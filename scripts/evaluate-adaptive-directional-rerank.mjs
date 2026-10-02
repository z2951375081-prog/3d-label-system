// Paired, offline-only validation of a per-sample deterministic fallback.
// Never consumes adjusted target positions when deciding which layout to use.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { acceptDirectionalAlternative, BASE_DIRECTIONAL_WEIGHTS, ADAPTIVE_DIRECTIONAL_WEIGHTS } from '../lib/adaptive-directional-rerank.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const read = async (filename) => JSON.parse(await fs.readFile(path.join(experiments, filename), 'utf8'));
const manifest = await read('dataset_manifest.json');
const model = await read('layout_model.json');
const prior = await read('manual_leader_length_prior.json');
const search = await read('directional_uniformity_upgrade_offline.json');
if (search.selection?.selected_policy !== 'density_4_8' || search.selection?.split !== 'val11_only') throw new Error('不存在 val11 预先选定的高方向权重候选');
const strong = BASE_DIRECTIONAL_WEIGHTS;
const alternative = ADAPTIVE_DIRECTIONAL_WEIGHTS;
const fields = [
  'directional_uniformity', 'directional_allocation_mismatch', 'intrinsic_quality_score',
  'multidimensional_quality_score', 'text_clarity', 'label_label_occlusion_ratio',
  'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio',
  'mesh_surface_intersection_ratio', 'multi_view_worst_overflow', 'leader_length_compliance_ratio',
  'mean_anchor_distance', 'manual_style_distance'
];
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const rounded = (value) => Number(value.toFixed(6));
function metricsSummary(rows, field) {
  const values = rows.map((item) => item[field] ?? item.manual_similarity?.[field]).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  return values.length ? rounded(mean(values)) : null;
}
async function evaluate(split) {
  const rows = [];
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const clean = cleanObj(await fs.readFile(path.join(root, sample.input.source_obj), 'utf8')).text;
    const bounds = boundsFromObj(clean), geometry = parseObjTriangles(clean);
    const manual = annotationsToLabels(JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8')));
    const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean, bounds), bounds, model);
    validateFixedLabelContract(manual, candidates, `${split}/${sample.category}/${sample.sample_id}/source`);
    const styled = applyLayoutModel(candidates, bounds, model).labels;
    const options = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true,
      iterations: 180, seed: 17, category: sample.category, leaderLengthPrior: prior,
      depthGrids: Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)])) };
    const evaluateOne = (weights, label) => {
      const labels = optimizeLabels(styled, bounds, { ...options, energyWeights: weights });
      validateFixedLabelContract(manual, labels, `${split}/${sample.category}/${sample.sample_id}/${label}`);
      // Evaluate both layouts with the original weights, so their objective
      // scores are numerically comparable even though search weights differ.
      return evaluateLayout(labels, bounds, { ...options, energyWeights: strong, geometry, manualReference: manual });
    };
    const baseline = evaluateOne(strong, 'incumbent');
    const alternativeMetrics = evaluateOne(alternative, 'alternative');
    const accepted = acceptDirectionalAlternative(baseline, alternativeMetrics);
    rows.push({ category: sample.category, sample_id: String(sample.sample_id), split, accepted,
      baseline: Object.fromEntries(fields.map((field) => [field, baseline[field] ?? baseline.manual_similarity?.[field] ?? null])),
      selected: Object.fromEntries(fields.map((field) => [field, (accepted ? alternativeMetrics : baseline)[field] ?? (accepted ? alternativeMetrics : baseline).manual_similarity?.[field] ?? null])) });
  }
  return rows;
}
function summarize(rows) {
  const output = { sample_count: rows.length, accepted_count: rows.filter((row) => row.accepted).length };
  for (const field of fields) {
    output[field] = { baseline: metricsSummary(rows.map((row) => row.baseline), field), selected: metricsSummary(rows.map((row) => row.selected), field) };
    output[field].delta = rounded(output[field].selected - output[field].baseline);
  }
  return output;
}
const validation = await evaluate('val');
const valSummary = summarize(validation);
const valPass = valSummary.accepted_count > 0
  && valSummary.directional_uniformity.delta > 0
  && valSummary.directional_allocation_mismatch.delta < 0
  && valSummary.multidimensional_quality_score.delta >= -0.01;
const test = valPass ? await evaluate('test') : [];
const report = {
  version: 'adaptive_directional_rerank_offline_v1', generated_at: new Date().toISOString(),
  original_weights: strong, challenger_weights: alternative, candidate_source: 'val11_selected_density_4_8',
  selection: { split: 'val11_only', passed: valPass,
    rule: 'per-sample intrinsic safety/quality, non-regressing penetration & mesh, uniformity +0.005, free-space mismatch -0.005; val mean quality tolerance -0.01',
    no_manual_target_metrics_used_to_select: true },
  validation: { summary: valSummary, rows: validation },
  test_confirmation: valPass ? { summary: summarize(test), rows: test } : null,
  test_used_for_selection: false, active_runtime_modified: false
};
await fs.writeFile(path.join(experiments, 'adaptive_directional_rerank_offline.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ selection: report.selection, val: valSummary, test: report.test_confirmation?.summary || null }, null, 2));
