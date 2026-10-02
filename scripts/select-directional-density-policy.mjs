import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const manifest = await readJson(path.join(experiments, 'dataset_manifest.json'));
const layoutModel = await readJson(path.join(experiments, 'layout_model.json'));
const leaderLengthPrior = await readJson(path.join(experiments, 'manual_leader_length_prior.json'));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const round = (value) => value === null ? null : Number(value.toFixed(6));
const configurations = [
  { id: 'off', directional_density: 0, directional_concentration: 0 },
  { id: 'light', directional_density: 0.6, directional_concentration: 0.4 },
  { id: 'medium', directional_density: 1.1, directional_concentration: 0.8 },
  { id: 'strong', directional_density: 1.6, directional_concentration: 1.2 }
];

async function evaluateSplit(split, configs) {
  const rows = [];
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
    const clean = cleanObj(raw), bounds = boundsFromObj(clean.text), geometry = parseObjTriangles(clean.text);
    const manual = annotationsToLabels(await readJson(path.join(root, sample.target.annotation_json)));
    const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, layoutModel);
    validateFixedLabelContract(manual, candidates, `${split}/${sample.category}/${sample.sample_id}/candidates`);
    const styled = applyLayoutModel(candidates, bounds, layoutModel).labels;
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    for (const config of configs) {
      const optimizerOptions = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true, iterations: 180, seed: 17, depthGrids, category: sample.category, leaderLengthPrior, energyWeights: { directional_density: config.directional_density, directional_concentration: config.directional_concentration } };
      const labels = optimizeLabels(styled, bounds, optimizerOptions);
      validateFixedLabelContract(manual, labels, `${split}/${sample.category}/${sample.sample_id}/${config.id}`);
      const metrics = evaluateLayout(labels, bounds, { ...optimizerOptions, geometry, manualReference: manual });
      rows.push({ split, category: sample.category, sample_id: String(sample.sample_id), configuration: config.id, metrics });
    }
  }
  return rows;
}

const fields = ['multidimensional_quality_score', 'objective_score', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'multi_view_worst_overflow', 'directional_allocation_mismatch', 'directional_concentration_excess', 'directional_uniformity', 'mean_anchor_distance', 'manual_style_distance'];
function summarize(rows, config) {
  const selected = rows.filter((row) => row.configuration === config.id);
  const result = { configuration: config.id, weights: { directional_density: config.directional_density, directional_concentration: config.directional_concentration }, sample_count: selected.length };
  for (const field of fields) {
    const values = selected.map((row) => row.metrics[field] ?? row.metrics.manual_similarity?.[field]).filter((value) => Number.isFinite(Number(value))).map(Number);
    result[field] = round(mean(values));
  }
  return result;
}

const valRows = await evaluateSplit('val', configurations);
const validation = configurations.map((config) => summarize(valRows, config));
const baseline = validation.find((item) => item.configuration === 'off');
for (const item of validation) {
  item.changes = {
    text_clarity: round(item.text_clarity - baseline.text_clarity),
    label_overlap: round(item.label_label_occlusion_ratio - baseline.label_label_occlusion_ratio),
    label_object_occlusion: round(item.label_object_occlusion_ratio - baseline.label_object_occlusion_ratio),
    object_label_occlusion: round(item.object_label_occlusion_ratio - baseline.object_label_occlusion_ratio),
    penetration: round(item.object_penetration_ratio - baseline.object_penetration_ratio),
    mesh_intersection: round(item.mesh_surface_intersection_ratio - baseline.mesh_surface_intersection_ratio),
    worst_overflow: round(item.multi_view_worst_overflow - baseline.multi_view_worst_overflow)
  };
  item.safety_pass = item.configuration !== 'off'
    && item.changes.text_clarity >= -0.10
    && item.changes.label_overlap <= 0.03
    && item.changes.label_object_occlusion <= 0.01
    && item.changes.object_label_occlusion <= 0.06
    && item.changes.penetration <= 0.005
    && item.changes.mesh_intersection <= 0.005
    && item.changes.worst_overflow <= 0.01;
  item.selection_score = item.configuration === 'off' ? null : round(
    (baseline.directional_allocation_mismatch - item.directional_allocation_mismatch) * 2
    + (baseline.directional_concentration_excess - item.directional_concentration_excess)
    + (item.directional_uniformity - baseline.directional_uniformity) * 0.5
    + (item.multidimensional_quality_score - baseline.multidimensional_quality_score) * 0.2
  );
}
const eligible = validation.filter((item) => item.safety_pass && item.directional_allocation_mismatch < baseline.directional_allocation_mismatch).sort((left, right) => right.selection_score - left.selection_score || left.objective_score - right.objective_score);
const selected = eligible[0];
if (!selected) throw new Error('没有方向密度权重通过 val11 安全门控并改善空间分配');
const selectedConfig = configurations.find((item) => item.id === selected.configuration);
const testRows = await evaluateSplit('test', [configurations[0], selectedConfig]);
const test = [configurations[0], selectedConfig].map((config) => summarize(testRows, config));
const report = {
  version: 'directional_density_policy_v1', generated_at: new Date().toISOString(), enabled: true,
  definition: { sectors: ['right', 'upper_right', 'up', 'upper_left', 'left', 'lower_left', 'down', 'lower_right'], available_space: 'per-view depth grid pixels without object occupancy inside normalized viewport', allocation: 'label count share compared with non-object free-space share', penalty: 'total-variation allocation mismatch plus concentration beyond expected count and one-label rounding tolerance' },
  selected_without_test: true, selected_configuration: selected.configuration, weights: selected.weights,
  validation_selection: { criterion: 'directional allocation improvement on val11 with deterministic safety non-regression', candidates: validation },
  test_confirmation: test,
  rows: { val: valRows, test: testRows }
};
await fs.writeFile(path.join(experiments, 'directional_density_policy.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ selected: selected.configuration, weights: selected.weights, validation, test_confirmation: test, output: 'experiments/directional_density_policy.json' }, null, 2));
