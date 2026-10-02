// Offline-only policy search: does not mutate the active model, optimizer,
// scoring service, or the ongoing LLM preference experiment.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const read = async (name) => JSON.parse(await fs.readFile(path.join(experiments, name), 'utf8'));
const manifest = await read('dataset_manifest.json');
const model = await read('layout_model.json');
const prior = await read('manual_leader_length_prior.json');
const incumbent = await read('directional_density_policy.json');
if (incumbent.selected_configuration !== 'strong' || incumbent.weights?.directional_density !== 1.6 || incumbent.weights?.directional_concentration !== 1.2) throw new Error('活动方向策略与离线搜索基准不一致');
const policies = [
  { id: 'incumbent', directional_density: 1.6, directional_concentration: 1.2 },
  { id: 'density_2_4', directional_density: 2.4, directional_concentration: 1.8 },
  { id: 'density_3_2', directional_density: 3.2, directional_concentration: 2.4 },
  { id: 'density_4_8', directional_density: 4.8, directional_concentration: 3.2 },
  { id: 'density_6_4', directional_density: 6.4, directional_concentration: 4.8 }
];
const fields = [
  'directional_uniformity', 'directional_allocation_mismatch', 'directional_concentration_excess',
  'multidimensional_quality_score', 'text_clarity', 'label_label_occlusion_ratio',
  'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio',
  'mesh_surface_intersection_ratio', 'multi_view_worst_overflow', 'mean_anchor_distance',
  'leader_length_compliance_ratio', 'manual_style_distance'
];
const mean = (values) => values.reduce((total, value) => total + value, 0) / values.length;
const round = (value) => Number(value.toFixed(6));
async function evaluate(split, candidates) {
  const result = [];
  for (const sample of manifest.samples.filter((entry) => entry.split === split)) {
    const clean = cleanObj(await fs.readFile(path.join(root, sample.input.source_obj), 'utf8')).text;
    const bounds = boundsFromObj(clean);
    const geometry = parseObjTriangles(clean);
    const manual = annotationsToLabels(JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8')));
    const inputs = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean, bounds), bounds, model);
    validateFixedLabelContract(manual, inputs, `${split}/${sample.category}/${sample.sample_id}/input`);
    const styled = applyLayoutModel(inputs, bounds, model).labels;
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    for (const policy of candidates) {
      const options = {
        viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing',
        fixedLabels: true, iterations: 180, seed: 17, category: sample.category, depthGrids,
        leaderLengthPrior: prior,
        energyWeights: { directional_density: policy.directional_density, directional_concentration: policy.directional_concentration }
      };
      const labels = optimizeLabels(styled, bounds, options);
      validateFixedLabelContract(manual, labels, `${split}/${sample.category}/${sample.sample_id}/${policy.id}`);
      const metrics = evaluateLayout(labels, bounds, { ...options, geometry, manualReference: manual });
      result.push({ split, category: sample.category, sample_id: String(sample.sample_id), policy: policy.id,
        metrics: Object.fromEntries(fields.map((name) => [name, metrics[name] ?? metrics.manual_similarity?.[name] ?? null])) });
    }
  }
  return result;
}
function summary(rows, policy) {
  const selected = rows.filter((row) => row.policy === policy.id);
  return { policy: policy.id, sample_count: selected.length, weights: {
    directional_density: policy.directional_density, directional_concentration: policy.directional_concentration
  }, ...Object.fromEntries(fields.map((field) => {
    const values = selected.map((row) => row.metrics[field]).filter((value) => value !== null && Number.isFinite(Number(value))).map(Number);
    return [field, values.length ? round(mean(values)) : null];
  })) };
}
const validationRows = await evaluate('val', policies);
const validation = policies.map((policy) => summary(validationRows, policy));
const baseline = validation[0];
const guarded = validation.slice(1).map((row) => {
  const changes = Object.fromEntries(fields.map((field) => [field, round(row[field] - baseline[field])]));
  const safe = changes.text_clarity >= -0.10 && changes.multidimensional_quality_score >= -0.02
    && changes.label_label_occlusion_ratio <= 0.01 && changes.label_object_occlusion_ratio <= 0.01
    && changes.object_label_occlusion_ratio <= 0.03 && changes.object_penetration_ratio <= 0.005
    && changes.mesh_surface_intersection_ratio <= 0.005 && changes.multi_view_worst_overflow <= 0.01
    && changes.leader_length_compliance_ratio >= -0.02;
  const improves = changes.directional_uniformity >= 0.02 && changes.directional_allocation_mismatch <= -0.01;
  return { ...row, changes, safety_pass: safe, directional_improvement_pass: improves, eligible: safe && improves };
});
const best = guarded.filter((row) => row.eligible).sort((a, b) => b.directional_uniformity - a.directional_uniformity || a.directional_allocation_mismatch - b.directional_allocation_mismatch)[0] || null;
const chosen = best ? policies.find((policy) => policy.id === best.policy) : null;
const testRows = chosen ? await evaluate('test', [policies[0], chosen]) : [];
const output = {
  version: 'directional_uniformity_upgrade_offline_v1', generated_at: new Date().toISOString(),
  source: { active_model: model.version, incumbent: incumbent.selected_configuration, seed: 17, iterations: 180, label_contract: 'manual_id_text_anchor_fixed' },
  selection: { split: 'val11_only', candidate_count: policies.length - 1, selected_policy: chosen?.id || null,
    rule: 'uniformity gain >= 0.02 and free-space mismatch decrease >= 0.01 vs strong, subject to multi-metric safety and quality non-regression' },
  validation: [baseline, ...guarded], test_confirmation: chosen ? [summary(testRows, policies[0]), summary(testRows, chosen)] : null,
  test_is_not_used_for_selection: true, active_runtime_modified: false
};
await fs.writeFile(path.join(experiments, 'directional_uniformity_upgrade_offline.json'), `${JSON.stringify(output, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(output, null, 2));
