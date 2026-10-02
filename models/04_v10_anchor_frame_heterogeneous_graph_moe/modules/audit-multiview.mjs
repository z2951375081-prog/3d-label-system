import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel, multiViewLayoutFeatures } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = async (file) => JSON.parse(await fs.readFile(path.join(root, file), 'utf8'));
const manifest = await readJson('experiments/dataset_manifest.json');
const current = await readJson('experiments/layout_model_multiview.json');
const previous = await readJson('experiments/layout_model_before_multiview.json');
const trainingReport = await readJson('experiments/layout_training_multiview_report.json');
const rows = [], contracts = [];
const methods = [['no_model', null], ['previous_mlp', previous], ['multiview_mlp', current]];
for (const sample of manifest.samples) {
  const text = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
  const clean = cleanObj(text), bounds = boundsFromObj(clean.text);
  const manual = annotationsToLabels(await readJson(sample.target.annotation_json));
  const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, current);
  const context = sample.category + '/' + sample.sample_id;
  validateFixedLabelContract(manual, candidates, context);
  if (candidates.some((label, index) => multiViewLayoutFeatures(label, bounds, index, candidates.length).length !== 51)) throw new Error(context + ': expected 51 features');
  contracts.push({ sample: context, split: sample.split, label_count: manual.length, valid: true });
  const geometry = parseObjTriangles(clean.text);
  const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
  const options = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', fixedLabels: true, optimizer: 'annealing', seed: 17, iterations: 180, depthGrids };
  const toCheck = sample.split === 'test' ? methods : [['multiview_mlp', current]];
  for (const [method, model] of toCheck) {
    const started = performance.now();
    const styled = model ? applyLayoutModel(candidates, bounds, model).labels : candidates;
    const labels = optimizeLabels(styled, bounds, options);
    validateFixedLabelContract(manual, labels, context + '/' + method);
    const metrics = evaluateLayout(labels, bounds, options);
    if (MULTI_VIEW_NAMES.some((view) => !metrics.view_metrics[view])) throw new Error(context + ': missing view metrics');
    if (sample.split === 'test') rows.push({ method, category: sample.category, sample_id: sample.sample_id, runtime_ms: Number((performance.now() - started).toFixed(3)), ...metrics });
  }
  console.log(context + ': fixed=' + manual.length + ', split=' + sample.split);
}
const fields = ['objective_score', 'olr', 'lcd', 'readability', 'object_occlusion_ratio', 'viewport_overflow_ratio', 'multi_view_worst_olr', 'multi_view_worst_overflow', 'runtime_ms'];
const summary = methods.map(([method]) => {
  const selected = rows.filter((row) => row.method === method);
  return { method, sample_count: selected.length, ...Object.fromEntries(fields.map((field) => [field, Number((selected.reduce((sum, row) => sum + row[field], 0) / Math.max(1, selected.length)).toFixed(6))])) };
});
const output = { generated_at: new Date().toISOString(), protocol: 'fixed manual label metadata; no manual centers in features; identical seed17 and 180-step five-view annealing; test11 final evaluation only', contract_samples: contracts.length, fixed_labels_valid: contracts.every((item) => item.valid), feature_dim: 51, architecture: current.architecture, training: trainingReport.metrics, summary, rows, contracts, limits: ['Engineering projection proxies, not original paper metrics.', 'Manual anchors, label identity and baseline sizes are known inputs: not automatic annotation discovery.', 'Supervised training loss is layout MSE with multiview inputs; five-view energy is a non-differentiable decoder, not an image CNN loss.', 'Local Qwen visual preference evidence is tracked separately from supervised layout metrics.'] };
await fs.writeFile(path.join(root, 'experiments', 'multiview_audit.json'), JSON.stringify(output, null, 2));
console.log(JSON.stringify(summary, null, 2));
