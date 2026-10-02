import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { evaluateReproductionMetrics, REPRODUCTION_METRIC_DIRECTIONS, REPRODUCTION_METRIC_PROTOCOL } from '../lib/reproduction-metrics.mjs';
import { buildCvFeatures } from '../lib/cv-feature-encoder.mjs';
import { buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { optimizeWithAdaptiveDirectionalGate } from '../lib/adaptive-directional-rerank.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';
import { selectSafetyConstrainedTrial } from '../lib/preference-policy.mjs';
import { predictPreference } from '../lib/preference-model.mjs';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputDir = path.join(root, 'experiments', 'comparisons', 'latest_reproduction_metrics');
const metricNames = Object.keys(REPRODUCTION_METRIC_DIRECTIONS);
const methodLabels = { latest_v10: '最新 v10 模型', BinoForce: 'BinoForce', hedgehog_1d: 'Hedgehog 1D', hedgehog_3d: 'Hedgehog 3D', manual: '人工优化标注' };

const read = (relative) => fs.readFile(path.join(root, relative), 'utf8');
const readJson = async (relative) => JSON.parse(await read(relative));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const round = (value) => value === null ? null : Number(value.toFixed(9));
function csv(items, columns) { return [columns.join(','), ...items.map((item) => columns.map((column) => JSON.stringify(item[column] ?? '')).join(','))].join('\n') + '\n'; }
function labelsFromHedgehog(data) { return annotationsToLabels(data); }
// Saved layouts can be reordered independently of the source annotation.
function alignLabelsById(manualLabels, candidateLabels, context) {
  if (!Array.isArray(candidateLabels) || candidateLabels.length !== manualLabels.length) throw new Error(`${context}: label count mismatch`);
  const byId = new Map();
  for (const label of candidateLabels) {
    const id = String(label?.id || '');
    if (!id || byId.has(id)) throw new Error(`${context}: missing or duplicate label id ${id || '<empty>'}`);
    byId.set(id, label);
  }
  return manualLabels.map((manual) => {
    const label = byId.get(String(manual.id));
    if (!label) throw new Error(`${context}: missing label id ${manual.id}`);
    return label;
  });
}

function labelsFromBinoForce(manualLabels, row, context) {
  const centers = row?.label_centers;
  if (!Array.isArray(centers) || centers.length !== manualLabels.length) throw new Error(`${context}: missing or incomplete BinoForce layout`);
  const hasIds = centers.every((center) => center?.group_id || center?.id);
  if (hasIds) {
    const byId = new Map(centers.map((center) => [String(center.group_id || center.id), center]));
    if (byId.size !== centers.length) throw new Error(`${context}: duplicate BinoForce label ids`);
    return manualLabels.map((manual) => {
      const center = byId.get(String(manual.id));
      if (!center || String(center.text) !== String(manual.text)) throw new Error(`${context}: BinoForce label id/text mismatch for ${manual.id}`);
      return { ...manual, center: center.center.map(Number) };
    });
  }
  // Archived BinoForce JSONL zips scene.texts and centers in annotation order.
  // Text alone is not an identifier: duplicate texts occur within a sample.
  return manualLabels.map((manual, index) => {
    const center = centers[index];
    if (String(center?.text) !== String(manual.text)) throw new Error(`${context}: BinoForce annotation-order/text mismatch at label ${index + 1}`);
    return { ...manual, center: center.center.map(Number) };
  });
}

async function main() {
  const manifest = await readJson('experiments/dataset_manifest.json');
  const model = await readJson('experiments/layout_model.json');
  const modelSha256 = createHash('sha256').update(await read('experiments/layout_model.json')).digest('hex');
  const preferenceModel = await readJson('experiments/preference_model.json').catch(() => null);
  const preferenceSha256 = preferenceModel ? createHash('sha256').update(await read('experiments/preference_model.json')).digest('hex') : null;
  const leaderLengthPrior = await readJson('experiments/manual_leader_length_prior.json');
  const binoLayouts = (await read('BinoForce_2025/results/binoforce_layouts.jsonl')).trim().split(/\r?\n/).map(JSON.parse);
  const binoIndex = new Map(binoLayouts.filter((row) => row.method === 'BinoForce').map((row) => [`${row.category}/${row.sample}/${row.view}`, row]));
  const rows = [];
  const viewRows = [];
  for (const sample of manifest.samples) {
    const rawObj = await read(sample.input.source_obj);
    const clean = cleanObj(rawObj);
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const cvFeatures = await buildCvFeatures({ objText: clean.text, bounds });
    const spatialContext = buildSpatialContext(geometry, bounds, { gridSize: model?.architecture?.spatial_grid?.grid_size || 20 });
    const manual = annotationsToLabels(await readJson(sample.target.annotation_json));
    const generated = generatedCandidatesFromCleanObj(clean.text, bounds);
    const candidates = fixedCandidatesForLayoutModel(manual, generated, bounds, model);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/latest-candidates`);
    const styled = applyLayoutModel(candidates, bounds, model, { geometry, geometryFeature: cvFeatures.geometry, visualFeature: cvFeatures.visual, spatialContext });
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const options = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', seed: 17, iterations: 180, depthGrids, fixedLabels: true, category: sample.category, leaderLengthPrior };
    let latest, selectedSeed = 17;
    if (preferenceModel) {
      const trials = [];
      for (let trial = 0; trial < 4; trial += 1) {
        const seed = options.seed + trial;
        const directional = optimizeWithAdaptiveDirectionalGate(styled.labels, bounds, { ...options, seed }, geometry);
        const viewSafety = evaluateViewConditionedLayout(directional.labels, bounds, depthGrids);
        Object.assign(directional.metrics, {
          weighted_leader_crossing_risk: viewSafety.weighted_leader_crossing_risk,
          worst_view_leader_crossing_risk: viewSafety.worst_view_leader_crossing_risk,
          cvar_view_leader_crossing_risk: viewSafety.cvar_view_leader_crossing_risk,
          worst_view_leader_crossing_count: viewSafety.worst_view_leader_crossing_count
        });
        trials.push({ seed, labels: directional.labels, metrics: directional.metrics });
      }
      const selection = selectSafetyConstrainedTrial(trials, preferenceModel, predictPreference);
      if (!selection.selected) throw new Error(`No safe v10 candidate for ${sample.category}/${sample.sample_id}`);
      latest = selection.selected.labels;
      selectedSeed = selection.selected.seed;
    } else latest = optimizeWithAdaptiveDirectionalGate(styled.labels, bounds, options, geometry).labels;
    validateFixedLabelContract(manual, latest, `${sample.category}/${sample.sample_id}/latest-output`);
    const sampleViewRows = [];
    for (const view of MULTI_VIEW_NAMES) {
      const bino = binoIndex.get(`${sample.category}/${sample.sample_id}/${view}`);
      const binoforce = labelsFromBinoForce(manual, bino, `${sample.category}/${sample.sample_id}/${view}`);
      const hedgehog1d = alignLabelsById(manual, labelsFromHedgehog(await readJson(`Hedgehog/results/layouts/${sample.category}/${sample.sample_id}/${view}/hedgehog_1d.json`)), `${sample.category}/${sample.sample_id}/${view}/hedgehog_1d`);
      const hedgehog3d = alignLabelsById(manual, labelsFromHedgehog(await readJson(`Hedgehog/results/layouts/${sample.category}/${sample.sample_id}/${view}/hedgehog_3d.json`)), `${sample.category}/${sample.sample_id}/${view}/hedgehog_3d`);
      const methods = { latest_v10: latest, BinoForce: binoforce, hedgehog_1d: hedgehog1d, hedgehog_3d: hedgehog3d, manual };
      for (const [method, labels] of Object.entries(methods)) {
        validateFixedLabelContract(manual, labels, `${sample.category}/${sample.sample_id}/${view}/${method}`);
        const metrics = evaluateReproductionMetrics({ labels, manualLabels: manual, geometry, bounds, view, dbvAvailable: !method.startsWith('hedgehog') });
        sampleViewRows.push({ category: sample.category, sample_id: sample.sample_id, split: sample.split, view, method, label: methodLabels[method], num_labels: labels.length, selected_seed: method === 'latest_v10' ? selectedSeed : null, ...metrics });
      }
    }
    viewRows.push(...sampleViewRows);
    for (const method of Object.keys(methodLabels)) {
      const subset = sampleViewRows.filter((row) => row.method === method);
      rows.push({ category: sample.category, sample_id: sample.sample_id, split: sample.split, view: 'five_view_mean', method, label: methodLabels[method], num_labels: subset[0].num_labels, selected_seed: method === 'latest_v10' ? selectedSeed : null, ...Object.fromEntries(metricNames.map((metric) => {
        const values = subset.map((row) => row[metric]).filter((value) => value !== null && Number.isFinite(Number(value))).map(Number);
        return [metric, round(mean(values))];
      })) });
    }
    console.log(`${sample.split.padEnd(5)} ${sample.category.padEnd(12)} ${sample.sample_id}`);
  }
  const summaries = [];
  for (const split of ['train', 'val', 'test', 'all']) for (const method of Object.keys(methodLabels)) {
    const subset = rows.filter((row) => row.method === method && (split === 'all' || row.split === split));
    const summary = { split, method, label: methodLabels[method], sample_count: subset.length };
    for (const metric of metricNames) {
      const values = subset.map((row) => row[metric]).filter((value) => value !== null && Number.isFinite(Number(value))).map(Number);
      summary[metric] = round(mean(values));
      summary[`${metric}_valid_samples`] = values.length;
    }
    summaries.push(summary);
  }
  const mainSummaries = [];
  for (const split of ['train', 'val', 'test', 'all']) for (const method of Object.keys(methodLabels)) {
    const subset = viewRows.filter((row) => row.view === 'main' && row.method === method && (split === 'all' || row.split === split));
    mainSummaries.push({ split, method, label: methodLabels[method], sample_count: subset.length, ...Object.fromEntries(metricNames.map((metric) => {
      const values = subset.map((row) => row[metric]).filter((value) => value !== null && Number.isFinite(Number(value))).map(Number);
      return [metric, round(mean(values))];
    })) });
  }
  const report = { version: 'latest_v10_reproduction_comparison_v3', generated_at: new Date().toISOString(), active_model: { file: 'experiments/layout_model.json', version: model.version, status: model.status, sha256: modelSha256, architecture: model.architecture }, preference_model: { file: preferenceModel ? 'experiments/preference_model.json' : null, sha256: preferenceSha256, status: preferenceModel?.status || 'not_loaded' }, protocol: { ...REPRODUCTION_METRIC_PROTOCOL, scope: 'final static layout: five-view per-sample mean and main view separately; latest v10 matches active browser four seed17-20 trials, directional gate, view-conditioned safety and active reward', metric_directions: REPRODUCTION_METRIC_DIRECTIONS, source_alignment: { Hedgehog: 'group_id aligned to manual annotation before evaluation', BinoForce: 'group_id when present; archived JSONL verified by its annotation-order plus text invariant' }, binoforce_dynamic_note: 'BinoForce source CSV is a 300-frame dynamic mean; these rows reproject only its final saved snapshot. Do not equate the two.', hedgehog_legacy_csv_note: 'Historical Hedgehog CSV has rows inconsistent with saved final layout JSON; comparison reprojects saved JSON. Python layout_metrics on the archived Faucet/1758/left/hedgehog_3d JSON matches this evaluator.', hedgehog_dbv: 'N/A: original Hedgehog reproduction did not evaluate stereo; not a measured zero', manual_pck_note: 'Manual layout is the PCK reference, so its PCK is tautologically 1' }, split_counts: manifest.counts, methods: methodLabels, summaries, main_summaries: mainSummaries, rows, view_rows: viewRows };
  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(path.join(outputDir, 'comparison.json'), `${JSON.stringify(report, null, 2)}\n`);
  await fs.writeFile(path.join(outputDir, 'rows.csv'), csv(rows, ['category', 'sample_id', 'split', 'view', 'method', 'label', 'num_labels', 'selected_seed', ...metricNames]));
  await fs.writeFile(path.join(outputDir, 'view_rows.csv'), csv(viewRows, ['category', 'sample_id', 'split', 'view', 'method', 'label', 'num_labels', 'selected_seed', ...metricNames]));
  await fs.writeFile(path.join(outputDir, 'summary.csv'), csv(summaries, ['split', 'method', 'label', 'sample_count', ...metricNames]));
  await fs.writeFile(path.join(outputDir, 'main_summary.csv'), csv(mainSummaries, ['split', 'method', 'label', 'sample_count', ...metricNames]));
  console.log(`Wrote ${path.relative(root, outputDir)} with ${rows.length} sample-method rows and ${viewRows.length} per-view rows.`);
}

main().catch((error) => { console.error(error.stack || error.message || error); process.exitCode = 1; });
