import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj } from './generate-artifacts.mjs';
import { MULTI_VIEW_NAMES, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { evaluateReproductionMetrics, reproductionQualityScoreV4, REPRODUCTION_METRIC_PROTOCOL_V4 } from '../lib/reproduction-metrics.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputDir = path.join(root, 'experiments', 'comparisons', 'final_eight_test11');
const baseUrl = process.env.BASE_URL || 'http://127.0.0.1:5173';
const methods = [
  ['v10_no_rerank', '原始 v10'],
  ['v10_historical_rerank', 'v10 + 基础奖励 MLP'],
  ['mdpo_no_rerank', 'v10 + MDPO'],
  ['mdpo_safe_rerank', 'v10 + MDPO + 基础奖励 MLP'],
  ['BinoForce', 'BinoForce'],
  ['hedgehog_3d', 'Hedgehog 3D'],
  ['hedgehog_1d', 'Hedgehog 1D'],
  ['manual', '人工优化布局']
];
const metricNames = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const finiteMean = (rows, name) => mean(rows.map((row) => row[name]).filter((value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))).map(Number));

async function visual(method, sample) {
  const response = await fetch(baseUrl + '/api/model-visualization', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method, sample: { category: sample.category, sample_id: sample.sample_id } })
  });
  const result = await response.json();
  if (!response.ok) throw new Error(method + ' ' + sample.category + '/' + sample.sample_id + ': ' + (result.error || response.status));
  return result;
}

const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const testSamples = manifest.samples.filter((sample) => sample.split === 'test');
if (testSamples.length !== 11) throw new Error('Expected 11 test samples, received ' + testSamples.length);
const rows = [];
const viewRows = [];
for (const sample of testSamples) {
  for (const [method, label] of methods) {
    const result = await visual(method, sample);
    const bounds = boundsFromObj(result.clean_obj);
    const geometry = parseObjTriangles(result.clean_obj);
    const manualLabels = result.manual_reference?.labels || annotationsToLabels(JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8')));
    const perView = [];
    for (const view of MULTI_VIEW_NAMES) {
      const labels = result.labels_by_view?.[view] || result.labels;
      const metrics = evaluateReproductionMetrics({ labels, manualLabels, geometry, bounds, view, dbvAvailable: !method.startsWith('hedgehog') });
      metrics.quality_score = reproductionQualityScoreV4(metrics, labels.length);
      const row = { category: sample.category, sample_id: sample.sample_id, split: 'test', view, method, label, num_labels: labels.length, ...metrics };
      perView.push(row);
      viewRows.push(row);
    }
    const sampleMean = { category: sample.category, sample_id: sample.sample_id, split: 'test', view: 'five_view_mean', method, label,
      num_labels: perView[0]?.num_labels || manualLabels.length,
      ...Object.fromEntries(metricNames.filter((name) => name !== 'quality_score').map((name) => [name, finiteMean(perView, name)])) };
    sampleMean.quality_score = reproductionQualityScoreV4(sampleMean, sampleMean.num_labels);
    rows.push(sampleMean);
    console.log(sample.category + '/' + sample.sample_id + ' · ' + method + ' · ' + sampleMean.quality_score.toFixed(3));
  }
}
const summaries = methods.map(([method, label]) => {
  const selected = rows.filter((row) => row.method === method);
  return { split: 'test', method, label, sample_count: selected.length,
    ...Object.fromEntries(metricNames.map((name) => [name, finiteMean(selected, name)])) };
});
const report = {
  version: 'final_eight_test11_unified_results_v2_pck_lcd',
  generated_at: new Date().toISOString(),
  requested_at: '2026-09-26',
  purpose: 'posthoc_test11_display_requested_by_user_not_used_for_training_selection_or_deployment',
  split: 'test', sample_count: 11, views: MULTI_VIEW_NAMES,
  protocol: { ...REPRODUCTION_METRIC_PROTOCOL_V4, scope: 'eight model/method test11 five-view means; PCK 35%, LCD 20%; per-sample normalized then averaged' },
  deployment_note: 'MDPO rows remain rejected diagnostic candidates and are not deployed.',
  summaries, rows, view_rows: viewRows
};
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(path.join(outputDir, 'comparison.json'), JSON.stringify(report, null, 2) + '\n');
console.log('Wrote ' + path.relative(root, path.join(outputDir, 'comparison.json')));
