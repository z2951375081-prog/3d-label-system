import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { parseObjTriangles } from '../lib/layout-optimizer.mjs';
import {
  evaluateReproductionMetrics,
  reproductionQualityScore,
  reproductionQualityScoreV4,
  REPRODUCTION_METRIC_PROTOCOL,
  REPRODUCTION_METRIC_PROTOCOL_V4,
  REPRODUCTION_QUALITY_WEIGHTS_V4
} from '../lib/reproduction-metrics.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relative) => fs.readFile(path.join(root, relative), 'utf8');
const readJson = async (relative) => JSON.parse(await read(relative));
const close = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

const report = await readJson('experiments/comparisons/latest_reproduction_metrics/comparison.json');
assert.equal(report.version, 'latest_v10_reproduction_comparison_v3');
assert.equal(report.split_counts.train, 33);
assert.equal(report.split_counts.val, 11);
assert.equal(report.split_counts.test, 11);
assert.equal(report.rows.length, 55 * 5);
assert.equal(report.view_rows.length, 55 * 5 * 5);
assert.equal(report.summaries.length, 4 * 5);
assert.equal(report.main_summaries.length, 4 * 5);
assert.equal(REPRODUCTION_METRIC_PROTOCOL.camera_target.join(','), '0,0,0');
close(Object.values(REPRODUCTION_QUALITY_WEIGHTS_V4).reduce((sum, weight) => sum + weight, 0), 1);
close(REPRODUCTION_QUALITY_WEIGHTS_V4.PCK_005 + REPRODUCTION_QUALITY_WEIGHTS_V4.PCK_010, 0.35);
close(REPRODUCTION_QUALITY_WEIGHTS_V4.crossing_degree, 0.20);
assert.equal(REPRODUCTION_METRIC_PROTOCOL_V4.id, 'unified_reproduction_metrics_v4_pck_lcd_emphasis');
close(REPRODUCTION_METRIC_PROTOCOL_V4.quality_score.pck_total_weight, 0.35);
close(REPRODUCTION_METRIC_PROTOCOL_V4.quality_score.lcd_weight, 0.20);

for (const row of report.rows.filter((item) => item.method === 'manual')) {
  assert.equal(row.PCK_005, 1);
  assert.equal(row.PCK_010, 1);
}
for (const row of report.rows.filter((item) => item.method.startsWith('hedgehog'))) assert.equal(row.DBV, null);
for (const row of report.rows) {
  assert.ok(Number.isFinite(reproductionQualityScore(row, row.num_labels)));
  assert.ok(Number.isInteger(row.intersections) || row.view === 'five_view_mean');
}
assert.ok(report.view_rows.some((row) => row.intersections > 0 && row.LCD > 0));
assert.ok(report.view_rows.some((row) => row.overlap_pairs > 0));

// Cross-language regression: Hedgehog's Python layout_metrics() gives these
// values for the saved Faucet/1758/left/hedgehog_3d final layout.
const annotation = await readJson('data/Layout/Faucet/1758/layout1/Annotation/1758.json');
const layout = await readJson('Hedgehog/results/layouts/Faucet/1758/left/hedgehog_3d.json');
const clean = cleanObj(await read('data/Layout/Faucet/1758/layout1/Obj-O/1758-main-O.obj'));
const metrics = evaluateReproductionMetrics({
  labels: annotationsToLabels(layout),
  manualLabels: annotationsToLabels(annotation),
  geometry: parseObjTriangles(clean.text),
  bounds: boundsFromObj(clean.text),
  view: 'left',
  dbvAvailable: false
});
close(metrics.PCK_005, 0);
close(metrics.PCK_010, 0.25);
close(metrics.OLR, 0.009027394, 1e-9);
close(metrics.LCD, 0);
close(metrics.avg_leader_length, 0.066828253, 1e-9);
assert.equal(metrics.overlap_pairs, 1);
assert.equal(metrics.occluded_points, 0);
assert.equal(metrics.intersections, 0);
close(metrics.quality_score, reproductionQualityScore(metrics, layout.groups.length), 1e-6);
assert.ok(Number.isFinite(reproductionQualityScoreV4(metrics, layout.groups.length)));
assert.equal(metrics.DBV, null);

console.log('Reproduction metric protocol tests passed.');
