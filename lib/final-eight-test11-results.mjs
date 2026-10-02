import { reproductionQualityScoreV4, REPRODUCTION_METRIC_PROTOCOL_V4 } from './reproduction-metrics.mjs';

export const FINAL_EIGHT_TEST11_METHODS = Object.freeze([
  Object.freeze({ id: 'v10_no_rerank', role: 'active' }),
  Object.freeze({ id: 'v10_historical_rerank', role: 'reward' }),
  Object.freeze({ id: 'mdpo_no_rerank', role: 'candidate' }),
  Object.freeze({ id: 'mdpo_safe_rerank', role: 'candidate' }),
  Object.freeze({ id: 'BinoForce', role: 'baseline' }),
  Object.freeze({ id: 'hedgehog_3d', role: 'baseline' }),
  Object.freeze({ id: 'hedgehog_1d', role: 'baseline' }),
  Object.freeze({ id: 'manual', role: 'reference' })
]);

export const FINAL_EIGHT_TEST11_VIEWS = Object.freeze(['main', 'right', 'left', 'up', 'down']);
export const FINAL_EIGHT_TEST11_SAMPLES = Object.freeze([
  'Chair/42397', 'Dishwasher/13175', 'Door/9290', 'Earphone/11934', 'Faucet/1988', 'Lamp/17343',
  'Laptop/11075', 'Microwave/7346', 'Refrigerator/12045', 'Scissors/11111', 'TrashCan/12366'
]);
export const FINAL_EIGHT_METRICS = Object.freeze([
  'PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'
]);

const SOURCE_VERSION = 'final_eight_test11_unified_results_v2_pck_lcd';
const BASE_METRICS = FINAL_EIGHT_METRICS.filter((name) => name !== 'quality_score');
const TOLERANCE = 1e-7;
const sampleKey = (row) => `${row.category}/${row.sample_id}`;
const finiteValues = (rows, name) => rows.map((row) => row[name])
  .filter((value) => value !== null && value !== undefined && value !== '')
  .map(Number).filter(Number.isFinite);
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const metricMean = (rows, name) => mean(finiteValues(rows, name));

function fail(message) {
  const error = new Error(message);
  error.statusCode = 409;
  error.code = 'FINAL_EIGHT_TEST11_AUDIT_FAILED';
  throw error;
}

function assertClose(actual, expected, context) {
  if (actual === null && expected === null) return 0;
  if (!Number.isFinite(Number(actual)) || !Number.isFinite(Number(expected))) fail(`${context}: expected finite matching values`);
  const delta = Math.abs(Number(actual) - Number(expected));
  if (delta > TOLERANCE) fail(`${context}: stored=${actual}, recomputed=${expected}, delta=${delta}`);
  return delta;
}

function verifyStoredRows(storedRows, computedRows, keyOf, kind) {
  if (!Array.isArray(storedRows) || storedRows.length !== computedRows.length) fail(`${kind}: expected ${computedRows.length} stored rows`);
  const storedByKey = new Map();
  for (const row of storedRows) {
    const key = keyOf(row);
    if (storedByKey.has(key)) fail(`${kind}: duplicate row ${key}`);
    storedByKey.set(key, row);
  }
  let maxDelta = 0;
  for (const computed of computedRows) {
    const key = keyOf(computed);
    const stored = storedByKey.get(key);
    if (!stored) fail(`${kind}: missing row ${key}`);
    for (const metric of FINAL_EIGHT_METRICS) maxDelta = Math.max(maxDelta, assertClose(stored[metric], computed[metric], `${kind} ${key} ${metric}`));
  }
  return maxDelta;
}

export function validateAndSummarizeFinalEightTest11(report) {
  if (!report || typeof report !== 'object') fail('Final eight test11 report is not an object');
  if (report.version !== SOURCE_VERSION) fail(`Unexpected report version: ${report.version || 'missing'}`);
  if (report.split !== 'test' || Number(report.sample_count) !== FINAL_EIGHT_TEST11_SAMPLES.length) fail('Final eight report must contain test11');
  if (report.protocol?.id !== REPRODUCTION_METRIC_PROTOCOL_V4.id) fail('Final eight report must use the v4 PCK/LCD protocol');
  if (JSON.stringify(report.views) !== JSON.stringify(FINAL_EIGHT_TEST11_VIEWS)) fail('Final eight report view list/order mismatch');

  const viewRows = report.view_rows;
  const expectedViewRows = FINAL_EIGHT_TEST11_METHODS.length * FINAL_EIGHT_TEST11_SAMPLES.length * FINAL_EIGHT_TEST11_VIEWS.length;
  if (!Array.isArray(viewRows) || viewRows.length !== expectedViewRows) fail(`Expected ${expectedViewRows} view rows, received ${viewRows?.length ?? 0}`);

  const allowedMethods = new Set(FINAL_EIGHT_TEST11_METHODS.map((method) => method.id));
  const allowedSamples = new Set(FINAL_EIGHT_TEST11_SAMPLES);
  const allowedViews = new Set(FINAL_EIGHT_TEST11_VIEWS);
  const uniqueViewRows = new Set();
  for (const row of viewRows) {
    const key = sampleKey(row);
    if (row.split !== 'test' || !allowedMethods.has(row.method) || !allowedSamples.has(key) || !allowedViews.has(row.view)) fail(`Invalid view-row provenance: ${row.method}/${key}/${row.view}`);
    const uniqueKey = `${row.method}|${key}|${row.view}`;
    if (uniqueViewRows.has(uniqueKey)) fail(`Duplicate view row: ${uniqueKey}`);
    uniqueViewRows.add(uniqueKey);
    if (!Number.isInteger(Number(row.num_labels)) || Number(row.num_labels) < 1) fail(`Invalid label count: ${uniqueKey}`);
    for (const metric of BASE_METRICS) {
      const value = row[metric];
      const nullableDbv = metric === 'DBV' && (row.method === 'hedgehog_3d' || row.method === 'hedgehog_1d');
      if (!(nullableDbv && (value === null || value === undefined)) && !Number.isFinite(Number(value))) fail(`Invalid ${metric}: ${uniqueKey}`);
    }
  }

  const sampleRows = [];
  for (const method of FINAL_EIGHT_TEST11_METHODS) {
    for (const key of FINAL_EIGHT_TEST11_SAMPLES) {
      const selected = viewRows.filter((row) => row.method === method.id && sampleKey(row) === key);
      if (selected.length !== FINAL_EIGHT_TEST11_VIEWS.length) fail(`${method.id}/${key}: expected five view rows`);
      const views = new Set(selected.map((row) => row.view));
      if (views.size !== FINAL_EIGHT_TEST11_VIEWS.length) fail(`${method.id}/${key}: view coverage mismatch`);
      const labelCounts = new Set(selected.map((row) => Number(row.num_labels)));
      if (labelCounts.size !== 1) fail(`${method.id}/${key}: inconsistent label counts across views`);
      const [category, sample_id] = key.split('/');
      const sampleRow = {
        category, sample_id, split: 'test', view: 'five_view_mean', method: method.id,
        num_labels: Number(selected[0].num_labels),
        ...Object.fromEntries(BASE_METRICS.map((metric) => [metric, metricMean(selected, metric)]))
      };
      sampleRow.quality_score = reproductionQualityScoreV4(sampleRow, sampleRow.num_labels);
      sampleRows.push(sampleRow);
    }
  }

  const summaries = FINAL_EIGHT_TEST11_METHODS.map((method) => {
    const selected = sampleRows.filter((row) => row.method === method.id);
    return {
      split: 'test', method: method.id, sample_count: selected.length,
      ...Object.fromEntries(FINAL_EIGHT_METRICS.map((metric) => [metric, metricMean(selected, metric)]))
    };
  });

  const sampleRowMaxDelta = verifyStoredRows(report.rows, sampleRows, (row) => `${row.method}|${sampleKey(row)}`, 'sample rows');
  const summaryMaxDelta = verifyStoredRows(report.summaries, summaries, (row) => row.method, 'method summaries');
  return {
    sampleRows,
    summaries,
    audit: {
      passed: true,
      protocol_id: REPRODUCTION_METRIC_PROTOCOL_V4.id,
      split: 'test',
      method_count: FINAL_EIGHT_TEST11_METHODS.length,
      sample_count: FINAL_EIGHT_TEST11_SAMPLES.length,
      views_per_sample: FINAL_EIGHT_TEST11_VIEWS.length,
      expected_view_row_count: expectedViewRows,
      actual_view_row_count: viewRows.length,
      recomputed_sample_row_count: sampleRows.length,
      sample_row_max_abs_delta: sampleRowMaxDelta,
      summary_max_abs_delta: summaryMaxDelta,
      aggregation: 'mean views per method/sample; compute v4 quality from sample mean using that sample label count; mean 11 sample scores per method'
    }
  };
}
