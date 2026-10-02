import assert from 'node:assert/strict';
import { compareMdpoUnifiedMetrics, evaluateMdpoTrainingUnifiedMetrics, MDPO_UNIFIED_NINE_METRICS } from '../lib/mdpo-unified-training-metrics.mjs';

const graphs = [{ category: 'Chair', sample_id: '1', manualLabels: [{}], geometry: {}, bounds: {}, value: 1 },
  { category: 'Lamp', sample_id: '2', manualLabels: [{}], geometry: {}, bounds: {}, value: 2 }];
const views = ['main', 'right', 'left', 'up', 'down'];
const offsets = Object.fromEntries(views.map((view, index) => [view, index]));
const make = (scale) => evaluateMdpoTrainingUnifiedMetrics({ network: { scale }, graphs, views, protocol: 'fixed-v2',
  forward: (network, graph) => ({ output: [[network.scale * graph.value]] }),
  labelsFromOutputs: (output) => [{ value: output[0][0] }],
  evaluateMetrics: ({ labels, view }) => Object.fromEntries(MDPO_UNIFIED_NINE_METRICS.map((name, index) => [name, labels[0].value + offsets[view] + index])) });
const baseline = make(1), candidate = make(2), delta = compareMdpoUnifiedMetrics(baseline, candidate);
assert.equal(baseline.sample_count, 2);
assert.equal(baseline.samples.length, 2);
assert.equal(baseline.samples[0].metrics.PCK_005, 3);
assert.equal(baseline.metrics.PCK_005, 3.5);
assert.equal(baseline.metrics.intersections, 21);
assert.equal(delta.PCK_005, 1.5);
assert.equal(delta.intersections, 3);
assert.throws(() => compareMdpoUnifiedMetrics(baseline, { ...candidate, metric_protocol: 'changed' }), /protocol mismatch/);
assert.throws(() => evaluateMdpoTrainingUnifiedMetrics({ network: {}, graphs: [], forward() {}, labelsFromOutputs() {}, evaluateMetrics() {}, views, protocol: 'x' }), /require model/);
console.log('v10-MDPO train33 unified nine-metric baseline/candidate/delta tests passed.');
