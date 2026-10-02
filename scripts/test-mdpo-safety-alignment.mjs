import assert from 'node:assert/strict';
import { evaluateMdpoVal11AlignedGate } from '../lib/mdpo-activation-gate.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';

const bounds = { center: [0, 0, 0], radius: 1, min: [-1, -1, -1], max: [1, 1, 1] };
const depth = { gridSize: 4, depth: new Float32Array(16).fill(Infinity), farDepth: new Float32Array(16).fill(-Infinity) };
const labels = [
  { center: [0, 0, 0], boxSize: [0.2, 0.2, 0.1], anchor: [0, 0, 0], text: 'A' },
  { center: [0.1, 0.1, 0], boxSize: [0.2, 0.2, 0.1], anchor: [0, 0, 0], text: 'B' }
];
const views = { main: depth, right: depth, left: depth, up: depth, down: depth };
const diagnostics = evaluateViewConditionedLayout(labels, bounds, views, { overlapPairWeight: 24, worstOverflowWeight: 20, cvarOverflowWeight: 12 });
assert.ok(diagnostics.per_view.main.overlap_pairs > 0, 'overlap pair count must detect overlap');
assert.ok(diagnostics.per_view.main.overlap_pair_risk > 0, 'aligned soft overlap-pair risk must be positive');
assert.ok(Number.isFinite(diagnostics.worst_view_overflow), 'worst-view overflow must be exposed');
assert.ok(Number.isFinite(diagnostics.cvar_view_overflow), 'overflow CVaR must be exposed');

const cohort = Array.from({ length: 11 }, (_, index) => 'Object/' + index);
const score = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4, manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: 0.2, PCK_010: 0.4, OLR: 0.1, LCD: 0, overlap_pairs: 0, occluded_points: 0, intersections: 0, worst_view_intersections: 0, object_occlusion: 0, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: 0, text_clarity: 4, leader_line_clarity: 4 };
const baseline = { sample_count: 11, cohort, scorer_model: 'qwen', prompt_version: 'p', views: ['before', 'main', 'right', 'left', 'up', 'down'], metric_protocol: 'm', score_means: score, metrics, test_used_for_selection: false };
const candidate = { ...baseline, score_means: { ...score, overall: 4.1 }, metrics: { ...metrics, PCK_005: 0.01 } };
const trainingAlignment = { protocol: 'safety_priority_v3_aligned', overlap_pair_weight: 24, worst_view_overflow_weight: 20, cvar_overflow_weight: 12, final_gate_metrics: ['overlap_pairs', 'worst_view_overflow'] };
const gate = evaluateMdpoVal11AlignedGate({ baseline, candidate, trainingAlignment });
assert.equal(gate.version, 'v10_mdpo_val11_activation_gate_v3');
assert.equal(gate.policy.id, 'safety_priority_v3_aligned');
assert.equal(gate.evidence.training_alignment, true);
assert.equal(gate.checks.pck_005_advisory, true);
assert.equal(gate.accepted, true);
console.log('MDPO safety alignment tests passed.');
