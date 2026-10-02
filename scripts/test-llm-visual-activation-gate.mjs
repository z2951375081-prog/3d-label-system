import assert from 'node:assert/strict';
import { evaluateVisualActivationGate, inferGeometrySafetyEligibility } from '../lib/llm-visual-activation-gate.mjs';

const cohort = Array.from({ length: 11 }, (_, index) => ({ sample: { category: 'C', sample_id: String(index) } }));
function summary(round, scores) { return { round, sample_count: 11, scorer_model: 'qwen3-vl:4b-instruct', score_means: scores, sample_scores: cohort }; }
const baselineScores = { manual_style_similarity: 4, spatial_balance: 4, visual_hierarchy: 4, composition_harmony: 4, overall: 4 };
const baseline = summary(0, baselineScores);
const accepted = evaluateVisualActivationGate({ baseline, candidate: summary(1, { ...baselineScores, overall: 4.1, composition_harmony: 3.995 }), geometryEligible: true });
assert.equal(accepted.accepted, true);
assert.ok(accepted.constraints.aesthetic_composite_gain >= 0.02);
assert.ok(accepted.constraints.composition_harmony_change >= -0.01);
const lowGain = evaluateVisualActivationGate({ baseline, candidate: summary(2, { ...baselineScores, overall: 4.05 }), geometryEligible: true });
assert.equal(lowGain.accepted, false);
assert.ok(lowGain.violations.includes('aesthetic_composite_gain'));
const compositionDrop = evaluateVisualActivationGate({ baseline, candidate: summary(3, { ...baselineScores, overall: 4.2, composition_harmony: 3.98 }), geometryEligible: true });
assert.equal(compositionDrop.accepted, false);
assert.ok(compositionDrop.violations.includes('composition_harmony_change'));
const unsafe = evaluateVisualActivationGate({ baseline, candidate: summary(4, { ...baselineScores, overall: 4.2 }), geometryEligible: false });
assert.equal(unsafe.accepted, false);
assert.ok(unsafe.violations.includes('geometry_safety_eligible'));
const legacySafeValidation = { constraints: { objective_relative_change: -0.03, label_object_occlusion_change: -0.002, depth_penetration_change: 0, mesh_surface_intersection_change: 0, worst_overflow_change: 0.001, text_clarity_change: 0.02, leader_crossings: 0, worst_view_leader_crossing_count: 0 } };
assert.equal(inferGeometrySafetyEligibility(legacySafeValidation), true, 'legacy checkpoints must be re-evaluated from geometry constraints');
assert.equal(inferGeometrySafetyEligibility({ ...legacySafeValidation, geometry_safety_eligible: false }), false, 'explicit new checkpoint eligibility must take precedence');
console.log('LLM visual activation gate tests passed.');
