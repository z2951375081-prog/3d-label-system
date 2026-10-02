import { LLM_AESTHETIC_DIMENSIONS, LLM_AESTHETIC_WEIGHTS, preferenceComposite } from '../public/preference-scoring.js';

export const LLM_VISUAL_ACTIVATION_THRESHOLDS = Object.freeze({
  minimum_aesthetic_composite_gain: 0.02,
  minimum_composition_harmony_change: -0.01,
  required_val_samples: 11
});

export const LLM_GEOMETRY_SAFETY_THRESHOLDS = Object.freeze({
  maximum_objective_relative_change: 0.01,
  maximum_label_object_occlusion_change: 0.01,
  maximum_depth_penetration_change: 0.005,
  maximum_mesh_surface_intersection_change: 0.005,
  maximum_worst_overflow_change: 0.01,
  minimum_text_clarity_change: -0.10,
  maximum_leader_crossings: 0,
  maximum_worst_view_leader_crossing_count: 0
});

function finite(value) { return Number.isFinite(Number(value)) ? Number(value) : null; }
function cohort(summary) { return (summary?.sample_scores || []).map((row) => `${row.sample?.category}/${row.sample?.sample_id}`).sort(); }
function sameJson(left, right) { return JSON.stringify(left) === JSON.stringify(right); }

export function inferGeometrySafetyEligibility(validation, thresholds = LLM_GEOMETRY_SAFETY_THRESHOLDS) {
  if (typeof validation?.geometry_safety_eligible === 'boolean') return validation.geometry_safety_eligible;
  const constraints = validation?.constraints;
  if (!constraints) return false;
  const required = [
    'objective_relative_change', 'label_object_occlusion_change', 'depth_penetration_change',
    'mesh_surface_intersection_change', 'worst_overflow_change', 'text_clarity_change',
    'leader_crossings', 'worst_view_leader_crossing_count'
  ];
  if (required.some((name) => finite(constraints[name]) === null)) return false;
  return Number(constraints.objective_relative_change) <= thresholds.maximum_objective_relative_change
    && Number(constraints.label_object_occlusion_change) <= thresholds.maximum_label_object_occlusion_change
    && Number(constraints.depth_penetration_change) <= thresholds.maximum_depth_penetration_change
    && Number(constraints.mesh_surface_intersection_change) <= thresholds.maximum_mesh_surface_intersection_change
    && Number(constraints.worst_overflow_change) <= thresholds.maximum_worst_overflow_change
    && Number(constraints.text_clarity_change) >= thresholds.minimum_text_clarity_change
    && Number(constraints.leader_crossings) <= thresholds.maximum_leader_crossings
    && Number(constraints.worst_view_leader_crossing_count) <= thresholds.maximum_worst_view_leader_crossing_count;
}

export function normalizedVisualSummary(summary) {
  if (!summary) return null;
  const scores = Object.fromEntries(LLM_AESTHETIC_DIMENSIONS.map((name) => [name, finite(summary.score_means?.[name])]));
  const complete = Object.values(scores).every((value) => value !== null);
  return {
    round: Number(summary.round),
    sample_count: Number(summary.sample_count || 0),
    scorer_model: summary.scorer_model || null,
    scores,
    composite_score: complete ? Number(preferenceComposite(scores).toFixed(6)) : null,
    cohort: cohort(summary)
  };
}

export function evaluateVisualActivationGate({ baseline, candidate, geometryEligible = false, thresholds = LLM_VISUAL_ACTIVATION_THRESHOLDS } = {}) {
  const base = normalizedVisualSummary(baseline);
  const next = normalizedVisualSummary(candidate);
  const evidence = {
    baseline_complete: Boolean(base?.composite_score !== null),
    candidate_complete: Boolean(next?.composite_score !== null),
    full_val11: Boolean(base?.sample_count === thresholds.required_val_samples && next?.sample_count === thresholds.required_val_samples),
    same_scorer: Boolean(base?.scorer_model && base.scorer_model === next?.scorer_model),
    same_cohort: Boolean(base?.cohort?.length && sameJson(base.cohort, next?.cohort || []))
  };
  const constraints = {
    aesthetic_composite_gain: base && next && base.composite_score !== null && next.composite_score !== null ? Number((next.composite_score - base.composite_score).toFixed(6)) : null,
    composition_harmony_change: base?.scores?.composition_harmony !== null && next?.scores?.composition_harmony !== null ? Number((next.scores.composition_harmony - base.scores.composition_harmony).toFixed(6)) : null,
    geometry_safety_eligible: Boolean(geometryEligible)
  };
  const evidenceComplete = Object.values(evidence).every(Boolean);
  const accepted = evidenceComplete
    && constraints.geometry_safety_eligible
    && constraints.aesthetic_composite_gain >= thresholds.minimum_aesthetic_composite_gain
    && constraints.composition_harmony_change >= thresholds.minimum_composition_harmony_change;
  const violations = [];
  if (!evidenceComplete) violations.push(...Object.entries(evidence).filter(([, ok]) => !ok).map(([name]) => name));
  if (!constraints.geometry_safety_eligible) violations.push('geometry_safety_eligible');
  if (constraints.aesthetic_composite_gain === null || constraints.aesthetic_composite_gain < thresholds.minimum_aesthetic_composite_gain) violations.push('aesthetic_composite_gain');
  if (constraints.composition_harmony_change === null || constraints.composition_harmony_change < thresholds.minimum_composition_harmony_change) violations.push('composition_harmony_change');
  return {
    version: 'llm_val11_five_dimension_activation_gate_v1',
    status: accepted ? 'accepted' : 'rejected',
    accepted,
    criterion: 'real val11 six-image Qwen five-dimension aesthetic composite gain >= 0.02; composition_harmony change >= -0.01; deterministic geometry safety gate must pass',
    formula: '0.30*overall + 0.25*composition_harmony + 0.20*visual_hierarchy + 0.15*spatial_balance + 0.10*manual_style_similarity',
    aesthetic_dimensions: LLM_AESTHETIC_DIMENSIONS,
    weights: LLM_AESTHETIC_WEIGHTS,
    thresholds,
    evidence,
    baseline: base,
    candidate: next,
    constraints,
    violations,
    test_not_used_for_activation: true
  };
}
