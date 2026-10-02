const AESTHETIC_WEIGHTS = Object.freeze({ overall: 0.30, composition_harmony: 0.25, visual_hierarchy: 0.20, spatial_balance: 0.15, manual_style_similarity: 0.10 });
export const MDPO_VAL11_POLICY_ID = 'safety_priority_v2';
export const MDPO_VAL11_ALIGNED_POLICY_ID = 'safety_priority_v3_aligned';
export const MDPO_VAL11_THRESHOLDS = Object.freeze({
  policy_id: MDPO_VAL11_POLICY_ID,
  pck_role: 'advisory_only',
  required_samples: 11, minimum_aesthetic_gain: 0.02, minimum_composition_change: -0.01,
  minimum_pck_005_change: null, minimum_pck_010_change: null, maximum_olr_change: 0.01,
  maximum_overlap_pairs_change: 0, maximum_occluded_points_change: 0,
  maximum_lcd_change: 0, maximum_object_occlusion_change: 0, maximum_penetration_change: 0,
  maximum_mesh_intersection_change: 0, maximum_overflow_change: 0,
  minimum_text_clarity_change: 0, minimum_leader_line_clarity_change: 0,
  maximum_intersections: 0, maximum_worst_view_intersections: 0
});
export const MDPO_VAL11_ALIGNED_THRESHOLDS = Object.freeze({
  ...MDPO_VAL11_THRESHOLDS,
  policy_id: MDPO_VAL11_ALIGNED_POLICY_ID,
  training_alignment_required: true,
  aligned_terms: Object.freeze(['overlap_pair_risk', 'worst_view_overflow', 'cvar_view_overflow'])
});
const METRICS = Object.freeze(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'overlap_pairs', 'occluded_points', 'intersections', 'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow', 'text_clarity', 'leader_line_clarity']);
const finite = (value) => value === null || value === undefined || value === '' || typeof value === 'boolean' || !Number.isFinite(Number(value)) ? null : Number(value);
const atLeast = (value, threshold) => value !== null && value + 1e-9 >= threshold;
const atMost = (value, threshold) => value !== null && value - 1e-9 <= threshold;
const cohort = (value) => (value?.cohort || value?.samples || []).map((item) => typeof item === 'string' ? item : `${item.category ?? item.sample?.category}/${item.sample_id ?? item.sample?.sample_id}`).sort();
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
function aesthetic(scores) { return Object.entries(AESTHETIC_WEIGHTS).reduce((sum, [name, weight]) => sum + weight * Number(scores[name]), 0); }

export function evaluateMdpoVal11Gate({ baseline, candidate, thresholds = MDPO_VAL11_THRESHOLDS } = {}) {
  const scoreNames = [...Object.keys(AESTHETIC_WEIGHTS), 'text_clarity', 'leader_line_clarity'];
  const baseScores = Object.fromEntries(scoreNames.map((name) => [name, finite(baseline?.score_means?.[name])]));
  const nextScores = Object.fromEntries(scoreNames.map((name) => [name, finite(candidate?.score_means?.[name])]));
  const baseMetrics = Object.fromEntries(METRICS.map((name) => [name, finite(baseline?.metrics?.[name])]));
  const nextMetrics = Object.fromEntries(METRICS.map((name) => [name, finite(candidate?.metrics?.[name])]));
  const baseCohort = cohort(baseline), nextCohort = cohort(candidate);
  const evidence = {
    baseline_complete: Object.values(baseScores).every((value) => value !== null) && Object.values(baseMetrics).every((value) => value !== null),
    candidate_complete: Object.values(nextScores).every((value) => value !== null) && Object.values(nextMetrics).every((value) => value !== null),
    full_val11: Number(baseline?.sample_count) === thresholds.required_samples && Number(candidate?.sample_count) === thresholds.required_samples && baseCohort.length === thresholds.required_samples && nextCohort.length === thresholds.required_samples && new Set(baseCohort).size === thresholds.required_samples && new Set(nextCohort).size === thresholds.required_samples,
    same_cohort: same(baseCohort, nextCohort),
    same_scorer: Boolean(baseline?.scorer_model && baseline.scorer_model === candidate?.scorer_model),
    same_prompt: Boolean(baseline?.prompt_version && baseline.prompt_version === candidate?.prompt_version),
    same_views: same(baseline?.views, candidate?.views) && same(candidate?.views, ['before', 'main', 'right', 'left', 'up', 'down']),
    same_protocol: Boolean(baseline?.metric_protocol && baseline.metric_protocol === candidate?.metric_protocol),
    test_not_used: baseline?.test_used_for_selection === false && candidate?.test_used_for_selection === false
  };
  const delta = {
    aesthetic: evidence.baseline_complete && evidence.candidate_complete ? aesthetic(nextScores) - aesthetic(baseScores) : null,
    composition_harmony: nextScores.composition_harmony === null || baseScores.composition_harmony === null ? null : nextScores.composition_harmony - baseScores.composition_harmony,
    PCK_005: nextMetrics.PCK_005 === null || baseMetrics.PCK_005 === null ? null : nextMetrics.PCK_005 - baseMetrics.PCK_005,
    PCK_010: nextMetrics.PCK_010 === null || baseMetrics.PCK_010 === null ? null : nextMetrics.PCK_010 - baseMetrics.PCK_010,
    OLR: nextMetrics.OLR === null || baseMetrics.OLR === null ? null : nextMetrics.OLR - baseMetrics.OLR,
    LCD: nextMetrics.LCD === null || baseMetrics.LCD === null ? null : nextMetrics.LCD - baseMetrics.LCD,
    overlap_pairs: nextMetrics.overlap_pairs === null || baseMetrics.overlap_pairs === null ? null : nextMetrics.overlap_pairs - baseMetrics.overlap_pairs,
    occluded_points: nextMetrics.occluded_points === null || baseMetrics.occluded_points === null ? null : nextMetrics.occluded_points - baseMetrics.occluded_points,
    object_occlusion: nextMetrics.object_occlusion === null || baseMetrics.object_occlusion === null ? null : nextMetrics.object_occlusion - baseMetrics.object_occlusion,
    penetration: nextMetrics.penetration === null || baseMetrics.penetration === null ? null : nextMetrics.penetration - baseMetrics.penetration,
    mesh_surface_intersection: nextMetrics.mesh_surface_intersection === null || baseMetrics.mesh_surface_intersection === null ? null : nextMetrics.mesh_surface_intersection - baseMetrics.mesh_surface_intersection,
    worst_view_overflow: nextMetrics.worst_view_overflow === null || baseMetrics.worst_view_overflow === null ? null : nextMetrics.worst_view_overflow - baseMetrics.worst_view_overflow,
    text_clarity: nextScores.text_clarity === null || baseScores.text_clarity === null ? null : nextScores.text_clarity - baseScores.text_clarity,
    leader_line_clarity: nextScores.leader_line_clarity === null || baseScores.leader_line_clarity === null ? null : nextScores.leader_line_clarity - baseScores.leader_line_clarity
  };
  const checks = {
    aesthetic_gain: atLeast(delta.aesthetic, thresholds.minimum_aesthetic_gain),
    composition_non_degradation: atLeast(delta.composition_harmony, thresholds.minimum_composition_change),
    pck_005_advisory: true,
    pck_010_advisory: true,
    zero_intersections: atMost(nextMetrics.intersections, thresholds.maximum_intersections),
    zero_worst_view_intersections: atMost(nextMetrics.worst_view_intersections, thresholds.maximum_worst_view_intersections),
    lcd_non_degradation: atMost(delta.LCD, thresholds.maximum_lcd_change),
    olr_limit: atMost(delta.OLR, thresholds.maximum_olr_change),
    overlap_pairs_non_degradation: atMost(delta.overlap_pairs, thresholds.maximum_overlap_pairs_change),
    occluded_points_non_degradation: atMost(delta.occluded_points, thresholds.maximum_occluded_points_change),
    object_occlusion_non_degradation: atMost(delta.object_occlusion, thresholds.maximum_object_occlusion_change),
    penetration_non_degradation: atMost(delta.penetration, thresholds.maximum_penetration_change),
    mesh_intersection_non_degradation: atMost(delta.mesh_surface_intersection, thresholds.maximum_mesh_intersection_change),
    overflow_non_degradation: atMost(delta.worst_view_overflow, thresholds.maximum_overflow_change),
    text_clarity_non_degradation: atLeast(delta.text_clarity, thresholds.minimum_text_clarity_change),
    leader_line_clarity_non_degradation: atLeast(delta.leader_line_clarity, thresholds.minimum_leader_line_clarity_change)
  };
  const nonNegative = (value) => Math.max(0, Number(value) || 0);
  const safety_penalty = 1000 * nonNegative(nextMetrics.intersections) + 1000 * nonNegative(nextMetrics.worst_view_intersections) + 100 * nonNegative(nextMetrics.overlap_pairs) + 100 * nonNegative(nextMetrics.occluded_points) + 100 * nonNegative(nextMetrics.object_occlusion) + 100 * nonNegative(nextMetrics.penetration) + 100 * nonNegative(nextMetrics.mesh_surface_intersection) + 20 * nonNegative(nextMetrics.worst_view_overflow) + 10 * nonNegative(nextMetrics.OLR) + 10 * nonNegative(nextMetrics.LCD);
  const violations = [...Object.entries(evidence).filter(([, ok]) => !ok).map(([name]) => `evidence:${name}`), ...Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => `gate:${name}`)];
  const accepted = violations.length === 0;
  return { version: 'v10_mdpo_val11_activation_gate_v2', policy: { id: MDPO_VAL11_POLICY_ID, objective: 'prioritize intersections, overlap, occlusion and worst-view safety; PCK is reported diagnostically and is not an activation criterion' }, status: accepted ? 'accepted' : 'rejected', accepted, deployment_status: accepted ? 'eligible_for_atomic_activation' : 'diagnostic_only', thresholds, evidence, checks, delta, safety_penalty, baseline: { scores: baseScores, metrics: baseMetrics, cohort: baseCohort }, candidate: { scores: nextScores, metrics: nextMetrics, cohort: nextCohort }, violations, test_not_used_for_activation: true };
}

export function evaluateMdpoVal11AlignedGate({ baseline, candidate, trainingAlignment } = {}) {
  const base = evaluateMdpoVal11Gate({ baseline, candidate, thresholds: MDPO_VAL11_ALIGNED_THRESHOLDS });
  const alignmentOk = trainingAlignment?.protocol === MDPO_VAL11_ALIGNED_POLICY_ID
    && Number(trainingAlignment.overlap_pair_weight) > 0
    && Number(trainingAlignment.worst_view_overflow_weight) > 0
    && Number(trainingAlignment.cvar_overflow_weight) > 0
    && Array.isArray(trainingAlignment.final_gate_metrics)
    && MDPO_VAL11_ALIGNED_THRESHOLDS.aligned_terms.every((name) => trainingAlignment.final_gate_metrics.includes(name === 'overlap_pair_risk' ? 'overlap_pairs' : name === 'cvar_view_overflow' ? 'worst_view_overflow' : name));
  const evidence = { ...base.evidence, training_alignment: alignmentOk };
  const violations = [...base.violations.filter((item) => !item.startsWith('evidence:training_alignment')), ...(alignmentOk ? [] : ['evidence:training_alignment'])];
  const accepted = violations.length === 0;
  return {
    ...base,
    version: 'v10_mdpo_val11_activation_gate_v3',
    policy: { ...base.policy, id: MDPO_VAL11_ALIGNED_POLICY_ID, objective: 'training and activation gate aligned on overlap-pair risk, worst-view overflow, CVaR overflow, occlusion, penetration and intersections; PCK remains advisory-only' },
    thresholds: MDPO_VAL11_ALIGNED_THRESHOLDS,
    evidence,
    status: accepted ? 'accepted' : 'rejected',
    accepted,
    deployment_status: accepted ? 'eligible_for_atomic_activation' : 'diagnostic_only',
    violations,
    training_alignment: trainingAlignment || null
  };
}
