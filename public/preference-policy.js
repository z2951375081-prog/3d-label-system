const number = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const requiredNumber = (value) => value === null || value === undefined || value === '' ? Infinity : number(value, Infinity);

export const SAFETY_RERANK_LIMITS = Object.freeze({
  // Objective is a soft aggregate; explicit occlusion/penetration constraints
  // below remain strict. A 1% aggregate tolerance produced no train pairs.
  objective_relative_increase: 0.30,
  label_label_occlusion_increase: 0.01,
  label_object_occlusion_increase: 0.01,
  object_label_occlusion_increase: 0.06,
  object_label_occlusion_ceiling: 0.10,
  object_penetration_increase: 0.005,
  mesh_surface_intersection_increase: 0.005,
  worst_olr_increase: 0.03,
  worst_overflow_increase: 0.01,
  text_clarity_drop: 0.10,
  leader_crossings_ceiling: 0,
  worst_view_leader_crossing_count_ceiling: 0,
  // The smooth risk remains diagnostic among layouts with exactly zero true
  // intersections; its nonzero floor must not eliminate the safe candidate pool.
  weighted_leader_crossing_risk_increase: 0.03,
  worst_view_leader_crossing_risk_increase: 0.05,
  cvar_view_leader_crossing_risk_increase: 0.04
});

function surfaceIntersection(metrics = {}) {
  const value = metrics.mesh_surface_intersection_ratio;
  return value === null || value === undefined ? 0 : number(value);
}

export function safetySnapshot(metrics = {}) {
  return {
    objective_score: number(metrics.objective_score, Infinity),
    label_label_occlusion_ratio: number(metrics.label_label_occlusion_ratio),
    label_object_occlusion_ratio: number(metrics.label_object_occlusion_ratio),
    object_label_occlusion_ratio: number(metrics.object_label_occlusion_ratio),
    object_penetration_ratio: number(metrics.object_penetration_ratio),
    mesh_surface_intersection_ratio: surfaceIntersection(metrics),
    multi_view_worst_olr: number(metrics.multi_view_worst_olr),
    multi_view_worst_overflow: number(metrics.multi_view_worst_overflow),
    text_clarity: number(metrics.text_clarity),
    leader_crossings: requiredNumber(metrics.leader_crossings),
    worst_view_leader_crossing_count: requiredNumber(metrics.worst_view_leader_crossing_count),
    weighted_leader_crossing_risk: number(metrics.weighted_leader_crossing_risk, Infinity),
    worst_view_leader_crossing_risk: number(metrics.worst_view_leader_crossing_risk, Infinity),
    cvar_view_leader_crossing_risk: number(metrics.cvar_view_leader_crossing_risk, Infinity)
  };
}

export function compareSafetyReference(leftMetrics, rightMetrics) {
  const left = safetySnapshot(leftMetrics);
  const right = safetySnapshot(rightMetrics);
  return left.worst_view_leader_crossing_count - right.worst_view_leader_crossing_count
    || left.leader_crossings - right.leader_crossings
    || left.worst_view_leader_crossing_risk - right.worst_view_leader_crossing_risk
    || left.mesh_surface_intersection_ratio - right.mesh_surface_intersection_ratio
    || left.object_penetration_ratio - right.object_penetration_ratio
    || left.objective_score - right.objective_score;
}

export function assessSafetyEligibility(metrics, referenceMetrics, limits = SAFETY_RERANK_LIMITS) {
  const candidate = safetySnapshot(metrics);
  const reference = safetySnapshot(referenceMetrics);
  const thresholds = {
    objective_score: reference.objective_score * (1 + limits.objective_relative_increase),
    label_label_occlusion_ratio: reference.label_label_occlusion_ratio + limits.label_label_occlusion_increase,
    label_object_occlusion_ratio: reference.label_object_occlusion_ratio + limits.label_object_occlusion_increase,
    object_label_occlusion_ratio: Math.min(limits.object_label_occlusion_ceiling, reference.object_label_occlusion_ratio + limits.object_label_occlusion_increase),
    object_penetration_ratio: reference.object_penetration_ratio + limits.object_penetration_increase,
    mesh_surface_intersection_ratio: reference.mesh_surface_intersection_ratio + limits.mesh_surface_intersection_increase,
    multi_view_worst_olr: reference.multi_view_worst_olr + limits.worst_olr_increase,
    multi_view_worst_overflow: reference.multi_view_worst_overflow + limits.worst_overflow_increase,
    text_clarity: reference.text_clarity - limits.text_clarity_drop,
    leader_crossings: limits.leader_crossings_ceiling,
    worst_view_leader_crossing_count: limits.worst_view_leader_crossing_count_ceiling,
    weighted_leader_crossing_risk: reference.weighted_leader_crossing_risk + limits.weighted_leader_crossing_risk_increase,
    worst_view_leader_crossing_risk: reference.worst_view_leader_crossing_risk + limits.worst_view_leader_crossing_risk_increase,
    cvar_view_leader_crossing_risk: reference.cvar_view_leader_crossing_risk + limits.cvar_view_leader_crossing_risk_increase
  };
  const violations = [];
  if (candidate.objective_score > thresholds.objective_score) violations.push('objective_score');
  if (candidate.label_label_occlusion_ratio > thresholds.label_label_occlusion_ratio) violations.push('label_label_occlusion_ratio');
  if (candidate.label_object_occlusion_ratio > thresholds.label_object_occlusion_ratio) violations.push('label_object_occlusion_ratio');
  if (candidate.object_label_occlusion_ratio > thresholds.object_label_occlusion_ratio) violations.push('object_label_occlusion_ratio');
  if (candidate.object_penetration_ratio > thresholds.object_penetration_ratio) violations.push('object_penetration_ratio');
  if (candidate.mesh_surface_intersection_ratio > thresholds.mesh_surface_intersection_ratio) violations.push('mesh_surface_intersection_ratio');
  if (candidate.multi_view_worst_olr > thresholds.multi_view_worst_olr) violations.push('multi_view_worst_olr');
  if (candidate.multi_view_worst_overflow > thresholds.multi_view_worst_overflow) violations.push('multi_view_worst_overflow');
  if (candidate.text_clarity < thresholds.text_clarity) violations.push('text_clarity');
  if (candidate.leader_crossings > thresholds.leader_crossings) violations.push('leader_crossings');
  if (candidate.worst_view_leader_crossing_count > thresholds.worst_view_leader_crossing_count) violations.push('worst_view_leader_crossing_count');
  if (candidate.weighted_leader_crossing_risk > thresholds.weighted_leader_crossing_risk) violations.push('weighted_leader_crossing_risk');
  if (candidate.worst_view_leader_crossing_risk > thresholds.worst_view_leader_crossing_risk) violations.push('worst_view_leader_crossing_risk');
  if (candidate.cvar_view_leader_crossing_risk > thresholds.cvar_view_leader_crossing_risk) violations.push('cvar_view_leader_crossing_risk');
  return { eligible: violations.length === 0, violations, candidate, reference, thresholds };
}

export function selectSafetyConstrainedTrial(trials, preferenceModel, predictPreference) {
  if (!Array.isArray(trials) || !trials.length) return null;
  const reference = [...trials].sort((left, right) => compareSafetyReference(left.metrics, right.metrics))[0];
  const assessed = trials.map((trial) => {
    const safety = assessSafetyEligibility(trial.metrics, reference.metrics);
    const preference = preferenceModel ? predictPreference(preferenceModel, trial.metrics) : null;
    return { ...trial, safety, preference };
  });
  const eligible = assessed.filter((trial) => trial.safety.eligible);
  const selected = !eligible.length ? null : preferenceModel
    ? [...eligible].sort((left, right) => (right.preference?.score ?? -Infinity) - (left.preference?.score ?? -Infinity) || number(left.metrics?.objective_score, Infinity) - number(right.metrics?.objective_score, Infinity))[0]
    : reference;
  return { selected, reference, assessed, eligible_count: eligible.length, total_count: assessed.length };
}
