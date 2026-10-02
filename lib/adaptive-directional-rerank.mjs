// A val-selected, per-sample free-space improvement gate. Neither its weights
// nor its decisions use the adjusted manual layout or an LLM visual score.
import { evaluateLayout, optimizeLabels } from './layout-optimizer.mjs';
export const BASE_DIRECTIONAL_WEIGHTS = Object.freeze({ directional_density: 1.6, directional_concentration: 1.2 });
export const ADAPTIVE_DIRECTIONAL_WEIGHTS = Object.freeze({ directional_density: 4.8, directional_concentration: 3.2 });

export function acceptDirectionalAlternative(base, candidate) {
  const required = ['directional_uniformity', 'directional_allocation_mismatch', 'intrinsic_quality_score',
    'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio',
    'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio',
    'multi_view_worst_overflow', 'leader_length_compliance_ratio'];
  if (!required.every((field) => Number.isFinite(base?.[field]) && Number.isFinite(candidate?.[field]))) return false;
  return candidate.directional_uniformity >= base.directional_uniformity + 0.005
    && candidate.directional_allocation_mismatch <= base.directional_allocation_mismatch - 0.005
    && candidate.intrinsic_quality_score >= base.intrinsic_quality_score - 0.01
    && candidate.text_clarity >= base.text_clarity - 0.04
    && candidate.label_label_occlusion_ratio <= base.label_label_occlusion_ratio + 0.003
    && candidate.label_object_occlusion_ratio <= base.label_object_occlusion_ratio + 0.003
    && candidate.object_label_occlusion_ratio <= base.object_label_occlusion_ratio + 0.01
    && candidate.object_penetration_ratio <= base.object_penetration_ratio
    && candidate.mesh_surface_intersection_ratio <= base.mesh_surface_intersection_ratio
    && candidate.multi_view_worst_overflow <= base.multi_view_worst_overflow + 0.003
    && candidate.leader_length_compliance_ratio >= base.leader_length_compliance_ratio - 0.01;
}

export function optimizeWithAdaptiveDirectionalGate(labels, bounds, options = {}, geometry = null) {
  const baseOptions = { ...options, energyWeights: { ...(options.energyWeights || {}), ...BASE_DIRECTIONAL_WEIGHTS } };
  const alternativeOptions = { ...options, energyWeights: { ...(options.energyWeights || {}), ...ADAPTIVE_DIRECTIONAL_WEIGHTS } };
  const baselineLabels = optimizeLabels(labels, bounds, baseOptions);
  const alternativeLabels = optimizeLabels(labels, bounds, alternativeOptions);
  // Compare with one common energy definition and no manual target metrics.
  const baselineMetrics = evaluateLayout(baselineLabels, bounds, { ...baseOptions, geometry });
  const alternativeMetrics = evaluateLayout(alternativeLabels, bounds, { ...baseOptions, geometry });
  const accepted = acceptDirectionalAlternative(baselineMetrics, alternativeMetrics);
  return {
    labels: accepted ? alternativeLabels : baselineLabels,
    metrics: accepted ? alternativeMetrics : baselineMetrics,
    baseline: { labels: baselineLabels, metrics: baselineMetrics },
    alternative: { labels: alternativeLabels, metrics: alternativeMetrics },
    accepted,
    policy: accepted ? 'adaptive_directional_4_8' : 'base_directional_1_6',
    selection_uses_manual_target: false
  };
}
