export const LLM_SCORE_NAMES = Object.freeze([
  'text_clarity', 'coverage', 'label_label_occlusion', 'object_occlusion',
  'object_penetration', 'leader_line_clarity', 'multiview_consistency',
  'binocular_consistency', 'manual_style_similarity', 'spatial_balance',
  'size_consistency', 'visual_hierarchy', 'composition_harmony', 'overall'
]);

// These scores remain visible diagnostics and safety checks. They are handled
// by supervised layout learning and the deterministic five-view energy gate;
// they do not decide which candidate becomes an LLM preference winner.
export const LLM_SAFETY_DIMENSIONS = Object.freeze([
  'text_clarity', 'coverage', 'label_label_occlusion', 'object_occlusion',
  'object_penetration', 'leader_line_clarity', 'multiview_consistency',
  'binocular_consistency', 'size_consistency'
]);

// LLM preference learning is deliberately aesthetic-only. Manual style is one
// influence, not the whole target, so the learned reward can improve beyond a
// literal copy of the adjusted reference layout.
export const LLM_AESTHETIC_DIMENSIONS = Object.freeze([
  'manual_style_similarity', 'spatial_balance', 'visual_hierarchy',
  'composition_harmony', 'overall'
]);

export const LLM_AESTHETIC_WEIGHTS = Object.freeze({
  manual_style_similarity: 0.10,
  spatial_balance: 0.15,
  visual_hierarchy: 0.20,
  composition_harmony: 0.25,
  overall: 0.30
});

// Backward-compatible export for analysis code that previously called these
// dimensions constraints.
export const LLM_CONSTRAINT_DIMENSIONS = LLM_SAFETY_DIMENSIONS;

function number(value) { return Number.isFinite(Number(value)) ? Number(value) : 0; }

export function preferenceComposite(scores = {}) {
  return LLM_AESTHETIC_DIMENSIONS.reduce((sum, name) => sum + number(scores[name]) * LLM_AESTHETIC_WEIGHTS[name], 0);
}

export function safetyDiagnosticMean(scores = {}) {
  return LLM_SAFETY_DIMENSIONS.reduce((sum, name) => sum + number(scores[name]), 0) / LLM_SAFETY_DIMENSIONS.length;
}
