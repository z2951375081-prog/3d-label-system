// Layout-level preference model contract.
// The trainer writes the same JSON format consumed by predictPreference.

export const PREFERENCE_FEATURES = [
  'label_count',
  'semantic_count',
  'mean_anchor_distance',
  'intrinsic_style_balance',
  'intrinsic_size_consistency',
  'aesthetic_radial_mean',
  'aesthetic_radial_spread',
  'aesthetic_leader_length_mean',
  'aesthetic_leader_length_spread',
  'aesthetic_spacing_consistency',
  'aesthetic_text_size_fit',
  'aesthetic_composition_harmony'
];

function number(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }

export function layoutFeatureVector(metrics = {}) {
  const labelCount = Math.min(1, number(metrics.label_count) / 20);
  const semanticCount = Math.min(1, number(metrics.semantic_count) / 20);
  return [
    labelCount,
    semanticCount,
    Math.min(1, number(metrics.mean_anchor_distance) / 4),
    number(metrics.intrinsic_style_balance),
    number(metrics.intrinsic_size_consistency),
    Math.min(1, number(metrics.aesthetic_radial_mean) / 4),
    Math.min(1, number(metrics.aesthetic_radial_spread) / 2),
    Math.min(1, number(metrics.aesthetic_leader_length_mean) / 4),
    Math.min(1, number(metrics.aesthetic_leader_length_spread) / 2),
    number(metrics.aesthetic_spacing_consistency),
    number(metrics.aesthetic_text_size_fit),
    number(metrics.aesthetic_composition_harmony)
  ];
}

function tanh(value) { return Math.tanh(value); }

export function predictPreference(model, metrics = {}) {
  if (!model?.weights) return null;
  const inputDim = model.weights.w1?.[0]?.length || 0;
  const x = layoutFeatureVector(metrics).slice(0, inputDim);
  if (!inputDim || x.length !== inputDim) return null;
  const hidden = model.weights.w1.map((row, index) => tanh(row.reduce((sum, weight, feature) => sum + weight * x[feature], model.weights.b1[index])));
  const score = model.weights.w2.reduce((sum, weight, index) => sum + weight * hidden[index], model.weights.b2);
  return { score: Number(score.toFixed(6)), model_status: model.status || 'trained', feature_dim: x.length };
}
