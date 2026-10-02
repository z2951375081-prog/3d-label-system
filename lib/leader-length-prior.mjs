function finite(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function quantile(values, probability) {
  const sorted = values.map(Number).filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * Math.max(0, Math.min(1, Number(probability)));
  const low = Math.floor(position), high = Math.ceil(position);
  return low === high ? sorted[low] : sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

export function summarizeLeaderLengths(values) {
  const clean = values.map(Number).filter(Number.isFinite);
  const probabilities = [0, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 1];
  const result = { count: clean.length };
  for (const probability of probabilities) {
    const key = probability === 0 ? 'min' : probability === 1 ? 'max' : `p${String(Math.round(probability * 100)).padStart(2, '0')}`;
    result[key] = Number(quantile(clean, probability).toFixed(6));
  }
  const mean = clean.reduce((sum, value) => sum + value, 0) / Math.max(1, clean.length);
  result.mean = Number(mean.toFixed(6));
  result.std = Number(Math.sqrt(clean.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, clean.length)).toFixed(6));
  return result;
}

export function rangeFromSummary(summary, policy = {}) {
  if (!summary) return null;
  const hardMin = finite(summary[policy.hard_min_quantile || 'p10'], finite(summary.p10));
  const preferredMin = finite(summary[policy.preferred_min_quantile || 'p20'], finite(summary.p20, hardMin));
  const target = finite(summary[policy.target_quantile || 'p50'], finite(summary.p50, preferredMin));
  const preferredMax = finite(summary[policy.preferred_max_quantile || 'p90'], finite(summary.p90, target));
  const hardMax = finite(summary[policy.hard_max_quantile || 'p95'], finite(summary.p95, preferredMax));
  if (![hardMin, preferredMin, target, preferredMax, hardMax].every(Number.isFinite)) return null;
  return {
    hard_min: Math.max(0.05, Math.min(hardMin, preferredMin)),
    preferred_min: Math.max(hardMin, preferredMin),
    target: Math.max(preferredMin, Math.min(target, preferredMax)),
    preferred_max: Math.max(preferredMin, preferredMax),
    hard_max: Math.max(preferredMax, hardMax)
  };
}

export function resolveLeaderLengthRange(prior, category = null) {
  if (!prior || prior.enabled === false) return null;
  const policy = prior.selected_policy || prior.policy || {};
  const categorySummary = category ? prior.categories?.[category] : null;
  const categoryMinimum = Math.max(1, finite(policy.minimum_category_labels, 12));
  const summary = categorySummary?.count >= categoryMinimum ? categorySummary : prior.global;
  const range = rangeFromSummary(summary, policy);
  if (!range) return null;
  return {
    ...range,
    source: categorySummary?.count >= categoryMinimum ? `train_category:${category}` : 'train_global',
    category: category || null,
    sample_count: summary.count,
    units: 'object_bounds_radius'
  };
}

export function normalizedLeaderLength(label, bounds) {
  return Math.hypot(...label.center.map((value, axis) => value - label.anchor[axis])) / Math.max(Number(bounds.radius) || 0, 1e-6);
}

export function clampLeaderLength(label, bounds, range, mode = 'hard') {
  if (!range) return { ...label, center: [...label.center] };
  const radius = Math.max(Number(bounds.radius) || 0, 1e-6);
  const lower = mode === 'preferred' ? range.preferred_min : range.hard_min;
  const upper = mode === 'preferred' ? range.preferred_max : range.hard_max;
  const offset = label.center.map((value, axis) => value - label.anchor[axis]);
  let length = Math.hypot(...offset);
  let direction = length > 1e-8 ? offset.map((value) => value / length) : label.anchor.map((value, axis) => value - bounds.center[axis]);
  let directionLength = Math.hypot(...direction);
  if (directionLength <= 1e-8) { direction = [0, 1, 0]; directionLength = 1; }
  direction = direction.map((value) => value / directionLength);
  const normalized = length / radius;
  const clamped = Math.max(lower, Math.min(upper, normalized));
  length = clamped * radius;
  return {
    ...label,
    center: label.anchor.map((value, axis) => value + direction[axis] * length),
    leader_length_constraint: {
      source: range.source,
      mode,
      before_normalized: Number(normalized.toFixed(6)),
      after_normalized: Number(clamped.toFixed(6)),
      preferred_range: [range.preferred_min, range.preferred_max],
      hard_range: [range.hard_min, range.hard_max]
    }
  };
}
