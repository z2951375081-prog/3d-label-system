export function compareToManual(generated, manual) {
  const higherBetter = ['readability', 'coverage', 'occlusion', 'anchor_coverage'];
  const lowerBetter = ['olr', 'lcd', 'dbv', 'viewport_overflow_ratio', 'mean_anchor_distance', 'binocular_disparity'];
  const metrics = [...higherBetter, ...lowerBetter];
  const gaps = {};
  const scores = [];
  for (const metric of metrics) {
    const reference = Number(manual?.[metric] ?? 0);
    const value = Number(generated?.[metric] ?? 0);
    const score = higherBetter.includes(metric)
      ? reference > 0 ? (value / reference) * 100 : value === 0 ? 100 : 0
      : reference <= 1e-6 ? (value <= 1e-6 ? 100 : 0) : Math.max(0, 100 - Math.max(0, value - reference) / reference * 100);
    scores.push(Math.max(0, Math.min(100, score)));
    gaps[metric] = { generated: value, manual: reference, delta: Number((value - reference).toFixed(5)), score: Number(score.toFixed(2)) };
  }
  return { manual_score: 100, generated_score: Number((scores.reduce((sum, value) => sum + value, 0) / Math.max(1, scores.length)).toFixed(2)), reference: 'manual_annotation', metrics: gaps };
}

