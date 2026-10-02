export const MDPO_UNIFIED_NINE_METRICS = Object.freeze(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length',
  'overlap_pairs', 'occluded_points', 'intersections', 'quality_score']);
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;

export function evaluateMdpoTrainingUnifiedMetrics({ network, graphs, forward, labelsFromOutputs, evaluateMetrics, views, protocol } = {}) {
  if (!network || !Array.isArray(graphs) || !graphs.length || typeof forward !== 'function'
      || typeof labelsFromOutputs !== 'function' || typeof evaluateMetrics !== 'function'
      || !Array.isArray(views) || views.length !== 5 || !protocol) throw new Error('MDPO train unified metrics require model, graphs, five views and fixed evaluator');
  const rows = graphs.map((graph) => {
    const labels = labelsFromOutputs(forward(network, graph).output, graph);
    const perView = Object.fromEntries(views.map((view) => [view, evaluateMetrics({ labels, manualLabels: graph.manualLabels,
      geometry: graph.geometry, bounds: graph.bounds, view, dbvAvailable: true })]));
    if (Object.values(perView).some((metrics) => MDPO_UNIFIED_NINE_METRICS.some((name) => !Number.isFinite(metrics?.[name]))))
      throw new Error(`MDPO train unified metrics incomplete: ${graph.category}/${graph.sample_id}`);
    const metrics = Object.fromEntries(MDPO_UNIFIED_NINE_METRICS.map((name) => [name, mean(views.map((view) => perView[view][name]))]));
    return { sample: `${graph.category}/${graph.sample_id}`, metrics, per_view: perView };
  });
  const summary = Object.fromEntries(MDPO_UNIFIED_NINE_METRICS.map((name) => [name, name === 'intersections'
    ? rows.reduce((sum, row) => sum + row.metrics[name], 0) : mean(rows.map((row) => row.metrics[name]))]));
  return { version: 'v10_mdpo_train_unified_metrics_v1', split: 'train', sample_count: rows.length,
    views: [...views], metric_protocol: protocol, metrics: summary, samples: rows };
}

export function compareMdpoUnifiedMetrics(baseline, candidate) {
  if (baseline?.sample_count !== candidate?.sample_count || baseline?.metric_protocol !== candidate?.metric_protocol
      || JSON.stringify(baseline?.views) !== JSON.stringify(candidate?.views)) throw new Error('MDPO train baseline/candidate unified metrics protocol mismatch');
  return Object.fromEntries(MDPO_UNIFIED_NINE_METRICS.map((name) => {
    const left = baseline.metrics?.[name], right = candidate.metrics?.[name];
    if (!Number.isFinite(left) || !Number.isFinite(right)) throw new Error(`MDPO train unified delta missing ${name}`);
    return [name, right - left];
  }));
}
