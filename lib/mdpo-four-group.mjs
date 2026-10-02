export const MDPO_VAL11_GROUPS = Object.freeze({
  v10_no_rerank: Object.freeze({ model_role: 'baseline', preference_rerank: false, label: 'v10 · reranker off', core_gate_role: 'baseline' }),
  v10_historical_rerank: Object.freeze({ model_role: 'baseline', preference_rerank: true, label: 'v10 + historical reward MLP', core_gate_role: null }),
  mdpo_no_rerank: Object.freeze({ model_role: 'candidate', preference_rerank: false, label: 'v10-MDPO · reranker off', core_gate_role: 'candidate' }),
  mdpo_safe_rerank: Object.freeze({ model_role: 'candidate', preference_rerank: true, label: 'v10-MDPO + safe reward reranking', core_gate_role: null })
});
const VIEWS = Object.freeze(['before', 'main', 'right', 'left', 'up', 'down']);
const SCORES = Object.freeze(['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity', 'text_clarity', 'leader_line_clarity']);
const METRICS = Object.freeze(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score', 'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow']);
const STRATEGY = Object.freeze(['generator', 'viewPolicy', 'groupPolicy', 'sizePolicy', 'optimizer', 'seed', 'iterations', 'preferenceRerank']);
const sampleKey = (sample) => `${sample?.category}/${sample?.sample_id}`;
const staticStrategy = (strategy) => Object.fromEntries(STRATEGY.filter((name) => name !== 'preferenceRerank').map((name) => [name, strategy?.[name]]));

export function mdpoVal11Group(group) {
  const value = MDPO_VAL11_GROUPS[String(group || '')];
  if (!value) throw new Error(`Invalid MDPO four-group val11 group: ${group}`);
  return value;
}

export function validateMdpoFourGroupSummaries(groups) {
  const names = Object.keys(MDPO_VAL11_GROUPS);
  if (!groups || names.some((name) => !groups[name])) throw new Error('MDPO four-group report is incomplete');
  const cohort = JSON.stringify(groups.v10_no_rerank.cohort);
  for (const name of names) {
    const expected = MDPO_VAL11_GROUPS[name], row = groups[name];
    if (row.group !== name || row.label !== expected.label || row.model_role !== expected.model_role
        || row.preference_rerank !== expected.preference_rerank || row.sample_count !== 11
        || !Array.isArray(row.cohort) || row.cohort.length !== 11 || new Set(row.cohort).size !== 11
        || JSON.stringify(row.cohort) !== cohort || row.test_used_for_selection !== false
        || !Array.isArray(row.sample_scores) || row.sample_scores.length !== 11) throw new Error(`MDPO four-group protocol mismatch: ${name}`);
    if (expected.preference_rerank && !/^[a-f0-9]{64}$/i.test(row.preference_model_sha256 || '')) throw new Error(`MDPO ${name} missing historical reward-model hash`);
    if (!expected.preference_rerank && row.preference_model_sha256 !== null) throw new Error(`MDPO ${name} unexpectedly used a reward model`);
  }
  if (groups.v10_historical_rerank.preference_model_sha256 !== groups.mdpo_safe_rerank.preference_model_sha256) throw new Error('MDPO reranked groups did not use the same historical reward model');
  const protocolRows = [];
  const scorerModels = new Set(names.map((name) => groups[name].scorer_model));
  const promptVersions = new Set(names.map((name) => groups[name].prompt_version));
  const metricProtocols = new Set(names.map((name) => groups[name].metric_protocol));
  if (scorerModels.size !== 1 || promptVersions.size !== 1 || metricProtocols.size !== 1
      || ![...scorerModels][0] || ![...promptVersions][0] || ![...metricProtocols][0]
      || names.some((name) => JSON.stringify(groups[name].views) !== JSON.stringify(VIEWS)))
    throw new Error('MDPO four-group scorer/prompt/six-view/metric protocol mismatch');
  for (let index = 0; index < 11; index++) {
    const sample = groups.v10_no_rerank.cohort[index];
    const entries = names.map((name) => ({ name, spec: MDPO_VAL11_GROUPS[name], entry: groups[name].sample_scores[index] }));
    for (const { name, spec, entry } of entries) {
      if (sampleKey(entry?.sample) !== sample || !entry.response_id
          || SCORES.some((dimension) => !Number.isFinite(entry.scores?.[dimension]))
          || METRICS.some((metric) => !Number.isFinite(entry.metrics?.[metric]))
          || VIEWS.some((view) => !/^[a-f0-9]{64}$/i.test(entry.view_sha256?.[view] || ''))
          || STRATEGY.some((field) => entry.generation_strategy?.[field] === undefined)
          || entry.generation_strategy.preferenceRerank !== spec.preference_rerank
          || !Number.isInteger(entry.generation_strategy.seed) || !Number.isInteger(entry.generation_strategy.iterations)
          || entry.evaluation?.group !== name || entry.evaluation?.role !== spec.model_role
          || entry.security?.test_used !== false || entry.security?.train_preference_created !== false
          || entry.security?.qwen_inference_input !== false
          || (spec.preference_rerank ? entry.preference_model_sha256 !== groups[name].preference_model_sha256 : entry.preference_model_sha256 !== null))
        throw new Error(`MDPO four-group sample evidence mismatch: ${name}/${sample}`);
    }
    const seeds = new Set(entries.map(({ entry }) => entry.generation_strategy.seed));
    const protocols = new Set(entries.map(({ entry }) => JSON.stringify(staticStrategy(entry.generation_strategy))));
    const references = new Set(entries.map(({ entry }) => entry.evaluation.reference_model_sha256));
    const baselineHashes = new Set(entries.filter(({ spec }) => spec.model_role === 'baseline').map(({ entry }) => entry.evaluation.candidate_sha256));
    const candidateHashes = new Set(entries.filter(({ spec }) => spec.model_role === 'candidate').map(({ entry }) => entry.evaluation.candidate_sha256));
    const candidateFiles = new Set(entries.filter(({ spec }) => spec.model_role === 'candidate').map(({ entry }) => entry.evaluation.candidate_file));
    if (seeds.size !== 1 || protocols.size !== 1 || references.size !== 1 || baselineHashes.size !== 1 || candidateHashes.size !== 1
        || candidateFiles.size !== 1 || [...entries.filter(({ spec }) => spec.model_role === 'baseline').map(({ entry }) => entry.evaluation.candidate_file)].some((file) => file !== null)
        || !/^[a-f0-9]{64}$/i.test([...references][0] || '') || [...baselineHashes][0] !== [...references][0]
        || !/^[a-f0-9]{64}$/i.test([...candidateHashes][0] || '') || ![...candidateFiles][0])
      throw new Error(`MDPO four-group paired generation/model protocol mismatch: ${sample}`);
    protocolRows.push({ sample, seed: [...seeds][0], static_strategy: JSON.parse([...protocols][0]),
      reference_model_sha256: [...references][0], candidate_sha256: [...candidateHashes][0], candidate_file: [...candidateFiles][0] });
  }
  if (new Set(protocolRows.map((row) => JSON.stringify({ ...row.static_strategy, seed: null }))).size !== 1
      || new Set(protocolRows.map((row) => row.reference_model_sha256)).size !== 1
      || new Set(protocolRows.map((row) => row.candidate_sha256)).size !== 1
      || new Set(protocolRows.map((row) => row.candidate_file)).size !== 1)
    throw new Error('MDPO four-group cohort changed generator protocol or frozen model');
  return { complete: true, groups: names, cohort: groups.v10_no_rerank.cohort,
    historical_reward_model_sha256: groups.v10_historical_rerank.preference_model_sha256,
    paired_sample_protocol: protocolRows };
}
