import { MDPO_DIMENSIONS } from './mdpo-continuous-policy.mjs';
import { assessSafetyEligibility, compareSafetyReference } from './preference-policy.mjs';
import { strictMdpoSafety } from './mdpo-collection.mjs';

const VIEWS = Object.freeze(['before', 'main', 'right', 'left', 'up', 'down']);
const SAFETY_METRICS = Object.freeze([
  'objective_score', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio',
  'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio',
  'multi_view_worst_olr', 'multi_view_worst_overflow', 'text_clarity',
  'leader_crossings', 'worst_view_leader_crossing_count', 'weighted_leader_crossing_risk',
  'worst_view_leader_crossing_risk', 'cvar_view_leader_crossing_risk'
]);

function checkCandidate(candidate, record, prefix) {
  if (!candidate || !Number.isInteger(candidate.seed) || !candidate.candidate_id
    || !Array.isArray(candidate.local_layout) || !candidate.local_layout.length
    || candidate.local_layout.some((row) => !Array.isArray(row) || row.length !== 6 || row.some((value) => !Number.isFinite(value)))) throw new Error(`${prefix} missing finite Nx6 local layout / seed / candidate_id`);
  if (!Array.isArray(candidate.label_ids) || candidate.label_ids.length !== candidate.local_layout.length
    || new Set(candidate.label_ids).size !== candidate.label_ids.length) throw new Error(`${prefix} missing ordered unique label IDs`);
  if (MDPO_DIMENSIONS.some((name) => !Number.isFinite(candidate.scores?.[name]) || candidate.scores[name] < 1 || candidate.scores[name] > 5)) throw new Error(`${prefix} missing seven Qwen 1–5 scores`);
  if (candidate.scorer_model !== record.scorer_model || candidate.prompt_version !== record.prompt_version
    || typeof candidate.response_id !== 'string' || !candidate.response_id.length) throw new Error(`${prefix} missing scorer/prompt/response provenance`);
  if (VIEWS.some((view) => !/^[a-f0-9]{64}$/i.test(candidate.view_sha256?.[view] || ''))) throw new Error(`${prefix} missing six verified view hashes`);
  const metrics = candidate.safety?.metrics;
  const reference = candidate.safety?.reference_metrics;
  if (!metrics || !reference || SAFETY_METRICS.some((name) => !Number.isFinite(metrics[name]) || !Number.isFinite(reference[name]))) throw new Error(`${prefix} missing measured deterministic safety metrics`);
  const assessment = assessSafetyEligibility(metrics, reference);
  if (!assessment.eligible || candidate.safety.eligible !== true) throw new Error(`${prefix} failed deterministic safety gate: ${assessment.violations.join(',')}`);
  const strict = strictMdpoSafety(metrics);
  if (!strict.eligible) throw new Error(`${prefix} failed strict MDPO zero-crossing/penetration/mesh gate: ${strict.violations.join(',')}`);
  if (candidate.label_contract_valid !== true) throw new Error(`${prefix} fixed label contract not verified`);
}

export function validateMdpoDataset(dataset, manifest, { requireComplete = true } = {}) {
  if (dataset?.version !== 'v10_mdpo_train_pairs_v1' || !Array.isArray(dataset.pairs)) throw new Error('Expected versioned v10 MDPO dataset');
  const train = new Map((manifest?.samples || []).filter((sample) => sample.split === 'train').map((sample) => [`${sample.category}/${sample.sample_id}`, sample]));
  const group = new Map();
  let scorer = null, prompt = null;
  for (const pair of dataset.pairs) {
    if (pair.split !== 'train') throw new Error('MDPO dataset cannot contain val or test pairs');
    const key = `${pair.category}/${pair.sample_id}`;
    const sample = train.get(key);
    if (!sample || pair.clean_obj_source !== sample.input?.source_obj) throw new Error(`Non-train or mismatched OBJ source: ${key}`);
    if (!pair.scorer_model || !pair.prompt_version || (scorer && scorer !== pair.scorer_model) || (prompt && prompt !== pair.prompt_version)) throw new Error('MDPO scorer model and prompt must be fixed across train33');
    scorer = pair.scorer_model; prompt = pair.prompt_version;
    checkCandidate(pair.candidate_a, pair, `${key}/A`);
    checkCandidate(pair.candidate_b, pair, `${key}/B`);
    const safer = compareSafetyReference(pair.candidate_a.safety.metrics, pair.candidate_b.safety.metrics) <= 0 ? pair.candidate_a : pair.candidate_b;
    const other = safer === pair.candidate_a ? pair.candidate_b : pair.candidate_a;
    const pairSafety = assessSafetyEligibility(other.safety.metrics, safer.safety.metrics);
    if (!pairSafety.eligible) throw new Error(`${key}: MDPO pair is not deterministic-safety-compatible: ${pairSafety.violations.join(',')}`);
    if (pair.candidate_a.candidate_id === pair.candidate_b.candidate_id || JSON.stringify(pair.candidate_a.local_layout) === JSON.stringify(pair.candidate_b.local_layout)) throw new Error(`${key}: degenerate duplicate layout pair`);
    if (JSON.stringify(pair.candidate_a.label_ids) !== JSON.stringify(pair.candidate_b.label_ids)) throw new Error(`${key}: label ordering differs between candidates`);
    const scoresA = pair.candidate_a.scores, scoresB = pair.candidate_b.scores;
    const margins = Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, (scoresA[name] - scoresB[name]) / 4]));
    if (!pair.dimension_margins || Object.keys(pair.dimension_margins).length !== MDPO_DIMENSIONS.length
        || MDPO_DIMENSIONS.some((name) => !Number.isFinite(pair.dimension_margins[name])
          || Math.abs(pair.dimension_margins[name] - margins[name]) > 1e-12)) {
      throw new Error(`${key}: saved seven-dimensional Qwen preference margins differ from candidate scores`);
    }
    if (MDPO_DIMENSIONS.every((name) => Math.abs(margins[name]) <= 0.025)) throw new Error(`${key}: all seven MDPO dimensions are tied`);
    if (!group.has(key)) group.set(key, { category: pair.category, pair_count: 0, candidate_ids: new Set() });
    const item = group.get(key);
    item.pair_count++;
    item.candidate_ids.add(pair.candidate_a.candidate_id);
    item.candidate_ids.add(pair.candidate_b.candidate_id);
  }
  if (requireComplete && (train.size !== 33 || group.size !== 33 || dataset.pairs.length < 198 || dataset.pairs.length > 396
      || [...group.values()].some(({ pair_count, candidate_ids }) => pair_count < 6 || pair_count > 12 || candidate_ids.size < 8))) {
    throw new Error(`Incomplete MDPO train33: ${group.size}/${train.size} objects and ${dataset.pairs.length} pairs; require 8 unique candidates and 6–12 pairs per object`);
  }
  return { split: 'train', sample_count: group.size, pair_count: dataset.pairs.length, scorer_model: scorer,
    prompt_version: prompt, category_pair_counts: Object.fromEntries([...group.values()].reduce((map, item) => {
      map.set(item.category, (map.get(item.category) || 0) + item.pair_count); return map;
    }, new Map())) };
}
