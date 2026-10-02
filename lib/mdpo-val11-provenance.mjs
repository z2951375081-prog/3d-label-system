import { createHash } from 'node:crypto';

const VIEWS = Object.freeze(['before', 'main', 'right', 'left', 'up', 'down']);
const SCORES = Object.freeze(['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity', 'text_clarity', 'leader_line_clarity']);
const METRICS = Object.freeze(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score', 'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow']);
const STRATEGY = Object.freeze(['generator', 'viewPolicy', 'groupPolicy', 'sizePolicy', 'optimizer', 'seed', 'iterations', 'preferenceRerank']);
const hash = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export function validateMdpoVal11Records({ baseline, candidate, expected, candidateId, referenceSha256, candidateSha256, scorerModel, promptVersion, metricProtocol } = {}) {
  if (!Array.isArray(expected) || expected.length !== 11 || new Set(expected).size !== 11
      || !Array.isArray(baseline) || baseline.length !== 11 || !Array.isArray(candidate) || candidate.length !== 11) throw new Error('MDPO val11 requires exactly 11 unique paired val samples');
  if (![candidateId, scorerModel, promptVersion, metricProtocol].every((item) => typeof item === 'string' && item.length)
      || ![referenceSha256, candidateSha256].every((item) => /^[a-f0-9]{64}$/i.test(item || ''))) throw new Error('MDPO val11 missing immutable comparison identity');
  const identities = [], strategyHashes = [];
  for (let index = 0; index < 11; index++) {
    const sample = expected[index];
    for (const [role, row] of [['baseline', baseline[index]], ['candidate', candidate[index]]]) {
      const expectedGroup = role === 'baseline' ? 'v10_no_rerank' : 'mdpo_no_rerank';
      if (row?.version !== 'v10_mdpo_val11_sample_v1' || row.split !== 'val' || row.group !== expectedGroup || row.role !== role || row.candidate_id !== candidateId
          || `${row.sample?.category}/${row.sample?.sample_id}` !== sample) throw new Error(`MDPO ${role}/${sample} sample identity mismatch`);
      if (row.scorer?.model !== scorerModel || row.scorer?.prompt_version !== promptVersion || !row.scorer.response_id
          || row.metric_protocol !== metricProtocol || JSON.stringify(row.views) !== JSON.stringify(VIEWS)
          || VIEWS.some((view) => !/^[a-f0-9]{64}$/i.test(row.view_sha256?.[view] || ''))) throw new Error(`MDPO ${role}/${sample} Qwen/view/metric provenance mismatch`);
      if (SCORES.some((name) => !Number.isFinite(row.scores?.[name]) || row.scores[name] < 1 || row.scores[name] > 5)
          || METRICS.some((name) => !Number.isFinite(row.metrics?.[name]))) throw new Error(`MDPO ${role}/${sample} scores or deterministic metrics incomplete`);
      if (row.evaluation?.candidate_id !== candidateId || row.evaluation.role !== role
          || row.evaluation.reference_model_sha256 !== referenceSha256
          || row.evaluation.candidate_sha256 !== (role === 'candidate' ? candidateSha256 : referenceSha256)
          || (role === 'baseline' && row.evaluation.candidate_file !== null)
          || (role === 'candidate' && !row.evaluation.candidate_file)) throw new Error(`MDPO ${role}/${sample} checkpoint hash mismatch`);
      if (!row.generation_strategy || STRATEGY.some((field) => row.generation_strategy[field] === undefined)
          || row.generation_strategy.preferenceRerank !== false
          || !Number.isInteger(row.generation_strategy.seed) || !Number.isInteger(row.generation_strategy.iterations)) throw new Error(`MDPO ${role}/${sample} generation strategy incomplete`);
      if (row.security?.test_used !== false || row.security.train_preference_created !== false
          || row.security.qwen_inference_input !== false) throw new Error(`MDPO ${role}/${sample} test/train/Qwen inference leak`);
    }
    const left = baseline[index], right = candidate[index];
    if (right.evaluation.candidate_file !== candidate[0].evaluation.candidate_file
        || JSON.stringify(left.generation_strategy) !== JSON.stringify(right.generation_strategy)) throw new Error(`MDPO val11 ${sample} baseline/candidate generation mismatch`);
    // Seeds vary across samples, but all other generator settings must be identical.
    const staticStrategy = (row) => Object.fromEntries(STRATEGY.filter((key) => key !== 'seed').map((key) => [key, row.generation_strategy[key]]));
    if (hash(staticStrategy(left)) !== hash(staticStrategy(baseline[0]))
        || hash(staticStrategy(right)) !== hash(staticStrategy(baseline[0]))) throw new Error(`MDPO val11 ${sample} generator protocol differs across cohort`);
    strategyHashes.push(hash(left.generation_strategy));
    identities.push({ sample, seed: left.generation_strategy.seed,
      baseline_response_id: left.scorer.response_id, candidate_response_id: right.scorer.response_id,
      baseline_views: left.view_sha256, candidate_views: right.view_sha256 });
  }
  return { version: 'v10_mdpo_val11_paired_provenance_v1', paired_samples: 11, cohort_sha256: hash(identities.map((item) => item.sample)),
    generation_protocol_sha256: hash({ static: Object.fromEntries(STRATEGY.filter((key) => key !== 'seed').map((key) => [key, baseline[0].generation_strategy[key]])), per_sample: strategyHashes }),
    scorer_model: scorerModel, prompt_version: promptVersion, metric_protocol: metricProtocol,
    reference_model_sha256: referenceSha256, candidate_sha256: candidateSha256, samples: identities };
}
