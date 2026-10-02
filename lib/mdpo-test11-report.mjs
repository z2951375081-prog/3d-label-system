import { createHash } from 'node:crypto';
import { MDPO_VAL11_GROUPS } from './mdpo-four-group.mjs';

const VIEWS = Object.freeze(['before', 'main', 'right', 'left', 'up', 'down']);
const SCORES = Object.freeze(['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity', 'text_clarity', 'leader_line_clarity']);
const METRICS = Object.freeze(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score', 'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow']);
const STRATEGY = Object.freeze(['generator', 'viewPolicy', 'groupPolicy', 'sizePolicy', 'optimizer', 'seed', 'iterations', 'preferenceRerank']);
const hash = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export function validateMdpoTest11Records({ groups, cohort, runId, lockSha256, baselineSha256, activeSha256, preferenceSha256, scorerModel, promptVersion, metricProtocol, allowIncomplete = false } = {}) {
  if (!Array.isArray(cohort) || cohort.length !== 11 || new Set(cohort).size !== 11 || !/^[a-zA-Z0-9_-]{8,100}$/.test(String(runId || ''))
      || ![lockSha256, baselineSha256, activeSha256, preferenceSha256].every((item) => /^[a-f0-9]{64}$/i.test(item || ''))
      || ![scorerModel, promptVersion, metricProtocol].every((item) => typeof item === 'string' && item.length)) throw new Error('MDPO test11 immutable identity is incomplete');
  const identities = [];
  for (const [groupName, spec] of Object.entries(MDPO_VAL11_GROUPS)) {
    const rows = groups?.[groupName];
    if (!Array.isArray(rows) || (allowIncomplete ? rows.length > 11 : rows.length !== 11)) throw new Error(`MDPO test11 ${groupName} requires ${allowIncomplete ? 'at most' : 'exactly'} 11 rows`);
    const bySample = new Map(rows.map((row) => [`${row.sample?.category}/${row.sample?.sample_id}`, row]));
    if (bySample.size !== rows.length || [...bySample.keys()].some((sample) => !cohort.includes(sample))) throw new Error(`MDPO test11 ${groupName} contains duplicate or unexpected samples`);
    for (const sample of cohort.filter((item) => bySample.has(item))) {
      const row = bySample.get(sample);
      if (row?.version !== 'v10_mdpo_test11_sample_v1' || row.split !== 'test' || row.group !== groupName || row.role !== spec.model_role
          || row.test_run_id !== runId || row.lock_sha256 !== lockSha256 || `${row.sample?.category}/${row.sample?.sample_id}` !== sample) throw new Error(`MDPO test11 ${groupName}/${sample} identity mismatch`);
      if (row.scorer?.model !== scorerModel || row.scorer?.prompt_version !== promptVersion || !row.scorer?.response_id
          || row.metric_protocol !== metricProtocol || JSON.stringify(row.views) !== JSON.stringify(VIEWS)
          || VIEWS.some((view) => !/^[a-f0-9]{64}$/i.test(row.view_sha256?.[view] || ''))) throw new Error(`MDPO test11 ${groupName}/${sample} Qwen/view provenance mismatch`);
      if (SCORES.some((name) => !Number.isFinite(row.scores?.[name]) || row.scores[name] < 1 || row.scores[name] > 5)
          || METRICS.some((name) => !Number.isFinite(row.metrics?.[name]))) throw new Error(`MDPO test11 ${groupName}/${sample} scores or metrics incomplete`);
      if (row.evaluation?.phase !== 'test11' || row.evaluation?.test_run_id !== runId || row.evaluation?.lock_sha256 !== lockSha256
          || row.evaluation?.role !== spec.model_role || row.evaluation?.group !== groupName
          || row.evaluation?.reference_model_sha256 !== baselineSha256
          || row.evaluation?.candidate_sha256 !== (spec.model_role === 'candidate' ? activeSha256 : baselineSha256)) throw new Error(`MDPO test11 ${groupName}/${sample} locked model mismatch`);
      if (!row.generation_strategy || STRATEGY.some((name) => row.generation_strategy[name] === undefined)
          || row.generation_strategy.preferenceRerank !== spec.preference_rerank || !Number.isInteger(row.generation_strategy.seed)) throw new Error(`MDPO test11 ${groupName}/${sample} generation protocol incomplete`);
      const rewardHash = row.preference_model?.model_sha256 ?? null;
      if (spec.preference_rerank ? rewardHash !== preferenceSha256 : rewardHash !== null) throw new Error(`MDPO test11 ${groupName}/${sample} reward-model provenance mismatch`);
      if (row.security?.test_used !== true || row.security?.test_used_for_training !== false || row.security?.test_used_for_selection !== false
          || row.security?.train_preference_created !== false || row.security?.qwen_inference_input !== false) throw new Error(`MDPO test11 ${groupName}/${sample} split or inference leak`);
      identities.push({ group: groupName, sample, response_id: row.scorer.response_id, view_sha256: row.view_sha256 });
    }
  }
  for (let index = 0; index < cohort.length; index += 1) {
    const rows = Object.keys(MDPO_VAL11_GROUPS).map((name) => groups[name].find((row) => `${row.sample?.category}/${row.sample?.sample_id}` === cohort[index])).filter(Boolean);
    if (!rows.length) continue;
    if (!allowIncomplete && rows.length !== Object.keys(MDPO_VAL11_GROUPS).length) throw new Error(`MDPO test11 ${cohort[index]} is missing groups`);
    const seeds = new Set(rows.map((row) => row.generation_strategy.seed));
    if (seeds.size !== 1) throw new Error(`MDPO test11 ${cohort[index]} groups used different seeds`);
    const staticStrategy = (row) => Object.fromEntries(STRATEGY.filter((name) => !['seed', 'preferenceRerank'].includes(name)).map((name) => [name, row.generation_strategy[name]]));
    if (new Set(rows.map((row) => hash(staticStrategy(row)))).size !== 1) throw new Error(`MDPO test11 ${cohort[index]} groups used different generation protocols`);
  }
  const recordCount = Object.values(groups).reduce((sum, rows) => sum + rows.length, 0);
  if (!allowIncomplete && recordCount !== 44) throw new Error('MDPO test11 requires all 44 immutable records');
  return { version: 'v10_mdpo_test11_provenance_v1', sample_count: 11, group_count: 4, record_count: recordCount, complete: recordCount === 44,
    cohort_sha256: hash(cohort), evidence_sha256: hash(identities), test_used_for_training: false, test_used_for_selection: false };
}
