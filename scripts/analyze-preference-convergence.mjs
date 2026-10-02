import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { LLM_AESTHETIC_DIMENSIONS, LLM_SAFETY_DIMENSIONS, LLM_SCORE_NAMES } from '../public/preference-scoring.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = process.env.EXPERIMENTS_DIR ? path.resolve(process.env.EXPERIMENTS_DIR) : path.join(root, 'experiments');
const STANDARD_ROUNDS = [0, 1, 2, 4, 8];
const THRESHOLDS = { minimum_composite_improvement: 0.05, maximum_safety_dimension_drop: 0.15, maximum_round4_to_round8_composite_change: 0.1, maximum_round4_to_round8_mean_dimension_change: 0.15 };

function parseArgs(argv) {
  const options = { runId: '', output: path.join(experiments, 'llm_preference_convergence.json') };
  for (let index = 0; index < argv.length; index += 1) if (argv[index].startsWith('--')) {
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const readOptional = async (file) => { try { return await readJson(file); } catch (error) { if (error.code === 'ENOENT') return null; throw error; } };
const outputFile = path.resolve(String(options.output));
const latest = await readOptional(path.join(experiments, 'llm_preference_run_latest.json'));
const runId = String(options.runId || latest?.run_id || '');
if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('未找到有效的 LLM run_id；请用 --runId 指定');

const runReport = await readOptional(path.join(experiments, `llm_preference_run_${runId}.json`)) || (latest?.run_id === runId ? latest : null);
const visualReport = await readOptional(path.join(experiments, `llm_visual_validation_${runId}.json`));
const ledger = await readOptional(path.join(experiments, `llm_preference_checkpoints_${runId}.json`));
const testReportCandidate = await readOptional(path.join(experiments, 'preference_test11_report.json'));
const testReport = testReportCandidate?.run_id === runId ? testReportCandidate : null;
const comparison = await readOptional(path.join(experiments, 'comparisons', 'round1', 'comparison.json'));

const visualByRound = new Map((visualReport?.rounds || []).map((item) => [Number(item.round), item]));
const ledgerByRound = new Map((ledger?.checkpoints || []).map((item) => [Number(item.round), item]));
const sampleCohort = (item) => [...(item?.sample_scores || [])].map((row) => `${row.sample?.category}/${row.sample?.sample_id}`).sort();
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const roundRows = [];

for (const round of STANDARD_ROUNDS) {
  const visual = visualByRound.get(round) || null;
  const checkpoint = ledgerByRound.get(round) || null;
  let checkpointVerified = round === 0 ? visual?.checkpoint?.kind === 'no_reward_baseline' : false;
  let actualSha256 = null;
  let verificationError = null;
  if (round > 0 && visual && checkpoint) {
    try {
      const file = path.join(experiments, 'llm_preference_checkpoints', runId, `round_${round}.json`);
      const bytes = await fs.readFile(file);
      actualSha256 = createHash('sha256').update(bytes).digest('hex').toUpperCase();
      checkpointVerified = actualSha256 === checkpoint.reward_model_sha256 && actualSha256 === visual.checkpoint?.reward_model_sha256 && JSON.parse(bytes.toString('utf8')).training?.run_id_filter === runId;
      if (!checkpointVerified) verificationError = 'checkpoint hash or run_id mismatch';
    } catch (error) { verificationError = error.message; }
  } else if (round > 0 && visual) verificationError = 'visual score exists without checkpoint ledger entry';
  roundRows.push({
    round,
    status: visual ? 'evaluated' : 'unavailable',
    reason: visual ? null : `缺少第 ${round} 轮同一 val11 的真实六图十四维评分`,
    sample_count: visual?.sample_count ?? 0,
    scorer_model: visual?.scorer_model || null,
    composite_score: visual?.composite_score ?? null,
    score_means: visual?.score_means || null,
    geometry_proxy_means: visual?.geometry_proxy_means || null,
    sample_cohort: visual ? sampleCohort(visual) : [],
    checkpoint: visual?.checkpoint || null,
    checkpoint_ledger: checkpoint ? { reward_model_file: checkpoint.reward_model_file, reward_model_sha256: checkpoint.reward_model_sha256, validation_status: checkpoint.validation_status, activated: checkpoint.activated } : null,
    checkpoint_verified: Boolean(checkpointVerified),
    checkpoint_actual_sha256: actualSha256,
    verification_error: verificationError
  });
}

const evaluated = roundRows.filter((item) => item.status === 'evaluated');
for (let index = 1; index < evaluated.length; index += 1) {
  const previous = evaluated[index - 1], current = evaluated[index];
  current.delta_vs_previous = {
    previous_round: previous.round,
    composite_score: Number((current.composite_score - previous.composite_score).toFixed(6)),
    score_means: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, Number(((current.score_means?.[name] ?? 0) - (previous.score_means?.[name] ?? 0)).toFixed(6))]))
  };
}

const baseline = visualByRound.get(0) || null;
const standardVisuals = STANDARD_ROUNDS.map((round) => visualByRound.get(round) || null);
const referenceCohort = sampleCohort(standardVisuals[0]);
const referenceModel = standardVisuals[0]?.scorer_model || null;
const fullSameCohort = standardVisuals.every((item) => item?.sample_count === 11 && item.scorer_model === referenceModel && sameJson(sampleCohort(item), referenceCohort));
const hashesVerified = roundRows.every((item) => item.status !== 'evaluated' || item.checkpoint_verified);
const evidenceSufficient = Boolean(referenceModel && fullSameCohort && hashesVerified);
const trained = [1, 2, 4, 8].map((round) => visualByRound.get(round)).filter(Boolean);
const peak = [...trained].sort((left, right) => Number(right.composite_score) - Number(left.composite_score))[0] || null;
const improved = Boolean(evidenceSufficient && baseline && peak && peak.composite_score >= baseline.composite_score + THRESHOLDS.minimum_composite_improvement && LLM_SAFETY_DIMENSIONS.every((name) => (peak.score_means?.[name] ?? 0) >= (baseline.score_means?.[name] ?? 0) - THRESHOLDS.maximum_safety_dimension_drop));
const round4 = visualByRound.get(4) || null, round8 = visualByRound.get(8) || null;
const meanDimensionChange4to8 = round4 && round8 ? LLM_AESTHETIC_DIMENSIONS.reduce((sum, name) => sum + Math.abs((round8.score_means?.[name] ?? 0) - (round4.score_means?.[name] ?? 0)), 0) / LLM_AESTHETIC_DIMENSIONS.length : null;
const plateau4to8 = Boolean(evidenceSufficient && round4 && round8 && Math.abs(round8.composite_score - round4.composite_score) <= THRESHOLDS.maximum_round4_to_round8_composite_change && meanDimensionChange4to8 <= THRESHOLDS.maximum_round4_to_round8_mean_dimension_change);
const riseThenStable = evidenceSufficient && improved && plateau4to8;

const unified = comparison?.unified_snapshot_evaluation || {};
const benchmarkMethods = ['current_fixed_label_seed17', 'dataset_camera_candidate_seed17', 'hedgehog_1d', 'hedgehog_3d', 'BinoForce_final_snapshot'];
const benchmarkSummary = (unified.summary || []).filter((row) => benchmarkMethods.includes(row.method)).map((row) => ({
  method: row.method, sample_count: row.sample_count, multidimensional_quality_score: row.multidimensional_quality_score,
  objective_score: row.objective_score, text_clarity: row.text_clarity,
  label_object_occlusion_ratio: row.label_object_occlusion_ratio, object_penetration_ratio: row.object_penetration_ratio,
  mesh_surface_intersection_ratio: row.mesh_surface_intersection_ratio, manual_style_distance: row.manual_style_distance
}));
const benchmarkBootstrap = (unified.paired_bootstrap?.comparisons || []).filter((row) => ['hedgehog_1d', 'hedgehog_3d'].includes(row.other_method) && ['multidimensional_quality_score', 'objective_score', 'text_clarity', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'manual_style_distance'].includes(row.metric)).map((row) => ({
  reference_method: row.reference_method, other_method: row.other_method, metric: row.metric, direction: row.direction,
  mean_advantage_reference_better: row.mean_advantage_reference_better, wins_ties_losses: [row.reference_wins, row.ties, row.reference_losses], bootstrap_95_ci: row.bootstrap_95_ci
}));

const report = {
  version: 'llm_preference_convergence_v4_aesthetic_reward_safety_gate', generated_at: new Date().toISOString(), run_id: runId,
  primary_evidence: 'independent_val11_six_image_external_fourteen_dimension_scores_with_aesthetic_only_preference',
  evidence_sufficient: evidenceSufficient, improved_before_plateau: improved, plateau_round4_to_round8: plateau4to8, rise_then_stable: riseThenStable,
  thresholds: THRESHOLDS, standard_rounds: STANDARD_ROUNDS, scorer_model: referenceModel, validation_sample_cohort: referenceCohort,
  checkpoints: roundRows,
  round4_to_round8: { composite_change: round4 && round8 ? Number((round8.composite_score - round4.composite_score).toFixed(6)) : null, mean_absolute_dimension_change: meanDimensionChange4to8 === null ? null : Number(meanDimensionChange4to8.toFixed(6)) },
  selected_visual_peak: peak ? { round: peak.round, composite_score: peak.composite_score, score_means: peak.score_means } : null,
  final_test11: testReport ? { selected_round: testReport.selected_round, selection_policy: testReport.selection_policy, test_samples: testReport.test_samples, candidate_activated: testReport.candidate_activated, summary: testReport.summary } : null,
  benchmark_context: {
    protocol: comparison?.camera_protocol || null,
    note: 'BinoForce/Hedgehog are deterministic unified-camera geometry benchmarks; their metric scale is not interchangeable with external LLM 1-5 scores.',
    summary: benchmarkSummary, paired_bootstrap: benchmarkBootstrap
  },
  interpretation: !evidenceSufficient ? '缺少同一外部模型、同一 val11 样本集的完整 0/1/2/4/8 六图十四维评分，不能判断美学偏好是否先提升后稳定。' : riseThenStable ? '真实 val11 六图美学综合分先提升，且第4到第8轮进入预设稳定阈值；安全诊断未越过退化阈值。' : '已有完整真实 val11 六图十四维证据，但不满足预先固定的“美学先提升且第4到第8轮稳定”联合判据。',
  policy: { reward_target: 'aesthetic_dimensions_only', aesthetic_dimensions: LLM_AESTHETIC_DIMENSIONS, safety_dimensions_diagnostic_only: LLM_SAFETY_DIMENSIONS, deterministic_energy_safety_gate: true },
  provenance: { visual_report_file: visualReport ? `experiments/llm_visual_validation_${runId}.json` : null, checkpoint_ledger_file: ledger ? `experiments/llm_preference_checkpoints_${runId}.json` : null, historical_run_report_file: runReport ? `experiments/llm_preference_run_${runId}.json` : null },
  security: { external_requests_made_by_this_analysis: 0, api_key_used: false, geometry_proxy_used_as_visual_convergence_evidence: false, test_not_used_to_select_or_train: true }
};

const previous = await readOptional(outputFile);
if (previous?.version === 'llm_preference_convergence_v1') {
  const legacyFile = path.join(path.dirname(outputFile), `llm_preference_convergence_legacy_geometry_${runId}.json`);
  try { await fs.writeFile(legacyFile, JSON.stringify(previous, null, 2) + '\n', { encoding: 'utf8', flag: 'wx' }); } catch (error) { if (error.code !== 'EEXIST') throw error; }
  report.provenance.legacy_geometry_report_preserved = path.relative(root, legacyFile).split(path.sep).join('/');
}
await fs.mkdir(path.dirname(outputFile), { recursive: true });
await fs.writeFile(outputFile, JSON.stringify(report, null, 2) + '\n', 'utf8');
if (path.resolve(experiments, `llm_preference_convergence_${runId}.json`) === outputFile) {
  await fs.writeFile(path.join(experiments, 'llm_preference_convergence_latest.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
}
console.log(JSON.stringify({ output: path.relative(root, outputFile).split(path.sep).join('/'), run_id: runId, evidence_sufficient: evidenceSufficient, rise_then_stable: riseThenStable, missing_rounds: roundRows.filter((item) => item.status !== 'evaluated').map((item) => item.round), scorer_model: referenceModel, external_requests: 0 }, null, 2));
