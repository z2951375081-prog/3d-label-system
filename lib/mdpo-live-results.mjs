import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const readJson = (file) => fs.readFile(file, 'utf8').then(JSON.parse);
const readOptionalJson = (file) => readJson(file).catch((error) => {
  if (error?.code === 'ENOENT') return null;
  throw error;
});
const processAlive = (pid) => {
  if (!Number.isInteger(Number(pid)) || Number(pid) <= 0) return false;
  try { process.kill(Number(pid), 0); return true; }
  catch (error) { return error?.code !== 'ESRCH'; }
};

async function parseLastEpoch(directory, root) {
  const logFile = path.join(directory, 'stdout.log');
  const [text, stat, checkpointNames] = await Promise.all([
    fs.readFile(logFile, 'utf8').catch(() => ''),
    fs.stat(logFile).catch(() => null),
    fs.readdir(path.join(directory, 'checkpoints')).catch(() => [])
  ]);
  const lines = text.split(/\r?\n/).filter(Boolean);
  const latestCheckpoint = checkpointNames.filter((name) => /^epoch_\d+\.json$/.test(name)).sort().at(-1);
  const checkpoint = latestCheckpoint
    ? await fs.readFile(path.join(directory, 'checkpoints', latestCheckpoint), 'utf8').then(JSON.parse, () => null)
    : null;
  const checkpointHistory = checkpoint?.history?.at(-1) || null;
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const row = JSON.parse(lines[index]);
      if (row.progress === 'epoch' && Number.isFinite(Number(row.epoch))) return {
        epoch: Number(row.epoch), epochs: Number(row.epochs), train_total: Number(row.train_total),
        geometry_safe: Number(checkpointHistory?.train?.geometry_safe ?? row.train?.geometry_safe),
        val_total: Number(row.val?.total), val_mse: Number(row.val?.mse),
        center_tail_cvar: Number(row.val?.center_tail_cvar), best_epoch: Number(row.best_epoch),
        stale: Number(row.stale), updated_at: stat?.mtime?.toISOString() || null,
        source: path.relative(root, logFile).split(path.sep).join('/')
      };
    } catch {}
  }
  return null;
}

const ALIGNED_VAL11_GROUPS = [
  { group: 'v10_no_rerank', label: '原始 v10（不重排）', model_role: 'baseline', preference_rerank: false },
  { group: 'v10_historical_rerank', label: 'v10 + 基础奖励 MLP', model_role: 'baseline', preference_rerank: true },
  { group: 'mdpo_no_rerank', label: 'v10 + MDPO（不重排）', model_role: 'candidate', preference_rerank: false },
  { group: 'mdpo_safe_rerank', label: 'v10 + MDPO + 基础奖励 MLP', model_role: 'candidate', preference_rerank: true }
];
const ALIGNED_LIVE_METRICS = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];

async function alignedVal11CandidateMetrics({ mdpo, candidate, group = 'mdpo_no_rerank' }) {
  if (!candidate?.id) return null;
  const definition = ALIGNED_VAL11_GROUPS.find((item) => item.group === group);
  if (!definition) throw new Error('Unknown aligned val11 group: ' + group);
  const directory = path.join(mdpo, 'val11_visual', candidate.id, group);
  const names = await fs.readdir(directory).catch((error) => {
    if (error?.code === 'ENOENT') return [];
    throw error;
  });
  const rows = [];
  for (const name of names.filter((item) => item.endsWith('.json'))) {
    try {
      const [row, stat] = await Promise.all([readJson(path.join(directory, name)), fs.stat(path.join(directory, name))]);
      if (row.version !== 'v10_mdpo_val11_sample_v1' || row.split !== 'val' || row.candidate_id !== candidate.id
          || row.group !== group || row.role !== definition.model_role
          || row.metric_protocol !== 'unified_reproduction_metrics_v3_safety_weighted_low_pck'
          || row.security?.test_used !== false || row.security?.qwen_inference_input !== false) continue;
      if (definition.model_role === 'candidate'
          && ((candidate.candidate_sha256 && row.evaluation?.candidate_sha256 !== candidate.candidate_sha256)
            || (candidate.candidate_file && row.evaluation?.candidate_file !== candidate.candidate_file))) continue;
      rows.push({
        ...row,
        sample_key: row.sample ? String(row.sample.category) + '/' + String(row.sample.sample_id) : path.basename(name, '.json'),
        effective_scored_at: row.scored_at || stat.mtime.toISOString(),
        mtime_ms: stat.mtimeMs
      });
    } catch {}
  }
  rows.sort((left, right) => right.mtime_ms - left.mtime_ms);
  const average = (name) => {
    const values = rows.map((row) => row.metrics?.[name])
      .filter((value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)))
      .map(Number);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  return {
    candidate_id: candidate.id,
    group,
    label: definition.label,
    model_role: definition.model_role,
    preference_rerank: definition.preference_rerank,
    record_count: rows.length,
    completed: rows.length,
    required_records: 11,
    required: 11,
    complete: rows.length === 11,
    metric_protocol: 'unified_reproduction_metrics_v3_safety_weighted_low_pck',
    metrics: rows.length ? Object.fromEntries(ALIGNED_LIVE_METRICS.map((name) => [name, average(name)])) : null,
    latest_sample: rows[0]?.sample_key || null,
    updated_at: rows[0]?.effective_scored_at || null
  };
}

async function alignedVal11GroupProgress({ mdpo, candidate }) {
  if (!candidate?.id) return null;
  const groups = await Promise.all(ALIGNED_VAL11_GROUPS.map((definition) => alignedVal11CandidateMetrics({ mdpo, candidate, group: definition.group })));
  const completed = groups.reduce((sum, group) => sum + group.completed, 0);
  const latest = groups.map((group) => group.updated_at).filter(Boolean).sort((left, right) => Date.parse(right) - Date.parse(left))[0] || null;
  const active = groups.find((group) => !group.complete) || null;
  return {
    candidate_id: candidate.id,
    completed,
    required: 44,
    complete: completed === 44,
    active_group: active?.group || null,
    active_group_label: active?.label || null,
    updated_at: latest,
    groups
  };
}

export async function buildMdpoLiveResults({ root, experimentsDir = path.join(root, 'experiments') }) {
  const mdpo = path.join(experimentsDir, 'mdpo');
  const ledger = await readJson(path.join(mdpo, 'val11_sweep_ledger.json'));
  const [pipelineLedger, alignedTrainingLedger, alignedValLedger, alignedSelection, alignedAblationLedger] = await Promise.all([
    readOptionalJson(path.join(mdpo, 'pipeline', 'full_pipeline_ledger.json')),
    readOptionalJson(path.join(mdpo, 'aligned_sweep', 'aligned_sweep_ledger.json')),
    readOptionalJson(path.join(mdpo, 'aligned_val11_sweep_ledger.json')),
    readOptionalJson(path.join(mdpo, 'aligned_hyperparameter_selection.json')),
    readOptionalJson(path.join(mdpo, 'aligned_ablations', 'ablation_ledger.json'))
  ]);
  const evaluated = (ledger.configurations || []).filter((row) => row.status === 'evaluated' && row.report_file && row.four_group_report_file);
  const latest = evaluated.toSorted((a, b) => Date.parse(b.completed_at || 0) - Date.parse(a.completed_at || 0))[0] || null;
  const running = (ledger.configurations || []).find((row) => row.status === 'running') || null;
  let report = null;
  let fourGroup = null;
  if (latest) {
    const reportFile = path.resolve(root, latest.report_file);
    const fourGroupFile = path.resolve(root, latest.four_group_report_file);
    const mdpoPrefix = path.resolve(mdpo) + path.sep;
    if (!reportFile.startsWith(mdpoPrefix) || !fourGroupFile.startsWith(mdpoPrefix)) throw new Error('Live result path escaped MDPO directory');
    const [reportBytes, fourBytes] = await Promise.all([fs.readFile(reportFile), fs.readFile(fourGroupFile)]);
    if (digest(reportBytes) !== latest.report_sha256 || digest(fourBytes) !== latest.four_group_report_sha256) throw new Error('Latest val11 report hash mismatch');
    report = JSON.parse(reportBytes.toString('utf8'));
    fourGroup = JSON.parse(fourBytes.toString('utf8'));
    if (report.candidate_id !== latest.id || fourGroup.candidate_id !== latest.id || !Array.isArray(fourGroup.groups) || fourGroup.groups.length !== 4) throw new Error('Latest val11 report provenance mismatch');
  }

  const activeStage = (pipelineLedger?.stages || []).find((row) => row.status === 'running') || (pipelineLedger?.stages || []).at(-1) || null;
  const childAlive = Boolean(activeStage?.status === 'running' && processAlive(activeStage.pid));
  const evaluationStatus = running && activeStage?.status === 'running' ? 'evaluating'
    : activeStage?.status === 'failed' ? 'error'
      : Number(ledger.required_configuration_count || 81) > evaluated.length ? 'waiting' : 'completed';
  const lastActivityAt = [pipelineLedger?.updated_at, activeStage?.last_heartbeat?.checked_at, running?.started_at, latest?.completed_at]
    .filter(Boolean).sort((a, b) => Date.parse(b) - Date.parse(a))[0] || null;
  let execution = {
    status: evaluationStatus, stage_id: activeStage?.id || null, stage_status: activeStage?.status || null,
    current_candidate: running ? { id: running.id, hyperparameters: running.hyperparameters, started_at: running.started_at } : null,
    pipeline_pid: pipelineLedger?.collector_pid || null, child_pid: activeStage?.pid || null, child_alive: childAlive,
    pipeline_updated_at: pipelineLedger?.updated_at || null, last_activity_at: lastActivityAt,
    explanation: evaluationStatus === 'evaluating'
      ? (childAlive ? 'val11 评估子进程正在运行' : 'val11 已登记为运行中，等待子进程心跳确认')
      : evaluationStatus === 'waiting' ? '流水线当前没有正在写入的候选评估'
        : evaluationStatus === 'error' ? (activeStage?.error || '流水线阶段异常') : '81 组 val11 已完成'
  };

  const alignedRows = alignedTrainingLedger?.configurations || [];
  const alignedValRows = alignedValLedger?.configurations || [];
  const alignedTrained = alignedRows.filter((row) => row.status === 'trained').length;
  const alignedTrainingFailed = alignedRows.filter((row) => row.status === 'failed').length;
  const alignedEvaluated = alignedValRows.filter((row) => row.status === 'evaluated').length;
  const alignedEvaluationFailed = alignedValRows.filter((row) => row.status === 'failed').length;
  const alignedSweepDir = path.join(mdpo, 'aligned_sweep');
  const alignedSweepEntries = await fs.readdir(alignedSweepDir, { withFileTypes: true }).catch(() => []);
  const alignedStarted = alignedSweepEntries.some((entry) => entry.isDirectory() && entry.name.startsWith('lr_'));
  const alignedStatus = alignedEvaluated === 81 ? 'completed'
    : alignedTrained === 81 && alignedEvaluationFailed === 0 ? 'evaluating'
      : alignedTrainingFailed > 0 ? 'error' : alignedTrained > 0 || alignedStarted ? 'training' : 'pending';
  const alignedRunningVal = alignedValRows.find((row) => row.status === 'running') || null;
  let alignedCurrent = alignedTrainingLedger?.current || alignedRunningVal || null;
  if (!alignedCurrent && alignedStatus === 'training') {
    const trainedIds = new Set(alignedRows.filter((row) => row.status === 'trained').map((row) => row.id));
    const activeDirectories = alignedSweepEntries.filter((entry) => entry.isDirectory() && entry.name.startsWith('lr_') && !trainedIds.has(entry.name));
    const stats = await Promise.all(activeDirectories.map(async (entry) => ({
      id: entry.name,
      stat: await fs.stat(path.join(alignedSweepDir, entry.name)).catch(() => ({ mtimeMs: 0 }))
    })));
    const newest = stats.sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs)[0];
    if (newest) alignedCurrent = { id: newest.id, status: 'training_inferred_from_active_artifact' };
  }
  const aligned = {
    policy: 'safety_priority_v3_aligned', status: alignedStatus, current: alignedCurrent,
    training: { trained: alignedTrained, failed: alignedTrainingFailed, required: 81 },
    val11: { evaluated: alignedEvaluated, failed: alignedEvaluationFailed, required: 81 },
    updated_at: alignedValLedger?.updated_at || alignedTrainingLedger?.updated_at || null
  };
  if (alignedStatus === 'evaluating' || alignedStatus === 'completed') execution = {
    status: alignedStatus === 'completed' ? 'completed' : 'evaluating',
    stage_id: 'evaluate_81_grid_val11_aligned',
    stage_status: alignedRunningVal ? 'running' : alignedStatus === 'completed' ? 'completed' : 'waiting',
    current_candidate: alignedRunningVal ? { id: alignedRunningVal.id, hyperparameters: alignedRunningVal.hyperparameters, started_at: alignedRunningVal.started_at } : null,
    pipeline_pid: null,
    child_pid: null,
    child_alive: Boolean(alignedRunningVal),
    pipeline_updated_at: alignedValLedger?.updated_at || null,
    last_activity_at: alignedValLedger?.updated_at || alignedRunningVal?.started_at || null,
    explanation: alignedStatus === 'completed' ? '安全对齐版 81 组 val11 已完成'
      : alignedRunningVal ? '安全对齐版 val11 评估正在运行' : '安全对齐版 val11 等待下一候选'
  };
  const latestCompletedAligned = alignedRows.filter((row) => row.status === 'trained' && row.completed_at)
    .toSorted((left, right) => Date.parse(right.completed_at) - Date.parse(left.completed_at))[0] || null;
  if (latestCompletedAligned) aligned.latest_completed_training = {
    id: latestCompletedAligned.id,
    completed_at: latestCompletedAligned.completed_at,
    best_epoch: latestCompletedAligned.best_epoch ?? null,
    proxy_val: latestCompletedAligned.proxy_val || null,
    hyperparameters: latestCompletedAligned.hyperparameters || null,
    deployment_status: latestCompletedAligned.deployment_status || 'diagnostic_only_pending_authoritative_val11_v3_aligned_gate'
  };
  if (alignedCurrent?.id) aligned.current_metrics = await parseLastEpoch(path.join(alignedSweepDir, alignedCurrent.id), root);
  if (alignedRunningVal?.id) {
    aligned.val11_group_progress = await alignedVal11GroupProgress({ mdpo, candidate: alignedRunningVal });
    aligned.current_val11 = aligned.val11_group_progress?.groups?.find((group) => group.group === 'mdpo_no_rerank') || null;
    if (aligned.val11_group_progress?.updated_at) {
      aligned.updated_at = aligned.val11_group_progress.updated_at;
      execution.last_activity_at = aligned.val11_group_progress.updated_at;
    }
  }
  const alignedLatestRow = alignedValRows.filter((row) => row.status === 'evaluated' && row.report_file)
    .toSorted((a, b) => Date.parse(b.completed_at || 0) - Date.parse(a.completed_at || 0))[0] || null;
  if (alignedLatestRow) {
    try {
      const alignedReportFile = path.resolve(root, alignedLatestRow.report_file);
      if (alignedReportFile.startsWith(path.resolve(mdpo) + path.sep)) {
        const alignedReport = JSON.parse(await fs.readFile(alignedReportFile, 'utf8'));
        aligned.latest_val11 = { candidate_id: alignedLatestRow.id, completed_at: alignedLatestRow.completed_at,
          gate: alignedReport.gate, metrics: alignedReport.candidate?.metrics || null };
        aligned.latest_val11_group_results = await alignedVal11GroupProgress({ mdpo, candidate: alignedLatestRow });
        if (!aligned.current_val11) aligned.current_val11 = await alignedVal11CandidateMetrics({ mdpo, candidate: alignedLatestRow });
      }
    } catch {}
  }

  if (alignedSelection?.selected?.id) {
    const selectedCandidate = {
      id: alignedSelection.selected.id,
      candidate_file: alignedSelection.selected.candidate_file,
      candidate_sha256: alignedSelection.selected.candidate_sha256
    };
    aligned.selection = {
      selected_candidate_id: selectedCandidate.id,
      deployment_eligible: alignedSelection.deployment_eligible === true,
      status: alignedSelection.status || (alignedSelection.deployment_eligible ? 'accepted' : 'rejected'),
      generated_at: alignedSelection.generated_at || null,
      hyperparameters: alignedSelection.selected.hyperparameters || null,
      violations: alignedSelection.selected.gate?.violations || [],
      delta: alignedSelection.selected.gate?.delta || null
    };
    aligned.selected_val11_group_results = await alignedVal11GroupProgress({ mdpo, candidate: selectedCandidate });
  }

  if (alignedAblationLedger) {
    const completed = (alignedAblationLedger.configurations || []).filter((row) => row.status === 'trained');
    aligned.ablations = {
      completed: completed.length,
      required: Array.isArray(alignedAblationLedger.required) ? alignedAblationLedger.required.length : 9,
      current: alignedAblationLedger.current || null,
      latest_completed: completed.at(-1) || null
    };
    if (alignedAblationLedger.current?.id) {
      aligned.ablations.current_metrics = await parseLastEpoch(path.join(mdpo, 'aligned_ablations', alignedAblationLedger.current.id), root);
    }
  }

  const legacyGroups = (fourGroup?.groups || []).map((group) => ({ group: group.group, label: group.label || group.name,
    model_role: group.model_role, preference_rerank: group.preference_rerank, sample_count: group.sample_count,
    metric_protocol: group.metric_protocol, metrics: group.metrics, scores: group.score_means }));
  const alignedDisplayGroups = (aligned.selected_val11_group_results?.groups || aligned.latest_val11_group_results?.groups || [])
    .filter((group) => group.metrics);
  const displayGroups = alignedDisplayGroups.length === ALIGNED_VAL11_GROUPS.length
    ? alignedDisplayGroups.map((group) => ({ candidate_id: group.candidate_id, group: group.group, label: group.label,
      model_role: group.model_role, preference_rerank: group.preference_rerank,
      sample_count: group.record_count, metric_protocol: group.metric_protocol, metrics: group.metrics, scores: null }))
    : legacyGroups;

  return {
    version: 'v10_mdpo_live_results_v5', generated_at: new Date().toISOString(),
    source: 'server_live_aggregation_from_experiment_ledgers', execution, aligned,
    progress: { required: Number(ledger.required_configuration_count || 81), evaluated: evaluated.length,
      accepted: evaluated.filter((row) => row.gate_accepted === true).length,
      rejected: evaluated.filter((row) => row.gate_accepted === false).length,
      running: running ? { id: running.id, hyperparameters: running.hyperparameters, started_at: running.started_at } : null },
    latest: latest ? { candidate_id: latest.id, completed_at: latest.completed_at, hyperparameters: latest.hyperparameters,
      candidate_sha256: latest.candidate_sha256, gate: report.gate,
      status: report.gate?.accepted ? 'accepted_pending_formal_selection' : 'diagnostic_only_rejected', deployed: false } : null,
    groups: displayGroups,
    security: { test_used: false, qwen_inference_input: false, candidate_not_presented_as_deployed: true }
  };
}
