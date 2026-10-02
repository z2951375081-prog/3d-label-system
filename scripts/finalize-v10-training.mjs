import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');

function parseArgs(argv) {
  const options = { expectedEpochs: 80, activate: true, force: false, iterations: 120, output: path.join(experiments, 'v10_training_completion_audit.json') };
  for (let index = 0; index < argv.length; index += 1) if (argv[index].startsWith('--')) {
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const candidateFile = path.join(experiments, 'layout_model_v10_anchor_frame_candidate.json');
const reportFile = path.join(experiments, 'layout_training_v10_anchor_frame_report.json');
const selectionFile = path.join(experiments, 'v10_anchor_frame_inference_selection.json');
const activeFile = path.join(experiments, 'layout_model.json');
const logFile = path.join(experiments, 'service-logs', 'v10-finalization.log');
const outputFile = path.resolve(String(options.output));

const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const sha256 = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const readJsonIfPresent = async (file) => readJson(file).catch(() => null);
const modelCoreSha256 = (model) => createHash('sha256').update(JSON.stringify({
  version: model?.version,
  architecture: model?.architecture,
  network: model?.network,
  inference: {
    center_blend: model?.inference?.center_blend,
    size_blend: model?.inference?.size_blend,
    size_ratio_range: model?.inference?.size_ratio_range,
    generation_input: model?.inference?.generation_input
  }
})).digest('hex').toUpperCase();
const stamp = async (message) => {
  const line = `[${new Date().toISOString()}] ${message}`;
  await fs.mkdir(path.dirname(logFile), { recursive: true });
  await fs.appendFile(logFile, `${line}\n`, 'utf8');
  console.log(line);
};

function runNode(args, label, timeoutMinutes = 30) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); process.stdout.write(chunk); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); process.stderr.write(chunk); });
    const timer = setTimeout(() => { child.kill(); reject(new Error(`${label} exceeded ${timeoutMinutes} minutes`)); }, timeoutMinutes * 60 * 1000);
    child.on('error', (error) => { clearTimeout(timer); reject(error); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ code, stdout, stderr });
      else reject(new Error(`${label} failed with code ${code}: ${stderr || stdout}`));
    });
  });
}

function assertTrainingContract(model) {
  const architecture = model.architecture || {};
  const training = model.training || {};
  if (model.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw new Error('candidate is not v10');
  if (Number(model.hyperparameters?.epochs) < Number(options.expectedEpochs)) throw new Error(`candidate epochs ${model.hyperparameters?.epochs} < expected ${options.expectedEpochs}`);
  if (architecture.type !== 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe') throw new Error('architecture type mismatch');
  if (architecture.generation_input !== 'pure_3d' || architecture.visual_generation_input !== false) throw new Error('visual inputs enter initial 3D generation');
  if (architecture.node_input_dim !== 51 || architecture.geometry_feature_dim !== 64 || architecture.anchor_label_edge_dim !== 18 || architecture.label_label_edge_dim !== 10) throw new Error('feature dimensions mismatch');
  if (architecture.message_passing_layers !== 2 || architecture.transformer_layers !== 1) throw new Error('GNN/Transformer depth mismatch');
  if (architecture.output_semantics !== 'local_u_local_v_local_normal_distance_and_log_size_xyz') throw new Error('decoder is not local 6D');
  if (architecture.view_loss?.weights?.main !== 0.4 || architecture.view_loss?.type !== 'weighted_main_plus_worst_view_plus_cvar_plus_stereo') throw new Error('main/worst-view/CVaR view loss missing');
  if (!architecture.human_style_targets?.includes('per_view_free_space_distribution')) throw new Error('per-view free-space target missing');
  if (!architecture.human_style_targets?.includes('leader_line_non_crossing') || !architecture.view_loss?.terms?.includes('leader_crossing') || !(architecture.view_loss?.leader_crossing_weight > 0)) throw new Error('leader crossing base-model objective missing');
  if (!training.leader_direction_loss_used || !training.per_view_font_clarity_loss_used) throw new Error('leader direction or font clarity loss missing');
  if (!training.leader_crossing_loss_used || !Number.isFinite(training.metrics?.val?.leader_crossing)) throw new Error('leader crossing training evidence missing');
  if (training.test_used_in_gradient_or_checkpoint_selection !== false) throw new Error('test split leakage detected');
  for (const key of ['input_fnn', 'pre_gnn_fnn', 'feature_fusion', 'message_passing_gnn', 'anchor_to_label_messages', 'label_to_label_messages', 'transformer', 'moe_router', 'moe_experts']) {
    if (!(training.parameter_updates?.[key]?.changed_parameters > 0)) throw new Error(`${key} has no parameter updates`);
  }
}

const audit = {
  version: 'v10_training_completion_audit_v1',
  generated_at: new Date().toISOString(),
  status: 'running',
  files: {},
  checks: [],
  activation: null,
  next_action: null
};

try {
  await stamp('Starting v10 candidate completion audit.');
  const [candidate, report] = await Promise.all([readJson(candidateFile), readJson(reportFile)]);
  assertTrainingContract(candidate);
  if (report.version !== 'layout_training_v10_anchor_frame_report') throw new Error('training report version mismatch');
  audit.checks.push({ name: 'training_contract', passed: true, best_epoch: candidate.training.best_epoch, epochs: candidate.hyperparameters.epochs });
  await runNode(['scripts/test-v10-architecture-contract.mjs'], 'v10 architecture contract', 5);
  await runNode(['scripts/test-anchor-frame-and-view-evaluator.mjs'], 'anchor/view evaluator regression', 5);
  audit.checks.push({ name: 'architecture_and_view_tests', passed: true });

  const [activeBeforeSelection, existingSelection] = await Promise.all([readJsonIfPresent(activeFile), readJsonIfPresent(selectionFile)]);
  const reusableActivatedCandidate = activeBeforeSelection?.version === candidate.version
    && ['active_val_selected_v10_anchor_frame', 'active_user_override_v10_anchor_frame'].includes(activeBeforeSelection?.status)
    && modelCoreSha256(activeBeforeSelection) === modelCoreSha256(candidate)
    && candidate.validation_gate?.status === 'accepted'
    && existingSelection?.validation?.accepted === true;
  if (reusableActivatedCandidate) {
    await stamp('Reusing the existing accepted val11/test11 selection for the identical already-active v10 model.');
  } else {
    await stamp('Running val11 blend selection and test11 confirmation.');
    await runNode(['scripts/select-v9-3d-inference.mjs', '--model', candidateFile, '--previous', activeFile, '--output', selectionFile, '--iterations', String(options.iterations)], 'v10 val/test selection', 120);
  }
  const selectedCandidate = reusableActivatedCandidate ? candidate : await readJson(candidateFile);
  const selection = reusableActivatedCandidate ? existingSelection : await readJson(selectionFile);
  const accepted = selectedCandidate.validation_gate?.status === 'accepted' && selection.validation?.accepted === true;
  audit.checks.push({ name: 'val11_quality_and_safety_gate', passed: accepted, reused_existing_selection: reusableActivatedCandidate, validation_gate: selectedCandidate.validation_gate });

  if (options.activate !== false && String(options.activate).toLowerCase() !== 'false') {
    if (!accepted && !options.force) throw new Error('v10 val11 gate rejected candidate; active model was preserved');
    if (reusableActivatedCandidate && !options.force) {
      await stamp('The identical val-selected v10 model is already active; preserving its existing activation and backup history.');
    } else {
      const activationArgs = ['scripts/activate-v10-anchor-frame-model.mjs'];
      if (options.force) activationArgs.push('--force');
      await stamp(accepted ? 'Activating val-selected v10 candidate.' : 'Force activating v10 candidate with preserved diagnostics.');
      await runNode(activationArgs, 'v10 activation', 10);
    }
    const active = await readJson(activeFile);
    if (active.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw new Error('active model is not v10 after activation');
    audit.activation = { activated: true, forced: Boolean(options.force), reused_existing_activation: reusableActivatedCandidate && !options.force, status: active.status, validation_gate: active.validation_gate };
  } else audit.activation = { activated: false, reason: 'activation disabled by option' };

  for (const [script, label] of [
    ['scripts/test-layout-provenance.mjs', 'layout provenance'],
    ['scripts/test-model-preservation.mjs', 'model preservation'],
    ['scripts/test-multidimensional-evaluator.mjs', 'multidimensional evaluator'],
    ['scripts/test-directional-density.mjs', 'free-space directional evaluator'],
    ['scripts/test-adaptive-directional-rerank.mjs', 'adaptive safe rerank'],
    ['scripts/audit-fixed-label-contract.mjs', 'fixed-label contract']
  ]) await runNode([script], label, 20);
  audit.checks.push({ name: 'post_activation_regressions', passed: true });
  audit.files = {
    candidate: { path: path.relative(root, candidateFile).replaceAll('\\', '/'), sha256: await sha256(candidateFile) },
    training_report: { path: path.relative(root, reportFile).replaceAll('\\', '/'), sha256: await sha256(reportFile) },
    selection: { path: path.relative(root, selectionFile).replaceAll('\\', '/'), sha256: await sha256(selectionFile) },
    active: audit.activation?.activated ? { path: path.relative(root, activeFile).replaceAll('\\', '/'), sha256: await sha256(activeFile) } : null
  };
  audit.status = 'complete';
  audit.next_action = audit.activation?.activated
    ? 'Restart the local service, capture the v10 six-view visualization, then run the 3-sample × 4-candidate × 8-round local Qwen preference experiment with val11 checkpoints.'
    : 'Review the rejected val11 diagnostics before any explicit activation override.';
} catch (error) {
  audit.status = 'failed';
  audit.error = error.stack || error.message;
  audit.next_action = /val11 gate rejected/.test(error.message) ? 'Keep the previous active model and tune/retrain v10 from val diagnostics.' : 'Fix the failed audit item and rerun finalization.';
  process.exitCode = 2;
} finally {
  audit.generated_at = new Date().toISOString();
  await fs.writeFile(outputFile, `${JSON.stringify(audit, null, 2)}\n`, 'utf8');
  await stamp(`Finalization audit status=${audit.status}; report=${path.relative(root, outputFile).replaceAll('\\', '/')}`);
}
