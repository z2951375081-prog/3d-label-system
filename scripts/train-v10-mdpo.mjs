import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { MDPO_DIMENSIONS, MDPO_STD, MDPO_WEIGHTS, applyMdpoVarianceGradient, initializeMdpoVariance, materializeMdpoStd, mdpoPairLoss } from '../lib/mdpo-continuous-policy.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { loadMdpoTrainingGraphs } from '../lib/mdpo-graph-cache.mjs';
import { mdpoRng, restoreMdpoTrainingState } from '../lib/mdpo-training-resume.mjs';
import { writeImmutableJsonBundle } from '../lib/immutable-json-bundle.mjs';
import { splitMdpoTrainPreferencePairs } from '../lib/mdpo-preference-holdout.mjs';
import { evaluateMdpoWeightUpdateEvidence } from '../lib/mdpo-weight-update-evidence.mjs';
import { compareMdpoUnifiedMetrics, evaluateMdpoTrainingUnifiedMetrics } from '../lib/mdpo-unified-training-metrics.mjs';
import { evaluateReproductionMetrics, REPRODUCTION_METRIC_PROTOCOL } from '../lib/reproduction-metrics.mjs';
import { MULTI_VIEW_NAMES } from '../lib/layout-optimizer.mjs';
import { applyV10BiasGradient, applyV10LoraGradient, initializeV10BiasTuning, initializeV10Lora, materializeV10BiasTuning, materializeV10Lora, materializeV10LoraStep, v10LoraPaths } from '../lib/mdpo-lora.mjs';
import { backwardV10TrainingGraph, buildV10TrainingGraphs, forwardV10TrainingGraph, labelsFromV10TrainingOutputs, v10GeometrySafetyLoss, v10SupervisedObjectiveLoss, v10SupervisedOutputGradient } from './train-3d-human-style-layout-model.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaults = {
  dataset: path.join(root, 'experiments', 'mdpo', 'train_pairs.json'), manifest: path.join(root, 'experiments', 'dataset_manifest.json'),
  reference: path.join(root, 'experiments', 'layout_model.json'), outputDir: path.join(root, 'experiments', 'mdpo', 'training'),
  epochs: 30, learningRate: 1e-4, beta: 0.1, lambdaMulti: 0.3, lambdaDpo: 1, lambdaKl: 0.01,
  lambdaSup: 0.2, lambdaSafe: 1, lambdaCenterTail: 0.1, rank: 4, alpha: 8, dropout: 0.05, clipNorm: 1,
  compositionWeight: MDPO_WEIGHTS.composition_harmony, hierarchyWeight: MDPO_WEIGHTS.visual_hierarchy,
  balanceWeight: MDPO_WEIGHTS.spatial_balance, manualStyleWeight: MDPO_WEIGHTS.manual_style_similarity,
  textWeight: MDPO_WEIGHTS.text_clarity, leaderWeight: MDPO_WEIGHTS.leader_line_clarity,
  learnableVariance: false, noReference: false, varianceLearningRate: 1e-4, varianceMinimum: 0.03, varianceMaximum: 0.75,
  patience: 6, minDelta: 1e-6, seed: 17, styleGridSize: 20, allowIncomplete: false, resume: null,
  safetyAlignment: false, overlapPairWeight: 24, worstOverflowWeight: 20, cvarOverflowWeight: 12, preferenceSafetyWeight: 0.5
};
function args(argv) { const out = { ...defaults }; for (let i = 0; i < argv.length; i++) if (argv[i].startsWith('--')) { const key = argv[i].slice(2); out[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true; } return out; }
const finite = (value, name, minimum = -Infinity) => { const number = Number(value); if (!Number.isFinite(number) || number < minimum) throw new Error(`Invalid ${name}`); return number; };
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
function shuffle(values, random) { const out = [...values]; for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; } return out; }
function combine(left, right, leftScale, rightScale) { return left.map((row, node) => row.map((value, axis) => leftScale * value + rightScale * right[node][axis])); }
function centerTail(meanOutput, graph) {
  const rows = meanOutput.map((row, index) => ({ index, error: Math.hypot(...row.slice(0, 3).map((value, axis) => value - graph.nodes[index].target[axis])) })).sort((a, b) => b.error - a.error);
  const tail = rows.slice(0, Math.max(1, Math.ceil(rows.length * 0.25)));
  const gradient = meanOutput.map(() => Array(6).fill(0));
  for (const item of tail) for (let axis = 0; axis < 3; axis++) gradient[item.index][axis] = 2 * (meanOutput[item.index][axis] - graph.nodes[item.index].target[axis]) / (3 * tail.length);
  return { loss: mean(tail.map((item) => item.error ** 2)), gradient };
}
function evaluation(network, graphs) {
  const errors = [], nodeErrors = [];
  for (const graph of graphs) {
    const output = forwardV10TrainingGraph(network, graph).output;
    for (let node = 0; node < output.length; node++) {
      errors.push(...output[node].map((value, axis) => (value - graph.nodes[node].target[axis]) ** 2));
      nodeErrors.push(Math.hypot(...output[node].slice(0, 3).map((value, axis) => value - graph.nodes[node].target[axis])));
    }
  }
  const ordered = [...nodeErrors].sort((a, b) => b - a), tail = ordered.slice(0, Math.max(1, Math.ceil(ordered.length * 0.25)));
  // Local 3D radii are only a training diagnostic, never the authoritative
  // rendered five-view PCK used for val11 deployment or test11 reporting.
  return { total: mean(errors) + 0.1 * mean(tail.map((value) => value ** 2)), mse: mean(errors), center_tail_cvar: mean(tail.map((value) => value ** 2)), local_center_proxy_005: mean(nodeErrors.map((value) => Number(value <= 0.05))), local_center_proxy_010: mean(nodeErrors.map((value) => Number(value <= 0.10))), node_count: nodeErrors.length };
}
function supervisedOptions(overrides = {}, safetyAlignment = false) { return { baseWeight: 1, styleWeight: 0.2, directionWeight: 1.25, viewWeight: 0.25, worstViewWeight: 2, cvarViewWeight: 1, stereoWeight: 1, textClarityWeight: 3, leaderCrossingWeight: 3.5, ...(safetyAlignment ? { overlapPairWeight: 24, worstOverflowWeight: 20, cvarOverflowWeight: 12 } : {}), ...overrides }; }
async function immutableJson(file, value) { await fs.writeFile(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' }); const bytes = await fs.readFile(file); return digest(bytes); }

async function main() {
  const options = args(process.argv.slice(2));
  for (const key of ['epochs', 'learningRate', 'beta', 'lambdaMulti', 'lambdaDpo', 'lambdaKl', 'lambdaSup', 'lambdaSafe', 'lambdaCenterTail', 'rank', 'alpha', 'dropout', 'clipNorm', 'compositionWeight', 'hierarchyWeight', 'balanceWeight', 'manualStyleWeight', 'textWeight', 'leaderWeight', 'varianceLearningRate', 'varianceMinimum', 'varianceMaximum', 'patience', 'minDelta', 'seed', 'styleGridSize', 'overlapPairWeight', 'worstOverflowWeight', 'cvarOverflowWeight', 'preferenceSafetyWeight']) options[key] = finite(options[key], key, key === 'dropout' ? 0 : 0);
  options.epochs = Math.floor(options.epochs); options.rank = Math.floor(options.rank); options.patience = Math.floor(options.patience);
  options.learnableVariance = options.learnableVariance === true || String(options.learnableVariance).toLowerCase() === 'true';
  options.noReference = options.noReference === true || String(options.noReference).toLowerCase() === 'true';
  options.safetyAlignment = options.safetyAlignment === true || String(options.safetyAlignment).toLowerCase() === 'true';
  if (options.noReference && options.lambdaKl !== 0) throw new Error('No-reference ablation must also disable explicit reference KL');
  if (options.dropout >= 1 || options.epochs < 1 || options.rank < 1 || options.varianceMinimum >= options.varianceMaximum) throw new Error('Invalid MDPO rank/dropout/epochs/variance bounds');
  const dimensionWeights = {
    composition_harmony: options.compositionWeight, visual_hierarchy: options.hierarchyWeight,
    spatial_balance: options.balanceWeight, manual_style_similarity: options.manualStyleWeight,
    text_clarity: options.textWeight, leader_line_clarity: options.leaderWeight
  };
  const referenceBytes = await fs.readFile(path.resolve(String(options.reference))), referenceHash = digest(referenceBytes);
  const referenceModel = JSON.parse(referenceBytes.toString('utf8'));
  if (!String(referenceModel.version).startsWith('layout_model_v10_')) throw new Error('MDPO reference must be active v10');
  const manifestBytes = await fs.readFile(path.resolve(String(options.manifest)));
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const dataset = JSON.parse(await fs.readFile(path.resolve(String(options.dataset)), 'utf8'));
  if (dataset.reference_model_sha256 !== referenceHash) throw new Error('MDPO dataset/reference SHA-256 mismatch');
  const audit = validateMdpoDataset(dataset, manifest, { requireComplete: !(options.allowIncomplete === true || String(options.allowIncomplete).toLowerCase() === 'true') });
  const preferenceSplit = splitMdpoTrainPreferencePairs(dataset.pairs);
  const { trainGraphs, valGraphs, cache: graphCache } = await loadMdpoTrainingGraphs({ root, manifest, manifestBytes, gridSize: options.styleGridSize, directory: path.join(root, 'experiments', 'mdpo', 'graph_cache'), buildGraphs: buildV10TrainingGraphs });
  console.log(JSON.stringify({ progress: 'graph_cache', status: graphCache.status, file: path.relative(root, graphCache.file).split(path.sep).join('/'), sha256_key: graphCache.key, train_graphs: trainGraphs.length, val_graphs: valGraphs.length }));
  const graphMap = new Map(trainGraphs.map((graph) => [`${graph.category}/${graph.sample_id}`, graph]));
  const usedSamples = new Set(dataset.pairs.map((pair) => `${pair.category}/${pair.sample_id}`));
  const train = trainGraphs.filter((graph) => usedSamples.has(`${graph.category}/${graph.sample_id}`));
  for (const pair of dataset.pairs) {
    const graph = graphMap.get(`${pair.category}/${pair.sample_id}`);
    if (!graph || JSON.stringify(pair.candidate_a.label_ids) !== JSON.stringify(graph.nodes.map((node) => String(node.id)))) throw new Error(`MDPO pair/graph label order mismatch: ${pair.category}/${pair.sample_id}`);
  }
  await fs.mkdir(path.resolve(String(options.outputDir)), { recursive: true });
  const checkpointDir = path.join(path.resolve(String(options.outputDir)), 'checkpoints'); await fs.mkdir(checkpointDir, { recursive: true });
  let adapters = initializeV10Lora(referenceModel.network, { rank: options.rank, alpha: options.alpha, dropout: options.dropout, seed: options.seed });
  let biases = initializeV10BiasTuning(referenceModel.network);
  let variance = initializeMdpoVariance({ std: MDPO_STD, learnable: options.learnableVariance, minimum: options.varianceMinimum, maximum: options.varianceMaximum });
  let startEpoch = 1, history = [], ledger = [], best = null, bestVal = Infinity, stale = 0, randomState = null;
  const datasetHash = digest(await fs.readFile(path.resolve(String(options.dataset))));
  if (options.resume) {
    const resumed = await restoreMdpoTrainingState({ checkpointFile: String(options.resume), checkpointDir, root,
      referenceHash, datasetHash, options });
    adapters = resumed.adapters; biases = resumed.biases; variance = resumed.variance;
    startEpoch = resumed.startEpoch; history = resumed.history; ledger = resumed.ledger;
    best = resumed.best; bestVal = resumed.bestVal; stale = resumed.stale; randomState = resumed.randomState;
  }
  const referenceOutputs = new Map(train.map((graph) => [`${graph.category}/${graph.sample_id}`, forwardV10TrainingGraph(referenceModel.network, graph).output]));
  const random = mdpoRng(options.seed + 1, randomState);
  for (let epoch = startEpoch; epoch <= options.epochs; epoch++) {
    if (stale >= options.patience) break;
    const dimensionTotals = Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, []]));
    const losses = [], supervisedLosses = [], safetyLosses = [], gradients = [], varianceGradients = [];
    const lr = options.learningRate * 0.5 * (1 + Math.cos(Math.PI * (epoch - 1) / Math.max(1, options.epochs)));
    // One full original-v10 supervised step per train graph and epoch. This
    // retains center/size, style, leader direction, worst-view/CVaR and text
    // clarity without redundantly recomputing expensive finite differences
    // for every preference pair of the same object.
    for (const graph of shuffle(train, random)) {
      const loraStep = materializeV10LoraStep(referenceModel.network, adapters, { training: true, random });
      const policy = materializeV10BiasTuning(loraStep.network, biases);
      const run = forwardV10TrainingGraph(policy, graph);
      const supervisedConfig = supervisedOptions({ viewWeight: 0 });
      const safetyConfig = supervisedOptions({ baseWeight: 0, styleWeight: 0, directionWeight: 0 }, options.safetyAlignment);
      const supervised = v10SupervisedOutputGradient(run, graph, supervisedConfig);
      const safety = v10SupervisedOutputGradient(run, graph, safetyConfig);
      const tail = centerTail(run.output, graph);
      const supervisedAndSafe = combine(supervised, safety, options.lambdaSup, options.lambdaSafe);
      const outputGradient = combine(supervisedAndSafe, tail.gradient, 1, options.lambdaCenterTail);
      const networkGradient = backwardV10TrainingGraph(policy, graph, run, supervisedOptions(), outputGradient);
      const loraUpdate = applyV10LoraGradient(adapters, networkGradient, lr, options.clipNorm, loraStep.masks);
      const biasUpdate = applyV10BiasGradient(biases, networkGradient, lr, options.clipNorm);
      const supervisedObjective = v10SupervisedObjectiveLoss(run.output, graph, supervisedConfig);
      supervisedLosses.push(options.lambdaSup * supervisedObjective.total + options.lambdaCenterTail * tail.loss);
      safetyLosses.push(options.lambdaSafe * v10GeometrySafetyLoss(run.output, graph, safetyConfig));
      gradients.push(loraUpdate.gradient_norm + biasUpdate.gradient_norm);
    }
    for (const pair of shuffle(preferenceSplit.gradientPairs, random)) {
      const key = `${pair.category}/${pair.sample_id}`, graph = graphMap.get(key);
      const loraStep = materializeV10LoraStep(referenceModel.network, adapters, { training: true, random });
      const policy = materializeV10BiasTuning(loraStep.network, biases);
      const run = forwardV10TrainingGraph(policy, graph);
      const preference = mdpoPairLoss({ mean: run.output, referenceMean: referenceOutputs.get(key), candidateA: pair.candidate_a.local_layout, candidateB: pair.candidate_b.local_layout, scoresA: pair.candidate_a.scores, scoresB: pair.candidate_b.scores, policyStd: materializeMdpoStd(variance), referenceStd: MDPO_STD, beta: options.beta, lambdaDpo: options.lambdaDpo, lambdaMulti: options.lambdaMulti, lambdaKl: options.lambdaKl, useReference: !options.noReference, weights: dimensionWeights });
      const preferenceSafetyConfig = supervisedOptions({ baseWeight: 0, styleWeight: 0, directionWeight: 0 }, options.safetyAlignment);
      const preferenceSafetyGradient = options.safetyAlignment
        ? v10SupervisedOutputGradient(run, graph, preferenceSafetyConfig)
        : run.output.map((row) => row.map(() => 0));
      const alignedPreferenceGradient = combine(preference.gradient, preferenceSafetyGradient, 1, options.safetyAlignment ? options.lambdaSafe * options.preferenceSafetyWeight : 0);
      const networkGradient = backwardV10TrainingGraph(policy, graph, run, supervisedOptions(), alignedPreferenceGradient);
      const loraUpdate = applyV10LoraGradient(adapters, networkGradient, lr, options.clipNorm, loraStep.masks);
      const biasUpdate = applyV10BiasGradient(biases, networkGradient, lr, options.clipNorm);
      const varianceUpdate = applyMdpoVarianceGradient(variance, preference.logStdGradient, options.varianceLearningRate * 0.5 * (1 + Math.cos(Math.PI * (epoch - 1) / Math.max(1, options.epochs))), options.clipNorm);
      losses.push(preference.loss); gradients.push(loraUpdate.gradient_norm + biasUpdate.gradient_norm);
      varianceGradients.push(varianceUpdate.gradient_norm);
      for (const name of MDPO_DIMENSIONS) dimensionTotals[name].push(preference.perDimension[name].loss);
    }
    const policy = materializeV10BiasTuning(materializeV10Lora(referenceModel.network, adapters), biases);
    const heldoutDimensionTotals = Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, []]));
    const heldoutLosses = [], heldoutKl = [];
    const heldoutOutputs = new Map();
    for (const pair of preferenceSplit.holdoutPairs) {
      const key = `${pair.category}/${pair.sample_id}`, graph = graphMap.get(key);
      if (!heldoutOutputs.has(key)) heldoutOutputs.set(key, forwardV10TrainingGraph(policy, graph).output);
      const diagnostic = mdpoPairLoss({ mean: heldoutOutputs.get(key), referenceMean: referenceOutputs.get(key),
        candidateA: pair.candidate_a.local_layout, candidateB: pair.candidate_b.local_layout,
        scoresA: pair.candidate_a.scores, scoresB: pair.candidate_b.scores,
        policyStd: materializeMdpoStd(variance), referenceStd: MDPO_STD, beta: options.beta,
        lambdaDpo: options.lambdaDpo, lambdaMulti: options.lambdaMulti,
        lambdaKl: options.lambdaKl, useReference: !options.noReference, weights: dimensionWeights });
      heldoutLosses.push(diagnostic.loss); heldoutKl.push(diagnostic.kl);
      for (const name of MDPO_DIMENSIONS) heldoutDimensionTotals[name].push(diagnostic.perDimension[name].loss);
    }
    const preferenceHoldout = { scope: 'train33_heldout_pairs_not_val11', pair_count: preferenceSplit.holdoutPairs.length,
      total: mean(heldoutLosses), kl: mean(heldoutKl),
      ...Object.fromEntries(MDPO_DIMENSIONS.map((name) => [`loss_${name}`, mean(heldoutDimensionTotals[name])])) };
    const val = evaluation(policy, valGraphs), trainMetrics = evaluation(policy, train);
    const row = { epoch, learning_rate: lr, train: { total: mean(losses) + mean(supervisedLosses) + mean(safetyLosses), mdpo: mean(losses), original_v10_supervised: mean(supervisedLosses), geometry_safe: mean(safetyLosses), gradient_norm: mean(gradients), variance_gradient_norm: mean(varianceGradients), policy_std: materializeMdpoStd(variance), preference_pair_count: preferenceSplit.gradientPairs.length, ...Object.fromEntries(MDPO_DIMENSIONS.map((name) => [`loss_${name}`, mean(dimensionTotals[name])])), supervised_metrics: trainMetrics }, val: { ...val, preference_holdout: preferenceHoldout } };
    history.push(row);
    const improved = val.total < bestVal - options.minDelta;
    const nextBest = improved ? { epoch, adapters: structuredClone(adapters), biases: structuredClone(biases),
      variance: structuredClone(variance), val } : { epoch: best.epoch, adapters: best.adapters, biases: best.biases, variance: best.variance, val: best.val };
    const nextStale = improved ? 0 : stale + 1;
    const checkpoint = { version: 'v10_mdpo_checkpoint_v1', epoch, generated_at: new Date().toISOString(), reference_model_sha256: referenceHash, dataset_sha256: datasetHash, split_policy: { gradient: 'train33_only', selection: 'val11_only', test_used: false }, hyperparameters: { ...options, dimensionWeights }, adapters, biases, variance, history, ledger, best_state: nextBest, stale_epochs: nextStale, random_state: random.state() };
    const file = path.join(checkpointDir, `epoch_${String(epoch).padStart(3, '0')}.json`), sha256 = await immutableJson(file, checkpoint);
    ledger.push({ epoch, file: path.relative(root, file).split(path.sep).join('/'), sha256, val_total: val.total });
    bestVal = nextBest.val.total; stale = nextStale;
    best = { ...nextBest, checkpoint: ledger[nextBest.epoch - 1] };
    console.log(JSON.stringify({ progress: 'epoch', epoch, epochs: options.epochs, train_total: row.train.total, val, best_epoch: best?.epoch, stale }));
    if (stale >= options.patience) break;
  }
  if (!best) throw new Error('MDPO training produced no checkpoint');
  const finalReferenceHash = digest(await fs.readFile(path.resolve(String(options.reference))));
  if (finalReferenceHash !== referenceHash) throw new Error('Frozen v10 reference changed during MDPO training');
  if (digest(await fs.readFile(path.resolve(String(options.dataset)))) !== datasetHash) throw new Error('MDPO train33 dataset changed during training');
  const bestNetwork = materializeV10BiasTuning(materializeV10Lora(referenceModel.network, best.adapters), best.biases);
  const weightUpdateEvidence = evaluateMdpoWeightUpdateEvidence({ referenceNetwork: referenceModel.network,
    trainedNetwork: bestNetwork, adapters: best.adapters, biases: best.biases, graph: train[0],
    forward: forwardV10TrainingGraph, varianceParameters: best.variance.trainable_parameters });
  const trainUnifiedBaseline = evaluateMdpoTrainingUnifiedMetrics({ network: referenceModel.network, graphs: train,
    forward: forwardV10TrainingGraph, labelsFromOutputs: labelsFromV10TrainingOutputs,
    evaluateMetrics: evaluateReproductionMetrics, views: MULTI_VIEW_NAMES, protocol: REPRODUCTION_METRIC_PROTOCOL.id });
  const trainUnifiedCandidate = evaluateMdpoTrainingUnifiedMetrics({ network: bestNetwork, graphs: train,
    forward: forwardV10TrainingGraph, labelsFromOutputs: labelsFromV10TrainingOutputs,
    evaluateMetrics: evaluateReproductionMetrics, views: MULTI_VIEW_NAMES, protocol: REPRODUCTION_METRIC_PROTOCOL.id });
  const candidate = {
    version: 'layout_model_v10_mdpo_candidate', status: 'diagnostic_only_requires_full_val11_gate', architecture: { ...referenceModel.architecture, adaptation: 'MDPO_LoRA', gaussian_policy: options.learnableVariance ? 'learnable_diagonal_6d_ablation' : 'fixed_diagonal_6d', qwen_inference_input: false, lora_paths: v10LoraPaths(referenceModel.network).map((path) => path.join('.')) },
    reference: { version: referenceModel.version, sha256: referenceHash, frozen_verified: true }, network: bestNetwork,
    mdpo: { dimensions: MDPO_DIMENSIONS, dimension_weights: dimensionWeights, objective: { protocol: options.safetyAlignment ? 'safety_priority_v3_aligned' : 'safety_priority_v2_legacy_training', supervised: 'lambdaSup * original_v10_center_size_style_direction', overall_dpo: 'lambdaDpo * L_DPO_overall', multidimensional: 'lambdaMulti * sum(alpha_d * L_d)', geometry_safe: options.safetyAlignment ? 'lambdaSafe * (overlap_pair_risk + worst_view_overflow + overflow_cvar + object_occlusion + penetration + mesh_intersection + leader_crossing)' : 'lambdaSafe * weighted_main_worst_view_cvar_stereo_text_clarity_leader_crossing', reference_kl: 'lambdaKl * KL(policy || frozen_reference)', center_tail: 'lambdaCenterTail * center_error_CVaR' }, safety_alignment: options.safetyAlignment ? { protocol: 'safety_priority_v3_aligned', overlap_pair_weight: 24, worst_view_overflow_weight: 20, cvar_overflow_weight: 12, final_gate_metrics: ['overlap_pairs', 'worst_view_overflow', 'intersections', 'worst_view_intersections', 'occluded_points', 'object_occlusion', 'penetration', 'mesh_surface_intersection'] } : null, hyperparameters: { ...options, dimensionWeights }, dataset_audit: audit, dataset_sha256: datasetHash, preference_holdout: { policy: preferenceSplit.policy, holdout_sha256: preferenceSplit.holdout_sha256, gradient_pair_count: preferenceSplit.gradientPairs.length, heldout_pair_count: preferenceSplit.holdoutPairs.length }, train_unified_metrics: { baseline: trainUnifiedBaseline, candidate: trainUnifiedCandidate, delta: compareMdpoUnifiedMetrics(trainUnifiedBaseline, trainUnifiedCandidate) }, adapters: best.adapters, biases: best.biases, variance: best.variance, weight_update_evidence: weightUpdateEvidence, trainable_parameters: best.adapters.trainable_parameters + best.biases.trainable_parameters + best.variance.trainable_parameters, best_epoch: best.epoch, best_val: best.val, history, checkpoint_ledger: ledger },
    split_policy: { train33_gradient_only: true, train33_preference_holdout_no_gradient: true, val11_proxy_early_stopping_only: true, full_val11_visual_safety_checkpoint_selection_pending: true, test11_not_loaded_or_used: true },
    validation_gate: { status: 'pending_full_val11_visual_and_safety_gate', active: false, pck_values_in_training_history: 'local_3d_proxy_not_rendered_five_view_PCK' }
  };
  const output = path.join(path.resolve(String(options.outputDir)), 'layout_model_v10_mdpo_candidate.json');
  await writeImmutableJsonBundle([{ file: output, value: candidate }, { file: path.join(path.resolve(String(options.outputDir)), 'checkpoint_ledger.json'),
    value: { version: 'v10_mdpo_checkpoint_ledger_v1', reference_model_sha256: referenceHash, dataset_sha256: datasetHash, selected_without_test: true, best, checkpoints: ledger } }]);
  console.log(JSON.stringify({ output: path.relative(root, output).split(path.sep).join('/'), best_epoch: best.epoch, val: best.val, trainable_parameters: candidate.mdpo.trainable_parameters, reference_hash_unchanged: true }, null, 2));
}

main().catch((error) => { console.error(error.stack || error.message || error); process.exitCode = 1; });
