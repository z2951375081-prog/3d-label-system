import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { MULTI_VIEW_LAYOUT_FEATURE_NAMES, multiViewLayoutFeatures, normalizeGroupName, targetVector } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, projectPointToView, projectLabelToView } from '../lib/layout-optimizer.mjs';
import { DATASET_CAMERA_PROTOCOL } from '../public/dataset-camera.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const outputFile = path.join(root, 'experiments', 'layout_model_provenance_safe_candidate.json');
const reportFile = path.join(root, 'experiments', 'layout_training_provenance_safe_report.json');

const read = (file) => fs.readFile(file, 'utf8');
const repoFile = (relativePath) => path.join(root, relativePath.replaceAll('/', path.sep));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const square = (value) => value * value;

function parseArgs(argv) {
  const options = { epochs: 80, learningRate: 0.003, weightDecay: 0.0001, multiviewWeight: 0.35, seed: 17, output: outputFile, report: reportFile };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function createRng(seed) { let state = Number(seed) >>> 0 || 17; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
function zeros(rows, columns) { return Array.from({ length: rows }, () => Array.from({ length: columns }, () => 0)); }
function cloneRows(rows) { return rows.map((row) => [...row]); }
function dot(row, input) { return row.reduce((sum, value, index) => sum + value * input[index], 0); }

function styleOf(bounds, candidateCount) {
  const shape = bounds.size.map((value) => value / Math.max(bounds.radius, 1e-6));
  const aspect = Math.max(...shape) / Math.max(Math.min(...shape), 1e-6);
  if (aspect > 1.8) return 'elongated';
  if (candidateCount >= 8) return 'dense';
  const symmetry = 1 - Math.min(1, Math.abs(shape[0] - shape[2]) / Math.max(shape[0], shape[2], 1e-6));
  if (symmetry >= 0.88) return 'symmetric';
  return 'balanced';
}

async function buildExamples(manifest, split) {
  const examples = [];
  const selection = {};
  const matching = { matched: 0, fallback: 0 };
  let radiusSum = 0;
  let radiusCount = 0;
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const raw = await read(repoFile(sample.input.source_obj));
    const clean = cleanObj(raw);
    const bounds = boundsFromObj(clean.text);
    const annotation = sample.target.annotation_json ? JSON.parse(await read(repoFile(sample.target.annotation_json))) : null;
    const manualLabels = annotationsToLabels(annotation);
    const generated = generatedCandidatesFromCleanObj(clean.text, bounds);
    if (annotation?.version !== 'after_mannual_adjust' || annotation?.layout_type !== 'manual_adjusted') throw new Error(`${sample.category}/${sample.sample_id} 的目标不是人工调整后标注`);
    const candidates = fixedCandidatesWithoutTargetLayout(manualLabels, generated, bounds);
    validateFixedLabelContract(manualLabels, candidates, `${sample.category}/${sample.sample_id}/${split}`);
    radiusSum += bounds.radius;
    radiusCount += 1;
    const categorySelection = selection[sample.category] ||= { sample_count: 0, label_counts: [], group_stats: {} };
    categorySelection.sample_count += 1;
    categorySelection.label_counts.push(manualLabels.length);
    const positiveGroups = new Set(manualLabels.flatMap((label) => [...(label.targetGroups || []), ...(label.sourceObjs || [])]).map(normalizeGroupName));
    for (const candidate of candidates) {
      const key = normalizeGroupName(candidate.sourceObjs?.[0] || candidate.text);
      const stat = categorySelection.group_stats[key] ||= { occurrences: 0, positives: 0 };
      stat.occurrences += 1;
      if (positiveGroups.has(key)) stat.positives += 1;
    }
    const style = styleOf(bounds, candidates.length);
    manualLabels.forEach((manual, index) => {
      const candidate = candidates[index];
      if (candidate.initialization_source === 'matched_clean_geometry') matching.matched += 1;
      else matching.fallback += 1;
      const target = targetVector(manual, manual.anchor, bounds);
      const example = {
        category: sample.category,
        sample_id: sample.sample_id,
        label_id: manual.id,
        label_text: manual.text,
        style,
        features: multiViewLayoutFeatures(candidate, bounds, index, candidates.length),
        target,
        anchor: [...manual.anchor],
        bounds: { center: [...bounds.center], radius: bounds.radius }
      };
      example.projected_target = projectedLayoutVector(target, example);
      examples.push(example);
    });
  }
  const normalizedSelection = {};
  for (const [category, categorySelection] of Object.entries(selection)) {
    const group_priors = {};
    for (const [key, stat] of Object.entries(categorySelection.group_stats)) group_priors[key] = Number((stat.positives / Math.max(1, stat.occurrences)).toFixed(6));
    normalizedSelection[category] = { sample_count: categorySelection.sample_count, label_count_mean: Number(mean(categorySelection.label_counts).toFixed(4)), label_count_min: Math.min(...categorySelection.label_counts), label_count_max: Math.max(...categorySelection.label_counts), group_priors };
  }
  return { examples, matching, selection: { radius_reference: radiusCount ? radiusSum / radiusCount : 1, categories: normalizedSelection } };
}

function initNetwork(inputDim, outputDim, rng) {
  const dims = [inputDim, 128, 64, 32, outputDim];
  const layers = [];
  for (let layer = 1; layer < dims.length; layer += 1) layers.push({ weights: zeros(dims[layer], dims[layer - 1]).map((row) => row.map(() => (rng() - 0.5) * 0.08)), bias: Array.from({ length: dims[layer] }, () => 0), activation: layer === dims.length - 1 ? 'linear' : 'tanh' });
  return { architecture: { type: 'mlp', layers: dims, activation: 'tanh' }, layers };
}

function activate(value, type) { return type === 'relu' ? Math.max(0, value) : type === 'linear' ? value : Math.tanh(value); }
function derivative(value, type) { return type === 'relu' ? (value > 0 ? 1 : 0) : type === 'linear' ? 1 : 1 - value * value; }
function forward(network, features) { let values = [...features]; const cache = []; for (const layer of network.layers) { const pre = layer.weights.map((row, index) => row.reduce((sum, weight, column) => sum + weight * values[column], layer.bias[index])); const next = pre.map((value) => activate(value, layer.activation)); cache.push({ input: values, pre, output: next }); values = next; } return { output: values, cache }; }

function projectedLayoutVector(output, example) {
  const radius = Math.max(example.bounds.radius, 1e-6);
  const center = example.anchor.map((value, axis) => value + output[axis] * radius);
  const size = output.slice(3, 6).map((value) => Math.max(radius * 0.005, Math.abs(value) * radius));
  return MULTI_VIEW_NAMES.flatMap((view) => {
    const panel = projectLabelToView({center,anchor:example.anchor,boxSize:size}, example.bounds, view);
    return [panel.center.x, panel.center.y, panel.width * 2, panel.height * 2];
  });
}

function lossParts(output, example, multiviewWeight) {
  const regression = mean(output.map((value, index) => square(value - example.target[index])));
  const projected = projectedLayoutVector(output, example);
  const multiview = mean(projected.map((value, index) => square(value - example.projected_target[index])));
  return { regression, multiview, total: (1 - multiviewWeight) * regression + multiviewWeight * multiview };
}

function outputGradient(output, example, multiviewWeight) {
  const regressionGradient = output.map((value, index) => 2 * (value - example.target[index]) / output.length);
  const epsilon = 1e-4;
  const multiviewGradient = output.map((_, index) => {
    const above = [...output];
    const below = [...output];
    above[index] += epsilon;
    below[index] -= epsilon;
    const aboveProjection = projectedLayoutVector(above, example);
    const belowProjection = projectedLayoutVector(below, example);
    const aboveLoss = mean(aboveProjection.map((value, targetIndex) => square(value - example.projected_target[targetIndex])));
    const belowLoss = mean(belowProjection.map((value, targetIndex) => square(value - example.projected_target[targetIndex])));
    return (aboveLoss - belowLoss) / (2 * epsilon);
  });
  return regressionGradient.map((value, index) => (1 - multiviewWeight) * value + multiviewWeight * multiviewGradient[index]);
}

function evaluateNetwork(examples, network, multiviewWeight = 0.35) {
  if (!examples.length) return null;
  const parts = examples.map((example) => lossParts(forward(network, example.features).output, example, multiviewWeight));
  return {
    total: mean(parts.map((item) => item.total)),
    regression: mean(parts.map((item) => item.regression)),
    multiview: mean(parts.map((item) => item.multiview))
  };
}
function shuffled(examples, rng) {
  const result = [...examples];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const target = Math.floor(rng() * (index + 1));
    [result[index], result[target]] = [result[target], result[index]];
  }
  return result;
}

function fitExpert(examples, validationExamples, inputDim, outputDim, options, rng) {
  const network = initNetwork(inputDim, outputDim, rng);
  const lr = Number(options.learningRate);
  const multiviewWeight = Math.max(0, Math.min(1, Number(options.multiviewWeight)));
  let bestNetwork = structuredClone(network);
  let bestVal = Infinity;
  let bestEpoch = 0;
  const history = [];
  for (let epoch = 0; epoch < Number(options.epochs); epoch += 1) {
  for (const example of shuffled(examples, rng)) {
    const pass = forward(network, example.features);
    let gradient = outputGradient(pass.output, example, multiviewWeight);
    for (let layerIndex = network.layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
      const layer = network.layers[layerIndex];
      const current = pass.cache[layerIndex];
      const previousGradient = Array.from({ length: current.input.length }, () => 0);
      for (let row = 0; row < layer.weights.length; row += 1) {
        const localGradient = gradient[row] * derivative(current.output[row], layer.activation);
        for (let column = 0; column < layer.weights[row].length; column += 1) { previousGradient[column] += localGradient * layer.weights[row][column]; layer.weights[row][column] -= lr * (localGradient * current.input[column] + Number(options.weightDecay) * layer.weights[row][column]); }
        layer.bias[row] -= lr * localGradient;
      }
      gradient = previousGradient;
    }
  }
    const trainLoss = evaluateNetwork(examples, network, multiviewWeight);
    const valLoss = evaluateNetwork(validationExamples, network, multiviewWeight) ?? trainLoss;
    history.push({
      epoch: epoch + 1,
      train_loss: Number(trainLoss.total.toFixed(6)),
      val_loss: Number(valLoss.total.toFixed(6)),
      train_mse: Number(trainLoss.regression.toFixed(6)),
      val_mse: Number(valLoss.regression.toFixed(6)),
      train_multiview_projection_mse: Number(trainLoss.multiview.toFixed(6)),
      val_multiview_projection_mse: Number(valLoss.multiview.toFixed(6))
    });
    if (valLoss.total < bestVal) { bestVal = valLoss.total; bestEpoch = epoch + 1; bestNetwork = structuredClone(network); }
  }
  const bestMetrics = evaluateNetwork(validationExamples, bestNetwork, multiviewWeight) ?? evaluateNetwork(examples, bestNetwork, multiviewWeight);
  return {
    network: bestNetwork,
    examples: examples.length,
    validation_examples: validationExamples.length,
    best_epoch: bestEpoch,
    best_val_loss: Number(bestVal.toFixed(6)),
    best_val_mse: Number(bestMetrics.regression.toFixed(6)),
    best_val_multiview_projection_mse: Number(bestMetrics.multiview.toFixed(6)),
    history
  };
}

function evaluateRouted(examples, experts, multiviewWeight) {
  if (!examples.length) return null;
  const parts = examples.map((example) => {
    const prediction = forward((experts[example.style] || experts.balanced).network, example.features).output;
    return lossParts(prediction, example, multiviewWeight);
  });
  return {
    loss: Number(mean(parts.map((item) => item.total)).toFixed(6)),
    mse: Number(mean(parts.map((item) => item.regression)).toFixed(6)),
    multiview_projection_mse: Number(mean(parts.map((item) => item.multiview)).toFixed(6))
  };
}
function evaluateConstant(examples, target) {
  return Number(mean(examples.map((example) => mean(target.map((value, index) => square(value - example.target[index]))))).toFixed(6));
}
function styleCounts(examples) {
  return Object.fromEntries(['balanced', 'elongated', 'dense', 'symmetric'].map((style) => [style, examples.filter((example) => example.style === style).length]));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const resolvedOutput = path.resolve(String(options.output));
  const protectedNames = new Set(['layout_model.json', 'layout_model_multiview.json']);
  if (protectedNames.has(path.basename(resolvedOutput).toLowerCase())) {
    throw new Error('拒绝直接覆盖已有活动/历史布局模型；请输出到 *_candidate.json，并仅通过验证门控激活');
  }
  const manifest = JSON.parse(await read(manifestFile));
  const trainData = await buildExamples(manifest, 'train');
  const valData = await buildExamples(manifest, 'val');
  const testData = await buildExamples(manifest, 'test');
  const train = trainData.examples;
  const val = valData.examples;
  const test = testData.examples;
  if (!train.length) throw new Error('train split 没有可用人工标签，无法训练多视角布局模型');
  if (train.some((example) => example.features.length !== MULTI_VIEW_LAYOUT_FEATURE_NAMES.length)) throw new Error('多视角特征维度不是 51');
  const multiviewWeight = Math.max(0, Math.min(1, Number(options.multiviewWeight)));
  const rng = createRng(options.seed);
  const experts = {};
  for (const style of ['balanced', 'elongated', 'dense', 'symmetric']) {
    const subset = train.filter((example) => example.style === style);
    const valSubset = val.filter((example) => example.style === style);
    experts[style] = fitExpert(subset.length >= 4 ? subset : train, valSubset.length ? valSubset : val, train[0].features.length, train[0].target.length, options, rng);
    experts[style].source_style_examples = subset.length;
    experts[style].fallback_to_all_train = subset.length < 4;
    console.log('expert '+style+': trained '+options.epochs+' epochs');
  }
  const constant = train[0].target.map((_, index) => mean(train.map((example) => example.target[index])));
  const routed = {
    train: evaluateRouted(train, experts, multiviewWeight),
    val: evaluateRouted(val, experts, multiviewWeight),
    test: evaluateRouted(test, experts, multiviewWeight)
  };
  const metrics = {
    train_loss: routed.train.loss,
    val_loss: routed.val.loss,
    test_loss: routed.test.loss,
    train_mse: routed.train.mse,
    val_mse: routed.val.mse,
    test_mse: routed.test.mse,
    train_multiview_projection_mse: routed.train.multiview_projection_mse,
    val_multiview_projection_mse: routed.val.multiview_projection_mse,
    test_multiview_projection_mse: routed.test.multiview_projection_mse,
    constant_baseline_train_mse: evaluateConstant(train, constant),
    constant_baseline_val_mse: evaluateConstant(val, constant),
    constant_baseline_test_mse: evaluateConstant(test, constant)
  };
  const model = {
    version: 'layout_model_v4_multiview_provenance_safe',
    status: 'trained_supervised_fixed_labels_five_view_projection_loss',
    camera_protocol: DATASET_CAMERA_PROTOCOL.id,
    inference: { center_blend: 1, size_blend: 1, size_ratio_range: [0.65, 1.35], selection_status: 'requires_validation_gate' },
    architecture: {
      type: 'fixed_label_multiview_style_gated_mlp',
      implemented: true,
      input_dim: MULTI_VIEW_LAYOUT_FEATURE_NAMES.length,
      expert_mlp_layers: [51, 128, 64, 32, 6],
      expert_styles: Object.keys(experts),
      gate: 'deterministic_geometry_style_gate',
      supervised_loss: 'weighted_3d_parameter_mse_plus_five_view_projected_center_and_box_mse',
      projected_views: [...MULTI_VIEW_NAMES],
      projected_terms_per_view: ['center_x', 'center_y', 'box_width', 'box_height'],
      decoder: 'five_view_simulated_annealing',
      label_contract: 'manual_count_id_text_anchor_groups_locked',
      input_provenance: 'manual_contract_without_adjusted_center_or_box_size',
      projection_contract: 'perspective_camera_aligned_billboard_panel'
    },
    feature_names: MULTI_VIEW_LAYOUT_FEATURE_NAMES,
    split_policy: { train: 33, val: 11, test: 11, manual_is_supervision: true, validation_selects_checkpoint: true, test_labels_used_for_final_evaluation_only: true, preference_updates_exclude_test: true },
    hyperparameters: { epochs: Number(options.epochs), learning_rate: Number(options.learningRate), weight_decay: Number(options.weightDecay), multiview_projection_loss_weight: multiviewWeight, regression_loss_weight: 1 - multiviewWeight, finite_difference_epsilon: 0.0001, seed: Number(options.seed), shuffle_each_epoch: true },
    selection: { ...trainData.selection, status: 'fixed_manual_label_contract', selection_disabled_when_manual_labels_exist: true },
    gate: { thresholds: { dense_count: 8, elongated_aspect: 1.8, symmetric_shape_similarity: 0.88 } },
    experts,
    training: { train_examples: train.length, val_examples: val.length, test_examples: test.length, ...metrics, style_counts: { train: styleCounts(train), val: styleCounts(val), test: styleCounts(test) }, candidate_matching: { train: trainData.matching, val: valData.matching, test: testData.matching }, fixed_label_contract_validated: true },
    generated_at: new Date().toISOString()
  };
  await fs.writeFile(resolvedOutput, `${JSON.stringify(model, null, 2)}\n`, 'utf8');
  const report = { version: 'layout_training_provenance_safe_v4_report', generated_at: model.generated_at, model_file: path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/'), dataset: { manifest: 'experiments/dataset_manifest.json', samples: { train: 33, val: 11, test: 11 }, examples: { train: train.length, val: val.length, test: test.length } }, architecture: model.architecture, hyperparameters: model.hyperparameters, metrics, style_counts: model.training.style_counts, candidate_matching: model.training.candidate_matching, fixed_label_contract_validated: true, best_epochs: Object.fromEntries(Object.entries(experts).map(([style, expert]) => [style, expert.best_epoch])), interpretation: { lower_loss_is_better: true, preference_training_included: false, test_used_in_gradient_or_checkpoint_selection: false, all_five_views_used_as_inputs: true, all_five_views_used_in_supervised_loss: true, adjusted_label_center_or_box_size_used_as_input: false } };
  const resolvedReportFile = path.resolve(String(options.report));
  await fs.writeFile(resolvedReportFile, JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(`完成：train=${train.length}, val=${val.length}, test=${test.length}, train_loss=${metrics.train_loss}, val_loss=${metrics.val_loss}, test_loss=${metrics.test_loss}`);
  console.log(`三维 MSE：train=${metrics.train_mse}, val=${metrics.val_mse}, test=${metrics.test_mse}`);
  console.log(`五视角投影 MSE：train=${metrics.train_multiview_projection_mse}, val=${metrics.val_multiview_projection_mse}, test=${metrics.test_multiview_projection_mse}`);
  console.log(`常数基线：train=${metrics.constant_baseline_train_mse}, val=${metrics.constant_baseline_val_mse}, test=${metrics.constant_baseline_test_mse}`);
  console.log(`输出：${path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/')}`);
  console.log(`报告：${path.relative(root, resolvedReportFile).split(path.sep).join('/')}`);
}

main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });
