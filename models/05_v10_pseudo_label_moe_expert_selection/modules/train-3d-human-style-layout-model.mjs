import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { GRAPH_EDGE_FEATURE_NAMES, PURE_3D_LABEL_FEATURE_NAMES, forwardGraphNetwork, graphEdgeFeatures } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildCvFeatures, dgcnnGeometryFeature } from '../lib/cv-feature-encoder.mjs';
import { ANCHOR_LOCAL_LABEL_FEATURE_NAMES, buildSpatialContext, computeSpatialStyleMetrics, pure3DLabelFeatures, anchorLocal3DLabelFeatures, spatialStyleLoss } from '../lib/spatial-style-features.mjs';
import { HETEROGENEOUS_EDGE_FEATURE_NAMES, buildHeterogeneousLayoutGraph, decodeLocalLayoutOutput, heterogeneousArchitectureMetadata, localTargetVector } from '../lib/heterogeneous-layout-graph.mjs';
import { ANCHOR_LABEL_EDGE_FEATURE_NAMES, LABEL_RELATION_EDGE_FEATURE_NAMES } from '../lib/anchor-frame-features.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';
import { scoreLayoutStyleExperts } from '../lib/moe-layout-styles.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const activeModelFile = path.join(root, 'experiments', 'layout_model.json');
const defaultOutput = path.join(root, 'experiments', 'layout_model_v9_3d_human_style_candidate.json');
const defaultReport = path.join(root, 'experiments', 'layout_training_v9_3d_human_style_report.json');
const defaultV10Output = path.join(root, 'experiments', 'layout_model_v10_anchor_frame_candidate.json');
const defaultV10Report = path.join(root, 'experiments', 'layout_training_v10_anchor_frame_report.json');
const leaderPriorFile = path.join(root, 'experiments', 'manual_leader_length_prior.json');
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const square = (value) => value * value;
const clone = (value) => structuredClone(value);
const zeros = (rows, columns) => Array.from({ length: rows }, () => Array.from({ length: columns }, () => 0));
const vector = (length) => Array.from({ length }, () => 0);
const add = (...values) => values[0].map((_, index) => values.reduce((sum, value) => sum + value[index], 0));
const matVec = (weights, values) => weights.map((row) => row.reduce((sum, weight, index) => sum + weight * values[index], 0));
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const softmax = (values) => { const peak = Math.max(...values); const exp = values.map((value) => Math.exp(value - peak)); const total = exp.reduce((sum, value) => sum + value, 0); return exp.map((value) => value / Math.max(total, 1e-9)); };

function parseArgs(argv) {
  const options = { epochs: 80, learningRate: 0.002, styleWeight: 0.2, directionWeight: 1.25, viewWeight: 0.25, worstViewWeight: 2, cvarViewWeight: 1, stereoWeight: 1, textClarityWeight: 3, leaderCrossingWeight: 3.5, architecture: 'v9', hiddenDim: 64, preGnnFnnLayers: 1, messageLayers: 2, transformerLayers: 1, expertCount: 4, seed: 17, output: defaultOutput, report: defaultReport, warmStart: activeModelFile, manifest: manifestFile, styleGridSize: 20, routerStyleWeight: 0.15, expertStyleWeight: 0.35, targetSource: 'manual', pseudoLabels: '', expertSelection: '' };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function rng(seed = 17) { let state = Number(seed) >>> 0 || 17; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
function copyMatrix(value) { return value.map((row) => [...row]); }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function repoPath(relativePath) { return path.join(root, String(relativePath).replaceAll('/', path.sep)); }
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function readText(file) { return fs.readFile(file, 'utf8'); }

function targetVector(manual, candidate, bounds) {
  const radius = Math.max(bounds.radius, 1e-8);
  return [
    ...manual.center.map((value, axis) => (value - candidate.anchor[axis]) / radius),
    ...manual.boxSize.map((value, axis) => Math.log(Math.max(Math.abs(value), 1e-8) / Math.max(Math.abs(candidate.boxSize[axis]), 1e-8)))
  ];
}

function outputToLabel(output, node, bounds) {
  const radius = Math.max(bounds.radius, 1e-8);
  const center = node.frame
    ? node.frame.worldFromLocal({ u: output[0] * radius, v: output[1] * radius, normal: output[2] * radius })
    : node.anchor.map((value, axis) => value + output[axis] * radius);
  const boxSize = node.initial_size.map((value, axis) => Math.max(radius * (axis === 2 ? 0.005 : 0.01), Math.abs(value) * Math.exp(Math.max(-3, Math.min(3, output[axis + 3])))));
  return { ...node, center, boxSize };
}

function nodeTargetVector(node, bounds) {
  return node.frame
    ? localTargetVector(node.manual, node, node.frame, bounds)
    : targetVector(node.manual, node, bounds);
}

function labelsFromOutputs(outputs, graph) {
  return outputs.map((output, index) => outputToLabel(output, graph.nodes[index], graph.bounds));
}

function viewConditionedLoss(outputs, graph, options) {
  if (!graph.depthGrids) return 0;
  const labels = labelsFromOutputs(outputs, graph);
  const diagnostics = evaluateViewConditionedLayout(labels, graph.bounds, graph.depthGrids, {
    worstViewWeight: Number(options.worstViewWeight),
    cvarViewWeight: Number(options.cvarViewWeight),
    stereoWeight: Number(options.stereoWeight),
    textClarityWeight: Number(options.textClarityWeight),
    leaderCrossingWeight: Number(options.leaderCrossingWeight),
    overlapPairWeight: Number(options.overlapPairWeight ?? 0),
    worstOverflowWeight: Number(options.worstOverflowWeight ?? 0),
    cvarOverflowWeight: Number(options.cvarOverflowWeight ?? 0)
  });
  return Number(diagnostics.objective || 0);
}

function v10GeometrySafetyLoss(outputs, graph, options) {
  const viewWeight = Number(options.viewWeight);
  if (!Number.isFinite(viewWeight) || viewWeight < 0) throw new Error('Invalid v10 geometry-safety loss weight');
  return viewWeight * viewConditionedLoss(outputs, graph, options) / Math.max(1, graph.nodes.length);
}

function makeV9Network(source, options) {
  if (!source?.network) throw new Error('训练需要从已有图模型继承 GNN/Transformer 参数；当前活动模型没有 network');
  const network = clone(source.network);
  const hiddenDim = Number(options.hiddenDim);
  if (hiddenDim !== network.input.weights.length) throw new Error(`hiddenDim=${hiddenDim} 与 warm start=${network.input.weights.length} 不一致`);
  if (network.input.weights[0].length !== PURE_3D_LABEL_FEATURE_NAMES.length) throw new Error('生成器输入不是 51D');
  network.fusion.visual_dim = 0;
  network.fusion.weights = network.fusion.weights.map((row) => row.slice(0, network.fusion.label_dim + network.fusion.geometry_dim));
  const requestedEdgeDim = options.architecture === 'v10' ? HETEROGENEOUS_EDGE_FEATURE_NAMES.length : GRAPH_EDGE_FEATURE_NAMES.length;
  network.message_layers = network.message_layers.map((layer) => {
    const current = layer.edge_weights?.[0]?.length || 0;
    const expanded = current === requestedEdgeDim ? layer.edge_weights : layer.edge_weights.map((row) => [...row, ...Array(Math.max(0, requestedEdgeDim - current)).fill(0)]);
    if (options.architecture !== 'v10') return { ...layer, edge_weights: expanded };
    return {
      ...layer,
      edge_weights: expanded,
      relation_edge_weights: layer.relation_edge_weights || expanded.map((row) => row.slice(0, LABEL_RELATION_EDGE_FEATURE_NAMES.length)),
      anchor_edge_weights: layer.anchor_edge_weights || expanded.map((row) => row.slice(LABEL_RELATION_EDGE_FEATURE_NAMES.length, LABEL_RELATION_EDGE_FEATURE_NAMES.length + ANCHOR_LABEL_EDGE_FEATURE_NAMES.length))
    };
  });
  const names = ['spherical_contour_style', 'rectangular_edge_style', 'anchor_surround_style', 'balanced_style', 'directional_style'];
  if (Number(options.expertCount) !== 4 && Number(options.expertCount) !== 5) throw new Error('v9 MoE 专家数只能是 4 或 5');
  network.moe.experts = network.moe.experts.slice(0, Number(options.expertCount));
  while (network.moe.experts.length < Number(options.expertCount)) {
    const template = network.moe.experts[0];
    network.moe.experts.push(clone(template));
  }
  network.moe.experts.forEach((expert, index) => { expert.name = names[index]; });
  network.moe.router.weights = network.moe.router.weights.slice(0, Number(options.expertCount));
  network.moe.router.bias = network.moe.router.bias.slice(0, Number(options.expertCount));
  return network;
}

function forwardHidden(network, graph) {
  const features = graph.nodeFeatures || graph.nodes.map((node, index) => pure3DLabelFeatures(node, graph.bounds, index, graph.nodes.length, graph.spatialContext, graph.nodes));
  const inputHidden = features.map((values) => matVec(network.input.weights, values).map((value, index) => Math.tanh(value + network.input.bias[index])));
  let hidden = inputHidden;
  const preGnnCaches = [];
  for (const layer of network.pre_gnn_fnn_layers || []) {
    const input = hidden;
    const ffnHidden = input.map((values) => add(matVec(layer.hidden_weights, values), layer.hidden_bias).map(Math.tanh));
    const delta = ffnHidden.map((values) => add(matVec(layer.output_weights, values), layer.output_bias).map(Math.tanh));
    hidden = input.map((values, index) => add(values, delta[index]));
    preGnnCaches.push({ input, ffnHidden, delta, output: hidden });
  }
  const fusionInput = hidden.map((values) => [...values, ...graph.geometryFeature]);
  const fusionHidden = fusionInput.map((values) => add(matVec(network.fusion.weights, values), network.fusion.bias).map(Math.tanh));
  hidden = fusionHidden;
  const messageCaches = [];
  for (const layer of network.message_layers) {
    const previous = hidden;
    hidden = previous.map((self, targetIndex) => {
      const heterogeneous = Boolean(graph.heterogeneousGraph && layer.anchor_edge_weights && layer.relation_edge_weights);
      const messages = previous.flatMap((source, sourceIndex) => {
        if (sourceIndex === targetIndex) return [];
        const relation = heterogeneous ? graph.heterogeneousGraph.relationEdges[targetIndex][sourceIndex] : graph.edges[targetIndex][sourceIndex];
        const relationWeights = heterogeneous ? layer.relation_edge_weights : layer.edge_weights;
        return [add(matVec(layer.neighbor_weights, source), matVec(relationWeights, relation))];
      });
      const aggregate = messages.length ? messages[0].map((_, feature) => mean(messages.map((message) => message[feature]))) : vector(self.length);
      const anchorMessages = heterogeneous ? graph.heterogeneousGraph.anchorEdgeFeatures[targetIndex].map((features) => matVec(layer.anchor_edge_weights, features)) : [];
      const anchorContext = anchorMessages.length ? anchorMessages[0].map((_, feature) => mean(anchorMessages.map((message) => message[feature]))) : vector(self.length);
      return add(matVec(layer.self_weights, self), aggregate, anchorContext, layer.bias).map(Math.tanh);
    });
    messageCaches.push({ input: previous, output: hidden });
  }
  const transformerCaches = [];
  for (const layer of network.transformer_layers || []) {
    const scale = Math.sqrt(Math.max(1, hidden[0]?.length || 1));
    const queries = hidden.map((values) => add(matVec(layer.query_weights, values), layer.query_bias));
    const keys = hidden.map((values) => add(matVec(layer.key_weights, values), layer.key_bias));
    const values = hidden.map((input) => add(matVec(layer.value_weights, input), layer.value_bias));
    const attention = queries.map((query) => softmax(keys.map((key) => dot(query, key) / scale)));
    const context = attention.map((row) => values[0].map((_, feature) => row.reduce((sum, weight, source) => sum + weight * values[source][feature], 0)));
    const attentionOutput = hidden.map((input, index) => add(input, matVec(layer.output_weights, context[index]).map((value, feature) => Math.tanh(value + layer.output_bias[feature]))));
    const ffnHidden = attentionOutput.map((input) => add(matVec(layer.ffn_in_weights, input), layer.ffn_in_bias).map(Math.tanh));
    const output = attentionOutput.map((input, index) => add(input, matVec(layer.ffn_out_weights, ffnHidden[index]), layer.ffn_out_bias));
    transformerCaches.push({ input: hidden, queries, keys, values, attention, context, attentionOutput, ffnHidden, output });
    hidden = output;
  }
  const gates = hidden.map((values) => softmax(add(matVec(network.moe.router.weights, values), network.moe.router.bias)));
  const expertHeads = hidden.map((values) => network.moe.experts.map((expert) => add(matVec(expert.hidden_weights, values), expert.hidden_bias).map(Math.tanh)));
  const expertOutputs = expertHeads.map((heads) => heads.map((head, expertIndex) => add(matVec(network.moe.experts[expertIndex].weights, head), network.moe.experts[expertIndex].bias)));
  const output = expertOutputs.map((experts, index) => experts[0].map((_, feature) => experts.reduce((sum, expert, expertIndex) => sum + gates[index][expertIndex] * expert[feature], 0)));
  return { output, cache: { inputHidden, preGnnCaches, fusionInput, fusionHidden, messageCaches, transformerCaches, finalHidden: hidden, gates, expertHeads, expertOutputs }, features };
}

function nodeStyleLoss(output, node, graph) {
  const predicted = outputToLabel(output, node, graph.bounds);
  const target = node.manual;
  const radius = Math.max(graph.bounds.radius, 1e-8);
  const predictedLeader = distance(predicted.anchor, predicted.center) / radius;
  const targetLeader = distance(target.anchor, target.center) / radius;
  const predictedRadial = distance(predicted.center, graph.bounds.center) / radius;
  const targetRadial = distance(target.center, graph.bounds.center) / radius;
  const predictedAspect = Math.log(Math.max(predicted.boxSize[0], 1e-8) / Math.max(predicted.boxSize[1], 1e-8));
  const targetAspect = Math.log(Math.max(target.boxSize[0], 1e-8) / Math.max(target.boxSize[1], 1e-8));
  return ((predictedLeader - targetLeader) / 0.5) ** 2 + ((predictedRadial - targetRadial) / 1.5) ** 2 + ((predictedAspect - targetAspect) / 2) ** 2;
}

function nodeLeaderDirectionLoss(output, node, graph) {
  const predicted = outputToLabel(output, node, graph.bounds);
  const target = node.manual;
  const predictedDirection = predicted.center.map((value, axis) => value - predicted.anchor[axis]);
  const targetDirection = target.center.map((value, axis) => value - target.anchor[axis]);
  const predictedLength = Math.max(distance(predicted.anchor, predicted.center), 1e-8);
  const targetLength = Math.max(distance(target.anchor, target.center), 1e-8);
  const cosine = predictedDirection.reduce((sum, value, axis) => sum + value * targetDirection[axis], 0) / (predictedLength * targetLength);
  return 1 - Math.max(-1, Math.min(1, cosine));
}

function leaderDirectionLoss(outputs, graph) {
  return mean(outputs.map((output, index) => nodeLeaderDirectionLoss(output, graph.nodes[index], graph)));
}

function targetRouterWeights(network, graph) {
  const styleWeights = graph.expertSelection?.router_target || graph.styleRouting?.weights || {};
  const raw = network.moe.experts.map((expert) => {
    const name = String(expert.name || '').toLowerCase();
    if (name.includes('spherical')) return Number(styleWeights.spherical || 0);
    if (name.includes('rectangular')) return Number(styleWeights.rectangular || 0);
    if (name.includes('surround')) return Number(styleWeights.surround || 0);
    return 0.02;
  });
  const total = raw.reduce((sum, value) => sum + Math.max(0, value), 0) || 1;
  return raw.map((value) => Math.max(0, value) / total);
}

function routerStyleLoss(run, network, graph) {
  if (!graph.styleRouting?.weights && !graph.expertSelection?.router_target || !run.cache?.gates?.length) return 0;
  const target = targetRouterWeights(network, graph);
  return mean(run.cache.gates.map((gates) => -target.reduce((sum, value, index) => sum + value * Math.log(Math.max(gates[index], 1e-8)), 0)));
}
function supervisedObjectiveLoss(outputs, graph, options) {
  const regression = mean(outputs.flatMap((output, index) => output.map((value, axis) => square(value - graph.nodes[index].target[axis]))));
  const style = mean(outputs.map((output, index) => nodeStyleLoss(output, graph.nodes[index], graph)));
  const direction = leaderDirectionLoss(outputs, graph);
  const router_style = 0;
  const total = Number(options.baseWeight ?? 1) * regression + Number(options.styleWeight) * style + Number(options.directionWeight) * direction;
  if (![regression, style, direction, router_style, total].every(Number.isFinite)) throw new Error('Non-finite v10 supervised objective');
  return { regression, style, leader_direction: direction, router_style, total };
}

function updateVector(target, gradient, learningRate) { for (let index = 0; index < target.length; index += 1) { const base = Number.isFinite(Number(target[index])) ? Number(target[index]) : 0; const delta = Number(gradient?.[index]); target[index] = Number.isFinite(delta) ? base - learningRate * delta : base; } }
function updateMatrix(target, gradient, learningRate) { for (let row = 0; row < target.length; row += 1) for (let column = 0; column < target[row].length; column += 1) { const base = Number.isFinite(Number(target[row][column])) ? Number(target[row][column]) : 0; const delta = Number(gradient?.[row]?.[column]); target[row][column] = Number.isFinite(delta) ? base - learningRate * delta : base; } }
function vectorAddInPlace(target, source, scale = 1) { for (let index = 0; index < target.length; index += 1) target[index] += source[index] * scale; }
function transposeVec(weights, values) { return weights[0].map((_, column) => weights.reduce((sum, row, index) => sum + row[column] * values[index], 0)); }
function outerAdd(target, left, right, scale = 1) { for (let row = 0; row < target.length; row += 1) for (let column = 0; column < target[row].length; column += 1) target[row][column] += left[row] * right[column] * scale; }

function zeroGrad(network) {
  return {
    input: { weights: zeros(network.input.weights.length, network.input.weights[0].length), bias: vector(network.input.bias.length) },
    pre_gnn_fnn_layers: network.pre_gnn_fnn_layers.map((layer) => ({ hidden_weights: zeros(layer.hidden_weights.length, layer.hidden_weights[0].length), hidden_bias: vector(layer.hidden_bias.length), output_weights: zeros(layer.output_weights.length, layer.output_weights[0].length), output_bias: vector(layer.output_bias.length) })),
    fusion: { weights: zeros(network.fusion.weights.length, network.fusion.weights[0].length), bias: vector(network.fusion.bias.length) },
    message_layers: network.message_layers.map((layer) => ({
      self_weights: zeros(layer.self_weights.length, layer.self_weights[0].length),
      neighbor_weights: zeros(layer.neighbor_weights.length, layer.neighbor_weights[0].length),
      edge_weights: zeros(layer.edge_weights.length, layer.edge_weights[0].length),
      relation_edge_weights: layer.relation_edge_weights ? zeros(layer.relation_edge_weights.length, layer.relation_edge_weights[0].length) : null,
      anchor_edge_weights: layer.anchor_edge_weights ? zeros(layer.anchor_edge_weights.length, layer.anchor_edge_weights[0].length) : null,
      bias: vector(layer.bias.length)
    })),
    transformer_layers: network.transformer_layers.map((layer) => ({ query_weights: zeros(layer.query_weights.length, layer.query_weights[0].length), query_bias: vector(layer.query_bias.length), key_weights: zeros(layer.key_weights.length, layer.key_weights[0].length), key_bias: vector(layer.key_bias.length), value_weights: zeros(layer.value_weights.length, layer.value_weights[0].length), value_bias: vector(layer.value_bias.length), output_weights: zeros(layer.output_weights.length, layer.output_weights[0].length), output_bias: vector(layer.output_bias.length), ffn_in_weights: zeros(layer.ffn_in_weights.length, layer.ffn_in_weights[0].length), ffn_in_bias: vector(layer.ffn_in_bias.length), ffn_out_weights: zeros(layer.ffn_out_weights.length, layer.ffn_out_weights[0].length), ffn_out_bias: vector(layer.ffn_out_bias.length) })),
    moe: { router: { weights: zeros(network.moe.router.weights.length, network.moe.router.weights[0].length), bias: vector(network.moe.router.bias.length) }, experts: network.moe.experts.map((expert) => ({ hidden_weights: zeros(expert.hidden_weights.length, expert.hidden_weights[0].length), hidden_bias: vector(expert.hidden_bias.length), weights: zeros(expert.weights.length, expert.weights[0].length), bias: vector(expert.bias.length) })) }
  };
}

function backwardTransformer(layer, cache, outputGrad, gradient) {
  const count = cache.input.length;
  const hiddenDim = cache.input[0].length;
  const attentionOutputGrad = Array.from({ length: count }, () => vector(hiddenDim));
  for (let index = 0; index < count; index += 1) {
    vectorAddInPlace(attentionOutputGrad[index], outputGrad[index]);
    outerAdd(gradient.ffn_out_weights, outputGrad[index], cache.ffnHidden[index]);
    vectorAddInPlace(gradient.ffn_out_bias, outputGrad[index]);
    const ffnGrad = transposeVec(layer.ffn_out_weights, outputGrad[index]).map((value, feature) => value * (1 - cache.ffnHidden[index][feature] ** 2));
    outerAdd(gradient.ffn_in_weights, ffnGrad, cache.attentionOutput[index]);
    vectorAddInPlace(gradient.ffn_in_bias, ffnGrad);
    vectorAddInPlace(attentionOutputGrad[index], transposeVec(layer.ffn_in_weights, ffnGrad));
  }
  const inputGrad = attentionOutputGrad.map((values) => [...values]);
  const contextGrad = Array.from({ length: count }, () => vector(hiddenDim));
  for (let index = 0; index < count; index += 1) {
    const preGrad = attentionOutputGrad[index].map((value, feature) => value * (1 - cache.attentionOutput[index][feature] ** 2));
    outerAdd(gradient.output_weights, preGrad, cache.context[index]);
    vectorAddInPlace(gradient.output_bias, preGrad);
    vectorAddInPlace(contextGrad[index], transposeVec(layer.output_weights, preGrad));
  }
  const queryGrad = Array.from({ length: count }, () => vector(hiddenDim));
  const keyGrad = Array.from({ length: count }, () => vector(hiddenDim));
  const valueGrad = Array.from({ length: count }, () => vector(hiddenDim));
  const scale = Math.sqrt(Math.max(1, hiddenDim));
  for (let target = 0; target < count; target += 1) {
    const attentionValueGrad = cache.values.map((values) => dot(contextGrad[target], values));
    const weighted = cache.attention[target].reduce((sum, weight, source) => sum + weight * attentionValueGrad[source], 0);
    for (let source = 0; source < count; source += 1) {
      vectorAddInPlace(valueGrad[source], contextGrad[target], cache.attention[target][source]);
      const scoreGrad = cache.attention[target][source] * (attentionValueGrad[source] - weighted);
      vectorAddInPlace(queryGrad[target], cache.keys[source], scoreGrad / scale);
      vectorAddInPlace(keyGrad[source], cache.queries[target], scoreGrad / scale);
    }
  }
  for (let index = 0; index < count; index += 1) {
    outerAdd(gradient.query_weights, queryGrad[index], cache.input[index]); vectorAddInPlace(gradient.query_bias, queryGrad[index]);
    outerAdd(gradient.key_weights, keyGrad[index], cache.input[index]); vectorAddInPlace(gradient.key_bias, keyGrad[index]);
    outerAdd(gradient.value_weights, valueGrad[index], cache.input[index]); vectorAddInPlace(gradient.value_bias, valueGrad[index]);
    vectorAddInPlace(inputGrad[index], transposeVec(layer.query_weights, queryGrad[index]));
    vectorAddInPlace(inputGrad[index], transposeVec(layer.key_weights, keyGrad[index]));
    vectorAddInPlace(inputGrad[index], transposeVec(layer.value_weights, valueGrad[index]));
  }
  return inputGrad;
}

function outputGradient(run, graph, options) {
  const baseWeight = options.baseWeight === undefined ? 1 : Number(options.baseWeight);
  const styleWeight = Number(options.styleWeight);
  const directionWeight = Number(options.directionWeight);
  const viewWeight = Number(options.viewWeight);
  if (![baseWeight, styleWeight, directionWeight, viewWeight].every(Number.isFinite)) throw new Error('Invalid v10 output-loss weights');
  return run.output.map((output, nodeIndex) => {
    const node = graph.nodes[nodeIndex];
    const base = output.map((value, axis) => baseWeight * 2 * (value - node.target[axis]) / 6);
    const epsilon = 1e-4;
    const style = output.map((_, axis) => {
      if (styleWeight === 0) return 0;
      const above = [...output], below = [...output];
      above[axis] += epsilon; below[axis] -= epsilon;
      return (nodeStyleLoss(above, node, graph) - nodeStyleLoss(below, node, graph)) / (2 * epsilon);
    });
    const direction = output.map((_, axis) => {
      if (directionWeight === 0) return 0;
      const above = [...output], below = [...output];
      above[axis] += epsilon; below[axis] -= epsilon;
      return (nodeLeaderDirectionLoss(above, node, graph) - nodeLeaderDirectionLoss(below, node, graph)) / (2 * epsilon);
    });
    const view = output.map((_, axis) => {
      if (!graph.depthGrids || viewWeight === 0 || options.largeDataset) return 0;
      const epsilon = 1e-4;
      const above = [...output];
      const below = [...output];
      above[axis] += epsilon;
      below[axis] -= epsilon;
      const aboveOutputs = run.output.map((values, index) => index === nodeIndex ? above : values);
      const belowOutputs = run.output.map((values, index) => index === nodeIndex ? below : values);
      return (viewConditionedLoss(aboveOutputs, graph, options) - viewConditionedLoss(belowOutputs, graph, options)) / (2 * epsilon);
    });
    return base.map((value, axis) => (value + styleWeight * style[axis] + directionWeight * direction[axis] + viewWeight * view[axis]) / Math.max(1, graph.nodes.length));
  });
}

function backward(network, graph, run, options, explicitOutputGradient = null) {
  const gradient = zeroGrad(network);
  const nodeCount = graph.nodes.length;
  let hiddenGrad = Array.from({ length: nodeCount }, () => vector(run.cache.finalHidden[0].length));
  let outputsGradient = explicitOutputGradient || outputGradient(run, graph, options);
  if (outputsGradient.length !== nodeCount || outputsGradient.some((row) => row.length !== 6 || row.some((value) => !Number.isFinite(value)))) {
    options.nonFiniteGradientCount = Number(options.nonFiniteGradientCount || 0) + 1;
    outputsGradient = Array.from({ length: nodeCount }, (_, nodeIndex) => Array.from({ length: 6 }, (_, axis) => Number.isFinite(Number(outputsGradient?.[nodeIndex]?.[axis])) ? Number(outputsGradient[nodeIndex][axis]) : 0));
  }
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const outputGrad = outputsGradient[nodeIndex];
    const hidden = run.cache.finalHidden[nodeIndex];
    const gates = run.cache.gates[nodeIndex];
    const sensitivities = network.moe.experts.map((_, expertIndex) => dot(outputGrad, run.cache.expertOutputs[nodeIndex][expertIndex]));
    const gateMean = gates.reduce((sum, gate, expertIndex) => sum + gate * sensitivities[expertIndex], 0);
    const routerGrad = gates.map((gate, expertIndex) => gate * (sensitivities[expertIndex] - gateMean));
    if (Number(options.routerStyleWeight || 0) && graph.styleRouting?.weights) {
      const targetGate = targetRouterWeights(network, graph);
      for (let expertIndex = 0; expertIndex < routerGrad.length; expertIndex += 1) routerGrad[expertIndex] += Number(options.routerStyleWeight) * (gates[expertIndex] - targetGate[expertIndex]) / Math.max(1, nodeCount);
    }
    outerAdd(gradient.moe.router.weights, routerGrad, hidden); vectorAddInPlace(gradient.moe.router.bias, routerGrad); vectorAddInPlace(hiddenGrad[nodeIndex], transposeVec(network.moe.router.weights, routerGrad));
    network.moe.experts.forEach((expert, expertIndex) => {
      const expertOutputGrad = outputGrad.map((value) => value * gates[expertIndex]);
      if (graph.expertTargets && Number(options.expertStyleWeight || 0)) {
        const style = expertIndex === 0 ? 'spherical' : expertIndex === 1 ? 'rectangular' : expertIndex === 2 ? 'surround' : null;
        const target = style && graph.expertTargets[style]?.[nodeIndex];
        if (target) for (let axis = 0; axis < expertOutputGrad.length; axis += 1) expertOutputGrad[axis] += Number(options.expertStyleWeight) * 2 * (run.cache.expertOutputs[nodeIndex][expertIndex][axis] - target[axis]) / 6 / Math.max(1, nodeCount);
      }
      const head = run.cache.expertHeads[nodeIndex][expertIndex];
      outerAdd(gradient.moe.experts[expertIndex].weights, expertOutputGrad, head); vectorAddInPlace(gradient.moe.experts[expertIndex].bias, expertOutputGrad);
      const headGrad = transposeVec(expert.weights, expertOutputGrad).map((value, feature) => value * (1 - head[feature] ** 2));
      outerAdd(gradient.moe.experts[expertIndex].hidden_weights, headGrad, hidden); vectorAddInPlace(gradient.moe.experts[expertIndex].hidden_bias, headGrad); vectorAddInPlace(hiddenGrad[nodeIndex], transposeVec(expert.hidden_weights, headGrad));
    });
  }
  for (let index = network.transformer_layers.length - 1; index >= 0; index -= 1) hiddenGrad = backwardTransformer(network.transformer_layers[index], run.cache.transformerCaches[index], hiddenGrad, gradient.transformer_layers[index]);
  for (let layerIndex = network.message_layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
    const layer = network.message_layers[layerIndex], cache = run.cache.messageCaches[layerIndex];
    const inputGrad = Array.from({ length: nodeCount }, () => vector(cache.input[0].length));
    const preGrad = hiddenGrad.map((values, nodeIndex) => values.map((value, feature) => value * (1 - cache.output[nodeIndex][feature] ** 2)));
    for (let target = 0; target < nodeCount; target += 1) {
      outerAdd(gradient.message_layers[layerIndex].self_weights, preGrad[target], cache.input[target]); vectorAddInPlace(gradient.message_layers[layerIndex].bias, preGrad[target]); vectorAddInPlace(inputGrad[target], transposeVec(layer.self_weights, preGrad[target]));
      const heterogeneous = Boolean(graph.heterogeneousGraph && layer.anchor_edge_weights && layer.relation_edge_weights);
      if (heterogeneous) {
        const anchorDivisor = Math.max(1, graph.heterogeneousGraph.anchorEdgeFeatures[target].length);
        for (const features of graph.heterogeneousGraph.anchorEdgeFeatures[target]) outerAdd(gradient.message_layers[layerIndex].anchor_edge_weights, preGrad[target], features, 1 / anchorDivisor);
      }
      const divisor = Math.max(1, nodeCount - 1);
      for (let source = 0; source < nodeCount; source += 1) {
        if (source === target) continue;
        outerAdd(gradient.message_layers[layerIndex].neighbor_weights, preGrad[target], cache.input[source], 1 / divisor);
        if (heterogeneous) outerAdd(gradient.message_layers[layerIndex].relation_edge_weights, preGrad[target], graph.heterogeneousGraph.relationEdges[target][source], 1 / divisor);
        else outerAdd(gradient.message_layers[layerIndex].edge_weights, preGrad[target], graph.edges[target][source], 1 / divisor);
        vectorAddInPlace(inputGrad[source], transposeVec(layer.neighbor_weights, preGrad[target]), 1 / divisor);
      }
    }
    hiddenGrad = inputGrad;
  }
  const fusionGrad = hiddenGrad.map((values, nodeIndex) => values.map((value, feature) => value * (1 - run.cache.fusionHidden[nodeIndex][feature] ** 2)));
  const labelGrad = Array.from({ length: nodeCount }, () => vector(network.fusion.label_dim));
  for (let index = 0; index < nodeCount; index += 1) {
    outerAdd(gradient.fusion.weights, fusionGrad[index], run.cache.fusionInput[index]); vectorAddInPlace(gradient.fusion.bias, fusionGrad[index]);
    vectorAddInPlace(labelGrad[index], transposeVec(network.fusion.weights, fusionGrad[index]).slice(0, network.fusion.label_dim));
  }
  hiddenGrad = labelGrad;
  for (let layerIndex = network.pre_gnn_fnn_layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
    const layer = network.pre_gnn_fnn_layers[layerIndex], cache = run.cache.preGnnCaches[layerIndex];
    const inputGrad = hiddenGrad.map((values) => [...values]);
    for (let index = 0; index < nodeCount; index += 1) {
      const deltaGrad = hiddenGrad[index].map((value, feature) => value * (1 - cache.delta[index][feature] ** 2));
      outerAdd(gradient.pre_gnn_fnn_layers[layerIndex].output_weights, deltaGrad, cache.ffnHidden[index]); vectorAddInPlace(gradient.pre_gnn_fnn_layers[layerIndex].output_bias, deltaGrad);
      const ffnGrad = transposeVec(layer.output_weights, deltaGrad).map((value, feature) => value * (1 - cache.ffnHidden[index][feature] ** 2));
      outerAdd(gradient.pre_gnn_fnn_layers[layerIndex].hidden_weights, ffnGrad, cache.input[index]); vectorAddInPlace(gradient.pre_gnn_fnn_layers[layerIndex].hidden_bias, ffnGrad); vectorAddInPlace(inputGrad[index], transposeVec(layer.hidden_weights, ffnGrad));
    }
    hiddenGrad = inputGrad;
  }
  for (let index = 0; index < nodeCount; index += 1) {
    const inputGrad = hiddenGrad[index].map((value, feature) => value * (1 - run.cache.inputHidden[index][feature] ** 2));
    outerAdd(gradient.input.weights, inputGrad, run.features[index]); vectorAddInPlace(gradient.input.bias, inputGrad);
  }
  return gradient;
}

function applyGradient(network, gradient, learningRate) {
  updateMatrix(network.input.weights, gradient.input.weights, learningRate); updateVector(network.input.bias, gradient.input.bias, learningRate);
  network.pre_gnn_fnn_layers.forEach((layer, index) => { const grad = gradient.pre_gnn_fnn_layers[index]; updateMatrix(layer.hidden_weights, grad.hidden_weights, learningRate); updateVector(layer.hidden_bias, grad.hidden_bias, learningRate); updateMatrix(layer.output_weights, grad.output_weights, learningRate); updateVector(layer.output_bias, grad.output_bias, learningRate); });
  updateMatrix(network.fusion.weights, gradient.fusion.weights, learningRate); updateVector(network.fusion.bias, gradient.fusion.bias, learningRate);
  network.message_layers.forEach((layer, index) => {
    const grad = gradient.message_layers[index];
    updateMatrix(layer.self_weights, grad.self_weights, learningRate);
    updateMatrix(layer.neighbor_weights, grad.neighbor_weights, learningRate);
    updateMatrix(layer.edge_weights, grad.edge_weights, learningRate);
    if (layer.relation_edge_weights) updateMatrix(layer.relation_edge_weights, grad.relation_edge_weights, learningRate);
    if (layer.anchor_edge_weights) updateMatrix(layer.anchor_edge_weights, grad.anchor_edge_weights, learningRate);
    updateVector(layer.bias, grad.bias, learningRate);
  });
  network.transformer_layers.forEach((layer, index) => { const grad = gradient.transformer_layers[index]; for (const prefix of ['query', 'key', 'value', 'output', 'ffn_in', 'ffn_out']) { updateMatrix(layer[`${prefix}_weights`], grad[`${prefix}_weights`], learningRate); updateVector(layer[`${prefix}_bias`], grad[`${prefix}_bias`], learningRate); } });
  updateMatrix(network.moe.router.weights, gradient.moe.router.weights, learningRate); updateVector(network.moe.router.bias, gradient.moe.router.bias, learningRate);
  network.moe.experts.forEach((expert, index) => { const grad = gradient.moe.experts[index]; updateMatrix(expert.hidden_weights, grad.hidden_weights, learningRate); updateVector(expert.hidden_bias, grad.hidden_bias, learningRate); updateMatrix(expert.weights, grad.weights, learningRate); updateVector(expert.bias, grad.bias, learningRate); });
}

function trainGraph(network, graph, options) {
  const run = forwardHidden(network, graph);
  applyGradient(network, backward(network, graph, run, options), Number(options.learningRate));
  const labels = labelsFromOutputs(run.output, graph);
  const regression = mean(run.output.flatMap((output, index) => output.map((value, axis) => square(value - graph.nodes[index].target[axis]))));
  const style = spatialStyleLoss(computeSpatialStyleMetrics(labels, graph.bounds, graph.spatialContext), graph.styleTarget, { includeDirectionalUniformity: graph.architecture !== 'v10' });
  const view = options.fastMetrics ? 0 : viewConditionedLoss(run.output, graph, options);
  const direction = leaderDirectionLoss(run.output, graph);
  const crossing = options.fastMetrics ? 0 : evaluateViewConditionedLayout(labels, graph.bounds, graph.depthGrids, { worstViewWeight: Number(options.worstViewWeight), cvarViewWeight: Number(options.cvarViewWeight), leaderCrossingWeight: Number(options.leaderCrossingWeight) }).leader_crossing_objective;
  const router = routerStyleLoss(run, network, graph);
  const expert_style = expertStyleLoss(run, graph);
  return { regression, style, leader_direction: direction, leader_crossing: crossing, view, router_style: router, expert_style, total: regression + Number(options.styleWeight) * style + Number(options.directionWeight) * direction + Number(options.viewWeight) * view + Number(options.routerStyleWeight || 0) * router + Number(options.expertStyleWeight || 0) * expert_style };
}

function graphPredictions(network, graph) {
  const run = forwardHidden(network, graph);
  return { run, labels: labelsFromOutputs(run.output, graph) };
}

function pseudoKey(category, sampleId) { return `${category}/${sampleId}`; }
function buildPseudoLabelMap(bundle) {
  const rows = Array.isArray(bundle?.rows) ? bundle.rows : [];
  return new Map(rows.map((row) => [pseudoKey(row.category, row.sample_id), row.labels]));
}
function buildExpertSelectionMap(bundle) {
  const rows = Array.isArray(bundle?.rows) ? bundle.rows : [];
  return new Map(rows.map((row) => [pseudoKey(row.category, row.sample_id), row]));
}

function resolveTargetLabels(sample, manual, pseudoMap, targetSource) {
  const usePseudo = String(targetSource || 'manual') !== 'manual' && pseudoMap?.has(pseudoKey(sample.category, sample.sample_id));
  if (!usePseudo) return { labels: manual, source: 'manual_adjusted_annotation' };
  const pseudo = pseudoMap.get(pseudoKey(sample.category, sample.sample_id));
  validateFixedLabelContract(manual, pseudo, `${sample.category}/${sample.sample_id}/pseudo-target-contract`);
  return { labels: pseudo, source: 'moe_unsupervised_pseudolabel' };
}
function buildExpertTargets(expertSelection, candidates, heterogeneous, bounds) {
  if (!expertSelection?.experts || !heterogeneous?.frames) return null;
  return Object.fromEntries(Object.entries(expertSelection.experts).map(([style, expert]) => [style, (expert.labels || []).map((label, index) => localTargetVector(label, candidates[index], heterogeneous.frames[index], bounds))]));
}

function expertStyleLoss(run, graph) {
  if (!graph.expertTargets || !run.cache?.expertOutputs?.length) return 0;
  const terms = [];
  for (const [style, targets] of Object.entries(graph.expertTargets)) {
    const expertIndex = style === 'spherical' ? 0 : style === 'rectangular' ? 1 : style === 'surround' ? 2 : undefined;
    if (expertIndex === undefined) continue;
    for (let node = 0; node < targets.length; node += 1) terms.push(mean(run.cache.expertOutputs[node][expertIndex].map((value, axis) => square(value - targets[node][axis]))));
  }
  return mean(terms);
}

async function buildGraphs(manifest, split, gridSize, architecture = 'v9', pseudoMap = null, targetSource = 'manual', expertSelectionMap = null, expertSelectionSource = null, compactGeometry = false) {
  const graphs = [];
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const clean = cleanObj(await readText(repoPath(sample.input.source_obj)));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const geometryFeature = architecture === 'v10'
      ? dgcnnGeometryFeature(clean.text, { pointCount: 1024, neighborCount: 4 })
      : (await buildCvFeatures({ objText: clean.text, bounds })).geometry;
    const annotation = await readJson(repoPath(sample.target.annotation_json));
    if (annotation?.version !== 'after_mannual_adjust' || annotation?.layout_type !== 'manual_adjusted') throw new Error(`${sample.category}/${sample.sample_id} 不是人工调整目标`);
    const manual = annotationsToLabels(annotation);
    const targetResolution = resolveTargetLabels(sample, manual, pseudoMap, targetSource);
    const targetLabels = targetResolution.labels;
    const candidates = fixedCandidatesWithoutTargetLayout(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/${split}`);
    const spatialContext = buildSpatialContext(geometry, bounds, { gridSize, pointLimit: compactGeometry ? 1024 : 0 });
    const heterogeneous = architecture === 'v10' ? buildHeterogeneousLayoutGraph(candidates, geometry, bounds, spatialContext) : null;
    const nodes = candidates.map((candidate, index) => ({
      ...candidate,
      manual: targetLabels[index],
      target_source: targetResolution.source,
      frame: heterogeneous?.frames?.[index] || null,
      target: heterogeneous ? localTargetVector(targetLabels[index], candidate, heterogeneous.frames[index], bounds) : targetVector(targetLabels[index], candidate, bounds),
      initial_center: [...candidate.center],
      initial_size: [...candidate.boxSize],
      features: heterogeneous ? heterogeneous.nodeFeatures[index] : pure3DLabelFeatures(candidate, bounds, index, candidates.length, spatialContext, candidates)
    }));
    const edges = architecture === 'v10' ? null : (heterogeneous?.edgeFeatures || nodes.map((_, targetIndex) => nodes.map((__, sourceIndex) => sourceIndex === targetIndex ? null : graphEdgeFeatures(nodes[sourceIndex], nodes[targetIndex], bounds))));
    const compactHeterogeneous = architecture === 'v10' && heterogeneous ? { frames: heterogeneous.frames, nodeFeatures: heterogeneous.nodeFeatures, anchorEdgeFeatures: heterogeneous.anchorEdgeFeatures, relationEdges: heterogeneous.relationEdges } : heterogeneous;
    const styleTarget = computeSpatialStyleMetrics(targetLabels, bounds, spatialContext);
    const styleRouting = scoreLayoutStyleExperts(candidates, bounds, geometry, { view: 'main' });
    const expertSelection = expertSelectionMap?.get(pseudoKey(sample.category, sample.sample_id)) || null;
    const expertTargets = buildExpertTargets(expertSelection, candidates, heterogeneous, bounds);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    graphs.push({ category: sample.category, sample_id: sample.sample_id, split, architecture, bounds, geometry: compactGeometry ? null : geometry, geometryFeature, visualFeature: null, spatialContext, nodes, edges, nodeFeatures: compactHeterogeneous?.nodeFeatures || null, heterogeneousGraph: compactHeterogeneous, depthGrids: compactGeometry ? null : depthGrids, manualLabels: manual, targetLabels, target_source: targetResolution.source, expertSelection, expertTargets, expert_selection_source: expertSelection ? expertSelectionSource : null, styleTarget, styleRouting });
  }
  return graphs;
}

function evaluateNetwork(network, graphs, options) {
  const losses = graphs.map((graph) => {
    const { run, labels } = graphPredictions(network, graph);
    const regression = mean(run.output.flatMap((output, index) => output.map((value, axis) => square(value - graph.nodes[index].target[axis]))));
    const style = spatialStyleLoss(computeSpatialStyleMetrics(labels, graph.bounds, graph.spatialContext), graph.styleTarget, { includeDirectionalUniformity: graph.architecture !== 'v10' });
    const direction = leaderDirectionLoss(run.output, graph);
    const view = options.fastMetrics ? 0 : viewConditionedLoss(run.output, graph, options);
    const crossing = options.fastMetrics ? 0 : evaluateViewConditionedLayout(labels, graph.bounds, graph.depthGrids, { worstViewWeight: Number(options.worstViewWeight), cvarViewWeight: Number(options.cvarViewWeight), leaderCrossingWeight: Number(options.leaderCrossingWeight) }).leader_crossing_objective;
    const router = routerStyleLoss(run, network, graph);
    const expert_style = expertStyleLoss(run, graph);
    return { regression, style, leader_direction: direction, leader_crossing: crossing, view, router_style: router, expert_style, total: regression + Number(options.styleWeight) * style + Number(options.directionWeight) * direction + Number(options.viewWeight) * view + Number(options.routerStyleWeight || 0) * router + Number(options.expertStyleWeight || 0) * expert_style };
  });
  return { total: mean(losses.map((row) => row.total)), regression: mean(losses.map((row) => row.regression)), human_style: mean(losses.map((row) => row.style)), leader_direction: mean(losses.map((row) => row.leader_direction)), leader_crossing: mean(losses.map((row) => row.leader_crossing)), view_conditioned: mean(losses.map((row) => row.view)), router_style: mean(losses.map((row) => row.router_style)), expert_style: mean(losses.map((row) => row.expert_style)) };
}

function routingSummary(network, graphs) {
  const gates = graphs.flatMap((graph) => forwardHidden(network, graph).cache.gates);
  const names = network.moe.experts.map((expert) => expert.name);
  return { average_weights: Object.fromEntries(names.map((name, index) => [name, Number(mean(gates.map((row) => row[index])).toFixed(6))])), min_weight: Math.min(...gates.flat()), max_weight: Math.max(...gates.flat()) };
}

async function evaluateViewMetrics(network, graphs, leaderPrior) {
  const rows = [];
  for (const graph of graphs) {
    const { labels } = graphPredictions(network, graph);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(graph.geometry, graph.bounds, view)]));
    const safe = optimizeLabels(labels, graph.bounds, { fixedLabels: true, groupPolicy: 'all', viewPolicy: 'binocular', optimizer: 'annealing', iterations: 80, seed: 17, depthGrids, category: graph.category, leaderLengthPrior: leaderPrior });
    validateFixedLabelContract(graph.manualLabels, safe, `${graph.category}/${graph.sample_id}/v9-eval`);
    rows.push({ view: evaluateLayout(safe, graph.bounds, { fixedLabels: true, viewPolicy: 'binocular', depthGrids, category: graph.category, leaderLengthPrior: leaderPrior, manualReference: graph.manualLabels, geometry: graph.geometry }), spatial: computeSpatialStyleMetrics(safe, graph.bounds, graph.spatialContext) });
  }
  const keys = ['multidimensional_quality_score', 'objective_score', 'olr', 'lcd', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'viewport_overflow_ratio', 'multi_view_worst_overflow', 'multi_view_worst_penetration', 'manual_style_distance', 'leader_length_compliance_ratio', 'directional_uniformity', 'air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance', 'min_3d_spacing'];
  const viewKeys = keys.slice(0, 15);
  const spatialKeys = keys.slice(15);
  return { sample_count: rows.length, ...Object.fromEntries(viewKeys.map((key) => [key, Number(mean(rows.map((row) => finite(row.view[key]))).toFixed(6))])), ...Object.fromEntries(spatialKeys.map((key) => [key, Number(mean(rows.map((row) => finite(row.spatial[key]))).toFixed(6))])), rows };
}

function parameterUpdateEvidence(before, after) {
  const groups = [
    ['input_fnn', before.input, after.input],
    ['pre_gnn_fnn', before.pre_gnn_fnn_layers, after.pre_gnn_fnn_layers],
    ['feature_fusion', before.fusion, after.fusion],
    ['message_passing_gnn', before.message_layers, after.message_layers],
    ...(after.message_layers.some((layer) => layer.anchor_edge_weights) ? [
      ['anchor_to_label_messages', before.message_layers.map((layer) => layer.anchor_edge_weights), after.message_layers.map((layer) => layer.anchor_edge_weights)],
      ['label_to_label_messages', before.message_layers.map((layer) => layer.relation_edge_weights), after.message_layers.map((layer) => layer.relation_edge_weights)]
    ] : []),
    ['transformer', before.transformer_layers, after.transformer_layers],
    ['moe_router', before.moe.router, after.moe.router],
    ['moe_experts', before.moe.experts, after.moe.experts]
  ];
  return Object.fromEntries(groups.map(([name, left, right]) => {
    const flatten = (value) => Array.isArray(value)
      ? value.flatMap(flatten)
      : (value && typeof value === 'object' ? Object.values(value).flatMap(flatten) : (Number.isFinite(Number(value)) ? [Number(value)] : []));
    const a = flatten(left), b = flatten(right);
    if (a.length !== b.length) throw new Error(`${name} 参数审计维度不一致：${a.length} vs ${b.length}`);
    const deltas = a.map((value, index) => Math.abs(value - b[index]));
    return [name, { parameter_count: a.length, changed_parameters: deltas.filter((value) => value > 1e-12).length, max_abs_delta: deltas.length ? Math.max(...deltas) : 0, l2_delta: Math.sqrt(deltas.reduce((sum, value) => sum + value * value, 0)) }];
  }));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const requestedArchitecture = String(options.architecture).toLowerCase();
  const architecture = requestedArchitecture === 'v9' && /200train_50test/i.test(String(options.manifest || '')) ? 'v10' : requestedArchitecture;
  if (!['v9', 'v10'].includes(architecture)) throw new Error(`不支持的 architecture=${architecture}，可选 v9/v10`);
  options.architecture = architecture;
  if (architecture !== 'v10') { options.directionWeight = 0; options.viewWeight = 0; }
  if (architecture === 'v10') {
    if (String(options.output) === defaultOutput) options.output = defaultV10Output;
    if (String(options.report) === defaultReport) options.report = defaultV10Report;
  }
  const output = path.resolve(String(options.output));
  if (path.basename(output).toLowerCase() === 'layout_model.json') throw new Error('训练禁止直接覆盖活动模型');
  const manifest = await readJson(path.resolve(String(options.manifest)));
  const source = await readJson(String(options.warmStart));
  const leaderPrior = await fs.readFile(leaderPriorFile, 'utf8').then(JSON.parse).catch(() => null);
  const pseudoBundle = options.pseudoLabels ? await readJson(path.resolve(String(options.pseudoLabels))) : null;
  const pseudoMap = pseudoBundle ? buildPseudoLabelMap(pseudoBundle) : null;
  const expertSelectionBundle = options.expertSelection ? await readJson(path.resolve(String(options.expertSelection))) : null;
  const expertSelectionMap = expertSelectionBundle ? buildExpertSelectionMap(expertSelectionBundle) : null;
  const trainTargetSource = pseudoMap && String(options.targetSource) !== 'manual' ? 'moe_unsupervised_pseudolabel' : 'manual';
  const compactTrainGeometry = manifest.samples.filter((item) => item.split === 'train').length > 100;
  const train = await buildGraphs(manifest, 'train', Number(options.styleGridSize), architecture, pseudoMap, trainTargetSource, expertSelectionMap, expertSelectionBundle ? 'geometry_llm_expert_selection' : null, compactTrainGeometry);
  const val = await buildGraphs(manifest, 'val', Number(options.styleGridSize), architecture, null, 'manual', null, null, false);
  const test = await buildGraphs(manifest, 'test', Number(options.styleGridSize), architecture, null, 'manual', null, null, false);
  const largeDataset = train.length > 100;
  options.largeDataset = largeDataset;
  options.fastMetrics = largeDataset;
  const network = makeV9Network(source, options);
  const before = clone(network);
  const history = [];
  let best = clone(network);
  let bestVal = Infinity;
  let bestEpoch = 0;
  const hasValidation = val.length > 0;
  const startedAt = Date.now();
  let lastHourlyReport = startedAt;
  for (let epoch = 1; epoch <= Number(options.epochs); epoch += 1) {
    const trainLoss = train.map((graph) => trainGraph(network, graph, options));
    const trainMetrics = evaluateNetwork(network, train, options);
    const valMetrics = evaluateNetwork(network, val, options);
    history.push({ epoch, train: { total: mean(trainLoss.map((row) => row.total)), regression: mean(trainLoss.map((row) => row.regression)), human_style: mean(trainLoss.map((row) => row.style)), leader_direction: mean(trainLoss.map((row) => row.leader_direction)), leader_crossing: mean(trainLoss.map((row) => row.leader_crossing)), view_conditioned: mean(trainLoss.map((row) => row.view)) }, val: valMetrics });
    if (!hasValidation || valMetrics.total < bestVal) { bestVal = valMetrics.total; best = clone(network); bestEpoch = epoch; }
    if (global.gc) global.gc();
    if (Date.now() - lastHourlyReport >= 10 * 60 * 1000) {
      lastHourlyReport = Date.now();
      console.log(JSON.stringify({ progress: 'ten_minute', epoch, epochs: Number(options.epochs), elapsed_hours: Number(((Date.now() - startedAt) / 3600000).toFixed(2)), train: history.at(-1).train, val: valMetrics, best_epoch: bestEpoch }, null, 2));
    }
  }
  options.fastMetrics = false;
  const metrics = { train: evaluateNetwork(best, train, { ...options, fastMetrics: true }), val: evaluateNetwork(best, val, options), test: evaluateNetwork(best, test, options) };
  const viewMetrics = { val: await evaluateViewMetrics(best, val, leaderPrior), test: await evaluateViewMetrics(best, test, leaderPrior) };
  const updates = parameterUpdateEvidence(before, best);
  const isV10 = architecture === 'v10';
  const graphMetadata = isV10 ? heterogeneousArchitectureMetadata({ messageLayers: options.messageLayers }) : {};
  const architectureMetadata = {
    type: isV10 ? 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe' : 'fixed_label_3d_human_style_relational_graph_transformer_moe',
    implemented: true,
    generation_input: 'pure_3d',
    node_input_dim: isV10 ? ANCHOR_LOCAL_LABEL_FEATURE_NAMES.length : PURE_3D_LABEL_FEATURE_NAMES.length,
    label_feature_semantics: isV10 ? '51D_anchor_local_coordinate_surface_patch_3d_style' : '51D_pure_3d_anchor_center_box_geometry_spacing_voxel_style',
    label_encoded_dim: 64,
    obj_surface_points: 1024,
    dgcnn_edgeconv_layers: 2,
    dgcnn_neighbor_count: 4,
    geometry_feature_dim: 64,
    geometry_encoder_training: 'deterministic_clean_obj_surface_encoder',
    multiview_image_count: 5,
    visual_backbone: isV10 ? null : 'frozen_lightweight_cnn_2conv',
    visual_feature_dim: isV10 ? 0 : 32,
    visual_input_source: isV10 ? 'not_computed_during_training' : 'evaluation_and_safety_only_clean_obj_renders',
    visual_generation_input: false,
    visual_feature_computed_during_training: !isV10,
    fused_feature_dim: 128,
    fusion_projection: [128, 64],
    edge_input_dim: isV10 ? HETEROGENEOUS_EDGE_FEATURE_NAMES.length : GRAPH_EDGE_FEATURE_NAMES.length,
    hidden_dim: Number(options.hiddenDim),
    pre_gnn_fnn_layers: Number(options.preGnnFnnLayers),
    pre_gnn_fnn_hidden_dim: Number(options.hiddenDim) * 2,
    pre_gnn_fnn_residual: true,
    message_passing_layers: Number(options.messageLayers),
    transformer_layers: Number(options.transformerLayers),
    attention: 'scaled_dot_product_full_label_set_self_attention',
    transformer_residual: true,
    transformer_ffn_dim: Number(options.hiddenDim) * 2,
    moe_router: 'learned_per_node_softmax_human_style',
    moe_expert_count: best.moe.experts.length,
    moe_expert_names: best.moe.experts.map((expert) => expert.name),
    moe_fusion: 'soft_weighted_sum_of_expert_outputs',
    output_head: [64, 32, 6],
    output_semantics: isV10 ? 'local_u_local_v_local_normal_distance_and_log_size_xyz' : '3d_center_residual_xyz_and_log_size_ratio_xyz',
    node_semantics: 'one_node_per_fixed_label',
    edge_semantics: isV10 ? 'anchor_label_heterogeneous_edges_plus_label_label_3d_relations' : 'directed_3d_label_relationship_with_relative_geometry_and_shared_parts',
    layer_semantics: ['pure_3d_label_fnn_51_to_64_to_128_to_64', 'clean_obj_1024_surface_points_2x_edgeconv_64d', '128d_3d_generation_fusion_to_64d', '2_layer_3d_relational_gnn', '1_layer_global_transformer', 'human_style_moe'],
    supervised_target: 'manual_adjusted_3d_center_and_box_size',
    supervised_loss: isV10 ? 'local_frame_3d_center_size_mse_plus_leader_direction_plus_leader_crossing_plus_weighted_main_worst_view_cvar_stereo_text_clarity_loss' : '3d_center_size_mse_plus_manual_human_style_auxiliary_loss',
    human_style_targets: ['uniform_label_distribution', '3d_spacing', 'directional_distribution', 'leader_length', 'leader_direction', 'leader_line_non_crossing', 'font_text_clarity', 'air_space_utilization', 'clearance', 'label_occupancy', 'per_view_free_space_distribution'],
    spatial_grid: { type: 'axis_aligned_surface_voxel_grid', grid_size: Number(options.styleGridSize), metrics: ['air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance', 'min_3d_spacing'] },
    view_loss: { type: 'weighted_main_plus_worst_view_plus_cvar_plus_stereo', weights: { main: 0.4, right: 0.15, left: 0.15, up: 0.15, down: 0.15 }, worst_view_weight: Number(options.worstViewWeight), cvar_weight: Number(options.cvarViewWeight), stereo_weight: Number(options.stereoWeight), text_clarity_weight: Number(options.textClarityWeight), leader_crossing_weight: Number(options.leaderCrossingWeight), terms: ['object_occlusion', 'depth_penetration', 'label_overlap', 'leader_crossing', 'overflow', 'font_text_clarity', 'free_space_distribution_mismatch'], main_view_priority: true },
    style_moe: { router_targets: ['spherical_contour_style', 'rectangular_edge_style', 'anchor_surround_style'], source_features: ['object_contour_point_cloud', 'projected_aspect', 'contour_regularity', 'anchor_angular_entropy', 'anchor_radial_uniformity'], ambiguous_boundary_policy: 'soft_normalized_probabilities_for_middle_regions' },
    view_role: 'five_view_evaluation_safety_optimization_only',
    decoder: isV10 ? 'local_anchor_frame_3d_generator_then_five_view_safety_optimizer' : 'three_d_generator_then_five_view_safety_optimizer',
    label_contract: 'manual_count_id_text_anchor_groups_locked',
    input_provenance: 'fixed_manual_contract_anchor_and_metadata_clean_obj_without_adjusted_center_or_box_size',
    ...(isV10 ? graphMetadata : {})
  };
  const model = {
    version: isV10 ? 'layout_model_v10_anchor_frame_heterogeneous_graph_moe' : 'layout_model_v9_3d_human_style_moe',
    status: 'trained_supervised_manual_3d_human_style_candidate',
    camera_protocol: 'dataset_multiview_reproduction_v1',
    inference: { center_blend: 1, size_blend: 1, size_ratio_range: [0.55, 1.8], generation_input: 'pure_3d', selection_status: 'candidate_requires_val_gate' },
    architecture: architectureMetadata,
    feature_names: isV10 ? ANCHOR_LOCAL_LABEL_FEATURE_NAMES : PURE_3D_LABEL_FEATURE_NAMES, edge_feature_names: isV10 ? HETEROGENEOUS_EDGE_FEATURE_NAMES : GRAPH_EDGE_FEATURE_NAMES,
    split_policy: { train: train.length, val: val.length, test: test.length, manual_center_and_box_are_targets_only: true, validation_selects_checkpoint: hasValidation, test_used_for_final_evaluation_only: true, preference_updates_exclude_test: true },
    hyperparameters: { epochs: Number(options.epochs), learning_rate: Number(options.learningRate), style_weight: Number(options.styleWeight), direction_weight: Number(options.directionWeight), view_weight: Number(options.viewWeight), worst_view_weight: Number(options.worstViewWeight), cvar_view_weight: Number(options.cvarViewWeight), stereo_weight: Number(options.stereoWeight), text_clarity_weight: Number(options.textClarityWeight), leader_crossing_weight: Number(options.leaderCrossingWeight), hidden_dim: Number(options.hiddenDim), pre_gnn_fnn_layers: Number(options.preGnnFnnLayers), message_passing_layers: Number(options.messageLayers), transformer_layers: Number(options.transformerLayers), moe_expert_count: best.moe.experts.length, warm_start: String(options.warmStart), seed: Number(options.seed), spatial_grid_size: Number(options.styleGridSize), router_style_weight: Number(options.routerStyleWeight || 0), expert_style_weight: Number(options.expertStyleWeight || 0), large_dataset_fast_metrics: Boolean(largeDataset), compact_train_geometry: Boolean(compactTrainGeometry), target_source: trainTargetSource, pseudo_labels: options.pseudoLabels ? path.relative(root, path.resolve(String(options.pseudoLabels))).split(path.sep).join('/') : null, expert_selection: options.expertSelection ? path.relative(root, path.resolve(String(options.expertSelection))).split(path.sep).join('/') : null, visual_features_used_by_generator: false },
    network: best,
    training: { train_graphs: train.length, val_graphs: val.length, test_graphs: test.length, train_examples: train.reduce((sum, graph) => sum + graph.nodes.length, 0), val_examples: val.reduce((sum, graph) => sum + graph.nodes.length, 0), test_examples: test.reduce((sum, graph) => sum + graph.nodes.length, 0), best_epoch: bestEpoch, history, metrics, view_metrics: viewMetrics, routing: { train: routingSummary(best, train), val: routingSummary(best, val), test: routingSummary(best, test), style_targets: { train: train.map((graph) => graph.styleRouting.selected), val: val.map((graph) => graph.styleRouting.selected), test: test.map((graph) => graph.styleRouting.selected) } }, parameter_updates: updates, fixed_label_contract_validated: true, training_target_source: trainTargetSource, pseudo_target_count: train.filter((graph) => graph.target_source === 'moe_unsupervised_pseudolabel').length, expert_selection_source: expertSelectionBundle ? 'geometry_llm_expert_selection' : null, expert_selection_count: train.filter((graph) => graph.expertSelection).length, expert_style_supervision_used: train.some((graph) => graph.expertTargets), expert_style_weight: Number(options.expertStyleWeight || 0), adjusted_target_layout_used_as_input: false, test_used_in_gradient_or_checkpoint_selection: false, visual_branch_used_in_generation: false, manual_style_loss_used: true, leader_direction_loss_used: isV10, leader_crossing_loss_used: isV10, per_view_font_clarity_loss_used: isV10 },
    validation_gate: { status: 'candidate_requires_val_gate', criterion: isV10 ? 'v10_val_local_frame_heterogeneous_graph_and_weighted_main_worst_view_cvar_stereo_gate' : 'v9_val_multidimensional_safety_and_pure_3d_generation_gate', selected_without_test: true, test_confirmation_pending: true }
  };
  await fs.writeFile(output, JSON.stringify(model, null, 2) + '\n', 'utf8');
  await fs.writeFile(path.resolve(String(options.report)), JSON.stringify({ version: isV10 ? 'layout_training_v10_anchor_frame_report' : 'layout_training_v9_3d_human_style_report', generated_at: new Date().toISOString(), model_file: path.relative(root, output).split(path.sep).join('/'), architecture: model.architecture, hyperparameters: model.hyperparameters, metrics, view_metrics: viewMetrics, routing: model.training.routing, parameter_updates: updates, best_epoch: bestEpoch, graph_counts: { train: train.length, val: val.length, test: test.length }, fixed_label_contract_validated: true, interpretation: { pure_3d_generation: true, visual_branch_used_by_generator: false, manual_centers_and_sizes_are_targets_only: true, manual_style_loss_used: true, leader_direction_loss_used: isV10, leader_crossing_loss_used: isV10, per_view_font_clarity_loss_used: isV10, view_conditioned_loss_used: isV10, training_target_source: trainTargetSource, pseudo_target_count: train.filter((graph) => graph.target_source === 'moe_unsupervised_pseudolabel').length, main_view_priority: true, worst_view_and_cvar_used: isV10, stereo_consistency_used: isV10, labels_are_nodes: true, label_relationships_are_edges: true, test_used_for_selection: false } }, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ output: path.relative(root, output).split(path.sep).join('/'), best_epoch: bestEpoch, metrics, view_metrics: { val: Object.fromEntries(Object.entries(viewMetrics.val).filter(([key]) => key !== 'rows')), test: Object.fromEntries(Object.entries(viewMetrics.test).filter(([key]) => key !== 'rows')) }, parameter_updates: updates }, null, 2));
}

export { forwardHidden as forwardV10TrainingGraph, backward as backwardV10TrainingGraph, outputGradient as v10SupervisedOutputGradient,
  supervisedObjectiveLoss as v10SupervisedObjectiveLoss, v10GeometrySafetyLoss,
  buildGraphs as buildV10TrainingGraphs, labelsFromOutputs as labelsFromV10TrainingOutputs };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.stack || error.message || error); process.exitCode = 1; });
}









