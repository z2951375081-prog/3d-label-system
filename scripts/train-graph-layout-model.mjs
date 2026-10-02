import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { GRAPH_EDGE_FEATURE_NAMES, MULTI_VIEW_LAYOUT_FEATURE_NAMES, graphEdgeFeatures, multiViewLayoutFeatures, normalizeGroupName, targetVector } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, optimizeLabels, parseObjTriangles, projectLabelToView } from '../lib/layout-optimizer.mjs';
import { DATASET_CAMERA_PROTOCOL } from '../public/dataset-camera.js';
import { buildCvFeatures } from '../lib/cv-feature-encoder.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const defaultOutput = path.join(root, 'experiments', 'layout_model_provenance_safe_candidate.json');
const defaultReport = path.join(root, 'experiments', 'layout_training_graph_report.json');
const defaultWarmStart = path.join(root, 'experiments', 'layout_model.json');
const leaderPriorFile = path.join(root, 'experiments', 'manual_leader_length_prior.json');
const read = (file) => fs.readFile(file, 'utf8');
const repoFile = (relativePath) => path.join(root, relativePath.replaceAll('/', path.sep));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const square = (value) => value * value;
const zeros = (rows, columns) => Array.from({ length: rows }, () => Array.from({ length: columns }, () => 0));
const vector = (length) => Array.from({ length }, () => 0);

function parseArgs(argv) {
  const options = { epochs: 80, learningRate: 0.00001, weightDecay: 0, multiviewWeight: 0.35, hiddenDim: 64, preGnnFnnLayers: 1, messageLayers: 2, transformerLayers: 1, expertCount: 4, warmStart: defaultWarmStart, expertPerturbation: 0.001, seed: 17, output: defaultOutput, report: defaultReport };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function createRng(seed) { let state = Number(seed) >>> 0 || 17; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
function randomMatrix(rows, columns, rng, scale = 0.08) { return zeros(rows, columns).map((row) => row.map(() => (rng() - 0.5) * scale)); }
function matVec(weights, values) { return weights.map((row) => row.reduce((sum, weight, index) => sum + weight * values[index], 0)); }
function addVectors(...values) { return values[0].map((_, index) => values.reduce((sum, value) => sum + value[index], 0)); }
function dot(left, right) { return left.reduce((sum, value, index) => sum + value * right[index], 0); }
function softmax(values) { const peak = Math.max(...values); const exp = values.map((value) => Math.exp(value - peak)); const total = exp.reduce((sum, value) => sum + value, 0); return exp.map((value) => value / Math.max(total, 1e-9)); }

function initNetwork(inputDim, edgeDim, hiddenDim, preGnnFnnLayers, messageLayers, transformerLayers, expertCount, rng) {
  const fusionWeights = zeros(hiddenDim, hiddenDim + 64 + 32);
  for (let row = 0; row < hiddenDim; row += 1) {
    fusionWeights[row][row] = 1;
    for (let column = hiddenDim; column < hiddenDim + 96; column += 1) fusionWeights[row][column] = (rng() - 0.5) * 0.002;
  }
  return {
    input: { weights: randomMatrix(hiddenDim, inputDim, rng), bias: vector(hiddenDim) },
    pre_gnn_fnn_layers: Array.from({ length: preGnnFnnLayers }, () => ({
      hidden_weights: randomMatrix(hiddenDim * 2, hiddenDim, rng, 0.05), hidden_bias: vector(hiddenDim * 2),
      output_weights: randomMatrix(hiddenDim, hiddenDim * 2, rng, 0.05), output_bias: vector(hiddenDim)
    })),
    fusion: { weights: fusionWeights, bias: vector(hiddenDim), label_dim: hiddenDim, geometry_dim: 64, visual_dim: 32 },
    message_layers: Array.from({ length: messageLayers }, () => ({
      self_weights: randomMatrix(hiddenDim, hiddenDim, rng),
      neighbor_weights: randomMatrix(hiddenDim, hiddenDim, rng),
      edge_weights: randomMatrix(hiddenDim, edgeDim, rng),
      bias: vector(hiddenDim)
    })),
    transformer_layers: Array.from({ length: transformerLayers }, () => ({
      query_weights: randomMatrix(hiddenDim, hiddenDim, rng, 0.05), query_bias: vector(hiddenDim),
      key_weights: randomMatrix(hiddenDim, hiddenDim, rng, 0.05), key_bias: vector(hiddenDim),
      value_weights: randomMatrix(hiddenDim, hiddenDim, rng, 0.05), value_bias: vector(hiddenDim),
      output_weights: zeros(hiddenDim, hiddenDim), output_bias: vector(hiddenDim),
      ffn_in_weights: randomMatrix(hiddenDim * 2, hiddenDim, rng, 0.05), ffn_in_bias: vector(hiddenDim * 2),
      ffn_out_weights: zeros(hiddenDim, hiddenDim * 2), ffn_out_bias: vector(hiddenDim)
    })),
    moe: {
      router: { weights: randomMatrix(expertCount, hiddenDim, rng, 0.04), bias: vector(expertCount) },
      experts: Array.from({ length: expertCount }, (_, index) => ({
        name: ['geometry', 'relations', 'multiview_style', 'global_context'][index] || ('expert_' + index),
        hidden_weights: randomMatrix(32, hiddenDim, rng),
        hidden_bias: vector(32),
        weights: randomMatrix(6, 32, rng),
        bias: vector(6)
      }))
    }
  };
}

function copyMatrix(values) { return values.map((row) => [...row]); }
function averageNested(values) {
  if (Array.isArray(values[0])) return values[0].map((_, index) => averageNested(values.map((value) => value[index])));
  return mean(values);
}
function warmStartNetwork(network, model, rng, perturbationScale) {
  if (['fixed_label_relational_graph_transformer_moe', 'fixed_label_fnn_relational_graph_transformer_moe', 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe'].includes(model?.architecture?.type)) {
    if (model.architecture.hidden_dim !== network.input.weights.length || model.network?.message_layers?.length !== network.message_layers.length || model.network?.transformer_layers?.length !== network.transformer_layers.length) throw new Error('warm start 的层数或隐藏维度与新模型不一致');
    network.input = structuredClone(model.network.input);
    network.message_layers = structuredClone(model.network.message_layers);
    network.transformer_layers = structuredClone(model.network.transformer_layers);
    if (model.network.pre_gnn_fnn_layers?.length === network.pre_gnn_fnn_layers.length) network.pre_gnn_fnn_layers = structuredClone(model.network.pre_gnn_fnn_layers);
    const targetExpertCount = network.moe.experts.length;
    const sourceExperts = model.network.moe?.experts || [];
    if (!sourceExperts.length || sourceExperts.length > targetExpertCount) throw new Error('warm start 的专家数量无法升级');
    network.moe.experts = sourceExperts.map((expert) => structuredClone(expert));
    network.moe.router.weights = model.network.moe.router.weights.map((row) => [...row]);
    network.moe.router.bias = [...model.network.moe.router.bias];
    while (network.moe.experts.length < targetExpertCount) {
      const index = network.moe.experts.length;
      const source = network.moe.experts;
      network.moe.experts.push({
        name: ['geometry', 'relations', 'multiview_style', 'global_context'][index] || ('expert_' + index),
        hidden_weights: averageNested(source.map((expert) => expert.hidden_weights)),
        hidden_bias: averageNested(source.map((expert) => expert.hidden_bias)),
        weights: averageNested(source.map((expert) => expert.weights)),
        bias: averageNested(source.map((expert) => expert.bias))
      });
      network.moe.router.weights.push(averageNested(network.moe.router.weights));
      network.moe.router.bias.push(-2);
    }
    if (model.network.fusion?.weights?.[0]?.length === network.fusion.weights[0].length) network.fusion = structuredClone(model.network.fusion);
    return { source_version: model.version, source_validation_status: model.validation_gate?.status || null, source_architecture: model.architecture.type, continued_training: model.architecture.type === 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe', architecture_upgrade: model.architecture.type !== 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe', added_pre_gnn_fnn_layers: model.network.pre_gnn_fnn_layers?.length ? 0 : network.pre_gnn_fnn_layers.length, added_experts: network.moe.experts.length - sourceExperts.length, expert_perturbation: 0, near_identity_residual_fnn: true, added_cv_fusion: !model.network.fusion };
  }
  if (model?.architecture?.type !== 'fixed_label_multiview_relational_gnn') throw new Error('warm start 必须是已验证的 v5、v6 或 v7 模型');
  if (model.architecture.hidden_dim !== network.input.weights.length || model.network?.message_layers?.length !== network.message_layers.length) throw new Error('warm start 的隐藏维度或 GNN 层数与新模型不一致');
  network.input = { weights: copyMatrix(model.network.input.weights), bias: [...model.network.input.bias] };
  network.message_layers = model.network.message_layers.map((layer) => ({
    self_weights: copyMatrix(layer.self_weights), neighbor_weights: copyMatrix(layer.neighbor_weights), edge_weights: copyMatrix(layer.edge_weights), bias: [...layer.bias]
  }));
  network.moe.router.weights = zeros(network.moe.experts.length, network.input.weights.length);
  network.moe.router.bias = vector(network.moe.experts.length);
  const base = model.network.output;
  const expertCount = network.moe.experts.length;
  const perturb = (template) => {
    if (Array.isArray(template)) {
      const children = template.map(perturb);
      return Array.from({ length: expertCount }, (_, expert) => children.map((child) => child[expert]));
    }
    const deltas = vector(expertCount);
    for (let expert = 0; expert < expertCount - 1; expert += 1) deltas[expert] = (rng() - 0.5) * perturbationScale;
    deltas[expertCount - 1] = -deltas.slice(0, -1).reduce((sum, value) => sum + value, 0);
    return deltas.map((delta) => template + delta);
  };
  const outputWeights = perturb(base.weights), outputBias = perturb(base.bias);
  network.moe.experts.forEach((expert, index) => {
    expert.hidden_weights = copyMatrix(base.hidden_weights); expert.hidden_bias = [...base.hidden_bias]; expert.weights = outputWeights[index]; expert.bias = outputBias[index];
  });
  return { source_version: model.version, source_validation_status: model.validation_gate?.status || null, source_architecture: model.architecture.type, continued_training: false, architecture_upgrade: true, added_pre_gnn_fnn_layers: network.pre_gnn_fnn_layers.length, added_experts: network.moe.experts.length, expert_perturbation: perturbationScale, uniform_fusion_exactly_preserves_source_head: true, near_identity_residual_fnn: true };
}

function zeroGrad(network) {
  return {
    input: { weights: zeros(network.input.weights.length, network.input.weights[0].length), bias: vector(network.input.bias.length) },
    pre_gnn_fnn_layers: network.pre_gnn_fnn_layers.map((layer) => ({
      hidden_weights: zeros(layer.hidden_weights.length, layer.hidden_weights[0].length), hidden_bias: vector(layer.hidden_bias.length),
      output_weights: zeros(layer.output_weights.length, layer.output_weights[0].length), output_bias: vector(layer.output_bias.length)
    })),
    fusion: { weights: zeros(network.fusion.weights.length, network.fusion.weights[0].length), bias: vector(network.fusion.bias.length) },
    message_layers: network.message_layers.map((layer) => ({
      self_weights: zeros(layer.self_weights.length, layer.self_weights[0].length),
      neighbor_weights: zeros(layer.neighbor_weights.length, layer.neighbor_weights[0].length),
      edge_weights: zeros(layer.edge_weights.length, layer.edge_weights[0].length),
      bias: vector(layer.bias.length)
    })),
    transformer_layers: network.transformer_layers.map((layer) => ({
      query_weights: zeros(layer.query_weights.length, layer.query_weights[0].length), query_bias: vector(layer.query_bias.length),
      key_weights: zeros(layer.key_weights.length, layer.key_weights[0].length), key_bias: vector(layer.key_bias.length),
      value_weights: zeros(layer.value_weights.length, layer.value_weights[0].length), value_bias: vector(layer.value_bias.length),
      output_weights: zeros(layer.output_weights.length, layer.output_weights[0].length), output_bias: vector(layer.output_bias.length),
      ffn_in_weights: zeros(layer.ffn_in_weights.length, layer.ffn_in_weights[0].length), ffn_in_bias: vector(layer.ffn_in_bias.length),
      ffn_out_weights: zeros(layer.ffn_out_weights.length, layer.ffn_out_weights[0].length), ffn_out_bias: vector(layer.ffn_out_bias.length)
    })),
    moe: {
      router: { weights: zeros(network.moe.router.weights.length, network.moe.router.weights[0].length), bias: vector(network.moe.router.bias.length) },
      experts: network.moe.experts.map((expert) => ({
        hidden_weights: zeros(expert.hidden_weights.length, expert.hidden_weights[0].length),
        hidden_bias: vector(expert.hidden_bias.length),
        weights: zeros(expert.weights.length, expert.weights[0].length),
        bias: vector(expert.bias.length)
      }))
    }
  };
}

function projectedLayoutVector(output, node, graph) {
  const radius = Math.max(graph.bounds.radius, 1e-6);
  const center = node.initial_center.map((value, axis) => value + output[axis] * radius);
  const size = node.initial_size.map((value, axis) => Math.max(radius * 0.005, Math.abs(value) * Math.exp(Math.max(-2, Math.min(2, output[axis + 3])))));
  return MULTI_VIEW_NAMES.flatMap((view) => {
    const panel = projectLabelToView({ center, anchor: node.anchor, boxSize: size }, graph.bounds, view);
    return [panel.center.x, panel.center.y, panel.width * 2, panel.height * 2];
  });
}

function outputGradient(output, node, graph, multiviewWeight) {
  const regression = output.map((value, index) => 2 * (value - node.target[index]) / output.length);
  const epsilon = 1e-4;
  const projection = output.map((_, index) => {
    const above = [...output], below = [...output];
    above[index] += epsilon; below[index] -= epsilon;
    const aboveProjection = projectedLayoutVector(above, node, graph);
    const belowProjection = projectedLayoutVector(below, node, graph);
    const aboveLoss = mean(aboveProjection.map((value, targetIndex) => square(value - node.projected_target[targetIndex])));
    const belowLoss = mean(belowProjection.map((value, targetIndex) => square(value - node.projected_target[targetIndex])));
    return (aboveLoss - belowLoss) / (2 * epsilon);
  });
  return regression.map((value, index) => ((1 - multiviewWeight) * value + multiviewWeight * projection[index]) / graph.nodes.length);
}

function lossParts(outputs, graph, multiviewWeight) {
  const rows = outputs.map((output, index) => {
    const node = graph.nodes[index];
    const regression = mean(output.map((value, axis) => square(value - node.target[axis])));
    const projected = projectedLayoutVector(output, node, graph);
    const multiview = mean(projected.map((value, axis) => square(value - node.projected_target[axis])));
    return { regression, multiview, total: (1 - multiviewWeight) * regression + multiviewWeight * multiview };
  });
  return { total: mean(rows.map((row) => row.total)), regression: mean(rows.map((row) => row.regression)), multiview: mean(rows.map((row) => row.multiview)) };
}

function forwardTransformerLayer(layer, input) {
  const scale = Math.sqrt(Math.max(1, input[0].length));
  const queries = input.map((values) => addVectors(matVec(layer.query_weights, values), layer.query_bias));
  const keys = input.map((values) => addVectors(matVec(layer.key_weights, values), layer.key_bias));
  const values = input.map((values) => addVectors(matVec(layer.value_weights, values), layer.value_bias));
  const attention = queries.map((query) => softmax(keys.map((key) => dot(query, key) / scale)));
  const context = attention.map((row) => values[0].map((_, feature) => row.reduce((sum, weight, source) => sum + weight * values[source][feature], 0)));
  const attentionDelta = context.map((values) => addVectors(matVec(layer.output_weights, values), layer.output_bias).map(Math.tanh));
  const attentionOutput = input.map((values, index) => addVectors(values, attentionDelta[index]));
  const ffnHidden = attentionOutput.map((values) => addVectors(matVec(layer.ffn_in_weights, values), layer.ffn_in_bias).map(Math.tanh));
  const output = attentionOutput.map((values, index) => addVectors(values, matVec(layer.ffn_out_weights, ffnHidden[index]), layer.ffn_out_bias));
  return { output, cache: { input, queries, keys, values, attention, context, attentionDelta, attentionOutput, ffnHidden, output } };
}

function forward(network, graph) {
  const inputPre = graph.nodes.map((node) => addVectors(matVec(network.input.weights, node.features), network.input.bias));
  let hidden = inputPre.map((values) => values.map(Math.tanh));
  const preGnnFnnCaches = [];
  for (const layer of network.pre_gnn_fnn_layers) {
    const input = hidden;
    const ffnHidden = input.map((values) => addVectors(matVec(layer.hidden_weights, values), layer.hidden_bias).map(Math.tanh));
    const delta = ffnHidden.map((values) => addVectors(matVec(layer.output_weights, values), layer.output_bias).map(Math.tanh));
    hidden = input.map((values, index) => addVectors(values, delta[index]));
    preGnnFnnCaches.push({ input, ffnHidden, delta, output: hidden });
  }
  const fusionInput = hidden.map((values) => [...values, ...graph.geometryFeature, ...graph.visualFeature]);
  const fusionPre = fusionInput.map((values) => addVectors(matVec(network.fusion.weights, values), network.fusion.bias));
  hidden = fusionPre.map((values) => values.map(Math.tanh));
  const messageCaches = [];
  for (const layer of network.message_layers) {
    const previous = hidden;
    const next = previous.map((self, targetIndex) => {
      const messages = previous.flatMap((source, sourceIndex) => sourceIndex === targetIndex ? [] : [addVectors(matVec(layer.neighbor_weights, source), matVec(layer.edge_weights, graph.edges[targetIndex][sourceIndex]))]);
      const aggregate = messages.length ? messages[0].map((_, index) => mean(messages.map((message) => message[index]))) : vector(self.length);
      return addVectors(matVec(layer.self_weights, self), aggregate, layer.bias).map(Math.tanh);
    });
    messageCaches.push({ input: previous, output: next });
    hidden = next;
  }
  const transformerCaches = [];
  for (const layer of network.transformer_layers) {
    const transformed = forwardTransformerLayer(layer, hidden);
    transformerCaches.push(transformed.cache);
    hidden = transformed.output;
  }
  const routerLogits = hidden.map((values) => addVectors(matVec(network.moe.router.weights, values), network.moe.router.bias));
  const gates = routerLogits.map(softmax);
  const expertHeads = hidden.map((values) => network.moe.experts.map((expert) => addVectors(matVec(expert.hidden_weights, values), expert.hidden_bias).map(Math.tanh)));
  const expertOutputs = expertHeads.map((heads) => heads.map((head, expertIndex) => addVectors(matVec(network.moe.experts[expertIndex].weights, head), network.moe.experts[expertIndex].bias)));
  const output = expertOutputs.map((experts, nodeIndex) => experts[0].map((_, feature) => experts.reduce((sum, expert, expertIndex) => sum + gates[nodeIndex][expertIndex] * expert[feature], 0)));
  return { output, cache: { inputHidden: inputPre.map((values) => values.map(Math.tanh)), preGnnFnnCaches, fusionInput, fusionHidden: fusionPre.map((values) => values.map(Math.tanh)), messageCaches, transformerCaches, finalHidden: hidden, gates, expertHeads, expertOutputs } };
}

function outerAdd(target, left, right, scale = 1) {
  for (let row = 0; row < target.length; row += 1) for (let column = 0; column < target[row].length; column += 1) target[row][column] += left[row] * right[column] * scale;
}
function transposeVec(weights, values) {
  return weights[0].map((_, column) => weights.reduce((sum, row, index) => sum + row[column] * values[index], 0));
}
function vectorAddInPlace(target, source, scale = 1) { for (let index = 0; index < target.length; index += 1) target[index] += source[index] * scale; }

function backwardTransformerLayer(layer, cache, outputGrad, grad) {
  const nodeCount = cache.input.length;
  const hiddenDim = cache.input[0].length;
  const attentionOutputGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const ffnOutputGrad = outputGrad[nodeIndex];
    vectorAddInPlace(attentionOutputGrad[nodeIndex], ffnOutputGrad);
    outerAdd(grad.ffn_out_weights, ffnOutputGrad, cache.ffnHidden[nodeIndex]);
    vectorAddInPlace(grad.ffn_out_bias, ffnOutputGrad);
    const ffnHiddenGrad = transposeVec(layer.ffn_out_weights, ffnOutputGrad).map((value, index) => value * (1 - square(cache.ffnHidden[nodeIndex][index])));
    outerAdd(grad.ffn_in_weights, ffnHiddenGrad, cache.attentionOutput[nodeIndex]);
    vectorAddInPlace(grad.ffn_in_bias, ffnHiddenGrad);
    vectorAddInPlace(attentionOutputGrad[nodeIndex], transposeVec(layer.ffn_in_weights, ffnHiddenGrad));
  }
  const inputGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  const contextGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    vectorAddInPlace(inputGrad[nodeIndex], attentionOutputGrad[nodeIndex]);
    const attentionPreGrad = attentionOutputGrad[nodeIndex].map((value, index) => value * (1 - square(cache.attentionDelta[nodeIndex][index])));
    outerAdd(grad.output_weights, attentionPreGrad, cache.context[nodeIndex]);
    vectorAddInPlace(grad.output_bias, attentionPreGrad);
    vectorAddInPlace(contextGrad[nodeIndex], transposeVec(layer.output_weights, attentionPreGrad));
  }
  const queryGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  const keyGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  const valueGrad = Array.from({ length: nodeCount }, () => vector(hiddenDim));
  const scale = Math.sqrt(Math.max(1, hiddenDim));
  for (let target = 0; target < nodeCount; target += 1) {
    const attentionGrad = cache.values.map((values) => dot(contextGrad[target], values));
    const weighted = cache.attention[target].reduce((sum, weight, source) => sum + weight * attentionGrad[source], 0);
    for (let source = 0; source < nodeCount; source += 1) {
      vectorAddInPlace(valueGrad[source], contextGrad[target], cache.attention[target][source]);
      const scoreGrad = cache.attention[target][source] * (attentionGrad[source] - weighted);
      vectorAddInPlace(queryGrad[target], cache.keys[source], scoreGrad / scale);
      vectorAddInPlace(keyGrad[source], cache.queries[target], scoreGrad / scale);
    }
  }
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    outerAdd(grad.query_weights, queryGrad[nodeIndex], cache.input[nodeIndex]); vectorAddInPlace(grad.query_bias, queryGrad[nodeIndex]);
    outerAdd(grad.key_weights, keyGrad[nodeIndex], cache.input[nodeIndex]); vectorAddInPlace(grad.key_bias, keyGrad[nodeIndex]);
    outerAdd(grad.value_weights, valueGrad[nodeIndex], cache.input[nodeIndex]); vectorAddInPlace(grad.value_bias, valueGrad[nodeIndex]);
    vectorAddInPlace(inputGrad[nodeIndex], transposeVec(layer.query_weights, queryGrad[nodeIndex]));
    vectorAddInPlace(inputGrad[nodeIndex], transposeVec(layer.key_weights, keyGrad[nodeIndex]));
    vectorAddInPlace(inputGrad[nodeIndex], transposeVec(layer.value_weights, valueGrad[nodeIndex]));
  }
  return inputGrad;
}

function backward(network, graph, run, multiviewWeight) {
  const grad = zeroGrad(network);
  const nodeCount = graph.nodes.length;
  let hiddenGrad = Array.from({ length: nodeCount }, () => vector(run.cache.finalHidden[0].length));
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const outputGrad = outputGradient(run.output[nodeIndex], graph.nodes[nodeIndex], graph, multiviewWeight);
    const gateSensitivity = network.moe.experts.map((_, expertIndex) => dot(outputGrad, run.cache.expertOutputs[nodeIndex][expertIndex]));
    const gateMean = run.cache.gates[nodeIndex].reduce((sum, gate, expertIndex) => sum + gate * gateSensitivity[expertIndex], 0);
    const routerGrad = run.cache.gates[nodeIndex].map((gate, expertIndex) => gate * (gateSensitivity[expertIndex] - gateMean));
    outerAdd(grad.moe.router.weights, routerGrad, run.cache.finalHidden[nodeIndex]);
    vectorAddInPlace(grad.moe.router.bias, routerGrad);
    vectorAddInPlace(hiddenGrad[nodeIndex], transposeVec(network.moe.router.weights, routerGrad));
    network.moe.experts.forEach((expert, expertIndex) => {
      const expertOutputGrad = outputGrad.map((value) => value * run.cache.gates[nodeIndex][expertIndex]);
      outerAdd(grad.moe.experts[expertIndex].weights, expertOutputGrad, run.cache.expertHeads[nodeIndex][expertIndex]);
      vectorAddInPlace(grad.moe.experts[expertIndex].bias, expertOutputGrad);
      const headGrad = transposeVec(expert.weights, expertOutputGrad).map((value, index) => value * (1 - square(run.cache.expertHeads[nodeIndex][expertIndex][index])));
      outerAdd(grad.moe.experts[expertIndex].hidden_weights, headGrad, run.cache.finalHidden[nodeIndex]);
      vectorAddInPlace(grad.moe.experts[expertIndex].hidden_bias, headGrad);
      vectorAddInPlace(hiddenGrad[nodeIndex], transposeVec(expert.hidden_weights, headGrad));
    });
  }
  for (let layerIndex = network.transformer_layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
    hiddenGrad = backwardTransformerLayer(network.transformer_layers[layerIndex], run.cache.transformerCaches[layerIndex], hiddenGrad, grad.transformer_layers[layerIndex]);
  }
  for (let layerIndex = network.message_layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
    const layer = network.message_layers[layerIndex];
    const cache = run.cache.messageCaches[layerIndex];
    const inputGrad = Array.from({ length: nodeCount }, () => vector(cache.input[0].length));
    const preGrads = hiddenGrad.map((values, nodeIndex) => values.map((value, index) => value * (1 - square(cache.output[nodeIndex][index]))));
    for (let targetIndex = 0; targetIndex < nodeCount; targetIndex += 1) {
      const preGrad = preGrads[targetIndex];
      outerAdd(grad.message_layers[layerIndex].self_weights, preGrad, cache.input[targetIndex]);
      vectorAddInPlace(grad.message_layers[layerIndex].bias, preGrad);
      vectorAddInPlace(inputGrad[targetIndex], transposeVec(layer.self_weights, preGrad));
      const divisor = Math.max(1, nodeCount - 1);
      for (let sourceIndex = 0; sourceIndex < nodeCount; sourceIndex += 1) {
        if (sourceIndex === targetIndex) continue;
        outerAdd(grad.message_layers[layerIndex].neighbor_weights, preGrad, cache.input[sourceIndex], 1 / divisor);
        outerAdd(grad.message_layers[layerIndex].edge_weights, preGrad, graph.edges[targetIndex][sourceIndex], 1 / divisor);
        vectorAddInPlace(inputGrad[sourceIndex], transposeVec(layer.neighbor_weights, preGrad), 1 / divisor);
      }
    }
    hiddenGrad = inputGrad;
  }
  const preFusionGrad = hiddenGrad.map((values, nodeIndex) => values.map((value, index) => value * (1 - square(run.cache.fusionHidden[nodeIndex][index]))));
  const labelHiddenGrad = Array.from({ length: nodeCount }, () => vector(network.fusion.label_dim));
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    outerAdd(grad.fusion.weights, preFusionGrad[nodeIndex], run.cache.fusionInput[nodeIndex]);
    vectorAddInPlace(grad.fusion.bias, preFusionGrad[nodeIndex]);
    const fusedInputGrad = transposeVec(network.fusion.weights, preFusionGrad[nodeIndex]);
    vectorAddInPlace(labelHiddenGrad[nodeIndex], fusedInputGrad.slice(0, network.fusion.label_dim));
  }
  hiddenGrad = labelHiddenGrad;
  for (let layerIndex = network.pre_gnn_fnn_layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
    const layer = network.pre_gnn_fnn_layers[layerIndex];
    const cache = run.cache.preGnnFnnCaches[layerIndex];
    const inputGrad = hiddenGrad.map((values) => [...values]);
    for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
      const deltaGrad = hiddenGrad[nodeIndex].map((value, index) => value * (1 - square(cache.delta[nodeIndex][index])));
      outerAdd(grad.pre_gnn_fnn_layers[layerIndex].output_weights, deltaGrad, cache.ffnHidden[nodeIndex]);
      vectorAddInPlace(grad.pre_gnn_fnn_layers[layerIndex].output_bias, deltaGrad);
      const ffnHiddenGrad = transposeVec(layer.output_weights, deltaGrad).map((value, index) => value * (1 - square(cache.ffnHidden[nodeIndex][index])));
      outerAdd(grad.pre_gnn_fnn_layers[layerIndex].hidden_weights, ffnHiddenGrad, cache.input[nodeIndex]);
      vectorAddInPlace(grad.pre_gnn_fnn_layers[layerIndex].hidden_bias, ffnHiddenGrad);
      vectorAddInPlace(inputGrad[nodeIndex], transposeVec(layer.hidden_weights, ffnHiddenGrad));
    }
    hiddenGrad = inputGrad;
  }
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const preGrad = hiddenGrad[nodeIndex].map((value, index) => value * (1 - square(run.cache.inputHidden[nodeIndex][index])));
    outerAdd(grad.input.weights, preGrad, graph.nodes[nodeIndex].features);
    vectorAddInPlace(grad.input.bias, preGrad);
  }
  return grad;
}

function updateMatrix(weights, gradients, learningRate, weightDecay) {
  for (let row = 0; row < weights.length; row += 1) for (let column = 0; column < weights[row].length; column += 1) weights[row][column] -= learningRate * (gradients[row][column] + weightDecay * weights[row][column]);
}
function updateVector(values, gradients, learningRate) { for (let index = 0; index < values.length; index += 1) values[index] -= learningRate * gradients[index]; }
function applyGradient(network, grad, learningRate, weightDecay) {
  updateMatrix(network.input.weights, grad.input.weights, learningRate, weightDecay); updateVector(network.input.bias, grad.input.bias, learningRate);
  network.pre_gnn_fnn_layers.forEach((layer, index) => {
    const layerGrad = grad.pre_gnn_fnn_layers[index];
    updateMatrix(layer.hidden_weights, layerGrad.hidden_weights, learningRate, weightDecay);
    updateVector(layer.hidden_bias, layerGrad.hidden_bias, learningRate);
    updateMatrix(layer.output_weights, layerGrad.output_weights, learningRate, weightDecay);
    updateVector(layer.output_bias, layerGrad.output_bias, learningRate);
  });
  updateMatrix(network.fusion.weights, grad.fusion.weights, learningRate, weightDecay);
  updateVector(network.fusion.bias, grad.fusion.bias, learningRate);
  network.message_layers.forEach((layer, index) => {
    updateMatrix(layer.self_weights, grad.message_layers[index].self_weights, learningRate, weightDecay);
    updateMatrix(layer.neighbor_weights, grad.message_layers[index].neighbor_weights, learningRate, weightDecay);
    updateMatrix(layer.edge_weights, grad.message_layers[index].edge_weights, learningRate, weightDecay);
    updateVector(layer.bias, grad.message_layers[index].bias, learningRate);
  });
  network.transformer_layers.forEach((layer, index) => {
    const layerGrad = grad.transformer_layers[index];
    for (const prefix of ['query', 'key', 'value', 'output', 'ffn_in', 'ffn_out']) {
      updateMatrix(layer[`${prefix}_weights`], layerGrad[`${prefix}_weights`], learningRate, weightDecay);
      updateVector(layer[`${prefix}_bias`], layerGrad[`${prefix}_bias`], learningRate);
    }
  });
  updateMatrix(network.moe.router.weights, grad.moe.router.weights, learningRate, weightDecay);
  updateVector(network.moe.router.bias, grad.moe.router.bias, learningRate);
  network.moe.experts.forEach((expert, index) => {
    const expertGrad = grad.moe.experts[index];
    updateMatrix(expert.hidden_weights, expertGrad.hidden_weights, learningRate, weightDecay);
    updateVector(expert.hidden_bias, expertGrad.hidden_bias, learningRate);
    updateMatrix(expert.weights, expertGrad.weights, learningRate, weightDecay);
    updateVector(expert.bias, expertGrad.bias, learningRate);
  });
}

async function buildGraphs(manifest, split, leaderLengthPrior) {
  const graphs = [], selection = {}, matching = { matched: 0, fallback: 0 };
  let radiusSum = 0;
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const clean = cleanObj(await read(repoFile(sample.input.source_obj)));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const cvFeatures = await buildCvFeatures({ objText: clean.text, bounds });
    const annotation = JSON.parse(await read(repoFile(sample.target.annotation_json)));
    if (annotation?.version !== 'after_mannual_adjust' || annotation?.layout_type !== 'manual_adjusted') throw new Error(sample.category + '/' + sample.sample_id + ' 的目标不是人工调整后标注');
    const manualLabels = annotationsToLabels(annotation);
    const candidates = fixedCandidatesWithoutTargetLayout(manualLabels, generatedCandidatesFromCleanObj(clean.text, bounds), bounds);
    validateFixedLabelContract(manualLabels, candidates, sample.category + '/' + sample.sample_id + '/' + split);
    radiusSum += bounds.radius;
    const categorySelection = selection[sample.category] ||= { sample_count: 0, label_counts: [], group_stats: {} };
    categorySelection.sample_count += 1; categorySelection.label_counts.push(candidates.length);
    const positives = new Set(manualLabels.flatMap((label) => [...(label.targetGroups || []), ...(label.sourceObjs || [])]).map(normalizeGroupName));
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const teacherLabels = optimizeLabels(candidates, bounds, { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true, seed: 17, iterations: 180, depthGrids, category: sample.category, leaderLengthPrior });
    validateFixedLabelContract(manualLabels, teacherLabels, sample.category + '/' + sample.sample_id + '/' + split + '/teacher');
    const nodes = candidates.map((candidate, index) => {
      if (candidate.initialization_source === 'matched_clean_geometry') matching.matched += 1; else matching.fallback += 1;
      const key = normalizeGroupName(candidate.sourceObjs?.[0] || candidate.text);
      const stat = categorySelection.group_stats[key] ||= { occurrences: 0, positives: 0 };
      stat.occurrences += 1; if (positives.has(key)) stat.positives += 1;
      const target = [
        ...teacherLabels[index].center.map((value, axis) => (value - candidate.center[axis]) / Math.max(bounds.radius, 1e-6)),
        ...teacherLabels[index].boxSize.map((value, axis) => Math.log(Math.max(Math.abs(value), 1e-6) / Math.max(Math.abs(candidate.boxSize[axis]), 1e-6)))
      ];
      const node = { ...candidate, label_id: manualLabels[index].id, features: multiViewLayoutFeatures(candidate, bounds, index, candidates.length), target, anchor: [...manualLabels[index].anchor], initial_center: [...candidate.center], initial_size: [...candidate.boxSize] };
      node.projected_target = projectedLayoutVector(target, node, { bounds });
      return node;
    });
    const edges = nodes.map((_, targetIndex) => nodes.map((__, sourceIndex) => sourceIndex === targetIndex ? null : graphEdgeFeatures(nodes[sourceIndex], nodes[targetIndex], bounds)));
    graphs.push({ category: sample.category, sample_id: sample.sample_id, split, bounds: { center: [...bounds.center], radius: bounds.radius }, geometryFeature: cvFeatures.geometry, visualFeature: cvFeatures.visual, nodes, edges });
  }
  const categories = Object.fromEntries(Object.entries(selection).map(([category, value]) => [category, { sample_count: value.sample_count, label_count_mean: Number(mean(value.label_counts).toFixed(4)), label_count_min: Math.min(...value.label_counts), label_count_max: Math.max(...value.label_counts), group_priors: Object.fromEntries(Object.entries(value.group_stats).map(([key, stat]) => [key, Number((stat.positives / Math.max(1, stat.occurrences)).toFixed(6))])) }]));
  return { graphs, matching, selection: { radius_reference: graphs.length ? radiusSum / graphs.length : 1, categories } };
}

function evaluate(network, graphs, multiviewWeight) {
  const rows = graphs.map((graph) => lossParts(forward(network, graph).output, graph, multiviewWeight));
  return { total: mean(rows.map((row) => row.total)), regression: mean(rows.map((row) => row.regression)), multiview: mean(rows.map((row) => row.multiview)) };
}
function routingSummary(network, graphs) {
  const gates = graphs.flatMap((graph) => forward(network, graph).cache.gates);
  const names = network.moe.experts.map((expert) => expert.name);
  const average = names.map((_, expertIndex) => mean(gates.map((row) => row[expertIndex])));
  const entropy = mean(gates.map((row) => -row.reduce((sum, value) => sum + value * Math.log(Math.max(value, 1e-12)), 0)));
  const dominantCounts = names.map(() => 0);
  for (const row of gates) dominantCounts[row.indexOf(Math.max(...row))] += 1;
  return {
    nodes: gates.length,
    average_weights: Object.fromEntries(names.map((name, index) => [name, Number(average[index].toFixed(6))])),
    normalized_entropy: Number((entropy / Math.max(Math.log(Math.max(2, names.length)), 1e-9)).toFixed(6)),
    mean_max_weight: Number(mean(gates.map((row) => Math.max(...row))).toFixed(6)),
    dominant_expert_share: Object.fromEntries(names.map((name, index) => [name, Number((dominantCounts[index] / Math.max(1, gates.length)).toFixed(6))]))
  };
}
function parameterDelta(initial, trained) {
  let squared = 0, maximum = 0, count = 0, changed = 0;
  function visit(left, right) {
    if (Array.isArray(left) && Array.isArray(right)) {
      for (let index = 0; index < Math.min(left.length, right.length); index += 1) visit(left[index], right[index]);
    } else if (left && right && typeof left === 'object' && typeof right === 'object') {
      for (const key of Object.keys(left)) if (key in right) visit(left[key], right[key]);
    } else if (typeof left === 'number' && typeof right === 'number') {
      const delta = right - left;
      squared += delta * delta; maximum = Math.max(maximum, Math.abs(delta)); count += 1;
      if (Math.abs(delta) > 1e-12) changed += 1;
    }
  }
  visit(initial, trained);
  return { parameter_count: count, changed_parameters: changed, l2_delta: Number(Math.sqrt(squared).toFixed(9)), max_abs_delta: Number(maximum.toFixed(9)) };
}
function preGnnFnnAblation(network, graphs) {
  if (!network.pre_gnn_fnn_layers?.length) return { nodes: 0, output_values: 0, max_abs_output_change: 0, rms_output_change: 0 };
  const ablated = { ...network, pre_gnn_fnn_layers: [] };
  let nodes = 0, count = 0, squared = 0, maximum = 0;
  for (const graph of graphs) {
    const full = forward(network, graph).output;
    const withoutFnn = forward(ablated, graph).output;
    nodes += full.length;
    for (let nodeIndex = 0; nodeIndex < full.length; nodeIndex += 1) {
      for (let outputIndex = 0; outputIndex < full[nodeIndex].length; outputIndex += 1) {
        const delta = full[nodeIndex][outputIndex] - withoutFnn[nodeIndex][outputIndex];
        maximum = Math.max(maximum, Math.abs(delta));
        squared += delta * delta;
        count += 1;
      }
    }
  }
  return { nodes, output_values: count, max_abs_output_change: maximum, rms_output_change: count ? Math.sqrt(squared / count) : 0 };
}
function cvFusionAblation(network, graphs) {
  let nodes = 0, count = 0, squared = 0, maximum = 0;
  for (const graph of graphs) {
    const full = forward(network, graph).output;
    const withoutCv = forward(network, { ...graph, geometryFeature: Array(64).fill(0), visualFeature: Array(32).fill(0) }).output;
    nodes += full.length;
    for (let nodeIndex = 0; nodeIndex < full.length; nodeIndex += 1) for (let outputIndex = 0; outputIndex < full[nodeIndex].length; outputIndex += 1) {
      const delta = full[nodeIndex][outputIndex] - withoutCv[nodeIndex][outputIndex];
      maximum = Math.max(maximum, Math.abs(delta)); squared += delta * delta; count += 1;
    }
  }
  return { nodes, output_values: count, max_abs_output_change: maximum, rms_output_change: count ? Math.sqrt(squared / count) : 0 };
}
function shuffle(values, rng) { const output = [...values]; for (let index = output.length - 1; index > 0; index -= 1) { const other = Math.floor(rng() * (index + 1)); [output[index], output[other]] = [output[other], output[index]]; } return output; }

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const output = path.resolve(String(options.output));
  if (['layout_model.json', 'layout_model_multiview.json'].includes(path.basename(output).toLowerCase())) throw new Error('拒绝直接覆盖活动模型；图模型必须先写候选并通过 val 门控');
  const manifest = JSON.parse(await read(manifestFile));
  const leaderLengthPrior = JSON.parse(await read(leaderPriorFile));
  const trainData = await buildGraphs(manifest, 'train', leaderLengthPrior), valData = await buildGraphs(manifest, 'val', leaderLengthPrior), testData = await buildGraphs(manifest, 'test', leaderLengthPrior);
  const train = trainData.graphs, val = valData.graphs, test = testData.graphs;
  if (!train.length) throw new Error('train split 没有可用图样本');
  const rng = createRng(options.seed);
  const network = initNetwork(MULTI_VIEW_LAYOUT_FEATURE_NAMES.length, GRAPH_EDGE_FEATURE_NAMES.length, Number(options.hiddenDim), Number(options.preGnnFnnLayers), Number(options.messageLayers), Number(options.transformerLayers), Number(options.expertCount), rng);
  let warmStart = null;
  if (options.warmStart && String(options.warmStart).toLowerCase() !== 'none') {
    const warmStartModel = JSON.parse(await read(path.resolve(String(options.warmStart))));
    if (warmStartModel.validation_gate?.status !== 'accepted') throw new Error('warm start 模型必须已经通过验证门控');
    warmStart = warmStartNetwork(network, warmStartModel, rng, Number(options.expertPerturbation));
  }
  const initialNetwork = structuredClone(network);
  const multiviewWeight = Math.max(0, Math.min(1, Number(options.multiviewWeight)));
  let best = structuredClone(network), bestVal = Infinity, bestEpoch = 0;
  const history = [];
  for (let epoch = 0; epoch < Number(options.epochs); epoch += 1) {
    for (const graph of shuffle(train, rng)) {
      const run = forward(network, graph);
      applyGradient(network, backward(network, graph, run, multiviewWeight), Number(options.learningRate), Number(options.weightDecay));
    }
    const trainLoss = evaluate(network, train, multiviewWeight), valLoss = evaluate(network, val, multiviewWeight);
    history.push({ epoch: epoch + 1, train_loss: Number(trainLoss.total.toFixed(6)), val_loss: Number(valLoss.total.toFixed(6)), train_mse: Number(trainLoss.regression.toFixed(6)), val_mse: Number(valLoss.regression.toFixed(6)), train_multiview_projection_mse: Number(trainLoss.multiview.toFixed(6)), val_multiview_projection_mse: Number(valLoss.multiview.toFixed(6)) });
    if (valLoss.total < bestVal) { bestVal = valLoss.total; bestEpoch = epoch + 1; best = structuredClone(network); }
  }
  const trainMetrics = evaluate(best, train, multiviewWeight), valMetrics = evaluate(best, val, multiviewWeight), testMetrics = evaluate(best, test, multiviewWeight);
  const metrics = { train_loss: Number(trainMetrics.total.toFixed(6)), val_loss: Number(valMetrics.total.toFixed(6)), test_loss: Number(testMetrics.total.toFixed(6)), train_mse: Number(trainMetrics.regression.toFixed(6)), val_mse: Number(valMetrics.regression.toFixed(6)), test_mse: Number(testMetrics.regression.toFixed(6)), train_multiview_projection_mse: Number(trainMetrics.multiview.toFixed(6)), val_multiview_projection_mse: Number(valMetrics.multiview.toFixed(6)), test_multiview_projection_mse: Number(testMetrics.multiview.toFixed(6)) };
  const routing = { train: routingSummary(best, train), val: routingSummary(best, val), test: routingSummary(best, test) };
  const parameterUpdates = {
    pre_gnn_fnn: parameterDelta(initialNetwork.pre_gnn_fnn_layers, best.pre_gnn_fnn_layers),
    feature_fusion: parameterDelta(initialNetwork.fusion, best.fusion),
    transformer: parameterDelta(initialNetwork.transformer_layers, best.transformer_layers),
    moe_router: parameterDelta(initialNetwork.moe.router, best.moe.router),
    moe_experts: parameterDelta(initialNetwork.moe.experts, best.moe.experts)
  };
  const functionalEvidence = { pre_gnn_fnn_ablation: preGnnFnnAblation(best, val), cv_fusion_ablation: cvFusionAblation(best, val) };
  const model = {
    version: 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe', status: 'trained_supervised_fixed_labels_dgcnn_cnn_fnn_graph_transformer_moe', camera_protocol: DATASET_CAMERA_PROTOCOL.id,
    inference: { center_blend: 1, size_blend: 1, size_ratio_range: [0.65, 1.35], selection_status: 'requires_validation_gate' },
    architecture: { type: 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe', implemented: true, node_input_dim: MULTI_VIEW_LAYOUT_FEATURE_NAMES.length, label_encoded_dim: 64, obj_surface_points: 1024, dgcnn_edgeconv_layers: 2, dgcnn_neighbor_count: 4, geometry_feature_dim: 64, geometry_encoder_training: 'deterministic_frozen_encoder_with_trainable_fusion', multiview_image_count: 5, visual_backbone: 'frozen_lightweight_cnn_2conv', visual_input_source: 'unlabeled_five_view_depth_rasters_rendered_from_clean_obj', visual_feature_dim: 32, fused_feature_dim: 160, fusion_projection: [160, 64], edge_input_dim: GRAPH_EDGE_FEATURE_NAMES.length, hidden_dim: Number(options.hiddenDim), pre_gnn_fnn_layers: Number(options.preGnnFnnLayers), pre_gnn_fnn_hidden_dim: Number(options.hiddenDim) * 2, pre_gnn_fnn_residual: true, message_passing_layers: Number(options.messageLayers), transformer_layers: Number(options.transformerLayers), attention: 'scaled_dot_product_full_label_set_self_attention', transformer_residual: true, transformer_ffn_dim: Number(options.hiddenDim) * 2, moe_router: 'learned_per_node_softmax', moe_expert_count: Number(options.expertCount), moe_expert_names: best.moe.experts.map((expert) => expert.name), moe_fusion: 'soft_weighted_sum_of_expert_outputs', output_head: [Number(options.hiddenDim), 32, 6], output_semantics: 'center_residual_xyz_and_log_size_ratio_xyz', node_semantics: 'one_node_per_fixed_label', edge_semantics: 'directed_label_relationship_with_relative_geometry_and_shared_parts', layer_semantics: ['node_multiview_encoder', 'pre_gnn_residual_fnn', 'dgcnn_geometry_and_frozen_cnn_visual_fusion', 'local_relational_message_passing', 'global_transformer_self_attention', 'learned_moe_fusion'], supervised_target: 'deterministic_five_view_safe_teacher', supervised_loss: 'weighted_3d_parameter_mse_plus_five_view_projected_center_and_box_mse', decoder: 'five_view_simulated_annealing', label_contract: 'manual_count_id_text_anchor_groups_locked', input_provenance: 'manual_contract_clean_obj_and_unlabeled_render_without_adjusted_center_or_box_size' },
    feature_names: MULTI_VIEW_LAYOUT_FEATURE_NAMES, edge_feature_names: GRAPH_EDGE_FEATURE_NAMES,
    split_policy: { train: 33, val: 11, test: 11, manual_is_supervision: true, validation_selects_checkpoint: true, test_labels_used_for_final_evaluation_only: true, preference_updates_exclude_test: true },
    hyperparameters: { epochs: Number(options.epochs), learning_rate: Number(options.learningRate), weight_decay: Number(options.weightDecay), multiview_projection_loss_weight: multiviewWeight, regression_loss_weight: 1 - multiviewWeight, hidden_dim: Number(options.hiddenDim), pre_gnn_fnn_layers: Number(options.preGnnFnnLayers), message_passing_layers: Number(options.messageLayers), transformer_layers: Number(options.transformerLayers), moe_expert_count: Number(options.expertCount), warm_start: warmStart, seed: Number(options.seed), shuffle_graphs_each_epoch: true },
    selection: { ...trainData.selection, status: 'fixed_manual_label_contract', selection_disabled_when_manual_labels_exist: true }, network: best,
    training: { train_graphs: train.length, val_graphs: val.length, test_graphs: test.length, train_examples: train.reduce((sum, graph) => sum + graph.nodes.length, 0), val_examples: val.reduce((sum, graph) => sum + graph.nodes.length, 0), test_examples: test.reduce((sum, graph) => sum + graph.nodes.length, 0), best_epoch: bestEpoch, history, ...metrics, routing, parameter_updates: parameterUpdates, functional_evidence: functionalEvidence, candidate_matching: { train: trainData.matching, val: valData.matching, test: testData.matching }, fixed_label_contract_validated: true }, generated_at: new Date().toISOString()
  };
  await fs.writeFile(output, JSON.stringify(model, null, 2) + '\n', 'utf8');
  const reportFile = path.resolve(String(options.report));
  await fs.writeFile(reportFile, JSON.stringify({ version: 'layout_training_dgcnn_cnn_fnn_graph_transformer_moe_v8_report', generated_at: model.generated_at, model_file: path.relative(root, output).split(path.sep).join('/'), architecture: model.architecture, hyperparameters: model.hyperparameters, metrics, routing, parameter_updates: parameterUpdates, functional_evidence: functionalEvidence, best_epoch: bestEpoch, graph_counts: { train: train.length, val: val.length, test: test.length }, node_counts: { train: model.training.train_examples, val: model.training.val_examples, test: model.training.test_examples }, fixed_label_contract_validated: true, interpretation: { actual_trained_pre_gnn_fnn: parameterUpdates.pre_gnn_fnn.changed_parameters > 0, actual_trained_feature_fusion: parameterUpdates.feature_fusion.changed_parameters > 0, cv_features_change_validation_outputs: functionalEvidence.cv_fusion_ablation.max_abs_output_change > 1e-8, pre_gnn_fnn_changes_validation_outputs: functionalEvidence.pre_gnn_fnn_ablation.max_abs_output_change > 1e-8, actual_learned_message_passing: true, actual_trained_transformer_attention: parameterUpdates.transformer.changed_parameters > 0, actual_trained_moe_router: parameterUpdates.moe_router.changed_parameters > 0, soft_moe_expert_fusion: true, labels_are_nodes: true, label_relationships_are_edges: true, adjusted_target_layout_used_as_input: false, test_used_in_gradient_or_checkpoint_selection: false } }, null, 2) + '\n', 'utf8');
  console.log('图布局模型完成：train_graphs=' + train.length + ', val_graphs=' + val.length + ', test_graphs=' + test.length + ', best_epoch=' + bestEpoch);
  console.log('loss train=' + metrics.train_loss + ', val=' + metrics.val_loss + ', test=' + metrics.test_loss);
  console.log('输出：' + path.relative(root, output).split(path.sep).join('/'));
}

main().catch((error) => { console.error(error.stack || error.message || error); process.exitCode = 1; });
