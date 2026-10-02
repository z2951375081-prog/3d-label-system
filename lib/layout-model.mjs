// Trainable layout-style model contract used by the local Node pipeline.
// v1 uses 16 geometry features; v2 adds five-view projections (51 total).

import { MULTI_VIEW_NAMES, projectPointToView, projectLabelToView } from './layout-optimizer.mjs';
import { DATASET_CAMERA_PROTOCOL } from '../public/dataset-camera.js';
import { PURE_3D_LABEL_FEATURE_NAMES, pure3DLabelFeatures } from './spatial-style-features.mjs';
import { HETEROGENEOUS_NODE_FEATURE_NAMES, buildHeterogeneousLayoutGraph, decodeLocalLayoutOutput } from './heterogeneous-layout-graph.mjs';

export const LAYOUT_FEATURE_NAMES = [
  'anchor_x', 'anchor_y', 'anchor_z', 'center_x', 'center_y', 'center_z',
  'box_x', 'box_y', 'box_z', 'source_count', 'target_count', 'distance',
  'shape_x', 'shape_y', 'shape_z', 'candidate_index'
];

export const MULTI_VIEW_LAYOUT_FEATURE_NAMES = [
  ...LAYOUT_FEATURE_NAMES,
  ...MULTI_VIEW_NAMES.flatMap((view) => [
    `${view}_anchor_x`, `${view}_anchor_y`, `${view}_center_x`, `${view}_center_y`,
    `${view}_center_depth`, `${view}_box_width`, `${view}_box_height`
  ])
];

export { PURE_3D_LABEL_FEATURE_NAMES };
export { HETEROGENEOUS_NODE_FEATURE_NAMES };

export const GRAPH_EDGE_FEATURE_NAMES = [
  'anchor_dx', 'anchor_dy', 'anchor_dz',
  'center_dx', 'center_dy', 'center_dz',
  'anchor_distance', 'center_distance',
  'same_source_part', 'same_target_group',
  'box_log_ratio_x', 'box_log_ratio_y', 'box_log_ratio_z'
];

export function normalizeGroupName(value) { return String(value || '').toLowerCase().replace(/\.obj$/i, '').replace(/(?:^|[_-])new[_-]?\d+(?=$|[_-])/g, '_').replace(/[_-]\d+(?=$|[_-])/g, '_').replace(/[^a-z0-9]+/g, ' ').replace(/\s+/g, ' ').trim(); }

export function groupSelectionScore(candidate, category, model) {
  const key = normalizeGroupName(candidate.sourceObjs?.[0] || candidate.targetGroups?.[0] || candidate.text);
  const categoryModel = model?.selection?.categories?.[category] || {};
  const prior = Number(categoryModel.group_priors?.[key] || 0);
  const geometry = Math.min(1, Number(candidate.sourceObjs?.length || 0) / 3);
  const centerDistance = Math.min(1, Math.hypot(...candidate.anchor) / Math.max(1, model?.selection?.radius_reference || 1));
  return prior * 0.72 + geometry * 0.16 + (1 - centerDistance) * 0.12;
}

export function selectLayoutCandidates(candidates, category, model) {
  if (!model?.selection?.categories?.[category] || candidates.length <= 1) return { candidates, selection: { status: 'not_loaded', selected_count: candidates.length } };
  const categoryModel = model.selection.categories[category];
  const desired = Math.max(1, Math.round(Number(categoryModel.label_count_mean || candidates.length)));
  const ranked = candidates.map((candidate) => ({ candidate, score: groupSelectionScore(candidate, category, model) })).sort((a, b) => b.score - a.score);
  const selected = ranked.slice(0, Math.min(desired, ranked.length)).map((item) => ({ ...item.candidate, selection_score: Number(item.score.toFixed(5)), selection_source: 'trained_group_prior' }));
  return { candidates: selected, selection: { status: model.status || 'trained', desired_count: desired, input_count: candidates.length, selected_count: selected.length, category } };
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function norm(value, center, radius) { return (value - center) / Math.max(radius, 1e-6); }

export function layoutFeatures(label, bounds, index = 0, total = 1) {
  const radius = Math.max(bounds.radius, 1e-6);
  index = label.candidate_index ?? index;
  total = label.candidate_total ?? total;
  return [
    norm(label.anchor[0], bounds.center[0], radius), norm(label.anchor[1], bounds.center[1], radius), norm(label.anchor[2], bounds.center[2], radius),
    norm(label.center[0], bounds.center[0], radius), norm(label.center[1], bounds.center[1], radius), norm(label.center[2], bounds.center[2], radius),
    label.boxSize[0] / radius, label.boxSize[1] / radius, label.boxSize[2] / radius,
    Math.min(1, (label.sourceObjs?.length || 0) / 4), Math.min(1, (label.targetGroups?.length || 0) / 4),
    Math.min(1, Math.hypot(...label.center) / (radius * 4)),
    bounds.size[0] / radius, bounds.size[1] / radius, bounds.size[2] / radius,
    total > 1 ? index / (total - 1) : 0.5
  ];
}

export function multiViewLayoutFeatures(label, bounds, index = 0, total = 1) {
  const radius = Math.max(bounds.radius, 1e-6);
  const features = layoutFeatures(label, bounds, index, total);
  for (const view of MULTI_VIEW_NAMES) {
    const anchor = projectPointToView(label.anchor, bounds, view);
    const center = projectPointToView(label.center, bounds, view);
    const panel = projectLabelToView(label,bounds,view);
    features.push(anchor.x, anchor.y, center.x, center.y, Math.min(2, center.depth / DATASET_CAMERA_PROTOCOL.cameraDistance), panel.width, panel.height);
  }
  return features;
}

function linearPredict(weights, features) {
  const input = [1, ...features];
  return weights.map((row) => row.reduce((sum, weight, index) => sum + weight * input[index], 0));
}

function activate(value, type) { return type === 'relu' ? Math.max(0, value) : type === 'linear' ? value : Math.tanh(value); }
function forwardNetwork(network, features) {
  let values = [...features];
  for (const layer of network.layers) {
    const next = layer.weights.map((row, rowIndex) => activate(row.reduce((sum, weight, index) => sum + weight * values[index], layer.bias[rowIndex]), layer.activation));
    values = next;
  }
  return values;
}

function matrixVector(weights, values) {
  return weights.map((row) => row.reduce((sum, weight, index) => sum + weight * values[index], 0));
}

function addVectors(...vectors) {
  return vectors[0].map((_, index) => vectors.reduce((sum, vector) => sum + vector[index], 0));
}

function dot(left, right) { return left.reduce((sum, value, index) => sum + value * right[index], 0); }
function softmax(values) {
  const peak = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - peak));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / Math.max(total, 1e-9));
}

function sharedNormalizedName(left = [], right = []) {
  const names = new Set(left.map(normalizeGroupName).filter(Boolean));
  return right.some((value) => names.has(normalizeGroupName(value))) ? 1 : 0;
}

export function graphEdgeFeatures(source, target, bounds) {
  const radius = Math.max(bounds.radius, 1e-6);
  const anchorDelta = target.anchor.map((value, axis) => (value - source.anchor[axis]) / radius);
  const centerDelta = target.center.map((value, axis) => (value - source.center[axis]) / radius);
  return [
    ...anchorDelta,
    ...centerDelta,
    Math.min(4, Math.hypot(...anchorDelta)),
    Math.min(4, Math.hypot(...centerDelta)),
    sharedNormalizedName(source.sourceObjs, target.sourceObjs),
    sharedNormalizedName(source.targetGroups, target.targetGroups),
    ...target.boxSize.map((value, axis) => Math.max(-3, Math.min(3, Math.log(Math.max(Math.abs(value), 1e-6) / Math.max(Math.abs(source.boxSize[axis]), 1e-6)))))
  ];
}

function forwardTransformerLayer(layer, input) {
  const scale = Math.sqrt(Math.max(1, input[0].length));
  const queries = input.map((values) => addVectors(matrixVector(layer.query_weights, values), layer.query_bias));
  const keys = input.map((values) => addVectors(matrixVector(layer.key_weights, values), layer.key_bias));
  const projectedValues = input.map((values) => addVectors(matrixVector(layer.value_weights, values), layer.value_bias));
  const attention = queries.map((query) => softmax(keys.map((key) => dot(query, key) / scale)));
  const context = attention.map((row) => projectedValues[0].map((_, feature) => row.reduce((sum, weight, source) => sum + weight * projectedValues[source][feature], 0)));
  const attentionDelta = context.map((values) => addVectors(matrixVector(layer.output_weights, values), layer.output_bias).map(Math.tanh));
  const attentionOutput = input.map((values, index) => addVectors(values, attentionDelta[index]));
  const ffnHidden = attentionOutput.map((values) => addVectors(matrixVector(layer.ffn_in_weights, values), layer.ffn_in_bias).map(Math.tanh));
  return attentionOutput.map((values, index) => addVectors(values, matrixVector(layer.ffn_out_weights, ffnHidden[index]), layer.ffn_out_bias));
}

export function forwardGraphNetwork(network, labels, bounds, nodeFeatures = null, options = {}) {
  const heterogeneous = Boolean(options.heterogeneousGraph || network.input?.weights?.[0]?.length === HETEROGENEOUS_NODE_FEATURE_NAMES.length && network.message_layers?.[0]?.edge_weights?.[0]?.length >= 28);
  const pure3D = Boolean(options.pure3D || heterogeneous || network.input?.weights?.[0]?.length === PURE_3D_LABEL_FEATURE_NAMES.length);
  const features = nodeFeatures || (pure3D
    ? (heterogeneous ? options.heterogeneousGraph.nodeFeatures : labels.map((label, index) => pure3DLabelFeatures(label, bounds, index, labels.length, options.spatialContext, labels)))
    : labels.map((label, index) => multiViewLayoutFeatures(label, bounds, index, labels.length)));
  let hidden = features.map((values) => matrixVector(network.input.weights, values).map((value, index) => Math.tanh(value + network.input.bias[index])));
  if (Array.isArray(network.pre_gnn_fnn_layers)) {
    for (const layer of network.pre_gnn_fnn_layers) {
      hidden = hidden.map((values) => {
        const ffnHidden = addVectors(matrixVector(layer.hidden_weights, values), layer.hidden_bias).map(Math.tanh);
        const delta = addVectors(matrixVector(layer.output_weights, ffnHidden), layer.output_bias).map(Math.tanh);
        return addVectors(values, delta);
      });
    }
  }
  if (network.fusion) {
    const geometry = options.geometryFeature || Array(network.fusion.geometry_dim || 64).fill(0);
    const visual = network.fusion.visual_dim ? (options.visualFeature || Array(network.fusion.visual_dim).fill(0)) : [];
    hidden = hidden.map((values) => addVectors(matrixVector(network.fusion.weights, [...values, ...geometry, ...visual]), network.fusion.bias).map(Math.tanh));
  }
  for (const layer of network.message_layers) {
    const previous = hidden;
    hidden = previous.map((self, targetIndex) => {
      const relationTyped = Boolean(options.heterogeneousGraph && layer.anchor_edge_weights && layer.relation_edge_weights);
      const messages = previous.flatMap((source, sourceIndex) => {
        if (sourceIndex === targetIndex) return [];
        const relation = relationTyped
          ? options.heterogeneousGraph.relationEdges[targetIndex][sourceIndex]
          : (options.heterogeneousGraph?.edgeFeatures?.[targetIndex]?.[sourceIndex] || graphEdgeFeatures(labels[sourceIndex], labels[targetIndex], bounds));
        const relationWeights = relationTyped ? layer.relation_edge_weights : layer.edge_weights;
        return [addVectors(matrixVector(layer.neighbor_weights, source), matrixVector(relationWeights, relation))];
      });
      const aggregate = messages.length
        ? messages[0].map((_, index) => messages.reduce((sum, message) => sum + message[index], 0) / messages.length)
        : self.map(() => 0);
      const anchorMessages = relationTyped ? options.heterogeneousGraph.anchorEdgeFeatures[targetIndex].map((features) => matrixVector(layer.anchor_edge_weights, features)) : [];
      const anchorContext = anchorMessages.length
        ? anchorMessages[0].map((_, index) => anchorMessages.reduce((sum, message) => sum + message[index], 0) / anchorMessages.length)
        : self.map(() => 0);
      return addVectors(matrixVector(layer.self_weights, self), aggregate, anchorContext, layer.bias).map(Math.tanh);
    });
  }
  if (Array.isArray(network.transformer_layers)) {
    for (const layer of network.transformer_layers) hidden = forwardTransformerLayer(layer, hidden);
  }
  if (network.moe?.router && Array.isArray(network.moe?.experts) && network.moe.experts.length) {
    const gates = hidden.map((values) => softmax(addVectors(matrixVector(network.moe.router.weights, values), network.moe.router.bias)));
    const expertOutputs = hidden.map((values) => network.moe.experts.map((expert) => {
      const head = addVectors(matrixVector(expert.hidden_weights, values), expert.hidden_bias).map(Math.tanh);
      return addVectors(matrixVector(expert.weights, head), expert.bias);
    }));
    const output = expertOutputs.map((experts, nodeIndex) => experts[0].map((_, feature) => experts.reduce((sum, expert, expertIndex) => sum + gates[nodeIndex][expertIndex] * expert[feature], 0)));
    return options.details ? { output, gates } : output;
  }
  const output = hidden.map((values) => {
    const head = matrixVector(network.output.hidden_weights, values).map((value, index) => Math.tanh(value + network.output.hidden_bias[index]));
    return matrixVector(network.output.weights, head).map((value, index) => value + network.output.bias[index]);
  });
  return options.details ? { output, gates: null } : output;
}

export function styleGate(bounds, candidateCount, model) {
  const shape = bounds.size.map((value) => value / Math.max(bounds.radius, 1e-6));
  const aspect = Math.max(...shape) / Math.max(Math.min(...shape), 1e-6);
  const thresholds = model?.gate?.thresholds || { dense_count: 8, elongated_aspect: 1.8, symmetric_shape_similarity: 0.88 };
  const symmetry = 1 - Math.min(1, Math.abs(shape[0] - shape[2]) / Math.max(shape[0], shape[2], 1e-6));
  const raw = { balanced: 0.3, elongated: aspect > thresholds.elongated_aspect ? 0.8 : 0.1, dense: candidateCount >= thresholds.dense_count ? 0.7 : 0.1, symmetric: symmetry >= thresholds.symmetric_shape_similarity ? 0.55 : 0.12 };
  const total = Object.values(raw).reduce((sum, value) => sum + value, 0);
  return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, value / total]));
}

export function applyLayoutModel(labels, bounds, model, options = {}) {
  const graphType = model?.architecture?.type;
  const fnnHybrid = graphType === 'fixed_label_fnn_relational_graph_transformer_moe';
  const cvHybrid = graphType === 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe';
  const heterogeneous = model?.architecture?.graph_type === 'anchor_label_heterogeneous_graph' || graphType === 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe';
  const pure3D = heterogeneous || model?.architecture?.generation_input === 'pure_3d' || graphType === 'fixed_label_3d_human_style_relational_graph_transformer_moe';
  const hybrid = graphType === 'fixed_label_relational_graph_transformer_moe' || fnnHybrid || cvHybrid || pure3D;
  if ((graphType === 'fixed_label_multiview_relational_gnn' || hybrid) && model?.network) {
    const heteroGraph = options.heterogeneousGraph || (heterogeneous && options.geometry ? buildHeterogeneousLayoutGraph(labels, options.geometry, bounds, options.spatialContext) : null);
    const run = forwardGraphNetwork(model.network, labels, bounds, null, { details: true, pure3D, heterogeneousGraph: heteroGraph, spatialContext: options.spatialContext, geometryFeature: options.geometryFeature, visualFeature: pure3D ? null : options.visualFeature });
    const predictions = run.output;
    const expertNames = model.network.moe?.experts?.map((expert) => expert.name) || [];
    const averageWeights = run.gates ? Object.fromEntries(expertNames.map((name, expertIndex) => [name, Number((run.gates.reduce((sum, row) => sum + row[expertIndex], 0) / Math.max(1, run.gates.length)).toFixed(6))])) : null;
    const centerBlend = clamp(Number(model.inference?.center_blend ?? 1), 0, 1);
    const sizeBlend = clamp(Number(model.inference?.size_blend ?? 1), 0, 1);
    const sizeRatioMin = Math.max(0.05, Number(model.inference?.size_ratio_range?.[0] ?? 0.05));
    const sizeRatioMax = Math.max(sizeRatioMin, Number(model.inference?.size_ratio_range?.[1] ?? 20));
    const output = labels.map((label, index) => {
      const prediction = predictions[index];
      const radius = bounds.radius;
      const predictedCenter = heterogeneous ? heteroGraph.frames[index].worldFromLocal({ u: prediction[0] * radius, v: prediction[1] * radius, normal: prediction[2] * radius }) : label.center.map((value, axis) => value + prediction[axis] * radius);
      const predictedSize = label.boxSize.map((value, axis) => Math.max(radius * (axis === 2 ? 0.005 : 0.01), Math.abs(value) * Math.exp(clamp(prediction[axis + 3], -2, 2))));
      const center = label.center.map((value, axis) => value + (predictedCenter[axis] - value) * centerBlend);
      const boxSize = label.boxSize.map((value, axis) => value + (clamp(predictedSize[axis], value * sizeRatioMin, value * sizeRatioMax) - value) * sizeBlend);
      const routerWeights = run.gates ? Object.fromEntries(expertNames.map((name, expertIndex) => [name, Number(run.gates[index][expertIndex].toFixed(6))])) : null;
      return { ...label, center, boxSize, layout_model: { architecture: heterogeneous ? 'anchor_frame_heterogeneous_graph_transformer_moe' : (pure3D ? '3d_human_style_relational_graph_transformer_moe' : (cvHybrid ? 'dgcnn_cnn_fnn_relational_graph_transformer_moe' : (fnnHybrid ? 'fnn_relational_graph_transformer_moe' : (hybrid ? 'relational_graph_transformer_moe' : 'relational_gnn')))), generation_input: pure3D ? 'pure_3d' : 'multiview_or_legacy', graph_type: heterogeneous ? 'anchor_label_heterogeneous_graph' : null, learned: true, node_feature_dim: heterogeneous ? HETEROGENEOUS_NODE_FEATURE_NAMES.length : (pure3D ? PURE_3D_LABEL_FEATURE_NAMES.length : MULTI_VIEW_LAYOUT_FEATURE_NAMES.length), fused_feature_dim: pure3D ? model.architecture.fused_feature_dim : (cvHybrid ? model.architecture.fused_feature_dim : null), visual_feature_dim: pure3D ? 0 : (model.architecture.visual_feature_dim || null), edge_feature_dim: model.architecture.edge_input_dim || GRAPH_EDGE_FEATURE_NAMES.length, pre_gnn_fnn_layers: model.architecture.pre_gnn_fnn_layers || 0, message_passing_layers: model.architecture.message_passing_layers, transformer_layers: model.architecture.transformer_layers || 0, transformer_global: true, moe_router_weights: routerWeights, version: model.version, center_blend: centerBlend, size_blend: sizeBlend } };
    });
    return hybrid
      ? { labels: output, expert: pure3D ? '3d_human_style_moe' : (cvHybrid ? 'dgcnn_cnn_fnn_graph_transformer_moe' : (fnnHybrid ? 'fnn_graph_transformer_moe' : 'graph_transformer_moe')), gate: { type: 'learned_per_node_softmax_moe', fusion: 'soft_weighted_sum', average_weights: averageWeights, human_style: pure3D } }
      : { labels: output, expert: 'relational_gnn', gate: { type: 'learned_message_passing', selected: 'relational_gnn' } };
  }
  if (!model?.experts) return { labels, expert: 'untrained', gate: {} };
  const gate = styleGate(bounds, labels[0]?.candidate_total ?? labels.length, model);
  const expertNames = Object.keys(model.experts);
  const selected = expertNames.reduce((best, name) => (gate[name] > (gate[best] || -Infinity) ? name : best), expertNames[0]);
  const expert = model.experts[selected];
  const inputDim = expert.network?.layers?.[0]?.weights?.[0]?.length || Math.max(0, Number(expert.weights?.[0]?.length || 1) - 1);
  const centerBlend = clamp(Number(model.inference?.center_blend ?? 1), 0, 1);
  const sizeBlend = clamp(Number(model.inference?.size_blend ?? 1), 0, 1);
  const sizeRatioMin = Math.max(0.05, Number(model.inference?.size_ratio_range?.[0] ?? 0.05));
  const sizeRatioMax = Math.max(sizeRatioMin, Number(model.inference?.size_ratio_range?.[1] ?? 20));
  const output = labels.map((label, index) => {
    const features = inputDim === PURE_3D_LABEL_FEATURE_NAMES.length
      ? pure3DLabelFeatures(label, bounds, index, labels.length, options.spatialContext, labels)
      : inputDim === MULTI_VIEW_LAYOUT_FEATURE_NAMES.length
        ? multiViewLayoutFeatures(label, bounds, index, labels.length)
        : layoutFeatures(label, bounds, index, labels.length);
    if (inputDim && features.length !== inputDim) throw new Error(`layout model input mismatch: expected ${inputDim}, got ${features.length}`);
    const prediction = expert.network ? forwardNetwork(expert.network, features) : linearPredict(expert.weights, features);
    const radius = bounds.radius;
    const predictedCenter = [label.anchor[0] + prediction[0] * radius, label.anchor[1] + prediction[1] * radius, label.anchor[2] + prediction[2] * radius];
    const predictedSize = [Math.max(radius * 0.01, Math.abs(prediction[3] * radius)), Math.max(radius * 0.01, Math.abs(prediction[4] * radius)), Math.max(radius * 0.005, Math.abs(prediction[5] * radius))];
    const center = label.center.map((value, axis) => value + (predictedCenter[axis] - value) * centerBlend);
    const boxSize = label.boxSize.map((value, axis) => {
      const limited = clamp(predictedSize[axis], value * sizeRatioMin, value * sizeRatioMax);
      return value + (limited - value) * sizeBlend;
    });
    return { ...label, center, boxSize, layout_model: { expert: selected, gate, learned: true, feature_dim: features.length, version: model.version, center_blend: centerBlend, size_blend: sizeBlend } };
  });
  return { labels: output, expert: selected, gate };
}

export function targetVector(manualLabel, anchor, bounds) {
  const radius = Math.max(bounds.radius, 1e-6);
  return [(manualLabel.center[0] - anchor[0]) / radius, (manualLabel.center[1] - anchor[1]) / radius, (manualLabel.center[2] - anchor[2]) / radius, manualLabel.boxSize[0] / radius, manualLabel.boxSize[1] / radius, manualLabel.boxSize[2] / radius];
}
