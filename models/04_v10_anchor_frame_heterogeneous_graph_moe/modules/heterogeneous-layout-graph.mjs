import { ANCHOR_LOCAL_LABEL_FEATURE_NAMES, anchorLocal3DLabelFeatures } from './spatial-style-features.mjs';
import { ANCHOR_LABEL_EDGE_FEATURE_NAMES, LABEL_RELATION_EDGE_FEATURE_NAMES, anchorLabelEdgeFeatures, buildAnchorFrames, labelRelationEdgeFeatures } from './anchor-frame-features.mjs';

export const HETEROGENEOUS_NODE_FEATURE_NAMES = ANCHOR_LOCAL_LABEL_FEATURE_NAMES;
export const HETEROGENEOUS_EDGE_FEATURE_NAMES = [...LABEL_RELATION_EDGE_FEATURE_NAMES, ...ANCHOR_LABEL_EDGE_FEATURE_NAMES];

export function buildHeterogeneousLayoutGraph(labels, geometry, bounds, spatialContext, options = {}) {
  const frames = buildAnchorFrames(labels, geometry, bounds, options);
  const nodeFeatures = labels.map((label, index) => anchorLocal3DLabelFeatures(label, frames[index], bounds, index, labels.length, spatialContext, labels));
  const anchorEdgeFeatures = labels.map((targetLabel) => labels.map((sourceLabel, source) => anchorLabelEdgeFeatures(targetLabel, frames[source], bounds, sourceLabel)));
  const anchorEdges = labels.map((_, index) => anchorEdgeFeatures[index][index]);
  const relationEdges = labels.map((_, target) => labels.map((__, source) => source === target ? null : labelRelationEdgeFeatures(labels[source], labels[target], bounds)));
  const edgeFeatures = labels.map((_, target) => labels.map((__, source) => {
    if (source === target) return null;
    return [...relationEdges[target][source], ...anchorEdges[target]];
  }));
  return {
    frames,
    nodeTypes: { anchor: labels.length, label: labels.length },
    anchorNodes: labels.map((label, index) => ({ id: `anchor:${label.id ?? index}`, position: [...label.anchor], frame: frames[index] })),
    labelNodes: labels.map((label, index) => ({ id: `label:${label.id ?? index}`, label_index: index })),
    relationTypes: {
      anchor_to_label: anchorEdgeFeatures.flatMap((row, target) => row.map((features, source) => ({ source, target, features }))),
      label_to_label: relationEdges.flatMap((row, target) => row.flatMap((features, source) => features ? [{ source, target, features }] : []))
    },
    nodeFeatures,
    anchorEdges,
    anchorEdgeFeatures,
    relationEdges,
    edgeFeatures,
    node_feature_dim: HETEROGENEOUS_NODE_FEATURE_NAMES.length,
    relation_edge_dim: LABEL_RELATION_EDGE_FEATURE_NAMES.length,
    anchor_edge_dim: ANCHOR_LABEL_EDGE_FEATURE_NAMES.length,
    edge_feature_dim: HETEROGENEOUS_EDGE_FEATURE_NAMES.length,
    graph_type: 'anchor_label_heterogeneous_graph'
  };
}

export function localTargetVector(manualLabel, anchorLabel, frame, bounds) {
  const radius = Math.max(bounds.radius, 1e-8);
  const local = frame.localCoordinates(manualLabel.center);
  return [local.u / radius, local.v / radius, local.normal / radius, ...manualLabel.boxSize.map((value, axis) => Math.log(Math.max(Math.abs(value), 1e-8) / Math.max(Math.abs(anchorLabel.boxSize[axis]), 1e-8)))];
}

export function decodeLocalLayoutOutput(output, label, frame, bounds) {
  const radius = Math.max(bounds.radius, 1e-8);
  const center = frame.worldFromLocal({ u: output[0] * radius, v: output[1] * radius, normal: output[2] * radius });
  const boxSize = label.boxSize.map((value, axis) => Math.max(radius * (axis === 2 ? 0.005 : 0.01), Math.abs(value) * Math.exp(Math.max(-3, Math.min(3, output[axis + 3])))));
  return { ...label, center, boxSize };
}

export function heterogeneousArchitectureMetadata(options = {}) {
  return {
    graph_type: 'anchor_label_heterogeneous_graph',
    anchor_node_semantics: 'one_anchor_surface_patch_context_per_label',
    label_node_semantics: 'one_label_node_per_fixed_label',
    anchor_label_edge_dim: ANCHOR_LABEL_EDGE_FEATURE_NAMES.length,
    label_label_edge_dim: LABEL_RELATION_EDGE_FEATURE_NAMES.length,
    combined_edge_dim: HETEROGENEOUS_EDGE_FEATURE_NAMES.length,
    relation_parameterization: 'separate_anchor_to_label_and_label_to_label_message_weights',
    anchor_nodes_are_fixed_geometric_context: true,
    anchor_label_edge_features: ANCHOR_LABEL_EDGE_FEATURE_NAMES,
    label_label_edge_features: LABEL_RELATION_EDGE_FEATURE_NAMES,
    local_coordinate_frame: 'weighted_surface_patch_pca_plus_area_weighted_normals',
    frame_fallbacks: ['nearest_triangle_patch', 'normal_sign_alignment_to_object_center', 'tangent_axis_fallback_for_degenerate_patch'],
    decoder_semantics: 'local_tangent_u_local_tangent_v_surface_normal_distance_and_log_size_xyz',
    relation_layers: Number(options.messageLayers || 2),
    equivariance_target: 'E3_local_frame_consistent_with_rotation_aligned_surface_frame'
  };
}
