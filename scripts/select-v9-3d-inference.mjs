import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildSpatialContext, computeSpatialStyleMetrics } from '../lib/spatial-style-features.mjs';
import { dgcnnGeometryFeature } from '../lib/cv-feature-encoder.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = async (file) => JSON.parse(await fs.readFile(path.resolve(file), 'utf8'));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

function parseArgs(argv) {
  const options = { model: 'experiments/layout_model_v9_3d_human_style_candidate.json', previous: 'experiments/layout_model.json', manifest: 'experiments/dataset_manifest.json', output: 'experiments/v9_3d_inference_selection.json', limit: 0, iterations: 120 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function assertPure3DModel(model) {
  const architecture = model?.architecture || {};
  const isV10 = model?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe';
  const isV9 = model?.version === 'layout_model_v9_3d_human_style_moe';
  if (!isV9 && !isV10) throw new Error('候选版本不是 v9/v10 纯三维人类风格模型');
  const expectedType = isV10 ? 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe' : 'fixed_label_3d_human_style_relational_graph_transformer_moe';
  if (architecture.type !== expectedType || architecture.generation_input !== 'pure_3d') throw new Error('候选不是纯三维生成架构');
  if (architecture.node_input_dim !== 51 || architecture.geometry_feature_dim !== 64 || architecture.obj_surface_points !== 1024 || architecture.dgcnn_edgeconv_layers < 2 || architecture.message_passing_layers !== 2 || architecture.transformer_layers !== 1 || ![4, 5].includes(architecture.moe_expert_count)) throw new Error('纯三维结构参数不满足目标');
  const visualMetadataValid = isV10 ? [0, 32].includes(architecture.visual_feature_dim) : architecture.visual_feature_dim === 32;
  if (architecture.visual_generation_input !== false || !visualMetadataValid || architecture.fused_feature_dim !== 128) throw new Error('五视角视觉分支职责或生成融合维度错误');
  if (isV10 && (architecture.graph_type !== 'anchor_label_heterogeneous_graph' || architecture.anchor_label_edge_dim !== 18 || architecture.label_label_edge_dim !== 10 || model.network?.message_layers?.[0]?.anchor_edge_weights?.[0]?.length !== 18 || model.network?.message_layers?.[0]?.relation_edge_weights?.[0]?.length !== 10)) throw new Error('v10 异构关系参数不完整');
  const names = new Set(['compact_style', 'balanced_style', 'spacious_style', 'long_text_style', 'directional_style']);
  if (!model.network?.moe?.experts?.every((expert) => names.has(expert.name))) throw new Error('MoE 专家不是人类风格专家');
  if (model.network?.fusion?.visual_dim !== 0 || model.network?.fusion?.weights?.[0]?.length !== 128) throw new Error('v9 生成器仍包含视觉输入');
  const updates = model.training?.parameter_updates || {};
  for (const key of ['input_fnn', 'pre_gnn_fnn', 'feature_fusion', 'message_passing_gnn', 'transformer', 'moe_router', 'moe_experts']) if (!(updates[key]?.changed_parameters > 0)) throw new Error(`${key} 没有实际训练更新`);
}

async function evaluateModel(model, manifest, split, options) {
  const rows = [];
  for (const sample of manifest.samples.filter((item) => item.split === split).slice(0, Number(options.limit) || undefined)) {
    const clean = cleanObj(await fs.readFile(path.join(root, sample.input.source_obj), 'utf8'));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const manual = annotationsToLabels(await readJson(path.join(root, sample.target.annotation_json)));
    const candidates = fixedCandidatesWithoutTargetLayout(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/${split}/candidate`);
    const spatialContext = buildSpatialContext(geometry, bounds, { gridSize: model.architecture.spatial_grid?.grid_size || 20 });
    const generated = applyLayoutModel(candidates, bounds, model, { geometry, geometryFeature: dgcnnGeometryFeature(clean.text, { pointCount: 1024, neighborCount: 4 }), spatialContext }).labels;
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const optionsForOptimizer = { fixedLabels: true, groupPolicy: 'all', viewPolicy: 'binocular', optimizer: 'annealing', iterations: Number(options.iterations), seed: 17, depthGrids, category: sample.category, manualReference: manual, geometry };
    const safe = optimizeLabels(generated, bounds, optionsForOptimizer);
    validateFixedLabelContract(manual, safe, `${sample.category}/${sample.sample_id}/${split}/output`);
    rows.push({ category: sample.category, sample_id: sample.sample_id, view: evaluateLayout(safe, bounds, optionsForOptimizer), spatial: computeSpatialStyleMetrics(safe, bounds, spatialContext) });
  }
  const viewKeys = ['multidimensional_quality_score', 'objective_score', 'olr', 'lcd', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'viewport_overflow_ratio', 'multi_view_worst_overflow', 'multi_view_worst_penetration', 'manual_style_distance', 'leader_length_compliance_ratio', 'directional_uniformity'];
  const spatialKeys = ['air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance', 'min_3d_clearance', 'mean_3d_spacing', 'min_3d_spacing', 'spacing_std', 'leader_length_mean', 'leader_length_std', 'out_of_sight_ratio'];
  return { sample_count: rows.length, ...Object.fromEntries(viewKeys.map((key) => [key, Number(mean(rows.map((row) => finite(row.view[key]))).toFixed(6))])), ...Object.fromEntries(spatialKeys.map((key) => [key, Number(mean(rows.map((row) => finite(row.spatial[key]))).toFixed(6))])), rows };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const candidate = await readJson(options.model);
  const previous = await readJson(options.previous);
  const manifest = await readJson(options.manifest);
  assertPure3DModel(candidate);
  const isV10 = candidate.version.startsWith('layout_model_v10_');
  const previousValidation = await evaluateModel(previous, manifest, 'val', options);
  const blendGrid = [[0.2, 0.6], [0.35, 0.6], [0.5, 0.6], [0.65, 0.8], [0.8, 1], [1, 1]];
  const candidates = [];
  for (const [centerBlend, sizeBlend] of blendGrid) {
    const trial = structuredClone(candidate);
    trial.inference = { ...trial.inference, center_blend: centerBlend, size_blend: sizeBlend };
    const metrics = await evaluateModel(trial, manifest, 'val', options);
    const safety = {
      label_object_occlusion_change: metrics.label_object_occlusion_ratio - previousValidation.label_object_occlusion_ratio,
      penetration_change: metrics.object_penetration_ratio - previousValidation.object_penetration_ratio,
      mesh_intersection_change: metrics.mesh_surface_intersection_ratio - previousValidation.mesh_surface_intersection_ratio,
      overflow_change: metrics.multi_view_worst_overflow - previousValidation.multi_view_worst_overflow,
      leader_crossing_change: metrics.lcd - previousValidation.lcd
    };
    const qualityDelta = metrics.multidimensional_quality_score - previousValidation.multidimensional_quality_score;
    const objectiveDelta = metrics.objective_score - previousValidation.objective_score;
    const safetyEligible = safety.label_object_occlusion_change <= 0.02 && safety.penetration_change <= 0.01 && safety.mesh_intersection_change <= 0.02 && safety.overflow_change <= 0.01 && (!isV10 || safety.leader_crossing_change <= 0.005);
    candidates.push({ center_blend: centerBlend, size_blend: sizeBlend, metrics, quality_delta: qualityDelta, objective_delta: objectiveDelta, safety, safety_eligible: safetyEligible, accepted: safetyEligible && qualityDelta >= -0.02 && objectiveDelta <= 0.05 });
  }
  const selected = candidates.filter((item) => item.accepted).sort((left, right) => (right.quality_delta - right.objective_delta * 0.05) - (left.quality_delta - left.objective_delta * 0.05))[0] || candidates.sort((left, right) => (right.quality_delta - right.objective_delta * 0.05) - (left.quality_delta - left.objective_delta * 0.05))[0];
  candidate.inference = { ...candidate.inference, center_blend: selected.center_blend, size_blend: selected.size_blend };
  const validation = { candidate: selected.metrics, previous: previousValidation, searched_candidates: candidates };
  const qualityDelta = validation.candidate.multidimensional_quality_score - validation.previous.multidimensional_quality_score;
  const objectiveDelta = validation.candidate.objective_score - validation.previous.objective_score;
  const safety = {
    label_object_occlusion_change: validation.candidate.label_object_occlusion_ratio - validation.previous.label_object_occlusion_ratio,
    penetration_change: validation.candidate.object_penetration_ratio - validation.previous.object_penetration_ratio,
    mesh_intersection_change: validation.candidate.mesh_surface_intersection_ratio - validation.previous.mesh_surface_intersection_ratio,
    overflow_change: validation.candidate.multi_view_worst_overflow - validation.previous.multi_view_worst_overflow,
    leader_crossing_change: validation.candidate.lcd - validation.previous.lcd
  };
  const accepted = qualityDelta >= -0.02 && objectiveDelta <= 0.05 && safety.label_object_occlusion_change <= 0.02 && safety.penetration_change <= 0.01 && safety.mesh_intersection_change <= 0.02 && safety.overflow_change <= 0.01 && (!isV10 || safety.leader_crossing_change <= 0.005);
  candidate.validation_gate = { status: accepted ? 'accepted' : 'rejected', criterion: isV10 ? 'v10 local-frame heterogeneous pure-3D generator; val quality/objective/safety gate; main-weighted worst-view/CVaR terms are trained before deterministic five-view optimization; test is not used for activation' : 'v9 pure-3D generator; val quality/objective/safety gate; five-view metrics are evaluator-only; test is not used for activation', selected_without_test: true, selected_blend: { center_blend: selected.center_blend, size_blend: selected.size_blend }, quality_delta: Number(qualityDelta.toFixed(6)), objective_delta: Number(objectiveDelta.toFixed(6)), safety, previous_active: validation.previous, candidate: validation.candidate, searched_candidates: candidates.map((item) => ({ center_blend: item.center_blend, size_blend: item.size_blend, quality_delta: Number(item.quality_delta.toFixed(6)), objective_delta: Number(item.objective_delta.toFixed(6)), safety: item.safety, safety_eligible: item.safety_eligible, accepted: item.accepted })) };
  await fs.writeFile(path.resolve(options.model), JSON.stringify(candidate, null, 2) + '\n', 'utf8');
  const test = { candidate: await evaluateModel(candidate, manifest, 'test', options), previous: await evaluateModel(previous, manifest, 'test', options) };
  const report = { version: isV10 ? 'v10_anchor_frame_inference_selection_v1' : 'v9_3d_inference_selection_v1', generated_at: new Date().toISOString(), candidate_model_file: String(options.model).replaceAll('\\', '/'), validation: { accepted, ...candidate.validation_gate }, test_confirmation: test, policy: { test_not_used_for_activation: true, visual_features_not_used_by_generator: true, fixed_label_contract_validated: true, qwen_role: 'safe_candidate_aesthetic_reranking_only' } };
  await fs.writeFile(path.resolve(options.output), JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ accepted, validation: candidate.validation_gate, test: { candidate: Object.fromEntries(Object.entries(test.candidate).filter(([key]) => key !== 'rows')), previous: Object.fromEntries(Object.entries(test.previous).filter(([key]) => key !== 'rows')) } }, null, 2));
}

main().catch((error) => { console.error(error.stack || error.message || error); process.exitCode = 1; });
