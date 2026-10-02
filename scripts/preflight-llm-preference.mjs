import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { LLM_AESTHETIC_DIMENSIONS, LLM_SAFETY_DIMENSIONS, LLM_SCORE_NAMES } from '../public/preference-scoring.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const options = { baseUrl: 'http://127.0.0.1:5173', samples: 3, candidates: 4, rounds: 8, visualValSamples: 11, output: 'experiments/llm_preference_preflight.json' };
  for (let index = 0; index < argv.length; index += 1) if (argv[index].startsWith('--')) {
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const experiments = process.env.EXPERIMENTS_DIR ? path.resolve(process.env.EXPERIMENTS_DIR) : path.join(root, 'experiments');
const output = path.resolve(root, String(options.output));
const checks = [];
const add = (name, passed, details = {}, blocking = true) => checks.push({ name, passed: Boolean(passed), blocking: Boolean(blocking), details });
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const exists = async (file) => fs.stat(file).then((stat) => stat.isFile(), () => false);

let config = null;
try {
  const response = await fetch(`${String(options.baseUrl).replace(/\/+$/, '')}/api/scoring-config`, { signal: AbortSignal.timeout(5000) });
  config = response.ok ? await response.json() : null;
  add('project_service', response.ok, { base_url: options.baseUrl, http_status: response.status });
} catch (error) {
  add('project_service', false, { base_url: options.baseUrl, error: error.message });
}
add('local_qwen_visual_connection', config?.configured && config?.tested, {
  configured: Boolean(config?.configured), tested: Boolean(config?.tested), tested_at: config?.testedAt || null,
  model: config?.model || null, format: config?.format || null, scoring_protocol: config?.scoringProtocol || null, local_only: config?.localOnly === true, credentials_required: config?.credentialsRequired === true
});

let manifest = null;
try {
  manifest = await readJson(path.join(experiments, 'dataset_manifest.json'));
  const counts = manifest.counts || {};
  add('dataset_split', manifest.samples?.length === 55 && counts.train === 33 && counts.val === 11 && counts.test === 11, { total: manifest.samples?.length, train: counts.train, val: counts.val, test: counts.test, policy: manifest.policy });
} catch (error) { add('dataset_split', false, { error: error.message }); }

let adjustedSamples = 0, labels = 0, annotationErrors = [];
if (manifest?.samples) for (const sample of manifest.samples) {
  try {
    const annotation = await readJson(path.join(root, sample.target.annotation_json));
    const valid = annotation.version === 'after_mannual_adjust' && annotation.layout_type === 'manual_adjusted' && Array.isArray(annotation.groups) && annotation.groups.every((group) => Array.isArray(group.label?.center) && Array.isArray(group.label?.box_size));
    if (!valid) annotationErrors.push(`${sample.category}/${sample.sample_id}`);
    else { adjustedSamples += 1; labels += annotation.groups.length; }
  } catch { annotationErrors.push(`${sample.category}/${sample.sample_id}`); }
}
add('manual_adjusted_supervision', adjustedSamples === 55 && labels === 406 && annotationErrors.length === 0, { adjusted_samples: adjustedSamples, labels, invalid_samples: annotationErrors });

try {
  const schema = await readJson(path.join(experiments, 'score-schema.json'));
  const required = schema.properties?.scores?.required || [];
  add('fourteen_dimension_schema_aesthetic_preference', required.length === 14 && LLM_SCORE_NAMES.every((name) => required.includes(name)), { required_dimensions: required, preference_dimensions: LLM_AESTHETIC_DIMENSIONS, safety_diagnostics: LLM_SAFETY_DIMENSIONS, policy: 'deterministic energy safety gate, then LLM aesthetic reward' });
} catch (error) { add('fourteen_dimension_schema_aesthetic_preference', false, { error: error.message }); }

try {
  const protocol = await readJson(path.join(experiments, 'dataset_camera_protocol.json'));
  const calibration = await readJson(path.join(experiments, 'dataset_camera_calibration.json'));
  const viewNames = protocol.observed_from_dataset?.view_names || [];
  const calibratedViews = Object.keys(calibration.views || {});
  const heldOut = calibratedViews.every((name) => calibration.views[name]?.cases?.val === 11 && calibration.views[name]?.cases?.test === 11);
  add('dataset_camera_protocol', viewNames.length === 5 && viewNames.every((name) => calibratedViews.includes(name)) && heldOut, {
    status: protocol.status, calibration_status: calibration.status, view_names: viewNames,
    camera_distance_fixed: protocol.shared_reproduction_parameters?.camera_distance_world_units,
    held_out_val_test_per_view: heldOut,
    limitation: 'image-derived reproduction estimate; not original camera metadata'
  });
} catch (error) { add('dataset_camera_protocol', false, { error: error.message }); }

try {
  const comparison = await readJson(path.join(experiments, 'comparisons', 'round1', 'comparison.json'));
  const unified = comparison.unified_snapshot_evaluation || {};
  const methods = [...new Set((unified.rows || []).map((row) => row.method))];
  const hasBino = methods.some((name) => /binoforce/i.test(name));
  const hasHedge1 = methods.includes('hedgehog_1d'), hasHedge3 = methods.includes('hedgehog_3d');
  add('binoforce_hedgehog_comparison', hasBino && hasHedge1 && hasHedge3 && (unified.paired_bootstrap?.comparisons?.length || 0) > 0, {
    camera_status: comparison.camera_protocol?.status, methods, snapshot_rows: unified.rows?.length || 0,
    paired_bootstrap_comparisons: unified.paired_bootstrap?.comparisons?.length || 0,
    dynamic_and_final_snapshot_separated: Boolean(comparison.source_protocol?.BinoForce_metrics && comparison.source_protocol?.BinoForce_layout_coordinates)
  });
} catch (error) { add('binoforce_hedgehog_comparison', false, { error: error.message }); }

try {
  const bytes = await fs.readFile(path.join(experiments, 'layout_model.json'));
  const model = JSON.parse(bytes.toString('utf8'));
  const digest = createHash('sha256').update(bytes).digest('hex').toUpperCase();
  const v5Active = model.version === 'layout_model_v5_multiview_relational_gnn' && model.architecture?.type === 'fixed_label_multiview_relational_gnn';
  const v6Active = model.version === 'layout_model_v6_relational_graph_transformer_moe'
    && model.architecture?.type === 'fixed_label_relational_graph_transformer_moe'
    && model.architecture?.transformer_layers >= 1
    && model.network?.transformer_layers?.length === model.architecture.transformer_layers
    && model.architecture?.moe_expert_count >= 2
    && model.network?.moe?.experts?.length === model.architecture.moe_expert_count
    && model.training?.parameter_updates?.transformer?.changed_parameters > 0
    && model.training?.parameter_updates?.moe_router?.changed_parameters > 0;
  const v7Active = model.version === 'layout_model_v7_fnn_relational_graph_transformer_moe'
    && model.architecture?.type === 'fixed_label_fnn_relational_graph_transformer_moe'
    && model.architecture?.pre_gnn_fnn_layers === 1
    && model.network?.pre_gnn_fnn_layers?.length === 1
    && model.architecture?.transformer_layers >= 1
    && model.network?.transformer_layers?.length === model.architecture.transformer_layers
    && model.architecture?.moe_expert_count === 4
    && model.network?.moe?.experts?.length === 4
    && model.training?.parameter_updates?.pre_gnn_fnn?.changed_parameters > 0
    && model.training?.parameter_updates?.transformer?.changed_parameters > 0
    && model.training?.parameter_updates?.moe_router?.changed_parameters > 0;
  const v8Active = model.version === 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe'
    && model.architecture?.type === 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe'
    && model.architecture?.obj_surface_points === 1024
    && model.architecture?.dgcnn_edgeconv_layers >= 2
    && model.architecture?.geometry_feature_dim === 64
    && model.architecture?.visual_feature_dim === 32
    && model.architecture?.fused_feature_dim === 160
    && model.network?.fusion?.weights?.[0]?.length === 160
    && model.training?.parameter_updates?.feature_fusion?.changed_parameters > 0
    && model.training?.functional_evidence?.cv_fusion_ablation?.max_abs_output_change > 1e-8;
  const v9Active = model.version === 'layout_model_v9_3d_human_style_moe'
    && model.architecture?.type === 'fixed_label_3d_human_style_relational_graph_transformer_moe'
    && model.architecture?.generation_input === 'pure_3d'
    && model.architecture?.node_input_dim === 51
    && model.architecture?.obj_surface_points === 1024
    && model.architecture?.dgcnn_edgeconv_layers >= 2
    && model.architecture?.geometry_feature_dim === 64
    && model.architecture?.message_passing_layers === 2
    && model.architecture?.transformer_layers === 1
    && [4, 5].includes(model.architecture?.moe_expert_count)
    && model.architecture?.visual_generation_input === false
    && model.network?.fusion?.visual_dim === 0
    && model.training?.parameter_updates?.input_fnn?.changed_parameters > 0
    && model.training?.parameter_updates?.message_passing_gnn?.changed_parameters > 0
    && model.training?.parameter_updates?.transformer?.changed_parameters > 0
    && model.training?.parameter_updates?.moe_router?.changed_parameters > 0;
  const v10Active = model.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
    && model.architecture?.type === 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe'
    && model.architecture?.generation_input === 'pure_3d'
    && model.architecture?.node_input_dim === 51
    && model.architecture?.obj_surface_points === 1024
    && model.architecture?.dgcnn_edgeconv_layers >= 2
    && model.architecture?.geometry_feature_dim === 64
    && model.architecture?.graph_type === 'anchor_label_heterogeneous_graph'
    && model.architecture?.anchor_label_edge_dim === 18
    && model.architecture?.label_label_edge_dim === 10
    && model.architecture?.edge_input_dim === 28
    && model.architecture?.message_passing_layers === 2
    && model.architecture?.transformer_layers === 1
    && model.architecture?.output_semantics === 'local_u_local_v_local_normal_distance_and_log_size_xyz'
    && model.architecture?.visual_generation_input === false
    && model.network?.fusion?.visual_dim === 0
    && model.network?.message_layers?.[0]?.anchor_edge_weights?.[0]?.length === 18
    && model.network?.message_layers?.[0]?.relation_edge_weights?.[0]?.length === 10
    && model.training?.parameter_updates?.anchor_to_label_messages?.changed_parameters > 0
    && model.training?.parameter_updates?.label_to_label_messages?.changed_parameters > 0
    && model.training?.leader_direction_loss_used === true
    && model.training?.leader_crossing_loss_used === true
    && model.training?.per_view_font_clarity_loss_used === true;
  const activationOverride = ['activated_by_user_request', 'activated_by_user_override'].includes(model.validation_gate?.status) && model.validation_gate?.activation_override === true;
  const accepted = model.validation_gate?.status === 'accepted' || activationOverride;
  const legacyGraphActive = (v5Active || v6Active || v7Active || v8Active || v9Active) && model.architecture?.edge_input_dim === 13;
  const graphActive = (legacyGraphActive || v10Active) && model.architecture?.node_input_dim === 51 && model.architecture?.message_passing_layers >= 1 && accepted;
  add('active_layout_model', graphActive, {
    version: model.version, status: model.status, sha256: digest, architecture: model.architecture, validation_gate: model.validation_gate?.status || null
  });
} catch (error) { add('active_layout_model', false, { error: error.message }); }

try {
  const prior = await readJson(path.join(experiments, 'manual_leader_length_prior.json'));
  const policy = prior.selected_policy || {};
  add('train_manual_leader_length_prior', prior.enabled === true && prior.source?.split === 'train_only' && prior.source?.sample_count === 33 && prior.source?.label_count === 248 && prior.source?.val_used_for_distribution === false && prior.source?.test_used_for_distribution === false && policy.preferred_min_quantile === 'p10' && policy.preferred_max_quantile === 'p90' && prior.validation_selection?.selected_without_test === true, {
    version: prior.version, source: prior.source, selected_policy: policy,
    global_train_distribution: prior.global,
    validation_selected: prior.validation_selection?.selected_configuration,
    test_used_after_selection_only: Boolean(prior.test_confirmation?.used_after_selection)
  });
} catch (error) { add('train_manual_leader_length_prior', false, { error: error.message }); }

try {
  const policy = await readJson(path.join(experiments, 'directional_density_policy.json'));
  const active = await readJson(path.join(experiments, 'layout_model.json')).catch(() => null);
  const v10PerView = active?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
    && active?.architecture?.view_loss?.type === 'weighted_main_plus_worst_view_plus_cvar_plus_stereo'
    && active?.architecture?.view_loss?.weights?.main === 0.4
    && active?.architecture?.human_style_targets?.includes('per_view_free_space_distribution');
  const legacyPolicy = policy.enabled === true && policy.selected_without_test === true && policy.selected_configuration === 'strong' && Number(policy.weights?.directional_density) === 1.6 && Number(policy.weights?.directional_concentration) === 1.2;
  add('per_view_free_space_distribution', v10PerView || legacyPolicy, {
    active_v10_per_view_free_space_distribution: v10PerView,
    version: policy.version, sectors: policy.definition?.sectors, available_space: policy.definition?.available_space,
    selected_configuration: policy.selected_configuration, weights: policy.weights,
    selection_split: 'val11', test_confirmation_after_selection: Array.isArray(policy.test_confirmation)
  });
} catch (error) { add('directional_density_free_space_penalty', false, { error: error.message }); }

const samples = Math.max(1, Math.min(33, Number(options.samples) || 3));
const candidates = Math.max(2, Math.min(8, Number(options.candidates) || 4));
const rounds = Math.max(1, Math.min(10, Number(options.rounds) || 8));
const visualValSamples = Math.max(1, Math.min(11, Number(options.visualValSamples) || 11));
const trainRequests = samples * candidates * rounds;
const validationRequests = visualValSamples * (rounds + 1);
const totalRequests = trainRequests + validationRequests;
add('experiment_scope', rounds >= 8 && visualValSamples === 11, {
  train_samples: samples, candidates_per_sample_per_round: candidates, rounds, visual_val_samples: visualValSamples,
  train_requests: trainRequests, validation_requests: validationRequests, total_external_requests_excluding_probe: totalRequests,
  images_per_request: 6, total_images_excluding_probe: totalRequests * 6,
  standard_visual_checkpoints: [0, 1, 2, 4, 8], can_test_rise_then_stable: rounds >= 8 && visualValSamples === 11
});

const requiredFiles = [
  'public/dataset-camera.js', 'scripts/calibrate-dataset-camera.py', 'scripts/run-llm-preference-headless.mjs',
  'experiments/dataset_camera_protocol.json', 'experiments/dataset_camera_calibration.json',
  'experiments/manual_leader_length_prior.json', 'experiments/leader_length_policy_selection.json', 'experiments/directional_density_policy.json',
  'experiments/directional_uniformity_upgrade_offline.json', 'experiments/adaptive_directional_rerank_offline.json',
  'experiments/comparisons/round1/comparison.json', 'experiments/comparisons/round1/paired_bootstrap.csv'
];
const missing = [];
for (const file of requiredFiles) if (!await exists(path.join(root, file))) missing.push(file);
add('required_artifacts', missing.length === 0, { required_files: requiredFiles, missing });

const blockingFailures = checks.filter((check) => check.blocking && !check.passed);
const report = {
  version: 'llm_preference_preflight_v1', generated_at: new Date().toISOString(), ready: blockingFailures.length === 0,
  requested_experiment: { samples, candidates, rounds, visual_val_samples: visualValSamples },
  request_budget: { local_requests_excluding_connection_probe: totalRequests, connection_probe_requests: 1, images_per_request: 6, images_excluding_probe: totalRequests * 6 },
  checks,
  blocking_failures: blockingFailures.map((check) => check.name),
  next_action: blockingFailures.some((check) => check.name === 'local_qwen_visual_connection') ? '启动本机 Ollama，确认 qwen3-vl:4b-instruct 已安装，并通过六图与 JSON 测试。' : '可启动正式 LLM 偏好实验。'
};
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, JSON.stringify(report, null, 2) + '\n', 'utf8');
console.log(JSON.stringify(report, null, 2));
if (!report.ready) process.exitCode = 2;
