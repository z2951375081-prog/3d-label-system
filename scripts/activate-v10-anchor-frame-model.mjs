import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { activatePreservingPrevious } from '../lib/model-preservation.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const candidateFile = path.join(experiments, 'layout_model_v10_anchor_frame_candidate.json');
const activationCandidateFile = path.join(experiments, 'layout_model_v10_anchor_frame_activated.json');
const activeFile = path.join(experiments, 'layout_model.json');
const historyFile = path.join(experiments, 'layout_model_activation_history.jsonl');
const force = process.argv.slice(2).includes('--force');

const candidate = JSON.parse(await fs.readFile(candidateFile, 'utf8'));
const architecture = candidate.architecture || {};
if (candidate.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw new Error('candidate is not v10');
if (architecture.type !== 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe' || architecture.generation_input !== 'pure_3d') throw new Error('v10 architecture validation failed');
if (architecture.node_input_dim !== 51 || architecture.edge_input_dim !== 28 || architecture.obj_surface_points !== 1024 || architecture.dgcnn_edgeconv_layers < 2 || architecture.geometry_feature_dim !== 64 || architecture.message_passing_layers !== 2 || architecture.transformer_layers !== 1 || ![4, 5].includes(architecture.moe_expert_count)) throw new Error('v10 base structure validation failed');
if (architecture.graph_type !== 'anchor_label_heterogeneous_graph' || architecture.anchor_label_edge_dim !== 18 || architecture.label_label_edge_dim !== 10) throw new Error('heterogeneous edge contract validation failed');
if (architecture.visual_generation_input !== false || candidate.network?.fusion?.visual_dim !== 0) throw new Error('visual features still enter generator');
if (candidate.network?.fusion?.weights?.[0]?.length !== 128 || candidate.network?.message_layers?.length !== 2 || candidate.network?.message_layers?.[0]?.edge_weights?.[0]?.length !== 28 || candidate.network?.message_layers?.[0]?.anchor_edge_weights?.[0]?.length !== 18 || candidate.network?.message_layers?.[0]?.relation_edge_weights?.[0]?.length !== 10 || candidate.network?.transformer_layers?.length !== 1 || candidate.network?.moe?.experts?.length !== architecture.moe_expert_count) throw new Error('v10 network dimensions incomplete');
if (architecture.relation_parameterization !== 'separate_anchor_to_label_and_label_to_label_message_weights') throw new Error('v10 relation-specific parameters missing');
if (architecture.output_semantics !== 'local_u_local_v_local_normal_distance_and_log_size_xyz') throw new Error('decoder is not local-frame semantics');
if (!architecture.view_loss?.terms?.includes('leader_crossing') || !(architecture.view_loss?.leader_crossing_weight > 0) || candidate.training?.leader_crossing_loss_used !== true) throw new Error('v10 base model was not trained against leader crossing');
for (const key of ['input_fnn', 'pre_gnn_fnn', 'feature_fusion', 'message_passing_gnn', 'anchor_to_label_messages', 'label_to_label_messages', 'transformer', 'moe_router', 'moe_experts']) if (!(candidate.training?.parameter_updates?.[key]?.changed_parameters > 0)) throw new Error(`${key} has no parameter updates`);
const qualityGateStatus = candidate.validation_gate?.status || 'unknown';
if (!force && qualityGateStatus !== 'accepted') throw new Error(`v10 candidate validation gate is ${qualityGateStatus}; use --force only after an explicit user override`);

const activated = structuredClone(candidate);
activated.status = force ? 'active_user_override_v10_anchor_frame' : 'active_val_selected_v10_anchor_frame';
activated.inference = { ...activated.inference, selection_status: force ? 'activated_by_explicit_user_override' : 'activated_after_val11_quality_and_safety_gate' };
activated.validation_gate = { ...activated.validation_gate, status: force ? 'activated_by_user_override' : 'accepted', activation_override: force, activation_reason: force ? 'explicit --force activation after preserving val/test diagnostics' : 'v10 candidate passed val11 quality and safety selection', quality_gate_result_preserved: qualityGateStatus, safety_gate_result_preserved: candidate.validation_gate?.safety || null, activated_at: new Date().toISOString() };
activated.activation_provenance = { policy: force ? 'explicit_user_override' : 'val11_quality_and_safety_gate', previous_active_preserved: true, previous_active_expected_version: candidate.version, val_test_results_are_diagnostic_not_hidden: true, visual_generation_input: false, qwen_role: 'safe_candidate_aesthetic_reranking_only' };
await fs.writeFile(activationCandidateFile, JSON.stringify(activated, null, 2) + '\n', 'utf8');

const result = await activatePreservingPrevious({
  candidateFile: activationCandidateFile,
  activeFile,
  historyFile,
  validateCandidate(model) {
    if (!['active_val_selected_v10_anchor_frame', 'active_user_override_v10_anchor_frame'].includes(model.status)) throw new Error('activation copy status invalid');
    if (!force && model.validation_gate?.status !== 'accepted') throw new Error('accepted val gate missing');
    if (force && model.validation_gate?.activation_override !== true) throw new Error('activation override audit missing');
    if (model.architecture?.generation_input !== 'pure_3d' || model.architecture?.visual_generation_input !== false) throw new Error('generation role validation failed');
  },
  metadata: { gate: force ? 'explicit_user_override' : 'val11_quality_and_safety', quality_gate: qualityGateStatus, fixed_label_contract: true, generation_input: 'pure_3d', graph_type: 'anchor_label_heterogeneous_graph', five_view_role: 'evaluation_safety_optimization_only', qwen_role: 'safe_candidate_aesthetic_reranking_only' }
});
console.log(JSON.stringify({ activated: result.candidate.version, status: result.candidate.status, validation_gate: result.candidate.validation_gate, activation: result.record }, null, 2));
