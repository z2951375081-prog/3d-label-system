import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { activatePreservingPrevious } from '../lib/model-preservation.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const candidateFile = path.join(experiments, 'layout_model_v9_3d_human_style_candidate.json');
const activationCandidateFile = path.join(experiments, 'layout_model_v9_3d_human_style_activated.json');
const activeFile = path.join(experiments, 'layout_model.json');
const historyFile = path.join(experiments, 'layout_model_activation_history.jsonl');

const candidate = JSON.parse(await fs.readFile(candidateFile, 'utf8'));
if (candidate.version !== 'layout_model_v9_3d_human_style_moe') throw new Error('候选不是最新 v9 三维人类风格模型');
const architecture = candidate.architecture || {};
if (architecture.type !== 'fixed_label_3d_human_style_relational_graph_transformer_moe' || architecture.generation_input !== 'pure_3d') throw new Error('v9 纯三维生成架构校验失败');
if (architecture.node_input_dim !== 51 || architecture.obj_surface_points !== 1024 || architecture.dgcnn_edgeconv_layers < 2 || architecture.geometry_feature_dim !== 64 || architecture.message_passing_layers !== 2 || architecture.transformer_layers !== 1 || ![4, 5].includes(architecture.moe_expert_count)) throw new Error('v9 基础结构校验失败');
if (architecture.visual_generation_input !== false || candidate.network?.fusion?.visual_dim !== 0) throw new Error('五视角视觉特征仍进入生成器，拒绝激活');
if (candidate.network?.fusion?.weights?.[0]?.length !== 128) throw new Error('v9 生成融合应为 128D→64D');
if (candidate.network?.message_layers?.length !== 2 || candidate.network?.transformer_layers?.length !== 1 || candidate.network?.moe?.experts?.length !== architecture.moe_expert_count) throw new Error('v9 权重层数不完整');
for (const key of ['input_fnn', 'pre_gnn_fnn', 'feature_fusion', 'message_passing_gnn', 'transformer', 'moe_router', 'moe_experts']) if (!(candidate.training?.parameter_updates?.[key]?.changed_parameters > 0)) throw new Error(`${key} 没有实际训练更新`);

const activated = structuredClone(candidate);
activated.status = 'active_latest_user_requested_v9_pure_3d';
activated.inference = { ...activated.inference, selection_status: 'activated_latest_user_requested_without_quality_gate' };
activated.validation_gate = {
  ...activated.validation_gate,
  status: 'activated_by_user_request',
  activation_override: true,
  activation_reason: '用户明确要求先替换最新模型，不因 val/test 质量下降拒绝 v9',
  quality_gate_result_preserved: candidate.validation_gate?.status || 'unknown',
  safety_gate_result_preserved: candidate.validation_gate?.safety || null,
  activated_at: new Date().toISOString()
};
activated.activation_provenance = {
  policy: 'latest_candidate_first_user_override',
  previous_active_preserved: true,
  previous_active_expected_version: 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe',
  val_test_results_are_diagnostic_not_hidden: true,
  visual_generation_input: false
};
await fs.writeFile(activationCandidateFile, JSON.stringify(activated, null, 2) + '\n', 'utf8');

const result = await activatePreservingPrevious({
  candidateFile: activationCandidateFile,
  activeFile,
  historyFile,
  validateCandidate(model) {
    if (model.status !== 'active_latest_user_requested_v9_pure_3d') throw new Error('v9 激活副本状态不正确');
    if (model.validation_gate?.activation_override !== true) throw new Error('缺少用户激活覆盖审计');
  },
  metadata: {
    gate: 'user_requested_latest_model_override',
    quality_gate: 'preserved_as_diagnostic',
    fixed_label_contract: true,
    generation_input: 'pure_3d',
    five_view_role: 'evaluation_safety_optimization_only',
    local_preference_scorer: 'qwen3-vl:4b-instruct'
  }
});
console.log(JSON.stringify({ activated: result.candidate.version, status: result.candidate.status, validation_gate: result.candidate.validation_gate, activation: result.record }, null, 2));
