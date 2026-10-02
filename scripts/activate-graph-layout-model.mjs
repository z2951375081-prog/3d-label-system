import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { activatePreservingPrevious } from '../lib/model-preservation.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const candidateFile = path.join(experiments, 'layout_model_v8_cv_clean_candidate.json');
const activeFile = path.join(experiments, 'layout_model.json');
const result = await activatePreservingPrevious({
  candidateFile,
  activeFile,
  historyFile: path.join(experiments, 'layout_model_activation_history.jsonl'),
  validateCandidate(candidate) {
    const architecture = candidate?.architecture || {};
    if (candidate?.version !== 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe') throw new Error('候选不是 v8 DGCNN-CNN-FNN-图 Transformer-MoE 模型');
    if (architecture.type !== 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe' || architecture.node_input_dim !== 51 || architecture.obj_surface_points !== 1024 || architecture.dgcnn_edgeconv_layers < 2 || architecture.geometry_feature_dim !== 64 || architecture.visual_feature_dim !== 32 || architecture.fused_feature_dim !== 160 || architecture.edge_input_dim !== 13 || architecture.pre_gnn_fnn_layers !== 1 || architecture.pre_gnn_fnn_hidden_dim !== 128 || architecture.message_passing_layers !== 2 || architecture.transformer_layers !== 1 || ![4, 5].includes(architecture.moe_expert_count)) throw new Error('v8 CV 融合图模型结构校验失败');
    if (architecture.input_provenance !== 'manual_contract_clean_obj_and_unlabeled_render_without_adjusted_center_or_box_size' || architecture.visual_input_source !== 'unlabeled_five_view_depth_rasters_rendered_from_clean_obj') throw new Error('候选输入来源不安全');
    if (architecture.supervised_target !== 'deterministic_five_view_safe_teacher') throw new Error('候选监督目标不是确定性安全教师');
    if (candidate?.network?.message_layers?.length !== architecture.message_passing_layers) throw new Error('消息传递层权重不完整');
    if (candidate?.network?.pre_gnn_fnn_layers?.length !== architecture.pre_gnn_fnn_layers) throw new Error('GNN 前置 FNN 权重不完整');
    if (candidate?.network?.transformer_layers?.length !== architecture.transformer_layers) throw new Error('Transformer 层权重不完整');
    if (candidate?.network?.moe?.experts?.length !== architecture.moe_expert_count || candidate?.network?.moe?.router?.weights?.length !== architecture.moe_expert_count) throw new Error('MoE 路由器或专家权重不完整');
    if (candidate?.network?.fusion?.weights?.length !== 64 || candidate?.network?.fusion?.weights?.[0]?.length !== 160) throw new Error('160D 特征融合权重不完整');
    if (!(candidate?.training?.parameter_updates?.pre_gnn_fnn?.changed_parameters > 0) || !(candidate?.training?.parameter_updates?.transformer?.changed_parameters > 0) || !(candidate?.training?.parameter_updates?.moe_router?.changed_parameters > 0)) throw new Error('前置 FNN、Transformer 或 MoE 路由器没有实际训练更新');
    if (!(candidate?.training?.functional_evidence?.pre_gnn_fnn_ablation?.max_abs_output_change > 1e-8)) throw new Error('前置 FNN 消融未证明其对验证输出有实际影响');
    if (!(candidate?.training?.parameter_updates?.feature_fusion?.changed_parameters > 0) || !(candidate?.training?.functional_evidence?.cv_fusion_ablation?.max_abs_output_change > 1e-8)) throw new Error('CV 融合分支没有实际训练更新或功能影响');
    if (candidate?.validation_gate?.status !== 'accepted') throw new Error('候选尚未通过 val 门控');
  },
  metadata: { gate: 'dgcnn_cnn_fnn_graph_transformer_moe_replacement_multidimensional_validation', fixed_label_contract: true, local_preference_scorer: 'qwen3-vl:4b-instruct' }
});
console.log(JSON.stringify({ activated: result.candidate.version, architecture: result.candidate.architecture, inference: result.candidate.inference, validation_gate: result.candidate.validation_gate, activation: result.record }, null, 2));
