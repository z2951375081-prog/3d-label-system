# v10 基础布局与本地 Qwen 安全偏好实验：5d3aabe9-2e20-455c-a9a9-43b7db28cd57

生成时间：2026-09-21T12:29:21.350Z。运行结论：基础 v10 模型已激活；本次 8 轮 Qwen 奖励候选训练完成，但 **val11 美学门控拒绝激活**。活动奖励模型仍来自 run 15eb60ec-bcc3-4428-88ed-408ba5d245ce。

## 模型与安全边界

- 基础布局：纯三维 clean OBJ 的 DGCNN EdgeConv 64D 几何、锚点局部 PCA 帧、51D 标签特征、51→64→128→64 FNN、18D anchor-label/10D label-label 异构关系、2 层关系 GNN、1 层全局 Transformer、4 专家人类风格 MoE、64→32→6 局部坐标解码。模型 80 epoch 训练，已通过 val11 基础模型激活门控。
- 五视角主视角 0.40、其余各 0.15；worst-view 与 CVaR 控制遮挡和引导线风险，确定性修复确保可交付候选的五视角精确零交叉；字体清晰度和引导线方向均计入基础训练及门控。Qwen 仅用五维美学评分重排已安全候选，不生成初始三维坐标。
- 基础模型 SHA-256：3D4AAD72F9EAD8CD29D61B65D5AC4DA4368FE7A2CC1C4DE01046516C981A8A29；本次拒绝的奖励候选 SHA-256：FD51CCA2F880016649F3D2864F2958236E664A8847D8F29DAA47BEE2C13DDD30；保留的活动奖励 SHA-256：62F0EC9F576CB1DBEB1832052619F57815A87C9D55ECE631437956F98D18A1BD。

## 真实六图 val11 美学评分与检查点

| 轮次 | 美学综合分（1–5，越高越好） | 冻结 SHA-256 验证 |
|---:|---:|---|
| 0 | 4.3515 | 通过 |
| 1 | 4.2279 | 通过 |
| 2 | 4.2515 | 通过 |
| 4 | 4.2006 | 通过 |
| 8 | 4.2415 | 通过 |

0→8 轮完整 99 条 val 样本记录；所有主视角和最差视角精确交叉数均为 0，穿模与网格表面相交率均为 0。train 共 60 条偏好，chosen/rejected 双侧精确零交叉。
基线 4.3515；训练后最佳 4.2515（第 2 轮）；第 4→8 轮变化 0.0409。完整证据=true，先提升后稳定=false。不能用“趋于稳定”代替“达到基线以上”。

## test11 仅作最终诊断，不用于选择或激活

本次仅诊断的检查点为第 7 轮；选择策略 best_rejected_val_visual_aesthetic_diagnostic_only；活动状态：未激活。几何指标改善不构成 val 美学通过证据。

| test11 指标 | 基线 | 拒绝的候选（仅诊断） |
|---|---:|---:|
| multidimensional_quality_score | 4.4735 | 4.5275 |
| objective_score | 1.0057 | 0.9557 |
| text_clarity | 4.9011 | 4.9181 |
| readability | 4.7165 | 4.7521 |
| olr | 0.0214 | 0.0144 |
| label_label_occlusion_ratio | 0.0214 | 0.0144 |
| label_object_occlusion_ratio | 0.0093 | 0.0055 |
| object_penetration_ratio | 0.0006 | 0.0004 |
| mesh_surface_intersection_ratio | 0.0174 | 0.0020 |
| leader_crossings | 0.0000 | 0.0000 |
| worst_view_leader_crossing_count | 不可比/未记录 | 0.0000 |
| directional_allocation_mismatch | 0.4209 | 0.4329 |

注意：旧基线没有记录 worst-view 精确交叉数，表中的“未记录”不能当作 0；部分 test11 样本仍存在非零网格相交率，不能宣称 test11 全部零穿模。

## 可视化与固定标签契约

无头六图可视化通过 3 类 train 样本，每类完整 6 张含文字视图；11 张 v10 指标卡均有真实数值。截图：`experiments/final_v10_visual_5d3aabe9-2e20-455c-a9a9-43b7db28cd57/headless_six_view_check.png`；指标：`experiments/final_v10_visual_5d3aabe9-2e20-455c-a9a9-43b7db28cd57/headless_six_view_check.json`。
全部 55 个样本、406 个标签已按候选→模型推理→确定性优化三个阶段核验标签 ID、原文、锚点 XYZ、sourceObjs 与 targetGroups 恒定。

原始证据：`experiments/llm_visual_validation_5d3aabe9-2e20-455c-a9a9-43b7db28cd57.json`、`experiments/llm_preference_checkpoints_5d3aabe9-2e20-455c-a9a9-43b7db28cd57.json`、`experiments/llm_preference_convergence_5d3aabe9-2e20-455c-a9a9-43b7db28cd57.json`、`experiments/preference_test11_report.json`。旧无效 run 的偏好未混入本 run。
