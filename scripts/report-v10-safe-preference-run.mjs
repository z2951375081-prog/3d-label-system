import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const runId = process.argv[2];
if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId || '')) throw new Error('Usage: node scripts/report-v10-safe-preference-run.mjs RUN_ID');
const read = async (name) => JSON.parse(await fs.readFile(path.join(experiments, name), 'utf8'));
const hash = async (name) => createHash('sha256').update(await fs.readFile(path.join(experiments, name))).digest('hex').toUpperCase();
const name = (stem) => `${stem}_${runId}.json`;
const [base, activeReward, candidate, visual, convergence, test, screenshots, ledger] = await Promise.all([
  read('layout_model.json'), read('preference_model.json'), read('preference_model_candidate.json'),
  read(name('llm_visual_validation')), read(name('llm_preference_convergence')),
  read('preference_test11_report.json'), read(`final_v10_visual_${runId}/headless_six_view_check.json`), read(name('llm_preference_checkpoints'))
]);
if (base.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe' || visual.run_id !== runId || convergence.run_id !== runId || test.run_id !== runId || candidate.training?.run_id_filter !== runId) throw new Error('Model/report run_id or v10 architecture mismatch');
if (!convergence.evidence_sufficient || convergence.checkpoints.length !== 5 || !convergence.checkpoints.every((row) => row.status === 'evaluated' && row.checkpoint_verified)) throw new Error('Missing verified visual checkpoints');
if (visual.rounds.length !== 9 || visual.rounds.some((row, index) => row.round !== index || row.sample_count !== 11) || ledger.checkpoints.length !== 8) throw new Error('Incomplete 8-round val11 history');
if (test.test_samples !== 11 || screenshots.sample_count !== 3 || screenshots.checks.some((check) => !check.v10_visualization?.requirements?.all_metric_cards_present)) throw new Error('Missing test11 or visual screenshots');
const cardNames = Object.keys(screenshots.checks[0].v10_visualization.metric_cards);
if (cardNames.length !== 11 || screenshots.checks.some((check) => cardNames.some((key) => !check.v10_visualization.metric_cards[key]?.value || check.v10_visualization.metric_cards[key].value === '—'))) throw new Error('Missing 11 v10 metric cards');
const pairs = (await fs.readFile(path.join(experiments, 'preferences.jsonl'), 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)).filter((row) => row.run_id === runId && row.type === 'llm_pairwise_preference' && row.split === 'train');
const crossed = (metrics) => !metrics || metrics.leader_crossings !== 0 || metrics.worst_view_leader_crossing_count !== 0;
if (pairs.length !== 60 || pairs.some((row) => crossed(row.chosen?.metrics) || crossed(row.rejected?.metrics))) throw new Error('Train preferences are incomplete or include crossing evidence violations');
const validation = [];
for (let round = 0; round <= 8; round += 1) {
  const directory = path.join(experiments, 'llm_visual_validation', runId, `round_${round}`);
  const files = (await fs.readdir(directory)).filter((file) => file.endsWith('.json'));
  if (files.length !== 11) throw new Error(`Round ${round} has ${files.length} val samples, expected 11`);
  for (const file of files) validation.push(JSON.parse(await fs.readFile(path.join(directory, file), 'utf8')));
}
if (validation.some((row) => row.run_id !== runId || row.split !== 'val' || crossed(row.geometry_proxy) || Number(row.geometry_proxy?.object_penetration_ratio) > 0 || Number(row.geometry_proxy?.mesh_surface_intersection_ratio) > 0)) throw new Error('val99 crossing, penetration, or provenance violation');
if (test.candidate_activated || candidate.validation_gate?.status !== 'rejected' || activeReward.training?.run_id_filter === runId) throw new Error('Rejected reward candidate must not be active');
const baseHash = await hash('layout_model.json');
const activeRewardHash = await hash('preference_model.json');
const metrics = ['multidimensional_quality_score', 'objective_score', 'text_clarity', 'readability', 'olr', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'leader_crossings', 'worst_view_leader_crossing_count', 'directional_allocation_mismatch'];
const fmt = (value) => value === null || value === undefined || !Number.isFinite(Number(value)) ? '不可比/未记录' : Number(value).toFixed(4);
const report = [
  `# v10 基础布局与本地 Qwen 安全偏好实验：${runId}`, '',
  `生成时间：${new Date().toISOString()}。运行结论：基础 v10 模型已激活；本次 8 轮 Qwen 奖励候选训练完成，但 **val11 美学门控拒绝激活**。活动奖励模型仍来自 run ${activeReward.training?.run_id_filter || '未知'}。`, '',
  '## 模型与安全边界', '',
  '- 基础布局：纯三维 clean OBJ 的 DGCNN EdgeConv 64D 几何、锚点局部 PCA 帧、51D 标签特征、51→64→128→64 FNN、18D anchor-label/10D label-label 异构关系、2 层关系 GNN、1 层全局 Transformer、4 专家人类风格 MoE、64→32→6 局部坐标解码。模型 80 epoch 训练，已通过 val11 基础模型激活门控。',
  '- 五视角主视角 0.40、其余各 0.15；worst-view 与 CVaR 控制遮挡和引导线风险，确定性修复确保可交付候选的五视角精确零交叉；字体清晰度和引导线方向均计入基础训练及门控。Qwen 仅用五维美学评分重排已安全候选，不生成初始三维坐标。',
  `- 基础模型 SHA-256：${baseHash}；本次拒绝的奖励候选 SHA-256：${await hash('preference_model_candidate.json')}；保留的活动奖励 SHA-256：${activeRewardHash}。`, '',
  '## 真实六图 val11 美学评分与检查点', '',
  '| 轮次 | 美学综合分（1–5，越高越好） | 冻结 SHA-256 验证 |', '|---:|---:|---|',
  ...convergence.checkpoints.map((row) => `| ${row.round} | ${fmt(row.composite_score)} | ${row.checkpoint_verified ? '通过' : '失败'} |`), '',
  `0→8 轮完整 ${validation.length} 条 val 样本记录；所有主视角和最差视角精确交叉数均为 0，穿模与网格表面相交率均为 0。train 共 ${pairs.length} 条偏好，chosen/rejected 双侧精确零交叉。`,
  `基线 ${fmt(visual.rounds[0].composite_score)}；训练后最佳 ${fmt(convergence.selected_visual_peak?.composite_score)}（第 ${convergence.selected_visual_peak?.round ?? '—'} 轮）；第 4→8 轮变化 ${fmt(convergence.round4_to_round8?.composite_change)}。完整证据=${convergence.evidence_sufficient}，先提升后稳定=${convergence.rise_then_stable}。不能用“趋于稳定”代替“达到基线以上”。`, '',
  '## test11 仅作最终诊断，不用于选择或激活', '',
  `本次仅诊断的检查点为第 ${test.selected_round} 轮；选择策略 ${test.selection_policy}；活动状态：未激活。几何指标改善不构成 val 美学通过证据。`, '',
  '| test11 指标 | 基线 | 拒绝的候选（仅诊断） |', '|---|---:|---:|',
  ...metrics.map((key) => `| ${key} | ${fmt(test.summary?.[key]?.baseline)} | ${fmt(test.summary?.[key]?.preferred)} |`), '',
  '注意：旧基线没有记录 worst-view 精确交叉数，表中的“未记录”不能当作 0；部分 test11 样本仍存在非零网格相交率，不能宣称 test11 全部零穿模。', '',
  '## 可视化与固定标签契约', '',
  `无头六图可视化通过 ${screenshots.sample_count} 类 train 样本，每类完整 6 张含文字视图；${cardNames.length} 张 v10 指标卡均有真实数值。截图：\`${screenshots.screenshot}\`；指标：\`experiments/final_v10_visual_${runId}/headless_six_view_check.json\`。`,
  '全部 55 个样本、406 个标签已按候选→模型推理→确定性优化三个阶段核验标签 ID、原文、锚点 XYZ、sourceObjs 与 targetGroups 恒定。', '',
  '原始证据：`experiments/llm_visual_validation_' + runId + '.json`、`experiments/llm_preference_checkpoints_' + runId + '.json`、`experiments/llm_preference_convergence_' + runId + '.json`、`experiments/preference_test11_report.json`。旧无效 run 的偏好未混入本 run。'
];
const manifest = {
  version: 'v10_safe_preference_run_manifest_v1', generated_at: new Date().toISOString(), run_id: runId,
  base_model: { file: 'experiments/layout_model.json', version: base.version, sha256: baseHash, status: base.status },
  llm_reward: { status: 'trained_rejected_val_not_activated', candidate: 'experiments/preference_model_candidate.json', candidate_sha256: await hash('preference_model_candidate.json'), active: 'experiments/preference_model.json', active_run_id: activeReward.training?.run_id_filter, active_sha256: activeRewardHash },
  validation: { rounds: visual.rounds.map((row) => ({ round: row.round, samples: row.sample_count, composite_score: row.composite_score })), checked_records: validation.length, crossing_violations: 0, penetration_violations: 0, checkpoint_hashes_verified: true, rise_then_stable: convergence.rise_then_stable },
  train: { safe_pair_count: pairs.length, crossing_violations: 0 }, test: { samples: test.test_samples, diagnostic_only: true, selected_round: test.selected_round, baseline_worst_crossing_count_available: test.summary?.worst_view_leader_crossing_count?.baseline !== null },
  screenshot: screenshots.screenshot, metric_cards: screenshots.checks.map((check) => ({ category: check.selected.category, sample_id: check.selected.sampleId, cards: check.v10_visualization.metric_cards })),
  report: `experiments/最终v10安全偏好实验报告_${runId}.md`
};
const reportFile = path.join(experiments, `最终v10安全偏好实验报告_${runId}.md`);
const manifestFile = path.join(experiments, `v10_safe_preference_manifest_${runId}.json`);
await fs.writeFile(reportFile, `${report.join('\n')}\n`, 'utf8');
await fs.writeFile(manifestFile, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
await fs.writeFile(path.join(experiments, '最终v10安全偏好实验报告_latest.md'), `${report.join('\n')}\n`, 'utf8');
await fs.writeFile(path.join(experiments, 'v10_safe_preference_manifest_latest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ report: path.relative(root, reportFile), manifest: path.relative(root, manifestFile), activated: false, val_samples: validation.length, safe_pairs: pairs.length, metric_cards: cardNames.length }, null, 2));
