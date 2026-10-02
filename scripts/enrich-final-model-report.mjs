import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const runIndex = process.argv.indexOf('--runId');
const runId = runIndex >= 0 ? String(process.argv[runIndex + 1] || '') : '';
if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('报告增强器需要有效 --runId');
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const exists = async (file) => fs.stat(file).then((item) => item.isFile(), () => false);
const number = (value, digits = 6) => value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const marker = '<!-- adaptive-directional-and-llm-final-model-v1 -->';
const sha256 = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const started = Date.now();
while (Date.now() - started < 18 * 60 * 60 * 1000) {
  const latest = path.join(experiments, '最终模型完整说明报告_latest.md');
  const testFile = path.join(experiments, 'preference_test11_report.json');
  const convergenceFile = path.join(experiments, 'llm_preference_convergence.json');
  if (await exists(latest) && await exists(testFile) && await exists(convergenceFile)) {
    const test = await readJson(testFile);
    const convergence = await readJson(convergenceFile);
    if (test.run_id === runId && convergence.run_id === runId && convergence.evidence_sufficient && test.candidate_activated) {
      const directional = await readJson(path.join(experiments, 'adaptive_directional_rerank_offline.json'));
      let text = await fs.readFile(latest, 'utf8');
      if (!text.includes(marker)) {
        const metrics = ['multidimensional_quality_score', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'mean_anchor_distance', 'leader_length_compliance_ratio', 'directional_allocation_mismatch', 'directional_uniformity', 'manual_style_distance'];
        const lines = [marker, '', '## 模型一与最终模型（LLM 创新点）', '',
          '- 模型一：活动基础布局 MLP、seed17、方向权重 1.6/1.2，不使用 LLM 奖励。',
          '- 最终模型：对每个候选先执行 val11 选定的方向自由空间安全回退，再经过确定性几何安全门控，最后由真实 LLM 五维美学奖励重排。',
          '- LLM 是最终模型的创新点：学习构图和谐、视觉层级、空间平衡、整体美感及少量人工风格；穿模、遮挡、文字、引导线、越界与方向密度仍由基础模型和确定性约束负责。', '',
          '| test11 指标 | 模型一 | 最终模型 | 变化 |', '|---|---:|---:|---:|',
          ...metrics.map((field) => { const base = test.summary?.[field]?.baseline, final = test.summary?.[field]?.preferred; const delta = Number.isFinite(Number(base)) && Number.isFinite(Number(final)) ? Number(final) - Number(base) : null; return `| ${field} | ${number(base)} | ${number(final)} | ${number(delta)} |`; }), '',
          '### 方向均匀度增强的冻结证据', '',
          `- val11：接纳 ${directional.validation.summary.accepted_count}/${directional.validation.summary.sample_count} 个样本；方向均匀度 ${number(directional.validation.summary.directional_uniformity.baseline)} → ${number(directional.validation.summary.directional_uniformity.selected)}；方向失配 ${number(directional.validation.summary.directional_allocation_mismatch.baseline)} → ${number(directional.validation.summary.directional_allocation_mismatch.selected)}。`,
          `- test11（仅冻结确认）：接纳 ${directional.test_confirmation.summary.accepted_count}/${directional.test_confirmation.summary.sample_count} 个样本；方向均匀度 ${number(directional.test_confirmation.summary.directional_uniformity.baseline)} → ${number(directional.test_confirmation.summary.directional_uniformity.selected)}；方向失配 ${number(directional.test_confirmation.summary.directional_allocation_mismatch.baseline)} → ${number(directional.test_confirmation.summary.directional_allocation_mismatch.selected)}。`,
          '- 增强候选搜索权重为 4.8/3.2；只有均匀度至少 +0.005、自由空间失配至少 -0.005，且穿模、网格相交、遮挡、文字、越界、引导线合规和候选内在质量通过逐样本门控时才采用，否则回退到 1.6/1.2。',
          '- 决策不读取人工调整后的中心或风格距离；人工布局只参与最终评估。', ''
        ];
        const section = `${lines.join('\n')}\n`;
        text = `${text.trimEnd()}\n\n${section}`;
        await fs.writeFile(latest, text, 'utf8');
        await fs.writeFile(path.join(experiments, `最终模型完整说明报告_${runId}.md`), text, 'utf8');
        const llmLatest = path.join(experiments, '最终LLM美学偏好实验报告_latest.md');
        if (await exists(llmLatest)) {
          const llmText = await fs.readFile(llmLatest, 'utf8');
          if (!llmText.includes(marker)) await fs.writeFile(llmLatest, `${llmText.trimEnd()}\n\n${section}`, 'utf8');
        }
        const layoutFile = path.join(experiments, 'layout_model.json');
        const preferenceFile = path.join(experiments, 'preference_model.json');
        const layout = await readJson(layoutFile);
        const preference = await readJson(preferenceFile);
        if (preference.training?.run_id_filter !== runId || preference.validation_gate?.status !== 'accepted') throw new Error('活动 LLM 奖励模型未绑定本次运行或未通过 val');
        const manifest = {
          version: 'final_model_manifest_v1', generated_at: new Date().toISOString(), run_id: runId,
          pipeline: ['fixed_label_layout_mlp', 'leader_length_train_prior', 'adaptive_directional_free_space_gate', 'five_view_geometry_safety_gate', 'llm_aesthetic_reward_rerank'],
          model_1: { file: 'experiments/layout_model.json', version: layout.version, sha256: await sha256(layoutFile), architecture: layout.architecture, inference: layout.inference },
          final_model: {
            base_layout_model_sha256: await sha256(layoutFile),
            adaptive_directional_gate: { base_weights: directional.original_weights, challenger_weights: directional.challenger_weights, rule: directional.selection.rule, validation: directional.validation.summary, frozen_test_confirmation: directional.test_confirmation.summary },
            llm_reward: { file: 'experiments/preference_model.json', sha256: await sha256(preferenceFile), architecture: preference.architecture, training: { run_id_filter: preference.training.run_id_filter, examples: preference.training.examples, epochs: preference.training.epochs, learning_rate: preference.training.learning_rate, seed: preference.training.seed }, validation_gate: preference.validation_gate },
            test11: { selected_round: test.selected_round, protocol: test.protocol, summary: test.summary }
          },
          invariants: { fixed_label_contract: true, manual_adjusted_coordinates_used_for_direction_selection: false, test_used_for_training_or_selection: false, api_key_persisted: false }
        };
        const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
        await fs.writeFile(path.join(experiments, `final_model_manifest_${runId}.json`), manifestText, 'utf8');
        await fs.writeFile(path.join(experiments, 'final_model_manifest_latest.json'), manifestText, 'utf8');
        console.log(`最终报告已补充模型一/最终模型和方向自适应证据：${runId}`);
      }
      process.exit(0);
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 5000));
}
throw new Error('等待最终报告超过 18 小时，未修改任何报告');
