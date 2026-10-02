// After the single frozen test evaluation, make the val-selected, already
// accepted checkpoint the served reward model as well. Test metrics never
// choose a checkpoint. The active model is backed up before any replacement.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experimentalDirIndex = process.argv.indexOf('--experimentsDir');
const exp = experimentalDirIndex >= 0 && process.argv[experimentalDirIndex + 1]
  ? path.resolve(process.argv[experimentalDirIndex + 1]) : path.join(root, 'experiments');
const runId = process.argv[process.argv.indexOf('--runId') + 1];
if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId || '')) throw new Error('必须指定合法的 --runId');
const read = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const sha = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const ledger = await read(path.join(exp, `llm_preference_checkpoints_${runId}.json`));
const test = await read(path.join(exp, 'preference_test11_report.json'));
const convergence = await read(path.join(exp, 'llm_preference_convergence.json'));
if ([ledger.run_id, test.run_id, convergence.run_id].some((id) => id !== runId) ||
    !convergence.evidence_sufficient || test.test_samples !== 11 || test.rows?.length !== 11 ||
    !test.candidate_activated || !String(test.selection_policy).startsWith('best_accepted_val_visual_aesthetic'))
  throw new Error('必须有同一运行的完整视觉 val 选择和冻结 test11；不允许用 test11 选模');
const selected = ledger.checkpoints?.find((item) => Number(item.round) === Number(test.selected_round));
if (!selected || !selected.activated || selected.validation_status !== 'accepted') throw new Error('所选轮次不是 val 已接受并激活的检查点');
const source = path.join(exp, 'llm_preference_checkpoints', runId, `round_${selected.round}.json`);
const selectedHash = await sha(source);
if (selectedHash !== selected.reward_model_sha256 || selectedHash !== test.candidate_model_sha256)
  throw new Error('冻结模型、检查点清单和 test11 结果哈希不一致');
const selectedModel = await read(source);
if (selectedModel.training?.run_id_filter !== runId || selectedModel.validation_gate?.status !== 'accepted')
  throw new Error('冻结模型不是本次运行已经通过 val 的奖励权重');
const activeFile = path.join(exp, 'preference_model.json');
const active = await read(activeFile);
if (active.training?.run_id_filter !== runId || active.validation_gate?.status !== 'accepted')
  throw new Error('活动模型不属于本次运行或没有通过 val');
const originalHash = await sha(activeFile);
let backup = null;
if (originalHash !== selectedHash) {
  backup = path.join(exp, `preference_model_before_final_val_selection_${runId}_${Date.now()}.json`);
  await fs.copyFile(activeFile, backup, fs.constants.COPYFILE_EXCL);
  if (await sha(backup) !== originalHash) throw new Error('活动奖励备份哈希校验失败');
  try {
    await fs.copyFile(source, activeFile);
    if (await sha(activeFile) !== selectedHash) throw new Error('所选检查点激活后的哈希校验失败');
  } catch (error) {
    await fs.copyFile(backup, activeFile);
    throw error;
  }
}
const receipt = {
  version: 'val_selected_reward_activation_v1', run_id: runId, selected_round: Number(selected.round),
  selection_policy: test.selection_policy, frozen_model_sha256: selectedHash, previous_active_sha256: originalHash,
  backup_file: backup ? path.relative(root, backup).split(path.sep).join('/') : null,
  changed: Boolean(backup), test_used_for_selection: false, generated_at: new Date().toISOString()
};
await fs.writeFile(path.join(exp, `final_reward_activation_${runId}.json`), `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(receipt));
