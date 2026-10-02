import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runId = 'test-final-reward-selection';
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'layout-reward-gate-'));
const write = (name, value) => fs.writeFile(path.join(temp, name), `${JSON.stringify(value, null, 2)}\n`);
const digest = (value) => createHash('sha256').update(`${JSON.stringify(value, null, 2)}\n`).digest('hex').toUpperCase();
const run = () => spawnSync(process.execPath, [path.join(root, 'scripts', 'activate-selected-reward-checkpoint.mjs'), '--runId', runId, '--experimentsDir', temp], { encoding: 'utf8', cwd: root });
try {
  const checkpointDir = path.join(temp, 'llm_preference_checkpoints', runId);
  await fs.mkdir(checkpointDir, { recursive: true });
  const selected = { training: { run_id_filter: runId, examples: 7 }, validation_gate: { status: 'accepted' }, weights: [0.2, 0.5] };
  const lastActive = { training: { run_id_filter: runId, examples: 9 }, validation_gate: { status: 'accepted' }, weights: [0.7, 0.8] };
  await fs.writeFile(path.join(checkpointDir, 'round_2.json'), `${JSON.stringify(selected, null, 2)}\n`);
  await write('preference_model.json', lastActive);
  await write(`llm_preference_checkpoints_${runId}.json`, { run_id: runId, checkpoints: [{ round: 2, activated: true, validation_status: 'accepted', reward_model_sha256: digest(selected) }] });
  await write('llm_preference_convergence.json', { run_id: runId, evidence_sufficient: true });
  const test = { run_id: runId, selected_round: 2, selection_policy: 'best_accepted_val_visual_aesthetic', candidate_activated: true, candidate_model_sha256: digest(selected), test_samples: 11, rows: Array.from({ length: 11 }, () => ({ split: 'test' })) };
  await write('preference_test11_report.json', test);
  let result = run();
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(temp, 'preference_model.json'))), selected);
  let receipt = JSON.parse(await fs.readFile(path.join(temp, `final_reward_activation_${runId}.json`)));
  assert.equal(receipt.frozen_model_sha256, digest(selected));
  assert.equal(receipt.changed, true);
  assert.deepEqual(JSON.parse(await fs.readFile(path.resolve(root, receipt.backup_file))), lastActive);
  result = run();
  assert.equal(result.status, 0, result.stderr);
  receipt = JSON.parse(await fs.readFile(path.join(temp, `final_reward_activation_${runId}.json`)));
  assert.equal(receipt.changed, false);
  assert.equal(receipt.backup_file, null);
  await write('preference_model.json', lastActive);
  await write('preference_test11_report.json', { ...test, candidate_activated: false });
  result = run();
  assert.notEqual(result.status, 0, 'rejected candidate must not activate');
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(temp, 'preference_model.json'))), lastActive);
  await write('preference_test11_report.json', { ...test, candidate_model_sha256: digest(lastActive) });
  result = run();
  assert.notEqual(result.status, 0, 'tampered test/frozen checkpoint hash must not activate');
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(temp, 'preference_model.json'))), lastActive);
  console.log('Final reward selection passed: val-selected frozen weight = served weight; backups and negative gates verified.');
} finally {
  const target = path.resolve(temp);
  if (path.dirname(target) !== path.resolve(os.tmpdir()) || !path.basename(target).startsWith('layout-reward-gate-'))
    throw new Error('拒绝清理不属于本次测试的临时目录');
  await fs.rm(target, { recursive: true, force: true });
}
