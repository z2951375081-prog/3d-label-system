import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { activatePreservingPrevious, sha256File } from '../lib/model-preservation.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const temporary = await fs.mkdtemp(path.join(experiments, '.model-preservation-'));

try {
  const activeFile = path.join(temporary, 'layout_model.json');
  const candidateFile = path.join(temporary, 'layout_model_candidate.json');
  const historyFile = path.join(temporary, 'activation_history.jsonl');
  const previous = { version: 'layout_model_v2_multiview', value: 'previous' };
  const candidate = { version: 'layout_model_v3_multiview_projection_loss', validation_gate: { status: 'accepted' }, value: 'candidate' };
  await fs.writeFile(activeFile, JSON.stringify(previous));
  await fs.writeFile(candidateFile, JSON.stringify(candidate));
  const previousHash = await sha256File(activeFile);

  const activated = await activatePreservingPrevious({
    candidateFile,
    activeFile,
    historyFile,
    validateCandidate(model) { assert.equal(model.validation_gate.status, 'accepted'); }
  });
  assert.equal(JSON.parse(await fs.readFile(activeFile, 'utf8')).value, 'candidate');
  assert.equal(JSON.parse(await fs.readFile(activated.record.backup_file, 'utf8')).value, 'previous');
  assert.equal(await sha256File(activated.record.backup_file), previousHash);
  assert.equal((await fs.readFile(historyFile, 'utf8')).trim().split(/\r?\n/).length, 1);

  const rollbackCandidateFile = path.join(temporary, 'rollback_candidate.json');
  await fs.writeFile(rollbackCandidateFile, JSON.stringify({ version: 'rollback_candidate', validation_gate: { status: 'accepted' }, value: 'must_rollback' }));
  const beforeRollbackHash = await sha256File(activeFile);
  let cleanupCalled = false;
  await assert.rejects(activatePreservingPrevious({
    candidateFile: rollbackCandidateFile,
    activeFile,
    historyFile,
    validateCandidate(model) { assert.equal(model.validation_gate.status, 'accepted'); },
    async afterActivate() { throw new Error('injected activation report failure'); },
    async rollbackAfterFailure() { cleanupCalled = true; }
  }), /injected activation report failure/);
  assert.equal(cleanupCalled, true);
  assert.equal(await sha256File(activeFile), beforeRollbackHash);
  assert.equal(JSON.parse(await fs.readFile(activeFile, 'utf8')).value, 'candidate');
  assert.equal((await fs.readFile(historyFile, 'utf8')).trim().split(/\r?\n/).length, 1);

  const protectedOutput = path.join(temporary, 'layout_model.json');
  const refused = spawnSync(process.execPath, ['scripts/train-layout-model.mjs', '--epochs', '0', '--output', protectedOutput], { cwd: root, encoding: 'utf8', windowsHide: true });
  assert.notEqual(refused.status, 0);
  assert.match(refused.stderr, /拒绝直接覆盖/);
  assert.equal(JSON.parse(await fs.readFile(activeFile, 'utf8')).value, 'candidate');
  console.log('Model preservation passed: candidate gate, verified backup/hash/history, and direct active-model overwrite refusal.');
} finally {
  const resolved = path.resolve(temporary), prefix = path.resolve(experiments) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('.model-preservation-')) throw new Error('Refusing unsafe model preservation cleanup target');
  await fs.rm(resolved, { recursive: true, force: true });
}
