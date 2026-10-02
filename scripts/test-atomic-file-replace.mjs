import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { replaceFileWithRetry } from '../lib/atomic-file-replace.mjs';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'mdpo-atomic-replace-'));
try {
  const target = path.join(directory, 'train_pairs.json');
  const temporary = path.join(directory, `train_pairs.${randomUUID()}.tmp`);
  await fs.writeFile(target, 'old-dataset\n');
  await fs.writeFile(temporary, 'new-dataset\n', { flag: 'wx' });

  let transientAttempts = 0;
  const transientOperations = {
    rename: async (...args) => {
      transientAttempts += 1;
      if (transientAttempts < 3) {
        const error = new Error('simulated Windows scanner lock');
        error.code = 'EPERM';
        throw error;
      }
      return fs.rename(...args);
    },
    rm: (...args) => fs.rm(...args)
  };
  const recovered = await replaceFileWithRetry(temporary, target, {
    operations: transientOperations, maxAttempts: 5, initialDelayMs: 0, maxDelayMs: 0, sleep: async () => {}
  });
  assert.equal(recovered.attempts, 3);
  assert.equal(transientAttempts, 3);
  assert.equal(await fs.readFile(target, 'utf8'), 'new-dataset\n');
  await assert.rejects(fs.access(temporary), { code: 'ENOENT' });

  const preservedTarget = path.join(directory, 'preserved_train_pairs.json');
  const failedTemporary = path.join(directory, `train_pairs.${randomUUID()}.tmp`);
  await fs.writeFile(preservedTarget, 'known-good-dataset\n');
  await fs.writeFile(failedTemporary, 'uncommitted-dataset\n', { flag: 'wx' });
  let permanentAttempts = 0;
  const permanentlyLockedOperations = {
    rename: async () => {
      permanentAttempts += 1;
      const error = new Error('simulated permanent Windows lock');
      error.code = 'EPERM';
      throw error;
    },
    rm: (...args) => fs.rm(...args)
  };
  await assert.rejects(
    replaceFileWithRetry(failedTemporary, preservedTarget, {
      operations: permanentlyLockedOperations, maxAttempts: 4, initialDelayMs: 0, maxDelayMs: 0, sleep: async () => {}
    }),
    (error) => error?.code === 'EPERM' && /owned temporary file removed/.test(error.message)
  );
  assert.equal(permanentAttempts, 4);
  assert.equal(await fs.readFile(preservedTarget, 'utf8'), 'known-good-dataset\n');
  await assert.rejects(fs.access(failedTemporary), { code: 'ENOENT' });

  console.log('Windows atomic replacement retry and old-dataset preservation tests passed.');
} finally {
  await fs.rm(directory, { recursive: true, force: true });
}
