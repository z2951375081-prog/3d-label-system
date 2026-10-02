import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { writeImmutableFileBundle, writeImmutableJsonBundle } from '../lib/immutable-json-bundle.mjs';

const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-immutable-bundle-'));
try {
  const first = path.join(temporary, 'first.json'), second = path.join(temporary, 'second.json');
  const result = await writeImmutableJsonBundle([{ file: first, value: { id: 1 } }, { file: second, value: { id: 2 } }]);
  assert.equal(result.length, 2);
  assert.equal(JSON.parse(await fs.readFile(first, 'utf8')).id, 1);
  assert.equal(JSON.parse(await fs.readFile(second, 'utf8')).id, 2);
  await assert.rejects(writeImmutableJsonBundle([{ file: first, value: { id: 3 } }, { file: path.join(temporary, 'third.json'), value: { id: 3 } }]), /already exists/);
  assert.equal(JSON.parse(await fs.readFile(first, 'utf8')).id, 1);
  await assert.rejects(writeImmutableJsonBundle([{ file: path.join(temporary, 'single.json'), value: {} }]), /at least two/);
  const jsonFile = path.join(temporary, 'metrics.json'), csvFile = path.join(temporary, 'metrics.csv');
  await writeImmutableFileBundle([{ file: jsonFile, text: '{"ok":true}\n' }, { file: csvFile, text: 'metric,value\nok,1\n' }]);
  assert.match(await fs.readFile(csvFile, 'utf8'), /metric,value/);
  assert.equal((await fs.readdir(temporary)).some((name) => name.includes('.pending-')), false);
  console.log('Immutable two-file JSON bundle publication and no-overwrite tests passed.');
} finally {
  const resolved = path.resolve(temporary), prefix = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('v10-mdpo-immutable-bundle-')) throw new Error('Unsafe immutable bundle test cleanup target');
  await fs.rm(resolved, { recursive: true, force: true });
}
