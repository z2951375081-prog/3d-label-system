import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

export async function writeImmutableFileBundle(entries) {
  if (!Array.isArray(entries) || entries.length < 2) throw new Error('Immutable file bundle requires at least two files');
  const files = entries.map((entry) => path.resolve(String(entry?.file || '')));
  if (new Set(files).size !== files.length || files.some((file) => !path.extname(file))) throw new Error('Immutable file bundle paths must be unique files with extensions');
  const prepared = entries.map(({ bytes, text }, index) => {
    const resolved = files[index], payload = Buffer.isBuffer(bytes) ? bytes : Buffer.from(String(text ?? ''));
    return { file: resolved, bytes: payload, sha256: digest(payload), temporary: path.join(path.dirname(resolved), `.${path.basename(resolved)}.pending-${randomUUID()}`) };
  });
  for (const item of prepared) {
    await fs.mkdir(path.dirname(item.file), { recursive: true });
    try { await fs.access(item.file); throw new Error(`Immutable JSON bundle target already exists: ${item.file}`); }
    catch (error) { if (error?.code !== 'ENOENT') throw error; }
  }
  const committed = [];
  try {
    for (const item of prepared) {
      await fs.writeFile(item.temporary, item.bytes, { flag: 'wx' });
      if (digest(await fs.readFile(item.temporary)) !== item.sha256) throw new Error(`Immutable JSON bundle staging hash mismatch: ${item.file}`);
    }
    // Hard-link publication is atomic and fails if a target appeared after the preflight check.
    for (const item of prepared) {
      await fs.link(item.temporary, item.file);
      if (digest(await fs.readFile(item.file)) !== item.sha256) throw new Error(`Immutable JSON bundle committed hash mismatch: ${item.file}`);
      committed.push(item);
    }
    return prepared.map(({ file, sha256 }) => ({ file, sha256 }));
  } catch (error) {
    for (const item of committed) {
      const bytes = await fs.readFile(item.file).catch(() => null);
      if (bytes && digest(bytes) === item.sha256) await fs.rm(item.file, { force: true });
    }
    throw error;
  } finally {
    for (const item of prepared) await fs.rm(item.temporary, { force: true }).catch(() => {});
  }
}

export async function writeImmutableJsonBundle(entries) {
  if (!Array.isArray(entries) || entries.length < 2) throw new Error('Immutable JSON bundle requires at least two files');
  if (entries.some((entry) => path.extname(String(entry?.file || '')).toLowerCase() !== '.json')) throw new Error('Immutable JSON bundle paths must be JSON files');
  return writeImmutableFileBundle(entries.map(({ file, value }) => ({ file, text: JSON.stringify(value, null, 2) + '\n' })));
}
