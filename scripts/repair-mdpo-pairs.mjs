import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { selectMdpoPairs } from '../lib/mdpo-collection.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.join(root, 'experiments', 'mdpo');
const pairFile = path.join(directory, 'train_pairs.json'), candidateFile = path.join(directory, 'train_candidates.jsonl');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const dataset = JSON.parse(await fs.readFile(pairFile, 'utf8'));
const candidateRows = (await fs.readFile(candidateFile, 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
const grouped = new Map();
for (const candidate of candidateRows) {
  const key = `${candidate.category}/${candidate.sample_id}`;
  if (!grouped.has(key)) grouped.set(key, []);
  if (!grouped.get(key).some((row) => row.candidate_id === candidate.candidate_id)) grouped.get(key).push(candidate);
}
const repaired = [], untouched = [];
for (const [key, candidates] of grouped) {
  const existing = dataset.pairs.filter((pair) => `${pair.category}/${pair.sample_id}` === key);
  if (existing.length >= 8) { untouched.push(...existing); continue; }
  const selected = selectMdpoPairs(candidates, { count: 8 });
  const template = existing[0];
  if (!template) throw new Error(`${key} has candidates but no pair provenance template`);
  repaired.push({ key, before: existing.length, after: selected.length });
  untouched.push(...selected.map((entry) => ({
    category: template.category, sample_id: template.sample_id, split: 'train', clean_obj_source: template.clean_obj_source,
    scorer_model: template.scorer_model, prompt_version: template.prompt_version, reference_model_sha256: template.reference_model_sha256,
    run_id: template.run_id, ...entry
  })));
}
const output = { ...dataset, pairs: untouched };
const audit = validateMdpoDataset(output, manifest, { requireComplete: false });
const backup = path.join(directory, `train_pairs.before_repair_${Date.now()}.json`);
await fs.copyFile(pairFile, backup, fs.constants.COPYFILE_EXCL);
const temporary = path.join(directory, `train_pairs.repair_${Date.now()}.tmp`);
await fs.writeFile(temporary, JSON.stringify(output, null, 2) + '\n', { flag: 'wx' });
await fs.rename(temporary, pairFile);
await fs.writeFile(path.join(directory, 'dataset_audit.json'), JSON.stringify({ ...audit, reference_model_sha256: output.reference_model_sha256, complete: false, repaired, updated_at: new Date().toISOString() }, null, 2) + '\n');
console.log(JSON.stringify({ repaired, backup: path.relative(root, backup).split(path.sep).join('/'), audit }, null, 2));
