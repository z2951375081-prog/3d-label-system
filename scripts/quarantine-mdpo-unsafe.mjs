import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { selectMdpoPairs, strictMdpoSafety } from '../lib/mdpo-collection.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.join(root, 'experiments', 'mdpo');
const datasetFile = path.join(directory, 'train_pairs.json');
const datasetBytes = await fs.readFile(datasetFile);
const dataset = JSON.parse(datasetBytes.toString('utf8'));
const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const source = (await fs.readFile(path.join(directory, 'train_candidates.jsonl'), 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
const runStamp = new Date().toISOString().replace(/[:.]/g, '-');

async function exists(file) {
  try { await fs.access(file); return true; } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function preserveExisting(file, label) {
  if (!await exists(file)) return null;
  const extension = path.extname(file);
  const archive = path.join(path.dirname(file), `${path.basename(file, extension)}.${label}_${runStamp}${extension}`);
  await fs.copyFile(file, archive, fs.constants.COPYFILE_EXCL);
  return archive;
}

const groups = new Map();
for (const candidate of source) {
  const key = `${candidate.category}/${candidate.sample_id}`;
  if (!groups.has(key)) groups.set(key, []);
  if (!groups.get(key).some((row) => row.candidate_id === candidate.candidate_id)) groups.get(key).push(candidate);
}
const quarantine = [], pairs = [];
for (const [key, candidates] of groups) {
  const unsafe = candidates.filter((candidate) => !strictMdpoSafety(candidate.safety?.metrics).eligible);
  const safe = candidates.filter((candidate) => strictMdpoSafety(candidate.safety?.metrics).eligible);
  const previous = dataset.pairs.filter((pair) => `${pair.category}/${pair.sample_id}` === key);
  let reason = null;
  try {
    if (!previous.length) throw new Error('missing original train-only pair provenance');
    const targetPairCount = Math.max(6, Math.min(12, previous.length));
    const selected = selectMdpoPairs(safe, { count: targetPairCount });
    const template = previous[0];
    pairs.push(...selected.map((entry) => ({ category: template.category, sample_id: template.sample_id, split: 'train', clean_obj_source: template.clean_obj_source, scorer_model: template.scorer_model, prompt_version: template.prompt_version, reference_model_sha256: template.reference_model_sha256, run_id: template.run_id, ...entry })));
  } catch (error) { reason = error.message; }
  quarantine.push({ sample: key, original_pairs: previous.length, candidates: candidates.length, strict_safe_candidates: safe.length, quarantined_candidate_ids: unsafe.map((item) => item.candidate_id), selected_pairs: pairs.filter((pair) => `${pair.category}/${pair.sample_id}` === key).length, needs_new_qwen_candidates: Boolean(reason), reason });
}
const output = { ...dataset, pairs };
const audit = validateMdpoDataset(output, manifest, { requireComplete: false });
const hash = createHash('sha256').update(datasetBytes).digest('hex');
const backup = path.join(directory, `train_pairs.before_strict_safety_${hash.slice(0, 16)}.json`);
if (await exists(backup)) {
  const existingBytes = await fs.readFile(backup);
  const existingHash = createHash('sha256').update(existingBytes).digest('hex');
  if (existingHash !== hash) throw new Error(`Existing strict-safety backup hash mismatch: ${backup}`);
} else {
  await fs.writeFile(backup, datasetBytes, { flag: 'wx' });
}
const temporary = path.join(directory, `train_pairs.strict_${hash.slice(0, 16)}_${runStamp}.tmp`);
await fs.writeFile(temporary, JSON.stringify(output, null, 2) + '\n', { flag: 'wx' });
await fs.rename(temporary, datasetFile);
const canonicalReport = path.join(directory, 'strict_safety_quarantine.json');
const archivedReport = await preserveExisting(canonicalReport, 'before');
const report = { version: 'v10_mdpo_strict_zero_geometry_quarantine_v2', created_at: new Date().toISOString(), input_sha256: hash, input_backup: path.relative(root, backup).split(path.sep).join('/'), prior_report_archive: archivedReport ? path.relative(root, archivedReport).split(path.sep).join('/') : null, original_pair_count: dataset.pairs.length, strict_pair_count: pairs.length, samples: quarantine };
const immutableReport = path.join(directory, `strict_safety_quarantine.${runStamp}.json`);
await fs.writeFile(immutableReport, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
await fs.writeFile(canonicalReport, JSON.stringify(report, null, 2) + '\n');
await fs.writeFile(path.join(directory, 'dataset_audit.json'), JSON.stringify({ ...audit, reference_model_sha256: output.reference_model_sha256, complete: false, strict_zero_geometry: true, updated_at: new Date().toISOString() }, null, 2) + '\n');
console.log(JSON.stringify({ original_pairs: dataset.pairs.length, strict_pairs: pairs.length, backup: path.relative(root, backup).split(path.sep).join('/'), report: path.relative(root, immutableReport).split(path.sep).join('/'), quarantine, audit }, null, 2));
