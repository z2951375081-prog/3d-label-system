import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { MDPO_DIMENSIONS } from '../lib/mdpo-continuous-policy.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const datasetFile = path.join(root, 'experiments', 'mdpo', 'train_pairs.json');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const modelFile = path.join(root, 'experiments', 'layout_model.json');
const outputFile = path.join(root, 'experiments', 'mdpo', 'dataset_audit_final.json');
const datasetBytes = await fs.readFile(datasetFile), modelBytes = await fs.readFile(modelFile);
const dataset = JSON.parse(datasetBytes.toString('utf8')), manifest = JSON.parse(await fs.readFile(manifestFile, 'utf8'));
const audit = validateMdpoDataset(dataset, manifest, { requireComplete: true });
const candidates = new Map();
for (const pair of dataset.pairs) for (const candidate of [pair.candidate_a, pair.candidate_b]) candidates.set(candidate.candidate_id, candidate);
const dimensionSignals = Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, { positive: 0, negative: 0, tie: 0, mean_absolute_margin: 0 }]));
for (const pair of dataset.pairs) for (const name of MDPO_DIMENSIONS) {
  const margin = (pair.candidate_a.scores[name] - pair.candidate_b.scores[name]) / 4;
  dimensionSignals[name][Math.abs(margin) <= 0.025 ? 'tie' : margin > 0 ? 'positive' : 'negative']++;
  dimensionSignals[name].mean_absolute_margin += Math.abs(margin) / dataset.pairs.length;
}
const referenceHash = createHash('sha256').update(modelBytes).digest('hex');
if (dataset.reference_model_sha256 !== referenceHash) throw new Error('Complete MDPO dataset no longer matches active frozen v10 reference');
const report = {
  version: 'v10_mdpo_dataset_audit_v1', generated_at: new Date().toISOString(), complete: true, ...audit,
  unique_candidate_count: candidates.size, mean_candidates_per_sample: candidates.size / audit.sample_count,
  dataset_sha256: createHash('sha256').update(datasetBytes).digest('hex'), reference_model_sha256: referenceHash,
  dimensions: dimensionSignals, split_integrity: { train_only: true, val_pairs: 0, test_pairs: 0 },
  provenance: { scorer_model: audit.scorer_model, prompt_version: audit.prompt_version, six_view_byte_hashes: true, qwen_response_ids: true, fixed_label_contract: true, deterministic_safety_recomputed: true }
};
const reportText = JSON.stringify(report, null, 2) + '\n';
const existing = await fs.readFile(outputFile, 'utf8').catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
if (existing !== null && existing !== reportText) throw new Error('Existing immutable complete MDPO dataset audit differs from current train33 evidence');
if (existing === null) await fs.writeFile(outputFile, reportText, { flag: 'wx' });
console.log(JSON.stringify(report, null, 2));
