import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { normalizedLeaderLength, summarizeLeaderLengths } from '../lib/leader-length-prior.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const manifestFile = path.join(experiments, 'dataset_manifest.json');
const outputFile = path.join(experiments, 'manual_leader_length_prior.json');
const manifestBytes = await fs.readFile(manifestFile);
const manifest = JSON.parse(manifestBytes.toString('utf8'));
const train = manifest.samples.filter((sample) => sample.split === 'train');
const rows = [];

for (const sample of train) {
  const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
  const bounds = boundsFromObj(cleanObj(raw).text);
  const annotation = JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8'));
  if (annotation.version !== 'after_mannual_adjust' || annotation.layout_type !== 'manual_adjusted') throw new Error(`${sample.category}/${sample.sample_id} 不是人工调整后标注`);
  for (const label of annotationsToLabels(annotation)) rows.push({ category: sample.category, sample_id: String(sample.sample_id), label_id: label.id, normalized_length: normalizedLeaderLength(label, bounds) });
}

const categories = Object.fromEntries([...new Set(rows.map((row) => row.category))].sort().map((category) => [category, summarizeLeaderLengths(rows.filter((row) => row.category === category).map((row) => row.normalized_length))]));
const report = {
  version: 'manual_leader_length_prior_v1',
  generated_at: new Date().toISOString(),
  enabled: true,
  source: {
    split: 'train_only',
    annotation_contract: 'after_mannual_adjust/manual_adjusted',
    sample_count: train.length,
    label_count: rows.length,
    normalization: 'euclidean(anchor, label.center) / clean_obj_bounds.radius',
    manifest_sha256: createHash('sha256').update(manifestBytes).digest('hex').toUpperCase(),
    val_used_for_distribution: false,
    test_used_for_distribution: false
  },
  selected_policy: {
    id: 'train_manual_category_band_p20_p90_v1',
    hard_min_quantile: 'p10',
    preferred_min_quantile: 'p20',
    target_quantile: 'p50',
    preferred_max_quantile: 'p90',
    hard_max_quantile: 'p95',
    minimum_category_labels: 12,
    rationale: 'P20 prevents systematically short leaders while retaining the lower tail; P90/P95 bound excessive distance. Category statistics fall back to the global train distribution when sparse.'
  },
  global: summarizeLeaderLengths(rows.map((row) => row.normalized_length)),
  categories,
  security: { contains_label_coordinates: false, contains_api_key: false },
  training_rows: rows.map((row) => ({ ...row, normalized_length: Number(row.normalized_length.toFixed(6)) }))
};

await fs.writeFile(outputFile, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ output: path.relative(root, outputFile).split(path.sep).join('/'), sample_count: train.length, label_count: rows.length, global: report.global, policy: report.selected_policy }, null, 2));
