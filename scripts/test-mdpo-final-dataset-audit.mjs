import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';

const referenceSha256 = 'a'.repeat(64);
const datasetBytes = Buffer.from(JSON.stringify({ version: 'v10_mdpo_train_pairs_v1', reference_model_sha256: referenceSha256, pairs: [] }));
const audit = { version: 'v10_mdpo_dataset_audit_v1', complete: true,
  dataset_sha256: createHash('sha256').update(datasetBytes).digest('hex'), reference_model_sha256: referenceSha256,
  sample_count: 33, pair_count: 198, unique_candidate_count: 264, mean_candidates_per_sample: 8,
  split_integrity: { train_only: true, val_pairs: 0, test_pairs: 0 },
  provenance: { six_view_byte_hashes: true, qwen_response_ids: true, fixed_label_contract: true, deterministic_safety_recomputed: true } };
assert.equal(validateMdpoFinalDatasetAudit({ datasetBytes, audit, referenceSha256 }).pairCount, 198);
assert.throws(() => validateMdpoFinalDatasetAudit({ datasetBytes: Buffer.from('{}'), audit, referenceSha256 }), /missing, incomplete/);
assert.throws(() => validateMdpoFinalDatasetAudit({ datasetBytes, audit: { ...audit, pair_count: 197 }, referenceSha256 }), /missing, incomplete/);
assert.throws(() => validateMdpoFinalDatasetAudit({ datasetBytes, audit: { ...audit, split_integrity: { ...audit.split_integrity, val_pairs: 1 } }, referenceSha256 }), /missing, incomplete/);
assert.throws(() => validateMdpoFinalDatasetAudit({ datasetBytes, audit, referenceSha256: 'b'.repeat(64) }), /missing, incomplete/);
console.log('v10-MDPO immutable final train33 dataset audit tests passed.');
