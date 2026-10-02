import { createHash } from 'node:crypto';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function validateMdpoFinalDatasetAudit({ datasetBytes, audit, referenceSha256 } = {}) {
  if (!Buffer.isBuffer(datasetBytes) || !/^[a-f0-9]{64}$/i.test(String(referenceSha256 || '')))
    throw new Error('MDPO final dataset audit requires train33 bytes and frozen reference hash');
  const dataset = JSON.parse(datasetBytes.toString('utf8'));
  if (audit?.version !== 'v10_mdpo_dataset_audit_v1' || audit.complete !== true
      || audit.dataset_sha256 !== digest(datasetBytes)
      || audit.reference_model_sha256 !== referenceSha256
      || dataset.reference_model_sha256 !== referenceSha256
      || audit.sample_count !== 33 || audit.pair_count < 198 || audit.pair_count > 396
      || audit.unique_candidate_count < 264 || audit.mean_candidates_per_sample < 8
      || audit.split_integrity?.train_only !== true || audit.split_integrity?.val_pairs !== 0
      || audit.split_integrity?.test_pairs !== 0
      || audit.provenance?.six_view_byte_hashes !== true || audit.provenance?.qwen_response_ids !== true
      || audit.provenance?.fixed_label_contract !== true || audit.provenance?.deterministic_safety_recomputed !== true)
    throw new Error('MDPO immutable final train33 audit is missing, incomplete, or differs from the dataset/reference');
  return { dataset, datasetSha256: audit.dataset_sha256, referenceSha256, sampleCount: 33, pairCount: audit.pair_count,
    uniqueCandidateCount: audit.unique_candidate_count };
}
