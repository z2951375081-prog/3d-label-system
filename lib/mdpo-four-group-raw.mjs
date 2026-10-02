import fs from 'node:fs/promises';
import path from 'node:path';
import { MDPO_VAL11_GROUPS, validateMdpoFourGroupSummaries } from './mdpo-four-group.mjs';

const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);

export async function validateMdpoFourGroupRawRecords({ directory, candidateId, fourGroup } = {}) {
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(String(candidateId || '')) || fourGroup?.candidate_id !== candidateId
      || fourGroup?.groups?.length !== 4) throw new Error('Invalid 44-record MDPO val11 raw evidence identity');
  const groups = Object.fromEntries(fourGroup.groups.map((row) => [row.group, row]));
  const protocol = validateMdpoFourGroupSummaries(groups);
  if (!same(protocol, fourGroup.protocol)) throw new Error('MDPO four-group protocol changed after report publication');
  let verified = 0;
  for (const [name, spec] of Object.entries(MDPO_VAL11_GROUPS)) {
    const groupDirectory = path.join(path.resolve(directory), candidateId, name);
    const files = (await fs.readdir(groupDirectory)).filter((entry) => entry.endsWith('.json'));
    if (files.length !== 11) throw new Error(`MDPO val11 ${candidateId}/${name} has ${files.length}/11 immutable Qwen records`);
    const expectedFiles = [];
    for (const entry of groups[name].sample_scores) {
      const sample = entry.sample;
      const filename = `${encodeURIComponent(`${sample.category}__${sample.sample_id}`)}.json`;
      expectedFiles.push(filename);
      const record = JSON.parse(await fs.readFile(path.join(groupDirectory, filename), 'utf8'));
      if (record.version !== 'v10_mdpo_val11_sample_v1' || record.split !== 'val'
          || record.candidate_id !== candidateId || record.group !== name || record.role !== spec.model_role
          || record.scorer?.response_id !== entry.response_id || record.scorer.model !== groups[name].scorer_model
          || record.scorer.prompt_version !== groups[name].prompt_version
          || record.metric_protocol !== groups[name].metric_protocol
          || !same(record.views, groups[name].views)
          || !same(record.sample, entry.sample) || !same(record.scores, entry.scores)
          || !same(record.metrics, entry.metrics) || !same(record.view_sha256, entry.view_sha256)
          || !same(record.generation_strategy, entry.generation_strategy)
          || !same(record.evaluation, entry.evaluation) || !same(record.security, entry.security)
          || (record.preference_model?.model_sha256 ?? null) !== entry.preference_model_sha256)
        throw new Error(`MDPO val11 immutable Qwen record differs from four-group report: ${candidateId}/${name}/${filename}`);
      verified++;
    }
    if (!same(files.sort(), expectedFiles.sort())) throw new Error(`MDPO val11 raw record names changed: ${candidateId}/${name}`);
  }
  return { candidate_id: candidateId, immutable_records_verified: verified, protocol };
}
