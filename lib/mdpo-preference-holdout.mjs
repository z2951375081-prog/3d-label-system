import { createHash } from 'node:crypto';

export function splitMdpoTrainPreferencePairs(pairs) {
  if (!Array.isArray(pairs) || pairs.length < 6) throw new Error('MDPO preference validation requires train33 pair records');
  const grouped = new Map();
  for (const pair of pairs) {
    if (pair?.split !== 'train' || !pair.category || !pair.sample_id
        || !pair.candidate_a?.candidate_id || !pair.candidate_b?.candidate_id)
      throw new Error('MDPO preference holdout accepts only identified train33 pairs');
    const key = `${pair.category}/${pair.sample_id}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(pair);
  }
  const gradientPairs = [], holdoutPairs = [], holdoutIds = [];
  for (const [sample, rows] of [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    if (rows.length < 6 || rows.length > 12) throw new Error(`MDPO train33 ${sample} needs 6–12 preference pairs for heldout diagnostic`);
    const ranked = rows.map((pair) => ({ pair, key: `${sample}|${[pair.candidate_a.candidate_id, pair.candidate_b.candidate_id].sort().join('|')}` }));
    if (new Set(ranked.map((row) => row.key)).size !== rows.length) throw new Error(`MDPO train33 ${sample} contains duplicate preference pairs`);
    ranked.sort((a, b) => createHash('sha256').update(a.key).digest('hex').localeCompare(createHash('sha256').update(b.key).digest('hex')));
    holdoutPairs.push(ranked[0].pair);
    holdoutIds.push(ranked[0].key);
    gradientPairs.push(...ranked.slice(1).map((row) => row.pair));
  }
  return { gradientPairs, holdoutPairs, holdoutIds,
    policy: 'exactly_one_deterministic_train33_pair_per_object_heldout_no_gradient_not_val11',
    holdout_sha256: createHash('sha256').update(JSON.stringify(holdoutIds)).digest('hex') };
}
