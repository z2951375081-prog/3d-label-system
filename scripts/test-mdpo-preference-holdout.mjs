import assert from 'node:assert/strict';
import { splitMdpoTrainPreferencePairs } from '../lib/mdpo-preference-holdout.mjs';

const samples = ['Chair/001', 'Lamp/002'];
const pairs = samples.flatMap((sample) => Array.from({ length: 6 }, (_, index) => {
  const [category, sample_id] = sample.split('/');
  return { category, sample_id, split: 'train', candidate_a: { candidate_id: `${sample}_a${index}` },
    candidate_b: { candidate_id: `${sample}_b${index}` } };
}));
const split = splitMdpoTrainPreferencePairs(pairs);
assert.equal(split.holdoutPairs.length, 2);
assert.equal(split.gradientPairs.length, 10);
assert.equal(split.policy, 'exactly_one_deterministic_train33_pair_per_object_heldout_no_gradient_not_val11');
assert.equal(split.holdout_sha256, splitMdpoTrainPreferencePairs([...pairs].reverse()).holdout_sha256);
assert.equal(new Set(split.holdoutPairs).size, 2);
assert.ok(split.holdoutPairs.every((pair) => !split.gradientPairs.includes(pair)));
assert.throws(() => splitMdpoTrainPreferencePairs([{ ...pairs[0], split: 'val' }, ...pairs.slice(1)]), /only identified train33/);
assert.throws(() => splitMdpoTrainPreferencePairs([...pairs, { ...pairs[0] }]), /duplicate preference pairs/);
console.log('v10-MDPO deterministic train33 heldout Qwen preference split tests passed.');
