import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fixedCandidatesWithoutTargetLayout, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { multiViewLayoutFeatures, targetVector } from '../lib/layout-model.mjs';
import { annotationsToLabels } from './generate-artifacts.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
let labelsChecked = 0;
for (const sample of manifest.samples) {
  const annotation = JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8'));
  assert.equal(annotation.version, 'after_mannual_adjust');
  assert.equal(annotation.layout_type, 'manual_adjusted');
  const manual = annotationsToLabels(annotation);
  const bounds = { center: [0, 0, 0], radius: 2, size: [2, 2, 2] };
  const generated = manual.map((label, index) => ({
    anchor: label.anchor,
    center: label.anchor.map((value, axis) => value + (axis === 0 ? 0.2 : 0.1)),
    boxSize: [0.18 + index * 0.01, 0.1, 0.02],
    text: label.text,
    sourceObjs: label.sourceObjs,
    targetGroups: label.targetGroups
  }));
  const first = fixedCandidatesWithoutTargetLayout(manual, generated, bounds);
  validateFixedLabelContract(manual, first, sample.category + '/' + sample.sample_id);
  const changedTargets = manual.map((label) => ({ ...label, center: label.center.map((v) => v + 999), boxSize: label.boxSize.map((v) => v + 999) }));
  const second = fixedCandidatesWithoutTargetLayout(changedTargets, generated, bounds);
  assert.deepEqual(first, second, 'adjusted target center/size must not affect input candidates');
  first.forEach((candidate, index) => {
    const a = multiViewLayoutFeatures(candidate, bounds, index, first.length);
    const b = multiViewLayoutFeatures(second[index], bounds, index, second.length);
    assert.deepEqual(a, b);
    assert.equal(a.length, 51);
    assert.notDeepEqual(targetVector(manual[index], manual[index].anchor, bounds), targetVector(changedTargets[index], changedTargets[index].anchor, bounds));
    labelsChecked += 1;
  });
}
assert.equal(manifest.samples.length, 55);
assert.equal(labelsChecked, 406);
console.log(JSON.stringify({ samples: manifest.samples.length, labels: labelsChecked, adjusted_center_or_size_in_candidate_features: false, target_source: 'after_mannual_adjust manual_adjusted JSON', status: 'passed' }));
