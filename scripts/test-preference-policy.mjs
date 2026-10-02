import assert from 'node:assert/strict';
import { selectSafetyConstrainedTrial } from '../lib/preference-policy.mjs';

const baseline = {
  seed: 17,
  metrics: {
    objective_score: 2,
    label_object_occlusion_ratio: 0.10,
    object_penetration_ratio: 0.02,
    mesh_surface_intersection_ratio: 0.01,
    multi_view_worst_overflow: 0.05,
    text_clarity: 4.2,
    aesthetic_composition_harmony: 0.55
    ,leader_crossings: 0
    ,worst_view_leader_crossing_count: 0
    ,weighted_leader_crossing_risk: 0.005
    ,worst_view_leader_crossing_risk: 0.01
    ,cvar_view_leader_crossing_risk: 0.007
  }
};
const unsafeBeautiful = {
  seed: 18,
  metrics: {
    ...baseline.metrics,
    object_penetration_ratio: 0.20,
    aesthetic_composition_harmony: 0.99
  }
};
const safeBeautiful = {
  seed: 19,
  metrics: {
    ...baseline.metrics,
    objective_score: 2.01,
    object_penetration_ratio: 0.022,
    aesthetic_composition_harmony: 0.82
  }
};
const model = { weights: true };
const predict = (_model, metrics) => ({ score: metrics.aesthetic_composition_harmony });
const result = selectSafetyConstrainedTrial([baseline, unsafeBeautiful, safeBeautiful], model, predict);

assert.equal(result.selected.seed, 19, 'highest aesthetic score among safe candidates should win');
assert.equal(result.assessed.find((item) => item.seed === 18).safety.eligible, false, 'penetrating candidate must be rejected');
assert.deepEqual(result.assessed.find((item) => item.seed === 18).safety.violations, ['object_penetration_ratio']);
assert.equal(result.reference.seed, 17, 'lowest deterministic energy candidate is the safety reference');
for (const key of ['leader_crossings', 'worst_view_leader_crossing_count']) {
  const crossed = { seed: 20, metrics: { ...safeBeautiful.metrics, [key]: 1, aesthetic_composition_harmony: 1 } };
  const checked = selectSafetyConstrainedTrial([baseline, crossed], model, predict);
  assert.equal(checked.selected.seed, 17, `${key} must not win an aesthetic rerank`);
  assert.ok(checked.assessed.find((trial) => trial.seed === 20).safety.violations.includes(key));
  for (const missing of [null, undefined]) {
    const missingEvidence = { seed: 22, metrics: { ...safeBeautiful.metrics, [key]: missing, aesthetic_composition_harmony: 1 } };
    const missingCheck = selectSafetyConstrainedTrial([baseline, missingEvidence], model, predict);
    assert.equal(missingCheck.selected.seed, 17, `${key} ${missing} must not pass the safety gate`);
    assert.ok(missingCheck.assessed.find((trial) => trial.seed === 22).safety.violations.includes(key));
  }
}
assert.equal(selectSafetyConstrainedTrial([{ seed: 21, metrics: { ...baseline.metrics, leader_crossings: 1 } }], model, predict).selected, null, 'unsafe reference must not be silently accepted');
console.log('Preference policy passed: deterministic safety gate rejects unsafe beauty before aesthetic reward ranking.');
