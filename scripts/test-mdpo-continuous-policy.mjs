import assert from 'node:assert/strict';
import { MDPO_DIMENSIONS, applyMdpoVarianceGradient, gaussianLayoutLogProbability, initializeMdpoVariance, materializeMdpoStd, mdpoPairLoss } from '../lib/mdpo-continuous-policy.mjs';

const ref = [[0, 0, 0, 0, 0, 0]];
const a = [[0.10, 0, 0, 0, 0, 0]];
const b = [[-0.10, 0, 0, 0, 0, 0]];
const scoresA = Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, 3]));
const scoresB = { ...scoresA, overall: 2 };
assert.ok(Number.isFinite(gaussianLayoutLogProbability(a, ref)));
assert.equal(gaussianLayoutLogProbability(a, ref), gaussianLayoutLogProbability(b, ref));
assert.throws(() => gaussianLayoutLogProbability([[Infinity, 0, 0, 0, 0, 0]], ref), /finite/);
const preferred = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: a, candidateB: b, scoresA, scoresB });
assert.ok(preferred.gradient[0][0] < 0, 'minimizing loss must move the mean toward preferred A');
assert.equal(preferred.perDimension.composition_harmony.loss, 0);
assert.equal(preferred.perDimension.composition_harmony.gradient_norm, 0);
const flipped = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: b, candidateB: a, scoresA, scoresB });
assert.ok(Math.abs(flipped.gradient[0][0] + preferred.gradient[0][0]) < 1e-12, 'swapping candidate layouts flips the output gradient');
const onlyText = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: a, candidateB: b,
  scoresA: { ...scoresA, overall: 3, text_clarity: 1 }, scoresB: { ...scoresA, text_clarity: 5 } });
assert.ok(onlyText.gradient[0][0] > 0, 'text clarity must independently reverse the gradient');
assert.equal(onlyText.perDimension.overall.gradient_norm, 0);
const tie = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: a, candidateB: b, scoresA, scoresB: { ...scoresA, overall: 2.95 } });
assert.deepEqual(tie.gradient, ref);
assert.deepEqual(tie.logStdGradient, [0, 0, 0, 0, 0, 0]);
const learnable = initializeMdpoVariance({ learnable: true });
const initialStd = materializeMdpoStd(learnable);
const varianceLoss = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: [[0.2, 0, 0, 0, 0, 0]], candidateB: [[0.05, 0, 0, 0, 0, 0]], scoresA, scoresB, policyStd: initialStd });
assert.ok(Math.abs(varianceLoss.logStdGradient[0]) > 1e-9, 'learnable variance must receive an independent output-scale gradient');
assert.equal(learnable.trainable_parameters, 6);
assert.equal(applyMdpoVarianceGradient(learnable, varianceLoss.logStdGradient, 1e-3).updated, true);
assert.notDeepEqual(materializeMdpoStd(learnable), initialStd);
const frozenVariance = initializeMdpoVariance();
assert.equal(applyMdpoVarianceGradient(frozenVariance, varianceLoss.logStdGradient, 1e-3).updated, false);
assert.deepEqual(materializeMdpoStd(frozenVariance), initialStd);
const noText = mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: a, candidateB: b,
  scoresA: { ...scoresA, overall: 3, text_clarity: 1 }, scoresB: { ...scoresA, text_clarity: 5 }, weights: { text_clarity: 0, leader_line_clarity: 0.1 } });
assert.deepEqual(noText.gradient, ref, 'zero text dimension weight must disable its gradient');
assert.ok(preferred.loss > 0 && Number.isFinite(preferred.loss));
const shiftedReference = [[0.25, 0, 0, 0, 0, 0]];
const withReference = mdpoPairLoss({ mean: ref, referenceMean: shiftedReference, candidateA: a, candidateB: b,
  scoresA, scoresB, lambdaKl: 0 });
const noReference = mdpoPairLoss({ mean: ref, referenceMean: shiftedReference, candidateA: a, candidateB: b,
  scoresA, scoresB, lambdaKl: 0, useReference: false });
assert.equal(withReference.useReference, true);
assert.equal(noReference.useReference, false);
assert.equal(noReference.kl, 0);
assert.notEqual(noReference.loss, withReference.loss, 'no-reference ablation must remove reference from DPO ratio, not only remove KL');
assert.notEqual(noReference.gradient[0][0], withReference.gradient[0][0]);
assert.throws(() => mdpoPairLoss({ mean: ref, referenceMean: ref, candidateA: a, candidateB: b, scoresA, scoresB,
  lambdaKl: 0.1, useReference: false }), /Invalid MDPO hyperparameters/);
console.log('MDPO Gaussian policy, seven-dimensional tie and gradient tests passed.');
