// The policy, reference and candidate layouts must use the same ordered
// anchor-local six-dimensional representation; screen/world coordinates are
// deliberately not accepted here.
export const MDPO_DIMENSIONS = Object.freeze([
  'overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance',
  'manual_style_similarity', 'text_clarity', 'leader_line_clarity'
]);
export const MDPO_STD = Object.freeze([0.15, 0.15, 0.10, 0.18, 0.18, 0.18]);
export const MDPO_WEIGHTS = Object.freeze({
  overall: 0.30, composition_harmony: 0.25, visual_hierarchy: 0.20,
  spatial_balance: 0.15, manual_style_similarity: 0.10,
  text_clarity: 0.10, leader_line_clarity: 0.10
});

function assertLayout(layout, expectedNodes = null) {
  if (!Array.isArray(layout) || !layout.length || (expectedNodes !== null && layout.length !== expectedNodes)
      || layout.some((row) => !Array.isArray(row) || row.length !== 6 || row.some((v) => !Number.isFinite(v)))) {
    throw new Error('MDPO requires ordered finite anchor-local Nx6 layouts with identical label count');
  }
}

function assertStd(std) {
  if (!Array.isArray(std) || std.length !== 6 || std.some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error('MDPO fixed policy standard deviations must contain six positive finite values');
  }
}

export function initializeMdpoVariance({ std = MDPO_STD, learnable = false, minimum = 0.03, maximum = 0.75 } = {}) {
  assertStd(std);
  if (![minimum, maximum].every((value) => Number.isFinite(value) && value > 0) || minimum >= maximum
      || std.some((value) => value < minimum || value > maximum)) throw new Error('Invalid MDPO variance bounds');
  return { version: 'v10_mdpo_diagonal_variance_v1', learnable: Boolean(learnable), minimum, maximum,
    reference_std: [...std], log_std: std.map(Math.log), trainable_parameters: learnable ? 6 : 0 };
}

export function materializeMdpoStd(variance) {
  if (!variance || !Array.isArray(variance.log_std) || variance.log_std.length !== 6) throw new Error('Invalid MDPO variance state');
  const std = variance.log_std.map((value) => Math.min(variance.maximum, Math.max(variance.minimum, Math.exp(value))));
  assertStd(std);
  return std;
}

export function applyMdpoVarianceGradient(variance, gradient, learningRate, clipNorm = 1) {
  if (!variance?.learnable) return { gradient_norm: 0, clipped: false, updated: false };
  if (!Array.isArray(gradient) || gradient.length !== 6 || gradient.some((value) => !Number.isFinite(value))
      || !Number.isFinite(learningRate) || learningRate <= 0 || !Number.isFinite(clipNorm) || clipNorm <= 0) throw new Error('Invalid learnable MDPO variance gradient');
  const norm = Math.hypot(...gradient), step = learningRate * Math.min(1, clipNorm / Math.max(norm, 1e-12));
  for (let axis = 0; axis < 6; axis += 1) {
    const next = variance.log_std[axis] - step * gradient[axis];
    variance.log_std[axis] = Math.min(Math.log(variance.maximum), Math.max(Math.log(variance.minimum), next));
  }
  return { gradient_norm: norm, clipped: norm > clipNorm, updated: true };
}

export function gaussianLayoutLogProbability(layout, mean, std = MDPO_STD) {
  assertStd(std);
  assertLayout(mean);
  assertLayout(layout, mean.length);
  let result = 0;
  for (let node = 0; node < layout.length; node += 1) {
    for (let axis = 0; axis < 6; axis += 1) {
      const normalized = (layout[node][axis] - mean[node][axis]) / std[axis];
      result -= 0.5 * normalized * normalized + Math.log(std[axis]) + 0.5 * Math.log(2 * Math.PI);
    }
  }
  if (!Number.isFinite(result)) throw new Error('MDPO Gaussian log probability is not finite');
  return result;
}

function softplus(value) { return Math.log1p(Math.exp(-Math.abs(value))) + Math.max(value, 0); }
function logistic(value) { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }

export function mdpoPairLoss({ mean, referenceMean, candidateA, candidateB, scoresA, scoresB,
  std = MDPO_STD, policyStd = std, referenceStd = MDPO_STD, beta = 0.1, lambdaDpo = 1, lambdaMulti = 0.3,
  weights = MDPO_WEIGHTS, tieThreshold = 0.025, lambdaKl = 0, useReference = true } = {}) {
  assertStd(policyStd);
  assertStd(referenceStd);
  assertLayout(mean);
  for (const layout of [referenceMean, candidateA, candidateB]) assertLayout(layout, mean.length);
  if (![beta, lambdaDpo, lambdaMulti, tieThreshold, lambdaKl].every((v) => Number.isFinite(v) && v >= 0)
      || typeof useReference !== 'boolean' || (!useReference && lambdaKl !== 0)) throw new Error('Invalid MDPO hyperparameters');
  if (MDPO_DIMENSIONS.some((name) => !Number.isFinite(scoresA?.[name]) || scoresA[name] < 1 || scoresA[name] > 5
    || !Number.isFinite(scoresB?.[name]) || scoresB[name] < 1 || scoresB[name] > 5)) throw new Error('MDPO requires seven complete Qwen 1–5 dimensions');
  const logRatioA = gaussianLayoutLogProbability(candidateA, mean, policyStd) - (useReference ? gaussianLayoutLogProbability(candidateA, referenceMean, referenceStd) : 0);
  const logRatioB = gaussianLayoutLogProbability(candidateB, mean, policyStd) - (useReference ? gaussianLayoutLogProbability(candidateB, referenceMean, referenceStd) : 0);
  const gradient = mean.map(() => Array(6).fill(0));
  const logStdGradient = Array(6).fill(0);
  const perDimension = {};
  const effectiveWeights = { ...MDPO_WEIGHTS, ...(weights || {}) };
  let loss = 0;
  for (const name of MDPO_DIMENSIONS) {
    const margin = (scoresA[name] - scoresB[name]) / 4;
    const dimensionWeight = name === 'overall' ? lambdaDpo : lambdaMulti * effectiveWeights[name];
    if (!Number.isFinite(dimensionWeight) || dimensionWeight < 0) throw new Error(`Invalid MDPO weight: ${name}`);
    if (Math.abs(margin) <= tieThreshold || !dimensionWeight) {
      perDimension[name] = { margin, tie: true, loss: 0, gradient_norm: 0 };
      continue;
    }
    const sign = Math.sign(margin);
    const logit = beta * sign * (logRatioA - logRatioB);
    const scale = dimensionWeight * Math.abs(margin);
    const termLoss = scale * softplus(-logit);
    // With fixed covariance, d(log p(A)-log p(B))/d mean = (A-B)/std².
    const factor = -scale * beta * sign * logistic(-logit);
    let gradientSquare = 0;
    for (let node = 0; node < mean.length; node += 1) for (let axis = 0; axis < 6; axis += 1) {
      const variance = policyStd[axis] ** 2;
      const update = factor * (candidateA[node][axis] - candidateB[node][axis]) / variance;
      gradient[node][axis] += update;
      const differenceA = candidateA[node][axis] - mean[node][axis];
      const differenceB = candidateB[node][axis] - mean[node][axis];
      logStdGradient[axis] += factor * (differenceA ** 2 - differenceB ** 2) / variance;
      gradientSquare += update * update;
    }
    loss += termLoss;
    perDimension[name] = { margin, tie: false, loss: termLoss, gradient_norm: Math.sqrt(gradientSquare) };
  }
  let kl = 0;
  if (useReference) for (let node = 0; node < mean.length; node += 1) for (let axis = 0; axis < 6; axis += 1) {
    const delta = mean[node][axis] - referenceMean[node][axis];
    const policyVariance = policyStd[axis] ** 2, referenceVariance = referenceStd[axis] ** 2;
    kl += Math.log(referenceStd[axis] / policyStd[axis]) + (policyVariance + delta ** 2) / (2 * referenceVariance) - 0.5;
    gradient[node][axis] += lambdaKl * delta / referenceVariance;
    logStdGradient[axis] += lambdaKl * (policyVariance / referenceVariance - 1);
  }
  loss += lambdaKl * kl;
  if (!Number.isFinite(loss) || gradient.some((row) => row.some((v) => !Number.isFinite(v))) || logStdGradient.some((value) => !Number.isFinite(value))) throw new Error('Non-finite MDPO loss or gradient');
  return { loss, kl, perDimension, gradient, logStdGradient, policyStd: [...policyStd], referenceStd: [...referenceStd], logRatioA, logRatioB, tieThreshold, useReference };
}
