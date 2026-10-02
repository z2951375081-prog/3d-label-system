// LoRA adapters for the exact matrix paths consumed by the existing v10
// heterogeneous GNN / Transformer / MoE forward pass.
function matrixAt(network, path) {
  let value = network;
  for (const part of path) value = value?.[part];
  if (!Array.isArray(value) || !value.length || !Array.isArray(value[0]) || !value[0].length
    || value.some((row) => row.length !== value[0].length)) throw new Error(`Invalid LoRA matrix: ${path.join('.')}`);
  return value;
}

function vectorAt(network, path) {
  let value = network;
  for (const part of path) value = value?.[part];
  if (!Array.isArray(value) || !value.length || value.some((item) => !Number.isFinite(item))) throw new Error(`Invalid trainable MDPO bias: ${path.join('.')}`);
  return value;
}

export function v10TrainableBiasPaths(network) {
  const paths = [['moe', 'router', 'bias'], ...network.moe.experts.map((_, index) => ['moe', 'experts', index, 'bias'])];
  paths.forEach((path) => vectorAt(network, path));
  return paths;
}

export function initializeV10BiasTuning(network) {
  const biases = v10TrainableBiasPaths(network).map((path) => ({ path, delta: vectorAt(network, path).map(() => 0) }));
  return { version: 'v10_mdpo_bias_delta_v1', biases, trainable_parameters: biases.reduce((sum, item) => sum + item.delta.length, 0) };
}

export function materializeV10BiasTuning(network, tuning) {
  const merged = structuredClone(network);
  for (const item of tuning.biases) {
    const target = vectorAt(merged, item.path);
    if (target.length !== item.delta.length) throw new Error('MDPO bias delta shape differs from frozen reference');
    for (let index = 0; index < target.length; index += 1) target[index] += item.delta[index];
  }
  return merged;
}

export function applyV10BiasGradient(tuning, networkGradient, learningRate, clipNorm = 1) {
  const values = tuning.biases.flatMap((item) => vectorAt(networkGradient, item.path));
  const norm = Math.hypot(...values);
  if (!Number.isFinite(norm)) throw new Error('Non-finite MDPO bias gradient');
  const step = learningRate * Math.min(1, clipNorm / Math.max(1e-12, norm));
  for (const item of tuning.biases) {
    const gradient = vectorAt(networkGradient, item.path);
    for (let index = 0; index < item.delta.length; index += 1) item.delta[index] -= step * gradient[index];
  }
  return { gradient_norm: norm, clipped: norm > clipNorm };
}

export function v10LoraPaths(network) {
  if (network?.message_layers?.length < 2 || !network.transformer_layers?.length || network.moe?.experts?.length !== 4) {
    throw new Error('MDPO LoRA requires the active v10 2-layer GNN, Transformer and four MoE experts');
  }
  const paths = [];
  for (const key of ['self_weights', 'neighbor_weights', 'relation_edge_weights', 'anchor_edge_weights']) paths.push(['message_layers', 1, key]);
  network.transformer_layers.forEach((_, index) => {
    for (const name of ['query', 'key', 'value', 'output', 'ffn_in', 'ffn_out']) paths.push(['transformer_layers', index, `${name}_weights`]);
  });
  paths.push(['moe', 'router', 'weights']);
  network.moe.experts.forEach((_, index) => {
    paths.push(['moe', 'experts', index, 'hidden_weights'], ['moe', 'experts', index, 'weights']);
  });
  paths.forEach((path) => matrixAt(network, path));
  return paths;
}

function generator(seed) {
  let state = (seed >>> 0) || 17;
  return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 2 ** 32; };
}

export function initializeV10Lora(network, { rank = 4, alpha = 8, dropout = 0, seed = 17 } = {}) {
  if (!Number.isInteger(rank) || rank < 1 || !Number.isFinite(alpha) || alpha <= 0 || !Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    throw new Error('v10 LoRA requires positive rank/alpha and dropout in [0,1)');
  }
  const random = generator(seed);
  const matrices = v10LoraPaths(network).map((path) => {
    const weights = matrixAt(network, path);
    const rows = weights.length, columns = weights[0].length;
    return {
      path, rows, columns,
      A: Array.from({ length: rank }, () => Array.from({ length: columns }, () => (random() - 0.5) * 0.02)),
      B: Array.from({ length: rows }, () => Array(rank).fill(0))
    };
  });
  return { version: 'v10_mdpo_lora_v1', rank, alpha, dropout, seed, matrices,
    trainable_parameters: matrices.reduce((sum, row) => sum + rank * (row.rows + row.columns), 0) };
}

export function materializeV10Lora(baseNetwork, adapters) {
  return materializeV10LoraStep(baseNetwork, adapters).network;
}

// One dropout mask per matrix input column is shared across all nodes of the
// same graph. Backward uses that exact mask; evaluation always disables it.
export function materializeV10LoraStep(baseNetwork, adapters, { training = false, random = Math.random } = {}) {
  const merged = structuredClone(baseNetwork);
  const scale = adapters.alpha / adapters.rank;
  const masks = [];
  for (const adapter of adapters.matrices) {
    const matrix = matrixAt(merged, adapter.path);
    if (matrix.length !== adapter.rows || matrix[0].length !== adapter.columns) throw new Error('LoRA adapter shape differs from frozen reference');
    const mask = Array.from({ length: adapter.columns }, () => training && adapters.dropout > 0
      ? (random() < adapters.dropout ? 0 : 1 / (1 - adapters.dropout)) : 1);
    masks.push(mask);
    for (let i = 0; i < adapter.rows; i += 1) for (let j = 0; j < adapter.columns; j += 1) {
      for (let k = 0; k < adapters.rank; k += 1) matrix[i][j] += scale * adapter.B[i][k] * adapter.A[k][j] * mask[j];
    }
  }
  return { network: merged, masks };
}

export function applyV10LoraGradient(adapters, networkGradient, learningRate, clipNorm = 1, masks = null) {
  if (!Number.isFinite(learningRate) || learningRate <= 0 || !Number.isFinite(clipNorm) || clipNorm <= 0) throw new Error('Invalid LoRA learning rate or clipping');
  const scale = adapters.alpha / adapters.rank;
  const gradients = adapters.matrices.map((adapter, index) => {
    const g = matrixAt(networkGradient, adapter.path);
    const mask = masks?.[index] || Array(adapter.columns).fill(1);
    if (mask.length !== adapter.columns) throw new Error('LoRA gradient dropout mask mismatch');
    const dA = adapter.A.map((row) => row.map(() => 0));
    const dB = adapter.B.map((row) => row.map(() => 0));
    for (let i = 0; i < adapter.rows; i += 1) for (let j = 0; j < adapter.columns; j += 1) for (let k = 0; k < adapters.rank; k += 1) {
      dB[i][k] += scale * g[i][j] * adapter.A[k][j] * mask[j];
      dA[k][j] += scale * g[i][j] * adapter.B[i][k] * mask[j];
    }
    return { adapter, dA, dB };
  });
  const norm = Math.hypot(...gradients.flatMap(({ dA, dB }) => [...dA.flat(), ...dB.flat()]));
  if (!Number.isFinite(norm)) throw new Error('Non-finite LoRA gradient');
  const step = learningRate * Math.min(1, clipNorm / Math.max(1e-12, norm));
  for (const { adapter, dA, dB } of gradients) {
    for (let i = 0; i < adapter.A.length; i++) for (let j = 0; j < adapter.A[i].length; j++) adapter.A[i][j] -= step * dA[i][j];
    for (let i = 0; i < adapter.B.length; i++) for (let j = 0; j < adapter.B[i].length; j++) adapter.B[i][j] -= step * dB[i][j];
  }
  return { gradient_norm: norm, clipped: norm > clipNorm };
}
