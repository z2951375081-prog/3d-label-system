// Shared V10 style adapter mixture-of-experts block.
// The shared graph encoder produces one hidden vector per label.  This module
// keeps all style-specific parameters in three small residual adapters and
// guarantees a six-dimensional local-frame layout output per expert.

export const STYLE_ADAPTER_KEYS = Object.freeze(['radial_ring', 'box_sides', 'anchor_adaptive']);
export const STYLE_ADAPTER_EXPERT_NAMES = Object.freeze([
  'radial_ring_expert',
  'box_sides_expert',
  'anchor_adaptive_expert'
]);
export const LOCAL_LAYOUT_OUTPUT_DIM = 6;

const add = (...values) => values[0].map((_, index) => values.reduce((sum, value) => sum + Number(value[index] || 0), 0));
const matrixVector = (weights, values) => weights.map((row) => row.reduce((sum, weight, index) => sum + Number(weight || 0) * Number(values[index] || 0), 0));
const softmax = (values) => {
  const peak = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(Number(value) - peak));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / Math.max(total, 1e-12));
};

function canonicalStyleKey(value) {
  const key = String(value || '').toLowerCase();
  if (key === 'spherical' || key === 'radial' || key === 'radial_ring') return 'radial_ring';
  if (key === 'rectangular' || key === 'box' || key === 'box_sides') return 'box_sides';
  if (key === 'surround' || key === 'anchor' || key === 'anchor_adaptive') return 'anchor_adaptive';
  return null;
}

function normalizeWeights(values) {
  const weights = STYLE_ADAPTER_KEYS.map((key) => Math.max(0, Number(values?.[key] ?? 0)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return total > 0 ? weights.map((value) => value / total) : null;
}

function oneHot(index) {
  return STYLE_ADAPTER_KEYS.map((_, candidate) => candidate === index ? 1 : 0);
}

function selectMixture(routerWeights, options = {}) {
  const forced = canonicalStyleKey(options.styleExpert || options.style || options.forcedStyle);
  if (forced) return { weights: oneHot(STYLE_ADAPTER_KEYS.indexOf(forced)), mode: 'forced', selected: forced };
  const explicit = normalizeWeights(options.styleWeights);
  if (explicit) {
    const selected = STYLE_ADAPTER_KEYS[explicit.indexOf(Math.max(...explicit))];
    return { weights: explicit, mode: 'explicit_blend', selected };
  }
  const selected = STYLE_ADAPTER_KEYS[routerWeights.indexOf(Math.max(...routerWeights))];
  return { weights: routerWeights, mode: 'router_blend', selected };
}

function validateMatrix(matrix, rows, columns, name) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== columns)) {
    throw new Error(`${name} must have shape ${rows}x${columns}`);
  }
}

function validateExpert(expert, hiddenDim, styleDim, adapterDim, expertIndex) {
  if (!expert || typeof expert !== 'object') throw new Error(`style expert ${expertIndex} is missing`);
  validateMatrix(expert.style_adapter?.down_weights, adapterDim, hiddenDim + styleDim, `expert[${expertIndex}].style_adapter.down_weights`);
  validateMatrix(expert.style_adapter?.up_weights, hiddenDim, adapterDim, `expert[${expertIndex}].style_adapter.up_weights`);
  if (expert.style_adapter.down_bias?.length !== adapterDim || expert.style_adapter.up_bias?.length !== hiddenDim) throw new Error(`style expert ${expertIndex} adapter bias shape mismatch`);
  validateMatrix(expert.hidden_weights, Math.max(1, expert.hidden_weights?.length || 0), hiddenDim, `expert[${expertIndex}].hidden_weights`);
  if (expert.weights?.length !== LOCAL_LAYOUT_OUTPUT_DIM || expert.bias?.length !== LOCAL_LAYOUT_OUTPUT_DIM) throw new Error(`style expert ${expertIndex} must output exactly 6 local layout values`);
}

/**
 * Run the style embedding -> three residual adapters -> 6D heads -> router.
 * `hidden` is the shared V10 graph representation for one label.
 */
export function runStyleAdapterMoE(hidden, moe, options = {}) {
  if (!Array.isArray(hidden) || !hidden.length) throw new Error('style adapter MoE requires a non-empty shared hidden vector');
  const experts = moe?.experts;
  if (!Array.isArray(experts) || experts.length !== STYLE_ADAPTER_KEYS.length) throw new Error('V10 style adapter MoE requires exactly three experts');
  const hiddenDim = hidden.length;
  const styleDim = Math.max(1, Number(moe?.style_adapter?.style_embedding_dim || moe?.style_embeddings?.[0]?.length || 0));
  const adapterDim = Math.max(1, Number(moe?.style_adapter?.adapter_dim || experts[0]?.style_adapter?.down_weights?.length || 0));
  const embeddings = moe.style_embeddings;
  if (!Array.isArray(embeddings) || embeddings.length !== experts.length || embeddings.some((embedding) => !Array.isArray(embedding) || embedding.length !== styleDim)) throw new Error('style_embeddings must contain one embedding per expert');
  if (!Array.isArray(moe.router?.weights) || moe.router.weights.length !== experts.length || moe.router.weights.some((row) => row.length !== hiddenDim)) throw new Error('style router weight shape mismatch');
  if (moe.router.bias?.length !== experts.length) throw new Error('style router bias shape mismatch');
  experts.forEach((expert, index) => validateExpert(expert, hiddenDim, styleDim, adapterDim, index));

  const logits = add(matrixVector(moe.router.weights, hidden), moe.router.bias);
  const routerWeights = softmax(logits);
  const mixture = selectMixture(routerWeights, options);
  const expertOutputs = experts.map((expert, expertIndex) => {
    const embedding = embeddings[expertIndex];
    const adapterInput = [...hidden, ...embedding];
    const adapterHidden = add(matrixVector(expert.style_adapter.down_weights, adapterInput), expert.style_adapter.down_bias).map(Math.tanh);
    const adapterDelta = add(matrixVector(expert.style_adapter.up_weights, adapterHidden), expert.style_adapter.up_bias).map(Math.tanh);
    const adaptedHidden = add(hidden, adapterDelta);
    const head = add(matrixVector(expert.hidden_weights, adaptedHidden), expert.hidden_bias).map(Math.tanh);
    const output = add(matrixVector(expert.weights, head), expert.bias);
    if (output.length !== LOCAL_LAYOUT_OUTPUT_DIM) throw new Error(`style expert ${expertIndex} output dimension is ${output.length}, expected 6`);
    return output;
  });
  const output = expertOutputs[0].map((_, feature) => expertOutputs.reduce((sum, expertOutput, expertIndex) => sum + mixture.weights[expertIndex] * expertOutput[feature], 0));
  return {
    output,
    expertOutputs,
    routerWeights,
    mixtureWeights: mixture.weights,
    selectedStyle: mixture.selected,
    fusionMode: mixture.mode,
    styleKeys: [...STYLE_ADAPTER_KEYS]
  };
}

export function styleAdapterMoEContract(moe = {}) {
  const styleDim = Number(moe.style_adapter?.style_embedding_dim || moe.style_embeddings?.[0]?.length || 0);
  const adapterDim = Number(moe.style_adapter?.adapter_dim || moe.experts?.[0]?.style_adapter?.down_weights?.length || 0);
  return {
    enabled: Boolean(moe.style_adapter?.enabled),
    version: moe.style_adapter?.version || 'style_adapter_moe_v1',
    expert_count: Array.isArray(moe.experts) ? moe.experts.length : 0,
    style_keys: [...STYLE_ADAPTER_KEYS],
    style_embedding_dim: styleDim,
    adapter_dim: adapterDim,
    output_dim: LOCAL_LAYOUT_OUTPUT_DIM,
    output_semantics: 'local_tangent_u_local_tangent_v_surface_normal_distance_and_log_size_xyz',
    fusion: 'router_softmax_or_forced_style_or_explicit_blend'
  };
}

export { canonicalStyleKey, normalizeWeights };

