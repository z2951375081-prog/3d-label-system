import { materializeV10BiasTuning, v10LoraPaths, v10TrainableBiasPaths } from './mdpo-lora.mjs';

const at = (network, parts) => parts.reduce((value, part) => value?.[part], network);
function group(path) {
  if (path[0] === 'message_layers') return 'relation_gnn_layer_2';
  if (path[0] === 'transformer_layers') return 'transformer_attention_and_ffn';
  if (path[1] === 'router') return 'human_style_moe_router';
  return `human_style_moe_expert_${Number(path[2]) + 1}_decoder`;
}
function countNumbers(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? 1 : 0;
  if (Array.isArray(value)) return value.reduce((sum, child) => sum + countNumbers(child), 0);
  return value && typeof value === 'object' ? Object.values(value).reduce((sum, child) => sum + countNumbers(child), 0) : 0;
}

export function evaluateMdpoWeightUpdateEvidence({ referenceNetwork, trainedNetwork, adapters, biases, graph, forward, varianceParameters = 0 } = {}) {
  if (!referenceNetwork || !trainedNetwork || !adapters || !biases || !graph || typeof forward !== 'function')
    throw new Error('MDPO weight-update evidence requires frozen reference, trained model, LoRA, graph and forward');
  const paths = v10LoraPaths(referenceNetwork);
  if (paths.length !== adapters.matrices?.length) throw new Error('MDPO LoRA update evidence has mismatched matrix count');
  const matrices = paths.map((parts, index) => {
    const base = at(referenceNetwork, parts), trained = at(trainedNetwork, parts);
    if (JSON.stringify(parts) !== JSON.stringify(adapters.matrices[index].path) || trained?.length !== base.length
        || trained.some((row, position) => !Array.isArray(row) || row.length !== base[position].length))
      throw new Error(`MDPO trained LoRA matrix shape/identity mismatch: ${parts.join('.')}`);
    let maximum = 0, l2Squared = 0, changed = 0, total = 0;
    for (let i = 0; i < base.length; i++) for (let j = 0; j < base[i].length; j++) {
      const delta = trained[i][j] - base[i][j];
      if (!Number.isFinite(delta)) throw new Error('Nonfinite MDPO LoRA update evidence');
      maximum = Math.max(maximum, Math.abs(delta)); l2Squared += delta * delta;
      if (delta !== 0) changed++;
      total++;
    }
    return { module: group(parts), path: parts.join('.'), rows: base.length, columns: base[0].length,
      maximum_absolute_update: maximum, l2_update: Math.sqrt(l2Squared), changed_weights: changed, total_weights: total };
  });
  const biasPaths = v10TrainableBiasPaths(referenceNetwork);
  const biasDeltas = biasPaths.map((parts, index) => {
    if (JSON.stringify(parts) !== JSON.stringify(biases.biases?.[index]?.path)) throw new Error('MDPO tuned bias identity mismatch');
    const base = at(referenceNetwork, parts), trained = at(trainedNetwork, parts), delta = biases.biases[index].delta;
    if (base.length !== trained?.length || base.length !== delta.length
        || base.some((item, position) => !Number.isFinite(trained[position]) || Math.abs(trained[position] - base[position] - delta[position]) > 1e-10))
      throw new Error('MDPO tuned bias does not match trained network');
    return { module: group(parts), path: parts.join('.'), maximum_absolute_update: Math.max(...delta.map(Math.abs)),
      changed_biases: delta.filter((item) => item !== 0).length, total_biases: delta.length };
  });
  const withoutLora = materializeV10BiasTuning(referenceNetwork, biases);
  const predicted = forward(trainedNetwork, graph).output, ablated = forward(withoutLora, graph).output;
  if (predicted.length !== ablated.length || predicted.some((row, node) => row.length !== ablated[node]?.length))
    throw new Error('MDPO LoRA ablation output shape differs');
  const outputChange = Math.max(...predicted.flatMap((row, node) => row.map((value, axis) => Math.abs(value - ablated[node][axis]))));
  if (!Number.isFinite(outputChange)) throw new Error('Nonfinite MDPO LoRA output ablation effect');
  const modules = Object.fromEntries([...new Set(matrices.map((row) => row.module))].map((name) => [name, {
    changed_matrices: matrices.filter((row) => row.module === name && row.changed_weights > 0).length,
    matrix_count: matrices.filter((row) => row.module === name).length,
    maximum_absolute_update: Math.max(...matrices.filter((row) => row.module === name).map((row) => row.maximum_absolute_update))
  }]));
  const trainable = adapters.trainable_parameters + biases.trainable_parameters + varianceParameters;
  const total = countNumbers(referenceNetwork) + trainable;
  return { version: 'v10_mdpo_weight_update_evidence_v1', matrix_count: matrices.length, matrices, bias_deltas: biasDeltas,
    modules, ablation: { comparison: 'trained_LoRA_plus_biases_vs_frozen_v10_plus_same_biases', sample: `${graph.category}/${graph.sample_id}`,
      maximum_absolute_local_6d_output_change: outputChange, effective: outputChange > 1e-10 },
    trainable_parameters: trainable, reference_numeric_parameters: total - trainable, total_numeric_parameters_including_adapters: total,
    trainable_fraction: trainable / total };
}
