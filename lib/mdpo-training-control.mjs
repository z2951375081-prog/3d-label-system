import { MDPO_WEIGHTS } from './mdpo-continuous-policy.mjs';

export const MDPO_UI_TRAINING_DEFAULTS = Object.freeze({
  model: 'v10_mdpo', epochs: 30, learningRate: 1e-4, beta: 0.1, lambdaMulti: 0.3,
  lambdaDpo: 1, lambdaKl: 0.01, lambdaSup: 0.2, lambdaSafe: 1, lambdaCenterTail: 0.1,
  rank: 4, alpha: 8, dropout: 0.05, clipNorm: 1, patience: 6, seed: 17017,
  learnableVariance: false, noReference: false,
  compositionWeight: MDPO_WEIGHTS.composition_harmony, hierarchyWeight: MDPO_WEIGHTS.visual_hierarchy,
  balanceWeight: MDPO_WEIGHTS.spatial_balance, manualStyleWeight: MDPO_WEIGHTS.manual_style_similarity,
  textWeight: MDPO_WEIGHTS.text_clarity, leaderWeight: MDPO_WEIGHTS.leader_line_clarity
});

const FINITE_FIELDS = Object.freeze({
  epochs: [1, 500, true], learningRate: [1e-7, 1, false], beta: [0.001, 10, false],
  lambdaMulti: [0, 20, false], lambdaDpo: [0, 20, false], lambdaKl: [0, 20, false],
  lambdaSup: [0, 20, false], lambdaSafe: [0.001, 50, false], lambdaCenterTail: [0, 20, false],
  rank: [1, 64, true], alpha: [0.01, 256, false], dropout: [0, 0.95, false],
  clipNorm: [0.01, 100, false], patience: [1, 100, true], seed: [0, 2147483647, true],
  compositionWeight: [0, 10, false], hierarchyWeight: [0, 10, false],
  balanceWeight: [0, 10, false], manualStyleWeight: [0, 10, false],
  textWeight: [0, 10, false], leaderWeight: [0, 10, false]
});

export function normalizeMdpoUiTrainingRequest(payload = {}) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('训练参数必须是 JSON 对象');
  if (payload.model !== undefined && payload.model !== 'v10_mdpo') throw new Error('当前训练控制台只允许选择 v10-MDPO');
  const allowed = new Set([...Object.keys(MDPO_UI_TRAINING_DEFAULTS)]);
  for (const name of Object.keys(payload)) if (!allowed.has(name)) throw new Error('不允许的训练参数：' + name);
  const normalized = { ...MDPO_UI_TRAINING_DEFAULTS, model: 'v10_mdpo' };
  for (const [name, [minimum, maximum, integer]] of Object.entries(FINITE_FIELDS)) {
    const value = payload[name] === undefined ? normalized[name] : Number(payload[name]);
    if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value)))
      throw new Error('参数 ' + name + ' 必须在 ' + minimum + '–' + maximum + ' 范围内' + (integer ? '且为整数' : ''));
    normalized[name] = value;
  }
  normalized.learnableVariance = payload.learnableVariance === true || String(payload.learnableVariance).toLowerCase() === 'true';
  normalized.noReference = payload.noReference === true || String(payload.noReference).toLowerCase() === 'true';
  if (normalized.noReference && normalized.lambdaKl !== 0) throw new Error('关闭冻结参考策略时，λ KL 必须设为 0');
  const multiWeight = ['compositionWeight', 'hierarchyWeight', 'balanceWeight', 'manualStyleWeight', 'textWeight', 'leaderWeight']
    .reduce((sum, name) => sum + normalized[name], 0);
  if (!(multiWeight > 0) && normalized.lambdaMulti > 0) throw new Error('启用多维 MDPO 时，六个多维权重之和必须大于 0');
  return normalized;
}

export function mdpoUiTrainingArgs(options, outputDir) {
  const args = ['scripts/train-v10-mdpo.mjs', '--outputDir', outputDir];
  for (const [name, value] of Object.entries(options)) {
    if (name === 'model') continue;
    args.push('--' + name, String(value));
  }
  return args;
}
