import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PREFERENCE_FEATURES, layoutFeatureVector } from '../lib/preference-model.mjs';
import { LLM_AESTHETIC_DIMENSIONS, LLM_AESTHETIC_WEIGHTS, LLM_SAFETY_DIMENSIONS, LLM_SCORE_NAMES, preferenceComposite } from '../public/preference-scoring.js';
import { annotationsToLabels } from './generate-artifacts.mjs';
import { validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { assessSafetyEligibility } from '../lib/preference-policy.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultInput = path.join(root, 'experiments', 'preferences.jsonl');
const defaultOutput = path.join(root, 'experiments', 'preference_model.json');
const LEGACY_SCORE_NAMES = ['readability', 'coverage', 'occlusion', 'balance', 'binocular_consistency', 'size_consistency', 'group_economy', 'overall'];

function parseArgs(argv) {
  const options = { input: defaultInput, output: defaultOutput, epochs: 80, learningRate: 0.01, seed: 17, source: 'all', runId: '' };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function random(seed) { let state = (Number(seed) >>> 0) || 17; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
function sigmoid(value) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value)))); }
function tanhDerivative(value) { return 1 - value * value; }
function dot(row, vector, bias) { return row.reduce((sum, weight, index) => sum + weight * vector[index], bias); }
function zeros(count) { return Array.from({ length: count }, () => 0); }
function matrix(rows, columns, fill = 0) { return Array.from({ length: rows }, () => Array.from({ length: columns }, () => fill)); }
function addInPlace(target, source, scale = 1) { for (let index = 0; index < target.length; index += 1) target[index] += source[index] * scale; }

async function loadExamples(inputFile, source = 'all', runId = '') {
  let text;
  try { text = await fs.readFile(inputFile, 'utf8'); } catch { throw new Error(`找不到偏好文件：${inputFile}。先在网页中保存至少一条候选 A/B 偏好。`); }
  const examples = [];
  const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
  let skippedNonTrain = 0;
  let skippedInvalidContract = 0;
  let skippedSource = 0;
  let skippedRun = 0;
  let skippedInvalidScoringEvidence = 0;
  let skippedUnsafeCrossing = 0;
  source = String(source || 'all');
  if (!['all', 'llm', 'human'].includes(source)) throw new Error('偏好来源必须是 all、llm 或 human');
  const manualCache = new Map();
  for (const line of text.split(/\r?\n/).filter(Boolean)) {
    let record;
    try { record = JSON.parse(line); } catch { continue; }
    const recordSource = record.type === 'llm_pairwise_preference' ? 'llm' : record.type === 'human_pairwise_preference' ? 'human' : 'unknown';
    if ((source !== 'all' && recordSource !== source) || recordSource === 'unknown') { skippedSource += 1; continue; }
    if (runId && record.run_id !== runId) { skippedRun += 1; continue; }
    if (recordSource === 'llm') {
      const evidence = record.scoring_evidence;
      if (!evidence?.chosen_model || evidence.chosen_model !== evidence.rejected_model) { skippedInvalidScoringEvidence += 1; continue; }
    }
    const sample = record.sample || record.chosen?.sample;
    const manifestSample = manifest.samples.find((item) => item.category === sample?.category && item.sample_id === String(sample?.sample_id));
    if (manifestSample?.split !== 'train' || record.split !== 'train' || record.chosen?.split !== 'train' || record.rejected?.split !== 'train') { skippedNonTrain += 1; continue; }
    const sameSample = (candidate) => candidate?.sample?.category === manifestSample.category && String(candidate?.sample?.sample_id) === String(manifestSample.sample_id);
    if (!sameSample(record.chosen) || !sameSample(record.rejected)) { skippedInvalidContract += 1; continue; }
    const manualKey = manifestSample.category + '/' + manifestSample.sample_id;
    if (!manualCache.has(manualKey)) {
      const annotation = JSON.parse(await fs.readFile(path.join(root, manifestSample.target.annotation_json), 'utf8'));
      manualCache.set(manualKey, annotationsToLabels(annotation));
    }
    try {
      validateFixedLabelContract(manualCache.get(manualKey), record.chosen.labels, manualKey + '/chosen');
      validateFixedLabelContract(manualCache.get(manualKey), record.rejected.labels, manualKey + '/rejected');
    } catch { skippedInvalidContract += 1; continue; }
    const chosen = record.chosen?.metrics;
    const rejected = record.rejected?.metrics;
    if (!chosen || !rejected) continue;
    if (recordSource === 'llm' && (!assessSafetyEligibility(chosen, chosen).eligible || !assessSafetyEligibility(rejected, rejected).eligible)) { skippedUnsafeCrossing += 1; continue; }
    const complete = (scores, names) => scores && names.every((key) => Number.isFinite(Number(scores[key])) && Number(scores[key]) >= 1 && Number(scores[key]) <= 5);
    const normalizedScores = (scores) => {
      if (complete(scores, LLM_SCORE_NAMES)) return Object.fromEntries(LLM_SCORE_NAMES.map((key) => [key, Number(scores[key]) / 5]));
      if (process.argv.includes("--allowLegacy") && complete(scores, LEGACY_SCORE_NAMES)) return Object.fromEntries(LEGACY_SCORE_NAMES.map((key) => [key, Number(scores[key]) / 5]));
      return null;
    };
    const chosenScores = normalizedScores(record.chosen?.llm_scores);
    const rejectedScores = normalizedScores(record.rejected?.llm_scores);
    if (recordSource === 'llm' && (!chosenScores || !rejectedScores)) { skippedInvalidScoringEvidence += 1; continue; }
    // LLM scores are supervision strength, never inference inputs (test has no LLM labels).
    const composite = (scores) => scores.text_clarity === undefined
      ? 0.6 * scores.overall + 0.4 * LEGACY_SCORE_NAMES.filter((key) => key !== "overall").reduce((sum, key) => sum + scores[key], 0) / 7
      : preferenceComposite(scores);
    const scoreGap = chosenScores && rejectedScores ? Math.max(0, composite(chosenScores) - composite(rejectedScores)) : 0;
    const weight = chosenScores && rejectedScores ? 1 + 2 * scoreGap : 1;
    examples.push({ source: recordSource, chosen: layoutFeatureVector(chosen), rejected: layoutFeatureVector(rejected), llm_scores: { chosen: chosenScores, rejected: rejectedScores }, llm_composite_gap: scoreGap, weight });
  }
  if (!examples.length) throw new Error('偏好文件中没有包含 chosen.metrics / rejected.metrics 的有效样本。');
  return { examples, skippedNonTrain, skippedInvalidContract, skippedSource, skippedRun, skippedInvalidScoringEvidence, skippedUnsafeCrossing, source, runId };
}

function initModel(inputDim, hiddenDim, rng) {
  return { status: 'trained_pairwise_mlp', architecture: { type: 'pairwise_mlp', input_dim: inputDim, hidden_dim: hiddenDim, output_dim: 1 }, feature_names: PREFERENCE_FEATURES, weights: { w1: matrix(hiddenDim, inputDim).map((row) => row.map(() => (rng() - 0.5) * 0.08)), b1: zeros(hiddenDim), w2: Array.from({ length: hiddenDim }, () => (rng() - 0.5) * 0.08), b2: 0 } };
}

function forward(model, x) { const hidden = model.weights.w1.map((row, index) => Math.tanh(dot(row, x, model.weights.b1[index]))); return { hidden, score: dot(model.weights.w2, hidden, model.weights.b2) }; }

function trainStep(model, chosen, rejected, learningRate, weightDecay, weight = 1) {
  const left = forward(model, chosen);
  const right = forward(model, rejected);
  const difference = left.score - right.score;
  const probability = sigmoid(difference);
  const loss = -Math.log(Math.max(probability, 1e-8)) * weight;
  const outputGradient = (probability - 1) * weight;
  const leftHiddenGradient = model.weights.w2.map((weight, index) => outputGradient * weight * tanhDerivative(left.hidden[index]));
  const rightHiddenGradient = model.weights.w2.map((weight, index) => -outputGradient * weight * tanhDerivative(right.hidden[index]));
  for (let hidden = 0; hidden < model.weights.w1.length; hidden += 1) {
    for (let feature = 0; feature < model.weights.w1[hidden].length; feature += 1) {
      model.weights.w1[hidden][feature] -= learningRate * (leftHiddenGradient[hidden] * chosen[feature] + rightHiddenGradient[hidden] * rejected[feature] + weightDecay * model.weights.w1[hidden][feature]);
    }
    model.weights.b1[hidden] -= learningRate * (leftHiddenGradient[hidden] + rightHiddenGradient[hidden]);
    model.weights.w2[hidden] -= learningRate * (outputGradient * (left.hidden[hidden] - right.hidden[hidden]) + weightDecay * model.weights.w2[hidden]);
  }
  model.weights.b2 -= learningRate * outputGradient;
  return { loss, preferred: probability > 0.5 };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const { examples, skippedNonTrain, skippedInvalidContract, skippedSource, skippedRun, skippedInvalidScoringEvidence, skippedUnsafeCrossing, source, runId } = await loadExamples(path.resolve(String(options.input)), options.source, String(options.runId || ''));
  const epochs = Math.max(1, Number(options.epochs));
  const learningRate = Math.max(1e-5, Number(options.learningRate));
  const rng = random(options.seed);
  const model = initModel(PREFERENCE_FEATURES.length, 32, rng);
  model.scoring_protocol = process.argv.includes("--allowLegacy") ? "historical_eight_dimension_exploratory" : "fourteen_dimension_aesthetic_reward_v3";
  model.rerank_policy = 'deterministic_energy_safety_gate_then_aesthetic_reward';
  const history = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    let loss = 0;
    let correct = 0;
    for (const example of examples) { const result = trainStep(model, example.chosen, example.rejected, learningRate, 1e-4, example.weight); loss += result.loss; correct += result.preferred ? 1 : 0; }
    history.push({ epoch: epoch + 1, loss: Number((loss / examples.length).toFixed(6)), pair_accuracy: Number((correct / examples.length).toFixed(6)) });
  }
  const scoredExamples = examples.filter((example) => example.llm_scores.chosen && example.llm_scores.rejected);
  const output = { ...model, generated_at: new Date().toISOString(), training: { examples: examples.length, epochs, learning_rate: learningRate, seed: Number(options.seed), source_filter: source, run_id_filter: runId || null, source_counts: { llm: examples.filter((example) => example.source === 'llm').length, human: examples.filter((example) => example.source === 'human').length }, split_policy: 'manifest_train_only', skipped_non_train: skippedNonTrain, skipped_invalid_contract: skippedInvalidContract, skipped_source: skippedSource, skipped_run: skippedRun, skipped_invalid_scoring_evidence: skippedInvalidScoringEvidence, llm_scored_pairs: scoredExamples.length, llm_score_dimensions: LLM_SCORE_NAMES, llm_aesthetic_dimensions_used_for_preference: LLM_AESTHETIC_DIMENSIONS, llm_safety_dimensions_diagnostic_only: LLM_SAFETY_DIMENSIONS, llm_composite_weights: LLM_AESTHETIC_WEIGHTS, llm_composite_formula: '0.30*overall + 0.25*composition_harmony + 0.20*visual_hierarchy + 0.15*spatial_balance + 0.10*manual_style_similarity', mean_llm_composite_gap: scoredExamples.length ? Number((scoredExamples.reduce((sum, example) => sum + example.llm_composite_gap, 0) / scoredExamples.length).toFixed(6)) : null, llm_scores_usage: 'Only five aesthetic dimensions select the winner and weight pairwise gradients. Nine safety/legibility dimensions remain diagnostics; deterministic five-view energy and the safety gate control them before reward reranking. Inference never consumes LLM scores or manual target coordinates.', evaluation_note: 'pair accuracy is training fit, not held-out generalization', final: history.at(-1), history } };
  output.training.skipped_unsafe_leader_crossing = skippedUnsafeCrossing;
  if (process.argv.includes('--allowLegacy')) {
    output.training.llm_score_dimensions = LEGACY_SCORE_NAMES;
    output.training.llm_composite_formula = '0.6*overall + 0.4*mean(seven_historical_dimensions)';
    output.training.llm_scores_usage = 'Historical eight dimensions only; no fabricated penetration or manual-style scores. Exploratory reanalysis, not current twelve-dimension supervision.';
  }
  await fs.mkdir(path.dirname(path.resolve(String(options.output))), { recursive: true });
  await fs.writeFile(path.resolve(String(options.output)), `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  console.log(`完成：${examples.length} 条偏好对，最终 pair accuracy=${output.training.final.pair_accuracy}`);
  console.log(`输出：${path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/')}`);
}

main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });
