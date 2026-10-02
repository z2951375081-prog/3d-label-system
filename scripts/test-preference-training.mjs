import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { evaluateLayout, repairLeaderCrossings } from '../lib/layout-optimizer.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';
import { PREFERENCE_FEATURES, layoutFeatureVector, predictPreference } from '../lib/preference-model.mjs';
import { LLM_SCORE_NAMES } from '../public/preference-scoring.js';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experimentDir = path.join(root, 'experiments');
const temporary = await fs.mkdtemp(path.join(experimentDir, '.preference-test-'));
const manifest = JSON.parse(await fs.readFile(path.join(experimentDir, 'dataset_manifest.json'), 'utf8'));
const train = manifest.samples.find((item) => item.split === 'train');
const val = manifest.samples.find((item) => item.split === 'val');
async function samplePair(sample) {
  const annotation = JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8'));
  const labels = annotationsToLabels(annotation);
  const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
  const bounds = boundsFromObj(cleanObj(raw).text);
  const changed = labels.map((label, index) => ({ ...label, center: [label.center[0] + (index + 1) * bounds.radius * 0.06, ...label.center.slice(1)] }));
  const options = { viewPolicy: 'binocular' };
  const safeLabels = repairLeaderCrossings(labels, bounds, { viewPolicy: 'binocular', seed: 17 }).labels;
  const safeChanged = repairLeaderCrossings(changed, bounds, { viewPolicy: 'binocular', seed: 19 }).labels;
  const evaluated = (candidate) => {
    const metrics = evaluateLayout(candidate, bounds, { ...options, manualReference: labels });
    const crossing = evaluateViewConditionedLayout(candidate, bounds);
    Object.assign(metrics, {
      worst_view_leader_crossing_count: crossing.worst_view_leader_crossing_count,
      weighted_leader_crossing_risk: crossing.weighted_leader_crossing_risk,
      worst_view_leader_crossing_risk: crossing.worst_view_leader_crossing_risk,
      cvar_view_leader_crossing_risk: crossing.cvar_view_leader_crossing_risk
    });
    return metrics;
  };
  const chosenMetrics = evaluated(safeLabels);
  const rejectedMetrics = evaluated(safeChanged);
  const chosenScores = Object.fromEntries(LLM_SCORE_NAMES.map((key) => [key, 5]));
  const rejectedScores = Object.fromEntries(Object.keys(chosenScores).map((key) => [key, 2]));
  const identity = { category: sample.category, sample_id: sample.sample_id };
  return { type: 'llm_pairwise_preference', run_id: 'fixture-run-01', scoring_evidence: { chosen_model: 'fixture-vision', rejected_model: 'fixture-vision', chosen_receipt_id: 'fixture-receipt-a', rejected_receipt_id: 'fixture-receipt-b' }, sample: identity, split: sample.split, chosen: { sample: identity, split: sample.split, labels: safeLabels, metrics: chosenMetrics, llm_scores: chosenScores }, rejected: { sample: identity, split: sample.split, labels: safeChanged, metrics: rejectedMetrics, llm_scores: rejectedScores } };
}
try {
  const valid = await samplePair(train);
  assert.equal(PREFERENCE_FEATURES.length, 12);
  const vector = layoutFeatureVector(valid.chosen.metrics);
  assert.equal(vector.length, 12);
  assert.deepEqual(PREFERENCE_FEATURES.slice(-5), ['aesthetic_leader_length_mean', 'aesthetic_leader_length_spread', 'aesthetic_spacing_consistency', 'aesthetic_text_size_fit', 'aesthetic_composition_harmony']);
  assert.equal(PREFERENCE_FEATURES.some((name) => /occlusion|penetration|overflow|objective|clarity|crossing/.test(name)), false, 'reward input must not duplicate deterministic safety penalties');
  assert.ok(vector.at(-3) >= 0 && vector.at(-3) <= 1, 'candidate-only spacing aesthetics must reach reward');
  assert.ok(vector.at(-2) >= 0 && vector.at(-2) <= 1, 'candidate-only text-size aesthetics must reach reward');
  assert.ok(vector.at(-1) > 0, 'candidate-only composition harmony must reach reward');
  const withoutManual = { ...valid.chosen.metrics, manual_similarity: null, multidimensional_quality_score: -999, manual_center_distance_norm: 999, manual_style_distance: 999 };
  assert.deepEqual(layoutFeatureVector(withoutManual), vector, 'changing or removing manual targets must never change reward inputs');
  const invalid = structuredClone(valid); invalid.chosen.labels[0].anchor[0] += 1;
  const outside = await samplePair(val);
  const human = structuredClone(valid); human.type = 'human_pairwise_preference'; delete human.scoring_evidence;
  const otherRun = structuredClone(valid); otherRun.run_id = 'fixture-run-02';
  const input = path.join(temporary, 'preference-input.jsonl');
  const output = path.join(temporary, 'preference-model.json');
  await fs.writeFile(input, [valid, invalid, outside, human, otherRun].map((row) => JSON.stringify(row)).join(String.fromCharCode(10)));
  const run = spawnSync(process.execPath, ['scripts/train-preference-model.mjs', '--input', input, '--output', output, '--epochs', '5', '--source', 'llm', '--runId', 'fixture-run-01'], { cwd: root, encoding: 'utf8', windowsHide: true });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  const model = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.equal(model.training.examples, 1);
  assert.equal(model.training.skipped_non_train, 1);
  assert.equal(model.training.skipped_invalid_contract, 1);
  assert.equal(model.training.skipped_source, 1);
  assert.equal(model.training.skipped_run, 1);
  assert.equal(model.training.source_filter, 'llm');
  assert.equal(model.training.run_id_filter, 'fixture-run-01');
  assert.equal(model.training.source_counts.human, 0);
  assert.equal(model.training.llm_scored_pairs, 1);
  assert.equal(model.training.llm_composite_formula.startsWith('0.30*overall'), true);
  assert.equal(model.training.mean_llm_composite_gap, 0.6);
  assert.equal(model.architecture.input_dim, 12);
  assert.equal(model.architecture.hidden_dim, 32);
  assert.equal(predictPreference(model, valid.chosen.metrics).feature_dim, 12);
  console.log('Preference train fixture passed: 12D aesthetic-only reward, deterministic safety separation, train-only and anchor protection.');
} finally {
  const resolved = path.resolve(temporary), prefix = path.resolve(experimentDir) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('.preference-test-')) throw new Error('Refusing unsafe test cleanup target');
  await fs.rm(resolved, { recursive: true });
}
