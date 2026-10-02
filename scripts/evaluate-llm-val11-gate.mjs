import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluateVisualActivationGate, inferGeometrySafetyEligibility, LLM_VISUAL_ACTIVATION_THRESHOLDS } from '../lib/llm-visual-activation-gate.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const visual = JSON.parse(await fs.readFile(path.join(experiments, 'llm_visual_validation_latest.json'), 'utf8'));
const ledger = JSON.parse(await fs.readFile(path.join(experiments, 'llm_preference_checkpoints_latest.json'), 'utf8'));
const active = JSON.parse(await fs.readFile(path.join(experiments, 'preference_model.json'), 'utf8'));
const baseline = visual.rounds.find((row) => Number(row.round) === 0);
if (!baseline) throw new Error('latest visual validation has no round-0 val11 baseline');
const checkpoints = new Map((ledger.checkpoints || []).map((row) => [Number(row.round), row]));
const rounds = visual.rounds.map((row) => {
  if (Number(row.round) === 0) return { round: 0, role: 'no_reward_baseline', sample_count: row.sample_count, scorer_model: row.scorer_model, composite_score: row.composite_score, score_means: row.score_means, activation_gate: null };
  const checkpoint = checkpoints.get(Number(row.round));
  const gate = evaluateVisualActivationGate({ baseline, candidate: row, geometryEligible: inferGeometrySafetyEligibility(checkpoint?.validation) });
  return { round: Number(row.round), role: 'qwen_reward_checkpoint', sample_count: row.sample_count, scorer_model: row.scorer_model, reward_model_file: checkpoint?.reward_model_file || null, reward_model_sha256: checkpoint?.reward_model_sha256 || null, composite_score: row.composite_score, score_means: row.score_means, geometry_safety_eligible: gate.constraints.geometry_safety_eligible, activation_gate: gate };
});
const eligible = rounds.filter((row) => row.activation_gate?.accepted);
const best = [...rounds.filter((row) => row.round > 0)].sort((a, b) => b.composite_score - a.composite_score)[0] || null;
const report = {
  version: 'llm_val11_five_dimension_gate_comparison_v1', generated_at: new Date().toISOString(), run_id: visual.run_id,
  split: 'val', sample_count: baseline.sample_count, baseline_round: 0,
  formula: '0.30*overall + 0.25*composition_harmony + 0.20*visual_hierarchy + 0.15*spatial_balance + 0.10*manual_style_similarity',
  thresholds: LLM_VISUAL_ACTIVATION_THRESHOLDS,
  rounds,
  accepted_rounds: eligible.map((row) => row.round),
  best_trained_round: best ? { round: best.round, composite_score: best.composite_score, aesthetic_composite_gain: best.activation_gate.constraints.aesthetic_composite_gain, composition_harmony_change: best.activation_gate.constraints.composition_harmony_change, status: best.activation_gate.status } : null,
  active_reward: { file: 'experiments/preference_model.json', status: active.status, training_run_id: active.training?.run_id_filter || null, architecture: active.architecture, feature_names: active.feature_names, validation_gate: active.validation_gate },
  interpretation: eligible.length ? `Rounds ${eligible.map((row) => row.round).join(', ')} satisfy the new real-val11 five-dimension activation gate.` : 'No trained checkpoint in the latest run improves the real val11 five-dimension aesthetic composite by 0.02 while keeping composition harmony within -0.01 of round 0.'
};
const output = path.join(experiments, 'llm_val11_gate_comparison.json');
await fs.writeFile(output, JSON.stringify(report, null, 2) + '\n', 'utf8');
console.log(JSON.stringify({ output: path.relative(root, output), run_id: report.run_id, accepted_rounds: report.accepted_rounds, best_trained_round: report.best_trained_round }, null, 2));
