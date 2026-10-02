import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { LLM_SCORE_NAMES } from '../public/preference-scoring.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experimentsRoot = path.join(root, 'experiments');
const temporary = await fs.mkdtemp(path.join(experimentsRoot, '.convergence-test-'));
const runId = 'fixture_visual_convergence_20260918';
const rounds = [0, 1, 2, 4, 8];
const cohort = Array.from({ length: 11 }, (_, index) => ({ category: `Category${index + 1}`, sample_id: String(1000 + index) }));

function scoreRound(round) {
  const value = ({ 0: 3, 1: 3.1, 2: 3.2, 4: 3.3, 8: 3.35 })[round];
  return { value, scores: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, value])) };
}

const ledger = { version: 'llm_preference_checkpoint_ledger_v1', run_id: runId, checkpoints: [] };
const visual = { version: 'llm_visual_validation_v1', run_id: runId, split: 'val', rounds: [] };
for (const round of rounds) {
  const scored = scoreRound(round);
  let checkpoint = { run_id: runId, round: 0, kind: 'no_reward_baseline', reward_model_file: null, reward_model_sha256: null };
  if (round > 0) {
    const directory = path.join(temporary, 'llm_preference_checkpoints', runId);
    await fs.mkdir(directory, { recursive: true });
    const modelFile = path.join(directory, `round_${round}.json`);
    const bytes = Buffer.from(JSON.stringify({ version: 'fixture_reward', training: { run_id_filter: runId, source_filter: 'llm', round } }, null, 2) + '\n');
    await fs.writeFile(modelFile, bytes);
    const digest = createHash('sha256').update(bytes).digest('hex').toUpperCase();
    checkpoint = { run_id: runId, round, kind: 'frozen_reward_checkpoint', reward_model_file: path.relative(root, modelFile).split(path.sep).join('/'), reward_model_sha256: digest };
    ledger.checkpoints.push({ round, reward_model_file: checkpoint.reward_model_file, reward_model_sha256: digest, validation_status: 'accepted', activated: true });
  }
  visual.rounds.push({
    round, status: 'evaluated', sample_count: 11, scorer_model: 'fixture-vision-model', checkpoint,
    score_means: scored.scores, composite_score: scored.value,
    geometry_proxy_means: { multidimensional_quality_score: 4 + round / 100 },
    sample_scores: cohort.map((sample) => ({ sample, scores: scored.scores, composite_score: scored.value, response_id: `fixture-${round}-${sample.sample_id}` }))
  });
}
await fs.writeFile(path.join(temporary, `llm_visual_validation_${runId}.json`), JSON.stringify(visual, null, 2) + '\n');
await fs.writeFile(path.join(temporary, `llm_preference_checkpoints_${runId}.json`), JSON.stringify(ledger, null, 2) + '\n');
await fs.writeFile(path.join(temporary, `llm_preference_run_${runId}.json`), JSON.stringify({ version: 'fixture_run', run_id: runId }, null, 2) + '\n');
await fs.writeFile(path.join(temporary, 'preference_test11_report.json'), JSON.stringify({ run_id: runId, selected_round: 8, selection_policy: 'best_accepted_val_visual_quality', test_samples: 11, candidate_activated: true, summary: {} }, null, 2) + '\n');

const output = path.join(temporary, 'convergence.json');
const child = spawn(process.execPath, ['scripts/analyze-preference-convergence.mjs', '--runId', runId, '--output', output], { cwd: root, windowsHide: true, env: { ...process.env, EXPERIMENTS_DIR: temporary } });
let stdout = '', stderr = '';
child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('close', resolve); });

try {
  assert.equal(code, 0, stderr || stdout);
  const report = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.equal(report.version, 'llm_preference_convergence_v4_aesthetic_reward_safety_gate');
  assert.equal(report.policy.reward_target, 'aesthetic_dimensions_only');
  assert.equal(report.policy.deterministic_energy_safety_gate, true);
  assert.equal(report.evidence_sufficient, true);
  assert.equal(report.improved_before_plateau, true);
  assert.equal(report.plateau_round4_to_round8, true);
  assert.equal(report.rise_then_stable, true);
  assert.deepEqual(report.checkpoints.map((item) => item.round), rounds);
  assert.ok(report.checkpoints.every((item) => item.sample_count === 11 && item.checkpoint_verified));
  assert.equal(report.security.external_requests_made_by_this_analysis, 0);
  assert.equal(report.security.geometry_proxy_used_as_visual_convergence_evidence, false);
  console.log(JSON.stringify({ status: 'passed', evidence_sufficient: report.evidence_sufficient, rise_then_stable: report.rise_then_stable, checkpoints: report.checkpoints.length }));
} finally {
  const resolved = path.resolve(temporary), prefix = path.resolve(experimentsRoot) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('.convergence-test-')) throw new Error('Refusing unsafe convergence test cleanup target');
  await fs.rm(resolved, { recursive: true, force: true });
}
