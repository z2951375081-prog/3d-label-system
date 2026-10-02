import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { PREFERENCE_FEATURES } from '../lib/preference-model.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');

function aestheticModel(sign) {
  const row = Array.from({ length: PREFERENCE_FEATURES.length }, () => 0);
  row[PREFERENCE_FEATURES.indexOf('aesthetic_composition_harmony')] = sign * 20;
  return {
    status: 'trained_pairwise_mlp',
    architecture: { type: 'pairwise_mlp', input_dim: PREFERENCE_FEATURES.length, hidden_dim: 1, output_dim: 1 },
    feature_names: PREFERENCE_FEATURES,
    weights: { w1: [row], b1: [0], w2: [1], b2: 0 },
    scoring_protocol: 'fourteen_dimension_aesthetic_reward_v3',
    rerank_policy: 'deterministic_energy_safety_gate_then_aesthetic_reward',
    training: { source_filter: 'llm', run_id_filter: `gate-fixture-${sign}`, source_counts: { llm: 1, human: 0 } }
  };
}

async function runGate(sign) {
  const temporary = await fs.mkdtemp(path.join(experiments, '.preference-gate-'));
  try {
    const candidate = path.join(temporary, 'candidate.json');
    const active = path.join(temporary, 'active.json');
    const selection = path.join(temporary, 'selection.json');
    const testReport = path.join(temporary, 'test.json');
    await fs.writeFile(candidate, JSON.stringify(aestheticModel(sign), null, 2));
    const run = spawnSync(process.execPath, [
      'scripts/select-preference-model.mjs',
      '--candidate', candidate,
      '--active', active,
      '--manifest', path.join(experiments, 'dataset_manifest.json'),
      '--layoutModel', path.join(experiments, 'layout_model.json'),
      '--output', selection,
      '--testOutput', testReport,
      '--iterations', '20',
      '--limit', '1'
    ], { cwd: root, encoding: 'utf8', windowsHide: true });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    return { selection: JSON.parse(await fs.readFile(selection, 'utf8')), activeExists: await fs.stat(active).then(() => true, () => false) };
  } finally {
    const resolved = path.resolve(temporary), prefix = path.resolve(experiments) + path.sep;
    if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('.preference-gate-')) throw new Error('Refusing unsafe preference gate cleanup target');
    await fs.rm(resolved, { recursive: true });
  }
}

const preferLowAesthetic = await runGate(-1);
assert.ok(preferLowAesthetic.selection.validation.constraints.aesthetic_gain === null || typeof preferLowAesthetic.selection.validation.constraints.aesthetic_gain === 'number');
assert.equal(preferLowAesthetic.selection.activated, false, 'LLM reward must not activate before real visual val11');
assert.equal(preferLowAesthetic.activeExists, false);
assert.match(preferLowAesthetic.selection.validation.criterion, /real val11 six-image Qwen five-dimension/);

const preferHighAesthetic = await runGate(1);
assert.ok(preferHighAesthetic.selection.validation.constraints.aesthetic_gain === null || typeof preferHighAesthetic.selection.validation.constraints.aesthetic_gain === 'number');
assert.equal(preferHighAesthetic.selection.activated, false, 'geometry eligibility alone must not activate an LLM reward');
assert.equal(preferHighAesthetic.activeExists, false, 'pending visual-val11 reward candidate must never create an active model');
assert.equal(preferHighAesthetic.selection.validation.visual_activation_pending, preferHighAesthetic.selection.validation.geometry_safety_eligible);
assert.match(preferHighAesthetic.selection.validation.criterion, /real val11 six-image Qwen five-dimension/);
console.log('Preference validation gate passed: LLM candidates remain pending until real visual val11 activation.');
