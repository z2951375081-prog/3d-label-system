import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11AlignedGate } from '../lib/mdpo-activation-gate.mjs';
import { writeImmutableJsonBundle } from '../lib/immutable-json-bundle.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const sweep = path.join(mdpo, 'aligned_sweep');
const reports = path.join(mdpo, 'val11_reports_safety_priority_v3_aligned');
const ledger = JSON.parse(await fs.readFile(path.join(mdpo, 'aligned_val11_sweep_ledger.json'), 'utf8'));
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const expected = [5e-5, 1e-4, 2e-4].flatMap((learningRate) => [0.05, 0.10, 0.20].flatMap((beta) => [0.1, 0.3, 0.5].flatMap((lambdaMulti) => [2, 4, 8].map((rank) => ('lr_' + learningRate + '_beta_' + beta + '_multi_' + lambdaMulti + '_rank_' + rank).replaceAll('.', 'p')))));
const rows = [];
for (const id of expected) {
  const row = ledger.configurations.find((item) => item.id === id);
  if (!row || row.status !== 'evaluated') throw new Error('Aligned val11 is incomplete: ' + id);
  const reportBytes = await fs.readFile(path.join(root, row.report_file)), fourBytes = await fs.readFile(path.join(root, row.four_group_report_file));
  if (digest(reportBytes) !== row.report_sha256 || digest(fourBytes) !== row.four_group_report_sha256) throw new Error('Aligned val11 report hash mismatch: ' + id);
  const report = JSON.parse(reportBytes.toString('utf8')), candidateBytes = await fs.readFile(path.join(root, row.candidate_file)), candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (report.version !== 'v10_mdpo_val11_gate_report_v3' || report.evaluation_policy !== 'safety_priority_v3_aligned' || report.candidate_id !== id || report.candidate_model?.sha256 !== digest(candidateBytes) || candidate.mdpo?.safety_alignment?.protocol !== 'safety_priority_v3_aligned') throw new Error('Aligned selection provenance mismatch: ' + id);
  const gate = evaluateMdpoVal11AlignedGate({ baseline: report.baseline, candidate: report.candidate, trainingAlignment: report.training_alignment });
  if (JSON.stringify(gate) !== JSON.stringify(report.gate)) throw new Error('Aligned gate is not reproducible: ' + id);
  rows.push({ id, candidate_file: row.candidate_file, candidate_sha256: digest(candidateBytes), report_file: row.report_file, report_sha256: row.report_sha256, four_group_report_file: row.four_group_report_file, four_group_report_sha256: row.four_group_report_sha256, hyperparameters: row.hyperparameters, gate, score: { accepted: gate.accepted, safety_penalty: gate.safety_penalty, aesthetic_delta: gate.delta.aesthetic, composition_delta: gate.delta.composition_harmony, quality_score: report.candidate.metrics.quality_score } });
}
rows.sort((a, b) => Number(b.score.accepted) - Number(a.score.accepted) || a.score.safety_penalty - b.score.safety_penalty || b.score.aesthetic_delta - a.score.aesthetic_delta || b.score.composition_delta - a.score.composition_delta || b.score.quality_score - a.score.quality_score || a.id.localeCompare(b.id));
const selected = rows[0];
const output = { version: 'v10_mdpo_aligned_hyperparameter_selection_v1', generated_at: new Date().toISOString(), evaluation_policy: 'safety_priority_v3_aligned', status: 'selected_by_complete_aligned_val11', selected, val11_gate_complete: true, deployment_eligible: selected.gate.accepted === true, deployment_action: selected.gate.accepted ? 'candidate_may_proceed_to_separate_activation_audit' : 'retain_original_v10_selected_aligned_candidate_diagnostic_only', required_grid: ledger.required_grid, evaluated_configuration_count: rows.length, fixed_seed: ledger.fixed_seed, ranking_policy: ['accepted_gate_first', 'safety_penalty_asc', 'aesthetic_delta_desc', 'composition_delta_desc', 'PCK_advisory_only', 'quality_score_desc'], test_used_for_selection: false, ranked_candidates: rows };
const outputFile = path.join(mdpo, 'aligned_hyperparameter_selection.json');
await fs.writeFile(outputFile, JSON.stringify(output, null, 2) + '\n', { flag: 'w' });
if (!output.deployment_eligible) await fs.writeFile(path.join(mdpo, 'aligned_rejection_report.json'), JSON.stringify({ version: 'v10_mdpo_aligned_rejection_report_v1', generated_at: output.generated_at, evaluation_policy: 'safety_priority_v3_aligned', selected_candidate_id: selected.id, candidate_count: rows.length, rejected_candidates: rows.map((row) => ({ id: row.id, violations: row.gate.violations, safety_penalty: row.gate.safety_penalty })), original_v10_preserved: true, activation_skipped: true, test11_skipped: true }, null, 2) + '\n', { flag: 'w' });
console.log(JSON.stringify({ selected: selected.id, deployment_eligible: output.deployment_eligible, evaluation_policy: output.evaluation_policy, output: path.relative(root, outputFile).split(path.sep).join('/') }, null, 2));
