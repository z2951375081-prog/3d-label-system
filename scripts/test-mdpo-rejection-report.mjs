import assert from 'node:assert/strict';
import { buildMdpoRejectionReport } from '../lib/mdpo-rejection-report.mjs';

const hashes = { selectionSha256: 'a'.repeat(64), referenceSha256: 'b'.repeat(64), historicalRewardSha256: 'c'.repeat(64) };
const rows = Array.from({ length: 81 }, (_, index) => ({ id: `run_${index}`, candidate_sha256: 'd'.repeat(64),
  report_sha256: 'e'.repeat(64), gate: { accepted: false, status: 'rejected', deployment_status: 'diagnostic_only',
    violations: ['gate:aesthetic_gain'], delta: { aesthetic: -0.01 } } }));
const selection = { version: 'v10_mdpo_hyperparameter_selection_v1', val11_gate_complete: true, deployment_eligible: false,
  deployment_action: 'retain_original_v10_selected_candidate_diagnostic_only', test_used_for_selection: false,
  evaluated_configuration_count: 81, required_grid: { total: 81 }, ranked_candidates: rows, selected: rows[0] };
const report = buildMdpoRejectionReport(selection, hashes);
assert.equal(report.candidate_count, 81);
assert.equal(report.accepted_count, 0);
assert.equal(report.selected_status, 'diagnostic_only');
assert.equal(report.test11_not_run, true);
assert.deepEqual(report.rejected_candidates[0].violations, ['gate:aesthetic_gain']);
assert.throws(() => buildMdpoRejectionReport({ ...selection, ranked_candidates: rows.slice(1) }, hashes), /81-candidate/);
assert.throws(() => buildMdpoRejectionReport({ ...selection, ranked_candidates: rows.map((row, index) => index === 1 ? { ...row, gate: { ...row.gate, accepted: true } } : row) }, hashes), /81-candidate/);
assert.throws(() => buildMdpoRejectionReport({ ...selection, selected: rows[1] }, hashes), /81-candidate/);
assert.throws(() => buildMdpoRejectionReport(selection, { ...hashes, referenceSha256: '' }), /81-candidate/);
console.log('v10-MDPO immutable 81-candidate rejection evidence tests passed.');
