import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { MDPO_ABLATION_IDS, validateMdpoAblationEvidence } from '../lib/mdpo-ablation-evidence.mjs';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { MDPO_VAL11_GROUPS, validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-ablation-evidence-test-'));
const mdpo = path.join(root, 'experiments', 'mdpo');
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const referenceSha256 = 'a'.repeat(64), datasetSha256 = 'b'.repeat(64), rewardSha256 = 'c'.repeat(64);
const cohort = Array.from({ length: 11 }, (_, index) => `C/${index}`);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4,
  manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .3, OLR: .1, LCD: 0, avg_leader_length: 1,
  overlap_pairs: 0, occluded_points: 0, intersections: 0, quality_score: 4,
  worst_view_intersections: 0, object_occlusion: 0, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: 0 };
const views = ['before', 'main', 'right', 'left', 'up', 'down'], viewHash = 'd'.repeat(64);
const selection = { selected: { id: 'selected_grid' } };
const ledger = { test_used_for_selection: false, configurations: [] }, ablations = [];
try {
  for (const id of MDPO_ABLATION_IDS) {
    const candidateId = `ablation_${id}`, candidateFile = path.join(mdpo, 'ablations', id, 'layout_model_v10_mdpo_candidate.json');
    await fs.mkdir(path.dirname(candidateFile), { recursive: true });
    const candidate = { version: 'layout_model_v10_mdpo_candidate', reference: { sha256: referenceSha256 },
      mdpo: { dataset_sha256: datasetSha256, weight_update_evidence: { ablation: { effective: true } } } };
    const candidateBytes = Buffer.from(JSON.stringify(candidate)), candidateSha256 = digest(candidateBytes);
    await fs.writeFile(candidateFile, candidateBytes);
    const candidateRelative = path.relative(root, candidateFile).split(path.sep).join('/');
    ledger.configurations.push({ id, status: 'trained', candidate_file: candidateRelative });
    const groupRows = {};
    for (const [name, spec] of Object.entries(MDPO_VAL11_GROUPS)) {
      const groupScores = spec.model_role === 'candidate' ? { ...scores, overall: 4.2 } : scores;
      const sampleScores = cohort.map((sample, index) => {
        const entry = { sample: { category: 'C', sample_id: String(index) }, scores: groupScores, metrics,
          response_id: `${candidateId}-${name}-${index}`, view_sha256: Object.fromEntries(views.map((view) => [view, viewHash])),
          generation_strategy: { generator: 'v10', viewPolicy: 'five_views', groupPolicy: 'all', sizePolicy: 'relative',
            optimizer: 'annealing', seed: 17017 + index * 4, iterations: 180, preferenceRerank: spec.preference_rerank },
          evaluation: { candidate_id: candidateId, group: name, role: spec.model_role, reference_model_sha256: referenceSha256,
            candidate_sha256: spec.model_role === 'baseline' ? referenceSha256 : candidateSha256,
            candidate_file: spec.model_role === 'baseline' ? null : candidateRelative },
          preference_model_sha256: spec.preference_rerank ? rewardSha256 : null,
          security: { test_used: false, train_preference_created: false, qwen_inference_input: false } };
        return entry;
      });
      groupRows[name] = { group: name, name: spec.label, label: spec.label, model_role: spec.model_role,
        preference_rerank: spec.preference_rerank, preference_model_sha256: spec.preference_rerank ? rewardSha256 : null,
        sample_count: 11, cohort, scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'fixed-mdpo-v1',
        views, metric_protocol: 'fixed-metric-v1', test_used_for_selection: false, score_means: groupScores,
        metrics: { ...metrics, text_clarity: 4, leader_line_clarity: 4 }, sample_scores: sampleScores };
      const directory = path.join(mdpo, 'val11_visual', candidateId, name);
      await fs.mkdir(directory, { recursive: true });
      for (const entry of sampleScores) {
        const record = { version: 'v10_mdpo_val11_sample_v1', split: 'val', candidate_id: candidateId,
          group: name, role: spec.model_role, sample: entry.sample,
          scorer: { model: groupRows[name].scorer_model, prompt_version: groupRows[name].prompt_version, response_id: entry.response_id },
          views, metric_protocol: groupRows[name].metric_protocol, scores: entry.scores, metrics: entry.metrics,
          view_sha256: entry.view_sha256, generation_strategy: entry.generation_strategy, evaluation: entry.evaluation,
          preference_model: { model_sha256: entry.preference_model_sha256 }, security: entry.security };
        await fs.writeFile(path.join(directory, `${encodeURIComponent(`C__${entry.sample.sample_id}`)}.json`), JSON.stringify(record));
      }
    }
    const protocol = validateMdpoFourGroupSummaries(groupRows), baseline = groupRows.v10_no_rerank,
      candidateSummary = groupRows.mdpo_no_rerank, gate = evaluateMdpoVal11Gate({ baseline, candidate: candidateSummary });
    const val11 = { version: 'v10_mdpo_val11_gate_report_v1', candidate_id: candidateId,
      candidate_model: { file: candidateRelative, sha256: candidateSha256 }, paired_provenance: { paired_samples: 11 },
      four_group_protocol: protocol, baseline, candidate: candidateSummary, gate, test_not_used: true };
    const four = { version: 'v10_mdpo_four_group_val11_v1', candidate_id: candidateId,
      candidate_model: val11.candidate_model, protocol, groups: Object.values(groupRows), core_gate: gate, test_not_used: true };
    const reportFile = path.join(mdpo, 'val11_reports', `${candidateId}.json`), fourFile = path.join(mdpo, 'val11_reports', `${candidateId}.four_group.json`);
    await fs.mkdir(path.dirname(reportFile), { recursive: true });
    const reportBytes = Buffer.from(JSON.stringify(val11)), fourBytes = Buffer.from(JSON.stringify(four));
    await fs.writeFile(reportFile, reportBytes); await fs.writeFile(fourFile, fourBytes);
    ablations.push({ id, candidate_id: candidateId, candidate_file: candidateRelative, candidate_sha256: candidateSha256,
      trainable_parameters: 7676, gate_accepted: gate.accepted, delta: gate.delta,
      report_file: path.relative(root, reportFile).split(path.sep).join('/'), report_sha256: digest(reportBytes),
      four_group_report_file: path.relative(root, fourFile).split(path.sep).join('/'), four_group_report_sha256: digest(fourBytes) });
  }
  const report = { version: 'v10_mdpo_ablation_report_v1', status: 'complete_authoritative_val11',
    base_selected_id: selection.selected.id, sample_count: 11, test_used: false, ablations,
    sweep_sensitivity: { rank: [{}, {}, {}], beta: [{}, {}, {}] } };
  const fields = ['id', 'candidate_sha256', 'trainable_parameters', 'gate_accepted', 'delta_aesthetic',
    'delta_composition_harmony', 'delta_PCK_005', 'delta_PCK_010', 'delta_text_clarity', 'delta_leader_line_clarity'];
  const csv = `${fields.join(',')}\n${ablations.map((row) => [row.id, row.candidate_sha256, row.trainable_parameters,
    row.gate_accepted, row.delta.aesthetic, row.delta.composition_harmony, row.delta.PCK_005, row.delta.PCK_010,
    row.delta.text_clarity, row.delta.leader_line_clarity].join(',')).join('\n')}\n`;
  const input = { root, mdpo, selection, ledger, report, csvBytes: Buffer.from(csv), referenceSha256, datasetSha256 };
  assert.equal((await validateMdpoAblationEvidence(input)).evaluated, 9);
  await assert.rejects(() => validateMdpoAblationEvidence({ ...input, csvBytes: Buffer.from(`${csv}tampered`) }), /CSV differs/);
  const bad = structuredClone(report); bad.ablations[0].delta.aesthetic += 1;
  await assert.rejects(() => validateMdpoAblationEvidence({ ...input, report: bad }), /changed or failed/);
  console.log('v10-MDPO nine-ablation immutable candidate, val11, raw Qwen and JSON/CSV evidence tests passed.');
} finally {
  const resolved = path.resolve(root), prefix = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('v10-mdpo-ablation-evidence-test-')) throw new Error('Unsafe ablation evidence test cleanup');
  await fs.rm(resolved, { recursive: true, force: true });
}
