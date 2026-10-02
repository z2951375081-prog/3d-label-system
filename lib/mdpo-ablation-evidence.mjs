import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { evaluateMdpoVal11Gate } from './mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from './mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from './mdpo-four-group-raw.mjs';

export const MDPO_ABLATION_IDS = Object.freeze(['full_fixed_variance', 'no_multi_only_overall_dpo', 'no_overall_only_multidimensional',
  'no_reference_kl', 'no_text_clarity', 'no_leader_line_clarity', 'equal_multidimensional_weights',
  'clarity_emphasis_weights', 'learnable_variance']);
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const cell = (value) => { const text = String(value ?? ''); return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; };
const inside = (root, mdpo, relative) => {
  const file = path.resolve(root, String(relative || ''));
  if (!file.startsWith(path.resolve(mdpo) + path.sep) || path.extname(file).toLowerCase() !== '.json')
    throw new Error(`Unsafe MDPO ablation evidence path: ${relative}`);
  return file;
};

export async function validateMdpoAblationEvidence({ root, mdpo, selection, ledger, report, csvBytes,
  referenceSha256, datasetSha256 } = {}) {
  const trained = new Map((ledger?.configurations || []).filter((row) => row.status === 'trained').map((row) => [row.id, row]));
  if (trained.size !== 9 || MDPO_ABLATION_IDS.some((id) => !trained.has(id)) || ledger?.test_used_for_selection !== false
      || report?.version !== 'v10_mdpo_ablation_report_v1' || report.status !== 'complete_authoritative_val11'
      || report.test_used !== false || report.base_selected_id !== selection?.selected?.id
      || report.sample_count !== 11 || report.ablations?.length !== 9
      || JSON.stringify(report.ablations.map((row) => row.id)) !== JSON.stringify(MDPO_ABLATION_IDS)
      || report.sweep_sensitivity?.rank?.length !== 3 || report.sweep_sensitivity?.beta?.length !== 3
      || !Buffer.isBuffer(csvBytes) || !/^[a-f0-9]{64}$/i.test(referenceSha256 || '')
      || !/^[a-f0-9]{64}$/i.test(datasetSha256 || '')) throw new Error('Complete authoritative nine-ablation evidence is required');
  const fields = ['id', 'candidate_sha256', 'trainable_parameters', 'gate_accepted', 'delta_aesthetic',
    'delta_composition_harmony', 'delta_PCK_005', 'delta_PCK_010', 'delta_text_clarity', 'delta_leader_line_clarity'];
  const csvRows = [];
  for (const row of report.ablations) {
    const trainedRow = trained.get(row.id);
    const legacySynthetic = report.version === 'v10_mdpo_ablation_report_v1' && row.report_file === 'experiments/mdpo/val11_reports/' + row.candidate_id + '.json';
    const expectedReportFile = legacySynthetic ? 'experiments/mdpo/val11_reports/' + row.candidate_id + '.json' : 'experiments/mdpo/val11_reports_safety_priority_v2/' + row.candidate_id + '.json';
    const expectedFourGroupFile = legacySynthetic ? 'experiments/mdpo/val11_reports/' + row.candidate_id + '.four_group.json' : 'experiments/mdpo/val11_reports_safety_priority_v2/' + row.candidate_id + '.four_group.json';
    if (row.candidate_id !== `ablation_${row.id}` || row.candidate_file !== trainedRow.candidate_file
        || row.report_file !== expectedReportFile
        || row.four_group_report_file !== expectedFourGroupFile)
      throw new Error(`Invalid authoritative ablation identity: ${row.id}`);
    const [candidateBytes, reportBytes, fourBytes] = await Promise.all([
      fs.readFile(inside(root, mdpo, row.candidate_file)), fs.readFile(inside(root, mdpo, row.report_file)),
      fs.readFile(inside(root, mdpo, row.four_group_report_file))
    ]);
    const candidate = JSON.parse(candidateBytes.toString('utf8')), val11 = JSON.parse(reportBytes.toString('utf8')),
      four = JSON.parse(fourBytes.toString('utf8'));
    if (digest(candidateBytes) !== row.candidate_sha256 || digest(reportBytes) !== row.report_sha256
        || digest(fourBytes) !== row.four_group_report_sha256 || candidate.reference?.sha256 !== referenceSha256
        || candidate.mdpo?.dataset_sha256 !== datasetSha256 || candidate.mdpo?.weight_update_evidence?.ablation?.effective !== true
        || val11.candidate_id !== row.candidate_id || val11.candidate_model?.sha256 !== row.candidate_sha256
        || val11.test_not_used !== true || val11.paired_provenance?.paired_samples !== 11
        || four.candidate_id !== row.candidate_id || four.test_not_used !== true || four.groups?.length !== 4
        || JSON.stringify(evaluateMdpoVal11Gate({ baseline: val11.baseline, candidate: val11.candidate })) !== JSON.stringify(val11.gate)
        || JSON.stringify(four.core_gate) !== JSON.stringify(val11.gate)
        || JSON.stringify(validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((group) => [group.group, group])))) !== JSON.stringify(val11.four_group_protocol)
        || row.gate_accepted !== val11.gate.accepted || JSON.stringify(row.delta) !== JSON.stringify(val11.gate.delta))
      throw new Error(`Authoritative ablation artifacts changed or failed validation: ${row.id}`);
    await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: row.candidate_id, fourGroup: four });
    const csvRow = { id: row.id, candidate_sha256: row.candidate_sha256, trainable_parameters: row.trainable_parameters,
      gate_accepted: row.gate_accepted, delta_aesthetic: row.delta.aesthetic,
      delta_composition_harmony: row.delta.composition_harmony, delta_PCK_005: row.delta.PCK_005,
      delta_PCK_010: row.delta.PCK_010, delta_text_clarity: row.delta.text_clarity,
      delta_leader_line_clarity: row.delta.leader_line_clarity };
    csvRows.push(fields.map((field) => cell(csvRow[field])).join(','));
  }
  const expectedCsv = `${fields.join(',')}\n${csvRows.join('\n')}\n`;
  if (csvBytes.toString('utf8') !== expectedCsv) throw new Error('Authoritative nine-ablation CSV differs from JSON evidence');
  return { complete: true, trained: 9, evaluated: 9, ids: MDPO_ABLATION_IDS, csvSha256: digest(csvBytes) };
}
