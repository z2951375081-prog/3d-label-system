import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { activatePreservingPrevious } from '../lib/model-preservation.mjs';
import { materializeActiveMdpoModel, validateMdpoActivationBundle } from '../lib/mdpo-model-activation.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';
import { validateMdpoAblationEvidence } from '../lib/mdpo-ablation-evidence.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments'), mdpo = path.join(experiments, 'mdpo');
const selectionFile = path.join(mdpo, 'hyperparameter_selection.json'), activeFile = path.join(experiments, 'layout_model.json');
const preferenceFile = path.join(experiments, 'preference_model.json'), historyFile = path.join(experiments, 'layout_model_activation_history.jsonl');
const selection = JSON.parse(await fs.readFile(selectionFile, 'utf8'));
const resolveInside = (relative) => {
  const target = path.resolve(root, String(relative || '')), prefix = path.resolve(mdpo) + path.sep;
  if (!target.startsWith(prefix) || path.extname(target).toLowerCase() !== '.json') throw new Error(`MDPO activation artifact outside experiments/mdpo: ${relative}`);
  return target;
};
const candidateFile = resolveInside(selection.selected?.candidate_file), reportFile = resolveInside(selection.selected?.report_file), fourGroupReportFile = resolveInside(selection.selected?.four_group_report_file);
const [candidateBytes, activeBytes, reportBytes, fourGroupReportBytes, preferenceBytes] = await Promise.all([fs.readFile(candidateFile), fs.readFile(activeFile), fs.readFile(reportFile), fs.readFile(fourGroupReportFile), fs.readFile(preferenceFile)]);
const validated = validateMdpoActivationBundle({ selection, candidateBytes, activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes });
const datasetBytes = await fs.readFile(path.join(mdpo, 'train_pairs.json'));
const finalDataset = validateMdpoFinalDatasetAudit({ datasetBytes,
  audit: await fs.readFile(path.join(mdpo, 'dataset_audit_final.json'), 'utf8').then(JSON.parse), referenceSha256: validated.activeHash });
if (validated.candidate.mdpo?.dataset_sha256 !== finalDataset.datasetSha256) throw new Error('MDPO activation candidate train33 hash differs from immutable final dataset audit');
const [ablationLedger, ablationReport, ablationCsv] = await Promise.all([
  fs.readFile(path.join(mdpo, 'ablations', 'ablation_ledger.json'), 'utf8').then(JSON.parse),
  fs.readFile(path.join(mdpo, 'ablations', 'ablation_report.json'), 'utf8').then(JSON.parse),
  fs.readFile(path.join(mdpo, 'ablations', 'ablation_report.csv'))
]);
await validateMdpoAblationEvidence({ root, mdpo, selection, ledger: ablationLedger, report: ablationReport,
  csvBytes: ablationCsv, referenceSha256: validated.activeHash, datasetSha256: finalDataset.datasetSha256 });
await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: selection.selected.candidate_id,
  fourGroup: validated.fourGroup });
const preferenceHash = createHash('sha256').update(preferenceBytes).digest('hex');
const active = materializeActiveMdpoModel(validated, { selectionFile: path.relative(root, selectionFile).split(path.sep).join('/'), reportFile: path.relative(root, reportFile).split(path.sep).join('/'), preferenceModelSha256: preferenceHash });
const activationDirectory = path.join(mdpo, 'activation');
await fs.mkdir(activationDirectory, { recursive: true });
const activationCandidateFile = path.join(activationDirectory, `layout_model_v10_mdpo_${validated.candidateHash.slice(0, 20)}.json`);
await fs.writeFile(activationCandidateFile, JSON.stringify(active, null, 2) + '\n', { flag: 'wx' });
const activationReportFile = path.join(mdpo, 'activation_report.json');
let report = null, activationReportBytes = null;
const result = await activatePreservingPrevious({
  candidateFile: activationCandidateFile, activeFile, historyFile,
  validateCandidate(model) {
    if (model.version !== 'layout_model_v10_mdpo' || model.status !== 'active_val11_selected_v10_mdpo'
        || model.validation_gate?.status !== 'accepted' || model.validation_gate?.active !== true
        || model.activation_provenance?.original_candidate_sha256 !== validated.candidateHash
        || model.activation_provenance?.frozen_reference_sha256 !== validated.activeHash
        || model.activation_provenance?.test11_used_for_activation !== false
        || model.architecture?.qwen_inference_input !== false) throw new Error('Materialized MDPO activation model failed validation');
  },
  metadata: { policy: 'v10_mdpo_complete_81_grid_authoritative_val11_no_force_override', selection_file: path.relative(root, selectionFile).split(path.sep).join('/'),
    val11_report_file: path.relative(root, reportFile).split(path.sep).join('/'), four_group_report_file: path.relative(root, fourGroupReportFile).split(path.sep).join('/'), candidate_sha256: validated.candidateHash,
    reference_sha256: validated.activeHash, historical_reward_model_sha256: preferenceHash, test_used: false },
  async afterActivate({ record }) {
    const preferenceAfter = createHash('sha256').update(await fs.readFile(preferenceFile)).digest('hex');
    if (preferenceAfter !== preferenceHash) throw new Error('Historical reward model changed during MDPO activation');
    report = { version: 'v10_mdpo_activation_report_v1', activated_at: record.activated_at, status: 'activated',
      candidate_sha256: validated.candidateHash, previous_active_sha256: validated.activeHash, active_sha256: record.activated_sha256.toLowerCase(),
      backup_file: record.backup_file ? path.relative(root, record.backup_file).split(path.sep).join('/') : null,
      historical_reward_model_sha256: preferenceHash, historical_reward_model_unchanged: true,
      selection_file: path.relative(root, selectionFile).split(path.sep).join('/'), val11_report_file: path.relative(root, reportFile).split(path.sep).join('/'),
      four_group_report_file: path.relative(root, fourGroupReportFile).split(path.sep).join('/'), four_group_report_sha256: validated.fourGroupHash, test11_pending: true };
    activationReportBytes = Buffer.from(JSON.stringify(report, null, 2) + '\n');
    await fs.writeFile(activationReportFile, activationReportBytes, { flag: 'wx' });
  },
  async rollbackAfterFailure() {
    if (!activationReportBytes) return;
    const existing = await fs.readFile(activationReportFile).catch(() => null);
    if (existing && createHash('sha256').update(existing).digest('hex') === createHash('sha256').update(activationReportBytes).digest('hex')) await fs.rm(activationReportFile, { force: true });
  }
});
const preferenceAfter = createHash('sha256').update(await fs.readFile(preferenceFile)).digest('hex');
if (preferenceAfter !== preferenceHash) throw new Error('Historical reward model changed during MDPO activation');
console.log(JSON.stringify(report, null, 2));
