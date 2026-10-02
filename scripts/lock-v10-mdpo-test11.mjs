import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validateExistingMdpoTest11Lock, validateMdpoTest11Prerequisites } from '../lib/mdpo-test11-lock.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), experiments = path.join(root, 'experiments'), mdpo = path.join(experiments, 'mdpo');
const activeFile = path.join(experiments, 'layout_model.json'), preferenceFile = path.join(experiments, 'preference_model.json');
const activationFile = path.join(mdpo, 'activation_report.json'), selectionFile = path.join(mdpo, 'hyperparameter_selection.json');
const lockFile = path.join(mdpo, 'test11_lock.json'), finalFile = path.join(mdpo, 'test11_final_report.json');
const [activeBytes, preferenceBytes, activationReport, selection, manifest] = await Promise.all([
  fs.readFile(activeFile), fs.readFile(preferenceFile), fs.readFile(activationFile, 'utf8').then(JSON.parse),
  fs.readFile(selectionFile, 'utf8').then(JSON.parse), fs.readFile(path.join(experiments, 'dataset_manifest.json'), 'utf8').then(JSON.parse)
]);
await fs.access(finalFile).then(() => { throw new Error('test11 final report already exists; the locked test may not be rerun'); }, (error) => { if (error?.code !== 'ENOENT') throw error; });
const baselineFile = path.resolve(root, String(activationReport.backup_file || '')), experimentsPrefix = path.resolve(experiments) + path.sep;
if (!baselineFile.startsWith(experimentsPrefix) || path.extname(baselineFile).toLowerCase() !== '.json') throw new Error('test11 frozen original-v10 backup path is invalid');
const baselineBytes = await fs.readFile(baselineFile);
const cohort = manifest.samples.filter((sample) => sample.split === 'test').map((sample) => `${sample.category}/${sample.sample_id}`);
const validated = validateMdpoTest11Prerequisites({ activeBytes, baselineBytes, preferenceBytes, activationReport, selection, cohort });
const reportFile = path.resolve(root, String(selection.selected.report_file || ''));
const fourGroupReportFile = path.resolve(root, String(selection.selected.four_group_report_file || ''));
const mdpoPrefix = path.resolve(mdpo) + path.sep;
if (![reportFile, fourGroupReportFile].every((file) => file.startsWith(mdpoPrefix) && path.extname(file).toLowerCase() === '.json')) throw new Error('test11 selected val11 evidence path invalid');
const [val11Bytes, fourGroupBytes] = await Promise.all([fs.readFile(reportFile), fs.readFile(fourGroupReportFile)]);
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
if (digest(val11Bytes) !== selection.selected.report_sha256 || digest(fourGroupBytes) !== selection.selected.four_group_report_sha256) throw new Error('test11 selected val11/four-group evidence changed since activation');
const val11 = JSON.parse(val11Bytes.toString('utf8'));
await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: selection.selected.candidate_id,
  fourGroup: JSON.parse(fourGroupBytes.toString('utf8')) });
if (val11.version !== 'v10_mdpo_val11_gate_report_v2' || val11.evaluation_policy !== 'safety_priority_v2' || val11.gate?.accepted !== true || val11.paired_provenance?.paired_samples !== 11
    || val11.baseline?.scorer_model !== val11.candidate?.scorer_model || val11.baseline?.prompt_version !== val11.candidate?.prompt_version
    || val11.baseline?.prompt_version !== 'aesthetic_safety_v3_fourteen_dimension_mdpo_teacher_v1'
    || JSON.stringify(val11.baseline?.views) !== JSON.stringify(['before', 'main', 'right', 'left', 'up', 'down'])) throw new Error('test11 selected val11 scorer/prompt/view protocol incomplete');
const codeFiles = ['server.mjs', 'public/app.js', 'lib/layout-model.mjs', 'lib/layout-optimizer.mjs', 'lib/heterogeneous-layout-graph.mjs',
  'lib/anchor-frame-features.mjs', 'lib/candidate-generator.mjs', 'lib/reproduction-metrics.mjs', 'lib/mdpo-continuous-policy.mjs',
  'lib/mdpo-test11-lock.mjs', 'lib/mdpo-test11-report.mjs', 'lib/mdpo-four-group.mjs', 'scripts/run-llm-preference-headless.mjs'];
const codeSha256 = Object.fromEntries(await Promise.all(codeFiles.map(async (file) => [file, createHash('sha256').update(await fs.readFile(path.join(root, file))).digest('hex')])));
const lock = { version: 'v10_mdpo_test11_lock_v1', locked_at: new Date().toISOString(), status: 'locked_pending_single_test11_evaluation',
  active_model: { file: 'experiments/layout_model.json', sha256: validated.activeSha256, version: validated.active.version },
  baseline_model: { file: path.relative(root, baselineFile).split(path.sep).join('/'), sha256: validated.baselineSha256 },
  historical_reward_model: { file: 'experiments/preference_model.json', sha256: validated.preferenceSha256, frozen: true },
  selection: { file: 'experiments/mdpo/hyperparameter_selection.json', candidate_sha256: selection.selected.candidate_sha256, val11_report_sha256: selection.selected.report_sha256,
    four_group_report_sha256: selection.selected.four_group_report_sha256 },
  scorer: { model: val11.baseline.scorer_model, prompt_version: val11.baseline.prompt_version, views: val11.baseline.views },
  cohort: validated.cohort, cohort_split: 'test', sample_count: 11, seed: 17017, code_sha256: codeSha256,
  policy: { one_final_test_only: true, test_not_used_for_training: true, test_not_used_for_selection: true, qwen_offline_evaluator_only: true, qwen_inference_input: false } };
const existingLock = await fs.readFile(lockFile, 'utf8').then(JSON.parse).catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
const effectiveLock = existingLock ? validateExistingMdpoTest11Lock(existingLock, { activeSha256: validated.activeSha256, baselineSha256: validated.baselineSha256,
  preferenceSha256: validated.preferenceSha256, candidateSha256: selection.selected.candidate_sha256, val11ReportSha256: selection.selected.report_sha256,
  fourGroupReportSha256: selection.selected.four_group_report_sha256, cohort: validated.cohort, codeSha256,
  scorerModel: val11.baseline.scorer_model, promptVersion: val11.baseline.prompt_version }) : lock;
if (!existingLock) await fs.writeFile(lockFile, JSON.stringify(lock, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ status: effectiveLock.status, reused_existing_lock: Boolean(existingLock), lock_file: path.relative(root, lockFile).split(path.sep).join('/'), sample_count: effectiveLock.sample_count,
  active_sha256: validated.activeSha256, baseline_sha256: validated.baselineSha256 }, null, 2));
