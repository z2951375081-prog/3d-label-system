import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { annotationsToLabels, boundsFromObj, cleanObj } from './scripts/generate-artifacts.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from './lib/layout-optimizer.mjs';
import { predictPreference } from './lib/preference-model.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from './lib/candidate-generator.mjs';
import { compareToManual } from './lib/layout-comparison.mjs';
import { applyLayoutModel, selectLayoutCandidates } from './lib/layout-model.mjs';
import { DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, VISION_KEYS, normalizeOllamaEndpoint, ollamaScoringRequest, parseOllamaScoringResponse } from './lib/ollama-vision-adapter.mjs';
import { activatePreservingPrevious } from './lib/model-preservation.mjs';
import { assessSafetyEligibility, compareSafetyReference, safetySnapshot, selectSafetyConstrainedTrial } from './lib/preference-policy.mjs';
import { optimizeWithAdaptiveDirectionalGate } from './lib/adaptive-directional-rerank.mjs';
import { LLM_AESTHETIC_DIMENSIONS, LLM_SAFETY_DIMENSIONS, LLM_SCORE_NAMES, preferenceComposite } from './public/preference-scoring.js';
import { buildCvFeatures } from './lib/cv-feature-encoder.mjs';
import { buildSpatialContext, computeSpatialStyleMetrics } from './lib/spatial-style-features.mjs';
import { evaluateViewConditionedLayout } from './lib/view-conditioned-evaluator.mjs';
import { evaluateVisualActivationGate, inferGeometrySafetyEligibility, LLM_VISUAL_ACTIVATION_THRESHOLDS } from './lib/llm-visual-activation-gate.mjs';
import { buildAnchorFrames } from './lib/anchor-frame-features.mjs';
import { localTargetVector } from './lib/heterogeneous-layout-graph.mjs';
import { MDPO_DIMENSIONS } from './lib/mdpo-continuous-policy.mjs';
import { hashMdpoViews, selectMdpoPairs, strictMdpoSafety } from './lib/mdpo-collection.mjs';
import { validateMdpoDataset } from './lib/mdpo-dataset.mjs';
import { validateMdpoFinalDatasetAudit } from './lib/mdpo-final-dataset-audit.mjs';
import { evaluateMdpoVal11AlignedGate, evaluateMdpoVal11Gate } from './lib/mdpo-activation-gate.mjs';
import { validateMdpoVal11Records } from './lib/mdpo-val11-provenance.mjs';
import { MDPO_VAL11_GROUPS, mdpoVal11Group, validateMdpoFourGroupSummaries } from './lib/mdpo-four-group.mjs';
import { assessMdpoDeployment } from './lib/mdpo-experiment-status.mjs';
import { buildMdpoRejectionReport } from './lib/mdpo-rejection-report.mjs';
import { writeImmutableFileBundle, writeImmutableJsonBundle } from './lib/immutable-json-bundle.mjs';
import { validateExistingMdpoTest11Lock, validateMdpoTest11Prerequisites } from './lib/mdpo-test11-lock.mjs';
import { validateMdpoTest11Records } from './lib/mdpo-test11-report.mjs';
import { evaluateReproductionMetrics, reproductionQualityScore, REPRODUCTION_METRIC_PROTOCOL, REPRODUCTION_METRIC_PROTOCOL_V4 } from './lib/reproduction-metrics.mjs';
import { validateAndSummarizeFinalEightTest11 } from './lib/final-eight-test11-results.mjs';
import { MDPO_PERTURBATION_MODES, perturbMdpoCandidate } from './lib/mdpo-candidate-perturbation.mjs';
import { replaceFileWithRetry } from './lib/atomic-file-replace.mjs';
import { validateCompletedMdpoTrainingRun } from './lib/mdpo-training-resume.mjs';
import { MDPO_UI_TRAINING_DEFAULTS, mdpoUiTrainingArgs, normalizeMdpoUiTrainingRequest } from './lib/mdpo-training-control.mjs';
import { buildMdpoLiveResults } from './lib/mdpo-live-results.mjs';
import { loadLatestCompletedAlignedModel, loadSelectedAlignedModel } from './lib/mdpo-aligned-display.mjs';

const root = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(root, 'public');
const dataDir = path.join(root, 'data');
const experimentsDir = process.env.EXPERIMENTS_DIR ? path.resolve(process.env.EXPERIMENTS_DIR) : path.join(root, 'experiments');
const evaluationLog = path.join(experimentsDir, 'evaluations.jsonl');
const preferenceLog = path.join(experimentsDir, 'preferences.jsonl');
const preferenceModelFile = path.join(experimentsDir, 'preference_model.json');
const preferenceCandidateFile = path.join(experimentsDir, 'preference_model_candidate.json');
const preferenceSelectionFile = path.join(experimentsDir, 'preference_validation_selection.json');
const preferenceTestReportFile = path.join(experimentsDir, 'preference_test11_report.json');
const preferenceCheckpointLatestFile = path.join(experimentsDir, 'llm_preference_checkpoints_latest.json');
const visualValidationLatestFile = path.join(experimentsDir, 'llm_visual_validation_latest.json');
const llmValGateComparisonFile = path.join(experimentsDir, 'llm_val11_gate_comparison.json');
const layoutModelFile = path.join(experimentsDir, 'layout_model.json');
const mdpoDir = path.join(experimentsDir, 'mdpo');
const mdpoPairFile = path.join(mdpoDir, 'train_pairs.json');
const mdpoCandidateFile = path.join(mdpoDir, 'train_candidates.jsonl');
const mdpoUiTrainingStateFile = path.join(mdpoDir, 'manual_training_job.json');
const mdpoUiTrainingRunsDir = path.join(mdpoDir, 'manual_runs');
const leaderLengthPriorFile = path.join(experimentsDir, 'manual_leader_length_prior.json');
const qualityComparisonFile = path.join(experimentsDir, 'comparisons', 'round1', 'comparison.json');
const reproductionComparisonFile = path.join(experimentsDir, 'comparisons', 'latest_reproduction_metrics', 'comparison.json');
const finalEightTest11File = path.join(experimentsDir, 'comparisons', 'final_eight_test11', 'comparison.json');
const preferenceConvergenceFile = path.join(experimentsDir, 'llm_preference_convergence.json');
const preferenceRunLatestFile = path.join(experimentsDir, 'llm_preference_run_latest.json');
const v9InferenceSelectionFile = path.join(experimentsDir, 'v9_3d_inference_selection.json');
const v10TrainingReportFile = path.join(experimentsDir, 'layout_training_v10_anchor_frame_report.json');
const scoringConnectionAuditLog = path.join(experimentsDir, 'scoring_connection_attempts.jsonl');
const port = Number(process.env.PORT || 5173);
const host = process.env.HOST || '127.0.0.1';
let preferenceModelCache = { mtimeMs: -1, model: null };
let layoutModelCache = { mtimeMs: -1, model: null };
let leaderLengthPriorCache = { mtimeMs: -1, prior: null };
let mdpoUiTrainingChild = null;
let mdpoUiTrainingStarting = false;
let initialScoringEndpoint;
try { initialScoringEndpoint = normalizeOllamaEndpoint(process.env.OLLAMA_HOST || DEFAULT_OLLAMA_URL); }
catch { initialScoringEndpoint = normalizeOllamaEndpoint(DEFAULT_OLLAMA_URL); }
let scoringConfig = { ...initialScoringEndpoint, model: process.env.OLLAMA_VISION_MODEL || DEFAULT_OLLAMA_MODEL };
let scoringConnectionTest = null;
const SCORING_PROTOCOL_ID = 'aesthetic_safety_v3_fourteen_dimension';

async function appendScoringConnectionAudit({ result, code = null, status = null, responseId = null }) {
  const record = {
    version: 'scoring_connection_attempt_v1',
    attempted_at: new Date().toISOString(),
    endpoint: scoringConfig.apiUrl,
    format: scoringConfig.format,
    model: scoringConfig.model,
    request: { kind: 'six_image_structured_scoring_probe', image_count: 6, persisted_to_training_data: false },
    result,
    code,
    http_status: status,
    response_id: responseId,
    security: { local_only: true, credentials_required: false, images_persisted: false }
  };
  try {
    await fs.mkdir(experimentsDir, { recursive: true });
    await fs.appendFile(scoringConnectionAuditLog, `${JSON.stringify(record)}\n`, 'utf8');
  } catch (error) {
    console.error(`连接审计写入失败：${error.message}`);
  }
}
const scoringReceipts = new Map();
const validationGenerationReceipts = new Map();
function scoringFingerprint(candidate) {
  return createHash('sha256').update(JSON.stringify({ run_id: candidate.run_id || null, sample: candidate.sample, split: candidate.split, metrics: candidate.metrics, labels: candidate.labels, strategy: candidate.strategy })).digest('hex');
}
function issueScoringReceipt(payload, response) {
  if (!safeScoringConfig().tested || !Array.isArray(payload?.labels) || !payload?.metrics || !payload?.sample || !payload?.strategy || payload?.split !== 'train') return null;
  const id = randomUUID();
  const images = payload.type === 'mdpo_train_candidate' ? hashMdpoViews(payload.visuals) : null;
  scoringReceipts.set(id, { id, run_id: payload.run_id || null, fingerprint: scoringFingerprint(payload), scores: response.result.scores, model: response.model, response_id: response.response_id, view_sha256: images, issued_at: Date.now() });
  while (scoringReceipts.size > 512) scoringReceipts.delete(scoringReceipts.keys().next().value);
  return id;
}
function verifiedLLMScore(candidate) {
  const receipt = scoringReceipts.get(candidate?.llm_score_receipt);
  if (!receipt || !candidate?.metrics || !Array.isArray(candidate?.labels) || !candidate?.sample || !candidate?.strategy || Date.now() - receipt.issued_at > 4 * 60 * 60 * 1000 || receipt.fingerprint !== scoringFingerprint(candidate)) return null;
  if (JSON.stringify(receipt.scores) !== JSON.stringify(candidate.llm_scores)) return null;
  return receipt;
}

let mdpoCollectionQueue = Promise.resolve();
async function collectMdpoTrainSample(payload) {
  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  const sample = manifest.samples.find((item) => item.category === payload.sample?.category && item.sample_id === String(payload.sample?.sample_id));
  if (sample?.split !== 'train' || payload.split !== 'train') throw new Error('MDPO candidate collection is train33 only');
  const minimumNewCandidates = payload.finalize_only === true ? 0 : 1;
  if (!/^[a-zA-Z0-9_-]{8,80}$/.test(String(payload.run_id || '')) || !Array.isArray(payload.candidates) || payload.candidates.length < minimumNewCandidates || payload.candidates.length > 12) throw new Error(`MDPO requires a run ID and ${minimumNewCandidates}-12 newly Qwen-scored candidates`);
  const modelBytes = await fs.readFile(layoutModelFile);
  const referenceHash = createHash('sha256').update(modelBytes).digest('hex');
  const referenceModel = JSON.parse(modelBytes.toString('utf8'));
  if (!String(referenceModel.version).startsWith('layout_model_v10_') || payload.reference_model_sha256 !== referenceHash) throw new Error('MDPO reference v10 SHA-256 mismatch');
  const fixed = annotationsToLabels(JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8')));
  const clean = cleanObj(await fs.readFile(path.join(root, sample.input.source_obj), 'utf8'));
  const bounds = boundsFromObj(clean.text), geometry = parseObjTriangles(clean.text);
  const initial = fixedCandidatesForLayoutModel(fixed, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, referenceModel);
  validateFixedLabelContract(fixed, initial, 'mdpo/reference-input');
  const frames = buildAnchorFrames(initial, geometry, bounds);
  const prepared = payload.candidates.map((item, index) => {
    const receipt = verifiedLLMScore(item);
    if (!receipt || receipt.run_id !== payload.run_id || !receipt.view_sha256 || !receipt.response_id || receipt.model !== scoringConfig.model) throw new Error(`MDPO candidate ${index} has no valid live Qwen image/score receipt`);
    if (item.split !== 'train' || item.sample?.category !== sample.category || String(item.sample?.sample_id) !== String(sample.sample_id) || item.type !== 'mdpo_train_candidate') throw new Error(`MDPO candidate ${index} belongs to a different split/sample/protocol`);
    const currentHashes = hashMdpoViews(item.visuals);
    if (JSON.stringify(currentHashes) !== JSON.stringify(receipt.view_sha256)) throw new Error(`MDPO candidate ${index} six images differ from images submitted to Qwen`);
    validateFixedLabelContract(fixed, item.labels, `mdpo/candidate-${index}`);
    const safety = assessSafetyEligibility(item.metrics, item.metrics);
    if (!safety.eligible) throw new Error(`MDPO candidate ${index} is unsafe: ${safety.violations.join(',')}`);
    const strict = strictMdpoSafety(item.metrics);
    if (!strict.eligible) throw new Error(`MDPO candidate ${index} violates strict geometry gate: ${strict.violations.join(',')}`);
    const perturbationIndex = Number(item.strategy?.mdpo_perturbation ?? item.strategy?.mdpoPerturbation);
    const isMdpoCandidate = item.strategy?.mdpo_candidate === true || item.strategy?.mdpoCandidate === true;
    if (!Number.isInteger(item.strategy?.seed) || !isMdpoCandidate
        || !Number.isInteger(perturbationIndex) || perturbationIndex < 0 || perturbationIndex >= MDPO_PERTURBATION_MODES.length
        || item.labels.length !== initial.length || MDPO_DIMENSIONS.some((name) => !Number.isFinite(receipt.scores[name]))) throw new Error(`MDPO candidate ${index} invalid seed, perturbation provenance or seven scores`);
    const localLayout = item.labels.map((label, position) => localTargetVector(label, initial[position], frames[position], bounds));
    if (localLayout.some((row) => row.some((value) => !Number.isFinite(value)))) throw new Error(`MDPO candidate ${index} non-finite local coordinates`);
    return {
      candidate_id: receipt.id, seed: item.strategy.seed, label_ids: item.labels.map((label) => String(label.id)), local_layout: localLayout,
      scores: Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, receipt.scores[name]])), scorer_model: receipt.model,
      prompt_version: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`, response_id: receipt.response_id,
      score_receipt_id: receipt.id, view_sha256: currentHashes,
      safety: { eligible: true, metrics: item.metrics, reference_metrics: item.metrics }, label_contract_valid: true,
      input_provenance: 'fixed_contract_clean_obj_only', reference_model_sha256: referenceHash,
      generation: {
        policy: 'frozen_active_v10_plus_bounded_anchor_local_perturbation',
        qwen_generated_coordinates: false,
        seed: item.strategy.seed,
        perturbation: { mode_index: perturbationIndex, mode: MDPO_PERTURBATION_MODES[perturbationIndex], bounded_anchor_local: true }
      }
    };
  });
  await fs.mkdir(mdpoDir, { recursive: true });
  const previous = await readOptionalJson(mdpoPairFile) || { version: 'v10_mdpo_train_pairs_v1', reference_model_sha256: referenceHash, pairs: [] };
  if (previous.reference_model_sha256 !== referenceHash) throw new Error('MDPO reference changed; start an explicit new dataset instead of mixing reference policies');
  const priorCandidateRows = await fs.readFile(mdpoCandidateFile, 'utf8').then((text) => text.split(/\r?\n/).filter(Boolean).map(JSON.parse), () => []);
  const newlyPersisted = prepared.filter((candidate) => !priorCandidateRows.some((row) => row.candidate_id === candidate.candidate_id));
  if (payload.stage_only === true) {
    if (newlyPersisted.length) await fs.appendFile(mdpoCandidateFile, newlyPersisted.map((candidate) => JSON.stringify({ ...candidate, category: sample.category, sample_id: sample.sample_id, split: 'train', clean_obj_source: sample.input.source_obj, run_id: payload.run_id })).join('\n') + '\n');
    return { sample: `${sample.category}/${sample.sample_id}`, staged: true, new_candidate_count: newlyPersisted.length, existing_pair_count: previous.pairs.filter((pair) => pair.category === sample.category && String(pair.sample_id) === String(sample.sample_id)).length };
  }
  const priorCandidates = priorCandidateRows.filter((candidate) => candidate.category === sample.category && String(candidate.sample_id) === String(sample.sample_id) && strictMdpoSafety(candidate.safety?.metrics).eligible);
  const mergedCandidates = [...priorCandidates, ...prepared].filter((candidate, index, rows) => rows.findIndex((row) => row.candidate_id === candidate.candidate_id) === index);
  if (new Set(mergedCandidates.map((candidate) => candidate.seed)).size < 8) throw new Error('MDPO requires at least eight distinct generation seeds across persisted and new candidates');
  const requestedPairCount = Math.min(12, Math.max(6, Number(payload.pair_count) || 8));
  const selected = selectMdpoPairs(mergedCandidates, { count: requestedPairCount });
  const pairs = selected.map((entry) => ({
    category: sample.category, sample_id: sample.sample_id, split: 'train', clean_obj_source: sample.input.source_obj,
    scorer_model: scoringConfig.model, prompt_version: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`,
    reference_model_sha256: referenceHash, run_id: payload.run_id, ...entry
  }));
  const retainedPairs = previous.pairs.filter((pair) => pair.category !== sample.category || String(pair.sample_id) !== String(sample.sample_id));
  const dataset = { ...previous, pairs: [...retainedPairs, ...pairs] };
  const audit = validateMdpoDataset(dataset, manifest, { requireComplete: false });
  const tempFile = path.join(mdpoDir, `train_pairs.${randomUUID()}.tmp`);
  await fs.writeFile(tempFile, JSON.stringify(dataset, null, 2) + '\n', { flag: 'wx' });
  await replaceFileWithRetry(tempFile, mdpoPairFile);
  if (newlyPersisted.length) await fs.appendFile(mdpoCandidateFile, newlyPersisted.map((candidate) => JSON.stringify({ ...candidate, category: sample.category, sample_id: sample.sample_id, split: 'train', clean_obj_source: sample.input.source_obj, run_id: payload.run_id })).join('\n') + '\n');
  await fs.writeFile(path.join(mdpoDir, 'dataset_audit.json'), JSON.stringify({ ...audit, reference_model_sha256: referenceHash, complete: audit.sample_count === 33 && audit.pair_count >= 198, updated_at: new Date().toISOString() }, null, 2) + '\n');
  return { sample: `${sample.category}/${sample.sample_id}`, new_candidate_count: prepared.length, total_candidate_count: mergedCandidates.length, candidate_count: prepared.length, pair_count: pairs.length, replaced_previous_pair_count: previous.pairs.length - retainedPairs.length, audit };
}
function safeScoringConfig() {
  const tested = Boolean(scoringConnectionTest && scoringConnectionTest.apiUrl === scoringConfig.apiUrl && scoringConnectionTest.model === scoringConfig.model && scoringConnectionTest.protocolId === SCORING_PROTOCOL_ID);
  return { baseUrl: scoringConfig.baseUrl, apiUrl: scoringConfig.apiUrl, format: scoringConfig.format, model: scoringConfig.model, configured: Boolean(scoringConfig.model), localOnly: true, credentialsRequired: false, tested, testedAt: tested ? scoringConnectionTest.testedAt : null, scoringProtocol: SCORING_PROTOCOL_ID };
}

function validateScoringConfig(input) {
  const candidateUrl = String(input?.baseUrl || input?.apiUrl || scoringConfig.baseUrl || DEFAULT_OLLAMA_URL).trim();
  const model = String(input?.model || DEFAULT_OLLAMA_MODEL).trim();
  let endpoint;
  try { endpoint = normalizeOllamaEndpoint(candidateUrl); }
  catch (error) { throw Object.assign(error, { statusCode: 400 }); }
  if (!model || model.length > 120 || !/^[a-zA-Z0-9._:/-]+$/.test(model)) throw Object.assign(new Error('请输入有效模型 ID'), { statusCode: 400 });
  return { ...endpoint, model };
}

async function getPreferenceModel() {
  try {
    const stat = await fs.stat(preferenceModelFile);
    if (stat.mtimeMs === preferenceModelCache.mtimeMs) return preferenceModelCache.model;
    const model = JSON.parse(await fs.readFile(preferenceModelFile, 'utf8'));
    preferenceModelCache = { mtimeMs: stat.mtimeMs, model };
    return model;
  } catch {
    preferenceModelCache = { mtimeMs: 0, model: null };
    return null;
  }
}

function validationGenerationFingerprint(candidate) {
  const { rendered_text_measurements, ...stableMetrics } = candidate.metrics || {};
  return createHash('sha256').update(JSON.stringify({
    sample: { category: candidate.sample?.category, sample_id: String(candidate.sample?.sample_id) },
    split: candidate.sample?.split || candidate.split,
    labels: candidate.labels,
    metrics: stableMetrics,
    preference_model: candidate.preference_model,
    mdpo_evaluation: candidate.layout_model?.evaluation || null
  })).digest('hex');
}

function issueValidationGenerationReceipt(candidate) {
  const legacy = candidate?.preference_model?.evaluation;
  const mdpo = candidate?.layout_model?.evaluation;
  const mdpoTest11 = mdpo?.phase === 'test11';
  const mdpoValidation = ['active_v10_baseline', 'frozen_v10_mdpo_diagnostic_checkpoint'].includes(mdpo?.kind) || mdpoTest11;
  const evaluation = legacy || (mdpoValidation ? mdpo : null);
  if (!evaluation || candidate?.sample?.split !== (mdpoTest11 ? 'test' : 'val')) return null;
  if (mdpoValidation) {
    const group = mdpoVal11Group(mdpo.group);
    if (group.model_role !== mdpo.role || group.preference_rerank !== candidate.strategy?.preferenceRerank) throw new Error('MDPO four-group generation strategy does not match requested group');
  }
  const id = randomUUID();
  validationGenerationReceipts.set(id, {
    id,
    kind: legacy ? 'reward_checkpoint' : mdpoTest11 ? 'mdpo_test11' : 'mdpo_checkpoint',
    run_id: legacy?.run_id || mdpo?.test_run_id || mdpo?.candidate_id,
    round: legacy?.round ?? null,
    role: mdpo?.role || null,
    group: mdpo?.group || null,
    candidate_sha256: mdpo?.candidate_sha256 || null,
    lock_sha256: mdpo?.lock_sha256 || null,
    phase: mdpo?.phase || null,
    generation_strategy: Object.fromEntries(['generator', 'viewPolicy', 'groupPolicy', 'sizePolicy', 'optimizer', 'seed', 'iterations', 'preferenceRerank'].map((name) => [name, candidate.strategy?.[name]])),
    fingerprint: validationGenerationFingerprint(candidate),
    issued_at: Date.now()
  });
  while (validationGenerationReceipts.size > 512) validationGenerationReceipts.delete(validationGenerationReceipts.keys().next().value);
  return id;
}

function verifiedValidationGeneration(payload) {
  const receipt = validationGenerationReceipts.get(payload?.validation_generation_receipt);
  if (!receipt || payload?.type !== 'llm_validation_checkpoint' || payload?.split !== 'val' || Date.now() - receipt.issued_at > 4 * 60 * 60 * 1000) return null;
  if (receipt.run_id !== payload.run_id || receipt.round !== Number(payload.round) || receipt.fingerprint !== validationGenerationFingerprint(payload)) return null;
  return receipt;
}

function verifyVisualValidationImages(payload) {
  const visuals = payload?.visuals;
  if (!visuals || VISION_KEYS.some((name) => typeof visuals[name] !== 'string' || !visuals[name].startsWith('data:image/'))) return false;
  const images = Object.fromEntries(VISION_KEYS.map((name) => [name, createHash('sha256').update(visuals[name]).digest('hex')]));
  const receipt = validationGenerationReceipts.get(payload.validation_generation_receipt);
  if (!receipt) return false;
  if (receipt.image_hashes && JSON.stringify(receipt.image_hashes) !== JSON.stringify(images)) return false;
  receipt.image_hashes = images;
  return true;
}

function verifiedMdpoValidationGeneration(payload) {
  const receipt = validationGenerationReceipts.get(payload?.validation_generation_receipt);
  if (!receipt || receipt.kind !== 'mdpo_checkpoint' || payload?.type !== 'mdpo_val11_checkpoint' || payload?.split !== 'val' || Date.now() - receipt.issued_at > 4 * 60 * 60 * 1000) return null;
  if (receipt.run_id !== payload.candidate_id || receipt.role !== payload.role || receipt.group !== payload.group || receipt.fingerprint !== validationGenerationFingerprint(payload)) return null;
  return receipt;
}

function verifiedMdpoTest11Generation(payload) {
  const receipt = validationGenerationReceipts.get(payload?.validation_generation_receipt);
  if (!receipt || receipt.kind !== 'mdpo_test11' || receipt.phase !== 'test11' || payload?.type !== 'mdpo_test11_checkpoint' || payload?.split !== 'test'
      || Date.now() - receipt.issued_at > 4 * 60 * 60 * 1000) return null;
  if (receipt.run_id !== payload.test_run_id || receipt.lock_sha256 !== payload.lock_sha256 || receipt.role !== payload.role || receipt.group !== payload.group
      || receipt.fingerprint !== validationGenerationFingerprint(payload)) return null;
  return receipt;
}

async function loadMdpoTest11Context({ allowCompleted = false } = {}) {
  const lockFile = path.join(mdpoDir, 'test11_lock.json'), finalFile = path.join(mdpoDir, 'test11_final_report.json');
  const [lockBytes, activeBytes, preferenceBytes, activationReport, selection, manifest] = await Promise.all([
    fs.readFile(lockFile), fs.readFile(layoutModelFile), fs.readFile(preferenceModelFile),
    fs.readFile(path.join(mdpoDir, 'activation_report.json'), 'utf8').then(JSON.parse),
    fs.readFile(path.join(mdpoDir, 'hyperparameter_selection.json'), 'utf8').then(JSON.parse),
    fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8').then(JSON.parse)
  ]).catch((error) => {
    if (error?.code === 'ENOENT') throw Object.assign(new Error('MDPO test11 is not locked and formally activated; test access denied'), { statusCode: 409, code: 'MDPO_TEST11_NOT_LOCKED' });
    throw error;
  });
  const lock = JSON.parse(lockBytes.toString('utf8')), lockSha256 = createHash('sha256').update(lockBytes).digest('hex');
  const finalReport = await readOptionalJson(finalFile);
  if (finalReport && !allowCompleted) throw Object.assign(new Error('Locked MDPO test11 final report already exists; final evaluation may not be rerun'), { statusCode: 409, code: 'MDPO_TEST11_ALREADY_COMPLETE' });
  const baselineFile = path.resolve(root, String(lock.baseline_model?.file || '')), experimentsPrefix = path.resolve(experimentsDir) + path.sep;
  if (!baselineFile.startsWith(experimentsPrefix) || path.extname(baselineFile).toLowerCase() !== '.json') throw Object.assign(new Error('MDPO test11 baseline backup path invalid'), { statusCode: 409 });
  const baselineBytes = await fs.readFile(baselineFile);
  const cohort = manifest.samples.filter((sample) => sample.split === 'test').map((sample) => `${sample.category}/${sample.sample_id}`);
  const validated = validateMdpoTest11Prerequisites({ activeBytes, baselineBytes, preferenceBytes, activationReport, selection, cohort });
  const val11File = path.resolve(root, String(selection.selected?.report_file || ''));
  const fourGroupFile = path.resolve(root, String(selection.selected?.four_group_report_file || ''));
  const mdpoPrefix = path.resolve(mdpoDir) + path.sep;
  if (![val11File, fourGroupFile].every((file) => file.startsWith(mdpoPrefix) && path.extname(file).toLowerCase() === '.json')) throw Object.assign(new Error('test11 val11 evidence path invalid'), { statusCode: 409 });
  const [val11Bytes, fourGroupBytes] = await Promise.all([fs.readFile(val11File), fs.readFile(fourGroupFile)]);
  if (createHash('sha256').update(val11Bytes).digest('hex') !== selection.selected.report_sha256
      || createHash('sha256').update(fourGroupBytes).digest('hex') !== selection.selected.four_group_report_sha256) throw Object.assign(new Error('test11 val11 evidence hash changed since selection'), { statusCode: 409 });
  const val11 = JSON.parse(val11Bytes.toString('utf8'));
  if (!['v10_mdpo_val11_gate_report_v1', 'v10_mdpo_val11_gate_report_v2'].includes(val11.version) || val11.gate?.accepted !== true || val11.paired_provenance?.paired_samples !== 11
      || val11.baseline?.scorer_model !== val11.candidate?.scorer_model || val11.baseline?.prompt_version !== val11.candidate?.prompt_version
      || val11.baseline?.prompt_version !== `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`
      || JSON.stringify(val11.baseline?.views) !== JSON.stringify(VISION_KEYS)) throw Object.assign(new Error('test11 val11 fixed scorer/prompt/view evidence invalid'), { statusCode: 409 });
  const codeSha256 = {};
  for (const relative of Object.keys(lock.code_sha256 || {})) {
    const file = path.resolve(root, relative);
    if (!(file === root || file.startsWith(`${root}${path.sep}`))) throw Object.assign(new Error('MDPO test11 code-lock path escaped project root'), { statusCode: 409 });
    codeSha256[relative] = createHash('sha256').update(await fs.readFile(file)).digest('hex');
  }
  validateExistingMdpoTest11Lock(lock, { activeSha256: validated.activeSha256, baselineSha256: validated.baselineSha256,
    preferenceSha256: validated.preferenceSha256, candidateSha256: selection.selected.candidate_sha256,
    val11ReportSha256: selection.selected.report_sha256, fourGroupReportSha256: selection.selected.four_group_report_sha256,
    cohort: validated.cohort, codeSha256, scorerModel: val11.baseline.scorer_model, promptVersion: val11.baseline.prompt_version });
  if (scoringConfig.model !== lock.scorer.model) throw Object.assign(new Error('test11 Qwen scorer differs from locked val11 scorer'), { statusCode: 409 });
  const testRunId = `test11_${lockSha256.slice(0, 20)}`;
  return { lock, lockBytes, lockSha256, testRunId, finalReport, activeBytes, activeModel: validated.active, activeSha256: validated.activeSha256,
    baselineFile, baselineBytes, baselineModel: JSON.parse(baselineBytes.toString('utf8')), baselineSha256: validated.baselineSha256,
    preferenceSha256: validated.preferenceSha256, cohort: validated.cohort, selection, activationReport };
}

async function resolveMdpoLayoutEvaluation(evaluation, sample, activeModel) {
  if (!evaluation) return null;
  if (evaluation.phase === 'test11') {
    if (sample?.split !== 'test') throw Object.assign(new Error('v10-MDPO locked test11 evaluation is test split only'), { statusCode: 400, code: 'MDPO_TEST_SPLIT_INVALID' });
    const context = await loadMdpoTest11Context();
    let group;
    try { group = mdpoVal11Group(evaluation.group); }
    catch (error) { throw Object.assign(error, { statusCode: 400 }); }
    if (evaluation.test_run_id !== context.testRunId || evaluation.lock_sha256 !== context.lockSha256 || evaluation.role !== group.model_role) throw Object.assign(new Error('Invalid locked MDPO test11 run/role/group identity'), { statusCode: 409 });
    const baseline = evaluation.role === 'baseline';
    return { model: baseline ? context.baselineModel : activeModel, evaluation: { phase: 'test11', test_run_id: context.testRunId, lock_sha256: context.lockSha256,
      group: evaluation.group, role: evaluation.role, model_file: baseline ? path.relative(root, context.baselineFile).split(path.sep).join('/') : 'experiments/layout_model.json',
      candidate_file: baseline ? null : 'experiments/layout_model.json', candidate_sha256: baseline ? context.baselineSha256 : context.activeSha256,
      reference_model_sha256: context.baselineSha256, kind: baseline ? 'locked_original_v10_test11_baseline' : 'locked_active_v10_mdpo_test11' } };
  }
  if (sample?.split !== 'val') throw Object.assign(new Error('v10-MDPO checkpoint evaluation is val11 only'), { statusCode: 400, code: 'MDPO_VAL_SPLIT_INVALID' });
  const candidateId = String(evaluation.candidate_id || '');
  let group;
  try { group = mdpoVal11Group(evaluation.group); }
  catch (error) { throw Object.assign(error, { statusCode: 400 }); }
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(candidateId) || evaluation.role !== group.model_role) throw Object.assign(new Error('Invalid MDPO val11 candidate ID/role/group'), { statusCode: 400 });
  const activeBytes = await fs.readFile(layoutModelFile), activeHash = createHash('sha256').update(activeBytes).digest('hex');
  if (evaluation.role === 'baseline') return { model: activeModel, evaluation: { candidate_id: candidateId, group: evaluation.group, role: 'baseline', candidate_file: null, candidate_sha256: activeHash, reference_model_sha256: activeHash, kind: 'active_v10_baseline' } };
  const file = path.resolve(root, String(evaluation.candidate_file || ''));
  const mdpoPrefix = path.resolve(mdpoDir) + path.sep;
  if (!file.startsWith(mdpoPrefix) || path.extname(file).toLowerCase() !== '.json') throw Object.assign(new Error('MDPO candidate file must remain inside experiments/mdpo'), { statusCode: 400 });
  const bytes = await fs.readFile(file), sha256 = createHash('sha256').update(bytes).digest('hex');
  if (sha256 !== String(evaluation.candidate_sha256 || '').toLowerCase()) throw Object.assign(new Error('MDPO candidate SHA-256 mismatch'), { statusCode: 409 });
  const model = JSON.parse(bytes.toString('utf8'));
  if (model.version !== 'layout_model_v10_mdpo_candidate' || model.status !== 'diagnostic_only_requires_full_val11_gate' || model.reference?.sha256 !== activeHash || model.architecture?.qwen_inference_input !== false) throw Object.assign(new Error('MDPO candidate provenance/reference/inference contract invalid'), { statusCode: 409 });
  return { model, evaluation: { candidate_id: candidateId, group: evaluation.group, role: 'candidate', candidate_file: path.relative(root, file).split(path.sep).join('/'), candidate_sha256: sha256, reference_model_sha256: activeHash, kind: 'frozen_v10_mdpo_diagnostic_checkpoint' } };
}

async function resolveValidationPreferenceModel(evaluation, sample) {
  if (!evaluation) return null;
  if (sample?.split !== 'val') throw Object.assign(new Error('冻结检查点视觉评价只允许 manifest 的 val 样本'), { statusCode: 400, code: 'LLM_VISUAL_VAL_SPLIT_INVALID' });
  const runId = validatedRunId(evaluation.run_id ?? evaluation.runId);
  const round = Number(evaluation.round);
  if (!Number.isInteger(round) || round < 0 || round > 64) throw Object.assign(new Error('视觉验证轮次无效'), { statusCode: 400, code: 'LLM_VISUAL_VAL_ROUND_INVALID' });
  if (round === 0) return { model: null, evaluation: { run_id: runId, round: 0, kind: 'no_reward_baseline', reward_model_file: null, reward_model_sha256: null } };
  const ledger = JSON.parse(await fs.readFile(path.join(experimentsDir, `llm_preference_checkpoints_${runId}.json`), 'utf8'));
  const checkpoint = ledger.checkpoints?.find((item) => Number(item.round) === round);
  if (!checkpoint) throw Object.assign(new Error(`运行 ${runId} 不存在第 ${round} 轮冻结检查点`), { statusCode: 409, code: 'LLM_VISUAL_VAL_CHECKPOINT_MISSING' });
  const modelFile = path.join(experimentsDir, 'llm_preference_checkpoints', runId, `round_${round}.json`);
  const bytes = await fs.readFile(modelFile);
  const digest = createHash('sha256').update(bytes).digest('hex').toUpperCase();
  const model = JSON.parse(bytes.toString('utf8'));
  if (digest !== checkpoint.reward_model_sha256 || model.training?.run_id_filter !== runId) throw Object.assign(new Error('视觉验证检查点的哈希或 run_id 不匹配'), { statusCode: 409, code: 'LLM_VISUAL_VAL_CHECKPOINT_MISMATCH' });
  return { model, evaluation: { run_id: runId, round, kind: 'frozen_reward_checkpoint', reward_model_file: checkpoint.reward_model_file, reward_model_sha256: digest } };
}

async function getLayoutModel() {
  try {
    const stat = await fs.stat(layoutModelFile);
    if (stat.mtimeMs === layoutModelCache.mtimeMs) return layoutModelCache.model;
    const model = JSON.parse(await fs.readFile(layoutModelFile, 'utf8'));
    layoutModelCache = { mtimeMs: stat.mtimeMs, model };
    return model;
  } catch {
    layoutModelCache = { mtimeMs: 0, model: null };
    return null;
  }
}

async function getLeaderLengthPrior() {
  try {
    const stat = await fs.stat(leaderLengthPriorFile);
    if (stat.mtimeMs === leaderLengthPriorCache.mtimeMs) return leaderLengthPriorCache.prior;
    const prior = JSON.parse(await fs.readFile(leaderLengthPriorFile, 'utf8'));
    if (prior.source?.split !== 'train_only' || prior.source?.val_used_for_distribution || prior.source?.test_used_for_distribution) throw new Error('引导线长度先验必须只来自 train split');
    leaderLengthPriorCache = { mtimeMs: stat.mtimeMs, prior };
    return prior;
  } catch (error) {
    if (error.code !== 'ENOENT') console.error(`引导线长度先验加载失败：${error.message}`);
    leaderLengthPriorCache = { mtimeMs: 0, prior: null };
    return null;
  }
}

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.obj': 'text/plain; charset=utf-8',
  '.mtl': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const result = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) result.push(...await walk(full));
    else result.push(full);
  }
  return result;
}

function relData(fullPath) {
  return path.relative(root, fullPath).split(path.sep).join('/');
}

async function buildCatalog() {
  const layoutDir = path.join(dataDir, 'Layout');
  const categoryEntries = await fs.readdir(layoutDir, { withFileTypes: true });
  const categories = [];
  let totals = { samples: 0, obj: 0, mtl: 0, json: 0, png: 0 };

  for (const categoryEntry of categoryEntries.filter((e) => e.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
    const categoryDir = path.join(layoutDir, categoryEntry.name);
    const sampleEntries = await fs.readdir(categoryDir, { withFileTypes: true });
    const samples = [];
    for (const sampleEntry of sampleEntries.filter((e) => e.isDirectory()).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))) {
      const sampleDir = path.join(categoryDir, sampleEntry.name, 'layout1');
      let files = [];
      try { files = await walk(sampleDir); } catch { continue; }
      const byExt = (ext) => files.filter((f) => path.extname(f).toLowerCase() === ext);
      const objFiles = byExt('.obj').map(relData);
      const mtlFiles = byExt('.mtl').map(relData);
      const annotation = byExt('.json').find((f) => /[\\/]annotation[\\/]/i.test(f));
      const views = byExt('.png').map(relData);
      const mainObj = objFiles.find((f) => /-main-o\.obj$/i.test(f)) || objFiles[0] || null;
      const sample = {
        id: sampleEntry.name,
        category: categoryEntry.name,
        objFiles,
        mainObj,
        annotation: annotation ? relData(annotation) : null,
        views,
        counts: { obj: objFiles.length, mtl: mtlFiles.length, json: annotation ? 1 : 0, png: views.length }
      };
      samples.push(sample);
      totals.samples += 1;
      totals.obj += sample.counts.obj;
      totals.mtl += sample.counts.mtl;
      totals.json += sample.counts.json;
      totals.png += sample.counts.png;
    }
    categories.push({ name: categoryEntry.name, sampleCount: samples.length, samples });
  }
  return { generatedAt: new Date().toISOString(), categories, totals };
}

let catalogPromise;
function getCatalog() {
  if (!catalogPromise) catalogPromise = buildCatalog();
  return catalogPromise;
}

function safePath(base, pathname) {
  const candidate = path.resolve(base, pathname);
  return candidate === base || candidate.startsWith(`${base}${path.sep}`) ? candidate : null;
}

async function sendFile(res, filePath) {
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) throw new Error('not a file');
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    const stream = (await import('node:fs')).createReadStream(filePath);
    stream.on('error', () => res.destroy());
    stream.pipe(res);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
}

async function readJsonBody(req) {
  let body = '';
  for await (const chunk of req) {
    body += chunk;
    if (body.length > 64_000_000) throw new Error('请求体超过 64 MB 限制');
  }
  return body ? JSON.parse(body) : {};
}

function sendJson(res, status, value) {
  res.writeHead(status, { 'Content-Type': mimeTypes['.json'], 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(value));
}

const QUALITY_COMPARISON_METHODS = [
  { id: 'BinoForce_final_snapshot', label: 'BinoForce' },
  { id: 'hedgehog_1d', label: 'Hedgehog 1D' },
  { id: 'hedgehog_3d', label: 'Hedgehog 3D' },
  { id: 'current_fixed_label_seed17', label: '模型一（基础模型）' }
];
const QUALITY_COMPARISON_FIELDS = [
  'multidimensional_quality_score', 'text_clarity', 'label_label_occlusion_ratio',
  'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio',
  'mean_anchor_distance', 'leader_length_compliance_ratio', 'directional_allocation_mismatch',
  'directional_uniformity', 'manual_style_distance', 'air_voxel_available_ratio',
  'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance',
  'min_3d_clearance', 'mean_3d_spacing', 'min_3d_spacing', 'leader_length_mean', 'out_of_sight_ratio'
];
const finiteOrNull = (value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
async function readOptionalJson(file) {
  try { return JSON.parse(await fs.readFile(file, 'utf8')); }
  catch { return null; }
}

async function buildMoeDashboardData() {
  const files = {
    routingAudit: path.join(experimentsDir, 'moe_style_routing_audit.json'),
    pseudoLabels: path.join(experimentsDir, 'moe_unsupervised_pseudolabels_train.json'),
    humanTemplate: path.join(experimentsDir, 'moe_style_human_score_template.csv'),
    llmStyleScores: path.join(experimentsDir, 'moe_style_llm_scores.json'),
    llmPriorReport: path.join(experimentsDir, 'moe_style_llm_vs_geometry_report.json'),
    candidateProjection: path.join(experimentsDir, 'moe_layout_llm_candidates.json'),
    previewManifest: path.join(experimentsDir, 'moe_layout_llm_candidate_previews', 'manifest.json'),
    candidateScores: path.join(experimentsDir, 'moe_layout_llm_candidate_scores.json'),
    pseudoModel: path.join(experimentsDir, 'layout_model_v10_moe_pseudo_candidate_diagnostic.json'),
    pseudoReport: path.join(experimentsDir, 'layout_training_v10_moe_pseudo_report.json'),
    expertLayoutsTrain: path.join(experimentsDir, 'moe_expert_layouts_train33.json'),
    expertSelectionTrain: path.join(experimentsDir, 'moe_expert_selection_train33.json'),
    expertLayoutsTest: path.join(experimentsDir, 'moe_expert_layouts_test11.json'),
    expertSelectionTest: path.join(experimentsDir, 'moe_expert_selection_test11.json'),
    expertPreviewManifestTest: path.join(experimentsDir, 'moe_expert_preview_test11', 'manifest.json'),
    extendedComparison: path.join(experimentsDir, 'comparisons', 'extended_200train_test50', 'comparison.json'),
    legacyComparison: path.join(experimentsDir, 'comparisons', 'latest_reproduction_metrics', 'comparison.json')
  };
  const [routingAudit, pseudoLabels, llmStyleScores, llmPriorReport, candidateProjection, previewManifest, candidateScores, pseudoModel, pseudoReport, expertLayoutsTrain, expertSelectionTrain, expertLayoutsTest, expertSelectionTest, expertPreviewManifestTest, extendedComparison, legacyComparison] = await Promise.all([
    readOptionalJson(files.routingAudit), readOptionalJson(files.pseudoLabels), readOptionalJson(files.llmStyleScores), readOptionalJson(files.llmPriorReport), readOptionalJson(files.candidateProjection), readOptionalJson(files.previewManifest), readOptionalJson(files.candidateScores), readOptionalJson(files.pseudoModel), readOptionalJson(files.pseudoReport), readOptionalJson(files.expertLayoutsTrain), readOptionalJson(files.expertSelectionTrain), readOptionalJson(files.expertLayoutsTest), readOptionalJson(files.expertSelectionTest), readOptionalJson(files.expertPreviewManifestTest), readOptionalJson(files.extendedComparison), readOptionalJson(files.legacyComparison)
  ]);
  const stat = async (file) => fs.stat(file).then((item) => ({ exists: true, bytes: item.size, updated_at: item.mtime.toISOString(), path: path.relative(root, file).split(path.sep).join('/') }), () => ({ exists: false, path: path.relative(root, file).split(path.sep).join('/') }));
  return {
    version: 'moe_dashboard_data_v1',
    generated_at: new Date().toISOString(),
    files: Object.fromEntries(await Promise.all(Object.entries(files).map(async ([key, file]) => [key, await stat(file)]))),
    workflow: [
      { step: '1 架构与数据审计', command: 'npm run audit:moe-styles', artifact: 'experiments/moe_style_routing_audit.json' },
      { step: '2 无监督伪标签', command: 'npm run generate:moe-pseudolabels -- --iterations 120', artifact: 'experiments/moe_unsupervised_pseudolabels_train.json' },
      { step: '3 人工/LLM 风格校准', command: 'npm run prepare:moe-style-llm && npm run run:moe-style-llm && npm run analyze:moe-style-llm-prior', artifact: 'experiments/moe_style_llm_vs_geometry_report.json' },
      { step: '4 扰动候选五视角偏好', command: 'npm run generate:moe-layout-llm-candidates && npm run render:moe-layout-llm-candidates && npm run score:moe-layout-llm-candidates', artifact: 'experiments/moe_layout_llm_candidate_scores.json' },
      { step: '5 v10 伪标签训练', command: 'npm run train:layout:v10:pseudo', artifact: 'experiments/layout_model_v10_moe_pseudo_candidate_diagnostic.json' }
    ],
    architecture: {
      model_files_to_operate: ['lib/moe-layout-styles.mjs', 'lib/layout-optimizer.mjs', 'lib/mdpo-candidate-perturbation.mjs', 'scripts/train-3d-human-style-layout-model.mjs', 'scripts/train-v10-mdpo.mjs'],
      data_files_to_operate: ['experiments/moe_style_routing_audit.json', 'experiments/moe_unsupervised_pseudolabels_train.json', 'experiments/moe_expert_layouts_train33.json', 'experiments/moe_expert_selection_train33.json', 'experiments/moe_expert_layouts_test11.json', 'experiments/moe_expert_selection_test11.json', 'experiments/moe_layout_llm_candidate_scores.json', 'experiments/layout_model_v10_moe_pseudo_candidate_diagnostic.json'],
      ui_entry: '/moe-dashboard.html'
    },
    summaries: {
      routing: routingAudit ? { count: routingAudit.count, summary: routingAudit.summary } : null,
      pseudo_labels: pseudoLabels ? { count: pseudoLabels.count, summary: pseudoLabels.summary, style_counts: pseudoLabels.style_counts } : null,
      llm_style: llmStyleScores ? { count: llmStyleScores.count, counts: Object.fromEntries(['spherical', 'rectangular', 'surround'].map((name) => [name, llmStyleScores.rows.filter((row) => row.selected === name).length])) } : null,
      llm_vs_geometry: llmPriorReport ? { n: llmPriorReport.n, agreement: llmPriorReport.agreement, counts: llmPriorReport.counts, dimensions: llmPriorReport.dimensions } : null,
      candidate_projection: candidateProjection ? { samples: candidateProjection.rows.length, candidates: candidateProjection.rows.reduce((sum, row) => sum + row.candidates.length, 0), modes: candidateProjection.modes } : null,
      candidate_scores: candidateScores ? { count: candidateScores.count, winners: candidateScores.winners, score_keys: candidateScores.score_keys } : null,
      pseudo_training: pseudoModel ? { version: pseudoModel.version, target_source: pseudoModel.hyperparameters?.target_source, pseudo_target_count: pseudoModel.training?.pseudo_target_count, val_quality: pseudoReport?.view_metrics?.val?.multidimensional_quality_score || pseudoModel.training?.view_metrics?.val?.multidimensional_quality_score, test_quality: pseudoReport?.view_metrics?.test?.multidimensional_quality_score || pseudoModel.training?.view_metrics?.test?.multidimensional_quality_score, routing: pseudoModel.training?.routing || null } : null,
      expert_selection: expertSelectionTrain ? { count: expertSelectionTrain.count, summary: expertSelectionTrain.summary, source: 'geometry_plus_llm_style_score', train_layouts: expertLayoutsTrain?.count || null } : null,
      test_expert_selection: expertSelectionTest ? { count: expertSelectionTest.count, summary: expertSelectionTest.summary, layouts: expertLayoutsTest?.count || null } : null,
      extended_comparison: extendedComparison ? { cohort: extendedComparison.cohort, methods: extendedComparison.methods, interpretation: extendedComparison.interpretation } : null,
      legacy_comparison: legacyComparison ? { split_counts: legacyComparison.split_counts, test_summaries: (legacyComparison.summaries || []).filter((row) => row.split === 'test') } : null
    },
    test_samples: expertSelectionTest ? expertSelectionTest.rows.map((selectionRow) => { const layoutRow = expertLayoutsTest?.rows?.find((row) => row.category === selectionRow.category && row.sample_id === selectionRow.sample_id); const previewRows = (expertPreviewManifestTest?.previews || []).filter((preview) => preview.category === selectionRow.category && preview.sample_id === selectionRow.sample_id); return { ...selectionRow, expert_metrics: Object.fromEntries(Object.entries(selectionRow.experts || {}).map(([style, expert]) => [style, { geometry_score: expert.geometry_score, llm_style_score: expert.llm_style_score, combined_score: expert.combined_score, metrics: expert.metrics }])), selected_labels: layoutRow?.experts?.[selectionRow.selected_expert]?.labels || [], previews: previewRows }; }) : [],
    previews: previewManifest?.previews || []
  };
}

const moeDashboardClients = new Set();
let moeDashboardSnapshot = '';
async function broadcastMoeDashboardUpdates() {
  if (!moeDashboardClients.size) return;
  try {
    const snapshot = JSON.stringify(await buildMoeDashboardData());
    if (snapshot === moeDashboardSnapshot) return;
    moeDashboardSnapshot = snapshot;
    for (const client of moeDashboardClients) { try { client.write(`data: ${snapshot}\n\n`); } catch { moeDashboardClients.delete(client); } }
  } catch (error) { for (const client of moeDashboardClients) { try { client.write(`event: error\ndata: ${JSON.stringify({ error: error.message })}\n\n`); } catch { moeDashboardClients.delete(client); } } }
}

const mdpoDigest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const mdpoProcessAlive = (pid) => {
  if (!Number.isInteger(Number(pid)) || Number(pid) <= 0) return false;
  try { process.kill(Number(pid), 0); return true; }
  catch (error) { if (error?.code === 'ESRCH') return false; return true; }
};
async function writeMdpoUiTrainingState(state) {
  await fs.mkdir(mdpoDir, { recursive: true });
  const temporary = path.join(mdpoDir, '.manual_training_job.' + randomUUID() + '.tmp');
  await fs.writeFile(temporary, JSON.stringify(state, null, 2) + '\n', { flag: 'wx' });
  await replaceFileWithRetry(temporary, mdpoUiTrainingStateFile);
}
async function mdpoLogTail(file, maximumLines = 24) {
  try { return (await fs.readFile(file, 'utf8')).split(/\r?\n/).filter(Boolean).slice(-maximumLines); }
  catch (error) { if (error?.code === 'ENOENT') return []; throw error; }
}
async function mdpoTrainingDatasetReadiness() {
  try {
    const [datasetBytes, manifest, referenceBytes] = await Promise.all([
      fs.readFile(mdpoPairFile), fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8').then(JSON.parse), fs.readFile(layoutModelFile)
    ]);
    const dataset = JSON.parse(datasetBytes.toString('utf8'));
    const audit = validateMdpoDataset(dataset, manifest, { requireComplete: false });
    const referenceSha256 = mdpoDigest(referenceBytes), datasetSha256 = mdpoDigest(datasetBytes);
    let complete = false, completionError = null;
    try {
      validateMdpoDataset(dataset, manifest, { requireComplete: true });
      const finalAudit = JSON.parse(await fs.readFile(path.join(mdpoDir, 'dataset_audit_final.json'), 'utf8'));
      validateMdpoFinalDatasetAudit({ datasetBytes, audit: finalAudit, referenceSha256 });
      complete = true;
    }
    catch (error) { completionError = error?.code === 'ENOENT' ? '等待 train33 完整数据与不可变正式数据审计' : error.message; }
    const referenceMatches = dataset.reference_model_sha256 === referenceSha256;
    const referenceIsOriginal = JSON.parse(referenceBytes.toString('utf8')).version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe';
    return { available: true, complete: complete && referenceMatches, sample_count: audit.sample_count,
      pair_count: audit.pair_count, reference_matches: referenceMatches, reference_sha256: referenceSha256,
      dataset_sha256: datasetSha256, error: !referenceIsOriginal ? '当前活动模型不是原始 v10，不能作为新的冻结参考策略'
        : referenceMatches ? completionError : 'train33 冻结参考哈希与当前活动 v10 不一致',
      reference_is_original_v10: referenceIsOriginal, ready_for_training: complete && referenceMatches && referenceIsOriginal };
  } catch (error) {
    return { available: false, complete: false, sample_count: 0, pair_count: 0, reference_matches: false, error: error.message };
  }
}
async function mdpoFormalPipelineState() {
  const ledger = await fs.readFile(path.join(mdpoDir, 'pipeline', 'full_pipeline_ledger.json'), 'utf8').then(JSON.parse).catch((error) => {
    if (error?.code === 'ENOENT') return null;
    throw error;
  });
  const running = (ledger?.stages || []).find((row) => row.status === 'running') || null;
  // Treat the entire formal run as exclusive, including gaps between stages.
  const terminal = ledger?.status === 'complete' || (!running && ledger?.stages?.some((row) => row.status === 'failed'));
  return { status: ledger?.status || (running ? 'running' : ledger?.stages?.some((row) => row.status === 'failed') ? 'failed' : ledger ? 'waiting' : 'not_started'), running_stage: running?.id || null,
    blocks_manual_training: Boolean(ledger && !terminal) };
}
async function reconcileMdpoUiTrainingJob() {
  const job = await fs.readFile(mdpoUiTrainingStateFile, 'utf8').then(JSON.parse).catch((error) => {
    if (error?.code === 'ENOENT') return null;
    throw error;
  });
  if (!job || job.status !== 'running' || mdpoProcessAlive(job.pid)) return job;
  let next;
  try {
    await validateCompletedMdpoTrainingRun({ outputDir: path.resolve(root, job.output_dir), root,
      referenceHash: job.reference_sha256, datasetHash: job.dataset_sha256, options: job.parameters });
    next = { ...job, status: 'completed_diagnostic_candidate', finished_at: new Date().toISOString(), exit_code: job.exit_code ?? 0,
      candidate_file: path.posix.join(job.output_dir.replaceAll('\\', '/'), 'layout_model_v10_mdpo_candidate.json'),
      checkpoint_ledger_file: path.posix.join(job.output_dir.replaceAll('\\', '/'), 'checkpoint_ledger.json') };
  } catch (error) {
    next = { ...job, status: 'failed', finished_at: new Date().toISOString(), error: job.error || error.message };
  }
  await writeMdpoUiTrainingState(next);
  return next;
}
async function mdpoUiTrainingControlStatus({ ignoreSubmitting = false, ignoreQueued = false } = {}) {
  const [job, dataset, pipeline] = await Promise.all([reconcileMdpoUiTrainingJob(), mdpoTrainingDatasetReadiness(), mdpoFormalPipelineState()]);
  const blockers = [];
  if (!dataset.ready_for_training) blockers.push(dataset.error || 'train33 尚未达到 33/33 与 198–396 对的严格门槛');
  if (job?.status === 'running') blockers.push('已有界面训练任务正在运行');
  if (job?.status === 'queued' && !ignoreQueued) blockers.push('已有界面训练任务排队中');
  if (pipeline.blocks_manual_training) blockers.push('正式流水线正在执行 ' + (pipeline.running_stage || '阶段切换') + '；独立训练须等待正式流水线结束');
  if (mdpoUiTrainingStarting && !ignoreSubmitting) blockers.push('训练任务正在提交，请勿重复点击');
  const safeJobFile = (relative) => {
    if (!relative) return null;
    const file = path.resolve(root, relative);
    if (!file.startsWith(mdpoUiTrainingRunsDir + path.sep)) throw new Error('训练日志路径不在独立训练目录中');
    return file;
  };
  const stdoutFile = safeJobFile(job?.stdout);
  const stderrFile = safeJobFile(job?.stderr);
  return { version: 'v10_mdpo_ui_training_control_v1', defaults: MDPO_UI_TRAINING_DEFAULTS,
    allowed_models: [{ id: 'v10_mdpo', label: 'v10-MDPO（冻结原始 v10 参考）' }], dataset, pipeline, blockers,
    can_start: blockers.length === 0, can_submit: blockers.length === 0 || (pipeline.blocks_manual_training && blockers.length === 1),
    job: job ? { ...job, alive: job.status === 'running' && mdpoProcessAlive(job.pid),
      stdout_tail: stdoutFile ? await mdpoLogTail(stdoutFile) : [], stderr_tail: stderrFile ? await mdpoLogTail(stderrFile) : [] } : null };
}
async function startMdpoUiTraining(payload, queuedJob = null) {
  if (mdpoUiTrainingStarting) { const error = new Error('界面训练任务正在提交'); error.statusCode = 409; throw error; }
  mdpoUiTrainingStarting = true;
  try {
  const options = normalizeMdpoUiTrainingRequest(payload);
  const current = await mdpoUiTrainingControlStatus({ ignoreSubmitting: true, ignoreQueued: Boolean(queuedJob) });
  if (!current.can_start && !(current.can_submit && !queuedJob)) {
    const error = new Error(current.blockers.join('；'));
    error.statusCode = 409; error.code = 'MDPO_TRAINING_NOT_READY'; error.status = current;
    throw error;
  }
  if (queuedJob && (queuedJob.run_id !== current.job?.run_id || queuedJob.dataset_sha256 !== current.dataset.dataset_sha256
      || queuedJob.reference_sha256 !== current.dataset.reference_sha256 || !current.can_start)) {
    throw Object.assign(new Error('排队训练的 train33 数据或冻结参考模型已变化；请重新提交'), { statusCode: 409 });
  }
  const runId = queuedJob?.run_id || 'ui_mdpo_' + new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14) + '_' + randomUUID().slice(0, 8);
  const outputDir = path.join(mdpoUiTrainingRunsDir, runId), stdoutFile = path.join(outputDir, 'stdout.log'), stderrFile = path.join(outputDir, 'stderr.log');
  const relativeOutput = path.relative(root, outputDir).split(path.sep).join('/');
  if (!current.can_start) {
    const job = { version: 'v10_mdpo_ui_training_job_v1', run_id: runId, model: options.model,
      status: 'queued', queued_at: new Date().toISOString(), pid: null, parameters: options,
      dataset_sha256: current.dataset.dataset_sha256, reference_sha256: current.dataset.reference_sha256,
      output_dir: relativeOutput, stdout: relativeOutput + '/stdout.log', stderr: relativeOutput + '/stderr.log',
      activation_policy: 'diagnostic_only_requires_full_val11_gate' };
    await writeMdpoUiTrainingState(job);
    return { ...job, message: '训练参数已保存；正式流水线结束后将自动启动独立诊断候选训练' };
  }
  await fs.mkdir(mdpoUiTrainingRunsDir, { recursive: true });
  await fs.mkdir(outputDir, { recursive: false });
  const args = mdpoUiTrainingArgs(options, outputDir);
  const stdout = await fs.open(stdoutFile, 'a'), stderr = await fs.open(stderrFile, 'a');
  let child;
  let earlySpawnError = null;
  try {
    child = spawn(process.execPath, args, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] });
    child.once('error', (error) => { earlySpawnError = error; });
  } finally {
    await stdout.close(); await stderr.close();
  }
  const job = { version: 'v10_mdpo_ui_training_job_v1', run_id: runId, model: options.model,
    status: 'running', started_at: new Date().toISOString(), pid: child.pid, parameters: options,
    dataset_sha256: current.dataset.dataset_sha256, reference_sha256: current.dataset.reference_sha256,
    output_dir: relativeOutput, stdout: path.relative(root, stdoutFile).split(path.sep).join('/'),
    stderr: path.relative(root, stderrFile).split(path.sep).join('/'), activation_policy: 'diagnostic_only_requires_full_val11_gate' };
  try { await writeMdpoUiTrainingState(job); }
  catch (error) { if (mdpoProcessAlive(child.pid)) child.kill(); throw error; }
  mdpoUiTrainingChild = child;
  let finalized = false;
  const finalize = async (code, spawnError = null) => {
    if (finalized) return; finalized = true;
    const latest = await readOptionalJson(mdpoUiTrainingStateFile);
    if (latest?.run_id !== runId) return;
    let next = { ...latest, exit_code: Number.isInteger(code) ? code : null, finished_at: new Date().toISOString() };
    try {
      if (spawnError) throw spawnError;
      if (code !== 0) throw new Error('v10-MDPO 训练进程退出码：' + code);
      await validateCompletedMdpoTrainingRun({ outputDir, root, referenceHash: job.reference_sha256,
        datasetHash: job.dataset_sha256, options });
      next = { ...next, status: 'completed_diagnostic_candidate',
        candidate_file: relativeOutput + '/layout_model_v10_mdpo_candidate.json', checkpoint_ledger_file: relativeOutput + '/checkpoint_ledger.json' };
    } catch (error) { next = { ...next, status: 'failed', error: error.message }; }
    await writeMdpoUiTrainingState(next);
    if (mdpoUiTrainingChild?.pid === child.pid) mdpoUiTrainingChild = null;
  };
  if (earlySpawnError) finalize(null, earlySpawnError).catch(console.error);
  child.once('error', (error) => { finalize(null, error).catch(console.error); });
  child.once('exit', (code) => { finalize(code).catch(console.error); });
  return { ...job, message: 'v10-MDPO 自定义候选训练已启动；结果不会自动激活或覆盖当前模型' };
  } finally { mdpoUiTrainingStarting = false; }
}
async function dispatchQueuedMdpoUiTraining() {
  if (mdpoUiTrainingStarting) return;
  const job = await readOptionalJson(mdpoUiTrainingStateFile);
  if (job?.status !== 'queued') return;
  const pipeline = await mdpoFormalPipelineState();
  if (pipeline.blocks_manual_training) return;
  try { await startMdpoUiTraining(job.parameters, job); }
  catch (error) {
    const latest = await readOptionalJson(mdpoUiTrainingStateFile);
    if (latest?.run_id === job.run_id && latest.status === 'queued')
      await writeMdpoUiTrainingState({ ...latest, status: 'failed', finished_at: new Date().toISOString(), error: error.message });
  }
}
function qualityRow(source, method, label) {
  return {
    method,
    label,
    ...Object.fromEntries(QUALITY_COMPARISON_FIELDS.map((field) => [field, finiteOrNull(source?.[field] ?? source?.manual_similarity?.[field])]))
  };
}
function meanQualityRows(rows, method, label) {
  const output = { method, label };
  for (const field of QUALITY_COMPARISON_FIELDS) {
    const values = rows.map((row) => finiteOrNull(row?.[field] ?? row?.manual_similarity?.[field])).filter((value) => value !== null);
    output[field] = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }
  return output;
}
function verifiedPreferenceComparison(report, runReport) {
  const runId = String(report?.run_id || '');
  const rows = Array.isArray(report?.rows) ? report.rows : [];
  const required = ['multidimensional_quality_score', 'leader_length_compliance_ratio', 'directional_allocation_mismatch', 'directional_uniformity'];
  return /^[a-zA-Z0-9_-]{8,80}$/.test(runId)
    && runReport?.run_id === runId
    && runReport?.evidence?.pair_count > 0
    && runReport?.test11?.test_samples === 11
    && rows.length === 11
    && rows.every((row) => row.split === 'test' && required.every((field) => finiteOrNull(row.preferred?.[field]) !== null));
}
function verifiedConvergence(convergence, runId) {
  const standard = Array.isArray(convergence?.checkpoints) ? convergence.checkpoints.filter((item) => [0, 1, 2, 4, 8].includes(Number(item.round))) : [];
  return convergence?.run_id === runId && convergence?.evidence_sufficient === true && standard.length === 5 && standard.every((item) => item.status === 'evaluated' && item.checkpoint_verified);
}
async function buildQualityComparison(category, sampleId) {
  const comparison = await readOptionalJson(qualityComparisonFile);
  if (!comparison?.unified_snapshot_evaluation) throw Object.assign(new Error('统一方法比较结果尚未生成'), { statusCode: 503, code: 'QUALITY_COMPARISON_MISSING' });
  const unified = comparison.unified_snapshot_evaluation;
  const sourcePaperMetrics = (comparison.source_summaries || [])
    .filter((row) => row.scope === 'test11_five_views' && ['hedgehog_1d', 'hedgehog_3d', 'BinoForce'].includes(row.method))
    .map((row) => ({ method: row.method, label: row.method === 'BinoForce' ? 'BinoForce' : row.method === 'hedgehog_1d' ? 'Hedgehog 1D' : 'Hedgehog 3D', samples: row.samples, PCK_005: finiteOrNull(row.PCK_005), PCK_010: finiteOrNull(row.PCK_010), OLR: finiteOrNull(row.OLR), LCD: finiteOrNull(row.LCD), DBV: finiteOrNull(row.DBV), avg_leader_length: finiteOrNull(row.avg_leader_length), quality_score: finiteOrNull(row.quality_score) }));
  const summaries = Array.isArray(unified.summary) ? unified.summary : [];
  const rows = Array.isArray(unified.rows) ? unified.rows : [];
  const averages = QUALITY_COMPARISON_METHODS.map((method) => {
    const source = summaries.find((row) => row.method === method.id && row.strict_label_contract !== false);
    return source ? qualityRow(source, method.id, method.label) : null;
  }).filter(Boolean);
  const all55Averages = QUALITY_COMPARISON_METHODS.map((method) => {
    const entries = rows.filter((row) => row.method === method.id && row.strict_label_contract !== false);
    return entries.length === 55 ? meanQualityRows(entries, method.id, method.label) : null;
  }).filter(Boolean);
  const sampleRows = category && sampleId ? QUALITY_COMPARISON_METHODS.map((method) => {
    const source = rows.find((row) => row.method === method.id && row.category === category && String(row.sample_id) === String(sampleId) && row.strict_label_contract !== false);
    return source ? qualityRow(source, method.id, method.label) : null;
  }).filter(Boolean) : [];

  const v9Selection = await readOptionalJson(v9InferenceSelectionFile);
  const v9Rows = Array.isArray(v9Selection?.test_confirmation?.candidate?.rows) ? v9Selection.test_confirmation.candidate.rows : [];
  const v9QualityRows = v9Rows.map((row) => ({ category: row.category, sample_id: row.sample_id, ...row.view, ...row.spatial }));
  if (v9QualityRows.length === 11) {
    averages.push(meanQualityRows(v9QualityRows, 'layout_model_v9_3d', 'v9 纯三维人类风格（test11）'));
    const v9Sample = v9QualityRows.find((row) => row.category === category && String(row.sample_id) === String(sampleId));
    if (v9Sample) sampleRows.push(qualityRow(v9Sample, 'layout_model_v9_3d', 'v9 纯三维人类风格'));
  }

  const v10Report = await readOptionalJson(v10TrainingReportFile);
  const v10Test = v10Report?.metrics?.test && v10Report?.view_metrics?.test;
  if (v10Test && Number(v10Report.view_metrics.test.sample_count) === 11) {
    averages.push(qualityRow({ ...v10Report.metrics.test, ...v10Report.view_metrics.test }, 'layout_model_v10_anchor_frame', 'v10 锚点局部坐标异构图（test11）'));
  }

  const preferenceReport = await readOptionalJson(preferenceTestReportFile);
  const preferenceRun = await readOptionalJson(preferenceRunLatestFile);
  const convergence = await readOptionalJson(preferenceConvergenceFile);
  const llmAvailable = verifiedPreferenceComparison(preferenceReport, preferenceRun);
  if (llmAvailable) {
    const preferredRows = preferenceReport.rows.map((row) => ({ category: row.category, sample_id: row.sample_id, ...row.preferred }));
    averages.push(meanQualityRows(preferredRows, 'llm_preference_after', '最终模型（LLM 创新）'));
    const samplePreferred = preferredRows.find((row) => row.category === category && String(row.sample_id) === String(sampleId));
    if (samplePreferred) sampleRows.push(qualityRow(samplePreferred, 'llm_preference_after', '最终模型（LLM 创新）'));
  }
  const convergenceRows = llmAvailable && verifiedConvergence(convergence, preferenceReport.run_id) ? convergence.checkpoints
    .filter((item) => [0, 1, 2, 4, 8].includes(Number(item.round)))
    .map((item) => ({ round: Number(item.round), composite_score: finiteOrNull(item.composite_score), score_means: item.score_means || null })) : [];
  return {
    generated_at: comparison.generated_at,
    protocol: { description: unified.protocol, camera: comparison.camera_protocol, cohort: 'frozen test11', sample_count: 11 },
    metrics: QUALITY_COMPARISON_FIELDS,
    sample: { category: category || null, sample_id: sampleId || null, available: sampleRows.length > 0, rows: sampleRows },
    averages,
    all55_averages: all55Averages,
    source_paper_metrics: {
      scope: 'test11_five_views',
      rows: sourcePaperMetrics,
      definitions: [
        { key: 'PCK@0.05 / PCK@0.10', direction: 'higher', meaning: 'Percentage of Correct Keypoints: projected label center lies within 5% / 10% of the image diagonal from the manual reference.' },
        { key: 'OLR', direction: 'lower', meaning: 'Overlap Ratio: average projected label-label and label-object overlap ratio.' },
        { key: 'LCD', direction: 'lower', meaning: 'Leader-line Crossing Degree: projected leader-line crossing penalty.' },
        { key: 'DBV', direction: 'lower', meaning: 'Double Vision Degree: left/right-eye overlap imbalance; unavailable for imported Hedgehog layouts.' },
        { key: 'Avg. Leader Length', direction: 'lower', meaning: 'Leader-line length normalized by the image diagonal.' },
        { key: 'Quality Score', direction: 'higher', meaning: 'Reproduction composite: PCK reward minus overlap/crossing penalties; not a source-paper user-study score.' }
      ],
      boundary: 'Source-schema reproduction metrics are shown separately from the unified Node evaluator and must not be merged into one ranking.'
    },
    llm_available: llmAvailable,
    llm_status: llmAvailable ? `真实运行 ${preferenceReport.run_id}，test11=${preferenceReport.test_samples}；偏好奖励${preferenceReport.candidate_activated ? '已通过 val 激活' : 'val 未通过、保留为诊断候选'}；${convergenceRows.length ? '0/1/2/4/8 趋势完整' : '单轮结果'}` : '等待真实 Qwen 运行与 test11 回执',
    convergence: convergenceRows
  };
}

function bboxLabel(bounds) {
  return [{ id: 'model-bounding-box', text: 'OBJ MODEL', anchor: [bounds.center[0], bounds.max[1], bounds.center[2]], center: [bounds.center[0], bounds.max[1] + bounds.radius * 0.7, bounds.center[2]], boxSize: [bounds.radius * 1.55, bounds.radius * 0.22, bounds.radius * 0.025], bendPoints: [], sourceObjs: [], targetGroups: [] }];
}

function normalizeLayoutOptions(strategy = {}) {
  const allowed = {
    viewPolicy: ['binocular', 'single'],
    groupPolicy: ['all', 'semantic-once', 'symmetric'],
    sizePolicy: ['relative', 'fixed', 'distance-aware'],
    optimizer: ['rules', 'annealing']
  };
  const options = {
    viewPolicy: allowed.viewPolicy.includes(strategy.viewPolicy) ? strategy.viewPolicy : 'binocular',
    groupPolicy: allowed.groupPolicy.includes(strategy.groupPolicy) ? strategy.groupPolicy : 'all',
    sizePolicy: allowed.sizePolicy.includes(strategy.sizePolicy) ? strategy.sizePolicy : 'relative',
    optimizer: allowed.optimizer.includes(strategy.optimizer) ? strategy.optimizer : 'annealing',
    seed: Number.isInteger(Number(strategy.seed)) ? Math.max(0, Number(strategy.seed)) : 17,
    iterations: Number.isInteger(Number(strategy.iterations)) ? Math.max(20, Math.min(2000, Number(strategy.iterations))) : 180,
    preferenceRerank: strategy.preferenceRerank !== false
  };
  options.mdpoCandidate = strategy.mdpoCandidate === true;
  options.mdpoPerturbation = Number.isInteger(Number(strategy.mdpoPerturbation)) ? Math.max(0, Math.min(MDPO_PERTURBATION_MODES.length - 1, Number(strategy.mdpoPerturbation))) : 0;
  return options;
}

async function generateLayout(payload) {
  const startedAt = performance.now();
  const timing = {};
  let rawObj = payload?.source_obj_text;
  let sample = payload?.sample || null;
  let referenceAnnotation = payload?.annotation || null;
  let sourceName = payload?.source_name || 'input.obj';
  if (sample?.category || sample?.sample_id) {
    if (!sample?.category || !sample?.sample_id) throw Object.assign(new Error('必须同时提供样本类别和编号'), { statusCode: 400, code: 'SAMPLE_INVALID' });
    const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
    const manifestSample = manifest.samples.find((item) => item.category === sample.category && item.sample_id === String(sample.sample_id));
    if (!manifestSample) throw Object.assign(new Error('manifest 中找不到该样本'), { statusCode: 404, code: 'SAMPLE_NOT_FOUND' });
    // Dataset samples always use the authoritative OBJ/annotation; request fields cannot redefine fixed labels.
    rawObj = await fs.readFile(path.join(root, manifestSample.input.source_obj), 'utf8');
    sample = { category: manifestSample.category, sample_id: manifestSample.sample_id, split: manifestSample.split };
    sourceName = path.basename(manifestSample.input.source_obj);
    referenceAnnotation = manifestSample.target.annotation_json ? JSON.parse(await fs.readFile(path.join(root, manifestSample.target.annotation_json), 'utf8')) : null;
  }
  if (typeof rawObj !== 'string' || !rawObj.trim()) throw Object.assign(new Error('请提供 sample 或 source_obj_text'), { statusCode: 400, code: 'INPUT_MISSING' });
  const cleanStartedAt = performance.now();
  const clean = cleanObj(rawObj);
  timing.clean_ms = Number((performance.now() - cleanStartedAt).toFixed(3));
  if (!clean.vertexCount || !clean.faceCount) throw Object.assign(new Error('OBJ 清洗后没有有效几何'), { statusCode: 422, code: 'EMPTY_CLEAN_GEOMETRY' });
  const bounds = boundsFromObj(clean.text);
  const geometry = parseObjTriangles(clean.text);
  const cvFeatures = await buildCvFeatures({ objText: clean.text, bounds });
  const activeLayoutModel = await getLayoutModel();
  const requestedDisplayModel = String(payload?.display_model || 'latest_completed_aligned');
  const displayModel = ['v10_no_rerank', 'v10_historical_rerank', 'mdpo_no_rerank', 'mdpo_safe_rerank', 'latest_completed_aligned'].includes(requestedDisplayModel)
    ? requestedDisplayModel : 'latest_completed_aligned';
  const alignedDisplayModel = !payload?.mdpo_evaluation && displayModel === 'latest_completed_aligned'
    ? await loadLatestCompletedAlignedModel({ root, experimentsDir })
    : !payload?.mdpo_evaluation && ['mdpo_no_rerank', 'mdpo_safe_rerank'].includes(displayModel)
      ? await loadSelectedAlignedModel({ root, experimentsDir }) : null;
  const mdpoEvaluation = await resolveMdpoLayoutEvaluation(payload?.mdpo_evaluation, sample, activeLayoutModel);
  const layoutModel = mdpoEvaluation?.model || alignedDisplayModel?.model || activeLayoutModel;
  const spatialContext = buildSpatialContext(geometry, bounds, { gridSize: layoutModel?.architecture?.spatial_grid?.grid_size || 20 });
  const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
  const generator = payload?.generator || 'annotation';
  const manualLabels = annotationsToLabels(referenceAnnotation);
  const generatedCandidates = generatedCandidatesFromCleanObj(clean.text, bounds);
  const candidates = manualLabels.length ? fixedCandidatesForLayoutModel(manualLabels, generatedCandidates, bounds, layoutModel) : generatedCandidates;
  validateFixedLabelContract(manualLabels, candidates, `${sample?.category || 'custom'}/${sample?.sample_id || sourceName}/candidates`);
  const model = layoutModel ? { encoder: layoutModel.architecture?.type === 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe' ? 'anchor_frame_heterogeneous_graph_transformer_moe_v10' : (layoutModel.architecture?.type === 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe' ? 'dgcnn_cnn_fnn_graph_transformer_moe_v8' : (layoutModel.architecture?.type === 'fixed_label_fnn_relational_graph_transformer_moe' ? 'fnn_graph_transformer_moe_v7' : (layoutModel.architecture?.type === 'fixed_label_relational_graph_transformer_moe' ? 'graph_transformer_moe_v6' : 'relational_gnn_v5'))), version: layoutModel.version, status: layoutModel.status, architecture: layoutModel.architecture } : { encoder: 'none', version: null, status: 'not_loaded' };
  const candidateStartedAt = performance.now();
  const selected = manualLabels.length
    ? { candidates, selection: { status: 'fixed_manual_label_contract', input_count: generatedCandidates.length, selected_count: candidates.length, category: sample?.category || null } }
    : layoutModel ? selectLayoutCandidates(candidates, sample?.category, layoutModel) : { candidates, selection: { status: 'not_loaded', selected_count: candidates.length } };
  const styled = layoutModel ? applyLayoutModel(selected.candidates, bounds, layoutModel, { geometry, geometryFeature: cvFeatures.geometry, visualFeature: cvFeatures.visual, spatialContext }) : { labels: candidates, expert: 'untrained', gate: {} };
  const scoredCandidates = styled.labels;
  const options = normalizeLayoutOptions(payload?.strategy);
  if (displayModel === 'v10_no_rerank' || displayModel === 'mdpo_no_rerank' || displayModel === 'latest_completed_aligned') options.preferenceRerank = false;
  if (displayModel === 'v10_historical_rerank' || displayModel === 'mdpo_safe_rerank') options.preferenceRerank = true;
  options.depthGrids = depthGrids;
  options.category = sample?.category || null;
  options.leaderLengthPrior = await getLeaderLengthPrior();
  if (manualLabels.length) { options.fixedLabels = true; options.groupPolicy = 'all'; options.viewPolicy = 'binocular'; }
  timing.candidate_ms = Number((performance.now() - candidateStartedAt).toFixed(3));
  const validationPreference = await resolveValidationPreferenceModel(payload?.preference_evaluation, sample);
  const preferenceModel = validationPreference ? validationPreference.model : options.preferenceRerank ? await getPreferenceModel() : null;
  const preferenceModelSha256 = validationPreference?.evaluation?.reward_model_sha256
    || (preferenceModel && options.preferenceRerank ? createHash('sha256').update(await fs.readFile(preferenceModelFile)).digest('hex') : null);
  const trials = [];
  let rerankPolicy = null;
  let labels;
  let metrics;
  const optimizeStartedAt = performance.now();
  if (preferenceModel && options.optimizer === 'annealing') {
    const rawTrials = [];
    for (let trial = 0; trial < 4; trial += 1) {
      const trialOptions = { ...options, seed: options.seed + trial };
      const directional = optimizeWithAdaptiveDirectionalGate(scoredCandidates, bounds, trialOptions, geometry);
      const viewSafety = evaluateViewConditionedLayout(directional.labels, bounds, depthGrids);
      Object.assign(directional.metrics, {
        weighted_leader_crossing_risk: viewSafety.weighted_leader_crossing_risk,
        worst_view_leader_crossing_risk: viewSafety.worst_view_leader_crossing_risk,
        cvar_view_leader_crossing_risk: viewSafety.cvar_view_leader_crossing_risk,
        worst_view_leader_crossing_count: viewSafety.worst_view_leader_crossing_count
      });
      rawTrials.push({ seed: trialOptions.seed, labels: directional.labels, metrics: directional.metrics, directional_policy: directional.policy, directional_alternative_accepted: directional.accepted });
    }
    const selection = selectSafetyConstrainedTrial(rawTrials, preferenceModel, predictPreference);
    if (!selection.selected) throw Object.assign(new Error('没有通过零引导线交叉硬门控的奖励重排候选'), { statusCode: 422, code: 'NO_ZERO_CROSSING_SAFE_CANDIDATE' });
    labels = selection.selected.labels;
    metrics = selection.selected.metrics;
    trials.push(...selection.assessed.map((trial) => ({
      seed: trial.seed,
      objective_score: trial.metrics.objective_score,
      preference_score: trial.preference?.score ?? null,
      safety_eligible: trial.safety.eligible,
      safety_violations: trial.safety.violations,
      safety_reference: trial.seed === selection.reference.seed,
      directional_policy: trial.directional_policy,
      directional_alternative_accepted: trial.directional_alternative_accepted
    })));
    rerankPolicy = {
      id: 'energy_safety_gate_then_aesthetic_reward_v1',
      safety_reference_seed: selection.reference.seed,
      eligible_candidates: selection.eligible_count,
      total_candidates: selection.total_count,
      selected_seed: selection.selected.seed
    };
  } else {
    const directional = optimizeWithAdaptiveDirectionalGate(scoredCandidates, bounds, options, geometry);
    labels = directional.labels;
    metrics = directional.metrics;
    rerankPolicy = { id: 'adaptive_directional_safety_gate_v1', directional_policy: directional.policy, directional_alternative_accepted: directional.accepted };
  }
  if (options.mdpoCandidate) {
    if (sample?.split !== 'train' || options.preferenceRerank || layoutModel?.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw Object.assign(new Error('MDPO perturbations are train-only, reward-reranker-off, frozen-active-v10 candidate generation'), { statusCode: 400, code: 'MDPO_PERTURBATION_SCOPE_INVALID' });
    const perturbed = perturbMdpoCandidate(labels, geometry, bounds, { mode: options.mdpoPerturbation, seed: options.seed });
    labels = perturbed.labels;
    rerankPolicy = { ...rerankPolicy, mdpo_candidate_perturbation: { mode: perturbed.mode, mode_index: options.mdpoPerturbation, bounded_anchor_local: true, qwen_generated_coordinates: false } };
  }
  // Manual coordinates are evaluation-only. They are added after candidate
  // generation/reranking so the optimizer cannot see val/test target layouts.
  metrics = evaluateLayout(labels, bounds, { ...options, geometry, manualReference: manualLabels.length ? manualLabels : undefined });
  const viewConditioned = evaluateViewConditionedLayout(labels, bounds, depthGrids, {
    viewWeights: layoutModel?.architecture?.view_loss?.weights,
    worstViewWeight: layoutModel?.architecture?.view_loss?.worst_view_weight,
    cvarViewWeight: layoutModel?.architecture?.view_loss?.cvar_weight,
    stereoWeight: layoutModel?.architecture?.view_loss?.stereo_weight,
    textClarityWeight: layoutModel?.architecture?.view_loss?.text_clarity_weight,
    leaderCrossingWeight: layoutModel?.architecture?.view_loss?.leader_crossing_weight
  });
  metrics.view_conditioned = viewConditioned;
  metrics.weighted_label_object_overlap = viewConditioned.weighted_label_object_overlap;
  metrics.worst_view_depth_occlusion = viewConditioned.worst_view_depth_occlusion;
  metrics.worst_view_penetration_v10 = viewConditioned.worst_view_penetration;
  metrics.cvar_view_occlusion = viewConditioned.cvar_view_occlusion;
  metrics.cvar_view_free_space_mismatch = viewConditioned.cvar_view_free_space_mismatch;
  metrics.worst_view_text_clarity_loss = viewConditioned.worst_view_text_clarity_loss;
  metrics.weighted_leader_crossing_risk = viewConditioned.weighted_leader_crossing_risk;
  metrics.worst_view_leader_crossing_risk = viewConditioned.worst_view_leader_crossing_risk;
  metrics.worst_view_leader_crossing_count = viewConditioned.worst_view_leader_crossing_count;
  metrics.cvar_view_leader_crossing_risk = viewConditioned.cvar_view_leader_crossing_risk;
  metrics.weighted_free_space_mismatch = viewConditioned.weighted_free_space_mismatch;
  metrics.view_conditioned_objective = viewConditioned.objective;
  if (manualLabels.length) {
    const reproductionRows = MULTI_VIEW_NAMES.map((view) => evaluateReproductionMetrics({ labels, manualLabels, geometry, bounds, view, dbvAvailable: true }));
    const fields = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
    metrics.mdpo_unified_metrics = {
      protocol: REPRODUCTION_METRIC_PROTOCOL.id,
      views: MULTI_VIEW_NAMES,
      per_view: Object.fromEntries(MULTI_VIEW_NAMES.map((view, index) => [view, reproductionRows[index]])),
      mean: Object.fromEntries(fields.map((name) => [name, reproductionRows.reduce((sum, row) => sum + Number(row[name] || 0), 0) / reproductionRows.length])),
      worst_view_intersections: Math.max(...reproductionRows.map((row) => Number(row.intersections || 0)))
    };
  }
  const spatialMetrics = computeSpatialStyleMetrics(labels, bounds, spatialContext);
  for (const [key, value] of Object.entries(spatialMetrics)) metrics[`spatial_${key}`] = value;
  timing.optimize_ms = Number((performance.now() - optimizeStartedAt).toFixed(3));
  const labelContract = validateFixedLabelContract(manualLabels, labels, `${sample?.category || 'custom'}/${sample?.sample_id || sourceName}/output`);
  const preference = predictPreference(preferenceModel, metrics);
  if (preference) metrics.preference_score = preference.score;
  const metricStartedAt = performance.now();
  const manualMetrics = manualLabels.length ? evaluateLayout(manualLabels, bounds, { ...options, manualReference: manualLabels }) : null;
  const comparison = manualMetrics ? compareToManual(metrics, manualMetrics) : null;
  timing.metrics_ms = Number((performance.now() - metricStartedAt).toFixed(3));
  timing.total_ms = Number((performance.now() - startedAt).toFixed(3));
  timing.labels_per_second = Number((labels.length / Math.max(timing.total_ms / 1000, 1e-6)).toFixed(3));
  const preferenceInfo = validationPreference
    ? { status: preference?.model_status || validationPreference.evaluation.kind, feature_dim: preference?.feature_dim ?? null, model_sha256: preferenceModelSha256, rerank_trials: trials.length, rerank_policy: rerankPolicy, evaluation: validationPreference.evaluation }
    : preference ? { status: preference.model_status, feature_dim: preference.feature_dim, model_sha256: preferenceModelSha256, rerank_trials: trials.length, rerank_policy: rerankPolicy } : { status: 'not_loaded', model_sha256: null, rerank_trials: 0 };
  const presentationEvaluation = alignedDisplayModel ? { kind: displayModel === 'latest_completed_aligned' ? 'latest_completed_safety_aligned_candidate' : 'selected_safety_aligned_candidate', candidate_id: alignedDisplayModel.id,
    candidate_file: alignedDisplayModel.file, candidate_sha256: alignedDisplayModel.sha256, completed_at: alignedDisplayModel.completed_at,
    best_epoch: alignedDisplayModel.best_epoch, hyperparameters: alignedDisplayModel.hyperparameters,
    deployment_status: alignedDisplayModel.deployment_status, formally_deployed: false } : null;
  const presentationModel = displayModel;
  const result = { version: 'generated_layout_v4_display_selection', generated_at: new Date().toISOString(), sample, source_name: sourceName, model, label_contract: labelContract, layout_model: layoutModel ? { version: layoutModel.version, status: layoutModel.status, expert: styled.expert, gate: styled.gate, selection: selected.selection, train_mse: layoutModel.training?.train_mse, val_mse: layoutModel.training?.val_mse, evaluation: mdpoEvaluation?.evaluation || presentationEvaluation } : { status: 'not_loaded', expert: 'untrained', selection: selected.selection }, preference_model: preferenceInfo, geometry: { clean_vertices: clean.vertexCount, clean_faces: clean.faceCount }, strategy: { generator, source: 'clean_obj_geometry', presentation_model: presentationModel, manual_annotation_used_for: manualLabels.length ? 'fixed_label_contract_and_supervised_reference' : 'none', five_view_energy: true, ...options }, metrics, manual_reference: manualMetrics ? { metrics: manualMetrics, labels: manualLabels } : null, manual_comparison: comparison, selection_trials: trials, timing, clean_obj: clean.text, labels };
  const validationReceipt = issueValidationGenerationReceipt(result);
  return validationReceipt ? { ...result, validation_generation_receipt: validationReceipt } : result;
}

function runPreferenceTraining(options = {}) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const source = ['all', 'llm', 'human'].includes(options.source) ? options.source : 'all';
    const args = [path.join(root, 'scripts', 'train-preference-model.mjs'), '--input', preferenceLog, '--output', preferenceCandidateFile, '--epochs', String(options.epochs || 80), '--learningRate', String(options.learningRate || 0.01), '--seed', String(options.seed || 17), '--source', source];
    if (options.runId) args.push('--runId', String(options.runId));
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

function runPreferenceValidationGate(options = {}) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const args = [
      path.join(root, 'scripts', 'select-preference-model.mjs'),
      '--candidate', preferenceCandidateFile,
      '--active', preferenceModelFile,
      '--manifest', path.join(experimentsDir, 'dataset_manifest.json'),
      '--layoutModel', layoutModelFile,
      '--output', preferenceSelectionFile,
      '--testOutput', preferenceTestReportFile,
      '--iterations', String(options.gateIterations || 180),
      '--limit', String(options.gateLimit || 0)
    ];
    if (options.skipTest) args.push('--skipTest', 'true');
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

function runPreferenceTestEvaluation(options = {}) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const args = [path.join(root, 'scripts', 'evaluate-preference-test.mjs'), '--iterations', String(options.iterations || 180), '--limit', String(options.limit || 0)];
    if (options.modelFile) args.push('--model', options.modelFile, '--runId', options.runId, '--round', String(options.round), '--selectionPolicy', options.selectionPolicy);
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, env: { ...process.env, EXPERIMENTS_DIR: experimentsDir } });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

function runLayoutTraining(options = {}) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const output = path.join(experimentsDir, 'layout_model_v10_anchor_frame_candidate.json');
    const report = path.join(experimentsDir, 'layout_training_v10_anchor_frame_report.json');
    const args = [path.join(root, 'scripts', 'train-3d-human-style-layout-model.mjs'), '--architecture', 'v10', '--epochs', String(options.epochs || 80), '--learningRate', String(options.learningRate || 0.002), '--styleWeight', String(options.styleWeight ?? 0.2), '--directionWeight', String(options.directionWeight ?? 1.25), '--viewWeight', String(options.viewWeight ?? 0.25), '--worstViewWeight', String(options.worstViewWeight ?? 2), '--cvarViewWeight', String(options.cvarViewWeight ?? 1), '--stereoWeight', String(options.stereoWeight ?? 1), '--textClarityWeight', String(options.textClarityWeight ?? 3), '--hiddenDim', String(options.hiddenDim || 64), '--preGnnFnnLayers', String(options.preGnnFnnLayers || 1), '--messageLayers', String(options.messageLayers || 2), '--transformerLayers', String(options.transformerLayers || 1), '--expertCount', String(options.expertCount || 4), '--seed', String(options.seed || 17), '--output', output, '--report', report];
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

function runScript(scriptName, args = []) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const child = spawn(process.execPath, [path.join(root, 'scripts', scriptName), ...args], { cwd: root, windowsHide: true });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

function runLayoutValidationGate() {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    const child = spawn(process.execPath, [path.join(root, 'scripts', 'select-v9-3d-inference.mjs'), '--model', path.join(experimentsDir, 'layout_model_v10_anchor_frame_candidate.json'), '--previous', layoutModelFile, '--output', path.join(experimentsDir, 'v10_anchor_frame_inference_selection.json')], { cwd: root, windowsHide: true });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', (error) => resolve({ ok: false, status: 500, error: error.message, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
    child.on('close', (code) => resolve({ ok: code === 0, status: code === 0 ? 200 : 422, code, stdout, stderr, elapsed_ms: Number((performance.now() - startedAt).toFixed(3)) }));
  });
}

async function activateMultiviewLayoutModel() {
  const candidateFile = path.join(experimentsDir, 'layout_model_v8_cv_clean_candidate.json');
  const activated = await activatePreservingPrevious({
    candidateFile,
    activeFile: layoutModelFile,
    historyFile: path.join(experimentsDir, 'layout_model_activation_history.jsonl'),
    validateCandidate(candidate) {
      const provenanceSafe = candidate?.version === 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe' && candidate?.architecture?.input_provenance === 'manual_contract_clean_obj_and_unlabeled_render_without_adjusted_center_or_box_size' && candidate?.architecture?.visual_input_source === 'unlabeled_five_view_depth_rasters_rendered_from_clean_obj';
      const projectionLossValid = candidate?.architecture?.supervised_loss === 'weighted_3d_parameter_mse_plus_five_view_projected_center_and_box_mse';
      const graphValid = candidate?.architecture?.type === 'fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe' && candidate?.architecture?.node_input_dim === 51 && candidate?.architecture?.obj_surface_points === 1024 && candidate?.architecture?.dgcnn_edgeconv_layers >= 2 && candidate?.architecture?.geometry_feature_dim === 64 && candidate?.architecture?.visual_feature_dim === 32 && candidate?.architecture?.fused_feature_dim === 160 && candidate?.network?.fusion?.weights?.[0]?.length === 160 && candidate?.architecture?.edge_input_dim === 13 && candidate?.architecture?.pre_gnn_fnn_layers === 1 && candidate?.network?.pre_gnn_fnn_layers?.length === 1 && candidate?.architecture?.message_passing_layers === 2 && candidate?.network?.message_layers?.length === 2;
      const transformerValid = candidate?.architecture?.transformer_layers >= 1 && candidate?.network?.transformer_layers?.length === candidate.architecture.transformer_layers && candidate?.training?.parameter_updates?.transformer?.changed_parameters > 0;
      const moeValid = candidate?.architecture?.moe_expert_count === 4 && candidate?.network?.moe?.experts?.length === 4 && candidate?.network?.moe?.router?.weights?.length === 4 && candidate?.training?.parameter_updates?.moe_router?.changed_parameters > 0;
      const fnnValid = candidate?.training?.parameter_updates?.pre_gnn_fnn?.changed_parameters > 0 && candidate?.training?.functional_evidence?.pre_gnn_fnn_ablation?.max_abs_output_change > 1e-8;
      const cvFusionValid = candidate?.training?.parameter_updates?.feature_fusion?.changed_parameters > 0 && candidate?.training?.functional_evidence?.cv_fusion_ablation?.max_abs_output_change > 1e-8;
      if (!provenanceSafe || !projectionLossValid || !graphValid || !transformerValid || !moeValid || !fnnValid || !cvFusionValid) throw new Error('新 v8 DGCNN-CNN-FNN 图模型结构、训练证据或输入来源校验失败，未替换活动模型');
      if (candidate?.validation_gate?.status !== 'accepted') throw new Error('候选模型尚未通过验证集五视角能量门控，已保留旧活动模型');
    },
    metadata: { gate: 'five_view_validation_energy', fixed_label_contract: true, architecture: 'dgcnn_cnn_fnn_relational_graph_transformer_moe_v8', visual_input_source: 'clean_obj_unlabeled_render' }
  });
  return activated.candidate;
}

const scoreSchema = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'score-schema.json'), 'utf8'));

async function scoreWithLocalQwen(payload) {
  const model = scoringConfig.model;
  if (!model) throw Object.assign(new Error('未配置本地 Ollama 视觉模型'), { statusCode: 503, code: 'OLLAMA_MODEL_MISSING' });
  const instruction = `你是 3D 标签布局的美学评审器。标签集合已经由人工标注固定，不允许因为重复而建议删除、合并或改名。系统的基础布局网络和确定性五视角能量函数负责重叠、越界、穿模、遮挡、文字清晰度、引导线交叉与双目稳定；你的偏好结论主要评价安全候选之间谁更美观，不要用安全项重复主导总体偏好。请综合原始模型和生成布局的主、右、左、俯、仰五个视角，以及请求中提供的确定性几何指标和人工调整后参考风格，按 1-5 分评分，5 分最好。\n\n安全诊断维度（保留用于审计和退化检查，但不进入偏好胜者加权）：\n- text_clarity: 实际文字像素高度、容纳度、裁切和五视角清晰度\n- coverage: 固定标签是否都可辨识\n- label_label_occlusion: 标签之间是否重叠遮挡\n- object_occlusion: 标签是否遮挡物体关键区域；同时注意物体挡住标签\n- object_penetration: 标签是否与物体表面发生穿插；优先参考 object_penetration_ratio 几何指标\n- leader_line_clarity: 引导线交叉、长度及标签与锚点对应是否清楚\n- multiview_consistency: 五视角表现是否稳定，不能只按主视角判断\n- binocular_consistency: 左右视角是否稳定可读\n- size_consistency: 标签尺寸是否适合文字和当前视距\n\n美学偏好维度（用于训练奖励模型）：\n- manual_style_similarity: 与人工调整后布局的视觉语言、径向分布与疏密节奏是否接近；只占一部分，不要求机械复制坐标\n- spatial_balance: 标签在物体四周的视觉重心和留白是否平衡\n- visual_hierarchy: 重要部件标签是否形成清楚的主次、层级和浏览顺序\n- composition_harmony: 标签、引导线、物体轮廓和负空间是否构成协调统一的整体\n- overall: 只表示整体美感与专业观感，不要把上述安全诊断项再次平均进 overall\n穿模不能只靠视觉猜测；请求内几何指标与图片冲突时，在理由中说明。只输出符合 JSON Schema 的结果，并给出可执行的美学布局改进建议。`;
  const format = 'ollama_chat';
  const calibratedInstruction = instruction + '\n\n审美连续刻度校准：明显拥挤或失衡约 2-3；可用但略有局部失衡约 3.5-4.0；美观但仍有可见优化空间约 4.1-4.7；几乎没有可见改进空间时才给 5.0。安全候选并不自动获得审美满分。主视角优先，并用其余视角核对留白、视觉重心、层级和引导线节奏。审美分数只根据实际图像和人工风格参考；安全数值仅用于安全诊断，不能用几何损失、seed 或数值差异制造偏好。确实看不出审美差别时保持平局。理由必须指向可见的布局细节。'
    + (['mdpo_train_candidate', 'mdpo_val11_checkpoint', 'mdpo_test11_checkpoint'].includes(payload?.type) ? '\n\nMDPO 七维离线教师协议 v1：分别独立评价 overall、composition_harmony、visual_hierarchy、spatial_balance、manual_style_similarity、text_clarity、leader_line_clarity；文字清晰度和引导线清晰度必须根据六张实际图像分别判断。仅观察视觉布局，不输出或猜测三维坐标；看不出差异时给相同分数。' : '');
  const visuals = payload?.visuals && typeof payload.visuals === 'object' ? payload.visuals : {};
  const payloadForText = {
    type: payload?.type,
    sample: payload?.sample,
    attached_images: VISION_KEYS,
    label_texts: Array.isArray(payload?.labels) ? payload.labels.map((label) => label.text) : [],
    manual_style_reference: payload?.type === 'mdpo_train_candidate' ? null : payload?.manual_comparison || null
  };
  if (payload?.sample?.category && payload?.sample?.sample_id && Array.isArray(payload.labels)) {
    const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
    const sample = manifest.samples.find(item=>item.category===payload.sample.category && item.sample_id===String(payload.sample.sample_id));
    if (!sample) throw Object.assign(new Error('评分样本不在数据集 manifest 中'),{statusCode:400,code:'SCORING_SAMPLE_INVALID'});
    const annotation = JSON.parse(await fs.readFile(path.join(root,sample.target.annotation_json),'utf8'));
    const manual = annotationsToLabels(annotation);
    try { validateFixedLabelContract(manual,payload.labels,'scoring/fixed-labels'); }
    catch(error) { throw Object.assign(error,{statusCode:400,code:'SCORING_LABEL_CONTRACT_INVALID'}); }
    const clean = cleanObj(await fs.readFile(path.join(root,sample.input.source_obj),'utf8'));
    const bounds = boundsFromObj(clean.text), geometry = parseObjTriangles(clean.text);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map(view=>[view,buildDepthGrid(geometry,bounds,view)]));
    const geometryMetrics = evaluateLayout(payload.labels,bounds,{viewPolicy:'binocular',depthGrids,geometry,manualReference:manual});
    if (payload?.type === 'mdpo_train_candidate') {
      const measured = evaluateViewConditionedLayout(payload.labels, bounds, depthGrids);
      const comparisons = {
        leader_crossings: geometryMetrics.leader_crossings,
        mesh_surface_intersection_ratio: geometryMetrics.mesh_surface_intersection_ratio,
        object_penetration_ratio: geometryMetrics.object_penetration_ratio,
        worst_view_leader_crossing_count: measured.worst_view_leader_crossing_count,
        worst_view_depth_occlusion: measured.worst_view_depth_occlusion,
        worst_view_penetration_v10: measured.worst_view_penetration
      };
      const mismatches = Object.entries(comparisons).filter(([name, actual]) =>
        !Number.isFinite(actual) || !Number.isFinite(payload.metrics?.[name]) || Math.abs(actual - payload.metrics[name]) > 1e-4);
      if (mismatches.length) throw Object.assign(new Error(`MDPO candidate measured geometry differs from client safety metrics: ${mismatches.map(([name]) => name).join(',')}`), { statusCode: 400, code: 'MDPO_GEOMETRY_MISMATCH' });
      if (geometryMetrics.leader_crossings > 0 || measured.worst_view_leader_crossing_count > 0
          || geometryMetrics.mesh_surface_intersection_ratio > 0 || geometryMetrics.object_penetration_ratio > 0)
        throw Object.assign(new Error('Measured MDPO geometry is unsafe; Qwen scoring forbidden'), { statusCode: 400, code: 'MDPO_UNSAFE_MEASURED_GEOMETRY' });
    }
    payloadForText.geometry_safety_evidence = Object.fromEntries([
      'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio',
      'mesh_surface_intersection_ratio', 'leader_crossings', 'text_clarity',
      'text_clipping_ratio', 'viewport_overflow_ratio'
    ].map((key) => [key, geometryMetrics[key] ?? null]));
    payloadForText.authoritative_manual_reference = {
      source: sample.target.annotation_json,
      version: annotation.version,
      layout_type: annotation.layout_type,
      label_count: manual.length
    };
  }
  let requestBody;
  try { requestBody = ollamaScoringRequest({ model, prompt: JSON.stringify(payloadForText), instruction: calibratedInstruction, visuals, schema: scoreSchema }); }
  catch (error) { throw Object.assign(error, { statusCode: 400, code: 'VISUALS_MISSING' }); }
  const response = await fetch(scoringConfig.apiUrl, { method: 'POST', redirect: 'error', signal: AbortSignal.timeout(300000), headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody) });
  const raw = await response.text();
  let data;
  try { data = JSON.parse(raw); } catch { data = { raw }; }
  if (!response.ok) {
    const remoteMessage = String(data?.error?.message || data?.error || data?.message || '').trim();
    throw Object.assign(new Error(remoteMessage || `本地 Ollama 视觉评分失败 (${response.status})`), { statusCode: response.status, code: 'OLLAMA_SCORING_ERROR' });
  }
  let parsed;
  try { parsed = parseOllamaScoringResponse(data, scoreSchema); }
  catch (error) {
    await fs.writeFile(path.join(experimentsDir, 'ollama_invalid_response_latest.json'), JSON.stringify({ received_at: new Date().toISOString(), model, response: data, error: error.message }, null, 2) + '\n', 'utf8').catch(() => {});
    throw Object.assign(error, { statusCode: 502, code: 'SCORING_RESPONSE_INVALID' });
  }
  return { model, format, ...parsed };
}

async function persistVisualValidationScore(payload, scored) {
  if (payload?.type !== 'llm_validation_checkpoint') return null;
  const receipt = verifiedValidationGeneration(payload);
  if (!receipt) throw Object.assign(new Error('val 视觉评分必须来自本服务刚刚用指定冻结检查点生成且未被修改的布局'), { statusCode: 409, code: 'LLM_VISUAL_VAL_GENERATION_RECEIPT_INVALID' });
  const runId = validatedRunId(payload.run_id);
  const round = Number(payload.round);
  const scores = scored?.result?.scores;
  if (!scores || LLM_SCORE_NAMES.some((name) => !Number.isFinite(Number(scores[name])) || Number(scores[name]) < 1 || Number(scores[name]) > 5)) throw Object.assign(new Error('val 视觉评分缺少完整的十四维 1-5 分结果'), { statusCode: 502, code: 'LLM_VISUAL_VAL_SCORE_INVALID' });
  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  const sample = manifest.samples.find((item) => item.category === payload.sample?.category && item.sample_id === String(payload.sample?.sample_id));
  if (!sample || sample.split !== 'val') throw Object.assign(new Error('视觉验证记录只允许 manifest 中的 val 样本'), { statusCode: 400, code: 'LLM_VISUAL_VAL_SAMPLE_INVALID' });
  const evaluation = payload.preference_model?.evaluation;
  if (!evaluation || evaluation.run_id !== runId || Number(evaluation.round) !== round) throw Object.assign(new Error('评分载荷与冻结检查点来源不一致'), { statusCode: 409, code: 'LLM_VISUAL_VAL_PROVENANCE_MISMATCH' });
  const record = {
    version: 'llm_visual_validation_sample_v1',
    scored_at: new Date().toISOString(),
    run_id: runId,
    round,
    split: 'val',
    sample: { category: sample.category, sample_id: sample.sample_id },
    scorer: { model: scored.model, format: scored.format, response_id: scored.response_id || null },
    checkpoint: evaluation,
    generation_strategy: validationGenerationReceipts.get(receipt.id)?.generation_strategy || null,
    scores: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, Number(scores[name])])),
    composite_score: Number(preferenceComposite(scores).toFixed(6)),
    rationale: scored.result.rationale || '',
    risks: scored.result.risks || [],
    suggested_changes: scored.result.suggested_changes || [],
    geometry_proxy: payload.metrics || null,
    security: { images_persisted: false, api_key_persisted: false, train_preference_created: false }
  };
  const directory = path.join(experimentsDir, 'llm_visual_validation', runId, `round_${round}`);
  await fs.mkdir(directory, { recursive: true });
  const file = path.join(directory, `${encodeURIComponent(`${sample.category}__${sample.sample_id}`)}.json`);
  await fs.writeFile(file, JSON.stringify(record, null, 2) + '\n', 'utf8');
  validationGenerationReceipts.delete(receipt.id);
  return { file: path.relative(root, file).split(path.sep).join('/'), record };
}

async function persistMdpoVisualValidationScore(payload, scored) {
  if (payload?.type !== 'mdpo_val11_checkpoint') return null;
  const receipt = verifiedMdpoValidationGeneration(payload);
  if (!receipt || !verifyVisualValidationImages(payload)) throw Object.assign(new Error('MDPO val11 score must match a just-rendered frozen model layout and six unchanged images'), { statusCode: 409, code: 'MDPO_VAL11_RECEIPT_INVALID' });
  const evaluation = payload.layout_model?.evaluation;
  const scores = scored?.result?.scores;
  const group = mdpoVal11Group(payload.group);
  if (!evaluation || evaluation.candidate_id !== payload.candidate_id || evaluation.group !== payload.group || evaluation.role !== payload.role
      || group.model_role !== payload.role || LLM_SCORE_NAMES.some((name) => !Number.isFinite(Number(scores?.[name])))) throw Object.assign(new Error('MDPO val11 model/group/Qwen score provenance mismatch'), { statusCode: 409 });
  const unified = payload.metrics?.mdpo_unified_metrics;
  if (unified?.protocol !== REPRODUCTION_METRIC_PROTOCOL.id || JSON.stringify(unified.views) !== JSON.stringify(MULTI_VIEW_NAMES)) throw Object.assign(new Error('MDPO val11 unified five-view metrics missing'), { statusCode: 409 });
  const record = {
    version: 'v10_mdpo_val11_sample_v1', scored_at: new Date().toISOString(), split: 'val', group: payload.group, role: payload.role,
    candidate_id: payload.candidate_id, sample: { category: payload.sample.category, sample_id: String(payload.sample.sample_id) },
    scorer: { model: scored.model, response_id: scored.response_id, prompt_version: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1` },
    evaluation, views: VISION_KEYS, metric_protocol: REPRODUCTION_METRIC_PROTOCOL.id, view_sha256: hashMdpoViews(payload.visuals),
    generation_strategy: validationGenerationReceipts.get(receipt.id)?.generation_strategy || null,
    preference_model: payload.preference_model || null,
    scores: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, Number(scores[name])])),
    metrics: { ...unified.mean, worst_view_intersections: unified.worst_view_intersections,
      object_occlusion: Number(payload.metrics.label_object_occlusion_ratio), penetration: Number(payload.metrics.object_penetration_ratio),
      mesh_surface_intersection: Number(payload.metrics.mesh_surface_intersection_ratio), worst_view_overflow: Number(payload.metrics.multi_view_worst_overflow) },
    geometry: { leader_crossings: payload.metrics.leader_crossings, worst_view_leader_crossing_count: payload.metrics.worst_view_leader_crossing_count,
      worst_view_depth_occlusion: payload.metrics.worst_view_depth_occlusion, worst_view_penetration: payload.metrics.worst_view_penetration_v10,
      cvar_view_occlusion: payload.metrics.cvar_view_occlusion },
    security: { images_persisted: false, test_used: false, train_preference_created: false, qwen_inference_input: false }
  };
  if (Object.values(record.metrics).some((value) => !Number.isFinite(value))) throw Object.assign(new Error('MDPO val11 record has incomplete deterministic metrics'), { statusCode: 409 });
  if (record.generation_strategy?.preferenceRerank !== group.preference_rerank) throw Object.assign(new Error('MDPO val11 group/reranker mismatch'), { statusCode: 409 });
  if (group.preference_rerank && !/^[a-f0-9]{64}$/i.test(record.preference_model?.model_sha256 || '')) throw Object.assign(new Error('MDPO reranked val11 group missing historical reward-model hash'), { statusCode: 409 });
  if (!group.preference_rerank && record.preference_model?.model_sha256 !== null) throw Object.assign(new Error('MDPO no-rerank val11 group unexpectedly loaded reward model'), { statusCode: 409 });
  const directory = path.join(mdpoDir, 'val11_visual', payload.candidate_id, payload.group);
  await fs.mkdir(directory, { recursive: true });
  const file = path.join(directory, `${encodeURIComponent(`${record.sample.category}__${record.sample.sample_id}`)}.json`);
  await fs.writeFile(file, JSON.stringify(record, null, 2) + '\n', { flag: 'wx' });
  validationGenerationReceipts.delete(receipt.id);
  return { file: path.relative(root, file).split(path.sep).join('/'), record };
}

async function persistMdpoTest11Score(payload, scored) {
  if (payload?.type !== 'mdpo_test11_checkpoint') return null;
  const receipt = verifiedMdpoTest11Generation(payload);
  if (!receipt || !verifyVisualValidationImages(payload)) throw Object.assign(new Error('MDPO test11 score must match a just-rendered locked layout and six unchanged images'), { statusCode: 409, code: 'MDPO_TEST11_RECEIPT_INVALID' });
  const context = await loadMdpoTest11Context();
  const evaluation = payload.layout_model?.evaluation, scores = scored?.result?.scores, group = mdpoVal11Group(payload.group);
  if (!evaluation || payload.test_run_id !== context.testRunId || payload.lock_sha256 !== context.lockSha256
      || evaluation.phase !== 'test11' || evaluation.test_run_id !== context.testRunId || evaluation.lock_sha256 !== context.lockSha256
      || evaluation.group !== payload.group || evaluation.role !== payload.role || group.model_role !== payload.role
      || LLM_SCORE_NAMES.some((name) => !Number.isFinite(Number(scores?.[name])))) throw Object.assign(new Error('MDPO test11 locked model/group/Qwen score provenance mismatch'), { statusCode: 409 });
  const unified = payload.metrics?.mdpo_unified_metrics;
  if (unified?.protocol !== REPRODUCTION_METRIC_PROTOCOL.id || JSON.stringify(unified.views) !== JSON.stringify(MULTI_VIEW_NAMES)) throw Object.assign(new Error('MDPO test11 unified five-view metrics missing'), { statusCode: 409 });
  const record = {
    version: 'v10_mdpo_test11_sample_v1', scored_at: new Date().toISOString(), split: 'test', group: payload.group, role: payload.role,
    test_run_id: context.testRunId, lock_sha256: context.lockSha256, sample: { category: payload.sample.category, sample_id: String(payload.sample.sample_id) },
    scorer: { model: scored.model, response_id: scored.response_id, prompt_version: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1` },
    evaluation, views: VISION_KEYS, metric_protocol: REPRODUCTION_METRIC_PROTOCOL.id, view_sha256: hashMdpoViews(payload.visuals),
    generation_strategy: validationGenerationReceipts.get(receipt.id)?.generation_strategy || null,
    preference_model: payload.preference_model || null,
    scores: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, Number(scores[name])])),
    metrics: { ...unified.mean, worst_view_intersections: unified.worst_view_intersections,
      object_occlusion: Number(payload.metrics.label_object_occlusion_ratio), penetration: Number(payload.metrics.object_penetration_ratio),
      mesh_surface_intersection: Number(payload.metrics.mesh_surface_intersection_ratio), worst_view_overflow: Number(payload.metrics.multi_view_worst_overflow) },
    geometry: { leader_crossings: payload.metrics.leader_crossings, worst_view_leader_crossing_count: payload.metrics.worst_view_leader_crossing_count,
      worst_view_depth_occlusion: payload.metrics.worst_view_depth_occlusion, worst_view_penetration: payload.metrics.worst_view_penetration_v10,
      cvar_view_occlusion: payload.metrics.cvar_view_occlusion },
    security: { test_used: true, test_used_for_training: false, test_used_for_selection: false, train_preference_created: false, qwen_inference_input: false }
  };
  if (Object.values(record.metrics).some((value) => !Number.isFinite(value))) throw Object.assign(new Error('MDPO test11 record has incomplete deterministic metrics'), { statusCode: 409 });
  if (record.generation_strategy?.preferenceRerank !== group.preference_rerank) throw Object.assign(new Error('MDPO test11 group/reranker mismatch'), { statusCode: 409 });
  if (group.preference_rerank && record.preference_model?.model_sha256 !== context.preferenceSha256) throw Object.assign(new Error('MDPO test11 reranked group historical reward-model hash mismatch'), { statusCode: 409 });
  if (!group.preference_rerank && record.preference_model?.model_sha256 !== null) throw Object.assign(new Error('MDPO test11 no-rerank group unexpectedly loaded reward model'), { statusCode: 409 });
  const directory = path.join(mdpoDir, 'test11_visual', context.testRunId, payload.group);
  await fs.mkdir(directory, { recursive: true });
  const file = path.join(directory, `${encodeURIComponent(`${record.sample.category}__${record.sample.sample_id}`)}.json`);
  await fs.writeFile(file, JSON.stringify(record, null, 2) + '\n', { flag: 'wx' });
  validationGenerationReceipts.delete(receipt.id);
  return { file: path.relative(root, file).split(path.sep).join('/'), record };
}

async function readMdpoTest11Records(context) {
  const groups = {};
  for (const groupName of Object.keys(MDPO_VAL11_GROUPS)) {
    const directory = path.join(mdpoDir, 'test11_visual', context.testRunId, groupName);
    const files = await fs.readdir(directory).catch((error) => { if (error?.code === 'ENOENT') return []; throw error; });
    const rows = [];
    for (const name of files.filter((file) => file.endsWith('.json'))) rows.push(JSON.parse(await fs.readFile(path.join(directory, name), 'utf8')));
    const bySample = new Map(rows.map((row) => [`${row.sample?.category}/${row.sample?.sample_id}`, row]));
    if (bySample.size !== rows.length || [...bySample.keys()].some((sample) => !context.cohort.includes(sample))) throw new Error(`Duplicate or unexpected MDPO test11 records in ${groupName}`);
    groups[groupName] = context.cohort.filter((sample) => bySample.has(sample)).map((sample) => bySample.get(sample));
  }
  return groups;
}

async function mdpoTest11Progress() {
  const context = await loadMdpoTest11Context();
  const groups = await readMdpoTest11Records(context), completed = [];
  for (const [groupName, rows] of Object.entries(groups)) for (const row of rows) completed.push({ key: `${groupName}|${row.sample.category}/${row.sample.sample_id}`,
    group: groupName, role: row.role, sample: `${row.sample.category}/${row.sample.sample_id}`,
    file: path.relative(root, path.join(mdpoDir, 'test11_visual', context.testRunId, groupName, `${encodeURIComponent(`${row.sample.category}__${row.sample.sample_id}`)}.json`)).split(path.sep).join('/') });
  if (completed.length) validateMdpoTest11Records({ groups, cohort: context.cohort,
    runId: context.testRunId, lockSha256: context.lockSha256, baselineSha256: context.baselineSha256, activeSha256: context.activeSha256,
    preferenceSha256: context.preferenceSha256, scorerModel: context.lock.scorer.model, promptVersion: context.lock.scorer.prompt_version, metricProtocol: REPRODUCTION_METRIC_PROTOCOL.id, allowIncomplete: true });
  return { test_run_id: context.testRunId, lock_sha256: context.lockSha256, seed: context.lock.seed, cohort: context.cohort,
    expected_records: 44, completed_records: completed.length, completed };
}

async function finalizeMdpoTest11() {
  const context = await loadMdpoTest11Context();
  const groups = await readMdpoTest11Records(context);
  const provenance = validateMdpoTest11Records({ groups, cohort: context.cohort, runId: context.testRunId, lockSha256: context.lockSha256,
    baselineSha256: context.baselineSha256, activeSha256: context.activeSha256, preferenceSha256: context.preferenceSha256,
    scorerModel: context.lock.scorer.model, promptVersion: context.lock.scorer.prompt_version, metricProtocol: REPRODUCTION_METRIC_PROTOCOL.id });
  const summaries = {};
  for (const [groupName, rows] of Object.entries(groups)) {
    const spec = MDPO_VAL11_GROUPS[groupName], average = (getter) => rows.reduce((sum, row) => sum + Number(getter(row)), 0) / rows.length;
    const score_means = Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, average((row) => row.scores[name])]));
    summaries[groupName] = { group: groupName, label: spec.label, model_role: spec.model_role, preference_rerank: spec.preference_rerank,
      preference_model_sha256: spec.preference_rerank ? context.preferenceSha256 : null, sample_count: 11, cohort: context.cohort,
      scorer_model: context.lock.scorer.model, prompt_version: context.lock.scorer.prompt_version, views: VISION_KEYS, metric_protocol: REPRODUCTION_METRIC_PROTOCOL.id,
      score_means, aesthetic_composite: preferenceComposite(score_means), metrics: {
        ...Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'quality_score', 'object_occlusion', 'penetration', 'mesh_surface_intersection'].map((name) => [name, average((row) => row.metrics[name])])),
        intersections: rows.reduce((sum, row) => sum + Number(row.metrics.intersections), 0), worst_view_intersections: Math.max(...rows.map((row) => Number(row.metrics.worst_view_intersections))),
        worst_view_overflow: Math.max(...rows.map((row) => Number(row.metrics.worst_view_overflow))), text_clarity: score_means.text_clarity, leader_line_clarity: score_means.leader_line_clarity
      }, sample_scores: rows.map((row) => ({ sample: row.sample, scores: row.scores, metrics: row.metrics, response_id: row.scorer.response_id })) };
  }
  const baseline = summaries.v10_no_rerank, candidate = summaries.mdpo_no_rerank;
  const unifiedRows = Object.values(summaries).map((row) => ({ group: row.group, label: row.label, sample_count: row.sample_count,
    ...Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, row.score_means[name]])), aesthetic_composite: row.aesthetic_composite,
    ...Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score',
      'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow'].map((name) => [name, row.metrics[name]])) }));
  const unified = { version: 'v10_mdpo_test11_unified_metrics_v1', generated_at: new Date().toISOString(), split: 'test', sample_count: 11,
    test_run_id: context.testRunId, lock_sha256: context.lockSha256, metric_protocol: REPRODUCTION_METRIC_PROTOCOL.id, rows: unifiedRows };
  const csvFields = Object.keys(unifiedRows[0]);
  const csvCell = (value) => { const text = String(value ?? ''); return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; };
  const csv = `${csvFields.map(csvCell).join(',')}\n${unifiedRows.map((row) => csvFields.map((field) => csvCell(row[field])).join(',')).join('\n')}\n`;
  const unifiedJsonBytes = Buffer.from(JSON.stringify(unified, null, 2) + '\n'), csvBytes = Buffer.from(csv);
  const report = { version: 'v10_mdpo_test11_final_report_v1', generated_at: new Date().toISOString(), status: 'complete_locked_single_test11',
    test_run_id: context.testRunId, lock_file: 'experiments/mdpo/test11_lock.json', lock_sha256: context.lockSha256,
    models: { original_v10_sha256: context.baselineSha256, active_v10_mdpo_sha256: context.activeSha256, historical_reward_model_sha256: context.preferenceSha256 },
    provenance, groups: summaries, core_no_rerank_delta: {
      aesthetic: candidate.aesthetic_composite - baseline.aesthetic_composite,
      composition_harmony: candidate.score_means.composition_harmony - baseline.score_means.composition_harmony,
      PCK_005: candidate.metrics.PCK_005 - baseline.metrics.PCK_005, PCK_010: candidate.metrics.PCK_010 - baseline.metrics.PCK_010,
      text_clarity: candidate.score_means.text_clarity - baseline.score_means.text_clarity,
      leader_line_clarity: candidate.score_means.leader_line_clarity - baseline.score_means.leader_line_clarity
    }, artifacts: { unified_json: 'experiments/mdpo/test11_unified_metrics.json', unified_json_sha256: createHash('sha256').update(unifiedJsonBytes).digest('hex'),
      unified_csv: 'experiments/mdpo/test11_unified_metrics.csv', unified_csv_sha256: createHash('sha256').update(csvBytes).digest('hex') },
    test_used_for_training: false, test_used_for_selection: false, model_changed_after_lock: false };
  await writeImmutableFileBundle([
    { file: path.join(mdpoDir, 'test11_final_report.json'), text: JSON.stringify(report, null, 2) + '\n' },
    { file: path.join(mdpoDir, 'test11_unified_metrics.json'), bytes: unifiedJsonBytes },
    { file: path.join(mdpoDir, 'test11_unified_metrics.csv'), bytes: csvBytes }
  ]);
  return report;
}

async function mdpoVal11Progress(input) {
  const candidateId = String(input?.candidate_id || '');
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(candidateId)) throw Object.assign(new Error('Invalid MDPO val11 candidate ID'), { statusCode: 400 });
  const candidateFile = path.resolve(root, String(input?.candidate_file || '')), mdpoPrefix = path.resolve(mdpoDir) + path.sep;
  if (!candidateFile.startsWith(mdpoPrefix) || path.extname(candidateFile).toLowerCase() !== '.json') throw Object.assign(new Error('MDPO val11 candidate must remain inside experiments/mdpo'), { statusCode: 400 });
  const [candidateBytes, activeBytes, manifest] = await Promise.all([
    fs.readFile(candidateFile), fs.readFile(layoutModelFile), fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8').then(JSON.parse)
  ]);
  const candidateHash = createHash('sha256').update(candidateBytes).digest('hex'), activeHash = createHash('sha256').update(activeBytes).digest('hex');
  if (candidateHash !== String(input?.candidate_sha256 || '').toLowerCase()) throw Object.assign(new Error('MDPO val11 resume candidate SHA-256 mismatch'), { statusCode: 409 });
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate'
      || candidate.reference?.sha256 !== activeHash || candidate.architecture?.qwen_inference_input !== false) throw Object.assign(new Error('MDPO val11 resume candidate/reference contract invalid'), { statusCode: 409 });
  const expected = Array.isArray(input?.samples) ? input.samples.map((item) => `${item.category}/${String(item.sampleId ?? item.sample_id)}`) : [];
  if (expected.length !== 11 || new Set(expected).size !== 11 || expected.some((key) => !manifest.samples.some((sample) => sample.split === 'val' && `${sample.category}/${sample.sample_id}` === key))) {
    throw Object.assign(new Error('MDPO val11 resume requires the exact 11 manifest validation samples'), { statusCode: 400 });
  }
  const expectedSet = new Set(expected), completed = [];
  for (const [groupName, groupSpec] of Object.entries(MDPO_VAL11_GROUPS)) {
    const directory = path.join(mdpoDir, 'val11_visual', candidateId, groupName);
    const files = await fs.readdir(directory).catch((error) => {
      if (error?.code === 'ENOENT') return [];
      throw error;
    });
    for (const name of files.filter((file) => file.endsWith('.json'))) {
      const file = path.join(directory, name), record = JSON.parse(await fs.readFile(file, 'utf8'));
      const sample = `${record.sample?.category}/${String(record.sample?.sample_id)}`;
      const scoresValid = LLM_SCORE_NAMES.every((dimension) => Number.isFinite(record.scores?.[dimension]) && record.scores[dimension] >= 1 && record.scores[dimension] <= 5);
      const metricsValid = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score', 'worst_view_intersections', 'object_occlusion', 'penetration', 'mesh_surface_intersection', 'worst_view_overflow']
        .every((metric) => Number.isFinite(record.metrics?.[metric]));
      const imagesValid = VISION_KEYS.every((view) => /^[a-f0-9]{64}$/i.test(record.view_sha256?.[view] || ''));
      const evaluation = record.evaluation || {};
      const modelHashValid = groupSpec.model_role === 'candidate'
        ? evaluation.candidate_sha256 === candidateHash && evaluation.reference_model_sha256 === activeHash && evaluation.candidate_file === path.relative(root, candidateFile).split(path.sep).join('/')
        : evaluation.candidate_sha256 === activeHash && evaluation.reference_model_sha256 === activeHash && evaluation.candidate_file === null;
      const rewardHash = record.preference_model?.model_sha256 ?? null;
      if (record.version !== 'v10_mdpo_val11_sample_v1' || record.split !== 'val' || record.candidate_id !== candidateId
          || record.group !== groupName || record.role !== groupSpec.model_role || evaluation.group !== groupName || evaluation.role !== groupSpec.model_role
          || !expectedSet.has(sample) || record.scorer?.model !== scoringConfig.model || record.scorer?.prompt_version !== `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`
          || !record.scorer?.response_id || JSON.stringify(record.views) !== JSON.stringify(VISION_KEYS) || record.metric_protocol !== REPRODUCTION_METRIC_PROTOCOL.id
          || !scoresValid || !metricsValid || !imagesValid || !modelHashValid || record.security?.test_used !== false || record.security?.train_preference_created !== false
          || record.security?.qwen_inference_input !== false || record.generation_strategy?.preferenceRerank !== groupSpec.preference_rerank
          || (groupSpec.preference_rerank ? !/^[a-f0-9]{64}$/i.test(rewardHash || '') : rewardHash !== null)) {
        throw Object.assign(new Error(`Existing MDPO val11 record failed resume verification: ${path.relative(root, file)}`), { statusCode: 409 });
      }
      const key = `${groupName}|${sample}`;
      if (completed.some((item) => item.key === key)) throw Object.assign(new Error(`Duplicate MDPO val11 resume record: ${key}`), { statusCode: 409 });
      completed.push({ key, group: groupName, role: groupSpec.model_role, sample, file: path.relative(root, file).split(path.sep).join('/') });
    }
  }
  return { candidate_id: candidateId, candidate_sha256: candidateHash, reference_model_sha256: activeHash, expected_records: 44, completed_records: completed.length, completed };
}

async function finalizeMdpoVal11(input) {
  const candidateId = String(input?.candidate_id || '');
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(candidateId)) throw Object.assign(new Error('Invalid MDPO candidate ID'), { statusCode: 400 });
  const expected = (input?.samples || []).map((item) => ({ category: String(item.category), sample_id: String(item.sampleId ?? item.sample_id) }));
  if (expected.length !== 11 || new Set(expected.map((item) => `${item.category}/${item.sample_id}`)).size !== 11) throw Object.assign(new Error('MDPO gate requires exactly 11 unique val samples'), { statusCode: 400 });
  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  if (expected.some((item) => !manifest.samples.some((sample) => sample.split === 'val' && sample.category === item.category && sample.sample_id === item.sample_id))) throw Object.assign(new Error('MDPO cohort contains non-val sample'), { statusCode: 400 });
  const byGroup = {};
  const groupRows = {};
  for (const [groupName, groupSpec] of Object.entries(MDPO_VAL11_GROUPS)) {
    const rows = [];
    for (const item of expected) {
      const file = path.join(mdpoDir, 'val11_visual', candidateId, groupName, `${encodeURIComponent(`${item.category}__${item.sample_id}`)}.json`);
      const row = JSON.parse(await fs.readFile(file, 'utf8'));
      if (row.candidate_id !== candidateId || row.group !== groupName || row.role !== groupSpec.model_role || row.scorer.model !== scoringConfig.model) throw new Error(`MDPO ${groupName} val11 provenance mismatch`);
      rows.push(row);
    }
    groupRows[groupName] = rows;
    const average = (getter) => rows.reduce((sum, row) => sum + Number(getter(row)), 0) / rows.length;
    const score_means = Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, average((row) => row.scores[name])]));
    const preferenceHashes = new Set(rows.map((row) => row.preference_model?.model_sha256 ?? null));
    if (preferenceHashes.size !== 1) throw new Error(`MDPO ${groupName} mixed reward-model provenance`);
    byGroup[groupName] = {
      group: groupName, name: groupSpec.label, label: groupSpec.label, model_role: groupSpec.model_role, preference_rerank: groupSpec.preference_rerank,
      preference_model_sha256: [...preferenceHashes][0], sample_count: 11, cohort: expected.map((item) => `${item.category}/${item.sample_id}`), scorer_model: scoringConfig.model,
      prompt_version: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`, views: VISION_KEYS, metric_protocol: REPRODUCTION_METRIC_PROTOCOL.id,
      test_used_for_selection: false, score_means,
      metrics: {
        ...Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'quality_score', 'object_occlusion', 'penetration', 'mesh_surface_intersection'].map((name) => [name, average((row) => row.metrics[name])])),
        intersections: rows.reduce((sum, row) => sum + Number(row.metrics.intersections), 0),
        worst_view_intersections: Math.max(...rows.map((row) => Number(row.metrics.worst_view_intersections))),
        worst_view_overflow: Math.max(...rows.map((row) => Number(row.metrics.worst_view_overflow))),
        text_clarity: score_means.text_clarity, leader_line_clarity: score_means.leader_line_clarity
      },
      evaluation: rows[0].evaluation, sample_scores: rows.map((row) => ({ sample: row.sample, scores: row.scores, metrics: row.metrics,
        response_id: row.scorer.response_id, view_sha256: row.view_sha256, generation_strategy: row.generation_strategy,
        evaluation: row.evaluation, preference_model_sha256: row.preference_model?.model_sha256 ?? null, security: row.security }))
    };
  }
  const fourGroupProtocol = validateMdpoFourGroupSummaries(byGroup);
  const baseline = byGroup.v10_no_rerank, candidate = byGroup.mdpo_no_rerank;
  const activeBytes = await fs.readFile(layoutModelFile), activeHash = createHash('sha256').update(activeBytes).digest('hex');
  const candidateFile = path.resolve(root, String(candidate.evaluation?.candidate_file || ''));
  const candidateBytes = await fs.readFile(candidateFile), candidateHash = createHash('sha256').update(candidateBytes).digest('hex');
  const pairedProvenance = validateMdpoVal11Records({
    baseline: groupRows.v10_no_rerank, candidate: groupRows.mdpo_no_rerank, expected: expected.map((item) => `${item.category}/${item.sample_id}`),
    candidateId, referenceSha256: activeHash, candidateSha256: candidateHash, scorerModel: scoringConfig.model,
    promptVersion: `${SCORING_PROTOCOL_ID}_mdpo_teacher_v1`, metricProtocol: REPRODUCTION_METRIC_PROTOCOL.id
  });
  const candidateModel = JSON.parse(candidateBytes.toString('utf8'));
  const alignedTraining = candidateModel.mdpo?.safety_alignment?.protocol === 'safety_priority_v3_aligned' ? candidateModel.mdpo.safety_alignment : null;
  const aligned = Boolean(alignedTraining);
  const gate = aligned ? evaluateMdpoVal11AlignedGate({ baseline, candidate, trainingAlignment: alignedTraining }) : evaluateMdpoVal11Gate({ baseline, candidate });
  if (candidateHash !== candidate.evaluation?.candidate_sha256) throw new Error('MDPO val11 candidate changed after rendered evaluation');
  const reportVersion = aligned ? 'v10_mdpo_val11_gate_report_v3' : 'v10_mdpo_val11_gate_report_v2';
  const evaluationPolicy = aligned ? 'safety_priority_v3_aligned' : 'safety_priority_v2';
  const reportDirectory = path.join(mdpoDir, aligned ? 'val11_reports_safety_priority_v3_aligned' : 'val11_reports_safety_priority_v2');
  const report = { version: reportVersion, generated_at: new Date().toISOString(), candidate_id: candidateId, evaluation_policy: evaluationPolicy,
    candidate_model: { file: path.relative(root, candidateFile).split(path.sep).join('/'), sha256: candidateHash, hyperparameters: candidateModel.mdpo?.hyperparameters || null },
    paired_provenance: pairedProvenance, four_group_protocol: fourGroupProtocol, baseline, candidate, gate, training_alignment: alignedTraining, test_not_used: true };
  const fourGroupReport = { version: aligned ? 'v10_mdpo_four_group_val11_v3' : 'v10_mdpo_four_group_val11_v2', generated_at: report.generated_at, candidate_id: candidateId, evaluation_policy: evaluationPolicy,
    candidate_model: report.candidate_model, protocol: fourGroupProtocol, groups: Object.values(byGroup), core_gate: gate, training_alignment: alignedTraining, test_not_used: true };
  await fs.mkdir(reportDirectory, { recursive: true });
  await writeImmutableJsonBundle([
    { file: path.join(reportDirectory, `${candidateId}.json`), value: report },
    { file: path.join(reportDirectory, `${candidateId}.four_group.json`), value: fourGroupReport }
  ]);
  await fs.writeFile(path.join(mdpoDir, aligned ? 'val11_gate_report_safety_priority_v3_aligned.json' : 'val11_gate_report.json'), JSON.stringify(report, null, 2) + '\n');
  await fs.writeFile(path.join(mdpoDir, aligned ? 'four_group_ablation_safety_priority_v3_aligned.json' : 'four_group_ablation.json'), JSON.stringify(fourGroupReport, null, 2) + '\n');
  return report;
}

function meanRecordMetric(rows, getter) {
  const values = rows.map(getter).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  return values.length ? Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(6)) : null;
}

async function finalizeVisualValidation(input) {
  const runId = validatedRunId(input?.runId);
  const round = Number(input?.round);
  if (!Number.isInteger(round) || round < 0 || round > 64) throw Object.assign(new Error('视觉验证汇总轮次无效'), { statusCode: 400, code: 'LLM_VISUAL_VAL_ROUND_INVALID' });
  const expected = Array.isArray(input?.samples) ? input.samples.map((item) => ({ category: String(item.category), sample_id: String(item.sampleId ?? item.sample_id) })) : [];
  if (!expected.length) throw Object.assign(new Error('必须声明本轮预计评分的 val 样本'), { statusCode: 400, code: 'LLM_VISUAL_VAL_EXPECTED_EMPTY' });
  if (new Set(expected.map((item) => `${item.category}/${item.sample_id}`)).size !== expected.length) throw Object.assign(new Error('本轮 val 视觉评分样本不得重复'), { statusCode: 400, code: 'LLM_VISUAL_VAL_EXPECTED_DUPLICATE' });
  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  for (const item of expected) if (!manifest.samples.some((sample) => sample.split === 'val' && sample.category === item.category && sample.sample_id === item.sample_id)) throw Object.assign(new Error(`预计样本 ${item.category}/${item.sample_id} 不是 val 样本`), { statusCode: 400, code: 'LLM_VISUAL_VAL_EXPECTED_INVALID' });
  const directory = path.join(experimentsDir, 'llm_visual_validation', runId, `round_${round}`);
  const rows = [];
  for (const item of expected) {
    const file = path.join(directory, `${encodeURIComponent(`${item.category}__${item.sample_id}`)}.json`);
    let row;
    try { row = JSON.parse(await fs.readFile(file, 'utf8')); }
    catch { throw Object.assign(new Error(`缺少 ${item.category}/${item.sample_id} 的第 ${round} 轮真实视觉评分`), { statusCode: 409, code: 'LLM_VISUAL_VAL_SAMPLE_MISSING' }); }
    if (row.run_id !== runId || Number(row.round) !== round || row.split !== 'val' || row.scorer?.model !== scoringConfig.model) throw Object.assign(new Error('视觉评分记录的 run_id、轮次、split 或模型不一致'), { statusCode: 409, code: 'LLM_VISUAL_VAL_RECORD_MISMATCH' });
    rows.push(row);
  }
  const score_means = Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, meanRecordMetric(rows, (row) => row.scores?.[name])]));
  const geometryFields = ['multidimensional_quality_score', 'objective_score', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'manual_style_distance', 'text_clarity'];
  const checkpoint = rows[0].checkpoint;
  if (rows.some((row) => JSON.stringify(row.checkpoint) !== JSON.stringify(checkpoint))) throw Object.assign(new Error('同轮 val 样本使用了不同的奖励检查点'), { statusCode: 409, code: 'LLM_VISUAL_VAL_CHECKPOINT_INCONSISTENT' });
  const protocolKey = (row) => JSON.stringify({ viewPolicy: row.generation_strategy?.viewPolicy, groupPolicy: row.generation_strategy?.groupPolicy, sizePolicy: row.generation_strategy?.sizePolicy, optimizer: row.generation_strategy?.optimizer, iterations: row.generation_strategy?.iterations, preferenceRerank: row.generation_strategy?.preferenceRerank });
  if (rows.some((row) => protocolKey(row) !== protocolKey(rows[0]))) throw Object.assign(new Error('同轮 val 样本没有使用同一生成与相机评价协议'), { statusCode: 409, code: 'LLM_VISUAL_VAL_PROTOCOL_INCONSISTENT' });
  const summary = {
    round,
    status: 'evaluated',
    sample_count: rows.length,
    scorer_model: scoringConfig.model,
    checkpoint,
    score_means,
    composite_score: meanRecordMetric(rows, (row) => row.composite_score),
    geometry_proxy_means: Object.fromEntries(geometryFields.map((name) => [name, meanRecordMetric(rows, (row) => row.geometry_proxy?.[name])])),
    sample_scores: rows.map((row) => ({ sample: row.sample, scores: row.scores, composite_score: row.composite_score, response_id: row.scorer.response_id, generation_strategy: row.generation_strategy }))
  };
  const reportFile = path.join(experimentsDir, `llm_visual_validation_${runId}.json`);
  let ledgerFile = null, ledger = null, ledgerCheckpoint = null;
  if (round > 0) {
    ledgerFile = path.join(experimentsDir, `llm_preference_checkpoints_${runId}.json`);
    ledger = JSON.parse(await fs.readFile(ledgerFile, 'utf8'));
    ledgerCheckpoint = ledger.checkpoints?.find((item) => Number(item.round) === round);
    if (!ledgerCheckpoint || ledgerCheckpoint.reward_model_sha256 !== checkpoint.reward_model_sha256) throw Object.assign(new Error('视觉汇总与检查点账本不一致'), { statusCode: 409, code: 'LLM_VISUAL_VAL_LEDGER_MISMATCH' });
  }
  let report = { version: 'llm_visual_validation_v1', run_id: runId, generated_at: new Date().toISOString(), split: 'val', images_per_request: 6, score_dimensions: LLM_SCORE_NAMES, rounds: [] };
  try { report = JSON.parse(await fs.readFile(reportFile, 'utf8')); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  report.generated_at = new Date().toISOString();
  report.rounds = [...(report.rounds || []).filter((item) => Number(item.round) !== round), summary].sort((left, right) => left.round - right.round);
  let visualActivationGate = null, activated = false, backupFile = null;
  if (round > 0) {
    const baseline = report.rounds.find((item) => Number(item.round) === 0) || null;
    visualActivationGate = evaluateVisualActivationGate({ baseline, candidate: summary, geometryEligible: inferGeometrySafetyEligibility(ledgerCheckpoint.validation) });
    summary.activation_gate = visualActivationGate;
    ledgerCheckpoint.visual_validation = { sample_count: summary.sample_count, scorer_model: summary.scorer_model, composite_score: summary.composite_score, score_means: summary.score_means, report_file: path.relative(root, reportFile).split(path.sep).join('/') };
    ledgerCheckpoint.visual_activation_gate = visualActivationGate;
    ledgerCheckpoint.validation_status = visualActivationGate.accepted ? 'accepted_visual_val11' : 'rejected_visual_val11';
    ledgerCheckpoint.activated = visualActivationGate.accepted;
    if (visualActivationGate.accepted) {
      const checkpointFile = path.join(experimentsDir, 'llm_preference_checkpoints', runId, `round_${round}.json`);
      const checkpointBytes = await fs.readFile(checkpointFile);
      const checkpointSha = createHash('sha256').update(checkpointBytes).digest('hex').toUpperCase();
      if (checkpointSha !== ledgerCheckpoint.reward_model_sha256) throw Object.assign(new Error('通过视觉门控的检查点哈希与账本不一致'), { statusCode: 409, code: 'LLM_VISUAL_ACTIVATION_HASH_MISMATCH' });
      try {
        await fs.access(preferenceModelFile);
        backupFile = path.join(experimentsDir, `preference_model_before_visual_activation_${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
        await fs.copyFile(preferenceModelFile, backupFile);
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
      const activeModel = JSON.parse(checkpointBytes.toString('utf8'));
      activeModel.validation_gate = { ...(activeModel.validation_gate || {}), status: 'accepted_visual_val11', visual_activation_gate: visualActivationGate, activated_at: new Date().toISOString() };
      await fs.writeFile(preferenceModelFile, JSON.stringify(activeModel, null, 2) + '\n', 'utf8');
      preferenceModelCache = { mtimeMs: -1, model: null };
      activated = true;
    }
    const selection = {
      generated_at: new Date().toISOString(), run_id: runId, round,
      candidate_model_file: ledgerCheckpoint.reward_model_file,
      active_model_file: activated ? 'experiments/preference_model.json' : null,
      activated, backup_file: backupFile ? path.relative(root, backupFile).split(path.sep).join('/') : null,
      validation: visualActivationGate,
      geometry_validation: ledgerCheckpoint.validation,
      policy: { train_pairs_only: true, validation_controls_activation: true, real_qwen_five_dimension_primary_gate: true, test_not_used_for_activation: true }
    };
    await fs.writeFile(preferenceSelectionFile, JSON.stringify(selection, null, 2) + '\n', 'utf8');
    ledger.generated_at = new Date().toISOString();
    await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n', 'utf8');
    await fs.writeFile(preferenceCheckpointLatestFile, JSON.stringify(ledger, null, 2) + '\n', 'utf8');
  }
  report.activation_policy = { version: 'llm_val11_five_dimension_activation_gate_v1', thresholds: LLM_VISUAL_ACTIVATION_THRESHOLDS, formula: '0.30*overall + 0.25*composition_harmony + 0.20*visual_hierarchy + 0.15*spatial_balance + 0.10*manual_style_similarity', composition_harmony_is_non_degradation_constraint: true };
  await fs.writeFile(reportFile, JSON.stringify(report, null, 2) + '\n', 'utf8');
  await fs.writeFile(visualValidationLatestFile, JSON.stringify(report, null, 2) + '\n', 'utf8');
  return { summary, activated, activation_gate: visualActivationGate, report_file: path.relative(root, reportFile).split(path.sep).join('/') };
}

async function scoringProbeVisuals() {
  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  const sample = manifest.samples.find((item) => item.split === 'train' && Array.isArray(item.views?.files) && item.views.files.length >= VISION_KEYS.length);
  if (!sample) throw Object.assign(new Error('找不到训练集的六张参考 PNG，无法测试视觉接口'), { statusCode: 503, code: 'SCORING_PROBE_IMAGES_MISSING' });
  const visuals = {};
  for (const key of VISION_KEYS) {
    const view = key === 'before' ? 'combined' : key;
    const relative = sample.views.files.find((file) => file.endsWith(`-${view}.png`));
    const absolute = relative && path.resolve(root, relative);
    if (!absolute || !absolute.startsWith(root + path.sep)) throw Object.assign(new Error(`六图参考视角 ${key} 路径无效`), { statusCode: 503, code: 'SCORING_PROBE_IMAGES_MISSING' });
    let bytes;
    try { bytes = await fs.readFile(absolute); }
    catch { throw Object.assign(new Error(`六图参考视角 ${key} 不存在`), { statusCode: 503, code: 'SCORING_PROBE_IMAGES_MISSING' }); }
    if (bytes.length < 1024 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw Object.assign(new Error(`六图参考视角 ${key} 不是完整 PNG`), { statusCode: 503, code: 'SCORING_PROBE_IMAGES_MISSING' });
    visuals[key] = `data:image/png;base64,${bytes.toString('base64')}`;
  }
  return { sample, visuals };
}
async function testScoringConnection() {
  const { sample, visuals } = await scoringProbeVisuals();
  const annotation = JSON.parse(await fs.readFile(path.join(root,sample.target.annotation_json),'utf8'));
  const result = await scoreWithLocalQwen({
    type: 'connection_probe',
    sample: { category: sample.category, sample_id: sample.sample_id },
    split: null,
    strategy: { view_policy: 'five_view_connection_probe' },
    labels: annotationsToLabels(annotation),
    metrics: { note: 'dataset reference images for protocol check only; not a generated-layout score and never used as preference data' },
    visuals
  });
  scoringConnectionTest = { apiUrl: scoringConfig.apiUrl, model: scoringConfig.model, protocolId: SCORING_PROTOCOL_ID, testedAt: new Date().toISOString() };
  return { ok: true, model: result.model, format: result.format, response_id: result.response_id, usage: result.usage || null, testedAt: scoringConnectionTest.testedAt, checks: { authorization: true, six_images_accepted: true, structured_fourteen_dimension_json: true, aesthetic_preference_dimensions: LLM_AESTHETIC_DIMENSIONS, safety_diagnostic_dimensions: LLM_SAFETY_DIMENSIONS, score_dimension_count: 14 } };
}

function validatedRunId(value) {
  const runId = String(value || '');
  if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw Object.assign(new Error('LLM 偏好运行 ID 无效'), { statusCode: 400, code: 'LLM_RUN_ID_INVALID' });
  return runId;
}

async function llmPreferencesForRun(runId) {
  try {
    return (await fs.readFile(preferenceLog, 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)).filter((record) => record.run_id === runId && record.type === 'llm_pairwise_preference');
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

async function writeLLMPreferenceCheckpoint(input) {
  const runId = validatedRunId(input?.runId);
  const round = Number(input?.round);
  if (!Number.isInteger(round) || round < 1 || round > 64) throw Object.assign(new Error('偏好检查点轮次无效'), { statusCode: 400, code: 'LLM_CHECKPOINT_ROUND_INVALID' });
  const candidateBytes = await fs.readFile(preferenceCandidateFile);
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (candidate.training?.source_filter !== 'llm' || candidate.training?.run_id_filter !== runId || candidate.training?.source_counts?.human !== 0) throw Object.assign(new Error('检查点候选模型不属于当前纯 LLM 运行'), { statusCode: 409, code: 'LLM_CHECKPOINT_MODEL_MISMATCH' });
  const selection = JSON.parse(await fs.readFile(preferenceSelectionFile, 'utf8'));
  const preferences = await llmPreferencesForRun(runId);
  const cumulative = preferences.filter((record) => Number(record.round) <= round);
  if (!cumulative.length) throw Object.assign(new Error('当前轮次尚无可保存的 LLM 偏好对'), { statusCode: 400, code: 'LLM_CHECKPOINT_EMPTY' });
  const checkpointDir = path.join(experimentsDir, 'llm_preference_checkpoints', runId);
  await fs.mkdir(checkpointDir, { recursive: true });
  const modelFile = path.join(checkpointDir, `round_${round}.json`);
  try { await fs.writeFile(modelFile, candidateBytes, { flag: 'wx' }); }
  catch (error) {
    if (error.code === 'EEXIST') throw Object.assign(new Error('本 run_id 的该轮冻结检查点已存在；不可覆盖，请另启新运行'), { statusCode: 409, code: 'LLM_CHECKPOINT_IMMUTABLE' });
    throw error;
  }
  const modelSha256 = createHash('sha256').update(candidateBytes).digest('hex').toUpperCase();
  const gate = candidate.validation_gate || selection.validation || null;
  const checkpoint = {
    round,
    standard_checkpoint: [1, 2, 4, 8].includes(round),
    saved_at: new Date().toISOString(),
    cumulative_pair_count: cumulative.length,
    reward_model_file: path.relative(root, modelFile).split(path.sep).join('/'),
    reward_model_sha256: modelSha256,
    validation_status: gate?.status || 'unknown',
    activated: Boolean(selection.activated && gate?.status === 'accepted'),
    validation_samples: gate?.validation_samples ?? null,
    validation: gate ? { baseline: gate.baseline, previous_active: gate.previous_active, candidate: gate.candidate, constraints: gate.constraints, criterion: gate.criterion, geometry_safety_eligible: gate.geometry_safety_eligible, visual_activation_pending: gate.visual_activation_pending } : null,
    training: candidate.training
  };
  const ledgerFile = path.join(experimentsDir, `llm_preference_checkpoints_${runId}.json`);
  let previous = [];
  try { previous = JSON.parse(await fs.readFile(ledgerFile, 'utf8')).checkpoints || []; } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const checkpoints = [...previous.filter((item) => Number(item.round) !== round), checkpoint].sort((left, right) => left.round - right.round);
  const ledger = { version: 'llm_preference_checkpoint_ledger_v1', run_id: runId, generated_at: new Date().toISOString(), standard_checkpoints: [1, 2, 4, 8], test_evaluated_at_checkpoints: false, checkpoints };
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n', 'utf8');
  await fs.writeFile(preferenceCheckpointLatestFile, JSON.stringify(ledger, null, 2) + '\n', 'utf8');
  return { checkpoint, ledger_file: path.relative(root, ledgerFile).split(path.sep).join('/') };
}

async function selectFinalLLMCheckpoint(runId) {
  const ledger = JSON.parse(await fs.readFile(path.join(experimentsDir, `llm_preference_checkpoints_${runId}.json`), 'utf8'));
  if (ledger.run_id !== runId || !Array.isArray(ledger.checkpoints) || !ledger.checkpoints.length) throw Object.assign(new Error('本次运行尚无可测试的验证集检查点'), { statusCode: 409 });
  let visual;
  try { visual = JSON.parse(await fs.readFile(path.join(experimentsDir, `llm_visual_validation_${runId}.json`), 'utf8')); }
  catch { throw Object.assign(new Error('最终检查点必须先完成真实 val11 六图视觉评分'), { statusCode: 409, code: 'LLM_VISUAL_VAL_REQUIRED' }); }
  const visualByRound = new Map((visual.rounds || []).map((item) => [Number(item.round), item]));
  const completeVisual = ledger.checkpoints.filter((item) => {
    const row = visualByRound.get(Number(item.round));
    return row?.sample_count === 11
      && Number.isFinite(Number(row.composite_score))
      && row.checkpoint?.reward_model_sha256 === item.reward_model_sha256
      && item.visual_activation_gate?.evidence?.full_val11 === true
      && item.visual_activation_gate?.evidence?.same_scorer === true
      && item.visual_activation_gate?.evidence?.same_cohort === true;
  });
  if (!completeVisual.length) throw Object.assign(new Error('没有完成同 scorer、同 cohort 的真实 val11 检查点'), { statusCode: 409, code: 'LLM_VISUAL_VAL_SELECTION_EMPTY' });
  const firstVisual = visualByRound.get(Number(completeVisual[0].round));
  const reference = JSON.stringify((firstVisual.sample_scores || []).map((row) => `${row.sample.category}/${row.sample.sample_id}`).sort());
  if (completeVisual.some((item) => {
    const row = visualByRound.get(Number(item.round));
    return row.scorer_model !== firstVisual.scorer_model || JSON.stringify((row.sample_scores || []).map((score) => `${score.sample.category}/${score.sample.sample_id}`).sort()) !== reference;
  })) throw Object.assign(new Error('各检查点的 val11 视觉评分样本或模型不一致，无法同口径选模'), { statusCode: 409, code: 'LLM_VISUAL_VAL_COHORT_MISMATCH' });
  const accepted = completeVisual.filter((item) => item.validation_status === 'accepted_visual_val11' && item.visual_activation_gate?.accepted === true && item.activated === true);
  const diagnosticOnly = accepted.length === 0;
  const pool = accepted.length ? accepted : completeVisual;
  const selected = [...pool].sort((left, right) => {
    const leftVisual = visualByRound.get(Number(left.round));
    const rightVisual = visualByRound.get(Number(right.round));
    return Number(rightVisual.composite_score) - Number(leftVisual.composite_score)
      || Number(rightVisual.score_means?.composition_harmony) - Number(leftVisual.score_means?.composition_harmony)
      || Number(left.validation?.candidate?.objective_score ?? Infinity) - Number(right.validation?.candidate?.objective_score ?? Infinity)
      || Number(left.round) - Number(right.round);
  })[0];
  if (!selected || !Number.isInteger(Number(selected.round))) throw Object.assign(new Error('验证集缺少可选择的检查点'), { statusCode: 409 });
  const modelFile = path.join(experimentsDir, 'llm_preference_checkpoints', runId, `round_${selected.round}.json`);
  const bytes = await fs.readFile(modelFile);
  const digest = createHash('sha256').update(bytes).digest('hex').toUpperCase();
  if (digest !== selected.reward_model_sha256 || JSON.parse(bytes.toString('utf8')).training?.run_id_filter !== runId) throw Object.assign(new Error('检查点模型哈希或运行来源不匹配'), { statusCode: 409 });
  return {
    runId,
    round: selected.round,
    modelFile,
    modelSha256: digest,
    validationStatus: selected.validation_status,
    diagnosticOnly,
    selectionPolicy: diagnosticOnly ? 'best_rejected_visual_val11_diagnostic_only' : 'best_accepted_visual_val11'
  };
}

function buildLLMConvergenceSummary(checkpointLedger, testReport, visualReport) {
  const standardRounds = [1, 2, 4, 8];
  const rounds = [0, ...standardRounds];
  const available = new Map((checkpointLedger?.checkpoints || []).map((item) => [Number(item.round), item]));
  const visual = new Map((visualReport?.rounds || []).map((item) => [Number(item.round), item]));
  const first = [...available.values()].sort((left, right) => left.round - right.round)[0];
  const geometryBaseline = first?.validation?.baseline || null;
  const geometryMetricNames = ['aesthetic_composition_harmony', 'intrinsic_style_balance', 'intrinsic_size_consistency', 'multidimensional_quality_score', 'objective_score', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'manual_style_distance', 'text_clarity'];
  const checkpoints = rounds.map((round) => {
    const item = available.get(round);
    const visualItem = visual.get(round) || null;
    const geometry = round === 0 ? geometryBaseline : item?.validation?.candidate || null;
    return {
      round,
      status: visualItem ? 'evaluated' : 'unavailable',
      reason: visualItem ? undefined : round === 0 ? '尚未执行无奖励基线的 val 六图评分' : `本次运行尚无第 ${round} 轮冻结检查点的 val 六图评分`,
      cumulative_pair_count: round === 0 ? 0 : item?.cumulative_pair_count ?? null,
      activated: item?.activated,
      validation_status: item?.validation_status,
      reward_model_file: item?.reward_model_file || null,
      reward_model_sha256: item?.reward_model_sha256 || null,
      visual_validation: visualItem,
      geometry_proxy: geometry,
      validation: geometry
    };
  });
  const visuallyEvaluated = checkpoints.filter((item) => item.visual_validation);
  for (let index = 1; index < visuallyEvaluated.length; index += 1) {
    const previous = visuallyEvaluated[index - 1].visual_validation;
    const current = visuallyEvaluated[index].visual_validation;
    visuallyEvaluated[index].visual_delta_vs_previous = {
      composite_score: Number((current.composite_score - previous.composite_score).toFixed(6)),
      score_means: Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, Number(((current.score_means?.[name] ?? 0) - (previous.score_means?.[name] ?? 0)).toFixed(6))]))
    };
  }
  const baseline = visual.get(0) || null;
  const trainedVisual = standardRounds.map((round) => visual.get(round)).filter(Boolean);
  const peak = [...trainedVisual].sort((left, right) => Number(right.composite_score) - Number(left.composite_score))[0] || null;
  const round4 = visual.get(4) || null;
  const round8 = visual.get(8) || null;
  const scoredRows = rounds.map((round) => visual.get(round));
  const referenceSamples = JSON.stringify(scoredRows[0]?.sample_scores?.map((item) => `${item.sample.category}/${item.sample.sample_id}`).sort() || []);
  const evidenceSufficient = scoredRows.every((item) => item?.sample_count === 11 && item.scorer_model === scoredRows[0]?.scorer_model && JSON.stringify(item.sample_scores?.map((row) => `${row.sample.category}/${row.sample.sample_id}`).sort()) === referenceSamples);
  const improved = Boolean(baseline && peak && peak.composite_score >= baseline.composite_score + 0.05 && LLM_SAFETY_DIMENSIONS.every((name) => (peak.score_means?.[name] ?? 0) >= (baseline.score_means?.[name] ?? 0) - 0.15));
  const meanDimensionChange = round4 && round8 ? LLM_AESTHETIC_DIMENSIONS.reduce((sum, name) => sum + Math.abs((round8.score_means?.[name] ?? 0) - (round4.score_means?.[name] ?? 0)), 0) / LLM_AESTHETIC_DIMENSIONS.length : Infinity;
  const plateau4to8 = Boolean(round4 && round8 && Math.abs(round8.composite_score - round4.composite_score) <= 0.1 && meanDimensionChange <= 0.15);
  const geometryDeltas = checkpoints.filter((item) => item.geometry_proxy);
  for (let index = 1; index < geometryDeltas.length; index += 1) geometryDeltas[index].geometry_delta_vs_previous = Object.fromEntries(geometryMetricNames.map((name) => [name, Number(((geometryDeltas[index].geometry_proxy?.[name] ?? 0) - (geometryDeltas[index - 1].geometry_proxy?.[name] ?? 0)).toFixed(6))]));
  return {
    version: 'llm_preference_convergence_v4_aesthetic_reward_safety_gate',
    standard_rounds: rounds,
    primary_evidence: 'independent_val_six_image_external_fourteen_dimension_scores_with_aesthetic_only_preference',
    geometry_proxy_reported_separately: true,
    evidence_sufficient: evidenceSufficient,
    improved_before_plateau: evidenceSufficient && improved,
    plateau_round4_to_round8: evidenceSufficient && plateau4to8,
    rise_then_stable: evidenceSufficient && improved && plateau4to8,
    thresholds: { minimum_composite_improvement: 0.05, maximum_safety_dimension_drop: 0.15, maximum_round4_to_round8_composite_change: 0.1, maximum_round4_to_round8_mean_dimension_change: 0.15 },
    checkpoints,
    final_test11: testReport ? { evaluated_once_after_final_checkpoint: true, selected_round: testReport.selected_round ?? null, selection_policy: testReport.selection_policy ?? null, candidate_validation_status: testReport.candidate_validation_status, candidate_activated: testReport.candidate_activated, test_samples: testReport.test_samples, summary: testReport.summary } : null,
    interpretation: !evidenceSufficient ? '尚未收集完整的真实 val 六图 0/1/2/4/8 轮十四维评分，不能判断美学偏好是否先提升后稳定。' : improved && plateau4to8 ? '本地 Qwen3-VL 在独立 val 六图上的美学综合分先提升，且第4到第8轮进入预设稳定阈值；安全诊断与几何代理分并列报告，test11仅作最终确认。' : '已有完整真实 val 六图 0/1/2/4/8 轮十四维评分，但不满足“美学先提升且4到8轮稳定”的预设联合判据。',
    policy: { reward_target: 'aesthetic_dimensions_only', deterministic_energy_controls_safety: true, safety_dimensions: LLM_SAFETY_DIMENSIONS, aesthetic_dimensions: LLM_AESTHETIC_DIMENSIONS, validation_visual_scores_select_and_track_checkpoints: true, validation_scores_never_create_train_preferences: true, immutable_checkpoint_hash_verified: true, test_evaluated_once_after_final_checkpoint: true, test_not_used_for_training_or_checkpoint_selection: true }
  };
}

async function writeLLMPreferenceRunReport(input) {
  const runId = validatedRunId(input?.runId);
  const config = safeScoringConfig();
  if (!config.tested) throw Object.assign(new Error('视觉评分连接尚未通过六图与 JSON 自检'), { statusCode: 409, code: 'SCORING_NOT_TESTED' });
  const preferences = await llmPreferencesForRun(runId);
  if (!preferences.length) throw Object.assign(new Error('该 run_id 尚无真实 LLM 偏好对'), { statusCode: 400, code: 'LLM_RUN_EMPTY' });
  if (preferences.some((record) => record.type !== 'llm_pairwise_preference' || record.split !== 'train' || !record.scoring_evidence?.chosen_receipt_id || !record.scoring_evidence?.rejected_receipt_id || record.scoring_evidence.chosen_receipt_id === record.scoring_evidence.rejected_receipt_id || record.scoring_evidence.chosen_model !== config.model || record.scoring_evidence.rejected_model !== config.model)) throw Object.assign(new Error('运行中的偏好记录缺少一致的真实评分证据'), { statusCode: 409, code: 'LLM_RUN_EVIDENCE_INVALID' });
  const preferenceModel = JSON.parse(await fs.readFile(preferenceCandidateFile, 'utf8'));
  if (preferenceModel.training?.source_filter !== 'llm' || preferenceModel.training?.run_id_filter !== runId || preferenceModel.training?.source_counts?.human !== 0) throw Object.assign(new Error('奖励模型不是由当前 run_id 的纯 LLM 偏好训练得到'), { statusCode: 409, code: 'LLM_RUN_MODEL_MISMATCH' });
  let testReport = null;
  try { const latestTest = JSON.parse(await fs.readFile(preferenceTestReportFile, 'utf8')); if (latestTest.run_id === runId) testReport = latestTest; } catch {}
  const activeRewardPresent = await fs.stat(preferenceModelFile).then(() => true, () => false);
  let validationSelection = null;
  try { validationSelection = JSON.parse(await fs.readFile(preferenceSelectionFile, 'utf8')); } catch {}
  let checkpointLedger = null;
  try { checkpointLedger = JSON.parse(await fs.readFile(path.join(experimentsDir, `llm_preference_checkpoints_${runId}.json`), 'utf8')); } catch {}
  let visualValidation = null;
  try { visualValidation = JSON.parse(await fs.readFile(path.join(experimentsDir, `llm_visual_validation_${runId}.json`), 'utf8')); } catch {}
  const convergence = buildLLMConvergenceSummary(checkpointLedger, testReport, visualValidation);
  const report = {
    version: 'llm_preference_run_report_v1',
    generated_at: new Date().toISOString(),
    run_id: runId,
    scorer: { api_url: config.apiUrl, format: config.format, model: config.model, connection_tested_at: config.testedAt, api_key_persisted: false },
    request: { sample: input?.sample || null, requested_rounds: Number(input?.rounds || 0), candidates_per_round: Number(input?.candidates || 0) },
    evidence: { pair_count: preferences.length, train_only: preferences.every((record) => record.split === 'train'), fixed_label_contract_checked_on_write: true, receipt_ids: preferences.flatMap((record) => [record.scoring_evidence.chosen_receipt_id, record.scoring_evidence.rejected_receipt_id]), response_ids: preferences.flatMap((record) => [record.scoring_evidence.chosen_response_id, record.scoring_evidence.rejected_response_id]).filter(Boolean) },
    reward_model: { file: 'experiments/preference_model_candidate.json', active_file: activeRewardPresent ? 'experiments/preference_model.json' : null, candidate_activated: preferenceModel.validation_gate?.status === 'accepted', status: preferenceModel.status, architecture: preferenceModel.architecture, validation_gate: preferenceModel.validation_gate, validation_selection: validationSelection, training: preferenceModel.training },
    checkpoints: checkpointLedger ? { file: `experiments/llm_preference_checkpoints_${runId}.json`, standard_rounds: checkpointLedger.standard_checkpoints, entries: checkpointLedger.checkpoints } : null,
    visual_validation: visualValidation ? { file: `experiments/llm_visual_validation_${runId}.json`, split: 'val', images_per_request: 6, score_dimensions: LLM_SCORE_NAMES, rounds: visualValidation.rounds } : null,
    convergence,
    test11: testReport ? { file: 'experiments/preference_test11_report.json', summary: testReport.summary, test_samples: testReport.test_samples } : null,
    limits: ['External scorer quality is provider-dependent.', 'Reward model reranks layout candidates; it does not directly update the base layout MLP.', 'API key is not written to this report or project files.']
  };
  const runFile = path.join(experimentsDir, `llm_preference_run_${runId}.json`);
  await fs.writeFile(runFile, JSON.stringify(report, null, 2) + '\n', 'utf8');
  await fs.writeFile(path.join(experimentsDir, 'llm_preference_run_latest.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
  const convergenceReport = { ...convergence, generated_at: report.generated_at, run_id: runId, scorer: report.scorer };
  await fs.writeFile(path.join(experimentsDir, `llm_preference_convergence_${runId}.json`), JSON.stringify(convergenceReport, null, 2) + '\n', 'utf8');
  await fs.writeFile(path.join(experimentsDir, 'llm_preference_convergence_latest.json'), JSON.stringify(convergenceReport, null, 2) + '\n', 'utf8');
  return { report, report_file: path.relative(root, runFile).split(path.sep).join('/'), latest_file: 'experiments/llm_preference_run_latest.json' };
}

const FINAL_EIGHT_METRICS = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];

function unifiedReproductionRows(report) {
  return (report.rows || []).map((row) => ({
    ...row,
    quality_score: reproductionQualityScore(row, row.num_labels)
  }));
}

function summarizeUnifiedReproductionRows(rows, method, split = 'val') {
  const selected = rows.filter((row) => row.method === method && (split === 'all' || row.split === split));
  const average = (metric) => {
    const values = selected.map((row) => row[metric])
      .filter((value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)))
      .map(Number);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  return {
    method,
    sample_count: selected.length,
    ...Object.fromEntries(FINAL_EIGHT_METRICS.map((metric) => [metric, average(metric)]))
  };
}

async function buildFinalEightModelResults() {
  const report = JSON.parse(await fs.readFile(finalEightTest11File, 'utf8'));
  const recomputed = validateAndSummarizeFinalEightTest11(report);
  const summaries = new Map(recomputed.summaries.map((row) => [row.method, row]));
  const projectMetrics = (source = {}) => Object.fromEntries(FINAL_EIGHT_METRICS.map((name) => [name, source?.[name] ?? null]));
  const resultRow = (id, label, status, role) => {
    const source = summaries.get(id);
    if (!source || source.sample_count !== 11) throw Object.assign(new Error('Missing complete test11 method in final results: ' + id), { statusCode: 409 });
    return { id, label, status, role, sample_count: source.sample_count, source: 'final_eight_test11', ...projectMetrics(source) };
  };
  const rows = [
    resultRow('v10_no_rerank', '原始 v10', 'test11 最终报告', 'active'),
    resultRow('v10_historical_rerank', 'v10 + 基础奖励 MLP', 'test11 最终报告', 'reward'),
    resultRow('mdpo_no_rerank', 'v10 + MDPO', 'test11 事后诊断 / 未部署', 'candidate'),
    resultRow('mdpo_safe_rerank', 'v10 + MDPO + 基础奖励 MLP', 'test11 事后诊断 / 未部署', 'candidate'),
    resultRow('BinoForce', 'BinoForce', 'test11 论文复现基线', 'baseline'),
    resultRow('hedgehog_3d', 'Hedgehog 3D', 'test11 论文复现基线', 'baseline'),
    resultRow('hedgehog_1d', 'Hedgehog 1D', 'test11 论文复现基线', 'baseline'),
    resultRow('manual', '人工优化布局', 'test11 参考上限', 'reference')
  ];
  return {
    version: 'final_eight_model_results_v3_test11_pck_lcd',
    generated_at: report.generated_at,
    audited_at: new Date().toISOString(),
    source_generated_at: { test11: report.generated_at },
    split: 'test', sample_count: 11,
    views: ['main', 'right', 'left', 'up', 'down'],
    protocol: { ...REPRODUCTION_METRIC_PROTOCOL_V4, scope: 'final dashboard table: eight model/method test11 five-view means; PCK 35%, LCD 20%' },
    metric_names: FINAL_EIGHT_METRICS,
    audit: recomputed.audit,
    rows,
    notes: [
      'MDPO candidates failed the val11 deployment gate; their test11 rows are post-hoc diagnostics requested on September 26, 2026 and do not change deployment status.',
      'DBV is unavailable for original Hedgehog reproductions.',
      'Manual PCK equals 1 because the manual layout is the PCK reference itself.',
      'Qwen aesthetic scores are omitted because they are not available for all eight methods.',
      'Safety quality scores for all eight methods use the same v4 formula with PCK 35% and LCD 20%: per-sample normalization followed by test11 averaging.'
    ]
  };
}

const MODEL_VISUALIZATION_METHODS = {
  v10_no_rerank: { label: '原始 v10', kind: 'generated', note: '正式活动 v10，不使用奖励重排' },
  v10_historical_rerank: { label: 'v10 + 基础奖励 MLP', kind: 'generated', note: '正式活动 v10，使用基础奖励 MLP 安全候选重排' },
  mdpo_no_rerank: { label: 'v10 + MDPO', kind: 'generated', note: '正式选中但未部署的 MDPO 诊断候选，不使用奖励重排' },
  mdpo_safe_rerank: { label: 'v10 + MDPO + 基础奖励 MLP', kind: 'generated', note: 'MDPO 诊断候选与基础奖励 MLP 的补充对照' },
  BinoForce: { label: 'BinoForce', kind: 'reproduction', note: 'BinoForce 三维动态标签布局复现' },
  hedgehog_3d: { label: 'Hedgehog 3D', kind: 'fixed_views', note: '每个固定相机分别复现的 3D Hedgehog 布局' },
  hedgehog_1d: { label: 'Hedgehog 1D', kind: 'fixed_views', note: '每个固定相机分别复现的 1D Pole 布局' },
  manual: { label: '人工优化布局', kind: 'reference', note: '数据集人工调整后的三维参考布局' }
};

function labelsWithReproducedCenters(manualLabels, centers) {
  const queues = new Map();
  for (const row of centers || []) {
    const key = String(row.text || '');
    if (!queues.has(key)) queues.set(key, []);
    queues.get(key).push(row.center);
  }
  return manualLabels.map((label) => {
    const queue = queues.get(String(label.text || '')) || [];
    const center = queue.shift();
    return { ...label, center: Array.isArray(center) && center.length === 3 ? center.map(Number) : [...label.center] };
  });
}

async function buildModelVisualization(payload) {
  const method = String(payload?.method || 'v10_historical_rerank');
  const definition = MODEL_VISUALIZATION_METHODS[method];
  if (!definition) throw Object.assign(new Error('不支持的模型/方法'), { statusCode: 400, code: 'MODEL_VISUALIZATION_METHOD_INVALID' });
  const category = String(payload?.sample?.category || '');
  const sampleId = String(payload?.sample?.sample_id || payload?.sample?.id || '');
  if (!category || !sampleId) throw Object.assign(new Error('请选择数据集样本后再切换模型'), { statusCode: 400, code: 'MODEL_VISUALIZATION_SAMPLE_REQUIRED' });
  if (definition.kind === 'generated') {
    const generated = await generateLayout({
      generator: 'annotation',
      sample: { category, sample_id: sampleId },
      display_model: method,
      strategy: { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', seed: 17017, iterations: 180,
        preferenceRerank: ['v10_historical_rerank', 'mdpo_safe_rerank'].includes(method) }
    });
    return { ...generated, method, method_label: definition.label, method_note: definition.note,
      labels_by_view: Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, generated.labels])), fixed_view_layout: false };
  }

  const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
  const manifestSample = manifest.samples.find((item) => item.category === category && String(item.sample_id) === sampleId);
  if (!manifestSample) throw Object.assign(new Error('样本不在数据集 manifest 中'), { statusCode: 404, code: 'MODEL_VISUALIZATION_SAMPLE_NOT_FOUND' });
  const [rawObj, annotation] = await Promise.all([
    fs.readFile(path.join(root, manifestSample.input.source_obj), 'utf8'),
    fs.readFile(path.join(root, manifestSample.target.annotation_json), 'utf8').then(JSON.parse)
  ]);
  const clean = cleanObj(rawObj);
  const manualLabels = annotationsToLabels(annotation);
  let labelsByView;
  if (method === 'manual') {
    labelsByView = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, manualLabels]));
  } else if (method === 'BinoForce') {
    const text = await fs.readFile(path.join(root, 'baselines', 'BinoForce_2025', 'results', 'binoforce_layouts.jsonl'), 'utf8');
    const rows = text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))
      .filter((row) => row.category === category && String(row.sample) === sampleId && row.method === 'BinoForce');
    const byView = new Map(rows.map((row) => [row.view, row]));
    labelsByView = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => {
      const row = byView.get(view);
      if (!row) throw Object.assign(new Error('BinoForce 缺少 ' + view + ' 视角结果'), { statusCode: 404, code: 'BINOFORCE_VIEW_MISSING' });
      return [view, labelsWithReproducedCenters(manualLabels, row.label_centers)];
    }));
  } else {
    labelsByView = Object.fromEntries(await Promise.all(MULTI_VIEW_NAMES.map(async (view) => {
      const file = path.join(root, 'baselines', 'Hedgehog', 'results', 'layouts', category, sampleId, view, method + '.json');
      const layout = await fs.readFile(file, 'utf8').then(JSON.parse).catch((error) => {
        if (error?.code === 'ENOENT') throw Object.assign(new Error(definition.label + ' 缺少 ' + view + ' 视角结果'), { statusCode: 404, code: 'HEDGEHOG_VIEW_MISSING' });
        throw error;
      });
      return [view, annotationsToLabels(layout)];
    })));
  }
  const labels = labelsByView.main || manualLabels;
  return {
    version: 'model_visualization_v1', generated_at: new Date().toISOString(),
    sample: { category, sample_id: sampleId, split: manifestSample.split }, source_name: path.basename(manifestSample.input.source_obj),
    method, method_label: definition.label, method_note: definition.note, fixed_view_layout: definition.kind === 'fixed_views',
    clean_obj: clean.text, labels, labels_by_view: labelsByView,
    manual_reference: { labels: manualLabels }, timing: null, metrics: null, manual_comparison: null,
    strategy: { presentation_model: method, five_view_energy: true, source: definition.kind }
  };
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method) && req.headers.origin) {
      const origin = new URL(req.headers.origin);
      if (origin.host !== url.host || !['http:', 'https:'].includes(origin.protocol)) return sendJson(res, 403, { error: '仅允许同源工作台修改本地设置', code: 'ORIGIN_FORBIDDEN' });
    }
    if (url.pathname === '/api/catalog') {
      const catalog = await getCatalog();
      res.writeHead(200, { 'Content-Type': mimeTypes['.json'] });
      res.end(JSON.stringify(catalog));
      return;
    }
    if (url.pathname === '/api/dataset-split') {
      const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
      return sendJson(res, 200, manifest);
    }
    if (url.pathname === '/api/quality-comparison' && req.method === 'GET') {
      try { return sendJson(res, 200, await buildQualityComparison(url.searchParams.get('category'), url.searchParams.get('sampleId'))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'QUALITY_COMPARISON_FAILED' }); }
    }
    if (url.pathname === '/api/final-eight-model-results' && req.method === 'GET') {
      try { return sendJson(res, 200, await buildFinalEightModelResults()); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'FINAL_EIGHT_MODEL_RESULTS_FAILED' }); }
    }
    if (url.pathname === '/api/model-visualization' && req.method === 'POST') {
      try { return sendJson(res, 200, await buildModelVisualization(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'MODEL_VISUALIZATION_FAILED' }); }
    }
    if (url.pathname === '/api/reproduction-comparison' && req.method === 'GET') {
      try {
        const report = JSON.parse(await fs.readFile(reproductionComparisonFile, 'utf8'));
        const rows = unifiedReproductionRows(report);
        const viewRows = (report.view_rows || []).map((row) => ({ ...row, quality_score: reproductionQualityScore(row, row.num_labels) }));
        const retainedMethods = new Set(['BinoForce', 'hedgehog_3d', 'hedgehog_1d', 'manual']);
        const split = ['train', 'val', 'test', 'all'].includes(url.searchParams.get('split')) ? url.searchParams.get('split') : 'test';
        const category = url.searchParams.get('category');
        const sampleId = url.searchParams.get('sampleId');
        const sampleRows = category && sampleId ? rows.filter((row) => row.category === category && row.sample_id === sampleId && retainedMethods.has(row.method)) : [];
        const sampleMainRows = category && sampleId ? viewRows.filter((row) => row.category === category && row.sample_id === sampleId && row.view === 'main' && retainedMethods.has(row.method)) : [];
        const average = (source, method, metric) => { const values = source.filter((row) => row.method === method && (split === 'all' || row.split === split) && row[metric] !== null && row[metric] !== undefined && row[metric] !== '').map((row) => Number(row[metric])).filter(Number.isFinite); return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null; };
        const metricNames = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
        const summaryRows = [...retainedMethods].map((method) => ({ method, label: report.methods?.[method] || method,
          sample_count: rows.filter((row) => row.method === method && (split === 'all' || row.split === split)).length,
          ...Object.fromEntries(metricNames.map((metric) => [metric, average(rows, method, metric)])) }));
        const mainSummaryRows = [...retainedMethods].map((method) => ({ method, label: report.methods?.[method] || method,
          sample_count: viewRows.filter((row) => row.view === 'main' && row.method === method && (split === 'all' || row.split === split)).length,
          ...Object.fromEntries(metricNames.map((metric) => [metric, average(viewRows.filter((row) => row.view === 'main'), method, metric)])) }));
        return sendJson(res, 200, { version: 'aligned_dashboard_comparison_v2_unified_safety_score', generated_at: new Date().toISOString(), active_model: report.active_model, preference_model: null, protocol: { ...REPRODUCTION_METRIC_PROTOCOL, scope: 'dashboard: unified v3 safety score for BinoForce, Hedgehog 3D/1D, and manual reference' }, split_counts: report.split_counts, methods: { BinoForce: 'BinoForce', hedgehog_3d: 'Hedgehog 3D', hedgehog_1d: 'Hedgehog 1D', manual: '人工优化布局' }, selected_split: split, sample: category && sampleId ? { category, sample_id: sampleId, split: sampleRows[0]?.split || null, rows: sampleRows, main_rows: sampleMainRows } : null, summary: summaryRows, main_summary: mainSummaryRows });
      } catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'REPRODUCTION_COMPARISON_FAILED' }); }
    }
    if (url.pathname === '/api/llm-val11-comparison' && req.method === 'GET') {
      try {
        const gate = JSON.parse(await fs.readFile(llmValGateComparisonFile, 'utf8'));
        const reproduction = JSON.parse(await fs.readFile(reproductionComparisonFile, 'utf8'));
        const requestedMetrics = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
        const currentRunActivated = gate.active_reward?.training_run_id === gate.run_id && gate.active_reward?.validation_gate?.status === 'accepted_visual_val11';
        const valMetrics = reproduction.summaries.filter((row) => row.split === 'val').map((row) => ({ method: row.method, label: row.method === 'latest_v10' ? currentRunActivated ? 'v10 + 本轮已激活 Qwen' : '当前部署 v10（非本轮 Qwen）' : row.label, sample_count: row.sample_count, ...Object.fromEntries(requestedMetrics.map((name) => [name, row[name]])) }));
        return sendJson(res, 200, { ...gate, current_run_activated: currentRunActivated, requested_metrics: requestedMetrics, val11_reproduction_metrics: valMetrics, base_model: reproduction.active_model, metric_protocol: reproduction.protocol });
      } catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'LLM_VAL11_COMPARISON_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-training-control' && req.method === 'GET') {
      try { return sendJson(res, 200, await mdpoUiTrainingControlStatus()); }
      catch (error) { return sendJson(res, 500, { error: error.message, code: 'MDPO_TRAINING_STATUS_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-training-control' && req.method === 'POST') {
      try { return sendJson(res, 202, await startMdpoUiTraining(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 400, { error: error.message, code: error.code || 'MDPO_TRAINING_START_FAILED', status: error.status || null }); }
    }
    if (url.pathname === '/api/mdpo-live-results' && req.method === 'GET') {
      try { return sendJson(res, 200, await buildMdpoLiveResults({ root, experimentsDir })); }
      catch (error) { return sendJson(res, 500, { error: error.message, code: 'MDPO_LIVE_RESULTS_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-experiment' && req.method === 'GET') {
      const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
      const dataset = await readOptionalJson(mdpoPairFile);
      const audit = dataset ? validateMdpoDataset(dataset, manifest, { requireComplete: false }) : null;
      const activeBytes = await fs.readFile(layoutModelFile), active = JSON.parse(activeBytes.toString('utf8'));
      const activeHash = createHash('sha256').update(activeBytes).digest('hex');
      const preferenceBytes = await fs.readFile(preferenceModelFile).catch(() => null);
      const preferenceHash = preferenceBytes ? createHash('sha256').update(preferenceBytes).digest('hex') : null;
      const selectionBytes = await fs.readFile(path.join(mdpoDir, 'hyperparameter_selection.json')).catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
      const selection = selectionBytes ? JSON.parse(selectionBytes.toString('utf8')) : null;
      const rejectionReport = await readOptionalJson(path.join(mdpoDir, 'val11_rejection_report.json'));
      const activationReport = await readOptionalJson(path.join(mdpoDir, 'activation_report.json'));
      const resolveMdpoArtifact = (relative, expectedBasename = null) => {
        if (!relative) return null;
        const target = path.resolve(root, String(relative));
        if (!(target === mdpoDir || target.startsWith(`${mdpoDir}${path.sep}`)) || path.extname(target).toLowerCase() !== '.json'
            || (expectedBasename && path.basename(target) !== expectedBasename)) return null;
        return target;
      };
      const selectedCandidateFile = resolveMdpoArtifact(selection?.selected?.candidate_file, 'layout_model_v10_mdpo_candidate.json');
      const selectedReportFile = resolveMdpoArtifact(selection?.selected?.report_file);
      const selectedFourGroupFile = resolveMdpoArtifact(selection?.selected?.four_group_report_file);
      const selectedCandidateBytes = selectedCandidateFile ? await fs.readFile(selectedCandidateFile).catch(() => null) : null;
      const selectedReportBytes = selectedReportFile ? await fs.readFile(selectedReportFile).catch(() => null) : null;
      const selectedFourGroupBytes = selectedFourGroupFile ? await fs.readFile(selectedFourGroupFile).catch(() => null) : null;
      const selectedCandidateSha256 = selectedCandidateBytes ? createHash('sha256').update(selectedCandidateBytes).digest('hex') : null;
      const selectedReportSha256 = selectedReportBytes ? createHash('sha256').update(selectedReportBytes).digest('hex') : null;
      const selectedFourGroupSha256 = selectedFourGroupBytes ? createHash('sha256').update(selectedFourGroupBytes).digest('hex') : null;
      const trained = selectedCandidateBytes ? JSON.parse(selectedCandidateBytes.toString('utf8')) : await readOptionalJson(path.join(mdpoDir, 'training', 'layout_model_v10_mdpo_candidate.json'));
      const checkpoint = selectedCandidateFile ? await readOptionalJson(path.join(path.dirname(selectedCandidateFile), 'checkpoint_ledger.json')) : await readOptionalJson(path.join(mdpoDir, 'training', 'checkpoint_ledger.json'));
      const selectedReport = selectedReportBytes ? JSON.parse(selectedReportBytes.toString('utf8')) : null;
      const gate = selectedReport?.baseline && selectedReport?.candidate ? evaluateMdpoVal11Gate({ baseline: selectedReport.baseline, candidate: selectedReport.candidate }) : null;
      let rejectionVerified = false;
      if (selectionBytes && rejectionReport && preferenceHash && gate?.accepted === false
          && selection?.rejection_report_file === 'experiments/mdpo/val11_rejection_report.json') {
        try {
          const expected = buildMdpoRejectionReport(selection, { selectionSha256: createHash('sha256').update(selectionBytes).digest('hex'),
            referenceSha256: activeHash, historicalRewardSha256: preferenceHash });
          rejectionVerified = active.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
            && dataset?.reference_model_sha256 === activeHash && JSON.stringify(gate) === JSON.stringify(selection.selected.gate)
            && selectedCandidateSha256 === selection.selected.candidate_sha256
            && selectedReportSha256 === selection.selected.report_sha256
            && selectedFourGroupSha256 === selection.selected.four_group_report_sha256
            && JSON.stringify(rejectionReport) === JSON.stringify(expected);
        } catch { rejectionVerified = false; }
      }
      const test11Lock = await readOptionalJson(path.join(mdpoDir, 'test11_lock.json'));
      const test11Report = await readOptionalJson(path.join(mdpoDir, 'test11_final_report.json'));
      const sweepLedger = await readOptionalJson(path.join(mdpoDir, 'sweep', 'sweep_ledger.json'));
      const val11SweepLedger = await readOptionalJson(path.join(mdpoDir, 'val11_sweep_ledger.json'));
      const ablationTrainingLedger = await readOptionalJson(path.join(mdpoDir, 'ablations', 'ablation_ledger.json'));
      const ablationReport = await readOptionalJson(path.join(mdpoDir, 'ablations', 'ablation_report.json'));
      const deployment = assessMdpoDeployment({ active, activeSha256: activeHash, gate, selection, activationReport, preferenceSha256: preferenceHash,
        selectedCandidateSha256, selectedReportSha256, selectedFourGroupSha256 });
      const activeMdpo = deployment.active;
      return sendJson(res, 200, {
        version: 'v10_mdpo_experiment_status_v1', generated_at: new Date().toISOString(),
        active_model: { version: active.version, status: active.status, sha256: activeHash, source: activeMdpo ? 'v10-MDPO validated and atomically activated' : active.version === 'layout_model_v10_mdpo' ? 'v10-MDPO present but deployment evidence verification failed' : 'original v10 active', mdpo_activated: activeMdpo },
        historical_reward_model: { present: Boolean(preferenceBytes), sha256: preferenceHash, preserved_for_activation: activeMdpo ? true : null },
        historical_reward_model_preserved: activeMdpo ? true : Boolean(preferenceBytes),
        architecture: { frozen_reference: dataset?.reference_model_sha256 || null, policy: 'anchor-local 6D fixed-diagonal Gaussian', lora_modules: ['GNN layer 2', 'Transformer Q/K/V/O FFN', 'Human-style MoE router + four experts', 'MoE 64-32-6 decoder biases'], qwen_inference_input: false, dimensions: MDPO_DIMENSIONS, deterministic_safety_first: true },
        dataset: { ...audit, collected_samples: dataset ? [...new Set(dataset.pairs.map((pair) => `${pair.category}/${pair.sample_id}`))].sort() : [], sample_pair_counts: dataset ? Object.fromEntries([...dataset.pairs.reduce((map, pair) => { const key = `${pair.category}/${pair.sample_id}`; map.set(key, (map.get(key) || 0) + 1); return map; }, new Map())]) : {}, complete: Boolean(audit && audit.sample_count === 33 && audit.pair_count >= 198 && audit.pair_count <= 396), required_samples: 33, required_pairs: [198, 396], reference_model_sha256: dataset?.reference_model_sha256 || null },
        training: trained ? { status: trained.status, source: selectedCandidateFile ? 'complete_81_grid_val11_selection' : 'single_diagnostic_training', selected_configuration: selection?.selected?.id || null, candidate_file: selection?.selected?.candidate_file || null, best_epoch: trained.mdpo?.best_epoch, trainable_parameters: trained.mdpo?.trainable_parameters, history: trained.mdpo?.history, preference_holdout: trained.mdpo?.preference_holdout || null, train_unified_metrics: trained.mdpo?.train_unified_metrics || null, weight_update_evidence: trained.mdpo?.weight_update_evidence || null, checkpoint_count: checkpoint?.checkpoints?.length || 0, best_checkpoint: checkpoint?.best?.checkpoint || null } : { status: 'awaiting_complete_train33', source: null, best_epoch: null, history: [], checkpoint_count: 0 },
        hyperparameter_selection: selection,
        val11_rejection: rejectionReport ? { status: rejectionReport.status, verified: rejectionVerified,
          selected_candidate_id: rejectionReport.selected_candidate_id, selected_status: rejectionVerified ? 'diagnostic_only' : 'unverified',
          candidate_count: rejectionReport.candidate_count, selected_violations: rejectionVerified ? rejectionReport.rejected_candidates?.[0]?.violations || [] : [] } : null,
        sweep: { status: sweepLedger ? 'training_in_progress_or_complete' : 'not_started', required_configurations: 81,
          trained: (sweepLedger?.configurations || []).filter((row) => row.status === 'trained').length,
          evaluated: (val11SweepLedger?.configurations || []).filter((row) => row.status === 'evaluated').length,
          configurations: sweepLedger?.configurations || [] },
        ablations: { status: ablationReport?.status || (ablationTrainingLedger ? 'training_in_progress_or_pending_val11' : 'not_started'),
          trained: (ablationTrainingLedger?.configurations || []).filter((row) => row.status === 'trained').length,
          report: ablationReport },
        val11_gate: gate || { status: 'not_evaluated', accepted: false, deployment_status: 'diagnostic_only', violations: ['missing_full_val11_same_protocol_seven_dimension_visual_and_geometry_evidence'] },
        activation: activationReport ? { ...activationReport, verification: deployment } : { status: 'not_activated', reason: 'complete_authoritative_val11_activation_report_missing', verification: deployment },
        four_group_ablation: selectedFourGroupBytes ? JSON.parse(selectedFourGroupBytes.toString('utf8')) : null,
        test11: test11Report || (test11Lock ? { status: test11Lock.status, lock: test11Lock, final_report: null } : { status: 'not_locked_before_successful_activation', lock: null, final_report: null })
      });
    }
    if (url.pathname === '/api/mdpo-reference' && req.method === 'GET') {
      const bytes = await fs.readFile(layoutModelFile);
      const model = JSON.parse(bytes.toString('utf8'));
      return sendJson(res, 200, { version: model.version, reference_model_sha256: createHash('sha256').update(bytes).digest('hex'), active: true });
    }
    if (url.pathname === '/api/mdpo-train-candidate-summaries' && req.method === 'GET') {
      const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
      const sample = manifest.samples.find((item) => item.category === url.searchParams.get('category') && item.sample_id === url.searchParams.get('sample_id'));
      if (sample?.split !== 'train') return sendJson(res, 400, { error: 'MDPO summaries are train33 only' });
      const raw = await fs.readFile(mdpoCandidateFile, 'utf8').catch(() => '');
      const candidates = raw.split(/\r?\n/).filter(Boolean).map(JSON.parse)
        .filter((row) => row.category === sample.category && String(row.sample_id) === String(sample.sample_id) && strictMdpoSafety(row.safety?.metrics).eligible)
        .map((row) => ({ candidate_id: row.candidate_id, seed: row.seed, llm_scores: row.scores, metrics: safetySnapshot(row.safety.metrics), persisted: true }));
      return sendJson(res, 200, { sample: `${sample.category}/${sample.sample_id}`, candidates });
    }
    if (url.pathname === '/api/model-config' && req.method === 'GET') {
      const aligned = url.searchParams.get('source') === 'latest_aligned' ? await loadLatestCompletedAlignedModel({ root, experimentsDir }) : null;
      const active = aligned?.model || await getLayoutModel();
      const preference = await getPreferenceModel();
      return sendJson(res, 200, {
        display_source: aligned ? 'latest_completed_aligned_candidate' : 'active_deployed_model',
        aligned_candidate: aligned ? { id: aligned.id, file: aligned.file, sha256: aligned.sha256, completed_at: aligned.completed_at, best_epoch: aligned.best_epoch, proxy_val: aligned.proxy_val, hyperparameters: aligned.hyperparameters, deployment_status: aligned.deployment_status, formally_deployed: false } : null,
        model: active ? {
          version: active.version,
          status: active.status,
          architecture: active.architecture,
          inference: active.inference,
          validation_gate: active.validation_gate,
          training: {
            best_epoch: active.mdpo?.best_epoch ?? active.training?.best_epoch ?? null,
            train_loss: active.training?.train_loss ?? null,
            val_loss: active.mdpo?.best_val?.total ?? active.training?.val_loss ?? null,
            test_loss: active.training?.test_loss ?? null,
            metrics: active.training?.metrics ?? null,
            view_metrics: active.training?.view_metrics ?? null,
            routing: active.training?.routing ?? null,
            parameter_updates: active.training?.parameter_updates ?? null,
            functional_evidence: active.training?.functional_evidence ?? null,
            fixed_label_contract_validated: active.training?.fixed_label_contract_validated ?? null,
            visual_branch_used_in_generation: active.training?.visual_branch_used_in_generation ?? null,
            manual_style_loss_used: active.training?.manual_style_loss_used ?? null
          }
        } : { status: 'not_loaded' },
        preference_model: preference ? {
          status: preference.status,
          architecture: preference.architecture,
          scoring_protocol: preference.scoring_protocol,
          rerank_policy: preference.rerank_policy,
          validation_gate: preference.validation_gate,
          training: {
            run_id_filter: preference.training?.run_id_filter ?? null,
            examples: preference.training?.examples ?? null,
            source_counts: preference.training?.source_counts ?? null
          }
        } : { status: 'not_loaded' },
        runtime: { default_view_policy: 'binocular', default_group_policy: 'all', default_size_policy: 'relative', default_optimizer: 'annealing', seed: 17, iterations: 180, five_views: ['main', 'right', 'left', 'up', 'down'] }
      });
    }
    if (url.pathname === '/api/scoring-config' && req.method === 'GET') return sendJson(res, 200, safeScoringConfig());
    if (url.pathname === '/api/scoring-config' && req.method === 'POST') {
      try {
        const config = validateScoringConfig(await readJsonBody(req));
        const connectionChanged = config.apiUrl !== scoringConfig.apiUrl || config.model !== scoringConfig.model;
        scoringConfig = config;
        if (connectionChanged) scoringConnectionTest = null;
        return sendJson(res, 200, safeScoringConfig());
      } catch (error) { return sendJson(res, error.statusCode || 400, { error: error.message, code: 'SCORING_CONFIG_INVALID' }); }
    }
    if (url.pathname === '/api/scoring-test' && req.method === 'POST') {
      try {
        const result = await testScoringConnection();
        await appendScoringConnectionAudit({ result: 'passed', code: 'SCORING_CONNECTION_TEST_PASSED', status: 200, responseId: result.response_id });
        return sendJson(res, 200, result);
      } catch (error) {
        const code = error.code || 'SCORING_CONNECTION_TEST_FAILED';
        const status = error.statusCode || 500;
        const classification = code === 'SCORING_RESPONSE_INVALID' ? 'structured_response_failed' : code === 'VISUALS_MISSING' || code === 'SCORING_PROBE_IMAGES_MISSING' ? 'six_image_request_invalid' : 'local_ollama_unavailable';
        await appendScoringConnectionAudit({ result: classification, code, status });
        return sendJson(res, status, { error: error.message, code });
      }
    }
    if (url.pathname === '/api/generate' && req.method === 'POST') {
      try { return sendJson(res, 200, await generateLayout(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message || '标签生成失败', code: error.code || 'GENERATION_FAILED' }); }
    }
    if (url.pathname === '/api/train' && req.method === 'POST') {
      const trainOptions = await readJsonBody(req);
      const result = await runPreferenceTraining(trainOptions);
      if (!result.ok) return sendJson(res, result.status, { error: result.stderr || result.stdout || '偏好训练失败', code: 'PREFERENCE_TRAINING_FAILED', output: result });
      const validation = await runPreferenceValidationGate(trainOptions);
      if (!validation.ok) return sendJson(res, validation.status, { error: validation.stderr || validation.stdout || '偏好模型验证门控失败', code: 'PREFERENCE_VALIDATION_FAILED', output: result, validation });
      const trained = JSON.parse(await fs.readFile(preferenceCandidateFile, 'utf8'));
      const selection = JSON.parse(await fs.readFile(preferenceSelectionFile, 'utf8'));
      if (selection.activated) preferenceModelCache = { mtimeMs: -1, model: null };
      return sendJson(res, 200, { message: selection.activated ? '候选偏好模型通过门控并已激活' : trained.validation_gate?.visual_activation_pending ? '几何安全门控已通过，等待真实 val11 六图五维评分后决定是否激活' : '候选偏好模型未通过几何安全门控；旧活动模型保持不变', output: result.stdout, validation_output: validation.stdout, candidate_model_file: 'experiments/preference_model_candidate.json', model_file: selection.activated ? 'experiments/preference_model.json' : null, selection_file: 'experiments/preference_validation_selection.json', activated: selection.activated, validation_gate: trained.validation_gate, training: trained.training, timing: { train_ms: result.elapsed_ms, validation_ms: validation.elapsed_ms, elapsed_ms: result.elapsed_ms + validation.elapsed_ms } });
    }
    if (url.pathname === '/api/evaluate-preference-test' && req.method === 'POST') {
      const testOptions = await readJsonBody(req);
      try {
        if (testOptions?.force || testOptions?.runId) throw Object.assign(new Error('force fresh test'), { code: 'FORCE_TEST' });
        const report = JSON.parse(await fs.readFile(preferenceTestReportFile, 'utf8'));
        return sendJson(res, 200, { message: 'test11 偏好重排对照完成', report_file: 'experiments/preference_test11_report.json', summary: report.summary, test_samples: report.test_samples, candidate_activated: report.candidate_activated, validation_status: report.candidate_validation_status, elapsed_ms: 0 });
      } catch {
        let selected = null;
        try { if (testOptions?.runId) selected = await selectFinalLLMCheckpoint(validatedRunId(testOptions.runId)); }
        catch (error) { return sendJson(res, error.statusCode || 409, { error: error.message, code: 'PREFERENCE_CHECKPOINT_SELECTION_FAILED' }); }
        const result = await runPreferenceTestEvaluation({ iterations: testOptions?.iterations, limit: testOptions?.limit, ...selected });
        if (!result.ok) return sendJson(res, result.status, { error: result.stderr || result.stdout || 'test11 偏好对照失败', code: 'PREFERENCE_TEST_FAILED' });
        const report = JSON.parse(await fs.readFile(preferenceTestReportFile, 'utf8'));
        return sendJson(res, 200, { message: 'test11 偏好重排对照完成', report_file: 'experiments/preference_test11_report.json', summary: report.summary, test_samples: report.test_samples, candidate_activated: report.candidate_activated, selected_round: report.selected_round, selection_policy: report.selection_policy, elapsed_ms: result.elapsed_ms });
      }
    }
    if (url.pathname === '/api/llm-checkpoint' && req.method === 'POST') {
      try { return sendJson(res, 200, await writeLLMPreferenceCheckpoint(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'LLM_CHECKPOINT_FAILED' }); }
    }
    if (url.pathname === '/api/llm-visual-validation/finalize' && req.method === 'POST') {
      try { return sendJson(res, 200, await finalizeVisualValidation(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'LLM_VISUAL_VAL_FINALIZE_FAILED' }); }
    }
    if (url.pathname === '/api/llm-run-report' && req.method === 'POST') {
      try { return sendJson(res, 200, await writeLLMPreferenceRunReport(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'LLM_RUN_REPORT_FAILED' }); }
    }
    if (url.pathname === '/api/train-layout' && req.method === 'POST') {
      const result = await runLayoutTraining(await readJsonBody(req));
      if (!result.ok) return sendJson(res, result.status, { error: result.stderr || result.stdout || '布局模型训练失败', code: 'LAYOUT_TRAINING_FAILED', output: result });
      const validation = await runLayoutValidationGate();
      if (!validation.ok) return sendJson(res, validation.status, { error: validation.stderr || validation.stdout || '验证集五视角门控失败', code: 'LAYOUT_VALIDATION_FAILED', output: result, validation });
      const activation = await runScript('activate-v10-anchor-frame-model.mjs');
      if (!activation.ok) return sendJson(res, activation.status, { error: activation.stderr || activation.stdout || 'v10 模型激活失败', code: 'LAYOUT_MODEL_ACTIVATION_FAILED', output: result, validation, activation });
      const activated = JSON.parse(await fs.readFile(layoutModelFile, 'utf8'));
      layoutModelCache = { mtimeMs: -1, model: null };
      return sendJson(res, 200, { message: 'v10 锚点局部坐标异构图模型训练、val 诊断并激活完成', output: result.stdout, validation: validation.stdout, activation: activation.stdout, candidate_model_file: 'experiments/layout_model_v10_anchor_frame_candidate.json', model_file: 'experiments/layout_model.json', model_version: activated.version, architecture: activated.architecture, validation_gate: activated.validation_gate, timing: { train_ms: result.elapsed_ms, validation_ms: validation.elapsed_ms, activation_ms: activation.elapsed_ms, elapsed_ms: result.elapsed_ms + validation.elapsed_ms + activation.elapsed_ms } });
    }
    if (url.pathname === '/api/evaluations' && req.method === 'GET') {
      try {
        const lines = (await fs.readFile(evaluationLog, 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
        return sendJson(res, 200, { evaluations: lines });
      } catch { return sendJson(res, 200, { evaluations: [] }); }
    }
    if (url.pathname === '/api/evaluations' && req.method === 'POST') {
      const evaluation = await readJsonBody(req);
      const record = { id: `eval_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, created_at: new Date().toISOString(), ...evaluation };
      await fs.mkdir(experimentsDir, { recursive: true });
      await fs.appendFile(evaluationLog, `${JSON.stringify(record)}\n`, 'utf8');
      return sendJson(res, 201, record);
    }
    if (url.pathname === '/api/preferences' && req.method === 'GET') {
      try { const lines = (await fs.readFile(preferenceLog, 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)); return sendJson(res, 200, { preferences: lines }); }
      catch { return sendJson(res, 200, { preferences: [] }); }
    }
    if (url.pathname === '/api/preferences' && req.method === 'POST') {
      const preference = await readJsonBody(req);
      if (!preference?.chosen || !preference?.rejected) return sendJson(res, 400, { error: 'preference 必须包含 chosen 和 rejected' });
      const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
      const sample = preference.sample || preference.chosen.sample;
      const manifestSample = manifest.samples.find((item) => item.category === sample?.category && item.sample_id === String(sample?.sample_id));
      if (manifestSample?.split !== 'train') return sendJson(res, 400, { error: '偏好训练数据只允许 manifest 中的 train 样本，val/test 只可评估', code: 'PREFERENCE_SPLIT_FORBIDDEN' });
      const sameSample = (candidate) => candidate?.sample?.category === sample.category && String(candidate?.sample?.sample_id) === String(sample.sample_id);
      if (!sameSample(preference.chosen) || !sameSample(preference.rejected)) return sendJson(res, 400, { error: 'chosen/rejected 必须来自同一个 train 样本' });
      preference.split = preference.chosen.split = preference.rejected.split = 'train';
      const annotation = JSON.parse(await fs.readFile(path.join(root, manifestSample.target.annotation_json), 'utf8'));
      const fixed = annotationsToLabels(annotation);
      try { validateFixedLabelContract(fixed, preference.chosen.labels, 'preference/chosen'); validateFixedLabelContract(fixed, preference.rejected.labels, 'preference/rejected'); }
      catch (error) { return sendJson(res, 400, { error: error.message, code: 'PREFERENCE_LABEL_MISMATCH' }); }
      if (preference.type === 'llm_pairwise_preference') {
        if (!/^[a-zA-Z0-9_-]{8,80}$/.test(String(preference.run_id || '')) || preference.chosen.run_id !== preference.run_id || preference.rejected.run_id !== preference.run_id) return sendJson(res, 400, { error: 'LLM 偏好及 chosen/rejected 必须携带同一个有效 run_id', code: 'LLM_RUN_ID_MISMATCH' });
        const chosenReceipt = verifiedLLMScore(preference.chosen);
        const rejectedReceipt = verifiedLLMScore(preference.rejected);
        if (!chosenReceipt || !rejectedReceipt || preference.chosen.llm_score_receipt === preference.rejected.llm_score_receipt) return sendJson(res, 400, { error: 'LLM 偏好必须来自本次服务实际评分的两个不同候选，且布局/评分不得被修改', code: 'LLM_SCORE_RECEIPT_INVALID' });
        if (chosenReceipt.run_id !== preference.run_id || rejectedReceipt.run_id !== preference.run_id) return sendJson(res, 400, { error: '评分凭据不属于当前 LLM 运行批次', code: 'LLM_SCORE_RUN_MISMATCH' });
        if (chosenReceipt.model !== rejectedReceipt.model) return sendJson(res, 400, { error: '同一偏好对的两个候选必须使用相同评分模型', code: 'LLM_SCORER_MISMATCH' });
        const chosenAbsoluteSafety = assessSafetyEligibility(preference.chosen.metrics, preference.chosen.metrics);
        const rejectedAbsoluteSafety = assessSafetyEligibility(preference.rejected.metrics, preference.rejected.metrics);
        if (!chosenAbsoluteSafety.eligible || !rejectedAbsoluteSafety.eligible) return sendJson(res, 400, { error: 'LLM 偏好候选必须在主视角及所有辅助视角保持零引导线交叉', code: 'LLM_LEADER_CROSSING_HARD_GATE_FAILED', violations: { chosen: chosenAbsoluteSafety.violations, rejected: rejectedAbsoluteSafety.violations } });
        const safer = compareSafetyReference(preference.chosen.metrics, preference.rejected.metrics) <= 0 ? preference.chosen : preference.rejected;
        const other = safer === preference.chosen ? preference.rejected : preference.chosen;
        if (!assessSafetyEligibility(other.metrics, safer.metrics).eligible) return sendJson(res, 400, { error: 'LLM 美学偏好只能从通过确定性几何安全门控的候选对学习', code: 'LLM_PREFERENCE_SAFETY_GATE_FAILED' });
        if (preferenceComposite(preference.chosen.llm_scores) - preferenceComposite(preference.rejected.llm_scores) < 0.01) return sendJson(res, 400, { error: 'chosen 的美学综合分必须高于 rejected', code: 'LLM_AESTHETIC_RANK_INVALID' });
        preference.scoring_evidence = { chosen_model: chosenReceipt.model, rejected_model: rejectedReceipt.model, chosen_receipt_id: chosenReceipt.id, rejected_receipt_id: rejectedReceipt.id, chosen_response_id: chosenReceipt.response_id, rejected_response_id: rejectedReceipt.response_id, issued_at: { chosen: new Date(chosenReceipt.issued_at).toISOString(), rejected: new Date(rejectedReceipt.issued_at).toISOString() } };
      }
      const record = { id: `pref_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, created_at: new Date().toISOString(), ...preference };
      await fs.mkdir(experimentsDir, { recursive: true });
      await fs.appendFile(preferenceLog, `${JSON.stringify(record)}\n`, 'utf8');
      return sendJson(res, 201, record);
    }
    if (url.pathname === '/api/score' && req.method === 'POST') {
      const payload = await readJsonBody(req);
      try {
        if (payload?.type === 'mdpo_train_candidate') {
          const manifest = JSON.parse(await fs.readFile(path.join(experimentsDir, 'dataset_manifest.json'), 'utf8'));
          const sample = manifest.samples.find((item) => item.category === payload.sample?.category && item.sample_id === String(payload.sample?.sample_id));
          if (sample?.split !== 'train' || payload.split !== 'train') throw Object.assign(new Error('MDPO Qwen training labels are train33 only'), { statusCode: 400 });
          if (!strictMdpoSafety(payload.metrics).eligible) throw Object.assign(new Error('MDPO strict geometry-unsafe candidate cannot reach Qwen'), { statusCode: 400, code: 'MDPO_UNSAFE_PRECHECK' });
          hashMdpoViews(payload.visuals);
        }
        if (payload?.type === 'mdpo_val11_checkpoint' && (!verifiedMdpoValidationGeneration(payload) || !verifyVisualValidationImages(payload))) throw Object.assign(new Error('MDPO val11 frozen checkpoint receipt/layout/six images invalid'), { statusCode: 409, code: 'MDPO_VAL11_GENERATION_RECEIPT_INVALID' });
        if (payload?.type === 'mdpo_test11_checkpoint' && (!verifiedMdpoTest11Generation(payload) || !verifyVisualValidationImages(payload))) throw Object.assign(new Error('MDPO test11 locked model receipt/layout/six images invalid'), { statusCode: 409, code: 'MDPO_TEST11_GENERATION_RECEIPT_INVALID' });
        if (payload?.type === 'llm_validation_checkpoint' && (!verifiedValidationGeneration(payload) || !verifyVisualValidationImages(payload))) throw Object.assign(new Error('冻结检查点 val 布局凭据无效、布局已修改或缺少六图；未调用本地 Qwen 评分'), { statusCode: 409, code: 'LLM_VISUAL_VAL_GENERATION_RECEIPT_INVALID' });
        const scored = await scoreWithLocalQwen(payload);
        const visualValidation = await persistVisualValidationScore(payload, scored);
        const mdpoValidation = await persistMdpoVisualValidationScore(payload, scored);
        const mdpoTest11 = await persistMdpoTest11Score(payload, scored);
        return sendJson(res, 200, { ...scored, score_receipt: issueScoringReceipt(payload, scored), visual_validation_record: visualValidation ? visualValidation.file : mdpoValidation ? mdpoValidation.file : mdpoTest11 ? mdpoTest11.file : null });
      } catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'SCORING_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-val11-finalize' && req.method === 'POST') {
      try { return sendJson(res, 200, await finalizeMdpoVal11(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'MDPO_VAL11_FINALIZE_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-val11-progress' && req.method === 'POST') {
      try { return sendJson(res, 200, await mdpoVal11Progress(await readJsonBody(req))); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'MDPO_VAL11_PROGRESS_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-test11-progress' && req.method === 'POST') {
      try { return sendJson(res, 200, await mdpoTest11Progress()); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'MDPO_TEST11_PROGRESS_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-test11-finalize' && req.method === 'POST') {
      try { return sendJson(res, 200, await finalizeMdpoTest11()); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: error.code || 'MDPO_TEST11_FINALIZE_FAILED' }); }
    }
    if (url.pathname === '/api/mdpo-train-sample' && req.method === 'POST') {
      const payload = await readJsonBody(req);
      const task = mdpoCollectionQueue.then(() => collectMdpoTrainSample(payload));
      mdpoCollectionQueue = task.catch(() => {});
      try { return sendJson(res, 201, await task); }
      catch (error) { return sendJson(res, 400, { error: error.message, code: 'MDPO_TRAIN_SAMPLE_INVALID' }); }
    }
    if (url.pathname === '/api/moe-dashboard' && req.method === 'GET') {
      try { return sendJson(res, 200, await buildMoeDashboardData()); }
      catch (error) { return sendJson(res, error.statusCode || 500, { error: error.message, code: 'MOE_DASHBOARD_FAILED' }); }
    }
    if (url.pathname === '/api/moe-dashboard/stream' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      const snapshot = await buildMoeDashboardData();
      const serialized = JSON.stringify(snapshot);
      moeDashboardSnapshot = serialized;
      res.write(`data: ${serialized}\n\n`);
      moeDashboardClients.add(res);
      req.on('close', () => moeDashboardClients.delete(res));
      return;
    }
    if (url.pathname.startsWith('/experiments/')) {
      const target = safePath(root, decodeURIComponent(url.pathname.slice(1)));
      if (target) return sendFile(res, target);
    }
    if (url.pathname.startsWith('/data/')) {
      const target = safePath(root, decodeURIComponent(url.pathname.slice(1)));
      if (target) return sendFile(res, target);
    }
    if (url.pathname === '/favicon.ico') { res.writeHead(204, { 'Cache-Control': 'public, max-age=86400' }); return res.end(); }
    const requested = url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/+/, '');
    const target = safePath(publicDir, decodeURIComponent(requested));
    if (target) return sendFile(res, target);
    res.writeHead(404); res.end('Not found');
  } catch (error) {
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(error instanceof Error ? error.message : 'Internal error');
  }
});

// Persisted queued requests resume after a server restart, without an open browser tab.
setInterval(() => { dispatchQueuedMdpoUiTraining().catch(console.error); }, 30_000).unref();
setImmediate(() => { dispatchQueuedMdpoUiTraining().catch(console.error); });
setInterval(() => { broadcastMoeDashboardUpdates().catch(console.error); }, 3000).unref();
server.listen(port, host, () => {
  console.log(`OBJ 3D Label Studio running at http://${host}:${port}`);
});
