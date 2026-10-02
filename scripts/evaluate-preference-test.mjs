import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { predictPreference } from '../lib/preference-model.mjs';
import { selectSafetyConstrainedTrial } from '../lib/preference-policy.mjs';
import { optimizeWithAdaptiveDirectionalGate } from '../lib/adaptive-directional-rerank.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildCvFeatures } from '../lib/cv-feature-encoder.mjs';
import { buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
function parseArgs(argv) {
  const options = { iterations: 180, limit: 0, output: null };
  for (let index = 0; index < argv.length; index += 1) if (argv[index].startsWith('--')) {
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}
const options = parseArgs(process.argv.slice(2));
const experiments = process.env.EXPERIMENTS_DIR ? path.resolve(process.env.EXPERIMENTS_DIR) : path.join(root, 'experiments');
const readJson = async (file) => JSON.parse(await fs.readFile(path.resolve(file), 'utf8'));
const manifest = await readJson(path.join(experiments, 'dataset_manifest.json'));
const layoutModel = await readJson(path.join(experiments, 'layout_model.json'));
const leaderLengthPrior = await readJson(path.join(experiments, 'manual_leader_length_prior.json'));
const selectedModelFile = options.model ? path.resolve(String(options.model)) : path.join(experiments, 'preference_model_candidate.json');
const preferenceBytes = await fs.readFile(selectedModelFile);
let preferenceModel;
try { preferenceModel = JSON.parse(preferenceBytes.toString('utf8')); }
catch { throw new Error('尚无本次运行的候选偏好模型，请先用 train 样本完成累计偏好训练'); }
if (options.runId && preferenceModel.training?.run_id_filter !== String(options.runId)) throw new Error('最终 test 检查点与本次 run_id 不一致');
const rows = [];
for (const sample of manifest.samples.filter((item) => item.split === 'test').slice(0, Number(options.limit) || undefined)) {
  const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
  const clean = cleanObj(raw), bounds = boundsFromObj(clean.text);
  const manual = annotationsToLabels(await readJson(path.join(root, sample.target.annotation_json)));
  const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, layoutModel);
  validateFixedLabelContract(manual, candidates, sample.category + '/' + sample.sample_id);
  const geometry = parseObjTriangles(clean.text);
  const cvFeatures = await buildCvFeatures({ objText: clean.text, bounds });
  const spatialContext = buildSpatialContext(geometry, bounds, { gridSize: layoutModel.architecture?.spatial_grid?.grid_size || 20 });
  const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
  const styled = applyLayoutModel(candidates, bounds, layoutModel, {
    geometry,
    geometryFeature: cvFeatures.geometry,
    visualFeature: layoutModel.architecture?.generation_input === 'pure_3d' ? null : cvFeatures.visual,
    spatialContext
  }).labels;
  const trials = [];
  let modelOne = null;
  for (let trial = 0; trial < 4; trial += 1) {
    const seed = 17 + trial;
    const optimizerOptions = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true, iterations: Number(options.iterations) || 180, seed, depthGrids, category: sample.category, leaderLengthPrior };
    const directional = optimizeWithAdaptiveDirectionalGate(styled, bounds, optimizerOptions, geometry);
    validateFixedLabelContract(manual, directional.labels, sample.category + '/' + sample.sample_id + '/seed' + seed);
    const metrics = evaluateLayout(directional.labels, bounds, { ...optimizerOptions, manualReference: manual, geometry });
    const viewSafety = evaluateViewConditionedLayout(directional.labels, bounds, depthGrids, {
      viewWeights: layoutModel.architecture?.view_loss?.weights,
      worstViewWeight: layoutModel.architecture?.view_loss?.worst_view_weight,
      cvarViewWeight: layoutModel.architecture?.view_loss?.cvar_weight,
      stereoWeight: layoutModel.architecture?.view_loss?.stereo_weight,
      textClarityWeight: layoutModel.architecture?.view_loss?.text_clarity_weight,
      leaderCrossingWeight: layoutModel.architecture?.view_loss?.leader_crossing_weight
    });
    Object.assign(metrics, { weighted_leader_crossing_risk: viewSafety.weighted_leader_crossing_risk, worst_view_leader_crossing_risk: viewSafety.worst_view_leader_crossing_risk, cvar_view_leader_crossing_risk: viewSafety.cvar_view_leader_crossing_risk, worst_view_leader_crossing_count: viewSafety.worst_view_leader_crossing_count });
    const baselineMetrics = evaluateLayout(directional.baseline.labels, bounds, { ...optimizerOptions, manualReference: manual, geometry });
    if (trial === 0) modelOne = { seed, metrics: baselineMetrics };
    trials.push({ seed, metrics, preference: predictPreference(preferenceModel, metrics), directional_policy: directional.policy, directional_alternative_accepted: directional.accepted });
  }
  const baseline = modelOne;
  const safetySelection = selectSafetyConstrainedTrial(trials, preferenceModel, predictPreference);
  const preferred = safetySelection.selected;
  if (!preferred) throw new Error(`${sample.category}/${sample.sample_id} 没有通过所有视角零引导线交叉硬门控的 test 候选`);
  rows.push({ category: sample.category, sample_id: sample.sample_id, split: 'test', baseline_model: 'model_1_base_seed17', final_model: 'adaptive_directional_gate_plus_llm_aesthetic_reward', baseline_seed: baseline.seed, selected_seed: preferred.seed, baseline: baseline.metrics, preferred: preferred.metrics, baseline_preference_score: predictPreference(preferenceModel, baseline.metrics)?.score ?? null, selected_preference_score: preferred.preference?.score ?? null, directional_policy: preferred.directional_policy, directional_alternative_accepted: preferred.directional_alternative_accepted, safety_reference_seed: safetySelection.reference.seed, safety_eligible_seeds: safetySelection.assessed.filter((item) => item.safety.eligible).map((item) => item.seed), rejected_unsafe_seeds: safetySelection.assessed.filter((item) => !item.safety.eligible).map((item) => ({ seed: item.seed, violations: item.safety.violations })), all_trial_seeds: trials.map((item) => item.seed) });
  console.log(sample.category + '/' + sample.sample_id + ': selected seed ' + preferred.seed);
}
const fields = ['aesthetic_composition_harmony', 'intrinsic_style_balance', 'intrinsic_size_consistency', 'multidimensional_quality_score', 'objective_score', 'olr', 'lcd', 'readability', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'mean_anchor_distance', 'leader_length_compliance_ratio', 'leader_length_shortfall', 'directional_allocation_mismatch', 'directional_concentration_excess', 'directional_uniformity', 'manual_center_distance_norm', 'manual_style_distance', 'viewport_overflow_ratio', 'multi_view_worst_olr', 'multi_view_worst_overflow', 'leader_crossings', 'worst_view_leader_crossing_count', 'weighted_leader_crossing_risk', 'worst_view_leader_crossing_risk', 'cvar_view_leader_crossing_risk'];
function meanMetric(items, key) {
  const values = items.map((metrics) => metrics[key] ?? metrics.manual_similarity?.[key]).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  return values.length ? Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(6)) : null;
}
const summary = Object.fromEntries(fields.map((key) => [key, { baseline: meanMetric(rows.map((row) => row.baseline), key), preferred: meanMetric(rows.map((row) => row.preferred), key) }]));
const output = { generated_at: new Date().toISOString(), run_id: options.runId || preferenceModel.training?.run_id_filter || null, selected_round: options.round ? Number(options.round) : null, selection_policy: options.selectionPolicy || (options.runId ? preferenceModel.validation_gate?.status === 'accepted' ? 'best_accepted_val_aesthetic_proxy' : 'best_rejected_val_aesthetic_proxy_diagnostic_only' : 'manual_candidate_evaluation'), layout_version: layoutModel.version, preference_status: preferenceModel.status, candidate_validation_status: preferenceModel.validation_gate?.status || 'unknown', candidate_activated: preferenceModel.validation_gate?.status === 'accepted', candidate_model_file: path.relative(root, selectedModelFile).split(path.sep).join('/'), candidate_model_sha256: createHash('sha256').update(preferenceBytes).digest('hex').toUpperCase(), test_samples: rows.length, split_policy: 'validation selects the LLM checkpoint; adaptive directional weights/rule were selected on val before frozen test confirmation; test never enters training or gating', protocol: 'model 1 = fixed labels seed17 with directional 1.6/1.2; final model = per-seed adaptive directional safety gate, then four-candidate deterministic safety gate, then LLM aesthetic reward', summary, rows, interpretation: 'The final-model comparison adds a val-selected deterministic directional fallback before LLM aesthetic reranking. Manual target coordinates do not decide the fallback. Unsafe candidates cannot be selected by aesthetic reward.' };
const outputFile = options.output ? path.resolve(String(options.output)) : path.join(experiments, 'preference_test11_report.json');
await fs.writeFile(outputFile, JSON.stringify(output, null, 2) + String.fromCharCode(10));
console.log(JSON.stringify({ test_samples: rows.length, summary }, null, 2));
