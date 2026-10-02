import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { predictPreference } from '../lib/preference-model.mjs';
import { selectSafetyConstrainedTrial } from '../lib/preference-policy.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildCvFeatures } from '../lib/cv-feature-encoder.mjs';
import { buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const options = {
    candidate: path.join(root, 'experiments', 'preference_model_candidate.json'),
    active: path.join(root, 'experiments', 'preference_model.json'),
    comparison: null,
    manifest: path.join(root, 'experiments', 'dataset_manifest.json'),
    layoutModel: path.join(root, 'experiments', 'layout_model.json'),
    leaderLengthPrior: path.join(root, 'experiments', 'manual_leader_length_prior.json'),
    output: path.join(root, 'experiments', 'preference_validation_selection.json'),
    testOutput: path.join(root, 'experiments', 'preference_test11_report.json'),
    iterations: 180,
    limit: 0,
    skipTest: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const readJson = async (file) => JSON.parse(await fs.readFile(path.resolve(String(file)), 'utf8'));
const candidate = await readJson(options.candidate);
const manifest = await readJson(options.manifest);
const layoutModel = await readJson(options.layoutModel);
const leaderLengthPrior = await readJson(options.leaderLengthPrior);
let previousActive = null;
try { previousActive = await readJson(options.comparison || options.active); } catch (error) { if (error.code !== 'ENOENT') throw error; }

async function buildTrials(split) {
  const rows = [];
  const samples = manifest.samples.filter((item) => item.split === split).slice(0, Number(options.limit) || undefined);
  for (const sample of samples) {
    const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
    const clean = cleanObj(raw), bounds = boundsFromObj(clean.text);
    const manual = annotationsToLabels(await readJson(path.join(root, sample.target.annotation_json)));
    const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, layoutModel);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/${split}/candidates`);
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
    for (let trial = 0; trial < 4; trial += 1) {
      const seed = 17 + trial;
      const optimizerOptions = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true, iterations: Number(options.iterations), seed, depthGrids, category: sample.category, leaderLengthPrior };
      const labels = optimizeLabels(styled, bounds, optimizerOptions);
      validateFixedLabelContract(manual, labels, `${sample.category}/${sample.sample_id}/${split}/seed${seed}`);
      const metrics = evaluateLayout(labels, bounds, { ...optimizerOptions, manualReference: manual, geometry });
      const viewSafety = evaluateViewConditionedLayout(labels, bounds, depthGrids, {
        viewWeights: layoutModel.architecture?.view_loss?.weights,
        worstViewWeight: layoutModel.architecture?.view_loss?.worst_view_weight,
        cvarViewWeight: layoutModel.architecture?.view_loss?.cvar_weight,
        stereoWeight: layoutModel.architecture?.view_loss?.stereo_weight,
        textClarityWeight: layoutModel.architecture?.view_loss?.text_clarity_weight,
        leaderCrossingWeight: layoutModel.architecture?.view_loss?.leader_crossing_weight
      });
      Object.assign(metrics, {
        weighted_leader_crossing_risk: viewSafety.weighted_leader_crossing_risk,
        worst_view_leader_crossing_risk: viewSafety.worst_view_leader_crossing_risk,
        cvar_view_leader_crossing_risk: viewSafety.cvar_view_leader_crossing_risk,
        worst_view_leader_crossing_count: viewSafety.worst_view_leader_crossing_count
      });
      trials.push({ seed, metrics });
    }
    rows.push({ category: sample.category, sample_id: sample.sample_id, split, trials });
    console.log(`${split} ${sample.category}/${sample.sample_id}: four candidates ready`);
  }
  return rows;
}

function selectTrial(row, model) {
  if (!model) return row.trials[0];
  const selection = selectSafetyConstrainedTrial(row.trials, model, predictPreference);
  return selection?.selected || null;
}

const fields = ['aesthetic_composition_harmony', 'intrinsic_style_balance', 'intrinsic_size_consistency', 'aesthetic_spacing_consistency', 'aesthetic_text_size_fit', 'multidimensional_quality_score', 'objective_score', 'olr', 'lcd', 'readability', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'manual_center_distance_norm', 'manual_style_distance', 'viewport_overflow_ratio', 'multi_view_worst_olr', 'multi_view_worst_overflow', 'leader_crossings', 'worst_view_leader_crossing_count', 'weighted_leader_crossing_risk', 'worst_view_leader_crossing_risk', 'cvar_view_leader_crossing_risk'];
function summarize(rows, model) {
  const selected = rows.map((row) => selectTrial(row, model));
  if (selected.some((item) => !item)) return null;
  return Object.fromEntries(fields.map((field) => { const values=selected.map(item=>item.metrics[field] ?? item.metrics.manual_similarity?.[field]).filter(value=>value!==null&&value!==undefined&&Number.isFinite(Number(value))).map(Number); return [field, values.length ? Number((values.reduce((sum,value)=>sum+value,0)/values.length).toFixed(6)) : null]; }));
}

function detailedRows(rows, model) {
  return rows.map((row) => {
    const baseline = row.trials[0];
    const preferred = selectTrial(row, model);
    if (!preferred) return { category: row.category, sample_id: row.sample_id, split: row.split, safety_gate: 'no_zero_crossing_candidate' };
    return { category: row.category, sample_id: row.sample_id, split: row.split, baseline_seed: baseline.seed, selected_seed: preferred.seed, baseline: baseline.metrics, preferred: preferred.metrics, baseline_preference_score: model ? predictPreference(model, baseline.metrics)?.score ?? null : null, selected_preference_score: preferred.preference?.score ?? null, all_trial_seeds: row.trials.map((item) => item.seed) };
  });
}

const validationTrials = await buildTrials('val');
if (!validationTrials.length) throw new Error('val split 没有可用样本，不能门控偏好模型');
const validationBaseline = summarize(validationTrials, null);
const validationCandidate = summarize(validationTrials, candidate);
const validationPrevious = previousActive ? summarize(validationTrials, previousActive) : null;
const referenceComposition = Math.max(validationBaseline.aesthetic_composition_harmony, validationPrevious?.aesthetic_composition_harmony ?? -Infinity);
const constraints = {
  composition_harmony_proxy_change: validationCandidate ? validationCandidate.aesthetic_composition_harmony - referenceComposition : null,
  aesthetic_gain: validationCandidate ? validationCandidate.aesthetic_composition_harmony - referenceComposition : null,
  objective_relative_change: validationCandidate ? validationCandidate.objective_score / validationBaseline.objective_score - 1 : null,
  label_object_occlusion_change: validationCandidate ? validationCandidate.label_object_occlusion_ratio - validationBaseline.label_object_occlusion_ratio : null,
  depth_penetration_change: validationCandidate ? validationCandidate.object_penetration_ratio - validationBaseline.object_penetration_ratio : null,
  mesh_surface_intersection_change: validationCandidate ? validationCandidate.mesh_surface_intersection_ratio - validationBaseline.mesh_surface_intersection_ratio : null,
  worst_overflow_change: validationCandidate ? validationCandidate.multi_view_worst_overflow - validationBaseline.multi_view_worst_overflow : null,
  text_clarity_change: validationCandidate ? validationCandidate.text_clarity - validationBaseline.text_clarity : null,
  leader_crossings: validationCandidate?.leader_crossings ?? null,
  worst_view_leader_crossing_count: validationCandidate?.worst_view_leader_crossing_count ?? null
};
const geometryEligible = Boolean(validationCandidate) && constraints.objective_relative_change <= 0.01 && constraints.label_object_occlusion_change <= 0.01 && constraints.depth_penetration_change <= 0.005 && constraints.mesh_surface_intersection_change <= 0.005 && constraints.worst_overflow_change <= 0.01 && constraints.text_clarity_change >= -0.10 && constraints.leader_crossings === 0 && constraints.worst_view_leader_crossing_count === 0;
const requiresVisualGate = candidate.training?.source_filter === 'llm' || Boolean(candidate.training?.run_id_filter);
const legacyAestheticAccepted = constraints.composition_harmony_proxy_change >= 0.0005;
const accepted = geometryEligible && !requiresVisualGate && legacyAestheticAccepted;
candidate.validation_gate = {
  status: requiresVisualGate ? geometryEligible ? 'safety_eligible_pending_visual_val11' : 'rejected_safety' : accepted ? 'accepted' : 'rejected',
  criterion: requiresVisualGate ? 'deterministic val11 geometry safety eligibility first; activation is deferred until real val11 six-image Qwen five-dimension aesthetic composite gain >= 0.02 and composition_harmony change >= -0.01' : 'legacy non-LLM preference gate',
  policy: 'deterministic energy safety gate first; LLM aesthetic reward ranks only eligible candidates; real Qwen val11 visual evidence controls LLM reward activation',
  baseline: validationBaseline,
  previous_active: validationPrevious,
  candidate: validationCandidate,
  reference_composition_proxy: referenceComposition,
  geometry_safety_eligible: geometryEligible,
  visual_activation_pending: requiresVisualGate && geometryEligible,
  constraints,
  relative_change_vs_baseline: validationCandidate ? Number((validationCandidate.objective_score / validationBaseline.objective_score - 1).toFixed(6)) : null,
  relative_change_vs_previous_active: validationCandidate && validationPrevious ? Number((validationCandidate.objective_score / validationPrevious.objective_score - 1).toFixed(6)) : null,
  validation_samples: validationTrials.length,
  test_not_used_for_activation: true
};
await fs.writeFile(path.resolve(String(options.candidate)), JSON.stringify(candidate, null, 2) + '\n', 'utf8');

let backupFile = null;
if (accepted) {
  try {
    await fs.access(path.resolve(String(options.active)));
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    backupFile = path.join(path.dirname(path.resolve(String(options.active))), `preference_model_before_activation_${stamp}.json`);
    await fs.copyFile(path.resolve(String(options.active)), backupFile);
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await fs.copyFile(path.resolve(String(options.candidate)), path.resolve(String(options.active)));
}

const skipTest = options.skipTest === true || String(options.skipTest).toLowerCase() === 'true';
let testTrials=[];
let testBaseline=null,testCandidate=null,testPrevious=null,candidateRows=[],testSummary=null;
if(!skipTest){
 testTrials = await buildTrials('test');
 testBaseline = summarize(testTrials, null);
 testCandidate = summarize(testTrials, candidate);
 testPrevious = previousActive ? summarize(testTrials, previousActive) : null;
 candidateRows = detailedRows(testTrials, candidate);
 testSummary = Object.fromEntries(fields.map((field) => [field, { baseline: testBaseline[field], preferred: testCandidate?.[field] ?? null }]));
}
const testReport = !skipTest ? {
  generated_at: new Date().toISOString(),
  run_id: candidate.training?.run_id_filter || null,
  layout_version: layoutModel.version,
  preference_status: candidate.status,
  candidate_model_file: path.relative(root, path.resolve(String(options.candidate))).split(path.sep).join('/'),
  candidate_validation_status: candidate.validation_gate.status,
  candidate_activated: accepted,
  test_samples: testTrials.length,
  split_policy: 'validation decides activation; test inference is final confirmation only and never added to training or gating',
  protocol: 'fixed labels, seeds 17..20, four annealed candidates per sample; candidate reward reranks against seed17 baseline',
  summary: testSummary,
  comparison: { no_reward: testBaseline, previous_active_reward: testPrevious, candidate_reward: testCandidate },
  rows: candidateRows,
  interpretation: 'Candidate test result is reported even when validation rejects activation; rejected candidates never replace the active reward model.'
} : null;
if(testReport) await fs.writeFile(path.resolve(String(options.testOutput)), JSON.stringify(testReport, null, 2) + '\n', 'utf8');

const selection = {
  generated_at: new Date().toISOString(),
  run_id: candidate.training?.run_id_filter || null,
  candidate_model_file: path.relative(root, path.resolve(String(options.candidate))).split(path.sep).join('/'),
  active_model_file: path.relative(root, path.resolve(String(options.active))).split(path.sep).join('/'),
  comparison_model_file: path.relative(root, path.resolve(String(options.comparison || options.active))).split(path.sep).join('/'),
  activated: accepted,
  backup_file: backupFile ? path.relative(root, backupFile).split(path.sep).join('/') : null,
  validation: candidate.validation_gate,
  test_confirmation: skipTest ? null : { no_reward: testBaseline, previous_active_reward: testPrevious, candidate_reward: testCandidate },
  policy: { train_pairs_only: true, validation_controls_activation: true, test_not_used_for_activation: true, test_skipped_for_intermediate_checkpoint: skipTest, rejected_candidate_preserved: true }
};
await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(selection, null, 2) + '\n', 'utf8');
console.log(JSON.stringify({ activated: accepted, validation: candidate.validation_gate, test_confirmation: selection.test_confirmation }, null, 2));
