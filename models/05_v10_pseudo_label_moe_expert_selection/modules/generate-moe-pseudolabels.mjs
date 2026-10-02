import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { computeSpatialStyleMetrics, buildSpatialContext } from '../lib/spatial-style-features.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const leaderPriorFile = path.join(root, 'experiments', 'manual_leader_length_prior.json');
const defaultOutput = path.join(root, 'experiments', 'moe_unsupervised_pseudolabels_train.json');

function parseArgs(argv) {
  const options = { split: 'train', output: defaultOutput, iterations: 180, seed: 17, limit: 0 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  options.iterations = Number(options.iterations || 180);
  options.seed = Number(options.seed || 17);
  options.limit = Number(options.limit || 0);
  return options;
}
function repoPath(relativePath) { return path.join(root, String(relativePath).replaceAll('/', path.sep)); }
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function readText(file) { return fs.readFile(file, 'utf8'); }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function mean(values) { return values.length ? values.reduce((sum, value) => sum + finite(value), 0) / values.length : 0; }
function metricSummary(rows) {
  const keys = ['objective_score', 'multidimensional_quality_score', 'olr', 'lcd', 'viewport_overflow_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'text_clarity', 'mean_anchor_distance', 'leader_length_compliance_ratio'];
  return Object.fromEntries(keys.map((key) => [key, Number(mean(rows.map((row) => row.metrics[key])).toFixed(6))]));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifest = await readJson(manifestFile);
  const leaderPrior = await fs.readFile(leaderPriorFile, 'utf8').then(JSON.parse).catch(() => null);
  let samples = manifest.samples.filter((sample) => options.split === 'all' || sample.split === options.split);
  if (options.limit > 0) samples = samples.slice(0, options.limit);
  const rows = [];
  for (const [sampleIndex, sample] of samples.entries()) {
    const clean = cleanObj(await readText(repoPath(sample.input.source_obj)));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const annotation = await readJson(repoPath(sample.target.annotation_json));
    const manual = annotationsToLabels(annotation);
    const candidates = fixedCandidatesWithoutTargetLayout(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/moe-pseudolabel-input`);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const pseudo = optimizeLabels(candidates, bounds, {
      fixedLabels: true,
      groupPolicy: 'all',
      viewPolicy: 'binocular',
      optimizer: 'annealing',
      iterations: options.iterations,
      seed: options.seed + sampleIndex * 104729,
      depthGrids,
      category: sample.category,
      leaderLengthPrior: leaderPrior,
      geometry,
      layoutStyle: 'auto',
      styleView: 'main'
    });
    validateFixedLabelContract(manual, pseudo, `${sample.category}/${sample.sample_id}/moe-pseudolabel-output`);
    const metrics = evaluateLayout(pseudo, bounds, { fixedLabels: true, viewPolicy: 'binocular', depthGrids, category: sample.category, leaderLengthPrior: leaderPrior, manualReference: manual, geometry });
    const spatial = computeSpatialStyleMetrics(pseudo, bounds, buildSpatialContext(geometry, bounds, { gridSize: 20 }));
    rows.push({
      category: sample.category,
      sample_id: sample.sample_id,
      split: sample.split,
      style: pseudo[0]?.moe_style_selected || null,
      style_weights: pseudo[0]?.moe_style_routing || null,
      label_count: pseudo.length,
      metrics,
      spatial,
      labels: pseudo.map((label) => ({
        id: label.id,
        text: label.text,
        anchor: label.anchor,
        center: label.center,
        boxSize: label.boxSize,
        sourceObjs: label.sourceObjs || [],
        targetGroups: label.targetGroups || [],
        layout_style_expert: label.layout_style_expert,
        moe_style_selected: label.moe_style_selected,
        moe_style_routing: label.moe_style_routing,
        objective: label.objective
      }))
    });
  }
  const result = {
    version: 'moe_unsupervised_pseudolabels_v1',
    generated_at: new Date().toISOString(),
    source_manifest: path.relative(root, manifestFile).replaceAll('\\', '/'),
    split: options.split,
    iterations: options.iterations,
    seed: options.seed,
    policy: {
      input: 'manual_label_contract_without_target_center_or_box_size',
      initialization: 'moe_style_auto_spherical_rectangular_surround_from_clean_obj_contour',
      optimizer: 'five_view_simulated_annealing_energy',
      target_use: 'pseudo_labels_can_replace_manual_centers_for_v10_training_after_val_gate'
    },
    count: rows.length,
    summary: metricSummary(rows),
    style_counts: rows.reduce((acc, row) => { acc[row.style] = (acc[row.style] || 0) + 1; return acc; }, {}),
    rows
  };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(result, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, count: rows.length, output: path.relative(root, path.resolve(String(options.output))).replaceAll('\\', '/'), summary: result.summary, style_counts: result.style_counts }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
