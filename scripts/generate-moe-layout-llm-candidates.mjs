import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj } from './generate-artifacts.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, parseObjTriangles, projectLabelToView } from '../lib/layout-optimizer.mjs';
import { perturbMdpoCandidate, MDPO_PERTURBATION_MODES } from '../lib/mdpo-candidate-perturbation.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const pseudoFile = path.join(root, 'experiments', 'moe_unsupervised_pseudolabels_train.json');
const leaderPriorFile = path.join(root, 'experiments', 'manual_leader_length_prior.json');
const defaultOutput = path.join(root, 'experiments', 'moe_layout_llm_candidates.json');

function parseArgs(argv) {
  const options = { input: pseudoFile, output: defaultOutput, samples: 3, modes: '0,12,13,14,15,16', seed: 17 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  options.samples = Number(options.samples || 3);
  options.seed = Number(options.seed || 17);
  options.modes = String(options.modes || '0').split(',').map((value) => Number(value.trim())).filter(Number.isFinite);
  return options;
}
function repoPath(relativePath) { return path.join(root, String(relativePath).replaceAll('/', path.sep)); }
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function readText(file) { return fs.readFile(file, 'utf8'); }
function sampleKey(category, sampleId) { return `${category}/${sampleId}`; }
function viewFilesForSample(sample) {
  return Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, sample.views.files.find((file) => file.toLowerCase().includes(`-${view}.png`)) || null]));
}
function projectedView(labels, bounds, view) {
  return labels.map((label) => {
    const item = projectLabelToView(label, bounds, view);
    return {
      id: label.id,
      text: label.text,
      anchor: { x: Number(item.anchor.x.toFixed(6)), y: Number(item.anchor.y.toFixed(6)), depth: Number(item.anchor.depth.toFixed(6)) },
      center: { x: Number(item.center.x.toFixed(6)), y: Number(item.center.y.toFixed(6)), depth: Number(item.center.depth.toFixed(6)) },
      width: Number(item.width.toFixed(6)),
      height: Number(item.height.toFixed(6)),
      style: label.layout_style_expert || label.moe_style_selected || null,
      perturbation: label.mdpo_perturbation?.mode || 'baseline'
    };
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const [manifest, pseudo, leaderPrior] = await Promise.all([
    readJson(manifestFile), readJson(path.resolve(String(options.input))), fs.readFile(leaderPriorFile, 'utf8').then(JSON.parse).catch(() => null)
  ]);
  const manifestByKey = new Map(manifest.samples.map((sample) => [sampleKey(sample.category, sample.sample_id), sample]));
  const rows = pseudo.rows.slice(0, options.samples);
  const outputRows = [];
  for (const [sampleIndex, row] of rows.entries()) {
    const sample = manifestByKey.get(sampleKey(row.category, row.sample_id));
    if (!sample) throw new Error(`Missing manifest sample ${row.category}/${row.sample_id}`);
    const clean = cleanObj(await readText(repoPath(sample.input.source_obj)));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const viewFiles = viewFilesForSample(sample);
    const candidates = [];
    for (const mode of options.modes) {
      const perturb = perturbMdpoCandidate(row.labels, geometry, bounds, { mode, seed: options.seed + sampleIndex * 1009 });
      const labels = perturb.labels;
      const metrics = evaluateLayout(labels, bounds, { fixedLabels: true, viewPolicy: 'binocular', depthGrids, category: row.category, leaderLengthPrior: leaderPrior, geometry });
      candidates.push({
        candidate_id: `${row.category}_${row.sample_id}_${perturb.mode}`.replace(/[^A-Za-z0-9_-]+/g, '_'),
        mode_index: mode,
        mode: perturb.mode,
        bounded: perturb.bounded,
        style: labels[0]?.moe_style_selected || row.style || null,
        style_weights: labels[0]?.moe_style_routing || row.style_weights || null,
        metrics,
        labels: labels.map((label) => ({ id: label.id, text: label.text, anchor: label.anchor, center: label.center, boxSize: label.boxSize, perturbation: label.mdpo_perturbation || null })),
        projected: Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, projectedView(labels, bounds, view)]))
      });
    }
    outputRows.push({ category: row.category, sample_id: row.sample_id, split: row.split, source_obj: sample.input.source_obj, view_files: viewFiles, bounds, candidates });
  }
  const result = {
    version: 'moe_layout_llm_candidate_projection_v1',
    generated_at: new Date().toISOString(),
    source_pseudolabels: path.relative(root, path.resolve(String(options.input))).split(path.sep).join('/'),
    image_order: MULTI_VIEW_NAMES,
    modes: options.modes.map((mode) => ({ mode_index: mode, mode: MDPO_PERTURBATION_MODES[mode] })),
    policy: {
      candidates: 'baseline_plus_bounded_anchor_local_perturbations',
      projection: 'dataset_camera_protocol_normalized_device_coordinates',
      llm_use: 'render PNG overlays then score five-view layout preference'
    },
    rows: outputRows
  };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(result, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/'), samples: outputRows.length, candidates: outputRows.reduce((sum, item) => sum + item.candidates.length, 0) }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
