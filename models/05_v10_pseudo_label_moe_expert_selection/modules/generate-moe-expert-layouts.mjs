import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { MOE_STYLE_EXPERTS } from '../lib/moe-layout-styles.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const leaderPriorFile = path.join(root, 'experiments', 'manual_leader_length_prior.json');
const defaultOutput = path.join(root, 'experiments', 'moe_expert_layouts_train33.json');

function parseArgs(argv) { const options = { split: 'train', output: defaultOutput, iterations: 120, seed: 17, limit: 0 }; for (let i=0;i<argv.length;i+=1) if(argv[i].startsWith('--')) { const k=argv[i].slice(2); options[k]=argv[i+1]&&!argv[i+1].startsWith('--')?argv[++i]:true; } options.iterations=Number(options.iterations||120); options.seed=Number(options.seed||17); options.limit=Number(options.limit||0); return options; }
function repoPath(value) { return path.join(root, String(value).replaceAll('/', path.sep)); }
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function readText(file) { return fs.readFile(file, 'utf8'); }
function key(category, id) { return `${category}/${id}`; }
function cloneLabel(label) { return { ...label, anchor: [...label.anchor], center: [...label.center], boxSize: [...label.boxSize], bendPoints: (label.bendPoints || []).map((point) => [...point]) }; }

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
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/expert-input`);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const experts = {};
    for (const style of MOE_STYLE_EXPERTS.slice(0, 3)) {
      const labels = optimizeLabels(candidates.map(cloneLabel), bounds, { fixedLabels: true, groupPolicy: 'all', viewPolicy: 'binocular', optimizer: 'annealing', iterations: options.iterations, seed: options.seed + sampleIndex * 1009 + style.length * 37, depthGrids, category: sample.category, leaderLengthPrior: leaderPrior, geometry, layoutStyle: style, styleView: 'main' });
      validateFixedLabelContract(manual, labels, `${sample.category}/${sample.sample_id}/${style}/expert-output`);
      experts[style] = { style, labels, metrics: evaluateLayout(labels, bounds, { fixedLabels: true, viewPolicy: 'binocular', depthGrids, category: sample.category, leaderLengthPrior: leaderPrior, manualReference: manual, geometry }) };
    }
    rows.push({ category: sample.category, sample_id: sample.sample_id, split: sample.split, source_obj: sample.input.source_obj, annotation_json: sample.target.annotation_json, bounds, expert_names: Object.keys(experts), experts });
  }
  const result = { version: 'moe_expert_layouts_v1', generated_at: new Date().toISOString(), split: options.split, iterations: options.iterations, seed: options.seed, experts: MOE_STYLE_EXPERTS.slice(0, 3), policy: { each_sample_generates_one_layout_per_expert: true, geometry_evaluation: 'five_view_energy_with_main_priority', fixed_label_contract: true }, count: rows.length, rows };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(result, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/'), count: rows.length, experts: result.experts }, null, 2));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
