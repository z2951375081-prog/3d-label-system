import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { compareToManual } from '../lib/layout-comparison.mjs';
import { applyLayoutModel, selectLayoutCandidates } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const outputDir = path.join(root, 'experiments', 'benchmarks');

function parseArgs(argv) {
  const options = { split: 'test', seed: 17, iterations: 180, output: outputDir, model: 'trained' };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function repoFile(relativePath) { return path.join(root, relativePath.replaceAll('/', path.sep)); }
function relativeRepo(file) { return path.relative(root, file).split(path.sep).join('/'); }
async function read(file) { return fs.readFile(file, 'utf8'); }
async function write(file, content) { await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, content, 'utf8'); }
function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function std(values) { const average = mean(values); return values.length ? Math.sqrt(mean(values.map((value) => (value - average) ** 2))) : 0; }
function round(value) { return Number(value.toFixed(4)); }

const conditions = [
  { id: 'rules_binocular_semantic_relative', viewPolicy: 'binocular', groupPolicy: 'semantic-once', sizePolicy: 'relative', optimizer: 'rules' },
  { id: 'annealing_binocular_semantic_relative', viewPolicy: 'binocular', groupPolicy: 'semantic-once', sizePolicy: 'relative', optimizer: 'annealing' },
  { id: 'annealing_binocular_all_relative', viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing' },
  { id: 'annealing_binocular_symmetric_relative', viewPolicy: 'binocular', groupPolicy: 'symmetric', sizePolicy: 'relative', optimizer: 'annealing' },
  { id: 'annealing_binocular_semantic_fixed', viewPolicy: 'binocular', groupPolicy: 'semantic-once', sizePolicy: 'fixed', optimizer: 'annealing' },
  { id: 'annealing_single_semantic_relative', viewPolicy: 'single', groupPolicy: 'semantic-once', sizePolicy: 'relative', optimizer: 'annealing' }
];

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (!['train', 'val', 'test', 'all'].includes(options.split)) throw new Error('--split 必须是 train/val/test/all');
  const seed = Number(options.seed);
  const iterations = Number(options.iterations);
  if (!Number.isInteger(seed) || seed < 0) throw new Error('--seed 必须是非负整数');
  if (!Number.isInteger(iterations) || iterations < 20 || iterations > 2000) throw new Error('--iterations 必须是 20-2000 的整数');
  const manifest = JSON.parse(await read(manifestFile));
  let layoutModel = null;
  if (!['trained', 'none'].includes(options.model)) throw new Error('--model 必须是 trained/none');
  if (options.model === 'trained') layoutModel = JSON.parse(await read(path.join(root, 'experiments', 'layout_model.json')));
  const leaderLengthPrior = JSON.parse(await read(path.join(root, 'experiments', 'manual_leader_length_prior.json')));
  const samples = manifest.samples.filter((sample) => options.split === 'all' || sample.split === options.split);
  const rows = [];
  for (const condition of conditions) {
    for (const sample of samples) {
      const raw = await read(repoFile(sample.input.source_obj));
      const clean = cleanObj(raw);
      const bounds = boundsFromObj(clean.text);
      const geometry = parseObjTriangles(clean.text);
      const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
      const annotation = sample.target.annotation_json ? JSON.parse(await read(repoFile(sample.target.annotation_json))) : null;
      const manualLabels = annotationsToLabels(annotation);
      const optimizerOptions = { ...condition, viewPolicy: manualLabels.length ? 'binocular' : condition.viewPolicy, groupPolicy: manualLabels.length ? 'all' : condition.groupPolicy, fixedLabels: Boolean(manualLabels.length), seed, iterations, depthGrids, category: sample.category, leaderLengthPrior };
       const started = performance.now();
       const generated = generatedCandidatesFromCleanObj(clean.text, bounds);
       const candidates = manualLabels.length ? fixedCandidatesForLayoutModel(manualLabels, generated, bounds, layoutModel) : generated;
       validateFixedLabelContract(manualLabels, candidates, `${sample.category}/${sample.sample_id}/benchmark-candidates`);
       const selected = manualLabels.length ? { candidates, selection: { status: 'fixed_manual_label_contract' } } : layoutModel ? selectLayoutCandidates(candidates, sample.category, layoutModel) : { candidates };
       const styled = layoutModel ? applyLayoutModel(selected.candidates, bounds, layoutModel) : { labels: candidates };
      const labels = optimizeLabels(styled.labels, bounds, optimizerOptions);
      validateFixedLabelContract(manualLabels, labels, `${sample.category}/${sample.sample_id}/benchmark-output`);
      const metrics = evaluateLayout(labels, bounds, { ...optimizerOptions, manualReference: manualLabels.length ? manualLabels : undefined });
      const manualMetrics = manualLabels.length ? evaluateLayout(manualLabels, bounds, { ...optimizerOptions, manualReference: manualLabels }) : null;
      const comparison = manualMetrics ? compareToManual(metrics, manualMetrics) : null;
      rows.push({ condition: condition.id, category: sample.category, sample_id: sample.sample_id, split: sample.split, runtime_ms: round(performance.now() - started), manual_score: comparison?.generated_score ?? null, ...metrics });
    }
  }
  const metricNames = ['label_count', 'semantic_count', 'readability', 'coverage', 'occlusion', 'overlap_pairs', 'overlap_ratio', 'olr', 'leader_crossings', 'lcd', 'dbv', 'viewport_overflow_ratio', 'anchor_coverage', 'mean_anchor_distance', 'leader_length_compliance_ratio', 'leader_length_shortfall', 'binocular_disparity', 'objective_score', 'manual_score', 'runtime_ms'];
  const summary = conditions.map((condition) => {
    const subset = rows.filter((row) => row.condition === condition.id);
    const aggregate = { condition: condition.id, split: options.split, sample_count: subset.length, strategy: condition };
    for (const metric of metricNames) { const values = subset.map((row) => Number(row[metric] || 0)); aggregate[`${metric}_mean`] = round(mean(values)); aggregate[`${metric}_std`] = round(std(values)); }
    return aggregate;
  });
  const run = { version: 'layout_benchmark_v1', generated_at: new Date().toISOString(), options: { split: options.split, seed, iterations, model: options.model }, layout_model: layoutModel ? { status: layoutModel.status, training: layoutModel.training } : { status: 'not_loaded' }, conditions, summary, rows };
  const jsonFile = path.join(options.output, 'layout_benchmark.json');
  const csvHeader = ['condition', 'category', 'sample_id', 'split', ...metricNames];
  const csv = [csvHeader.join(','), ...rows.map((row) => csvHeader.map((key) => JSON.stringify(row[key] ?? '')).join(','))].join('\n') + '\n';
  await write(jsonFile, `${JSON.stringify(run, null, 2)}\n`);
  await write(path.join(options.output, 'layout_benchmark.csv'), csv);
  console.log(`完成：${samples.length} 个 ${options.split} 样本 × ${conditions.length} 个条件`);
  for (const item of summary) console.log(`${item.condition}: manual=${item.manual_score_mean ?? '—'}, readability=${item.readability_mean}±${item.readability_std}, OLR=${item.olr_mean ?? '—'}, LCD=${item.lcd_mean ?? '—'}, DBV=${item.dbv_mean ?? '—'}, runtime=${item.runtime_ms_mean} ms`);
  console.log(`输出：${relativeRepo(jsonFile)} 和 ${relativeRepo(path.join(options.output, 'layout_benchmark.csv'))}`);
}

main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });
