import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { scoreLayoutStyleExperts } from '../lib/moe-layout-styles.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const defaultJson = path.join(root, 'experiments', 'moe_style_routing_audit.json');
const defaultCsv = path.join(root, 'experiments', 'moe_style_routing_audit.csv');

function parseArgs(argv) {
  const options = { split: 'all', output: defaultJson, csv: defaultCsv, limit: 0 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  options.limit = Number(options.limit || 0);
  return options;
}
function repoPath(relativePath) { return path.join(root, String(relativePath).replaceAll('/', path.sep)); }
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function readText(file) { return fs.readFile(file, 'utf8'); }
function csvCell(value) { return `"${String(value ?? '').replaceAll('"', '""')}"`; }
function summarize(rows) {
  const bySplit = {};
  const byCategory = {};
  for (const row of rows) {
    for (const bucket of [bySplit, byCategory]) {
      const key = bucket === bySplit ? row.split : row.category;
      bucket[key] ||= { count: 0, spherical: 0, rectangular: 0, surround: 0, selected: { spherical: 0, rectangular: 0, surround: 0 } };
      const item = bucket[key];
      item.count += 1;
      item.spherical += row.weights.spherical;
      item.rectangular += row.weights.rectangular;
      item.surround += row.weights.surround;
      item.selected[row.selected] += 1;
    }
  }
  const finish = (bucket) => Object.fromEntries(Object.entries(bucket).map(([key, item]) => [key, {
    count: item.count,
    selected: item.selected,
    average_weights: {
      spherical: Number((item.spherical / item.count).toFixed(6)),
      rectangular: Number((item.rectangular / item.count).toFixed(6)),
      surround: Number((item.surround / item.count).toFixed(6))
    }
  }]));
  return { by_split: finish(bySplit), by_category: finish(byCategory) };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifest = await readJson(manifestFile);
  let samples = manifest.samples.filter((sample) => options.split === 'all' || sample.split === options.split);
  if (options.limit > 0) samples = samples.slice(0, options.limit);
  const rows = [];
  for (const sample of samples) {
    const clean = cleanObj(await readText(repoPath(sample.input.source_obj)));
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const annotation = await readJson(repoPath(sample.target.annotation_json));
    const manual = annotationsToLabels(annotation);
    const candidates = fixedCandidatesWithoutTargetLayout(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds);
    validateFixedLabelContract(manual, candidates, `${sample.category}/${sample.sample_id}/moe-style-audit`);
    const routing = scoreLayoutStyleExperts(candidates, bounds, geometry, { view: 'main' });
    rows.push({
      category: sample.category,
      sample_id: sample.sample_id,
      split: sample.split,
      label_count: candidates.length,
      selected: routing.selected,
      weights: routing.weights,
      features: routing.features,
      contour: {
        hull_points: routing.contour.hull.length,
        projected_points: routing.contour.points.length,
        radial_cv: routing.contour.radial_cv,
        projected_aspect: routing.contour.projected_aspect
      },
      input: sample.input.source_obj,
      target: sample.target.annotation_json
    });
  }
  const result = {
    version: 'moe_style_routing_audit_v1',
    generated_at: new Date().toISOString(),
    source_manifest: path.relative(root, manifestFile).replaceAll('\\', '/'),
    policy: {
      labels: 'fixed_manual_contract_without_target_center_or_box_size',
      geometry: 'clean_obj_contour_point_cloud_main_view',
      routing: 'normalized_soft_probabilities_for_spherical_rectangular_surround'
    },
    count: rows.length,
    summary: summarize(rows),
    rows
  };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(result, null, 2) + '\n', 'utf8');
  const header = ['category', 'sample_id', 'split', 'label_count', 'selected', 'spherical', 'rectangular', 'surround', 'projected_aspect', 'aspect_3d', 'contour_regularity', 'anchor_angular_entropy', 'anchor_radial_uniformity', 'contour_radial_cv'];
  const csv = [header.join(',')].concat(rows.map((row) => [
    row.category, row.sample_id, row.split, row.label_count, row.selected,
    row.weights.spherical, row.weights.rectangular, row.weights.surround,
    row.features.projected_aspect, row.features.aspect_3d, row.features.contour_regularity,
    row.features.anchor_angular_entropy, row.features.anchor_radial_uniformity, row.features.contour_radial_cv
  ].map(csvCell).join(','))).join('\n') + '\n';
  await fs.writeFile(path.resolve(String(options.csv)), csv, 'utf8');
  console.log(JSON.stringify({ ok: true, count: rows.length, output: path.relative(root, path.resolve(String(options.output))).replaceAll('\\', '/'), csv: path.relative(root, path.resolve(String(options.csv))).replaceAll('\\', '/'), summary: result.summary }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
