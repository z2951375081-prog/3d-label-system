import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels as optimizeSharedLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { compareToManual } from '../lib/layout-comparison.mjs';
import { applyLayoutModel, selectLayoutCandidates } from '../lib/layout-model.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestFile = path.join(root, 'experiments', 'dataset_manifest.json');
const defaultOutput = path.join(root, 'experiments', 'artifacts');

function parseArgs(argv) {
  const options = { split: 'all', group: 'all', size: 'relative', optimizer: 'annealing', view: 'binocular', seed: 17, iterations: 180, output: defaultOutput, limit: 0 };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'help') { options.help = true; continue; }
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function printHelp() {
  console.log(`用法：
  npm run artifacts -- --split test --group semantic-once --size relative --optimizer annealing

参数：
  --split       all / train / val / test，默认 all
  --group       all / semantic-once / symmetric，默认 semantic-once
  --size        relative / fixed / distance-aware，默认 relative
  --optimizer   rules / annealing，默认 annealing
  --view        binocular / single，默认 binocular
  --seed        随机种子，默认 17
  --iterations  退火迭代次数，默认 180
  --output      输出目录，默认 experiments/artifacts
  --limit       只处理前 N 个样本，调试时使用
`);
}

async function readText(file) { return fs.readFile(file, 'utf8'); }
async function writeText(file, text) { await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, text, 'utf8'); }
function resolveRepoPath(relativePath) { return path.join(root, relativePath.replaceAll('/', path.sep)); }
function relativeRepoPath(file) { return path.relative(root, file).split(path.sep).join('/'); }

function isPresentationGroupName(name) {
  return /^(label_|leader_)/i.test(String(name || ''));
}

function isPresentationMaterialName(name) {
  return /^(label_|leader_)/i.test(String(name || ''));
}

function cleanObj(objText) {
  const vertices = [];
  const faces = [];
  let currentGroup = 'object';
  let skipGroup = false;
  let skipMaterial = false;
  for (const line of objText.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const parts = trimmed.split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) { vertices.push(parts.slice(1, 4)); continue; }
    if (parts[0] === 'g' || parts[0] === 'o') {
      const groupName = parts.slice(1).join('_') || 'object';
      if (parts[0] === 'g') currentGroup = groupName;
      skipGroup = parts.slice(1).some(isPresentationGroupName);
      skipMaterial = false;
      continue;
    }
    if (parts[0] === 'usemtl') { skipMaterial = isPresentationMaterialName(parts[1] || ''); continue; }
    if (parts[0] === 'f' && parts.length >= 4 && !skipGroup && !skipMaterial) {
      const refs = parts.slice(1).map((part) => Number(part.split('/')[0]));
      if (refs.length >= 3 && refs.every((ref) => Number.isInteger(ref) && ref !== 0)) faces.push({ refs, group: currentGroup });
    }
  }
  const used = new Map();
  const cleanVertices = [];
  const resolve = (ref) => {
    const oldIndex = ref > 0 ? ref - 1 : vertices.length + ref;
    if (oldIndex < 0 || oldIndex >= vertices.length || !vertices[oldIndex]) return null;
    if (!used.has(oldIndex)) { used.set(oldIndex, cleanVertices.length + 1); cleanVertices.push(vertices[oldIndex]); }
    return used.get(oldIndex);
  };
  const cleanFaces = [];
  for (const face of faces) {
    const resolved = face.refs.map(resolve);
    if (resolved.every((index) => index !== null)) cleanFaces.push({ indices: resolved, group: face.group });
  }
  const output = ['# clean input generated from Obj-O', '# presentation layers removed; anchor material styling normalized to object_default', 'o clean_model', 'g clean_model', 'usemtl object_default'];
  cleanVertices.forEach((vertex) => output.push(`v ${vertex.join(' ')}`));
  let previousGroup = null;
  cleanFaces.forEach((face) => { if (face.group !== previousGroup) { output.push(`g ${face.group}`); previousGroup = face.group; } output.push(`f ${face.indices.join(' ')}`); });
  return { text: `${output.join('\n')}\n`, vertexCount: cleanVertices.length, faceCount: cleanFaces.length };
}

function annotationsToLabels(annotation) {
  if (!annotation?.groups?.length) return [];
  const palette = [[0.93, 0.35, 0.25], [0.20, 0.55, 0.95], [0.95, 0.66, 0.16], [0.55, 0.34, 0.86], [0.14, 0.68, 0.56], [0.84, 0.27, 0.58], [0.42, 0.70, 0.25], [0.18, 0.72, 0.78]];
  return annotation.groups.map((group, index) => {
    const label = group.label || {};
    const anchor = group.anchor?.point || group.leader_line?.start || label.center || [0, 0, 0];
    const center = label.center || group.leader_line?.end || anchor;
    return {
      id: group.group_id || `label-${index + 1}`,
      text: label.text || group.group_id || `label-${index + 1}`,
      anchor: [...anchor],
      center: [...center],
      boxSize: [...(label.box_size || [0.25, 0.12, 0.02])],
      bendPoints: (group.leader_line?.bend_points || []).map((point) => [...point]),
      sourceObjs: [...(group.source_objs || [])],
      targetGroups: [...(group.target_g || [])],
      color: [...palette[index % palette.length]]
    };
  });
}

function boundsFromObj(cleanText) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const line of cleanText.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] !== 'v' || parts.length < 4) continue;
    const point = parts.slice(1, 4).map(Number);
    for (let index = 0; index < 3; index += 1) { min[index] = Math.min(min[index], point[index]); max[index] = Math.max(max[index], point[index]); }
  }
  const center = min.map((value, index) => (value + max[index]) / 2);
  const size = max.map((value, index) => value - min[index]);
  return { min, max, center, size, radius: Math.max(Math.hypot(...size) / 2, 0.001) };
}

function exportLabeledObj(cleanText, labels) {
  const vertexCount = (cleanText.match(/^v\s+/gm) || []).length;
  const output = [cleanText.trimEnd(), '', '# --- Generated 3D labels ---'];
  let nextVertex = vertexCount + 1;
  labels.forEach((label, index) => {
    const [cx, cy, cz] = label.center;
    const [sx, sy, sz] = label.boxSize;
    const [x, y, z] = [sx / 2, sy / 2, sz / 2];
    const vertices = [[cx - x, cy - y, cz - z], [cx + x, cy - y, cz - z], [cx + x, cy + y, cz - z], [cx - x, cy + y, cz - z], [cx - x, cy - y, cz + z], [cx + x, cy - y, cz + z], [cx + x, cy + y, cz + z], [cx - x, cy + y, cz + z]];
    output.push(`o label_${index + 1}_${String(label.text).replace(/\s+/g, '_')}`);
    vertices.forEach((vertex) => output.push(`v ${vertex.join(' ')}`));
    [[0, 1, 2, 0, 2, 3], [4, 7, 6, 4, 6, 5], [0, 4, 5, 0, 5, 1], [1, 5, 6, 1, 6, 2], [2, 6, 7, 2, 7, 3], [4, 0, 3, 4, 3, 7]].forEach((face) => output.push(`f ${face.map((value) => nextVertex + value).join(' ')}`));
    nextVertex += 8;
    const line = [label.anchor, ...(label.bendPoints || []), label.center];
    const indices = [];
    line.forEach((point) => { output.push(`v ${point.join(' ')}`); indices.push(nextVertex); nextVertex += 1; });
    output.push(`l ${indices.join(' ')}`);
  });
  return `${output.join('\n')}\n`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) { printHelp(); return; }
  if (!['all', 'train', 'val', 'test'].includes(options.split)) throw new Error('--split 必须是 all/train/val/test');
  if (!['all', 'semantic-once', 'symmetric'].includes(options.group)) throw new Error('--group 必须是 all/semantic-once/symmetric');
  if (!['relative', 'fixed', 'distance-aware'].includes(options.size)) throw new Error('--size 参数不正确');
  if (!['rules', 'annealing'].includes(options.optimizer)) throw new Error('--optimizer 必须是 rules/annealing');
  if (!['binocular', 'single'].includes(options.view)) throw new Error('--view 必须是 binocular/single');
  if (!Number.isInteger(Number(options.seed)) || Number(options.seed) < 0) throw new Error('--seed 必须是非负整数');
  if (!Number.isInteger(Number(options.iterations)) || Number(options.iterations) < 20 || Number(options.iterations) > 2000) throw new Error('--iterations 必须是 20-2000 的整数');
  const manifest = JSON.parse(await readText(manifestFile));
  const leaderLengthPrior = await fs.readFile(path.join(root, 'experiments', 'manual_leader_length_prior.json'), 'utf8').then((text) => JSON.parse(text)).catch(() => null);
  let samples = manifest.samples.filter((sample) => options.split === 'all' || sample.split === options.split);
  if (Number(options.limit) > 0) samples = samples.slice(0, Number(options.limit));
  const generated = [];
  for (const sample of samples) {
    const sourceFile = resolveRepoPath(sample.input.source_obj);
    const rawObj = await readText(sourceFile);
    const clean = cleanObj(rawObj);
    const bounds = boundsFromObj(clean.text);
    const geometry = parseObjTriangles(clean.text);
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    const annotation = sample.target.annotation_json ? JSON.parse(await readText(resolveRepoPath(sample.target.annotation_json))) : null;
    const manualLabels = annotationsToLabels(annotation);
    const optimizerOptions = { viewPolicy: manualLabels.length ? 'binocular' : options.view, groupPolicy: manualLabels.length ? 'all' : options.group, sizePolicy: options.size, optimizer: options.optimizer, seed: Number(options.seed), iterations: Number(options.iterations), depthGrids, fixedLabels: Boolean(manualLabels.length), category: sample.category, leaderLengthPrior };
    const layoutModel = await fs.readFile(path.join(root, 'experiments', 'layout_model.json'), 'utf8').then((text) => JSON.parse(text)).catch(() => null);
    const started = performance.now();
    const generatedCandidates = generatedCandidatesFromCleanObj(clean.text, bounds);
    const candidates = manualLabels.length ? fixedCandidatesForLayoutModel(manualLabels, generatedCandidates, bounds, layoutModel) : generatedCandidates;
    validateFixedLabelContract(manualLabels, candidates, `${sample.category}/${sample.sample_id}/candidates`);
    const selected = manualLabels.length ? { candidates, selection: { status: 'fixed_manual_label_contract', input_count: generatedCandidates.length, selected_count: candidates.length } } : layoutModel ? selectLayoutCandidates(candidates, sample.category, layoutModel) : { candidates, selection: { status: 'not_loaded', selected_count: candidates.length } };
    const styled = layoutModel ? applyLayoutModel(selected.candidates, bounds, layoutModel) : { labels: candidates, expert: 'untrained', gate: {} };
    const labels = optimizeSharedLabels(styled.labels, bounds, optimizerOptions);
    const labelContract = validateFixedLabelContract(manualLabels, labels, `${sample.category}/${sample.sample_id}/output`);
    const metrics = evaluateLayout(labels, bounds, { ...optimizerOptions, manualReference: manualLabels.length ? manualLabels : undefined });
    const manualMetrics = manualLabels.length ? evaluateLayout(manualLabels, bounds, { ...optimizerOptions, manualReference: manualLabels }) : null;
    const comparison = manualMetrics ? compareToManual(metrics, manualMetrics) : null;
    const timing = { total_ms: Number((performance.now() - started).toFixed(3)), labels_per_second: Number((labels.length / Math.max((performance.now() - started) / 1000, 1e-6)).toFixed(3)) };
    const artifactDir = path.join(options.output, sample.split, sample.category, sample.sample_id);
    const cleanFile = path.join(artifactDir, `${sample.sample_id}_clean.obj`);
    const labelsFile = path.join(artifactDir, `${sample.sample_id}_labels.json`);
    const labeledFile = path.join(artifactDir, `${sample.sample_id}_labeled.obj`);
    const metadataFile = path.join(artifactDir, 'metadata.json');
    const artifact = { version: 'artifact_v2_fixed_multiview', generated_at: new Date().toISOString(), sample: { category: sample.category, sample_id: sample.sample_id, split: sample.split }, input: { source_obj: sample.input.source_obj, clean_obj: relativeRepoPath(cleanFile), cleaning: 'remove_existing_label_and_leader_groups_normalize_anchor_region_materials' }, target: { labels_json: relativeRepoPath(labelsFile), labeled_obj: relativeRepoPath(labeledFile) }, model, label_contract: labelContract, layout_model: layoutModel ? { version: layoutModel.version, status: layoutModel.status, expert: styled.expert, selection: selected.selection, train_mse: layoutModel.training?.train_mse, val_mse: layoutModel.training?.val_mse } : { status: 'not_loaded', expert: 'untrained', selection: selected.selection }, strategy: { view_policy: options.view, views: MULTI_VIEW_NAMES, group_policy: optimizerOptions.groupPolicy, size_policy: options.size, optimizer: options.optimizer, source: 'clean_obj_geometry', manual_annotation_used_for: manualLabels.length ? 'fixed_label_contract_and_reference' : 'none', seed: Number(options.seed), iterations: Number(options.iterations) }, geometry: { clean_vertices: clean.vertexCount, clean_faces: clean.faceCount }, metrics, manual_reference: manualMetrics, manual_comparison: comparison, timing, labels };
    await writeText(cleanFile, clean.text);
    await writeText(labelsFile, `${JSON.stringify(artifact, null, 2)}\n`);
    await writeText(labeledFile, exportLabeledObj(clean.text, labels));
    await writeText(metadataFile, `${JSON.stringify({ ...artifact, target: { ...artifact.target, metadata: relativeRepoPath(metadataFile) } }, null, 2)}\n`);
    generated.push(artifact);
    console.log(`${sample.split.padEnd(5)} ${sample.category.padEnd(12)} ${sample.sample_id}: ${labels.length} labels, ${clean.vertexCount} vertices, ${clean.faceCount} faces`);
  }
  const runFile = path.join(options.output, 'run.json');
  await writeText(runFile, `${JSON.stringify({ version: 'artifact_run_v1', generated_at: new Date().toISOString(), options: { split: options.split, group: options.group, size: options.size, optimizer: options.optimizer, view: options.view, seed: Number(options.seed), iterations: Number(options.iterations), limit: Number(options.limit) || 0 }, count: generated.length, artifacts: generated }, null, 2)}\n`);
  console.log(`完成：${generated.length} 个样本，清单写入 ${relativeRepoPath(runFile)}`);
}

export { cleanObj, annotationsToLabels, boundsFromObj };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });
}
