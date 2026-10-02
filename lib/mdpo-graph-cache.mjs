import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { serialize, deserialize } from 'node:v8';

const VERSION = 'v10_mdpo_graph_cache_v1';
const GRAPH_BUILD_SCRIPTS = Object.freeze([
  'scripts/train-3d-human-style-layout-model.mjs', 'scripts/generate-artifacts.mjs',
  'lib/candidate-generator.mjs', 'lib/layout-model.mjs', 'lib/layout-optimizer.mjs',
  'lib/cv-feature-encoder.mjs', 'lib/spatial-style-features.mjs',
  'lib/heterogeneous-layout-graph.mjs', 'lib/anchor-frame-features.mjs',
  'lib/view-conditioned-evaluator.mjs'
]);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function frameData(frame) {
  if (!frame) return null;
  const { localCoordinates, worldFromLocal, ...data } = frame;
  return data;
}

export function reviveMdpoFrame(raw) {
  if (!raw || ![raw.origin, raw.tangent1, raw.tangent2, raw.normal].every((vector) => Array.isArray(vector) && vector.length === 3 && vector.every(Number.isFinite))) throw new Error('Invalid cached anchor frame');
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  return {
    ...raw,
    localCoordinates(position) {
      const delta = position.map((value, index) => value - raw.origin[index]);
      return { u: dot(delta, raw.tangent1), v: dot(delta, raw.tangent2), normal: dot(delta, raw.normal) };
    },
    worldFromLocal(local) {
      const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
      return raw.origin.map((value, index) => value + raw.tangent1[index] * finite(local?.u) + raw.tangent2[index] * finite(local?.v) + raw.normal[index] * finite(local?.normal));
    }
  };
}

function pack(graphs) {
  return graphs.map((graph) => ({
    ...graph,
    nodes: graph.nodes.map((node) => ({ ...node, frame: frameData(node.frame) })),
    heterogeneousGraph: graph.heterogeneousGraph && {
      ...graph.heterogeneousGraph,
      frames: graph.heterogeneousGraph.frames.map(frameData),
      anchorNodes: graph.heterogeneousGraph.anchorNodes.map((node) => ({ ...node, frame: frameData(node.frame) }))
    }
  }));
}

function unpack(graphs) {
  return graphs.map((graph) => {
    if (graph.heterogeneousGraph) {
      graph.heterogeneousGraph.frames = graph.heterogeneousGraph.frames.map(reviveMdpoFrame);
      graph.heterogeneousGraph.anchorNodes.forEach((node, index) => { node.frame = graph.heterogeneousGraph.frames[index]; });
      graph.nodes.forEach((node, index) => { node.frame = graph.heterogeneousGraph.frames[index]; });
    } else graph.nodes.forEach((node) => { if (node.frame) node.frame = reviveMdpoFrame(node.frame); });
    return graph;
  });
}

async function fingerprint(root, manifest, manifestBytes, gridSize) {
  const files = [
    ...GRAPH_BUILD_SCRIPTS,
    ...manifest.samples.filter((sample) => ['train', 'val'].includes(sample.split)).flatMap((sample) => [sample.input.source_obj, sample.target.annotation_json])
  ];
  const hash = createHash('sha256').update(VERSION).update(String(gridSize)).update(manifestBytes);
  for (const relative of [...new Set(files)].sort()) {
    const absolute = path.resolve(root, relative);
    if (!absolute.startsWith(root + path.sep)) throw new Error(`Unsafe graph cache dependency path: ${relative}`);
    hash.update(relative).update(sha256(await fs.readFile(absolute)));
  }
  return hash.digest('hex');
}

export async function loadMdpoTrainingGraphs({ root, manifest, manifestBytes, gridSize, directory, buildGraphs }) {
  const base = path.resolve(root);
  const key = await fingerprint(base, manifest, manifestBytes, gridSize);
  const cacheDirectory = path.resolve(directory);
  await fs.mkdir(cacheDirectory, { recursive: true });
  const file = path.join(cacheDirectory, `v10_grid_${gridSize}_${key.slice(0, 24)}.bin`);
  const expected = Object.fromEntries(['train', 'val'].map((split) => [split, manifest.samples.filter((sample) => sample.split === split).map((sample) => `${sample.category}/${sample.sample_id}`)]));
  try {
    const cached = deserialize(await fs.readFile(file));
    if (cached.version !== VERSION || cached.key !== key || JSON.stringify(cached.cohort) !== JSON.stringify(expected)) throw new Error('MDPO graph cache provenance mismatch');
    return { trainGraphs: unpack(cached.trainGraphs), valGraphs: unpack(cached.valGraphs), cache: { status: 'hit', file, key } };
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const trainGraphs = await buildGraphs(manifest, 'train', gridSize, 'v10');
  const valGraphs = await buildGraphs(manifest, 'val', gridSize, 'v10');
  const actual = Object.fromEntries([['train', trainGraphs], ['val', valGraphs]].map(([split, graphs]) => [split, graphs.map((graph) => `${graph.category}/${graph.sample_id}`)]));
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error('MDPO graph build cohort mismatch');
  const temporary = path.join(cacheDirectory, `.v10_grid_${gridSize}_${key.slice(0, 24)}_${randomUUID()}.tmp`);
  const bytes = serialize({ version: VERSION, key, cohort: expected, trainGraphs: pack(trainGraphs), valGraphs: pack(valGraphs) });
  await fs.writeFile(temporary, bytes, { flag: 'wx' });
  await fs.rename(temporary, file);
  return { trainGraphs, valGraphs, cache: { status: 'built', file, key, bytes: bytes.length } };
}
