import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadMdpoTrainingGraphs } from '../lib/mdpo-graph-cache.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fullManifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const samples = ['train', 'val'].map((split) => fullManifest.samples.find((row) => row.split === split));
const manifest = { samples };
const manifestBytes = Buffer.from(JSON.stringify(manifest));
const scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-graph-cache-test-'));
assert.ok(path.resolve(scratch).startsWith(path.resolve(os.tmpdir()) + path.sep));
let builds = 0;
const frame = {
  origin: [2, 3, 4], tangent1: [1, 0, 0], tangent2: [0, 1, 0], normal: [0, 0, 1],
  localCoordinates(position) { return { u: position[0] - 2, v: position[1] - 3, normal: position[2] - 4 }; },
  worldFromLocal(local) { return [2 + local.u, 3 + local.v, 4 + local.normal]; }
};
const buildGraphs = async (_manifest, split) => {
  builds++;
  return samples.filter((sample) => sample.split === split).map((sample) => ({
    category: sample.category, sample_id: sample.sample_id,
    nodes: [{ id: 'anchor-1', frame, target: [0, 0, 0, 0, 0, 0] }],
    heterogeneousGraph: { frames: [frame], anchorNodes: [{ id: 'anchor:anchor-1', frame }] },
    depthGrids: { main: { depth: new Float32Array([Infinity, -Infinity, 1.25]), farDepth: new Float32Array([-Infinity, Infinity, 2.5]), gridSize: 1 } }
  }));
};
try {
  const config = { root, manifest, manifestBytes, gridSize: 20, directory: scratch, buildGraphs };
  const built = await loadMdpoTrainingGraphs(config);
  assert.equal(built.cache.status, 'built');
  assert.equal(builds, 2);
  const cached = await loadMdpoTrainingGraphs(config);
  assert.equal(cached.cache.status, 'hit');
  assert.equal(builds, 2, 'cache hit cannot rebuild any graph');
  assert.equal(cached.trainGraphs.length, 1);
  assert.equal(cached.valGraphs.length, 1);
  const restored = cached.trainGraphs[0];
  assert.deepEqual(restored.nodes[0].frame.localCoordinates([3, 5, 7]), { u: 1, v: 2, normal: 3 });
  assert.deepEqual(restored.nodes[0].frame.worldFromLocal({ u: 1, v: 2, normal: 3 }), [3, 5, 7]);
  assert.equal(restored.nodes[0].frame, restored.heterogeneousGraph.frames[0]);
  assert.equal(restored.heterogeneousGraph.anchorNodes[0].frame, restored.nodes[0].frame);
  assert.ok(restored.depthGrids.main.depth instanceof Float32Array);
  assert.equal(restored.depthGrids.main.depth[0], Infinity);
  assert.equal(restored.depthGrids.main.depth[1], -Infinity);
  const changed = await loadMdpoTrainingGraphs({ ...config, gridSize: 21 });
  assert.equal(changed.cache.status, 'built', 'different spatial grid requires new cache');
  assert.equal(builds, 4);
  console.log('v10-MDPO graph cache provenance, typed-depth preservation and anchor-frame revival passed.');
} finally {
  if (!path.resolve(scratch).startsWith(path.resolve(os.tmpdir()) + path.sep) || !path.basename(scratch).startsWith('v10-mdpo-graph-cache-test-')) throw new Error('Unsafe test cleanup target');
  await fs.rm(scratch, { recursive: true, force: true });
}
