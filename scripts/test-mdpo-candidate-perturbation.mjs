import assert from 'node:assert/strict';
import { perturbMdpoCandidate, MDPO_PERTURBATION_MODES } from '../lib/mdpo-candidate-perturbation.mjs';

const bounds = { center: [0, 0, 0], radius: 1, min: [-1, -1, -1], max: [1, 1, 1] };
const geometry = { triangles: [ [[0,0,0],[1,0,0],[0,1,0]], [[0,0,0],[0,1,0],[0,0,1]] ] };
const labels = Array.from({ length: 8 }, (_, index) => ({ id: `L${index}`, anchor: [0.05 * index, 0, 0], center: [0.05 * index, 0.8, 0.2], boxSize: [0.4, 0.16, 0.02], sourceObjs: [], targetGroups: [] }));
const baseline = perturbMdpoCandidate(labels, geometry, bounds, { mode: 0, seed: 17 });
assert.deepEqual(baseline.labels, labels);
const signatures = MDPO_PERTURBATION_MODES.map((_, mode) => JSON.stringify(perturbMdpoCandidate(labels, geometry, bounds, { mode, seed: 17 }).labels.map((label) => [label.center, label.boxSize])));
assert.equal(new Set(signatures).size, MDPO_PERTURBATION_MODES.length);
assert.deepEqual(perturbMdpoCandidate(labels, geometry, bounds, { mode: 9, seed: 17 }), perturbMdpoCandidate(labels, geometry, bounds, { mode: 9, seed: 17 }));
assert.ok(perturbMdpoCandidate(labels, geometry, bounds, { mode: 3 }).labels.every((label, index) => label.boxSize[0] < labels[index].boxSize[0]));
assert.ok(perturbMdpoCandidate(labels, geometry, bounds, { mode: 4 }).labels.every((label, index) => label.boxSize[0] > labels[index].boxSize[0]));
console.log(`MDPO bounded anchor-local candidate perturbations passed: ${MDPO_PERTURBATION_MODES.length} deterministic modes.`);
