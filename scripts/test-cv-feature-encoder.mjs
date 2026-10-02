import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { buildCvFeatures, renderCleanObjFiveViewRasters, sampleObjSurfacePoints } from '../lib/cv-feature-encoder.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const raw = await fs.readFile(path.join(root, 'data', 'Layout', 'Chair', '38635', 'layout1', 'Obj-O', '38635-main-O.obj'), 'utf8');
const clean = cleanObj(raw);
const bounds = boundsFromObj(clean.text);
const pointsA = sampleObjSurfacePoints(clean.text, 1024);
const pointsB = sampleObjSurfacePoints(clean.text, 1024);
assert.equal(pointsA.length, 1024);
assert.deepEqual(pointsA, pointsB, 'surface sampling must be deterministic');
assert.ok(pointsA.flat().every(Number.isFinite));

const rasters = renderCleanObjFiveViewRasters(clean.text, bounds);
assert.equal(rasters.length, 5);
assert.ok(rasters.every((image) => image.length === 32 && image.every((row) => row.length === 32)));

const featuresA = await buildCvFeatures({ objText: clean.text, bounds });
const featuresB = await buildCvFeatures({ objText: clean.text, bounds });
assert.equal(featuresA.geometry.length, 64);
assert.equal(featuresA.visual.length, 32);
assert.equal(featuresA.fused_global.length, 96);
assert.deepEqual(featuresA, featuresB, 'frozen CV features must be deterministic');
assert.ok(featuresA.fused_global.every(Number.isFinite));
await assert.rejects(() => buildCvFeatures({ objText: clean.text, bounds, viewFiles: ['dataset-view-with-manual-labels.png'] }), /human labels/);
console.log('CV feature encoder passed: deterministic 1024-point sampling, 2-stage geometry path, five unlabeled clean-OBJ rasters, 64D+32D features, and dataset-PNG leakage guard.');
