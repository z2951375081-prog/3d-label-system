import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cameraFrame, DISPLAY_CAMERA_ORIENTATIONS } from '../public/webgl-viewer.js';
import { DATASET_CAMERA_PROTOCOL, datasetCameraForBounds } from '../public/dataset-camera.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocol = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_camera_protocol.json'), 'utf8'));
const upright = DISPLAY_CAMERA_ORIENTATIONS.upright;

assert.ok(upright, 'missing upright display orientation');
assert.equal(upright.theta, 0, 'large-view display camera must start from the front view');
assert.equal(upright.phi, 0, 'upright display camera must have zero pitch');
assert.equal(upright.distance, 3, 'upright reset must preserve the inspection distance');
assert.deepEqual([...upright.pan], [0, 0], 'upright reset must clear display panning');

const eye = [Math.sin(upright.theta) * upright.distance, 0, Math.cos(upright.theta) * upright.distance];
const interactiveFrame = cameraFrame(eye, [0, 0, 0]);
assert.ok(interactiveFrame.up[1] > 0.999999, 'interactive camera screen-up must follow world +Y');
assert.ok(Math.abs(interactiveFrame.up[0]) < 1e-9 && Math.abs(interactiveFrame.up[2]) < 1e-9, 'upright camera must have no roll');

const bounds = { center: [1, 2, 3], radius: 4 };
const mainCamera = datasetCameraForBounds(bounds, 'main');
assert.equal(mainCamera.distance, DATASET_CAMERA_PROTOCOL.cameraDistance);
assert.deepEqual(DATASET_CAMERA_PROTOCOL.viewOrder, ['main', 'right', 'left', 'up', 'down']);
assert.equal(DATASET_CAMERA_PROTOCOL.id, 'dataset_multiview_reproduction_v1');
assert.equal(protocol.shared_reproduction_parameters.camera_distance_world_units, DATASET_CAMERA_PROTOCOL.cameraDistance);

console.log('Display camera passed: large-view reset is front-facing and upright; fixed five-view dataset protocol is unchanged.');
