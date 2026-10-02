import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DATASET_CAMERA_PROTOCOL, DATASET_VIEW_NAMES } from '../public/dataset-camera.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocol = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_camera_protocol.json'), 'utf8'));
const calibration = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_camera_calibration.json'), 'utf8'));
assert.equal(calibration.status, 'image_derived_estimate_not_original_camera_metadata');
assert.equal(protocol.parameter_provenance.image_derived_calibration, 'experiments/dataset_camera_calibration.json');
assert.equal(DATASET_CAMERA_PROTOCOL.imageWidth, protocol.observed_from_dataset.image_width_px);
assert.equal(DATASET_CAMERA_PROTOCOL.imageHeight, protocol.observed_from_dataset.image_height_px);
assert.deepEqual(DATASET_VIEW_NAMES, protocol.observed_from_dataset.view_names);

for (const view of DATASET_VIEW_NAMES) {
  const result = calibration.views[view];
  assert.ok(result, `missing dataset image-derived calibration for ${view}`);
  assert.equal(result.cases.train, 12);
  assert.equal(result.cases.val, 11);
  assert.equal(result.cases.test, 11);
  assert.equal(result.camera.camera_distance_fixed, DATASET_CAMERA_PROTOCOL.cameraDistance);
  assert.ok(result.direction_change_from_reproduction_degrees < 1, `${view} direction differs too much`);
  assert.ok(Math.abs(result.camera.equivalent_focal_length_x_mm_for_36mm_sensor - DATASET_CAMERA_PROTOCOL.focalLengthMm) < 1);
  assert.ok(Math.abs(result.camera.equivalent_focal_length_y_mm_for_24mm_sensor - DATASET_CAMERA_PROTOCOL.focalLengthMm) < 1);
  for (const split of ['val', 'test']) {
    const original = result.red_line_alignment_rmse_pixels.reproduction_protocol[split];
    const fitted = result.red_line_alignment_rmse_pixels.calibrated[split];
    assert.ok(original > 0 && original < 2, `${view}/${split}: shared protocol not supported by image pixels`);
    assert.ok(fitted > 0 && fitted <= original + 0.01, `${view}/${split}: image-fit generalizes worse than shared protocol`);
  }
}
console.log('Camera calibration audit passed: five views, train-only fit, held-out val/test PNG alignment below 2px; distance/focal ambiguity retained.');
