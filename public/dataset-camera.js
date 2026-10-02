// Shared camera used by the browser renderer and the Node evaluation pipeline.
// Image size/view names are observed in data/Layout/**/Mutiviews. Intrinsic
// and extrinsic values follow the common BinoForce/Hedgehog reproduction
// protocol because the dataset does not ship an original camera matrix.

export const DATASET_CAMERA_PROTOCOL = Object.freeze({
  id: 'dataset_multiview_reproduction_v1',
  imageWidth: 750,
  imageHeight: 500,
  focalLengthMm: 50,
  sensorWidthMm: 36,
  sensorHeightMm: 24,
  cameraDistance: 10,
  perturbDegrees: 45,
  near: 1e-6,
  far: 1e6,
  mainDirection: [1, 1, 1],
  viewOrder: ['main', 'right', 'left', 'up', 'down']
});

function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale(a, value) { return [a[0] * value, a[1] * value, a[2] * value]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(a, fallback = [1, 0, 0]) {
  const length = Math.hypot(...a);
  return length > 1e-9 ? scale(a, 1 / length) : [...fallback];
}

function frameFromDirection(direction, upHint = [0, 1, 0]) {
  const outward = norm(direction);
  const forward = scale(outward, -1);
  const right = norm(cross(forward, norm(upHint)), [1, 0, 0]);
  const up = norm(cross(right, forward), [0, 1, 0]);
  return { outward, forward, right, up };
}

function rotateToward(direction, screenAxis, degrees) {
  const radians = degrees * Math.PI / 180;
  return norm(add(scale(norm(direction), Math.cos(radians)), scale(norm(screenAxis), Math.sin(radians))));
}

const mainFrame = frameFromDirection(DATASET_CAMERA_PROTOCOL.mainDirection);
const VIEW_DIRECTIONS = Object.freeze({
  main: mainFrame.outward,
  right: rotateToward(mainFrame.outward, mainFrame.right, DATASET_CAMERA_PROTOCOL.perturbDegrees),
  left: rotateToward(mainFrame.outward, scale(mainFrame.right, -1), DATASET_CAMERA_PROTOCOL.perturbDegrees),
  up: rotateToward(mainFrame.outward, mainFrame.up, DATASET_CAMERA_PROTOCOL.perturbDegrees),
  down: rotateToward(mainFrame.outward, scale(mainFrame.up, -1), DATASET_CAMERA_PROTOCOL.perturbDegrees)
});

export const DATASET_VIEW_NAMES = Object.freeze([...DATASET_CAMERA_PROTOCOL.viewOrder]);

export function datasetCameraForBounds(bounds, view = 'main') {
  const target = [...bounds.center];
  const direction = VIEW_DIRECTIONS[view] || VIEW_DIRECTIONS.main;
  const frame = frameFromDirection(direction, mainFrame.up);
  const eye = add(target, scale(frame.outward, DATASET_CAMERA_PROTOCOL.cameraDistance));
  return {
    ...frame,
    eye,
    target,
    view,
    distance: DATASET_CAMERA_PROTOCOL.cameraDistance,
    focalLengthMm: DATASET_CAMERA_PROTOCOL.focalLengthMm,
    sensorWidthMm: DATASET_CAMERA_PROTOCOL.sensorWidthMm,
    sensorHeightMm: DATASET_CAMERA_PROTOCOL.sensorHeightMm,
    horizontalFovRadians: 2 * Math.atan(DATASET_CAMERA_PROTOCOL.sensorWidthMm / (2 * DATASET_CAMERA_PROTOCOL.focalLengthMm)),
    verticalFovRadians: 2 * Math.atan(DATASET_CAMERA_PROTOCOL.sensorHeightMm / (2 * DATASET_CAMERA_PROTOCOL.focalLengthMm)),
    near: DATASET_CAMERA_PROTOCOL.near,
    far: DATASET_CAMERA_PROTOCOL.far
  };
}

export function datasetCameraDirections() {
  return Object.fromEntries(Object.entries(VIEW_DIRECTIONS).map(([name, direction]) => [name, [...direction]]));
}
