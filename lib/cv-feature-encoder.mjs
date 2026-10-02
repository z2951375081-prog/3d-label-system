import fs from 'node:fs/promises';
import zlib from 'node:zlib';
import { MULTI_VIEW_NAMES, buildDepthGrid, parseObjTriangles } from './layout-optimizer.mjs';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function seededRandom(seed = 17) {
  let state = Number(seed) >>> 0 || 17;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function seededMatrix(rows, columns, seed, scale) {
  const random = seededRandom(seed);
  return Array.from({ length: rows }, () => Array.from({ length: columns }, () => (random() - 0.5) * scale));
}

function matVec(weights, values) {
  return weights.map((row) => row.reduce((sum, weight, index) => sum + weight * values[index], 0));
}

function parseTriangles(objText) {
  const vertices = [];
  const triangles = [];
  for (const line of String(objText || '').split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) vertices.push(parts.slice(1, 4).map(Number));
    if (parts[0] !== 'f' || parts.length < 4) continue;
    const indices = parts.slice(1).map((token) => {
      const raw = Number(token.split('/')[0]);
      return raw < 0 ? vertices.length + raw : raw - 1;
    }).filter((index) => vertices[index]);
    for (let index = 1; index < indices.length - 1; index += 1) triangles.push([vertices[indices[0]], vertices[indices[index]], vertices[indices[index + 1]]]);
  }
  return triangles;
}

function triangleArea(triangle) {
  const [a, b, c] = triangle;
  const ab = b.map((value, axis) => value - a[axis]);
  const ac = c.map((value, axis) => value - a[axis]);
  const cross = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
  return Math.hypot(...cross) * 0.5;
}

function radicalInverse(index, base) {
  let value = 0;
  let fraction = 1 / base;
  while (index > 0) {
    value += (index % base) * fraction;
    index = Math.floor(index / base);
    fraction /= base;
  }
  return value;
}

export function sampleObjSurfacePoints(objText, count = 1024) {
  const triangles = parseTriangles(objText);
  if (!triangles.length) return Array.from({ length: count }, () => [0, 0, 0]);
  const areas = triangles.map(triangleArea);
  const cumulative = [];
  let totalArea = 0;
  for (const area of areas) {
    totalArea += Math.max(area, 1e-12);
    cumulative.push(totalArea);
  }
  const points = [];
  for (let index = 0; index < count; index += 1) {
    const target = ((index + 0.5) / count) * totalArea;
    let low = 0;
    let high = cumulative.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (cumulative[middle] < target) low = middle + 1;
      else high = middle;
    }
    const triangle = triangles[low];
    const u = radicalInverse(index + 1, 2);
    const v = radicalInverse(index + 1, 3);
    const su = Math.sqrt(u);
    const weights = [1 - su, su * (1 - v), su * v];
    points.push(triangle[0].map((_, axis) => triangle.reduce((sum, vertex, vertexIndex) => sum + vertex[axis] * weights[vertexIndex], 0)));
  }
  const center = [0, 1, 2].map((axis) => points.reduce((sum, point) => sum + point[axis], 0) / points.length);
  const radius = Math.max(1e-8, ...points.map((point) => Math.hypot(point[0] - center[0], point[1] - center[1], point[2] - center[2])));
  return points.map((point) => point.map((value, axis) => (value - center[axis]) / radius));
}

function nearestNeighbors(features, count) {
  return features.map((feature, index) => {
    const best = [];
    for (let other = 0; other < features.length; other += 1) {
      if (other === index) continue;
      let distance = 0;
      for (let axis = 0; axis < feature.length; axis += 1) {
        const delta = feature[axis] - features[other][axis];
        distance += delta * delta;
      }
      if (best.length < count || distance < best.at(-1).distance) {
        let insertion = best.length;
        while (insertion > 0 && distance < best[insertion - 1].distance) insertion -= 1;
        best.splice(insertion, 0, { index: other, distance });
        if (best.length > count) best.pop();
      }
    }
    return best.map((item) => item.index);
  });
}

function edgeConv(features, outputDim, seed, neighborCount = 4) {
  const inputDim = features[0]?.length || 0;
  const weights = seededMatrix(outputDim, inputDim * 2, seed, Math.sqrt(6 / Math.max(1, inputDim * 3)));
  const bias = Array.from({ length: outputDim }, (_, index) => Math.sin((index + 1) * (seed + 1)) * 0.01);
  const neighbors = nearestNeighbors(features, neighborCount);
  return features.map((self, pointIndex) => {
    const pooled = Array(outputDim).fill(-Infinity);
    for (const neighborIndex of neighbors[pointIndex]) {
      const neighbor = features[neighborIndex];
      const edge = [...self, ...neighbor.map((value, axis) => value - self[axis])];
      const encoded = matVec(weights, edge).map((value, axis) => Math.max(0, value + bias[axis]));
      for (let axis = 0; axis < outputDim; axis += 1) pooled[axis] = Math.max(pooled[axis], encoded[axis]);
    }
    return pooled.map((value) => Number.isFinite(value) ? value : 0);
  });
}

export function dgcnnGeometryFeature(objText, options = {}) {
  const pointCount = Number(options.pointCount || 1024);
  const sampled = sampleObjSurfacePoints(objText, pointCount);
  const layer1 = edgeConv(sampled, 16, 1701, Number(options.neighborCount || 4));
  const stride = Math.max(1, Math.floor(layer1.length / 256));
  const hierarchical = layer1.filter((_, index) => index % stride === 0).slice(0, 256);
  const layer2 = edgeConv(hierarchical, 64, 1702, Number(options.neighborCount || 4));
  return Array.from({ length: 64 }, (_, axis) => Math.max(...layer2.map((point) => point[axis])));
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function decodePng(buffer) {
  if (buffer.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error('Invalid PNG');
  let offset = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat = [];
  while (offset + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString('ascii');
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0); height = data.readUInt32BE(4); bitDepth = data[8]; colorType = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    offset += 12 + length;
  }
  if (bitDepth !== 8 || interlace !== 0 || ![0, 2, 4, 6].includes(colorType)) throw new Error(`Unsupported PNG: bitDepth=${bitDepth}, colorType=${colorType}, interlace=${interlace}`);
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  const rowBytes = width * channels;
  const inflated = zlib.inflateSync(Buffer.concat(idat));
  const rows = [];
  let cursor = 0;
  let previous = Buffer.alloc(rowBytes);
  for (let y = 0; y < height; y += 1) {
    const filter = inflated[cursor++];
    const raw = inflated.subarray(cursor, cursor + rowBytes);
    cursor += rowBytes;
    const row = Buffer.alloc(rowBytes);
    for (let x = 0; x < rowBytes; x += 1) {
      const left = x >= channels ? row[x - channels] : 0;
      const up = previous[x] || 0;
      const upperLeft = x >= channels ? previous[x - channels] : 0;
      const predictor = filter === 1 ? left : filter === 2 ? up : filter === 3 ? Math.floor((left + up) / 2) : filter === 4 ? paeth(left, up, upperLeft) : 0;
      row[x] = (raw[x] + predictor) & 255;
    }
    rows.push(row);
    previous = row;
  }
  return { width, height, channels, rows };
}

function resizeRgb(image, size = 32) {
  const output = Array.from({ length: size }, () => Array.from({ length: size }, () => [0, 0, 0]));
  for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    const sourceX = clamp(Math.floor((x + 0.5) / size * image.width), 0, image.width - 1);
    const sourceY = clamp(Math.floor((y + 0.5) / size * image.height), 0, image.height - 1);
    const offset = sourceX * image.channels;
    const row = image.rows[sourceY];
    if (image.channels === 1 || image.channels === 2) output[y][x] = [row[offset] / 255, row[offset] / 255, row[offset] / 255];
    else output[y][x] = [row[offset] / 255, row[offset + 1] / 255, row[offset + 2] / 255];
  }
  return output;
}

function frozenConv(input, inputChannels, outputChannels, seed) {
  const weights = seededMatrix(outputChannels, inputChannels * 9, seed, Math.sqrt(2 / (inputChannels * 9)));
  const height = input.length, width = input[0].length;
  const output = Array.from({ length: height }, () => Array.from({ length: width }, () => Array(outputChannels).fill(0)));
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const patch = [];
    for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) patch.push(...input[clamp(y + dy, 0, height - 1)][clamp(x + dx, 0, width - 1)]);
    output[y][x] = matVec(weights, patch).map((value) => Math.max(0, value));
  }
  return output;
}

function pool2x2(input) {
  const height = Math.floor(input.length / 2), width = Math.floor(input[0].length / 2);
  return Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => {
    const values = [input[y * 2][x * 2], input[y * 2 + 1][x * 2], input[y * 2][x * 2 + 1], input[y * 2 + 1][x * 2 + 1]];
    return values[0].map((_, channel) => Math.max(...values.map((value) => value[channel])));
  }));
}

export function frozenCnnImageFeature(pngBuffer) {
  return frozenCnnRasterFeature(resizeRgb(decodePng(pngBuffer), 32));
}

function frozenCnnRasterFeature(resized) {
  const first = pool2x2(frozenConv(resized, 3, 8, 2601));
  const second = pool2x2(frozenConv(first, 8, 16, 2602));
  const pixels = second.flat();
  const means = Array.from({ length: 16 }, (_, channel) => pixels.reduce((sum, pixel) => sum + pixel[channel], 0) / pixels.length);
  const maxima = Array.from({ length: 16 }, (_, channel) => Math.max(...pixels.map((pixel) => pixel[channel])));
  return [...means, ...maxima];
}

export function renderCleanObjFiveViewRasters(objText, bounds) {
  const geometry = parseObjTriangles(objText);
  const result = [];
  for (const view of MULTI_VIEW_NAMES) {
    const grid = buildDepthGrid(geometry, bounds, view, 64);
    const depths = Array.from(grid.depth).filter(Number.isFinite);
    const minDepth = depths.length ? Math.min(...depths) : 0;
    const maxDepth = depths.length ? Math.max(...depths) : 1;
    const valueAt = (x, y) => grid.depth[clamp(y, 0, 63) * 64 + clamp(x, 0, 63)];
    const image = Array.from({ length: 32 }, (_, y) => Array.from({ length: 32 }, (_, x) => {
      const gx = x * 2 + 1, gy = y * 2 + 1;
      const depth = valueAt(gx, gy);
      if (!Number.isFinite(depth)) return [0, 0, 0];
      const neighborX = valueAt(gx + 1, gy), neighborY = valueAt(gx, gy + 1);
      return [1, clamp((depth - minDepth) / Math.max(maxDepth - minDepth, 1e-8), 0, 1), clamp((Math.abs((Number.isFinite(neighborX) ? neighborX : depth) - depth) + Math.abs((Number.isFinite(neighborY) ? neighborY : depth) - depth)) / Math.max(bounds.radius, 1e-8), 0, 1)];
    }));
    result.push(image);
  }
  return result;
}

export function frozenCleanObjFiveViewFeature(objText, bounds) {
  const features = renderCleanObjFiveViewRasters(objText, bounds).map(frozenCnnRasterFeature);
  return Array.from({ length: 32 }, (_, axis) => features.reduce((sum, feature) => sum + feature[axis], 0) / features.length);
}

export async function frozenMultiviewCnnFeature(files) {
  const selected = files.filter((file) => /-(main|right|left|up|down)\.png$/i.test(file));
  if (selected.length !== 5) throw new Error(`Frozen CNN requires five views; got ${selected.length}`);
  const features = [];
  for (const file of selected) features.push(frozenCnnImageFeature(await fs.readFile(file)));
  return Array.from({ length: 32 }, (_, axis) => features.reduce((sum, feature) => sum + feature[axis], 0) / features.length);
}

export async function buildCvFeatures({ objText, bounds, viewFiles = [] }) {
  if (viewFiles.length) throw new Error('Dataset PNG contains human labels and cannot enter clean-OBJ CV input');
  if (!bounds) throw new Error('Clean OBJ five-view rendering requires bounds');
  const geometry = dgcnnGeometryFeature(objText, { pointCount: 1024, neighborCount: 4 });
  const visual = frozenCleanObjFiveViewFeature(objText, bounds);
  return { geometry, visual, fused_global: [...geometry, ...visual] };
}
