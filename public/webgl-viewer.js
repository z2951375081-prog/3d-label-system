import { DATASET_CAMERA_PROTOCOL, datasetCameraForBounds } from './dataset-camera.js';

// Interactive inspection only. This display preset never changes OBJ vertices,
// annotation coordinates, or the fixed five-view dataset camera protocol.
export const DISPLAY_CAMERA_ORIENTATIONS = Object.freeze({
  // Large interactive view starts directly along +Z, with world +Y as screen-up.
  // Fixed dataset five-view cameras remain defined separately in dataset-camera.js.
  upright: Object.freeze({ theta: 0, phi: 0, distance: 3, pan: Object.freeze([0, 0]) })
});

const vertexShaderSource = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
uniform mat4 uMatrix;
out vec3 vColor;
void main() { gl_Position = uMatrix * vec4(aPosition, 1.0); vColor = aColor; }
`;

const fragmentShaderSource = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) for (let row = 0; row < 4; row += 1) out[col * 4 + row] = a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3];
  return out;
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function lookAt(eye, target, up = [0, 1, 0]) {
  let z = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
  let len = Math.max(Math.hypot(...z), 1e-8);
  z = z.map((value) => value / len);
  let x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  len = Math.max(Math.hypot(...x), 1e-8);
  x = x.map((value) => value / len);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]), -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]), -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1]);
}

function meshColorArray(model) {
  const groups = model.triangleGroups || [];
  const colors = new Float32Array(groups.length * 9);
  const base = [0.43, 0.55, 0.70];
  const partColors = model.partColors || {};
  for (let triangle = 0; triangle < groups.length; triangle += 1) {
    const group = groups[triangle];
    let color = partColors[group] || partColors[group?.replace(/\.obj$/i, '')];
    if (!color && group) color = Object.entries(partColors).find(([key]) => group === key || group.includes(key) || key.includes(group))?.[1];
    color ||= base;
    for (let vertex = 0; vertex < 3; vertex += 1) colors.set(color, (triangle * 3 + vertex) * 3);
  }
  return colors;
}

export function cameraFrame(eye, target) {
  const forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const length = Math.max(Math.hypot(...forward), 1e-6);
  for (let index = 0; index < 3; index += 1) forward[index] /= length;
  // right = forward × world-up. The previous opposite sign rolled the
  // interactive camera by 180 degrees and displayed models upside down.
  let right = [-forward[2], 0, forward[0]];
  const rightLength = Math.max(Math.hypot(...right), 1e-6);
  right = right.map((value) => value / rightLength);
  const up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2], right[0] * forward[1] - right[1] * forward[0]];
  return { right, up };
}

function addBillboardBox(target, center, size, frame) {
  const halfWidth = size[0] / 2;
  const halfHeight = size[1] / 2;
  const point = (right, up) => [center[0] + frame.right[0] * halfWidth * right + frame.up[0] * halfHeight * up, center[1] + frame.right[1] * halfWidth * right + frame.up[1] * halfHeight * up, center[2] + frame.right[2] * halfWidth * right + frame.up[2] * halfHeight * up];
  const left = point(-1, -1), right = point(1, -1), topRight = point(1, 1), topLeft = point(-1, 1);
  [left, right, topRight, left, topRight, topLeft].forEach((value) => target.push(...value));
}

function offsetPoint(point, direction, amount) { return point.map((value, index) => value + direction[index] * amount); }

function projectPixel(matrix, point, width, height) {
  const x = point[0], y = point[1], z = point[2];
  const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
  const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
  const clipZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (!Number.isFinite(clipW) || clipW <= 0) return null;
  const ndcX = clipX / clipW, ndcY = clipY / clipW, ndcZ = clipZ / clipW;
  if (!Number.isFinite(ndcX + ndcY + ndcZ)) return null;
  return { x: (ndcX * 0.5 + 0.5) * width, y: (1 - (ndcY * 0.5 + 0.5)) * height, z: ndcZ };
}

function splitText(text, context, maxWidth, maxLines = 2) {
  const words = String(text || '').replace(/_+/g, ' ').trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const lines = [];
  let line = '';
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (line && context.measureText(next).width > maxWidth && lines.length < maxLines - 1) { lines.push(line); line = word; }
    else line = next;
  }
  if (line) lines.push(line);
  return lines.slice(0, maxLines);
}

export class ModelViewer {
  constructor(container) {
    this.container = container;
    container.querySelector('.viewport-empty')?.remove();
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'webgl-canvas';
    this.textCanvas = document.createElement('canvas');
    this.textCanvas.className = 'label-text-canvas';
    container.append(this.canvas, this.textCanvas);
    this.gl = this.canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.textContext = this.textCanvas.getContext('2d');
    if (!this.gl || !this.textContext) throw new Error('当前浏览器不支持 WebGL2 / Canvas2D');
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexShaderSource));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentShaderSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    this.program = program;
    this.positionLocation = gl.getAttribLocation(program, 'aPosition');
    this.colorLocation = gl.getAttribLocation(program, 'aColor');
    this.matrixLocation = gl.getUniformLocation(program, 'uMatrix');
    this.meshPosition = gl.createBuffer(); this.meshColor = gl.createBuffer();
    this.labelPosition = gl.createBuffer(); this.labelColor = gl.createBuffer();
    this.linePosition = gl.createBuffer(); this.lineColor = gl.createBuffer();
    this.model = null; this.labels = []; this.frame = null; this.preset = null;
    this.displayOrientation = 'upright';
    const upright = DISPLAY_CAMERA_ORIENTATIONS[this.displayOrientation];
    this.theta = upright.theta; this.phi = upright.phi; this.distance = upright.distance; this.pan = [...upright.pan]; this.drag = null;
    this.bindControls();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  bindControls() {
    this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    this.canvas.addEventListener('pointerdown', (event) => { this.canvas.setPointerCapture(event.pointerId); this.drag = { x: event.clientX, y: event.clientY, button: event.button }; });
    this.canvas.addEventListener('pointermove', (event) => {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX; this.drag.y = event.clientY;
      if (this.drag.button === 2) { this.pan[0] += dx * 0.004 * this.distance; this.pan[1] -= dy * 0.004 * this.distance; }
      else { this.theta += dx * 0.008; this.phi = Math.max(-1.45, Math.min(1.45, this.phi + dy * 0.008)); }
      this.draw();
    });
    this.canvas.addEventListener('pointerup', () => { this.drag = null; });
    this.canvas.addEventListener('pointercancel', () => { this.drag = null; });
    this.canvas.addEventListener('wheel', (event) => { event.preventDefault(); this.distance = Math.max(0.35, Math.min(25, this.distance * Math.exp(event.deltaY * 0.001))); this.draw(); }, { passive: false });
  }

  resetCamera() {
    const orientation = DISPLAY_CAMERA_ORIENTATIONS[this.displayOrientation] || DISPLAY_CAMERA_ORIENTATIONS.upright;
    this.theta = orientation.theta; this.phi = orientation.phi; this.distance = orientation.distance; this.pan = [...orientation.pan];
    this.draw();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, this.container.clientWidth), height = Math.max(1, this.container.clientHeight);
    for (const canvas of [this.canvas, this.textCanvas]) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); canvas.style.width = `${width}px`; canvas.style.height = `${height}px`; }
    this.draw();
  }

  setPreset(preset) { this.preset = preset || null; this.resetCamera(); }
  setContent(model, labels, frame) { this.model = model; this.labels = labels || []; this.frame = frame || model?.bounds; this.resetCamera(); }

  drawPrimitive(positionBuffer, colorBuffer, positions, colors, mode) {
    const gl = this.gl;
    if (!positions.length) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW); gl.enableVertexAttribArray(this.positionLocation); gl.vertexAttribPointer(this.positionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer); gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW); gl.enableVertexAttribArray(this.colorLocation); gl.vertexAttribPointer(this.colorLocation, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(mode, 0, positions.length / 3);
  }

  drawLabelText(matrix, camera) {
    const context = this.textContext, width = this.textCanvas.width, height = this.textCanvas.height;
    this.textMeasurements = [];
    context.clearRect(0, 0, width, height);
    for (const label of this.labels) {
      const center = projectPixel(matrix, label.center, width, height);
      const horizontal = projectPixel(matrix, offsetPoint(label.center, camera.right, label.boxSize[0] / 2), width, height);
      const vertical = projectPixel(matrix, offsetPoint(label.center, camera.up, label.boxSize[1] / 2), width, height);
      if (!center || !horizontal || !vertical || center.z < -1.2 || center.z > 1.2) continue;
      const boxWidth = Math.max(20, Math.hypot(horizontal.x - center.x, horizontal.y - center.y) * 2);
      const boxHeight = Math.max(12, Math.hypot(vertical.x - center.x, vertical.y - center.y) * 2);
      const text = String(label.text || label.id || '').replace(/_+/g, ' ').trim();
      if (!text) continue;
      let fontSize = Math.max(7, Math.min(24, boxHeight * 0.38));
      let lines = [];
      while (fontSize >= 7) {
        context.font = `700 ${fontSize}px Inter, "Segoe UI", Arial, sans-serif`;
        lines = splitText(text, context, boxWidth * 0.88, boxHeight > fontSize * 2.2 ? 2 : 1);
        if (lines.length && Math.max(...lines.map((line) => context.measureText(line).width)) <= boxWidth * 0.9 && lines.length * fontSize * 1.08 <= boxHeight * 0.88) break;
        fontSize -= 1;
      }
      context.font = `700 ${fontSize}px Inter, "Segoe UI", Arial, sans-serif`;
      context.textAlign = 'center'; context.textBaseline = 'middle'; context.lineJoin = 'round';
      context.lineWidth = Math.max(1.5, fontSize * 0.18); context.strokeStyle = 'rgba(15, 23, 42, .72)'; context.fillStyle = '#ffffff';
      const lineHeight = fontSize * 1.05, firstY = center.y - ((lines.length - 1) * lineHeight) / 2;
      const measuredLines = [];
      lines.forEach((line, index) => {
        const y = firstY + index * lineHeight, measurement = context.measureText(line);
        const compression = Math.min(1, boxWidth * 0.94 / Math.max(1, measurement.width));
        const glyphHeight = (measurement.actualBoundingBoxAscent || fontSize * 0.8) + (measurement.actualBoundingBoxDescent || fontSize * 0.2);
        const glyphWidth = measurement.width * compression;
        const clippedWidth = Math.max(0, center.x - glyphWidth / 2) < width ? Math.max(0, Math.min(width,center.x+glyphWidth/2)-Math.max(0,center.x-glyphWidth/2)) : 0;
        const clippedHeight = Math.max(0, Math.min(height,y+glyphHeight/2)-Math.max(0,y-glyphHeight/2));
        const clipping = 1 - clippedWidth * clippedHeight / Math.max(1,glyphWidth * glyphHeight);
        measuredLines.push({glyph_height_px:glyphHeight,glyph_width_px:glyphWidth,horizontal_scale:compression,clipping_ratio:Math.max(0,Math.min(1,clipping))});
        context.strokeText(line, center.x, y, boxWidth * 0.94); context.fillText(line, center.x, y, boxWidth * 0.94);
      });
      this.textMeasurements.push({label_id:label.id,text,font_size_px:fontSize,line_count:lines.length,canvas_width:width,canvas_height:height,lines:measuredLines});
    }
  }

  draw() {
    const gl = this.gl;
    if (!gl) return;
    if (!this.model || !this.frame) { this.textContext.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height); return; }
    const width = this.canvas.width, height = this.canvas.height;
    gl.viewport(0, 0, width, height); gl.clearColor(0.91, 0.935, 0.97, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.useProgram(this.program);
    const frame = this.frame, center = frame.center, radius = Math.max(frame.radius, 0.001);
    let eye, target, camera, projection;
    if (this.preset) {
      const datasetCamera = datasetCameraForBounds(frame, this.preset);
      eye = datasetCamera.eye; target = datasetCamera.target; camera = datasetCamera;
      projection = perspective(datasetCamera.verticalFovRadians, datasetCamera.sensorWidthMm / datasetCamera.sensorHeightMm, datasetCamera.near, datasetCamera.far);
    } else {
      const direction = [Math.sin(this.theta) * Math.cos(this.phi), Math.sin(this.phi), Math.cos(this.theta) * Math.cos(this.phi)];
      const length = Math.max(Math.hypot(...direction), 1e-6), normalized = direction.map((value) => value / length);
      eye = [center[0] + normalized[0] * this.distance * radius + this.pan[0], center[1] + normalized[1] * this.distance * radius + this.pan[1], center[2] + normalized[2] * this.distance * radius];
      target = [center[0] + this.pan[0], center[1] + this.pan[1], center[2]];
      camera = cameraFrame(eye, target);
      projection = perspective(0.72, width / height, radius * 0.01, radius * 100);
    }
    const view = lookAt(eye, target, camera.up);
    const matrix = mat4Multiply(projection, view);
    gl.uniformMatrix4fv(this.matrixLocation, false, matrix);
    this.drawPrimitive(this.meshPosition, this.meshColor, this.model.triangles, meshColorArray(this.model), gl.TRIANGLES);
    const boxPositions = [], boxColors = [], linePositions = [], lineColors = [];
    this.labels.forEach((label) => {
      addBillboardBox(boxPositions, label.center, label.boxSize, camera);
      const labelColor = label.color || [0.20, 0.76, 0.70];
      for (let index = 0; index < 18; index += 1) boxColors.push(...labelColor);
      const points = [label.anchor, ...(label.bendPoints || []), label.center];
      for (let index = 0; index < points.length - 1; index += 1) { linePositions.push(...points[index], ...points[index + 1]); lineColors.push(...(label.color || [0.95, 0.36, 0.28]), ...(label.color || [0.95, 0.36, 0.28])); }
    });
    gl.disable(gl.DEPTH_TEST);
    this.drawPrimitive(this.labelPosition, this.labelColor, boxPositions, new Float32Array(boxColors), gl.TRIANGLES);
    this.drawPrimitive(this.linePosition, this.lineColor, linePositions, new Float32Array(lineColors), gl.LINES);
    this.drawLabelText(matrix, camera);
  }

  toDataURL(type = 'image/jpeg', quality = 0.82) {
    // Evaluation captures use dataset resolution, never responsive CSS or DPR.
    const previousSize = [this.canvas.width, this.canvas.height];
    if (this.preset) for (const canvas of [this.canvas, this.textCanvas]) {
      canvas.width = DATASET_CAMERA_PROTOCOL.imageWidth;
      canvas.height = DATASET_CAMERA_PROTOCOL.imageHeight;
    }
    this.draw();
    const composite = document.createElement('canvas');
    this.lastCaptureTextMeasurements = structuredClone(this.textMeasurements || []);
    composite.width = this.canvas.width; composite.height = this.canvas.height;
    const context = composite.getContext('2d');
    context.drawImage(this.canvas, 0, 0); context.drawImage(this.textCanvas, 0, 0);
    const output = composite.toDataURL(type, quality);
    if (this.preset) {
      for (const canvas of [this.canvas, this.textCanvas]) { canvas.width = previousSize[0]; canvas.height = previousSize[1]; }
      this.draw();
    }
    return output;
  }
}

export function parseOBJ(text) {
  const vertices = [], triangles = [], triangleGroups = [];
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let currentGroup = 'object';
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed[0] === '#') continue;
    const parts = trimmed.split(/\s+/);
    if ((parts[0] === 'g' || parts[0] === 'o') && parts.length > 1) currentGroup = parts.slice(1).join('_');
    else if (parts[0] === 'v' && parts.length >= 4) {
      const point = parts.slice(1, 4).map(Number); vertices.push(point);
      for (let index = 0; index < 3; index += 1) { min[index] = Math.min(min[index], point[index]); max[index] = Math.max(max[index], point[index]); }
    } else if (parts[0] === 'f' && parts.length >= 4) {
      const indices = parts.slice(1).map((part) => { const raw = Number(part.split('/')[0]); return raw < 0 ? vertices.length + raw : raw - 1; });
      for (let index = 1; index < indices.length - 1; index += 1) { [indices[0], indices[index], indices[index + 1]].forEach((vertexIndex) => triangles.push(...vertices[vertexIndex])); triangleGroups.push(currentGroup); }
    }
  }
  if (!vertices.length) throw new Error('OBJ 中没有找到顶点 v');
  const center = min.map((value, index) => (value + max[index]) / 2), size = max.map((value, index) => value - min[index]);
  const radius = Math.max(Math.hypot(...size) / 2, 0.001);
  return { vertices, triangles, triangleGroups, bounds: { min, max, center, size, radius } };
}
