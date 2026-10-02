function bboxLabel(bounds) {
  return [{ id: 'model-bounding-box', text: 'OBJ MODEL', anchor: [bounds.center[0], bounds.max[1], bounds.center[2]], center: [bounds.center[0], bounds.max[1] + bounds.radius * 0.7, bounds.center[2]], boxSize: [bounds.radius * 1.55, bounds.radius * 0.22, bounds.radius * 0.025], bendPoints: [], sourceObjs: [], targetGroups: [] }];
}

function normalizedGroup(value) {
  return String(value || '').toLowerCase().replace(/\.obj$/i, '').replace(/[^a-z0-9]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function fixedCandidatesFromManual(manualLabels, generatedCandidates, bounds) {
  if (!Array.isArray(manualLabels) || !manualLabels.length) return generatedCandidates;
  const generated = Array.isArray(generatedCandidates) ? generatedCandidates : [];
  return manualLabels.map((manual, index) => {
    const manualKeys = [...(manual.targetGroups || []), ...(manual.sourceObjs || []), manual.text].map(normalizedGroup).filter(Boolean);
    const candidate = generated.find((item) => [...(item.targetGroups || []), ...(item.sourceObjs || []), item.text].map(normalizedGroup).some((key) => manualKeys.includes(key)));
    const anchor = [...manual.anchor];
    let offset;
    if (candidate) offset = candidate.center.map((value, axis) => value - candidate.anchor[axis]);
    else {
      const direction = anchor.map((value, axis) => value - bounds.center[axis]);
      const length = Math.max(Math.hypot(...direction), 1e-6);
      offset = direction.map((value) => value / length * bounds.radius * 0.65);
    }
    return {
      ...manual,
      id: manual.id,
      text: manual.text,
      anchor,
      center: anchor.map((value, axis) => value + offset[axis]),
      boxSize: [...manual.boxSize],
      bendPoints: [],
      sourceObjs: [...(manual.sourceObjs || [])],
      targetGroups: [...(manual.targetGroups || [])],
      candidate_index: index,
      candidate_total: manualLabels.length,
      fixed_label_contract: true,
      initialization_source: candidate ? 'matched_clean_geometry' : 'manual_anchor_radial_fallback'
    };
  });
}

// Only the fixed annotation contract may enter the input; final label positions and
// final box sizes remain supervision targets, never candidate features.
export function fixedCandidatesWithoutTargetLayout(manualLabels, generatedCandidates, bounds) {
  const candidates = fixedCandidatesFromManual(manualLabels, generatedCandidates, bounds);
  if (!manualLabels?.length) return candidates;
  const radius = Math.max(Number(bounds.radius) || 0, 1e-6);
  return candidates.map((candidate) => {
    const chars = [...String(candidate.text || '')].length;
    const width = radius * Math.max(0.22, Math.min(2.4, chars * 0.085));
    const height = radius * 0.16;
    return {
      ...candidate,
      boxSize: [width, height, radius * 0.02],
      size_initialization_source: 'fixed_text_length_and_clean_geometry_scale',
      input_provenance: 'manual_contract_without_adjusted_center_or_box_size'
    };
  });
}

export function fixedCandidatesForLayoutModel(manualLabels, generatedCandidates, bounds, model) {
  if (model?.architecture?.input_provenance === 'manual_contract_without_adjusted_center_or_box_size') {
    return fixedCandidatesWithoutTargetLayout(manualLabels, generatedCandidates, bounds);
  }
  return fixedCandidatesFromManual(manualLabels, generatedCandidates, bounds);
}

function sameStringArray(left, right) {
  return JSON.stringify((left || []).map(String)) === JSON.stringify((right || []).map(String));
}

function sameAnchor(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === 3 && right.length === 3 && left.every((value, axis) => Number.isFinite(Number(value)) && Number.isFinite(Number(right[axis])) && Math.abs(Number(value) - Number(right[axis])) <= 1e-8 * Math.max(1, Math.abs(Number(value))));
}

export function validateFixedLabelContract(manualLabels, generatedLabels, context = 'layout') {
  if (!Array.isArray(manualLabels) || !manualLabels.length) return { fixed: false, count: Array.isArray(generatedLabels) ? generatedLabels.length : 0 };
  if (!Array.isArray(generatedLabels)) throw new Error(`${context}: fixed label output is not an array`);
  if (manualLabels.length !== generatedLabels.length) throw new Error(`${context}: fixed label count mismatch, manual=${manualLabels.length}, generated=${generatedLabels.length}`);
  for (let index = 0; index < manualLabels.length; index += 1) {
    const manual = manualLabels[index];
    const generated = generatedLabels[index];
    if (String(manual.id) !== String(generated?.id)) throw new Error(`${context}: label ${index + 1} id mismatch`);
    if (String(manual.text) !== String(generated?.text)) throw new Error(`${context}: label ${index + 1} text mismatch`);
    if (!sameStringArray(manual.sourceObjs, generated?.sourceObjs)) throw new Error(`${context}: label ${manual.id} sourceObjs mismatch`);
if (!sameStringArray(manual.targetGroups, generated?.targetGroups)) throw new Error(`${context}: label ${manual.id} targetGroups mismatch`);
    if (!sameAnchor(manual.anchor, generated?.anchor)) throw new Error(`${context}: label ${manual.id} anchor mismatch`);
  }
  return { fixed: true, count: manualLabels.length, status: 'fixed_manual_label_contract' };
}

export function generatedCandidatesFromCleanObj(cleanText, bounds) {
  const vertices = [];
  const groups = new Map();
  let currentGroup = 'object';
  for (const line of cleanText.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) { vertices.push(parts.slice(1, 4).map(Number)); continue; }
    if ((parts[0] === 'g' || parts[0] === 'o') && parts.length > 1) { currentGroup = parts.slice(1).join('_'); continue; }
    if (parts[0] !== 'f' || parts.length < 4) continue;
    const indices = parts.slice(1).map((token) => { const raw = Number(token.split('/')[0]); return raw < 0 ? vertices.length + raw : raw - 1; }).filter((index) => vertices[index]);
    if (indices.length < 3) continue;
    if (!groups.has(currentGroup)) groups.set(currentGroup, new Set());
    indices.forEach((index) => groups.get(currentGroup).add(index));
  }
  const candidates = [];
  for (const [group, indexSet] of groups) {
    if (/^(clean_model|object)$/i.test(group)) continue;
    const points = [...indexSet].map((index) => vertices[index]).filter(Boolean);
    if (points.length < 3) continue;
    const min = [0, 1, 2].map((axis) => points.reduce((value, point) => Math.min(value, point[axis]), Infinity));
    const max = [0, 1, 2].map((axis) => points.reduce((value, point) => Math.max(value, point[axis]), -Infinity));
    const center = min.map((value, axis) => (value + max[axis]) / 2);
    const size = max.map((value, axis) => Math.max(value - min[axis], bounds.radius * 0.025));
    const directionRaw = center.map((value, axis) => value - bounds.center[axis]);
    const directionLength = Math.max(Math.hypot(...directionRaw), 1e-6);
    const direction = directionRaw.map((value) => value / directionLength);
    const labelCenter = center.map((value, axis) => value + direction[axis] * (bounds.radius * 0.55 + Math.max(...size) * 0.35));
    candidates.push({ id: `generated-${group}`, text: group.replace(/[_-]+/g, ' ').trim(), anchor: center, center: labelCenter, boxSize: [Math.max(size[0], bounds.radius * 0.18), Math.max(size[1], bounds.radius * 0.08), bounds.radius * 0.02], bendPoints: [], sourceObjs: [group], targetGroups: [group] });
  }
  const result = candidates.length ? candidates : bboxLabel(bounds);
  return result.map((candidate, index) => ({ ...candidate, candidate_index: index, candidate_total: result.length }));
}
