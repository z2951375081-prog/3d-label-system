import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { acceptDirectionalAlternative } from '../lib/adaptive-directional-rerank.mjs';

const base = { directional_uniformity: 0.70, directional_allocation_mismatch: 0.45, intrinsic_quality_score: 4.8,
  text_clarity: 4.6, label_label_occlusion_ratio: 0.01, label_object_occlusion_ratio: 0.01,
  object_label_occlusion_ratio: 0.02, object_penetration_ratio: 0, mesh_surface_intersection_ratio: 0,
  multi_view_worst_overflow: 0.001, leader_length_compliance_ratio: 0.95 };
assert.equal(acceptDirectionalAlternative(base, { ...base, directional_uniformity: 0.72, directional_allocation_mismatch: 0.43 }), true);
assert.equal(acceptDirectionalAlternative(base, { ...base, directional_uniformity: 0.72, directional_allocation_mismatch: 0.43, object_penetration_ratio: 0.001 }), false);
assert.equal(acceptDirectionalAlternative(base, { ...base, directional_uniformity: 0.72, directional_allocation_mismatch: 0.43, mesh_surface_intersection_ratio: 0.001 }), false);
const evidence = JSON.parse(await fs.readFile(new URL('../experiments/adaptive_directional_rerank_offline.json', import.meta.url), 'utf8'));
assert.equal(evidence.selection.passed, true);
assert.ok(evidence.test_confirmation.summary.directional_uniformity.delta > 0);
assert.ok(evidence.test_confirmation.summary.directional_allocation_mismatch.delta < 0);
assert.equal(evidence.active_runtime_modified, false);
console.log(JSON.stringify({ status: 'passed', test11_uniformity_delta: evidence.test_confirmation.summary.directional_uniformity.delta, test11_mismatch_delta: evidence.test_confirmation.summary.directional_allocation_mismatch.delta }, null, 2));
