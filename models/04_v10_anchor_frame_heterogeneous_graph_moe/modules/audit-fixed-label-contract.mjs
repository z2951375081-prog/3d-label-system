import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj, boundsFromObj, annotationsToLabels } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { dgcnnGeometryFeature } from '../lib/cv-feature-encoder.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const modelOption = process.argv.indexOf('--model');
const modelFile = modelOption >= 0 && process.argv[modelOption + 1]
  ? path.resolve(root, process.argv[modelOption + 1])
  : path.join(root, 'experiments', 'layout_model.json');
const model = JSON.parse(await fs.readFile(modelFile, 'utf8'));
const leaderLengthPrior = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'manual_leader_length_prior.json'), 'utf8'));
let count = 0, labelsSeen = 0;
for (const sample of manifest.samples) {
  const annotation = JSON.parse(await fs.readFile(path.join(root, sample.target.annotation_json), 'utf8'));
  const manual = annotationsToLabels(annotation);
  const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
  const clean = cleanObj(raw);
  const bounds = boundsFromObj(clean.text);
  const geometry = parseObjTriangles(clean.text);
  const spatialContext = buildSpatialContext(geometry, bounds, { gridSize: model.architecture?.spatial_grid?.grid_size || 20 });
  const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, model);
  const context = sample.category + '/' + sample.sample_id;
  validateFixedLabelContract(manual, candidates, context + '/candidates');
  const styled = applyLayoutModel(candidates, bounds, model, {
    geometry,
    geometryFeature: dgcnnGeometryFeature(clean.text, { pointCount: model.architecture?.obj_surface_points || 1024, neighborCount: model.architecture?.dgcnn_neighbor_count || 4 }),
    spatialContext
  }).labels;
  validateFixedLabelContract(manual, styled, context + '/inference');
  if (['fixed_label_relational_graph_transformer_moe', 'fixed_label_fnn_relational_graph_transformer_moe'].includes(model.architecture?.type)) {
    const expertNames = model.architecture.moe_expert_names;
    assert.ok(Array.isArray(expertNames) && expertNames.length === model.architecture.moe_expert_count, context + '/moe-expert-contract');
    if (model.architecture?.type === 'fixed_label_fnn_relational_graph_transformer_moe') {
      assert.equal(model.architecture.pre_gnn_fnn_layers, 1, context + '/pre-gnn-fnn-depth');
      assert.equal(model.network?.pre_gnn_fnn_layers?.length, 1, context + '/pre-gnn-fnn-weights');
    }
    for (const label of styled) {
      assert.equal(label.layout_model?.architecture, model.architecture?.type === 'fixed_label_fnn_relational_graph_transformer_moe' ? 'fnn_relational_graph_transformer_moe' : 'relational_graph_transformer_moe', context + '/hybrid-runtime-architecture');
      if (model.architecture?.type === 'fixed_label_fnn_relational_graph_transformer_moe') assert.equal(label.layout_model?.pre_gnn_fnn_layers, 1, context + '/pre-gnn-fnn-runtime-depth');
      assert.equal(label.layout_model?.transformer_layers, model.architecture.transformer_layers, context + '/transformer-runtime-depth');
      const weights = expertNames.map((name) => label.layout_model?.moe_router_weights?.[name]);
      assert.ok(weights.every((value) => Number.isFinite(value) && value > 0 && value < 1), context + '/moe-router-positive-weights');
      assert.ok(Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1) < 1e-4, context + '/moe-router-normalized');
    }
  }
  const generated = optimizeLabels(styled, bounds, { fixedLabels: true, groupPolicy: 'all', viewPolicy: 'binocular', sizePolicy: 'relative', optimizer: 'annealing', seed: 17, iterations: 20, category: sample.category, leaderLengthPrior });
  validateFixedLabelContract(manual, generated, context + '/optimization');
  assert.ok(generated.every((item) => item.layout_model?.feature_dim === 51 || (
    item.layout_model?.node_feature_dim === 51
    && (item.layout_model?.edge_feature_dim === 13 || (
      item.layout_model?.graph_type === 'anchor_label_heterogeneous_graph'
      && item.layout_model?.edge_feature_dim === 28
      && model.architecture?.anchor_label_edge_dim === 18
      && model.architecture?.label_label_edge_dim === 10
    ))
  )), context + '/graph-or-legacy-feature-contract');
  count += 1; labelsSeen += generated.length;
}
assert.equal(count, 55);
console.log(JSON.stringify({ model_file: path.relative(root, modelFile).split(path.sep).join('/'), model_version: model.version, sample_count: count, labels_checked: labelsSeen, stages: ['candidates', 'inference', 'optimization'], checks: ['ordered_id', 'exact_text', 'anchor_xyz', 'sourceObjs', 'targetGroups'], status: 'passed' }));
