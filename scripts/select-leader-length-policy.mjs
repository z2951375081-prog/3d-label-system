import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const manifest = await readJson(path.join(experiments, 'dataset_manifest.json'));
const layoutModel = await readJson(path.join(experiments, 'layout_model.json'));
const basePrior = await readJson(path.join(experiments, 'manual_leader_length_prior.json'));
const outputFile = path.join(experiments, 'leader_length_policy_selection.json');
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const round = (value) => value === null ? null : Number(value.toFixed(6));

const configurations = [
  { id: 'legacy_no_range', enabled: false },
  { id: 'train_p10_p90', enabled: true, preferred_min_quantile: 'p10' },
  { id: 'train_p20_p90', enabled: true, preferred_min_quantile: 'p20' },
  { id: 'train_p25_p90', enabled: true, preferred_min_quantile: 'p25' }
];

function priorFor(configuration) {
  if (!configuration.enabled) return null;
  return { ...basePrior, selected_policy: { ...basePrior.selected_policy, id: `${configuration.id}_v1`, preferred_min_quantile: configuration.preferred_min_quantile } };
}

async function evaluateSplit(split, selectedConfigurations) {
  const rows = [];
  for (const sample of manifest.samples.filter((item) => item.split === split)) {
    const raw = await fs.readFile(path.join(root, sample.input.source_obj), 'utf8');
    const clean = cleanObj(raw), bounds = boundsFromObj(clean.text), geometry = parseObjTriangles(clean.text);
    const manual = annotationsToLabels(await readJson(path.join(root, sample.target.annotation_json)));
    const candidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(clean.text, bounds), bounds, layoutModel);
    validateFixedLabelContract(manual, candidates, `${split}/${sample.category}/${sample.sample_id}/candidates`);
    const styled = applyLayoutModel(candidates, bounds, layoutModel).labels;
    const depthGrids = Object.fromEntries(MULTI_VIEW_NAMES.map((view) => [view, buildDepthGrid(geometry, bounds, view)]));
    for (const configuration of selectedConfigurations) {
      const generationPrior = priorFor(configuration);
      const optimizerOptions = { viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', fixedLabels: true, iterations: 180, seed: 17, depthGrids, category: sample.category, leaderLengthPrior: generationPrior };
      const labels = optimizeLabels(styled, bounds, optimizerOptions);
      validateFixedLabelContract(manual, labels, `${split}/${sample.category}/${sample.sample_id}/${configuration.id}`);
      // Every layout is evaluated against the selected train-derived band so
      // objective values are comparable even for the legacy generator.
      const evaluationPrior = generationPrior || priorFor(configurations.find((item) => item.id === 'train_p20_p90'));
      const metrics = evaluateLayout(labels, bounds, { ...optimizerOptions, leaderLengthPrior: evaluationPrior, geometry, manualReference: manual });
      rows.push({ split, category: sample.category, sample_id: String(sample.sample_id), configuration: configuration.id, metrics });
    }
  }
  return rows;
}

const fields = ['multidimensional_quality_score', 'objective_score', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_label_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'viewport_overflow_ratio', 'multi_view_worst_overflow', 'mean_anchor_distance', 'leader_length_compliance_ratio', 'leader_length_shortfall', 'manual_center_distance_norm', 'manual_style_distance'];

function summarize(rows, configuration) {
  const selected = rows.filter((row) => row.configuration === configuration.id);
  const summary = { configuration: configuration.id, sample_count: selected.length };
  for (const field of fields) {
    const values = selected.map((row) => row.metrics[field] ?? row.metrics.manual_similarity?.[field]).filter((value) => Number.isFinite(Number(value))).map(Number);
    summary[field] = round(mean(values));
  }
  const component = (name) => selected.map((row) => row.metrics.manual_similarity?.style_components?.[name]).filter((value) => Number.isFinite(Number(value))).map(Number);
  summary.manual_leader_mean_difference = round(mean(component('mean_leader_length_difference')));
  summary.manual_leader_spread_difference = round(mean(component('leader_length_spread_difference')));
  return summary;
}

const validationRows = await evaluateSplit('val', configurations);
const validation = configurations.map((configuration) => summarize(validationRows, configuration));
const baseline = validation.find((item) => item.configuration === 'legacy_no_range');
for (const item of validation) {
  item.constraints = {
    text_clarity_change: round(item.text_clarity - baseline.text_clarity),
    label_overlap_change: round(item.label_label_occlusion_ratio - baseline.label_label_occlusion_ratio),
    label_object_occlusion_change: round(item.label_object_occlusion_ratio - baseline.label_object_occlusion_ratio),
    object_label_occlusion_change: round(item.object_label_occlusion_ratio - baseline.object_label_occlusion_ratio),
    penetration_change: round(item.object_penetration_ratio - baseline.object_penetration_ratio),
    mesh_intersection_change: round(item.mesh_surface_intersection_ratio - baseline.mesh_surface_intersection_ratio),
    worst_overflow_change: round(item.multi_view_worst_overflow - baseline.multi_view_worst_overflow)
  };
  item.safety_pass = item.configuration !== 'legacy_no_range'
    && item.constraints.text_clarity_change >= -0.10
    && item.constraints.label_overlap_change <= 0.03
    && item.constraints.label_object_occlusion_change <= 0.01
    && item.constraints.object_label_occlusion_change <= 0.06
    && item.constraints.penetration_change <= 0.005
    && item.constraints.mesh_intersection_change <= 0.005
    && item.constraints.worst_overflow_change <= 0.01;
  item.selection_score = item.configuration === 'legacy_no_range' ? null : round(
    (item.leader_length_compliance_ratio - baseline.leader_length_compliance_ratio) * 1.5
    + (baseline.leader_length_shortfall - item.leader_length_shortfall) * 2
    + (baseline.manual_leader_mean_difference - item.manual_leader_mean_difference)
    + (item.multidimensional_quality_score - baseline.multidimensional_quality_score) * 0.25
  );
}

const eligible = validation.filter((item) => item.safety_pass).sort((left, right) => right.selection_score - left.selection_score || left.objective_score - right.objective_score);
const selected = eligible[0];
if (!selected) throw new Error('没有引导线长度方案通过 val11 安全门控，保持旧策略');
const selectedConfiguration = configurations.find((item) => item.id === selected.configuration);
const testRows = await evaluateSplit('test', [configurations[0], selectedConfiguration]);
const test = [configurations[0], selectedConfiguration].map((configuration) => summarize(testRows, configuration));

basePrior.selected_policy = {
  ...basePrior.selected_policy,
  id: `${selectedConfiguration.id}_v1`,
  preferred_min_quantile: selectedConfiguration.preferred_min_quantile,
  rationale: `${selectedConfiguration.preferred_min_quantile.toUpperCase()} lower bound and P90 preferred upper bound were selected on val11 from predeclared P10/P20/P25 candidates. Test11 was evaluated only after selection.`
};
basePrior.validation_selection = {
  selected_without_test: true,
  selected_configuration: selected.configuration,
  criterion: 'maximize train-derived leader band compliance and manual leader rhythm on val11 subject to deterministic safety non-regression',
  candidates: validation
};
basePrior.test_confirmation = { used_after_selection: true, selected_configuration: selected.configuration, summary: test };
await fs.writeFile(path.join(experiments, 'manual_leader_length_prior.json'), `${JSON.stringify(basePrior, null, 2)}\n`, 'utf8');
const report = { version: 'leader_length_policy_selection_v1', generated_at: new Date().toISOString(), distribution_source: 'train33 manual_adjusted only', selected_without_test: true, selected_configuration: selected.configuration, validation, test_confirmation: test, rows: { val: validationRows, test: testRows } };
await fs.writeFile(outputFile, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ selected: selected.configuration, validation, test_confirmation: test, output: path.relative(root, outputFile).split(path.sep).join('/') }, null, 2));
