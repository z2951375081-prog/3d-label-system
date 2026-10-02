import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { fixedCandidatesForLayoutModel, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'experiments', 'comparisons', 'round1');
const read = (p) => fs.readFile(path.join(root, p), 'utf8');
const json = async (p) => JSON.parse(await read(p));
const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const std = (a) => { const average = mean(a); return average === null ? null : Math.sqrt(mean(a.map((value) => (value - average) ** 2))); };
const rounded = (n) => n === null ? null : Number(n.toFixed(8));
function seededRandom(seedText) {
 let seed=2166136261;
 for(const char of seedText){seed^=char.charCodeAt(0);seed=Math.imul(seed,16777619);}
 return ()=>{seed+=0x6D2B79F5;let value=seed;value=Math.imul(value^(value>>>15),value|1);value^=value+Math.imul(value^(value>>>7),value|61);return ((value^(value>>>14))>>>0)/4294967296;};
}
function percentile(sorted,p){if(!sorted.length)return null;const index=(sorted.length-1)*p;const low=Math.floor(index),high=Math.ceil(index);return low===high?sorted[low]:sorted[low]+(sorted[high]-sorted[low])*(index-low);}
function pairedBootstrap(rows,referenceMethod,otherMethod,field,direction,iterations=10000){
 const reference=new Map(rows.filter(row=>row.method===referenceMethod&&row.strict_label_contract).map(row=>[row.category+'/'+row.sample_id,row]));
 const pairs=rows.filter(row=>row.method===otherMethod&&row.strict_label_contract&&reference.has(row.category+'/'+row.sample_id)).map(row=>{const ref=Number(reference.get(row.category+'/'+row.sample_id)[field]),other=Number(row[field]);if(!Number.isFinite(ref)||!Number.isFinite(other))return null;return {sample:row.category+'/'+row.sample_id,reference:ref,other,advantage:direction==='higher'?ref-other:other-ref};}).filter(Boolean);
 const differences=pairs.map(pair=>pair.advantage),epsilon=1e-12;
 const rng=seededRandom(referenceMethod+'/'+otherMethod+'/'+field);
 const bootstrap=[];
 for(let iteration=0;iteration<iterations&&differences.length;iteration+=1){let total=0;for(let index=0;index<differences.length;index+=1)total+=differences[Math.floor(rng()*differences.length)];bootstrap.push(total/differences.length);}
 bootstrap.sort((a,b)=>a-b);
 const meanAdvantage=mean(differences);
 return {reference_method:referenceMethod,other_method:otherMethod,metric:field,direction,paired_samples:pairs.length,mean_reference:rounded(mean(pairs.map(pair=>pair.reference))),mean_other:rounded(mean(pairs.map(pair=>pair.other))),mean_advantage_reference_better:rounded(meanAdvantage),reference_wins:differences.filter(value=>value>epsilon).length,ties:differences.filter(value=>Math.abs(value)<=epsilon).length,reference_losses:differences.filter(value=>value< -epsilon).length,bootstrap_iterations:iterations,bootstrap_95_ci:[rounded(percentile(bootstrap,0.025)),rounded(percentile(bootstrap,0.975))],bootstrap_probability_reference_better:rounded(bootstrap.filter(value=>value>0).length/Math.max(1,bootstrap.length)),per_sample:pairs};
}
function csvRows(text) { const lines = text.trim().split(/\r?\n/); const header = lines.shift().split(','); return lines.map(line => Object.fromEntries(line.split(',').map((v, i) => [header[i], v]))); }
const fields = ['PCK_005','PCK_010','OLR','LCD','DBV','avg_leader_length','quality_score','num_labels'];
function summarize(rows, scope) { return [...new Set(rows.map(r => r.method))].map(method => { const subset = rows.filter(r => r.method === method); const item = { scope, method, rows: subset.length, samples: new Set(subset.map(r => r.category + '/' + r.sample)).size }; for (const field of fields) { const values = subset.filter(r => r[field] !== '' && r[field] !== undefined && Number.isFinite(Number(r[field]))).map(r => Number(r[field])); item[field] = rounded(mean(values)); item[field + '_valid_rows'] = values.length; } return item; }); }
const manifest = await json('experiments/dataset_manifest.json');
const layoutModel = await json('experiments/layout_model.json');
const leaderLengthPrior = await json('experiments/manual_leader_length_prior.json');
const datasetCameraCandidate = await json('experiments/layout_model_dataset_camera_candidate.json').catch(()=>null);
const tests = manifest.samples.filter(s => s.split === 'test');
const keys = new Set(tests.map(s => s.category + '/' + s.sample_id));
const bino = csvRows(await read('BinoForce_2025/results/binoforce2025_results.csv'));
const hedge = csvRows(await read('Hedgehog/results/hedgehog_results.csv'));
const sourceSummaries = [ ...summarize(bino, 'all55_five_views'), ...summarize(bino.filter(r => keys.has(r.category + '/' + r.sample)), 'test11_five_views'), ...summarize(bino.filter(r => keys.has(r.category + '/' + r.sample) && r.view === 'main'), 'test11_main') ];
const hedgeIndex = new Map(hedge.map(r => [r.category + '/' + r.sample + '/' + r.view + '/' + r.method, r]));
const imported = bino.filter(r => ['manual','plane','hedgehog_1d','hedgehog_3d'].includes(r.method));
let differences = 0;
for (const row of imported) { const original = hedgeIndex.get(row.category + '/' + row.sample + '/' + row.view + '/' + row.method); if (!original || fields.filter(f => f !== 'DBV').some(f => row[f] !== original[f])) differences++; }
const binoLayouts = (await read('BinoForce_2025/results/binoforce_layouts.jsonl')).trim().split(/\r?\n/).map(JSON.parse);
const unifiedRows = [];
for (const sample of manifest.samples) {
 const rawText = await read(sample.input.source_obj);
 const cleanText = cleanObj(rawText).text;
 const bounds = boundsFromObj(cleanText);
 const geometry = parseObjTriangles(cleanText);
 const options = { viewPolicy: 'binocular', depthGrids: Object.fromEntries(MULTI_VIEW_NAMES.map(view => [view, buildDepthGrid(geometry, bounds, view)])), category: sample.category, leaderLengthPrior };
 const manual = annotationsToLabels(await json(sample.target.annotation_json));
 const fixedCandidates = fixedCandidatesForLayoutModel(manual, generatedCandidatesFromCleanObj(cleanText, bounds), bounds, layoutModel);
 const currentLabels = optimizeLabels(applyLayoutModel(fixedCandidates,bounds,layoutModel).labels, bounds, { ...options, fixedLabels:true, groupPolicy:'all', sizePolicy:'relative', optimizer:'annealing', seed:17, iterations:180 });
 validateFixedLabelContract(manual,currentLabels,sample.category+'/'+sample.sample_id+'/current');
 const layouts = [['manual', manual]];
 if (sample.split === 'test') {
  const folder = 'experiments/artifacts/round1/test/' + sample.category + '/' + sample.sample_id + '/';
  const artifact = await json(folder + 'metadata.json');
  layouts.unshift(['round1_mlp_all', artifact.labels]);
 }
 layouts.push(['current_fixed_label_seed17',currentLabels]);
 if(datasetCameraCandidate) {
  const candidateInputs=fixedCandidatesForLayoutModel(manual,generatedCandidatesFromCleanObj(cleanText,bounds),bounds,datasetCameraCandidate);
  const candidateLabels=optimizeLabels(applyLayoutModel(candidateInputs,bounds,datasetCameraCandidate).labels,bounds,{...options,fixedLabels:true,groupPolicy:'all',sizePolicy:'relative',optimizer:'annealing',seed:17,iterations:180});
  validateFixedLabelContract(manual,candidateLabels,sample.category+'/'+sample.sample_id+'/dataset-camera-candidate');
  layouts.push(['dataset_camera_candidate_seed17',candidateLabels]);
 }
 for (const method of ['hedgehog_1d','hedgehog_3d','plane']) layouts.push([method, annotationsToLabels(await json('Hedgehog/results/layouts/' + sample.category + '/' + sample.sample_id + '/main/' + method + '.json'))]);
 const saved = binoLayouts.find(r => r.category === sample.category && r.sample === sample.sample_id && r.view === 'main');
 if (!saved || saved.label_centers.length !== manual.length || saved.label_centers.some((c,i) => c.text !== manual[i].text)) throw Error('BinoForce exported label order mismatch: ' + sample.category + '/' + sample.sample_id);
 layouts.push(['BinoForce_final_snapshot', manual.map((l,i) => ({...l, center: saved.label_centers[i].center}))]);
 for (const [method, labels] of layouts) {
  const metrics = evaluateLayout(labels, bounds, { ...options, geometry, manualReference: manual });
  unifiedRows.push({
    category: sample.category, sample_id: sample.sample_id, split:sample.split, method, ...metrics,
   strict_label_contract: (() => { try { validateFixedLabelContract(manual,labels,method); return true; } catch { return false; } })(),
   manual_center_distance_norm: metrics.manual_similarity?.manual_center_distance_norm ?? null,
   manual_style_distance: metrics.manual_similarity?.manual_style_distance ?? null,
   manual_size_distance_log: metrics.manual_similarity?.manual_size_distance_log ?? null
  });
 }
}
const unifiedFields = ['label_count','multidimensional_quality_score','objective_score','readability','text_clarity','text_pixel_height','text_fit_ratio','text_clipping_ratio','label_label_occlusion_ratio','label_object_occlusion_ratio','object_label_occlusion_ratio','object_penetration_ratio','mesh_surface_intersection_ratio','lcd','dbv','viewport_overflow_ratio','mean_anchor_distance','leader_length_compliance_ratio','leader_length_shortfall','directional_allocation_mismatch','directional_concentration_excess','directional_uniformity','manual_center_distance_norm','manual_style_distance','manual_size_distance_log'];
const testUnifiedRows = unifiedRows.filter(row => row.split === 'test');
const unifiedSummary = [...new Set(testUnifiedRows.map(r=>r.method))].map(method => {
 const rows = testUnifiedRows.filter(r=>r.method===method);
 return { method, sample_count: rows.length, strict_label_contract: rows.every(row=>row.strict_label_contract), ...Object.fromEntries(unifiedFields.flatMap(field => { const values=rows.filter(row=>row[field]!==null && row[field]!==undefined).map(row=>Number(row[field])).filter(Number.isFinite); return [[field,rounded(mean(values))],[field+'_std',rounded(std(values))]]; })) };
});
const directions = { multidimensional_quality_score:'higher', objective_score:'lower', readability:'higher', text_clarity:'higher', text_pixel_height:'target_not_ranked', text_fit_ratio:'higher', text_clipping_ratio:'lower', label_label_occlusion_ratio:'lower', label_object_occlusion_ratio:'lower', object_label_occlusion_ratio:'lower', object_penetration_ratio:'lower', mesh_surface_intersection_ratio:'lower', lcd:'lower', dbv:'lower', viewport_overflow_ratio:'lower', mean_anchor_distance:'descriptive', leader_length_compliance_ratio:'higher', leader_length_shortfall:'lower', manual_center_distance_norm:'lower', manual_style_distance:'lower', manual_size_distance_log:'lower' };
const rankings = Object.fromEntries(Object.entries(directions).filter(([,direction])=>['lower','higher'].includes(direction)).map(([field,direction])=>[field,unifiedSummary.filter(row=>row.strict_label_contract && row[field]!==null).sort((a,b)=>direction==='lower'?a[field]-b[field]:b[field]-a[field]).map((row,index)=>({rank:index+1,method:row.method,value:row[field]}))]));
const pairedMetrics=['multidimensional_quality_score','objective_score','text_clarity','label_object_occlusion_ratio','object_penetration_ratio','mesh_surface_intersection_ratio','manual_style_distance'];
const pairedOpponents=['hedgehog_1d','hedgehog_3d','plane','current_fixed_label_seed17'];
const pairedComparisons=pairedOpponents.flatMap(method=>pairedMetrics.map(field=>pairedBootstrap(testUnifiedRows,'BinoForce_final_snapshot',method,field,directions[field],10000)));
const report = { generated_at: new Date().toISOString(), camera_protocol: await json('experiments/dataset_camera_protocol.json'), source_protocol: { BinoForce_metrics: '800 warmup frames followed by 300 evaluation frames averaged at each time step, as recorded by BinoForce_2025 reproduction', BinoForce_layout_coordinates: 'final state only; no per-frame coordinates are persisted in binoforce_layouts.jsonl', Hedgehog_metrics: 'static layouts evaluated in the reproduction schema', comparison_boundary: 'Source CSV dynamic averages and unified final-snapshot re-evaluation are reported separately and must not be merged.' }, source_summaries: sourceSummaries, source_audit: { imported_hedgehog_rows: imported.length, differing_import_rows: differences, missing_DBV_is_not_zero: true }, unified_snapshot_evaluation: { protocol: 'All 55 sample rows are reprojected with dataset_multiview_reproduction_v1 (750x500, 50mm/36x24mm, main [1,1,1], +/-45 degrees, distance 10) without re-optimization; rankings, summary, and paired bootstrap remain frozen test11 only.', row_scope: 'all55_for_current_sample_display', summary_scope: 'frozen_test11_only', metric_directions: directions, rankings, paired_bootstrap: { interpretation: 'Positive advantage means BinoForce is better after respecting each metric direction. Percentile intervals resample the same 11 test samples with replacement using deterministic seeds.', comparisons: pairedComparisons }, fairness_limits: ['Dataset files expose image size/view names but not original camera matrices; intrinsics/extrinsics are the shared BinoForce/Hedgehog reproduction assumption.', 'Unified rows use the final BinoForce snapshot because per-frame coordinates were not persisted; the source CSV separately contains 300-frame dynamic metric means.', 'Round1 predicts candidate anchors and sizes; reproduced methods use manual anchors and sizes.', 'Depth penetration is a five-view front/back interval proxy; mesh surface intersection is separately measured by triangle/billboard-OBB SAT.', 'No runtime comparison: reproduction CSVs do not record comparable runtime.'], summary: unifiedSummary, rows: unifiedRows } };
report.unified_snapshot_evaluation.fairness_limits.push('Historical round1_mlp_all violates the fixed label contract and is excluded from rankings. Current fixed-label inference uses the preserved model trained under the former projection; a new-protocol retraining is still required. Active v2 uses human-adjusted box sizes as input, unlike provenance-safe v4.');
report.unified_snapshot_evaluation.mesh_surface_intersection_note = 'mesh_surface_intersection fields use triangle/billboard-OBB SAT independently of the depth proxy. Fully enclosed boxes without surface contact are not detected by SAT.';
await fs.mkdir(output, { recursive: true });
await fs.writeFile(path.join(output,'comparison.json'), JSON.stringify(report,null,2)+'\n');
function csv(items, columns) { return [columns.join(','),...items.map(item=>columns.map(c=>JSON.stringify(item[c]??'')).join(','))].join('\n')+'\n'; }
await fs.writeFile(path.join(output,'source_summary.csv'),csv(sourceSummaries,['scope','method','rows','samples',...fields]));
await fs.writeFile(path.join(output,'unified_snapshot_summary.csv'),csv(unifiedSummary,['method','sample_count',...unifiedFields.flatMap(field=>[field,field+'_std'])]));
await fs.writeFile(path.join(output,'unified_snapshot_rows.csv'),csv(unifiedRows,['category','sample_id','split','method',...unifiedFields]));
await fs.writeFile(path.join(output,'paired_bootstrap.csv'),csv(pairedComparisons,['reference_method','other_method','metric','direction','paired_samples','mean_reference','mean_other','mean_advantage_reference_better','reference_wins','ties','reference_losses','bootstrap_iterations','bootstrap_95_ci','bootstrap_probability_reference_better']));
console.log(JSON.stringify({source_audit:report.source_audit,unifiedSummary},null,2));
