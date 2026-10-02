import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotationsToLabels, boundsFromObj, cleanObj } from './generate-artifacts.mjs';
import { fixedCandidatesWithoutTargetLayout, generatedCandidatesFromCleanObj, validateFixedLabelContract } from '../lib/candidate-generator.mjs';
import { applyLayoutModel } from '../lib/layout-model.mjs';
import { MULTI_VIEW_NAMES, buildDepthGrid, evaluateLayout, optimizeLabels, parseObjTriangles } from '../lib/layout-optimizer.mjs';
import { buildCvFeatures } from '../lib/cv-feature-encoder.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const readJson=async relative=>JSON.parse(await fs.readFile(path.join(root,relative),'utf8'));
function parseArgs(argv){const options={model:'experiments/layout_model_provenance_safe_candidate.json',previous:'experiments/layout_model.json',output:'experiments/multiview_provenance_safe_inference_selection.json'};for(let index=0;index<argv.length;index++){if(!argv[index].startsWith('--'))continue;const key=argv[index].slice(2);options[key]=argv[index+1]&&!argv[index+1].startsWith('--')?argv[++index]:true;}return options;}
const args=parseArgs(process.argv.slice(2));
const manifest=await readJson('experiments/dataset_manifest.json');
const model=await readJson(args.model),previous=await readJson(args.previous);

async function prepare(split){
 const result=[];
 for(const sample of manifest.samples.filter(item=>item.split===split)){
  const raw=await fs.readFile(path.join(root,sample.input.source_obj),'utf8');
  const clean=cleanObj(raw),bounds=boundsFromObj(clean.text),geometry=parseObjTriangles(clean.text);
  const cvFeatures=await buildCvFeatures({objText:clean.text,bounds});
  const manual=annotationsToLabels(await readJson(sample.target.annotation_json));
  const labels=fixedCandidatesWithoutTargetLayout(manual,generatedCandidatesFromCleanObj(clean.text,bounds),bounds);
  validateFixedLabelContract(manual,labels,sample.category+'/'+sample.sample_id);
  const depthGrids=Object.fromEntries(MULTI_VIEW_NAMES.map(view=>[view,buildDepthGrid(geometry,bounds,view)]));
  result.push({sample,bounds,geometry,cvFeatures,manual,labels,options:{viewPolicy:'binocular',groupPolicy:'all',sizePolicy:'relative',fixedLabels:true,optimizer:'annealing',seed:17,iterations:180,depthGrids}});
 }
 return result;
}
function run(prepared,layoutModel,{exact=false}={}){
 return prepared.map(item=>{
  const styled=layoutModel?applyLayoutModel(item.labels,item.bounds,layoutModel,{geometryFeature:item.cvFeatures.geometry,visualFeature:item.cvFeatures.visual}).labels:item.labels;
  const labels=optimizeLabels(styled,item.bounds,item.options);
  validateFixedLabelContract(item.manual,labels,item.sample.category+'/'+item.sample.sample_id);
  const metrics=evaluateLayout(labels,item.bounds,{...item.options,manualReference:item.manual,geometry:exact?item.geometry:undefined});
  return {category:item.sample.category,sample_id:item.sample.sample_id,...metrics,manual_center_distance_norm:metrics.manual_similarity?.manual_center_distance_norm??null,manual_style_distance:metrics.manual_similarity?.manual_style_distance??null};
 });
}
const fields=['multidimensional_quality_score','objective_score','olr','lcd','readability','text_clarity','label_label_occlusion_ratio','label_object_occlusion_ratio','object_label_occlusion_ratio','object_penetration_ratio','mesh_surface_intersection_ratio','manual_center_distance_norm','manual_style_distance','viewport_overflow_ratio','multi_view_worst_olr'];
function mean(rows,key){const values=rows.map(row=>row[key]).filter(value=>value!==null&&value!==undefined&&Number.isFinite(Number(value))).map(Number);return values.length?values.reduce((sum,value)=>sum+value,0)/values.length:null;}
function aggregate(rows){return Object.fromEntries(fields.map(key=>{const value=mean(rows,key);return [key,value===null?null:Number(value.toFixed(6))];}));}
function withInference(base,center_blend,size_blend,status='validation_search'){const candidate=structuredClone(base);candidate.inference={center_blend,size_blend,size_ratio_range:[0.65,1.35],selection_status:status};return candidate;}

const validation=await prepare('val');
const baselineSearch=aggregate(run(validation,null));
const previousSearch=aggregate(run(validation,previous));
const candidates=[];
for(const center_blend of [0,0.1,0.2,0.35,0.5,0.75,1])for(const size_blend of [0,0.15,0.3,0.6,1]){
 const candidate=withInference(model,center_blend,size_blend);
 candidates.push({center_blend,size_blend,...aggregate(run(validation,candidate))});
}
candidates.sort((a,b)=>b.multidimensional_quality_score-a.multidimensional_quality_score || a.objective_score-b.objective_score);
const searchedBest=candidates[0];
const selectedModel=withInference(model,searchedBest.center_blend,searchedBest.size_blend);
const baseline=aggregate(run(validation,null,{exact:true}));
const previousValidation=aggregate(run(validation,previous,{exact:true}));
const best=aggregate(run(validation,selectedModel,{exact:true}));
const qualityReference=Math.max(baseline.multidimensional_quality_score,previousValidation.multidimensional_quality_score);
const objectiveReference=Math.min(baseline.objective_score,previousValidation.objective_score);
const constraints={
 quality_gain:best.multidimensional_quality_score-qualityReference,
 quality_gain_vs_baseline:best.multidimensional_quality_score-baseline.multidimensional_quality_score,
 quality_gain_vs_previous:best.multidimensional_quality_score-previousValidation.multidimensional_quality_score,
 objective_relative_change:best.objective_score/objectiveReference-1,
 objective_relative_change_vs_baseline:best.objective_score/baseline.objective_score-1,
 objective_relative_change_vs_previous:best.objective_score/previousValidation.objective_score-1,
 label_object_occlusion_change:best.label_object_occlusion_ratio-baseline.label_object_occlusion_ratio,
 label_object_occlusion_change_vs_previous:best.label_object_occlusion_ratio-previousValidation.label_object_occlusion_ratio,
 depth_penetration_change:best.object_penetration_ratio-baseline.object_penetration_ratio,
 depth_penetration_change_vs_previous:best.object_penetration_ratio-previousValidation.object_penetration_ratio,
 mesh_surface_intersection_change:best.mesh_surface_intersection_ratio-baseline.mesh_surface_intersection_ratio,
 mesh_surface_intersection_change_vs_previous:best.mesh_surface_intersection_ratio-previousValidation.mesh_surface_intersection_ratio
};
const cvHybridReplacement=model?.architecture?.type==='fixed_label_dgcnn_cnn_fnn_relational_graph_transformer_moe';
const fnnHybridReplacement=model?.architecture?.type==='fixed_label_fnn_relational_graph_transformer_moe'||cvHybridReplacement;
const hybridReplacement=model?.architecture?.type==='fixed_label_relational_graph_transformer_moe'||fnnHybridReplacement;
const graphReplacement=model?.architecture?.type==='fixed_label_multiview_relational_gnn'||hybridReplacement;
const qualityAccepted=hybridReplacement
 ? constraints.quality_gain_vs_previous>=0
 : graphReplacement
 ? constraints.quality_gain_vs_baseline>=-0.005&&constraints.quality_gain_vs_previous>=0.02
 : constraints.quality_gain>=0.02;
const objectiveAccepted=hybridReplacement
 ? constraints.objective_relative_change_vs_previous<=0
 : graphReplacement
 ? constraints.objective_relative_change_vs_baseline<=0&&constraints.objective_relative_change_vs_previous<=-0.01
 : constraints.objective_relative_change<=-0.01;
const routingWeights=Object.values(model?.training?.routing?.val?.average_weights||{});
const architectureAccepted=!fnnHybridReplacement||(model?.architecture?.pre_gnn_fnn_layers===1&&model?.network?.pre_gnn_fnn_layers?.length===1&&model?.architecture?.moe_expert_count===4&&model?.network?.moe?.experts?.length===4&&model?.training?.parameter_updates?.pre_gnn_fnn?.changed_parameters>0&&model?.training?.functional_evidence?.pre_gnn_fnn_ablation?.max_abs_output_change>1e-8&&(!cvHybridReplacement||(model?.architecture?.obj_surface_points===1024&&model?.architecture?.dgcnn_edgeconv_layers>=2&&model?.architecture?.geometry_feature_dim===64&&model?.architecture?.visual_feature_dim===32&&model?.architecture?.fused_feature_dim===160&&model?.network?.fusion?.weights?.[0]?.length===160&&model?.training?.parameter_updates?.feature_fusion?.changed_parameters>0&&model?.training?.functional_evidence?.cv_fusion_ablation?.max_abs_output_change>1e-8)));
const routingAccepted=!hybridReplacement||(model?.training?.parameter_updates?.transformer?.changed_parameters>0&&model?.training?.parameter_updates?.moe_router?.changed_parameters>0&&routingWeights.length===model?.architecture?.moe_expert_count&&routingWeights.every(value=>value>0.01)&&Math.abs(routingWeights.reduce((sum,value)=>sum+value,0)-1)<1e-3);
const safetyAccepted=hybridReplacement
 ? constraints.label_object_occlusion_change_vs_previous<=0&&constraints.depth_penetration_change_vs_previous<=0&&constraints.mesh_surface_intersection_change_vs_previous<=0
 : constraints.label_object_occlusion_change<=0.01&&constraints.depth_penetration_change<=0.005&&constraints.mesh_surface_intersection_change<=0.005;
const accepted=(searchedBest.center_blend>0||searchedBest.size_blend>0)&&architectureAccepted&&qualityAccepted&&objectiveAccepted&&safetyAccepted&&routingAccepted;
model.inference={center_blend:searchedBest.center_blend,size_blend:searchedBest.size_blend,size_ratio_range:[0.65,1.35],selection_status:accepted?'accepted_on_multidimensional_validation':'rejected_on_multidimensional_validation'};
model.validation_gate={status:accepted?'accepted':'rejected',criterion:cvHybridReplacement?'v8 replacement: clean-OBJ unlabeled five-view CNN and 1024-point two-layer EdgeConv are fused into 160D; trained fusion and val ablation required; val quality/objective and safety must be no worse than active v7':fnnHybridReplacement?'v7 replacement: validation quality and objective must both be no worse than previous active; no safety regression; real trained FNN, Transformer, MoE updates and routing required':hybridReplacement?'v6 replacement: validation quality and objective must both be no worse than the previous active model; safety metrics must not regress and trained routing must be non-degenerate':graphReplacement?'graph replacement: val quality no worse than no-model by 0.005, quality +0.02 over previous active, objective no worse than no-model and -1% versus previous, with strict safety tolerances':'val multidimensional quality +0.02 and objective -1%, with strict safety tolerances',camera_protocol:model.camera_protocol||null,search_without_exact_mesh:true,baseline,previous_active:previousValidation,quality_reference:qualityReference,objective_reference:objectiveReference,search_selected:searchedBest,selected:best,constraints,architecture_accepted:architectureAccepted,routing_accepted:routingAccepted,safety_accepted:safetyAccepted,searched_candidates:candidates.length,selected_without_test:true};
await fs.writeFile(path.join(root,args.model),JSON.stringify(model,null,2)+'\n','utf8');

const test=await prepare('test');
const testReport={no_model:aggregate(run(test,null,{exact:true})),previous_active:aggregate(run(test,previous,{exact:true})),selected_candidate:aggregate(run(test,model,{exact:true}))};
const report={generated_at:new Date().toISOString(),candidate_model_file:String(args.model).replaceAll('\\','/'),comparison_model_file:String(args.previous).replaceAll('\\','/'),validation:{baseline_search:baselineSearch,previous_search:previousSearch,candidates,search_selected:searchedBest,baseline,previous_active:previousValidation,selected:best,constraints,accepted},test_confirmation:testReport,policy:{activation_requires_multidimensional_validation_acceptance:true,test_not_used_for_blend_selection_or_activation:true,active_model_replaced:false}};
await fs.writeFile(path.join(root,args.output),JSON.stringify(report,null,2)+'\n','utf8');
console.log(JSON.stringify({validation:{baseline,previous_active:previousValidation,selected:best,constraints,accepted},test_confirmation:testReport},null,2));
