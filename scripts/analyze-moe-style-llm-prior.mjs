import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const STYLE_KEYS = ['spherical', 'rectangular', 'surround'];
function parseArgs(argv) { const options = { llmScores: path.join(experiments, 'moe_style_llm_scores.json'), audit: path.join(experiments, 'moe_style_routing_audit.json'), output: path.join(experiments, 'moe_style_llm_vs_geometry_report.json') }; for (let i=0;i<argv.length;i+=1) if(argv[i].startsWith('--')) { const k=argv[i].slice(2); options[k]=argv[i+1]&&!argv[i+1].startsWith('--')?argv[++i]:true; } return options; }
function mean(v){return v.length?v.reduce((s,x)=>s+x,0)/v.length:0;}
function variance(v){const m=mean(v);return mean(v.map(x=>(x-m)**2));}
function pearson(x,y){if(x.length!==y.length||x.length<3)return null;const mx=mean(x),my=mean(y);const cov=mean(x.map((v,i)=>(v-mx)*(y[i]-my)));const den=Math.sqrt(variance(x)*variance(y));return den>1e-12?cov/den:null;}
function ranks(values){const sorted=values.map((value,index)=>({value,index})).sort((a,b)=>a.value-b.value);const out=Array(values.length);for(let i=0;i<sorted.length;){let j=i+1;while(j<sorted.length&&sorted[j].value===sorted[i].value)j++;const rank=(i+j+1)/2;for(let k=i;k<j;k++)out[sorted[k].index]=rank;i=j;}return out;}
function dominantScores(scores){return STYLE_KEYS.reduce((best,key)=>Number(scores[key])>Number(scores[best])?key:best,STYLE_KEYS[0]);}
function normalizeScores(scores){const values=STYLE_KEYS.map(k=>Number(scores[k]));const min=Math.min(...values), max=Math.max(...values); if(max-min<1e-8) return Object.fromEntries(STYLE_KEYS.map(k=>[k,1/3])); const raw=Object.fromEntries(STYLE_KEYS.map(k=>[k,(Number(scores[k])-min)/(max-min)+0.001])); const total=STYLE_KEYS.reduce((s,k)=>s+raw[k],0); return Object.fromEntries(STYLE_KEYS.map(k=>[k,raw[k]/total]));}
const options=parseArgs(process.argv.slice(2));
const [audit,llm]=await Promise.all([fs.readFile(path.resolve(String(options.audit)),'utf8').then(JSON.parse),fs.readFile(path.resolve(String(options.llmScores)),'utf8').then(JSON.parse)]);
const auditMap=new Map(audit.rows.map(r=>[`${r.category}/${r.sample_id}`,r]));
const rows=llm.rows.map(row=>{const key=`${row.category}/${row.sample_id}`;const geo=auditMap.get(key); const llmWeights=normalizeScores(row.scores); return {category:row.category,sample_id:row.sample_id,split:row.split,geometry_selected:geo?.selected,llm_selected:row.selected,geometry_weights:geo?.weights,llm_scores:row.scores,llm_weights:llmWeights,match:geo?.selected===row.selected,confidence:row.confidence,rationale:row.rationale};}).filter(r=>r.geometry_weights);
const dimensions=Object.fromEntries(STYLE_KEYS.map(key=>{const g=rows.map(r=>Number(r.geometry_weights[key])); const l=rows.map(r=>Number(r.llm_weights[key])); const p=pearson(g,l); const s=pearson(ranks(g),ranks(l)); return [key,{pearson_r:p===null?null:Number(p.toFixed(6)),spearman_rho:s===null?null:Number(s.toFixed(6)),geometry_mean:Number(mean(g).toFixed(6)),llm_mean:Number(mean(l).toFixed(6))}];}));
const confusion=Object.fromEntries(STYLE_KEYS.map(g=>[g,Object.fromEntries(STYLE_KEYS.map(l=>[l,0]))]));
for(const row of rows) confusion[row.geometry_selected][row.llm_selected]+=1;
const report={version:'moe_style_llm_vs_geometry_report_v1',generated_at:new Date().toISOString(),n:rows.length,agreement:Number((rows.filter(r=>r.match).length/Math.max(1,rows.length)).toFixed(6)),dimensions,confusion,counts:{geometry:Object.fromEntries(STYLE_KEYS.map(k=>[k,rows.filter(r=>r.geometry_selected===k).length])),llm:Object.fromEntries(STYLE_KEYS.map(k=>[k,rows.filter(r=>r.llm_selected===k).length]))},interpretation:'This compares LLM visual style labels with geometry-router priors only; it is a calibration screen, not a replacement for the requested human-vs-LLM significance test.',rows};
await fs.writeFile(path.resolve(String(options.output)),JSON.stringify(report,null,2)+'\n','utf8');
console.log(JSON.stringify({ok:true,output:path.relative(root,path.resolve(String(options.output))).replaceAll('\\','/'),n:report.n,agreement:report.agreement,counts:report.counts},null,2));
