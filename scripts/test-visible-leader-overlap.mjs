import assert from 'node:assert/strict';
import { evaluateLayout } from '../lib/layout-optimizer.mjs';
const bounds={min:[-1,-1,-1],max:[1,1,1],center:[0,0,0],size:[2,2,2],radius:Math.sqrt(3)};
const base={boxSize:[0.2,0.08,0.02],sourceObjs:[],targetGroups:[]};
const overlapping=[
  {...base,id:'a',text:'a',anchor:[-0.2,0,0],center:[0.6,0,0]},
  {...base,id:'b',text:'b',anchor:[0.2,0,0],center:[1.0,0,0]}
];
const separated=[
  {...base,id:'a',text:'a',anchor:[-0.2,0,0],center:[0.6,0.45,0]},
  {...base,id:'b',text:'b',anchor:[0.2,0,0],center:[1.0,-0.45,0]}
];
const overlapMetrics=evaluateLayout(overlapping,bounds,{viewPolicy:'single'});
const separatedMetrics=evaluateLayout(separated,bounds,{viewPolicy:'single'});
assert.ok(overlapMetrics.visible_leader_overlap_ratio > 0, 'parallel visually-overlapping leaders should be penalized');
assert.equal(separatedMetrics.visible_leader_overlap_ratio,0);
assert.ok(overlapMetrics.objective_score > separatedMetrics.objective_score);
console.log(JSON.stringify({ok:true,overlap:overlapMetrics.visible_leader_overlap_ratio,separated:separatedMetrics.visible_leader_overlap_ratio,overlap_objective:overlapMetrics.objective_score,separated_objective:separatedMetrics.objective_score},null,2));
