// Exact triangle/OBB surface test (separating-axis theorem), not a screen
// overlap heuristic. Billboard box orientation is camera-specific.
// Fully enclosed boxes with no surface contact require a separate volume test.
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub = (a,b) => a.map((value,index)=>value-b[index]);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const triangleBoundsCache = new WeakMap();
function triangleBounds(geometry) {
  if(triangleBoundsCache.has(geometry)) return triangleBoundsCache.get(geometry);
  const bounds = new Float64Array(geometry.triangles.length*6);
  for(let index=0;index<geometry.triangles.length;index++) {
    const triangle=geometry.triangles[index];
    for(let axis=0;axis<3;axis++) {
      bounds[index*6+axis]=Math.min(triangle[0][axis],triangle[1][axis],triangle[2][axis]);
      bounds[index*6+axis+3]=Math.max(triangle[0][axis],triangle[1][axis],triangle[2][axis]);
    }
  }
  triangleBoundsCache.set(geometry,bounds);
  return bounds;
}

export function triangleIntersectsLabelBox(triangle,label,basis) {
  const axes = [basis.right,basis.up,basis.forward];
  const half = label.boxSize.map(value=>Math.max(Math.abs(value)/2,1e-8));
  const vertices = triangle.map(point=>axes.map(axis=>dot(sub(point,label.center),axis)));
  for(let axis=0;axis<3;axis++) if(Math.min(...vertices.map(v=>v[axis]))>half[axis] || Math.max(...vertices.map(v=>v[axis])) < -half[axis]) return false;
  const edges = [sub(vertices[1],vertices[0]),sub(vertices[2],vertices[1]),sub(vertices[0],vertices[2])];
  const boxAxes = [[1,0,0],[0,1,0],[0,0,1]];
  const testAxes = [cross(edges[0],edges[1]),...edges.flatMap(edge=>boxAxes.map(axis=>cross(edge,axis)))];
  for(const axis of testAxes) {
    if(dot(axis,axis)<1e-20) continue;
    const projected = vertices.map(vertex=>dot(vertex,axis));
    const radius = half.reduce((sum,value,index)=>sum+value*Math.abs(axis[index]),0);
    if(Math.min(...projected)>radius+1e-10 || Math.max(...projected)<-radius-1e-10) return false;
  }
  return true;
}

export function meshLabelSurfaceIntersection(labels,geometry,basis) {
  if(!geometry?.triangles) return null;
  const bounds=triangleBounds(geometry);
  let intersecting = 0;
  for(const label of labels) {
    // Broad phase before transforming expensive mesh triangles.
    const radius = Math.hypot(...label.boxSize)/2;
    let hit=false;
    for(let index=0;index<geometry.triangles.length;index++) {
      const offset=index*6;
      if(bounds[offset]>label.center[0]+radius || bounds[offset+3]<label.center[0]-radius ||
         bounds[offset+1]>label.center[1]+radius || bounds[offset+4]<label.center[1]-radius ||
         bounds[offset+2]>label.center[2]+radius || bounds[offset+5]<label.center[2]-radius) continue;
      if(triangleIntersectsLabelBox(geometry.triangles[index],label,basis)) {hit=true;break;}
    }
    if(hit) intersecting++;
  }
  return { intersecting_label_count:intersecting, surface_intersection_ratio:intersecting/Math.max(1,labels.length), method:'triangle_billboard_obb_sat', fully_enclosed_volume_detection:false };
}
