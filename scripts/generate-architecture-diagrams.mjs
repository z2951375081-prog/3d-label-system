import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const outputDir = path.join(root, 'public', 'architecture-diagrams');
fs.mkdirSync(outputDir, { recursive: true });

const node = (id, x, y, w, h, lines, tone = 'blue') => ({ id, x, y, w, h, lines, tone });
const edge = (from, to, options = {}) => ({ from, to, ...options });
const tones = {
  blue: { fill: '#e4f2ff', stroke: '#cbdbe9', text: '#2791ff' },
  green: { fill: '#e8f8f2', stroke: '#c3e3d7', text: '#168a70' },
  amber: { fill: '#fff4dc', stroke: '#ead8ae', text: '#b87713' },
  violet: { fill: '#f0eaff', stroke: '#d9ccef', text: '#7656b5' },
  gray: { fill: '#f2f5f8', stroke: '#d7dee6', text: '#65778a' }
};

function escapeXml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

function renderBox(item) {
  const palette = tones[item.tone] || tones.blue;
  const lineHeight = 25;
  const firstY = item.y + item.h / 2 - ((item.lines.length - 1) * lineHeight) / 2 + 7;
  const tspans = item.lines.map((line, index) => '<tspan x="' + (item.x + item.w / 2) + '" y="' + (firstY + index * lineHeight) + '">' + escapeXml(line) + '</tspan>').join('');
  return '<g><rect x="' + item.x + '" y="' + item.y + '" width="' + item.w + '" height="' + item.h + '" rx="24" fill="' + palette.fill + '" stroke="' + palette.stroke + '" stroke-width="1.6"/><text class="box-text" text-anchor="middle">' + tspans + '</text></g>';
}

function renderEdge(link, byId) {
  const from = byId.get(link.from);
  const to = byId.get(link.to);
  const x1 = from.x + from.w / 2;
  const y1 = from.y + from.h;
  const x2 = to.x + to.w / 2;
  const y2 = to.y;
  if (link.via) {
    const points = [[x1, y1], ...link.via, [x2, y2]];
    return '<polyline points="' + points.map((point) => point.join(',')).join(' ') + '" class="edge"/>';
  }
  if (Math.abs(x1 - x2) < 2) return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="edge"/>';
  const middleY = link.middleY || Math.round((y1 + y2) / 2);
  return '<path d="M ' + x1 + ' ' + y1 + ' V ' + middleY + ' H ' + x2 + ' V ' + y2 + '" class="edge"/>';
}

function renderDiagram(diagram) {
  const byId = new Map(diagram.nodes.map((item) => [item.id, item]));
  const subtitle = diagram.subtitle ? '<text class="subtitle" x="380" y="76" text-anchor="middle">' + escapeXml(diagram.subtitle) + '</text>' : '';
  return '<?xml version="1.0" encoding="UTF-8"?>\n'
    + '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="' + diagram.height + '" viewBox="0 0 760 ' + diagram.height + '">'
    + '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6.5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#9aa3aa"/></marker>'
    + '<filter id="shadow" x="-10%" y="-10%" width="120%" height="125%"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#8ca0b3" flood-opacity="0.13"/></filter></defs>'
    + '<style>svg{background:#fff}.title{font:700 28px Microsoft YaHei,Noto Sans SC,Arial,sans-serif;fill:#274866}.subtitle{font:500 14px Microsoft YaHei,Noto Sans SC,Arial,sans-serif;fill:#7b8b99}.box-text{font:700 17px Microsoft YaHei,Noto Sans SC,Arial,sans-serif;fill:#2791ff}.edge{fill:none;stroke:#9aa3aa;stroke-width:1.5;marker-end:url(#arrow)}rect{filter:url(#shadow)}</style>'
    + '<rect width="760" height="' + diagram.height + '" fill="#ffffff"/>'
    + '<text class="title" x="380" y="42" text-anchor="middle">' + escapeXml(diagram.title) + '</text>' + subtitle
    + diagram.edges.map((link) => renderEdge(link, byId)).join('')
    + diagram.nodes.map(renderBox).join('')
    + '</svg>\n';
}

const baseV10Nodes = [
  node('obj', 40, 130, 275, 82, ['Clean OBJ']),
  node('contract', 420, 92, 300, 92, ['固定标签契约', '文字、ID、锚点、语义关系']),
  node('sample', 28, 250, 300, 86, ['确定性采样 1024 个表面点']),
  node('patch', 430, 222, 280, 82, ['锚点表面 Patch']),
  node('dgcnn', 40, 374, 275, 86, ['2层 DGCNN EdgeConv']),
  node('pca', 410, 342, 320, 92, ['加权 PCA 局部坐标系', 't1、t2、n']),
  node('geo', 52, 500, 250, 82, ['64D OBJ 几何特征']),
  node('feat', 430, 472, 280, 82, ['51D 标签节点特征']),
  node('fnn', 420, 592, 300, 92, ['残差 FNN', '51→64→128→64']),
  node('fusion', 230, 728, 300, 92, ['几何与标签融合', '128→64']),
  node('gnn', 210, 860, 340, 106, ['异构关系 GNN ×2', 'Anchor→Label 18D', 'Label→Label 10D']),
  node('transformer', 230, 1010, 300, 82, ['全局 Transformer ×1']),
  node('router', 230, 1132, 300, 82, ['Softmax Human-style MoE 路由器']),
  node('experts', 230, 1254, 300, 94, ['四个风格专家', '每个 64→32→6']),
  node('local', 215, 1392, 330, 94, ['局部 u、v、法向距离', 'log size x/y/z']),
  node('safety', 180, 1530, 400, 86, ['确定性五视角优化与安全门控'])
];
const baseV10Edges = [
  edge('obj', 'sample'), edge('sample', 'dgcnn'), edge('dgcnn', 'geo'),
  edge('contract', 'patch'), edge('patch', 'pca'), edge('pca', 'feat'), edge('feat', 'fnn'),
  edge('geo', 'fusion', { middleY: 680 }), edge('fnn', 'fusion', { middleY: 680 }),
  edge('fusion', 'gnn'), edge('gnn', 'transformer'), edge('transformer', 'router'), edge('router', 'experts'), edge('experts', 'local'), edge('local', 'safety')
];

const diagrams = [
  {
    slug: '01-original-v10', title: '原始 v10 架构', subtitle: '活动基础布局模型 · 纯三维生成', height: 1760,
    nodes: [...baseV10Nodes, node('final', 270, 1660, 220, 76, ['最终布局'], 'green')],
    edges: [...baseV10Edges, edge('safety', 'final')]
  },
  {
    slug: '02-v10-base-reward-mlp', title: 'v10 + 基础奖励 MLP 架构', subtitle: '安全候选内的美学重排组合', height: 2280,
    nodes: [...baseV10Nodes,
      node('candidates', 250, 1656, 260, 80, ['多个安全候选']),
      node('stats', 245, 1776, 270, 82, ['12D 布局统计特征']),
      node('mlp', 230, 1898, 300, 94, ['基础奖励 MLP', '12→32 tanh→1'], 'violet'),
      node('rerank', 220, 2034, 320, 82, ['安全候选美学重排'], 'violet'),
      node('final', 270, 2156, 220, 76, ['最终布局'], 'green')],
    edges: [...baseV10Edges, edge('safety', 'candidates'), edge('candidates', 'stats'), edge('stats', 'mlp'), edge('mlp', 'rerank'), edge('rerank', 'final')]
  },
  {
    slug: '03-v10-mdpo', title: 'v10 + MDPO 架构', subtitle: '七维偏好直接更新策略 · 奖励重排关闭', height: 2260,
    nodes: [
      node('pairs', 30, 126, 300, 92, ['train33 安全候选对', '六图可视化'], 'violet'),
      node('reference', 430, 126, 300, 92, ['冻结原始 v10', '参考策略 πref'], 'gray'),
      node('qwen', 35, 258, 290, 92, ['Qwen 七维视觉偏好', '仅离线评分'], 'violet'),
      node('chosen', 40, 390, 280, 82, ['chosen / rejected 偏好']),
      node('loss', 210, 520, 340, 106, ['七维 MDPO 损失', 'reference KL + 分维偏好'], 'violet'),
      node('constraints', 190, 668, 380, 106, ['v10 监督 + Center-tail CVaR', '确定性几何安全损失'], 'amber'),
      node('lora', 170, 816, 420, 106, ['LoRA + bias 更新', 'GNN2 · Transformer · MoE · Decoder'], 'green'),
      node('input', 190, 974, 380, 86, ['Clean OBJ + 固定标签契约']),
      node('encode', 185, 1102, 390, 94, ['DGCNN 64D + 51D 标签编码', '锚点局部坐标系']),
      node('fusion', 220, 1238, 320, 88, ['128D 融合 → 64D']),
      node('reason', 190, 1368, 380, 100, ['更新后的 GNN2 + Transformer', '全标签关系推理'], 'green'),
      node('moe', 190, 1510, 380, 100, ['更新后的 MoE 四专家', '64→32→6 解码'], 'green'),
      node('policy', 190, 1652, 380, 100, ['六维连续高斯策略', '局部中心与 log size']),
      node('safety', 180, 1794, 400, 92, ['确定性五视角优化与安全门控']),
      node('candidates', 245, 1928, 270, 82, ['安全候选', '奖励重排关闭']),
      node('final', 270, 2052, 220, 76, ['MDPO 候选布局'], 'amber'),
      node('status', 205, 2168, 350, 62, ['当前：被拒候选 / 未部署'], 'gray')],
    edges: [edge('pairs', 'qwen'), edge('qwen', 'chosen'), edge('chosen', 'loss'), edge('reference', 'loss', { middleY: 486 }), edge('loss', 'constraints'), edge('constraints', 'lora'), edge('lora', 'input'), edge('input', 'encode'), edge('encode', 'fusion'), edge('fusion', 'reason'), edge('reason', 'moe'), edge('moe', 'policy'), edge('policy', 'safety'), edge('safety', 'candidates'), edge('candidates', 'final'), edge('final', 'status')]
  },
  {
    slug: '04-v10-mdpo-base-reward-mlp', title: 'v10 + MDPO + 基础奖励 MLP', subtitle: 'MDPO 策略候选 + 历史奖励重排', height: 2540,
    nodes: [
      node('pairs', 30, 126, 300, 92, ['train33 安全候选对', '六图可视化'], 'violet'),
      node('reference', 430, 126, 300, 92, ['冻结原始 v10', '参考策略 πref'], 'gray'),
      node('qwen', 35, 258, 290, 92, ['Qwen 七维视觉偏好', '仅离线评分'], 'violet'),
      node('loss', 210, 402, 340, 106, ['MDPO + reference KL', '监督 + CVaR + 安全'], 'violet'),
      node('lora', 170, 550, 420, 106, ['LoRA + bias 更新', 'GNN2 · Transformer · MoE · Decoder'], 'green'),
      node('input', 190, 708, 380, 86, ['Clean OBJ + 固定标签契约']),
      node('encode', 185, 836, 390, 94, ['DGCNN 64D + 51D 标签编码', '锚点局部坐标系']),
      node('fusion', 220, 972, 320, 88, ['128D 融合 → 64D']),
      node('reason', 190, 1102, 380, 100, ['更新后的 GNN2 + Transformer', '全标签关系推理'], 'green'),
      node('moe', 190, 1244, 380, 100, ['更新后的 MoE 四专家', '64→32→6 解码'], 'green'),
      node('policy', 190, 1386, 380, 94, ['六维连续策略输出']),
      node('safety', 180, 1522, 400, 92, ['确定性五视角优化与安全门控']),
      node('candidates', 245, 1656, 270, 82, ['多个 MDPO 安全候选']),
      node('stats', 245, 1778, 270, 82, ['12D 布局统计特征']),
      node('mlp', 230, 1900, 300, 94, ['基础奖励 MLP', '12→32 tanh→1'], 'violet'),
      node('rerank', 220, 2036, 320, 82, ['安全候选美学重排'], 'violet'),
      node('final', 270, 2158, 220, 76, ['候选对照布局'], 'amber'),
      node('status', 205, 2274, 350, 62, ['当前：候选对照 / 未部署'], 'gray')],
    edges: [edge('pairs', 'qwen'), edge('qwen', 'loss'), edge('reference', 'loss', { middleY: 370 }), edge('loss', 'lora'), edge('lora', 'input'), edge('input', 'encode'), edge('encode', 'fusion'), edge('fusion', 'reason'), edge('reason', 'moe'), edge('moe', 'policy'), edge('policy', 'safety'), edge('safety', 'candidates'), edge('candidates', 'stats'), edge('stats', 'mlp'), edge('mlp', 'rerank'), edge('rerank', 'final'), edge('final', 'status')]
  },
  {
    slug: '05-binoforce', title: 'BinoForce 方法架构', subtitle: '双目视点下的动态五力布局复现', height: 1690,
    nodes: [
      node('scene', 50, 126, 280, 86, ['3D 场景 + OBJ 几何']),
      node('annotations', 430, 126, 280, 86, ['标签、锚点与尺寸']),
      node('camera', 210, 254, 340, 94, ['左右眼虚拟相机', 'IPD = 0.064 m'], 'green'),
      node('trajectory', 210, 390, 340, 94, ['确定性动态相机轨迹', '30 FPS 分段移动/暂停']),
      node('init', 230, 526, 300, 86, ['标签从锚点初始化']),
      node('forces', 130, 654, 500, 132, ['五类布局力', '排斥 · 吸引 · 双目重叠', '引导线交叉 · 圆形力'], 'blue'),
      node('weights', 155, 828, 450, 106, ['加权合力', '0.02 / 0.09 / 0.06 / 0.8 / 0.03']),
      node('move', 220, 976, 320, 92, ['逐帧位移更新', 'labels += force × 0.01']),
      node('warmup', 210, 1110, 340, 94, ['Warm-up 800 帧', '进入稳定布局']),
      node('face', 205, 1246, 350, 94, ['Camera-facing 标签投影', '双目 max 重叠约束']),
      node('snapshot', 220, 1382, 320, 86, ['保存最终动态快照']),
      node('final', 230, 1510, 300, 94, ['五视角统一评价', 'BinoForce 最终布局'], 'green')],
    edges: [edge('scene', 'camera', { middleY: 232 }), edge('annotations', 'camera', { middleY: 232 }), edge('camera', 'trajectory'), edge('trajectory', 'init'), edge('init', 'forces'), edge('forces', 'weights'), edge('weights', 'move'), edge('move', 'warmup'), edge('warmup', 'face'), edge('face', 'snapshot'), edge('snapshot', 'final')]
  },
  {
    slug: '06-hedgehog-3d', title: 'Hedgehog 3D 方法架构', subtitle: '径向 Pole + 局部图像平面三自由度优化', height: 1530,
    nodes: [
      node('obj', 45, 126, 290, 86, ['OBJ / 包围球几何']),
      node('annotations', 425, 126, 290, 86, ['人工标签锚点与尺寸']),
      node('center', 205, 254, 350, 92, ['物体包围球中心 + Anchor']),
      node('pole', 190, 388, 380, 94, ['构造径向 Hedgehog Pole', 'center → anchor']),
      node('init', 210, 524, 340, 92, ['最小 Pole 长度初始化', '0.08 × 物体对角线']),
      node('project', 205, 658, 350, 92, ['投影到当前视角', '标签矩形与引导线']),
      node('forces', 155, 792, 450, 118, ['屏幕空间布局力', '标签排斥 · 物体避让 · 锚点遮挡', '引导线交叉 · 回拉力']),
      node('update', 160, 952, 440, 110, ['3DOF 更新', 'Pole 长度 + 局部图像平面 X/Y', 'X/Y 位移限制在标签尺寸内'], 'green'),
      node('iterate', 220, 1104, 320, 86, ['确定性迭代 80 次']),
      node('world', 200, 1232, 360, 94, ['反投影回 3D 标签中心']),
      node('final', 230, 1368, 300, 92, ['五视角评价', 'Hedgehog 3D 布局'], 'green')],
    edges: [edge('obj', 'center', { middleY: 232 }), edge('annotations', 'center', { middleY: 232 }), edge('center', 'pole'), edge('pole', 'init'), edge('init', 'project'), edge('project', 'forces'), edge('forces', 'update'), edge('update', 'iterate'), edge('iterate', 'world'), edge('world', 'final')]
  },
  {
    slug: '07-hedgehog-1d', title: 'Hedgehog 1D 方法架构', subtitle: '标签仅沿径向 Pole 滑动', height: 1490,
    nodes: [
      node('obj', 45, 126, 290, 86, ['OBJ / 包围球几何']),
      node('annotations', 425, 126, 290, 86, ['人工标签锚点与尺寸']),
      node('center', 205, 254, 350, 92, ['物体包围球中心 + Anchor']),
      node('pole', 190, 388, 380, 94, ['构造径向 Hedgehog Pole', 'center → anchor']),
      node('init', 210, 524, 340, 92, ['最小 Pole 长度初始化', '0.08 × 物体对角线']),
      node('project', 205, 658, 350, 92, ['投影到当前视角', '标签矩形与引导线']),
      node('forces', 155, 792, 450, 118, ['屏幕空间布局力', '标签排斥 · 物体避让 · 锚点遮挡', '引导线交叉 · 回拉力']),
      node('update', 175, 952, 410, 100, ['1DOF 更新', '只改变 Pole 长度'], 'amber'),
      node('iterate', 220, 1094, 320, 86, ['确定性迭代 80 次']),
      node('world', 200, 1222, 360, 94, ['沿 Pole 得到 3D 标签中心']),
      node('final', 230, 1358, 300, 92, ['五视角评价', 'Hedgehog 1D 布局'], 'green')],
    edges: [edge('obj', 'center', { middleY: 232 }), edge('annotations', 'center', { middleY: 232 }), edge('center', 'pole'), edge('pole', 'init'), edge('init', 'project'), edge('project', 'forces'), edge('forces', 'update'), edge('update', 'iterate'), edge('iterate', 'world'), edge('world', 'final')]
  },
  {
    slug: '08-manual-optimized-layout', title: '人工优化布局流程', subtitle: '数据集参考布局 · 非学习模型', height: 1420,
    nodes: [
      node('obj', 45, 126, 290, 86, ['Clean OBJ 与零件结构']),
      node('contract', 425, 126, 290, 86, ['固定标签文字、ID、锚点']),
      node('views', 200, 260, 360, 94, ['主 / 左 / 右 / 俯 / 仰', '五视角人工观察']),
      node('place', 190, 396, 380, 100, ['人工拖拽标签中心', '调整 3D 位置与尺寸'], 'green'),
      node('leader', 200, 538, 360, 94, ['连接 Anchor 与标签', '检查引导线方向和长度']),
      node('quality', 150, 674, 460, 124, ['逐视角质量检查', '文字可读 · 标签重叠 · 物体遮挡', '交叉 · 越界 · 空间分布'], 'amber'),
      node('revise', 205, 840, 350, 92, ['不合格则继续人工调整']),
      node('save', 190, 974, 380, 94, ['保存 after_manual JSON', '中心、box size 与标签契约']),
      node('reference', 160, 1110, 440, 106, ['作为监督目标与 PCK 参考', 'PCK=1 为参考自一致'], 'violet'),
      node('final', 230, 1258, 300, 92, ['人工优化最终布局'], 'green')],
    edges: [edge('obj', 'views', { middleY: 238 }), edge('contract', 'views', { middleY: 238 }), edge('views', 'place'), edge('place', 'leader'), edge('leader', 'quality'), edge('quality', 'revise'), edge('revise', 'save'), edge('save', 'reference'), edge('reference', 'final')]
  }
];

const edgeCandidates = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
];
const edgeExecutable = edgeCandidates.find((candidate) => fs.existsSync(candidate));
const manifest = [];

for (const diagram of diagrams) {
  const svgName = diagram.slug + '.svg';
  const pngName = diagram.slug + '.png';
  const svgPath = path.join(outputDir, svgName);
  const pngPath = path.join(outputDir, pngName);
  fs.writeFileSync(svgPath, renderDiagram(diagram), 'utf8');
  let pngGenerated = false;
  if (edgeExecutable) {
    const profile = fs.mkdtempSync(path.join(process.env.TEMP || outputDir, 'architecture-edge-'));
    const result = spawnSync(edgeExecutable, [
      '--headless', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
      '--force-device-scale-factor=2', '--window-size=760,' + diagram.height,
      '--user-data-dir=' + profile, '--screenshot=' + pngPath, pathToFileURL(svgPath).href
    ], { encoding: 'utf8', timeout: 30000 });
    pngGenerated = result.status === 0 && fs.existsSync(pngPath);
  }
  manifest.push({ slug: diagram.slug, title: diagram.title, subtitle: diagram.subtitle, width: 760, height: diagram.height, svg: svgName, png: pngGenerated ? pngName : null });
}

const cards = manifest.map((item) => '<article><a href="' + item.svg + '" target="_blank"><img src="' + item.svg + '" alt="' + item.title + '"/></a><h2>' + item.title + '</h2><p>' + item.subtitle + '</p></article>').join('');
const gallery = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>8种模型/方法架构图</title><style>body{margin:0;padding:28px;background:#f4f7fb;color:#294b68;font-family:Microsoft YaHei,Noto Sans SC,Arial,sans-serif}header{max-width:1500px;margin:auto auto 22px}h1{margin:0 0 8px;font-size:28px}header p{margin:0;color:#718196}.grid{max-width:1500px;margin:auto;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{padding:14px;background:#fff;border:1px solid #dce7ee;border-radius:14px;box-shadow:0 8px 24px rgba(39,61,98,.06)}article a{display:block;height:720px;overflow:auto;border-radius:10px;background:#fff}img{display:block;width:100%;height:auto}h2{margin:12px 4px 5px;font-size:18px}article p{margin:0 4px 5px;color:#718196;font-size:13px}@media(max-width:900px){.grid{grid-template-columns:1fr}article a{height:620px}}</style></head><body><header><h1>8种模型/方法架构图</h1><p>统一依据当前项目实现与复现实验生成；点击图片可打开完整 SVG。</p></header><main class="grid">' + cards + '</main></body></html>';
fs.writeFileSync(path.join(outputDir, 'index.html'), gallery, 'utf8');
fs.writeFileSync(path.join(outputDir, 'manifest.json'), JSON.stringify({ generated_at: new Date().toISOString(), diagrams: manifest }, null, 2) + '\n', 'utf8');

console.log(JSON.stringify({ output_dir: outputDir, diagrams: manifest.length, png_count: manifest.filter((item) => item.png).length }, null, 2));
