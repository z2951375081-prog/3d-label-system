import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeOllamaEndpoint, DEFAULT_OLLAMA_MODEL } from '../lib/ollama-vision-adapter.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const manifestFile = path.join(experiments, 'dataset_manifest.json');
const auditFile = path.join(experiments, 'moe_style_routing_audit.json');
const defaultTemplate = path.join(experiments, 'moe_style_human_score_template.csv');
const defaultPromptManifest = path.join(experiments, 'moe_style_llm_prompt_manifest.json');
const defaultLlmScores = path.join(experiments, 'moe_style_llm_scores.json');
const defaultValidation = path.join(experiments, 'moe_style_llm_validation_report.json');
const STYLE_KEYS = ['spherical', 'rectangular', 'surround'];

function parseArgs(argv) {
  const options = { mode: 'prepare', split: 'all', limit: 0, humanScores: defaultTemplate, llmScores: defaultLlmScores, output: defaultValidation, promptManifest: defaultPromptManifest, template: defaultTemplate, callOllama: false, ollamaUrl: 'http://127.0.0.1:11434', model: DEFAULT_OLLAMA_MODEL };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  options.limit = Number(options.limit || 0);
  options.callOllama = options.callOllama === true || String(options.callOllama).toLowerCase() === 'true';
  return options;
}
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
function repoPath(relativePath) { return path.join(root, String(relativePath).replaceAll('/', path.sep)); }
function csvCell(value) { return `"${String(value ?? '').replaceAll('"', '""')}"`; }
function parseCsv(text) {
  const rows = [];
  let cell = '', row = [], quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n') { row.push(cell); if (row.some((value) => value !== '')) rows.push(row); row = []; cell = ''; }
    else if (c !== '\r') cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [header, ...body] = rows;
  return body.map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ''])));
}
function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function variance(values) { const avg = mean(values); return mean(values.map((value) => (value - avg) ** 2)); }
function pearson(x, y) {
  if (x.length !== y.length || x.length < 3) return null;
  const mx = mean(x), my = mean(y);
  const cov = mean(x.map((value, index) => (value - mx) * (y[index] - my)));
  const denom = Math.sqrt(variance(x) * variance(y));
  return denom > 1e-12 ? cov / denom : null;
}
function ranks(values) {
  const sorted = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const out = Array(values.length).fill(0);
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].value === sorted[i].value) j += 1;
    const rank = (i + j + 1) / 2;
    for (let k = i; k < j; k += 1) out[sorted[k].index] = rank;
    i = j;
  }
  return out;
}
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}
function normalCdf(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function correlationPValue(r, n) {
  if (!Number.isFinite(r) || n < 4 || Math.abs(r) >= 1) return null;
  const z = 0.5 * Math.log((1 + r) / (1 - r)) * Math.sqrt(n - 3);
  return 2 * (1 - normalCdf(Math.abs(z)));
}
function pairedT(reference, observed) {
  const diffs = reference.map((value, index) => observed[index] - value);
  const avg = mean(diffs);
  const sd = Math.sqrt(variance(diffs) * diffs.length / Math.max(1, diffs.length - 1));
  const t = sd > 1e-12 ? avg / (sd / Math.sqrt(diffs.length)) : 0;
  const p = 2 * (1 - normalCdf(Math.abs(t))); // normal approximation, adequate for screening report
  return { mean_difference: Number(avg.toFixed(6)), t: Number(t.toFixed(6)), p_value_normal_approx: Number(p.toFixed(6)) };
}
function dominant(row, prefix = '') { return STYLE_KEYS.reduce((best, key) => Number(row[`${prefix}${key}`]) > Number(row[`${prefix}${best}`]) ? key : best, STYLE_KEYS[0]); }
function dataUrlToOllamaImage(file) { return fs.readFile(file).then((buffer) => buffer.toString('base64')); }
function stylePrompt(row) {
  return `请只依据五张视角图判断这个 3D 物体最适合哪种外部标签布局风格，并给 1-5 分：\n` +
    `1 spherical：标签绕紧凑/近圆轮廓形成环状或圆弧分布。\n` +
    `2 rectangular：长条或矩形物体，标签沿包围物体的矩形框四边分布。\n` +
    `3 surround：不规则物体，标签按锚点方向自由围绕，不要求对齐。\n` +
    `返回 JSON：{"scores":{"spherical":1-5,"rectangular":1-5,"surround":1-5},"selected":"spherical|rectangular|surround","confidence":1-5,"rationale":"简短中文原因"}。样本 ${row.category}/${row.sample_id}。`;
}
function styleSchema() {
  return { type: 'object', properties: { scores: { type: 'object', properties: Object.fromEntries(STYLE_KEYS.map((key) => [key, { type: 'number', minimum: 1, maximum: 5 }])), required: STYLE_KEYS }, selected: { type: 'string', enum: STYLE_KEYS }, confidence: { type: 'number', minimum: 1, maximum: 5 }, rationale: { type: 'string' } }, required: ['scores', 'selected', 'confidence', 'rationale'] };
}
async function collectRows(options) {
  const [manifest, audit] = await Promise.all([readJson(manifestFile), readJson(auditFile)]);
  const byKey = new Map(audit.rows.map((row) => [`${row.category}/${row.sample_id}`, row]));
  let samples = manifest.samples.filter((sample) => options.split === 'all' || sample.split === options.split);
  if (options.limit > 0) samples = samples.slice(0, options.limit);
  return samples.map((sample) => {
    const auditRow = byKey.get(`${sample.category}/${sample.sample_id}`);
    const viewFiles = Object.fromEntries(['main', 'right', 'left', 'up', 'down'].map((view) => [view, sample.views.files.find((file) => file.toLowerCase().includes(`-${view}.png`)) || null]));
    return { ...sample, audit: auditRow, viewFiles };
  });
}
async function prepare(options) {
  const rows = await collectRows(options);
  const header = ['category', 'sample_id', 'split', 'selected_by_geometry', 'geometry_spherical_1_5', 'geometry_rectangular_1_5', 'geometry_surround_1_5', 'manual_spherical_1_5', 'manual_rectangular_1_5', 'manual_surround_1_5', 'manual_selected', 'notes'];
  const csv = [header.join(',')].concat(rows.map((row) => {
    const weights = row.audit?.weights || {};
    return [row.category, row.sample_id, row.split, row.audit?.selected || '', ...STYLE_KEYS.map((key) => Number((1 + 4 * Number(weights[key] || 0)).toFixed(4))), '', '', '', '', ''].map(csvCell).join(',');
  })).join('\n') + '\n';
  await fs.writeFile(path.resolve(String(options.template)), csv, 'utf8');
  const prompts = rows.map((row) => ({ category: row.category, sample_id: row.sample_id, split: row.split, prompt: stylePrompt(row), image_files: row.viewFiles, geometry_prior: row.audit?.weights || null }));
  await fs.writeFile(path.resolve(String(options.promptManifest)), JSON.stringify({ version: 'moe_style_llm_prompt_manifest_v1', generated_at: new Date().toISOString(), image_order: ['main', 'right', 'left', 'up', 'down'], schema: styleSchema(), prompts }, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, template: path.relative(root, path.resolve(String(options.template))).replaceAll('\\', '/'), prompt_manifest: path.relative(root, path.resolve(String(options.promptManifest))).replaceAll('\\', '/'), count: rows.length }, null, 2));
}
async function runLlm(options) {
  const rows = await collectRows(options);
  const endpoint = normalizeOllamaEndpoint(options.ollamaUrl);
  const schema = styleSchema();
  const scored = [];
  for (const row of rows) {
    const imageFiles = ['main', 'right', 'left', 'up', 'down'].map((view) => repoPath(row.viewFiles[view]));
    const images = await Promise.all(imageFiles.map(dataUrlToOllamaImage));
    const response = await fetch(endpoint.apiUrl, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: options.model, stream: false, format: schema, options: { temperature: 0, seed: 17, num_ctx: 16384 }, messages: [{ role: 'system', content: '你是 3D 外部标签布局风格评估器。必须输出 JSON，不要输出多余文本。' }, { role: 'user', content: stylePrompt(row) + '\n图像顺序：main,right,left,up,down', images }] }) });
    if (!response.ok) throw new Error(`Ollama request failed ${response.status}`);
    const data = await response.json();
    const content = String(data?.message?.content || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
    const parsed = JSON.parse(content);
    scored.push({ category: row.category, sample_id: row.sample_id, split: row.split, scores: Object.fromEntries(STYLE_KEYS.map((key) => [key, Number(parsed.scores[key])])), selected: parsed.selected, confidence: Number(parsed.confidence), rationale: parsed.rationale, model: options.model, response_id: data.created_at || null });
  }
  await fs.writeFile(path.resolve(String(options.llmScores)), JSON.stringify({ version: 'moe_style_llm_scores_v1', generated_at: new Date().toISOString(), model: options.model, count: scored.length, rows: scored }, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.llmScores))).replaceAll('\\', '/'), count: scored.length }, null, 2));
}
async function validate(options) {
  const humanRows = parseCsv(await fs.readFile(path.resolve(String(options.humanScores)), 'utf8')).filter((row) => STYLE_KEYS.every((key) => Number.isFinite(Number(row[`manual_${key}_1_5`]))));
  const llm = await readJson(path.resolve(String(options.llmScores)));
  const llmByKey = new Map((llm.rows || []).map((row) => [`${row.category}/${row.sample_id}`, row]));
  const paired = humanRows.map((row) => ({ human: row, llm: llmByKey.get(`${row.category}/${row.sample_id}`) })).filter((pair) => pair.llm);
  const dimensions = Object.fromEntries(STYLE_KEYS.map((key) => {
    const h = paired.map((pair) => Number(pair.human[`manual_${key}_1_5`]));
    const l = paired.map((pair) => Number(pair.llm.scores[key]));
    const pr = pearson(h, l);
    const sr = pearson(ranks(h), ranks(l));
    return [key, { n: h.length, pearson_r: pr === null ? null : Number(pr.toFixed(6)), pearson_p_approx: pr === null ? null : Number(correlationPValue(pr, h.length)?.toFixed(6)), spearman_rho: sr === null ? null : Number(sr.toFixed(6)), spearman_p_approx: sr === null ? null : Number(correlationPValue(sr, h.length)?.toFixed(6)), paired_t_bias: pairedT(h, l) }];
  }));
  const agreement = paired.filter((pair) => (pair.human.manual_selected || dominant(pair.human, 'manual_').replace(/$/, '_1_5')) === pair.llm.selected || dominant({ spherical: pair.human.manual_spherical_1_5, rectangular: pair.human.manual_rectangular_1_5, surround: pair.human.manual_surround_1_5 }) === pair.llm.selected).length / Math.max(1, paired.length);
  const report = { version: 'moe_style_llm_validation_report_v1', generated_at: new Date().toISOString(), n: paired.length, status: paired.length >= 10 ? 'ready_for_interpretation' : 'too_few_pairs', dimensions, dominant_style_agreement: Number(agreement.toFixed(6)), decision_rule: 'LLM可替代前置打标建议门槛：n>=10，三维度 Spearman rho 均值 >=0.55，dominant agreement >=0.60，且偏差检验无明显系统性偏置。', source_files: { human_scores: path.relative(root, path.resolve(String(options.humanScores))).replaceAll('\\', '/'), llm_scores: path.relative(root, path.resolve(String(options.llmScores))).replaceAll('\\', '/') } };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.output))).replaceAll('\\', '/'), n: paired.length, dominant_style_agreement: report.dominant_style_agreement, status: report.status }, null, 2));
}

const options = parseArgs(process.argv.slice(2));
if (options.mode === 'prepare') await prepare(options);
else if (options.mode === 'llm') await runLlm(options);
else if (options.mode === 'validate') await validate(options);
else throw new Error('mode must be prepare, llm, or validate');

