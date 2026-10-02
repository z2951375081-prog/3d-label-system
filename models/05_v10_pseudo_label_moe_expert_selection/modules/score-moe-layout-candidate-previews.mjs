import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeOllamaEndpoint, DEFAULT_OLLAMA_MODEL } from '../lib/ollama-vision-adapter.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultManifest = path.join(root, 'experiments', 'moe_layout_llm_candidate_previews', 'manifest.json');
const defaultOutput = path.join(root, 'experiments', 'moe_layout_llm_candidate_scores.json');
const VIEW_ORDER = ['main', 'right', 'left', 'up', 'down'];
const SCORE_KEYS = ['overall', 'style_fit', 'uniformity', 'leader_clarity', 'label_spacing', 'object_preservation', 'viewport_safety', 'text_readability'];

function parseArgs(argv) {
  const options = { manifest: defaultManifest, output: defaultOutput, limit: 0, concurrency: 2, ollamaUrl: 'http://127.0.0.1:11434', model: DEFAULT_OLLAMA_MODEL };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  options.limit = Number(options.limit || 0);
  return options;
}
function schema() {
  return { type: 'object', properties: { scores: { type: 'object', properties: Object.fromEntries(SCORE_KEYS.map((key) => [key, { type: 'number', minimum: 1, maximum: 5 }])), required: SCORE_KEYS }, selected_for_lora: { type: 'boolean' }, rationale: { type: 'string' }, risks: { type: 'array', items: { type: 'string' } }, suggested_changes: { type: 'array', items: { type: 'string' } } }, required: ['scores', 'selected_for_lora', 'rationale', 'risks', 'suggested_changes'] };
}
function prompt(item) {
  return `请评价这个 3D 外部标签布局候选。五张图分别是主视角、右、左、上、下视角，图中白色框是标签，彩色线是引导线。候选扰动模式：${item.mode}；目标风格：${item.style || 'unknown'}。请给 1-5 分，5 表示最好：overall 总体偏好，style_fit 是否符合目标布局风格，uniformity 标签分布是否均匀，leader_clarity 引导线是否自然且少交叉，label_spacing 标签之间距离是否合适，object_preservation 是否不遮挡物体，viewport_safety 是否不越界，text_readability 是否清晰。只输出 JSON。`;
}
async function imageBase64(relativePath) { return (await fs.readFile(path.join(root, relativePath.replaceAll('/', path.sep)))).toString('base64'); }
function parseContent(data) {
  const raw = String(data?.message?.content || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const parsed = JSON.parse(raw);
  const invalid = SCORE_KEYS.filter((key) => !Number.isFinite(Number(parsed?.scores?.[key])) || Number(parsed.scores[key]) < 1 || Number(parsed.scores[key]) > 5);
  if (invalid.length) throw new Error('Invalid candidate score keys: ' + invalid.join(', '));
  parsed.scores = Object.fromEntries(SCORE_KEYS.map((key) => [key, Number(Number(parsed.scores[key]).toFixed(4))]));
  return parsed;
}
function composite(scores) {
  const weights = { overall: 0.22, style_fit: 0.16, uniformity: 0.14, leader_clarity: 0.13, label_spacing: 0.10, object_preservation: 0.12, viewport_safety: 0.08, text_readability: 0.05 };
  return Object.entries(weights).reduce((sum, [key, weight]) => sum + weight * Number(scores[key]), 0);
}
function clamp01(value) { return Math.max(0, Math.min(1, Number(value) || 0)); }
function finite(value, fallback = 0) { const number = Number(value); return Number.isFinite(number) ? number : fallback; }
function deterministicSelectionScore(row) {
  const metrics = row.metrics || {};
  const labels = Math.max(1, finite(metrics.label_count, 1));
  const quality = finite(metrics.multidimensional_quality_score, finite(metrics.intrinsic_quality_score, 0));
  const penalties =
    0.45 * clamp01(finite(metrics.lcd, 0) / 0.08) +
    0.35 * clamp01(finite(metrics.viewport_overflow_ratio, 0) / 0.05) +
    0.30 * clamp01(finite(metrics.object_penetration_ratio, 0) / 0.05) +
    0.22 * clamp01(finite(metrics.label_object_occlusion_ratio, 0) / 0.05) +
    0.18 * clamp01(finite(metrics.overlap_pairs, 0) / labels);
  return Number((0.55 * finite(row.composite_score, 0) + 0.45 * quality - penalties).toFixed(6));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(await fs.readFile(path.resolve(String(options.manifest)), 'utf8'));
  const endpoint = normalizeOllamaEndpoint(options.ollamaUrl);
  const items = options.limit > 0 ? manifest.previews.slice(0, options.limit) : manifest.previews;
  const outputFile = path.resolve(String(options.output));
  const partialFile = outputFile + '.partial.json';
  let outputRows = Array(items.length);
  if (options.resume) {
    try {
      const partial = JSON.parse(await fs.readFile(partialFile, 'utf8'));
      if (partial.source_manifest === path.relative(root, path.resolve(String(options.manifest))).split(path.sep).join('/') && Array.isArray(partial.rows)) {
        outputRows = partial.rows.slice(0, items.length);
      }
    } catch {}
  }
  let nextIndex = 0;
  let completed = outputRows.filter(Boolean).length;
  let flushQueue = Promise.resolve();
  const flushPartial = () => {
    const payload = { version: 'moe_layout_llm_candidate_scores_partial_v1', generated_at: new Date().toISOString(), source_manifest: path.relative(root, path.resolve(String(options.manifest))).split(path.sep).join('/'), model: options.model, count: outputRows.filter(Boolean).length, rows: outputRows };
    flushQueue = flushQueue.then(() => fs.writeFile(partialFile, JSON.stringify(payload) + '\n', 'utf8')).catch(() => {});
    return flushQueue;
  };
  async function worker() {
    while (true) {
      const index = nextIndex++;
      if (index >= items.length) return;
      if (outputRows[index]) continue;
      const item = items[index];
      const images = await Promise.all(VIEW_ORDER.map((view) => imageBase64(item.views[view])));
      let lastError = null;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const response = await fetch(endpoint.apiUrl, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: options.model, stream: false, format: schema(), options: { temperature: 0, seed: 17, num_ctx: 16384 }, messages: [{ role: 'system', content: '你是 3D 标签布局多视角偏好评分器。必须严格输出 JSON。' }, { role: 'user', content: prompt(item) + '\n图像顺序：' + VIEW_ORDER.join(','), images }] }) });
          if (!response.ok) throw new Error(`Ollama request failed: ${response.status}`);
          const data = await response.json();
          const parsed = parseContent(data);
          const row = { ...item, llm_scores: parsed.scores, composite_score: Number(composite(parsed.scores).toFixed(6)), selected_for_lora: Boolean(parsed.selected_for_lora), rationale: parsed.rationale, risks: parsed.risks, suggested_changes: parsed.suggested_changes, model: options.model, response_id: data.created_at || null };
          row.selection_score = deterministicSelectionScore(row);
          outputRows[index] = row;
          completed += 1;
          if (completed % 4 === 0) await flushPartial();
          lastError = null;
          break;
        } catch (error) {
          lastError = error;
          if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
        }
      }
      if (lastError) {
        if (!options.skipFailures) throw new Error(`candidate ${item.candidate_id} failed after retries: ${lastError.message}`);
        outputRows[index] = { ...item, scoring_error: lastError.message, model: options.model };
        completed += 1;
        await flushPartial();
      }
    }
  }
  await Promise.all(Array.from({ length: options.concurrency }, () => worker()));
  await flushQueue;
  const grouped = new Map();
  for (const row of outputRows) {
    const key = `${row.category}/${row.sample_id}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  const winners = [...grouped.entries()].map(([, rows]) => rows.reduce((best, row) => {
    if (row.selection_score !== best.selection_score) return row.selection_score > best.selection_score ? row : best;
    if (finite(row.metrics?.multidimensional_quality_score) !== finite(best.metrics?.multidimensional_quality_score)) return finite(row.metrics?.multidimensional_quality_score) > finite(best.metrics?.multidimensional_quality_score) ? row : best;
    return row.composite_score > best.composite_score ? row : best;
  }, rows[0]));
  const result = { version: 'moe_layout_llm_candidate_scores_v2_clean_preview_safety_selection', generated_at: new Date().toISOString(), source_manifest: path.relative(root, path.resolve(String(options.manifest))).split(path.sep).join('/'), model: options.model, score_keys: SCORE_KEYS, selection_policy: { id: 'llm_preference_plus_deterministic_geometry_safety_v1', formula: '0.55*llm_composite + 0.45*multidimensional_quality - penalties(lcd, overflow, penetration, object_occlusion, label_overlap)', reason: 'Do not select a visually worse perturbation only because a small LLM-score difference is highest.' }, count: outputRows.filter((row) => row?.llm_scores).length, error_count: outputRows.filter((row) => row?.scoring_error).length, winners: winners.map((row) => ({ category: row.category, sample_id: row.sample_id, candidate_id: row.candidate_id, mode: row.mode, composite_score: row.composite_score, deterministic_quality_score: finite(row.metrics?.multidimensional_quality_score, null), selection_score: row.selection_score })), rows: outputRows };
  await fs.writeFile(outputFile, JSON.stringify(result, null, 2) + '\n', 'utf8');
  await fs.rm(partialFile, { force: true }).catch(() => {});
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/'), count: outputRows.filter((row) => row?.llm_scores).length, error_count: outputRows.filter((row) => row?.scoring_error).length, winners: result.winners }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
