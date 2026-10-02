import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultExpertFile = path.join(root, 'experiments', 'moe_expert_layouts_train33.json');
const defaultLlmFile = path.join(root, 'experiments', 'moe_style_llm_scores.json');
const defaultOutput = path.join(root, 'experiments', 'moe_expert_selection_train33.json');
const STYLES = ['spherical', 'rectangular', 'surround'];

function parseArgs(argv) {
  const options = { experts: defaultExpertFile, llm: defaultLlmFile, output: defaultOutput, temperature: 0.18 };
  for (let i = 0; i < argv.length; i += 1) {
    if (!argv[i].startsWith('--')) continue;
    const key = argv[i].slice(2);
    options[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true;
  }
  options.temperature = Math.max(0.03, Number(options.temperature || 0.18));
  return options;
}
async function readJson(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function sampleKey(row) { return `${row.category}/${row.sample_id}`; }
function softmax(values, temperature) {
  const peak = Math.max(...values);
  const exponentials = values.map((value) => Math.exp((value - peak) / temperature));
  const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  return exponentials.map((value) => value / total);
}
function safetyScore(metrics) {
  const penalty = 0.42 * Math.min(1, finite(metrics.olr) / 0.1)
    + 0.24 * Math.min(1, finite(metrics.lcd) / 0.1) + 0.28 * Math.min(1, finite(metrics.visible_leader_overlap_ratio) / 0.1)
    + 0.18 * Math.min(1, finite(metrics.viewport_overflow_ratio) / 0.05)
    + 0.16 * Math.min(1, finite(metrics.label_object_occlusion_ratio) / 0.05)
    + 0.14 * Math.min(1, finite(metrics.object_penetration_ratio) / 0.05);
  return Math.max(0, 1 - penalty);
}
function geometryScore(metrics) {
  const quality = finite(metrics.multidimensional_quality_score) / 5;
  const clarity = Math.min(1, Math.max(0, finite(metrics.text_clarity) / 5));
  return Math.max(0, Math.min(1, 0.78 * quality + 0.22 * clarity)) * safetyScore(metrics);
}
function normalizeLlm(row) {
  const scores = row?.scores || {};
  return Object.fromEntries(STYLES.map((style) => [style, Math.max(0, Math.min(1, finite(scores[style], 3) / 5))]));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const [expertBundle, llmBundle] = await Promise.all([
    readJson(path.resolve(String(options.experts))),
    readJson(path.resolve(String(options.llm)))
  ]);
  const llmMap = new Map((llmBundle.rows || []).map((row) => [sampleKey(row), normalizeLlm(row)]));
  const rows = [];
  for (const row of expertBundle.rows) {
    const llmPrior = llmMap.get(sampleKey(row)) || Object.fromEntries(STYLES.map((style) => [style, 1 / 3]));
    const perExpert = {};
    const rawScores = [];
    for (const style of STYLES) {
      const expert = row.experts[style];
      const geometry = geometryScore(expert.metrics);
      const llm = llmPrior[style];
      const combined = geometry;
      perExpert[style] = {
        style,
        geometry_score: Number(geometry.toFixed(6)),
        llm_style_score: Number(llm.toFixed(6)),
        combined_score: Number(combined.toFixed(6)),
        metrics: expert.metrics,
        labels: expert.labels
      };
      rawScores.push(combined);
    }
    const probabilities = softmax(rawScores, options.temperature);
    const routerTarget = Object.fromEntries(STYLES.map((style, index) => [style, Number(probabilities[index].toFixed(6))]));
    const selected = STYLES.reduce((best, style) => perExpert[style].combined_score > perExpert[best].combined_score ? style : best, STYLES[0]);
    rows.push({ category: row.category, sample_id: row.sample_id, split: row.split, selected_expert: selected, router_target: routerTarget, llm_style_prior: llmPrior, experts: perExpert });
  }
  const selectedCounts = Object.fromEntries(STYLES.map((style) => [style, rows.filter((row) => row.selected_expert === style).length]));
  const averageRouterTarget = Object.fromEntries(STYLES.map((style) => {
    const average = rows.reduce((sum, row) => sum + row.router_target[style], 0) / Math.max(1, rows.length);
    return [style, Number(average.toFixed(6))];
  }));
  const result = {
    version: 'moe_expert_selection_v1',
    generated_at: new Date().toISOString(),
    sources: {
      expert_layouts: path.relative(root, path.resolve(String(options.experts))).split(path.sep).join('/'),
      llm_style_scores: path.relative(root, path.resolve(String(options.llm))).split(path.sep).join('/')
    },
    policy: {
      geometry_weight: 1,
      llm_style_weight: 0,
      router_target: 'temperature_softmax_of_five_view_geometry_quality_score',
      ambiguous_samples: 'retain_soft_router_probabilities_for_gradient_update'
    },
    count: rows.length,
    summary: { selected_counts: selectedCounts, average_router_target: averageRouterTarget },
    rows
  };
  await fs.writeFile(path.resolve(String(options.output)), JSON.stringify(result, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ok: true, output: path.relative(root, path.resolve(String(options.output))).split(path.sep).join('/'), count: rows.length, summary: result.summary }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exit(1); });
