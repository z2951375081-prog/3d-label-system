const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const pathLink = (p) => p ? `<a href="/${p}" target="_blank"><code>${p}</code></a>` : '<code>未生成</code>';
const STYLES = ['spherical', 'rectangular', 'surround'];
let testSamples = [];
let latestSnapshot = null;
function card(label, value, note='') { return `<article class="card"><span>${label}</span><strong>${value}</strong>${note ? `<small>${note}</small>` : ''}</article>`; }
function renderFiles(title, files) { return `<div class="file-box"><h3>${title}</h3>${files.map(pathLink).join('')}</div>`; }
function averageTestMetrics() {
  const selected = testSamples.map((row) => row.experts?.[row.selected_expert]).filter(Boolean);
  const keys = ['geometry_score', 'llm_style_score', 'combined_score', 'multidimensional_quality_score', 'objective_score', 'olr', 'lcd', 'viewport_overflow_ratio', 'label_object_occlusion_ratio', 'visible_leader_overlap_ratio', 'object_penetration_ratio', 'text_clarity'];
  return Object.fromEntries(keys.map((key) => {
    const values = selected.map((expert) => key.includes('score') && key !== 'multidimensional_quality_score' ? Number(expert[key]) : Number(expert.metrics?.[key])).filter(Number.isFinite);
    return [key, values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null];
  }));
}
function renderTestAverage() {
  const average = averageTestMetrics();
  $('testAverage').innerHTML = [
    card('test11 样本数', testSamples.length, '全部测试样本'),
    card('平均几何质量', fmt(average.geometry_score), '五视角 OBJ 几何评价'),
    card('平均 LLM 风格分', fmt(average.llm_style_score), '候选风格参考'),
    card('平均综合分', fmt(average.combined_score), '当前专家选择分'),
    card('平均多维质量', fmt(average.multidimensional_quality_score), '1–5，越高越好'),
    card('平均 OLR', fmt(average.olr), '标签重叠，越低越好'),
    card('平均 LCD', fmt(average.lcd), '引导线交叉，越低越好'),
    card('平均线重叠', fmt(average.visible_leader_overlap_ratio), '肉眼可见线段重叠'),
    card('平均视野越界', fmt(average.viewport_overflow_ratio), '越低越好')
  ].join('');
}
function renderTestSample(index = 0) {
  const row = testSamples[index];
  if (!row) { $('testExpertPanel').innerHTML = '<p class="hint">暂无 test11 专家布局数据。</p>'; return; }
  const previewsById = new Map((row.previews || []).map((preview) => [preview.candidate_id, preview]));
  const expertCards = STYLES.map((style) => {
    const expert = row.experts?.[style];
    const preview = previewsById.get(`${row.category}_${row.sample_id}_${style}`);
    if (!expert) return '';
    const metrics = expert.metrics || {};
    const selected = row.selected_expert === style;
    return `<article class="expert-card ${selected ? 'selected' : ''}">
      <div class="expert-card-head"><div><b>${style}</b>${selected ? '<span class="selected-badge">最终选择</span>' : ''}</div><span>router ${fmt(row.router_target?.[style],3)}</span></div>
      <div class="expert-metrics"><span>几何 ${fmt(expert.geometry_score,3)}</span><span>LLM ${fmt(expert.llm_style_score,3)}</span><span>综合 ${fmt(expert.combined_score,3)}</span><span>质量 ${fmt(metrics.multidimensional_quality_score,3)}</span><span>OLR ${fmt(metrics.olr,3)}</span><span>LCD ${fmt(metrics.lcd,3)}</span><span>越界 ${fmt(metrics.viewport_overflow_ratio,3)}</span><span>遮挡 ${fmt(metrics.label_object_occlusion_ratio,3)}</span><span>线重叠 ${fmt(metrics.visible_leader_overlap_ratio,3)}</span></div>
      <div class="expert-preview-views">${preview ? Object.entries(preview.views || {}).map(([view, src]) => `<div class="view"><span>${view}</span><img src="/${src}?v=${encodeURIComponent(latestSnapshot?.generated_at || '')}" loading="lazy" alt="${row.category}/${row.sample_id} ${style} ${view}"></div>`).join('') : '<p class="hint">暂无预览</p>'}</div>
    </article>`;
  }).join('');
  $('testExpertPanel').innerHTML = `<div class="test-title"><b>${row.category}/${row.sample_id}</b><span>选中专家：${row.selected_expert} · router ${STYLES.map(style => `${style} ${fmt(row.router_target?.[style],3)}`).join(' / ')}</span></div><div class="expert-grid">${expertCards}</div>`;
}
function initTestSelector() {
  const selector = $('testSelector');
  const oldValue = selector.value;
  selector.innerHTML = testSamples.map((row, index) => `<option value="${index}">${row.category} / ${row.sample_id} · ${row.selected_expert}</option>`).join('');
  const next = Math.min(Math.max(Number(oldValue || 0), 0), Math.max(0, testSamples.length - 1));
  selector.value = String(next);
  selector.onchange = () => renderTestSample(Number(selector.value));
  renderTestAverage();
  renderTestSample(next);
}
function renderDashboard(data) {
  latestSnapshot = data;
  testSamples = data.test_samples || [];
  $('status').textContent = `已同步 ${data.generated_at} · SSE 实时更新已开启 · test11=${testSamples.length}`;
  const s = data.summaries || {};
  $('summaryGrid').innerHTML = [
    card('风格路由审计', s.routing?.count ?? '—', s.routing ? `train/val/test 已汇总` : '尚未运行 audit'),
    card('无监督伪标签', s.pseudo_labels?.count ?? '—', s.pseudo_labels ? `质量 ${fmt(s.pseudo_labels.summary?.multidimensional_quality_score)} · LCD ${fmt(s.pseudo_labels.summary?.lcd)}` : ''),
    card('LLM 风格初筛', s.llm_style?.count ?? '—', s.llm_style ? `sph ${s.llm_style.counts.spherical} / rect ${s.llm_style.counts.rectangular} / sur ${s.llm_style.counts.surround}` : ''),
    card('test11 专家布局', s.test_expert_selection?.count ?? '—', s.test_expert_selection ? `sph ${s.test_expert_selection.summary.selected_counts.spherical} / rect ${s.test_expert_selection.summary.selected_counts.rectangular} / sur ${s.test_expert_selection.summary.selected_counts.surround}` : ''),
    card('可视线重叠', fmt(testSamples.length ? testSamples.reduce((sum, row) => sum + Number(row.experts?.[row.selected_expert]?.metrics?.visible_leader_overlap_ratio || 0), 0) / testSamples.length : null), 'test11 选中专家平均'),
    card('v10 伪标签训练', s.pseudo_training?.pseudo_target_count ?? '—', s.pseudo_training ? `val ${fmt(s.pseudo_training.val_quality)} · test ${fmt(s.pseudo_training.test_quality)}` : '')
  ].join('');
  const comparison = s.extended_comparison;
  if (comparison) {
    const methods = comparison.methods || {};
    const cards = Object.entries(methods).map(([name, item]) => { const metrics = item.metrics || {}; const count = item.count ?? item.available_samples ?? 0; const total = item.total ?? comparison.cohort?.sample_count ?? 50; return `<article class=\"comparison-card\"><h3>${name}</h3><strong>${count} / ${total}</strong><span>可比样本覆盖率</span>${metrics.multidimensional_quality_score !== undefined ? `<small>质量 ${fmt(metrics.multidimensional_quality_score)} · OLR ${fmt(metrics.olr)} · LCD ${fmt(metrics.lcd)}</small>` : '<small>经典布局文件未覆盖全部 test50，保持 N/A</small>'}</article>`; }).join('');
    const legacy = s.legacy_comparison?.test_summaries || [];
    const legacyCards = legacy.filter(row => ['BinoForce','hedgehog_1d','hedgehog_3d'].includes(row.method)).map(row => `<article class=\"comparison-card legacy\"><h3>旧 test11 ${row.label}</h3><strong>${fmt(row.quality_score,2)}</strong><span>legacy test11 quality score · n=${row.sample_count}</span><small>OLR ${fmt(row.OLR)} · LCD ${fmt(row.LCD)}</small></article>`).join('');
    $('comparisonPanel').innerHTML = cards + (legacyCards ? `<div class=\"legacy-caption\">旧 55 样本 test11 经典方法对照（与扩展 test50 不混合）</div>${legacyCards}` : '');
  } else $('comparisonPanel').innerHTML = '<p class=\"hint\">尚未生成 extended test50 comparison。</p>';
  $('architecture').innerHTML = [renderFiles('模型代码操作点', data.architecture.model_files_to_operate),renderFiles('数据与训练产物', data.architecture.data_files_to_operate),renderFiles('主要实验文件', Object.values(data.files || {}).filter(f=>f.exists).map(f=>f.path))].join('');
  $('workflow').innerHTML = data.workflow.map((step) => `<article class="step"><b>${step.step}</b><div><code>${step.command}</code>${pathLink(step.artifact)}</div></article>`).join('');
  $('winners').innerHTML = (s.candidate_scores?.winners || []).map((w) => `<article class="winner"><b>${w.category}/${w.sample_id}</b><br>推荐扰动：${w.mode}<br>选择分：${fmt(w.selection_score ?? w.composite_score,2)}<br><small>LLM ${fmt(w.composite_score,2)} · 几何质量 ${fmt(w.deterministic_quality_score,2)}</small></article>`).join('') || '<p class="hint">尚未运行候选评分。</p>';
  initTestSelector();
  const previews = data.previews || [];
  $('previews').innerHTML = previews.slice(0, 20).map((item) => { const score = (data.summaries.candidate_scores?.winners || []).find((w) => w.candidate_id === item.candidate_id); return `<article class="preview-card"><div class="preview-head"><div><b>${item.category}/${item.sample_id}</b> · ${item.mode}<div class="metrics">style ${item.style || '—'} · selection ${score ? fmt(score.selection_score ?? score.composite_score,2) : '—'} · LLM ${score ? fmt(score.composite_score,2) : '—'}</div></div><div class="metrics">${item.candidate_id}</div></div><div class="views">${Object.entries(item.views || {}).map(([view, src]) => `<div class="view"><span>${view}</span><img src="/${src}?v=${encodeURIComponent(data.generated_at || '')}" loading="lazy" alt="${item.candidate_id} ${view}"></div>`).join('')}</div></article>`; }).join('') || '<p class="hint">尚未生成预览图。</p>';
}
async function loadDashboard() { const res = await fetch('/api/moe-dashboard', { cache: 'no-store' }); if (!res.ok) throw new Error(await res.text()); renderDashboard(await res.json()); }
async function main() {
  try { await loadDashboard(); } catch (error) { $('status').textContent = `读取失败：${error.message}`; $('status').style.color = '#b42318'; }
  if (window.EventSource) {
    const stream = new EventSource('/api/moe-dashboard/stream');
    stream.onmessage = (event) => { try { renderDashboard(JSON.parse(event.data)); } catch (error) { console.warn('MoE dashboard SSE parse failed', error); } };
    stream.onerror = () => { $('status').textContent = '实时连接暂时断开，等待自动重连…'; $('status').style.color = '#a15c00'; };
  }
  setInterval(() => { if (!window.EventSource || !latestSnapshot) loadDashboard().catch(() => {}); }, 10000);
}
main();
