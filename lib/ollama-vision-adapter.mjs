export const DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434';
// Local-only vision scoring adapter.
export const DEFAULT_OLLAMA_MODEL = 'qwen3-vl:4b-instruct';
export const VISION_KEYS = ['before', 'main', 'right', 'left', 'up', 'down'];

export function normalizeOllamaEndpoint(rawUrl = DEFAULT_OLLAMA_URL) {
  let parsed;
  try { parsed = new URL(String(rawUrl || DEFAULT_OLLAMA_URL).trim()); } catch { throw new Error('请输入本机 Ollama 地址，例如 http://127.0.0.1:11434'); }
  if (!['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname) || parsed.protocol !== 'http:' || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error('仅允许连接本机 HTTP Ollama 服务');
  const pathname = parsed.pathname.replace(/\/+$/, '').replace(/\/api\/chat$/, '');
  return { baseUrl: parsed.origin + pathname, apiUrl: parsed.origin + pathname + '/api/chat', format: 'ollama_chat' };
}

export function requireVisionImages(visuals) {
  const source = visuals && typeof visuals === 'object' ? visuals : {};
  if (VISION_KEYS.some((key) => typeof source[key] !== 'string' || !/^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(source[key]))) throw new Error('需要原始图和生成布局的主/右/左/俯/仰五视角，共 6 张有效图片');
  return source;
}

export function ollamaScoringRequest({ model, prompt, instruction, visuals, schema }) {
  const images = requireVisionImages(visuals);
  return {
    model,
    stream: false,
    format: schema,
    options: { temperature: 0, seed: 17, num_ctx: 32768 },
    messages: [
      { role: 'system', content: instruction + '\n所有 scores 字段必须在 1 到 5 之间，允许一位小数，5 表示最好。审美维度必须依据六张图像连续评分，例如 3.7、4.2、4.6。不要惯性地给安全候选整数 5，满分仅用于几乎没有可见改进空间的布局；看不出差别时允许平局，不能凭种子或几何目标捏造差异。输入中的 0 到 1 比例只是安全证据，不得原样复制；风险比例需反向换算为安全评分，正向质量比例需换算为质量评分。' },
      { role: 'user', content: prompt + '\n图像顺序：' + VISION_KEYS.join(', '), images: VISION_KEYS.map((key) => images[key].replace(/^data:image\/(?:png|jpeg|webp);base64,/, '')) }
    ]
  };
}

export function parseOllamaScoringResponse(data, schema) {
  const raw = String(data?.message?.content || '').trim().replace(/^\x60\x60\x60(?:json)?\s*/i, '').replace(/\s*\x60\x60\x60$/, '');
  let result;
  try { result = JSON.parse(raw); } catch { throw new Error('本地 Qwen 未返回可解析的 JSON 评分'); }
  const lowerIsBetterRatios = new Set(['label_label_occlusion', 'object_occlusion', 'object_penetration']);
  const normalizedFromRatios = [];
  const normalizedFromPercentages = [];
  for (const key of schema.properties.scores.required) {
    const value = Number(result?.scores?.[key]);
    if (Number.isFinite(value) && value >= 0 && value < 1) {
      result.scores[key] = lowerIsBetterRatios.has(key) ? 5 - 4 * value : 1 + 4 * value;
      normalizedFromRatios.push(key);
    } else if (Number.isFinite(value) && value > 5 && value <= 100) {
      const ratio = value / 100;
      result.scores[key] = lowerIsBetterRatios.has(key) ? 5 - 4 * ratio : 1 + 4 * ratio;
      normalizedFromPercentages.push(key);
    }
  }
  const invalid = schema.properties.scores.required.filter((key) => !Number.isFinite(Number(result?.scores?.[key])) || Number(result.scores[key]) < 1 || Number(result.scores[key]) > 5);
  if (invalid.length) throw new Error('本地 Qwen 返回的十四维评分不完整或超出 1-5 范围：' + invalid.join(', '));
  result.scores = Object.fromEntries(schema.properties.scores.required.map((key) => [key, Number(Number(result.scores[key]).toFixed(4))]));
  if (normalizedFromRatios.length) result.normalized_from_unit_interval = normalizedFromRatios;
  if (normalizedFromPercentages.length) result.normalized_from_percentage = normalizedFromPercentages;
  if (typeof result.rationale !== 'string' || !Array.isArray(result.risks) || !Array.isArray(result.suggested_changes)) throw new Error('本地 Qwen 返回的解释字段不符合评分 Schema');
  return { result, response_id: data?.created_at || null, usage: { prompt_eval_count: data?.prompt_eval_count ?? null, eval_count: data?.eval_count ?? null, total_duration_ns: data?.total_duration ?? null } };
}
