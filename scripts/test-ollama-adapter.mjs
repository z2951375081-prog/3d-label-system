import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_OLLAMA_MODEL, VISION_KEYS, normalizeOllamaEndpoint, ollamaScoringRequest, parseOllamaScoringResponse } from '../lib/ollama-vision-adapter.mjs';
import { LLM_SCORE_NAMES } from '../public/preference-scoring.js';

const schema = { properties: { scores: { required: LLM_SCORE_NAMES } } };
const scores = Object.fromEntries(LLM_SCORE_NAMES.map((name) => [name, 4]));
const visuals = Object.fromEntries(VISION_KEYS.map((name) => [name, 'data:image/png;base64,aGVsbG8=']));

test('only loopback Ollama endpoints are accepted', () => {
  assert.equal(normalizeOllamaEndpoint('http://127.0.0.1:11434').apiUrl, 'http://127.0.0.1:11434/api/chat');
  assert.equal(normalizeOllamaEndpoint('http://localhost:11434/api/chat').baseUrl, 'http://localhost:11434');
  assert.throws(() => normalizeOllamaEndpoint('https://api.openai.com/v1'));
  assert.throws(() => normalizeOllamaEndpoint('http://192.168.1.2:11434'));
});

test('Ollama request sends exactly six raw base64 images and JSON schema', () => {
  const body = ollamaScoringRequest({ model: DEFAULT_OLLAMA_MODEL, prompt: '{}', instruction: 'score', visuals, schema });
  assert.equal(body.model, 'qwen3-vl:4b-instruct');
  assert.equal(body.messages[1].images.length, 6);
  assert.equal(body.messages[1].images[0], 'aGVsbG8=');
  assert.equal(body.format, schema);
  assert.equal(body.stream, false);
  assert.match(body.messages[0].content, /3\.7/);
  assert.match(body.messages[0].content, /平局/);
});

test('Ollama structured response requires all fourteen dimensions', () => {
  const parsed = parseOllamaScoringResponse({ message: { content: JSON.stringify({ scores, rationale: 'ok', risks: [], suggested_changes: [] }) }, eval_count: 10 }, schema);
  assert.equal(parsed.result.scores.overall, 4);
  assert.equal(parsed.usage.eval_count, 10);
  assert.throws(() => parseOllamaScoringResponse({ message: { content: '{}' } }, schema));
});

test('unit-interval evidence accidentally copied by Qwen is calibrated to 1-5 scores', () => {
  const copied = { ...scores, label_label_occlusion: 0.1, object_penetration: 0, spatial_balance: 0.75, composition_harmony: 0.5 };
  const parsed = parseOllamaScoringResponse({ message: { content: JSON.stringify({ scores: copied, rationale: 'ok', risks: [], suggested_changes: [] }) } }, schema);
  assert.equal(parsed.result.scores.label_label_occlusion, 4.6);
  assert.equal(parsed.result.scores.object_penetration, 5);
  assert.equal(parsed.result.scores.spatial_balance, 4);
  assert.equal(parsed.result.scores.composition_harmony, 3);
});

test('genuine fractional aesthetics survive parsing without artificial tie breaks', () => {
  const fractional = { ...scores, overall: 4.3, composition_harmony: 4.6, spatial_balance: 4.1 };
  const parsed = parseOllamaScoringResponse({ message: { content: JSON.stringify({ scores: fractional, rationale: 'visible balance', risks: [], suggested_changes: [] }) } }, schema);
  assert.equal(parsed.result.scores.overall, 4.3);
  assert.equal(parsed.result.scores.composition_harmony, 4.6);
  assert.equal(parsed.result.scores.spatial_balance, 4.1);
});

test('percentage-style consistency values are calibrated to the common 1-5 scale', () => {
  const percentages = { ...scores, multiview_consistency: 100, binocular_consistency: 75 };
  const parsed = parseOllamaScoringResponse({ message: { content: JSON.stringify({ scores: percentages, rationale: 'ok', risks: [], suggested_changes: [] }) } }, schema);
  assert.equal(parsed.result.scores.multiview_consistency, 5);
  assert.equal(parsed.result.scores.binocular_consistency, 4);
  assert.deepEqual(parsed.result.normalized_from_percentage, ['multiview_consistency', 'binocular_consistency']);
});
