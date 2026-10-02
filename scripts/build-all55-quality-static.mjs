// Generate only the compact, public per-method summary; never expose the
// original annotations, raw image scores, credentials, or train samples.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(root, 'experiments', 'comparisons', 'round1', 'comparison.json');
const target = path.join(root, 'public', 'quality-all55.json');
const methods = [
  ['BinoForce_final_snapshot', 'BinoForce'], ['hedgehog_1d', 'Hedgehog 1D'],
  ['hedgehog_3d', 'Hedgehog 3D'], ['current_fixed_label_seed17', '模型一（基础模型）']
];
const fields = [
  'multidimensional_quality_score', 'text_clarity', 'label_label_occlusion_ratio',
  'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio',
  'mean_anchor_distance', 'leader_length_compliance_ratio', 'directional_allocation_mismatch',
  'directional_uniformity', 'manual_style_distance'
];
const raw = JSON.parse(await fs.readFile(source, 'utf8'));
const all = raw.unified_snapshot_evaluation?.rows;
if (!Array.isArray(all)) throw new Error('缺少统一相机逐样本比较结果');
const averages = methods.map(([method, label]) => {
  const rows = all.filter((row) => row.method === method && row.strict_label_contract !== false);
  const unique = new Set(rows.map((row) => `${row.category}/${row.sample_id}`));
  const splits = Object.fromEntries(['train', 'val', 'test'].map((split) => [split, rows.filter((row) => row.split === split).length]));
  if (rows.length !== 55 || unique.size !== 55 || splits.train !== 33 || splits.val !== 11 || splits.test !== 11) throw new Error(`${method}: 33/11/11 全 55 样本不完整`);
  return { method, label, sample_count: rows.length, ...Object.fromEntries(fields.map((field) => {
    const values = rows.map((row) => row[field] ?? row.manual_similarity?.[field])
      .filter((value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))).map(Number);
    return [field, values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null];
  })) };
});
const result = { protocol: 'unified_dataset_camera_55_samples', source: 'experiments/comparisons/round1/comparison.json', split_counts: { train: 33, val: 11, test: 11 }, averages };
const serialized = `${JSON.stringify(result, null, 2)}\n`;
if (process.argv.includes('--verify')) {
  if (await fs.readFile(target, 'utf8') !== serialized) throw new Error('全 55 样本静态汇总与源文件不一致');
  console.log('全 55 样本静态汇总与原始比较文件一致');
} else {
  await fs.writeFile(target, serialized, 'utf8');
  console.log(`生成全 55 样本汇总：${target}`);
}
