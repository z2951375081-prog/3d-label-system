import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dataRoot = path.join(root, 'data', 'Layout');
const outputFile = path.join(root, 'experiments', 'dataset_manifest.json');
const rel = (file) => path.relative(root, file).split(path.sep).join('/');
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });

async function listDirectories(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries.filter((entry) => entry.isDirectory()).map((entry) => path.join(dir, entry.name)).sort(natural);
}

async function listFiles(dir) {
  try { return (await fs.readdir(dir, { withFileTypes: true })).filter((entry) => entry.isFile()).map((entry) => path.join(dir, entry.name)).sort(natural); } catch { return []; }
}

async function main() {
  const samples = [];
  for (const categoryPath of await listDirectories(dataRoot)) {
    const category = path.basename(categoryPath);
    const samplePaths = await listDirectories(categoryPath);
    for (let index = 0; index < samplePaths.length; index += 1) {
      const samplePath = samplePaths[index];
      const sampleId = path.basename(samplePath);
      const layoutDir = path.join(samplePath, 'layout1');
      const objFiles = (await listFiles(path.join(layoutDir, 'Obj-O'))).filter((file) => file.toLowerCase().endsWith('.obj'));
      const mainObj = objFiles.find((file) => /-main-o\.obj$/i.test(file)) || objFiles[0];
      if (!mainObj) continue;
      const annotationPath = path.join(layoutDir, 'Annotation', `${sampleId}.json`);
      const annotationExists = await fs.stat(annotationPath).then(() => true).catch(() => false);
      const views = (await listFiles(path.join(layoutDir, 'Mutiviews'))).filter((file) => file.toLowerCase().endsWith('.png')).map(rel);
      samples.push({
        category,
        sample_id: sampleId,
        split: index % 5 < 3 ? 'train' : index % 5 === 3 ? 'val' : 'test',
        input: { source_obj: rel(mainObj), clean_obj: null, cleaning: { remove_groups: ['label_*', 'leader_*'], remove_materials: ['label_*', 'leader_*'], normalize_materials: ['anchor_region_*'], emitted_material: 'object_default', mode: 'rebuild_faces_and_reindex' } },
        target: { annotation_json: annotationExists ? rel(annotationPath) : null, labeled_obj: null, generated_at_runtime: true },
        views: { expected: 5, files: views }
      });
    }
  }
  const counts = { train: 0, val: 0, test: 0, by_category: {} };
  for (const sample of samples) { counts[sample.split] += 1; counts.by_category[sample.category] ??= { train: 0, val: 0, test: 0 }; counts.by_category[sample.category][sample.split] += 1; }
  const manifest = { version: '1.0', generated_at: new Date().toISOString(), policy: 'stratified_by_category_mod5', note: 'clean_obj is derived at runtime; raw Obj-O files are never overwritten.', counts, samples };
  await fs.mkdir(path.dirname(outputFile), { recursive: true });
  await fs.writeFile(outputFile, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`Wrote ${rel(outputFile)}: ${samples.length} samples (${counts.train}/${counts.val}/${counts.test})`);
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
