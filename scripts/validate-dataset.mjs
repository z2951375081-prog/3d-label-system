import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanObj } from './generate-artifacts.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (file) => fs.readFile(file, 'utf8');
const repoFile = (relativePath) => path.join(root, relativePath.replaceAll('/', path.sep));
const optionValue = (name, fallback) => { const index = process.argv.indexOf(`--${name}`); return index >= 0 && process.argv[index + 1] && !process.argv[index + 1].startsWith('--') ? process.argv[index + 1] : fallback; };

async function main() {
  const manifest = JSON.parse(await read(path.resolve(root, optionValue('manifest', 'experiments/dataset_manifest.json'))));
  const failures = [];
  let totalFaces = 0;
  let totalVertices = 0;
  for (const sample of manifest.samples) {
    const source = await read(repoFile(sample.input.source_obj));
    const clean = cleanObj(source);
    totalFaces += clean.faceCount;
    totalVertices += clean.vertexCount;
    const forbidden = clean.text.match(/^(?:g|o|usemtl)\s+(?:label_|leader_|anchor_region_)/im);
    if (!clean.vertexCount || !clean.faceCount || forbidden) failures.push(`${sample.category}/${sample.sample_id}: vertices=${clean.vertexCount}, faces=${clean.faceCount}, forbidden=${Boolean(forbidden)}`);
  }
  console.log(`validated ${manifest.samples.length} samples: ${totalVertices} clean vertices, ${totalFaces} clean faces`);
  if (failures.length) { console.error(failures.join('\n')); process.exitCode = 1; return; }
  console.log('OK: every clean OBJ has geometry and no label/leader layer or anchor-region material styling.');
}

main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });



