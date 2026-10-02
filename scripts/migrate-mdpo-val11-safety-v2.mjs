import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const oldDir = path.join(mdpo, 'val11_reports');
const newDir = path.join(mdpo, 'val11_reports_safety_priority_v2');
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
await fs.mkdir(newDir, { recursive: true });
const names = (await fs.readdir(oldDir)).filter((name) => name.endsWith('.json') && !name.endsWith('.four_group.json'));
const migrated = [];
for (const name of names) {
  const id = name.slice(0, -5);
  const oldReport = JSON.parse(await fs.readFile(path.join(oldDir, name), 'utf8'));
  const oldFour = JSON.parse(await fs.readFile(path.join(oldDir, `${id}.four_group.json`), 'utf8'));
  if (oldReport.version !== 'v10_mdpo_val11_gate_report_v1' || oldFour.version !== 'v10_mdpo_four_group_val11_v1' || oldReport.test_not_used !== true || oldFour.test_not_used !== true) continue;
  const gate = evaluateMdpoVal11Gate({ baseline: oldReport.baseline, candidate: oldReport.candidate });
  const report = { ...oldReport, version: 'v10_mdpo_val11_gate_report_v2', evaluation_policy: 'safety_priority_v2', generated_at: new Date().toISOString(), gate, four_group_protocol: oldReport.four_group_protocol };
  const four = { ...oldFour, version: 'v10_mdpo_four_group_val11_v2', evaluation_policy: 'safety_priority_v2', generated_at: report.generated_at, core_gate: gate };
  const reportBytes = Buffer.from(JSON.stringify(report, null, 2) + '\n');
  const fourBytes = Buffer.from(JSON.stringify(four, null, 2) + '\n');
  await fs.writeFile(path.join(newDir, name), reportBytes, { flag: 'wx' }).catch(async (error) => { if (error.code !== 'EEXIST') throw error; });
  await fs.writeFile(path.join(newDir, `${id}.four_group.json`), fourBytes, { flag: 'wx' }).catch(async (error) => { if (error.code !== 'EEXIST') throw error; });
  migrated.push({ id, accepted: gate.accepted, report_sha256: digest(reportBytes), four_group_report_sha256: digest(fourBytes) });
}
console.log(JSON.stringify({ policy: 'safety_priority_v2', migrated: migrated.length, reports: migrated }, null, 2));
