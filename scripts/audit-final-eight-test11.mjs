import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateAndSummarizeFinalEightTest11 } from '../lib/final-eight-test11-results.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const reportPath = path.join(root, 'experiments', 'comparisons', 'final_eight_test11', 'comparison.json');
const auditPath = path.join(root, 'experiments', 'comparisons', 'final_eight_test11', 'comparison_audit.json');
const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
const result = validateAndSummarizeFinalEightTest11(report);
const labels = new Map((report.summaries || []).map((row) => [row.method, row.label || row.method]));
const auditReport = {
  version: 'final_eight_test11_audit_v1',
  audited_at: new Date().toISOString(),
  source_file: path.relative(root, reportPath).replaceAll('\\', '/'),
  source_generated_at: report.generated_at,
  ...result.audit,
  summaries: result.summaries.map((row) => ({ ...row, label: labels.get(row.method) || row.method }))
};

await fs.writeFile(auditPath, JSON.stringify(auditReport, null, 2) + '\n');
console.log(JSON.stringify(auditReport, null, 2));
