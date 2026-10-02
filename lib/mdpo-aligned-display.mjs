import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

export async function loadLatestCompletedAlignedModel({ root, experimentsDir = path.join(root, 'experiments') }) {
  const mdpoDir = path.join(experimentsDir, 'mdpo');
  const sweepDir = path.join(mdpoDir, 'aligned_sweep');
  const ledgerFile = path.join(sweepDir, 'aligned_sweep_ledger.json');
  const [ledger, referenceBytes] = await Promise.all([
    fs.readFile(ledgerFile, 'utf8').then(JSON.parse),
    fs.readFile(path.join(experimentsDir, 'layout_model.json'))
  ]);
  const referenceSha256 = digest(referenceBytes);
  const trained = (ledger.configurations || []).filter((row) => row.status === 'trained' && row.id && row.completed_at)
    .toSorted((left, right) => Date.parse(right.completed_at) - Date.parse(left.completed_at));
  for (const row of trained) {
    const file = path.join(sweepDir, row.id, 'layout_model_v10_mdpo_candidate.json');
    const bytes = await fs.readFile(file).catch((error) => error?.code === 'ENOENT' ? null : Promise.reject(error));
    if (!bytes) continue;
    const model = JSON.parse(bytes.toString('utf8'));
    if (model.version !== 'layout_model_v10_mdpo_candidate'
      || model.status !== 'diagnostic_only_requires_full_val11_gate'
      || model.reference?.sha256 !== referenceSha256
      || model.mdpo?.objective?.protocol !== 'safety_priority_v3_aligned'
      || model.architecture?.qwen_inference_input !== false) continue;
    return {
      id: row.id,
      file: path.relative(root, file).split(path.sep).join('/'),
      sha256: digest(bytes),
      completed_at: row.completed_at,
      best_epoch: row.best_epoch ?? model.mdpo?.best_epoch ?? null,
      proxy_val: row.proxy_val || model.mdpo?.best_val || null,
      hyperparameters: row.hyperparameters || model.mdpo?.hyperparameters || null,
      deployment_status: row.deployment_status || model.status,
      reference_sha256: referenceSha256,
      model
    };
  }
  return null;
}

export async function loadSelectedAlignedModel({ root, experimentsDir = path.join(root, 'experiments') }) {
  const selectionFile = path.join(experimentsDir, 'mdpo', 'aligned_hyperparameter_selection.json');
  const [selection, referenceBytes] = await Promise.all([
    fs.readFile(selectionFile, 'utf8').then(JSON.parse),
    fs.readFile(path.join(experimentsDir, 'layout_model.json'))
  ]);
  const selected = selection?.selected;
  if (!selected?.id || !selected?.candidate_file) return null;
  const mdpoRoot = path.resolve(experimentsDir, 'mdpo');
  const file = path.resolve(root, selected.candidate_file);
  if (!file.startsWith(mdpoRoot + path.sep) || path.basename(file) !== 'layout_model_v10_mdpo_candidate.json') return null;
  const bytes = await fs.readFile(file);
  const sha256 = digest(bytes);
  const model = JSON.parse(bytes.toString('utf8'));
  const referenceSha256 = digest(referenceBytes);
  if (selected.candidate_sha256 && sha256 !== selected.candidate_sha256) throw new Error('Selected aligned MDPO candidate hash mismatch');
  if (model.version !== 'layout_model_v10_mdpo_candidate'
    || model.reference?.sha256 !== referenceSha256
    || model.mdpo?.objective?.protocol !== 'safety_priority_v3_aligned'
    || model.architecture?.qwen_inference_input !== false) throw new Error('Selected aligned MDPO candidate provenance is invalid');
  return {
    id: selected.id,
    file: path.relative(root, file).split(path.sep).join('/'),
    sha256,
    generated_at: selection.generated_at || null,
    hyperparameters: selected.hyperparameters || model.mdpo?.hyperparameters || null,
    deployment_status: selected.gate?.deployment_status || 'diagnostic_only',
    deployment_eligible: selection.deployment_eligible === true,
    model
  };
}
