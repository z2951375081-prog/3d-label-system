import fs from 'node:fs/promises';
import path from 'node:path';
import { constants as fsConstants } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';

export function sha256Buffer(buffer) {
  return createHash('sha256').update(buffer).digest('hex').toUpperCase();
}

export async function sha256File(file) {
  return sha256Buffer(await fs.readFile(file));
}

function activationStamp(date = new Date()) {
  return date.toISOString().replace(/[:.]/g, '-');
}

export async function activatePreservingPrevious({ candidateFile, activeFile, historyFile, validateCandidate = null, metadata = {}, afterActivate = null, rollbackAfterFailure = null }) {
  const candidatePath = path.resolve(candidateFile);
  const activePath = path.resolve(activeFile);
  if (candidatePath === activePath) throw new Error('候选模型与活动模型路径不能相同');

  const candidateBytes = await fs.readFile(candidatePath);
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (validateCandidate) await validateCandidate(candidate);
  const candidateHash = sha256Buffer(candidateBytes);

  await fs.mkdir(path.dirname(activePath), { recursive: true });
  let previousHash = null;
  let previousBytes = null;
  let backupFile = null;
  try {
    previousBytes = await fs.readFile(activePath);
    previousHash = sha256Buffer(previousBytes);
    const extension = path.extname(activePath) || '.json';
    const stem = path.basename(activePath, extension);
    backupFile = path.join(path.dirname(activePath), `${stem}_before_activation_${activationStamp()}_${previousHash.slice(0, 12)}${extension}`);
    await fs.copyFile(activePath, backupFile, fsConstants.COPYFILE_EXCL);
    if (await sha256File(backupFile) !== previousHash) throw new Error('活动模型备份哈希校验失败，已中止激活');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const pendingFile = path.join(path.dirname(activePath), `.${path.basename(activePath)}.pending-${randomUUID()}`);
  try {
    await fs.writeFile(pendingFile, candidateBytes, { flag: 'wx' });
    if (await sha256File(pendingFile) !== candidateHash) throw new Error('候选模型临时文件哈希校验失败，已中止激活');
    await fs.rename(pendingFile, activePath);
  } finally {
    await fs.rm(pendingFile, { force: true }).catch(() => {});
  }
  if (await sha256File(activePath) !== candidateHash) throw new Error('活动模型激活后哈希校验失败，请使用自动备份恢复');

  const record = {
    version: 'model_activation_record_v1',
    activated_at: new Date().toISOString(),
    active_file: activePath,
    candidate_file: candidatePath,
    backup_file: backupFile,
    previous_sha256: previousHash,
    activated_sha256: candidateHash,
    candidate_version: candidate.version || null,
    metadata
  };
  try {
    if (afterActivate) await afterActivate({ candidate, record, candidateBytes, previousBytes });
    if (historyFile) {
      const historyPath = path.resolve(historyFile);
      await fs.mkdir(path.dirname(historyPath), { recursive: true });
      await fs.appendFile(historyPath, `${JSON.stringify(record)}\n`, 'utf8');
    }
  } catch (error) {
    const rollbackErrors = [];
    try {
      const currentHash = await sha256File(activePath).catch(() => null);
      if (currentHash !== candidateHash) throw new Error('激活后置步骤失败，但活动模型已被并发修改，拒绝自动覆盖');
      if (previousBytes) {
        const rollbackFile = path.join(path.dirname(activePath), `.${path.basename(activePath)}.rollback-${randomUUID()}`);
        try {
          await fs.writeFile(rollbackFile, previousBytes, { flag: 'wx' });
          if (await sha256File(rollbackFile) !== previousHash) throw new Error('旧活动模型回滚临时文件哈希不匹配');
          await fs.rename(rollbackFile, activePath);
        } finally {
          await fs.rm(rollbackFile, { force: true }).catch(() => {});
        }
        if (await sha256File(activePath) !== previousHash) throw new Error('旧活动模型回滚后哈希不匹配');
      } else {
        await fs.rm(activePath, { force: true });
      }
    } catch (rollbackError) { rollbackErrors.push(rollbackError); }
    if (rollbackAfterFailure) {
      try { await rollbackAfterFailure({ candidate, record, candidateBytes, previousBytes, cause: error }); }
      catch (rollbackError) { rollbackErrors.push(rollbackError); }
    }
    if (rollbackErrors.length) throw new AggregateError([error, ...rollbackErrors], '模型激活后置步骤失败，且自动回滚不完整');
    throw error;
  }
  return { candidate, record };
}
