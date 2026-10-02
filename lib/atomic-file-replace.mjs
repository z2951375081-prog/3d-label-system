import fs from 'node:fs/promises';
import path from 'node:path';

const RETRYABLE_WINDOWS_RENAME_CODES = new Set(['EPERM', 'EACCES', 'EBUSY']);
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export function isRetryableWindowsRenameError(error) {
  return RETRYABLE_WINDOWS_RENAME_CODES.has(error?.code);
}

export async function replaceFileWithRetry(temporaryFile, targetFile, options = {}) {
  const temporary = path.resolve(String(temporaryFile || ''));
  const target = path.resolve(String(targetFile || ''));
  if (temporary === target) throw new Error('Atomic replacement requires distinct temporary and target files');

  const operations = options.operations || fs;
  const sleep = options.sleep || delay;
  const maxAttempts = Number.isInteger(options.maxAttempts) && options.maxAttempts > 0 ? options.maxAttempts : 7;
  const initialDelayMs = Number.isFinite(options.initialDelayMs) && options.initialDelayMs >= 0 ? options.initialDelayMs : 25;
  const maxDelayMs = Number.isFinite(options.maxDelayMs) && options.maxDelayMs >= initialDelayMs ? options.maxDelayMs : 800;

  let attempts = 0;
  while (attempts < maxAttempts) {
    attempts += 1;
    try {
      await operations.rename(temporary, target);
      return { attempts, temporary_file: temporary, target_file: target };
    } catch (error) {
      if (isRetryableWindowsRenameError(error) && attempts < maxAttempts) {
        await sleep(Math.min(maxDelayMs, initialDelayMs * (2 ** (attempts - 1))));
        continue;
      }

      let cleanupError = null;
      try { await operations.rm(temporary, { force: true }); }
      catch (failure) { cleanupError = failure; }
      if (cleanupError) {
        const aggregate = new AggregateError([error, cleanupError], `Atomic replacement failed after ${attempts} attempt(s), and its owned temporary file could not be removed`);
        aggregate.code = error?.code;
        aggregate.renameError = error;
        aggregate.cleanupError = cleanupError;
        throw aggregate;
      }
      error.message = `${error.message} (atomic replacement failed after ${attempts} attempt(s); owned temporary file removed)`;
      throw error;
    }
  }
  throw new Error('Atomic replacement retry loop terminated unexpectedly');
}
