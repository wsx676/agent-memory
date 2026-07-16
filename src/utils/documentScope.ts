import type { DocumentRecord, TaskResult } from "@/utils/api";

function normalizePath(value: string): string {
  return value.replace(/\\/g, "/").toLowerCase();
}

export function isPytestTempPath(sourcePath: string): boolean {
  const normalized = normalizePath(sourcePath);
  return normalized.includes("/appdata/local/temp/pytest-of-") || normalized.includes("/pytest-of-");
}

export function isOfficialDatasetPath(sourcePath: string): boolean {
  return normalizePath(sourcePath).includes("/public_dataset_a/raw/");
}

export function isOfficialDatasetDocument(document: DocumentRecord): boolean {
  return Boolean(document.sourcePath) && isOfficialDatasetPath(document.sourcePath) && !isPytestTempPath(document.sourcePath);
}

export function isOfficialGroupAQid(qid: string): boolean {
  return /^(fc|fin|ins|reg|res)_a_\d+$/i.test(qid);
}

export function shouldShowOfficialResult(result: TaskResult, officialDocumentIds: Set<string>): boolean {
  return isOfficialGroupAQid(result.qid) || result.evidence.some((item) => officialDocumentIds.has(item.docId));
}
