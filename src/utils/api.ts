export type DashboardSummary = {
  documents: number;
  processedDocuments: number;
  tasks: number;
  results: number;
  deliveryReady: boolean;
};

export type DocumentRecord = {
  documentId: string;
  title: string;
  fileType: string;
  sourcePath: string;
  status: string;
  chunkCount: number;
};

export type TaskResult = {
  taskId: string;
  qid: string;
  answer: string;
  evidence: Array<{
    docId: string;
    pageNo: number;
    clauseNo?: string;
    quotedText: string;
    supportsOption: string[];
    reasoning: string;
  }>;
  tokenUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  llmTrace: {
    enabled: boolean;
    attempted: boolean;
    used: string;
    requestedModel: string;
    actualModel: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    fallback: boolean;
    fallbackReason: string;
    error?: string | null;
    missingSettings: string[];
  };
  logs: string[];
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getDashboard: () => fetchJson<DashboardSummary>("/api/dashboard"),
  getDocuments: () => fetchJson<DocumentRecord[]>("/api/documents"),
  createDocument: (payload: { fileName: string; fileType: "pdf" | "txt"; sourcePath?: string }) =>
    fetchJson<{ documentId: string; status: string }>("/api/documents/upload", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  preprocessDocument: (payload: { documentId: string; enableOcr: boolean; enableTableRecovery: boolean }) =>
    fetchJson<{ taskId: string; status: string }>("/api/preprocess/start", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runTask: (payload: {
    mode: "A" | "B";
    qid: string;
    question: string;
    options: string[];
    answerFormat: "single" | "multi" | "judge";
    docIds?: string[];
  }) =>
    fetchJson<{ taskId: string; status: string }>("/api/tasks/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getTaskResult: (taskId: string) => fetchJson<TaskResult>(`/api/tasks/${taskId}`),
  getResults: () => fetchJson<TaskResult[]>("/api/results"),
  getQuality: () => fetchJson<Record<string, string>>("/api/quality"),
  getDelivery: () => fetchJson<Array<{ name: string; description: string; path: string }>>("/api/delivery"),
};
