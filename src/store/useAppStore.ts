import { create } from "zustand";

import { api, type DashboardSummary, type DocumentRecord, type TaskResult } from "@/utils/api";

type AppState = {
  dashboard: DashboardSummary | null;
  documents: DocumentRecord[];
  results: TaskResult[];
  loading: boolean;
  error: string | null;
  lastTaskId: string | null;
  refreshDashboard: () => Promise<void>;
  refreshDocuments: () => Promise<void>;
  refreshResults: () => Promise<void>;
  addDocument: (payload: { fileName: string; fileType: "pdf" | "txt"; sourcePath?: string }) => Promise<void>;
  preprocessDocument: (documentId: string) => Promise<void>;
  runQuestionTask: (payload: {
    mode: "A" | "B";
    qid: string;
    question: string;
    options: string[];
    answerFormat: "single" | "multi" | "judge";
    docIds?: string[];
  }) => Promise<void>;
};

export const useAppStore = create<AppState>((set, get) => ({
  dashboard: null,
  documents: [],
  results: [],
  loading: false,
  error: null,
  lastTaskId: null,
  refreshDashboard: async () => {
    const dashboard = await api.getDashboard();
    set({ dashboard });
  },
  refreshDocuments: async () => {
    const documents = await api.getDocuments();
    set({ documents });
  },
  refreshResults: async () => {
    const results = await api.getResults();
    set({ results });
  },
  addDocument: async (payload) => {
    set({ loading: true, error: null });
    try {
      await api.createDocument(payload);
      await get().refreshDocuments();
      await get().refreshDashboard();
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "上传失败" });
    } finally {
      set({ loading: false });
    }
  },
  preprocessDocument: async (documentId) => {
    set({ loading: true, error: null });
    try {
      await api.preprocessDocument({
        documentId,
        enableOcr: true,
        enableTableRecovery: true,
      });
      await Promise.all([get().refreshDocuments(), get().refreshDashboard()]);
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "预处理失败" });
    } finally {
      set({ loading: false });
    }
  },
  runQuestionTask: async (payload) => {
    set({ loading: true, error: null });
    try {
      const response = await api.runTask(payload);
      const result = await api.getTaskResult(response.taskId);
      set({ lastTaskId: response.taskId, results: [result, ...get().results.filter((item) => item.taskId !== result.taskId)] });
      await get().refreshDashboard();
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "执行失败" });
    } finally {
      set({ loading: false });
    }
  },
}));
