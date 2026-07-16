import { FormEvent, useEffect, useState } from "react";

import AppShell from "@/components/layout/AppShell";
import { useAppStore } from "@/store/useAppStore";

export default function DocumentsPage() {
  const { documents, loading, error, addDocument, preprocessDocument, refreshDocuments } = useAppStore();
  const [fileName, setFileName] = useState("示例文档.txt");
  const [sourcePath, setSourcePath] = useState("");

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await addDocument({ fileName, fileType: fileName.endsWith(".pdf") ? "pdf" : "txt", sourcePath });
    setSourcePath("");
  }

  return (
    <AppShell title="文档资产页" description="录入 PDF/TXT 文档、启动预处理任务，并查看结构化资产的准备情况。">
      <section className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
        <form onSubmit={handleSubmit} className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-white">文档导入</h3>
          <div className="mt-5 space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">文件名</span>
              <input
                value={fileName}
                onChange={(event) => setFileName(event.target.value)}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-300"
              />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">源文件路径</span>
              <input
                value={sourcePath}
                onChange={(event) => setSourcePath(event.target.value)}
                placeholder="例如：C:\data\finance\sample.txt"
                className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-300"
              />
            </label>
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-2xl bg-cyan-400 px-4 py-3 text-sm font-medium text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-cyan-700"
            >
              {loading ? "处理中..." : "登记文档"}
            </button>
            {error ? <p className="text-sm text-rose-300">{error}</p> : null}
          </div>
        </form>

        <section className="rounded-3xl border border-white/10 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">文档列表</h3>
            <button onClick={() => void refreshDocuments()} className="rounded-full border border-white/10 px-3 py-2 text-xs text-slate-300 hover:bg-white/5">
              刷新
            </button>
          </div>
          <div className="mt-5 space-y-3">
            {documents.map((document) => (
              <article key={document.documentId} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                  <div>
                    <p className="text-sm font-medium text-white">{document.title}</p>
                    <p className="mt-1 text-xs text-slate-400">{document.sourcePath || "尚未配置源路径"}</p>
                    <p className="mt-2 text-xs text-slate-500">状态：{document.status} | 切片数：{document.chunkCount}</p>
                  </div>
                  <button
                    onClick={() => void preprocessDocument(document.documentId)}
                    className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-400/20"
                  >
                    启动预处理
                  </button>
                </div>
              </article>
            ))}
            {documents.length === 0 ? <p className="text-sm text-slate-400">当前还没有录入文档。</p> : null}
          </div>
        </section>
      </section>
    </AppShell>
  );
}
