import { FormEvent, useEffect, useState } from "react";

import AppShell from "@/components/layout/AppShell";
import { useAppStore } from "@/store/useAppStore";
import { isOfficialDatasetDocument } from "@/utils/documentScope";

const defaultOptions = ["提供给付责任", "不提供给付责任", "责任未明确", "属于免责范围"];

export default function TasksPage() {
  const { loading, error, documents, runQuestionTask, refreshDocuments } = useAppStore();
  const [qid, setQid] = useState("Q-001");
  const [mode, setMode] = useState<"A" | "B">("A");
  const [question, setQuestion] = useState("根据文档内容，以下关于保险责任的说法哪一项正确？");
  const [docIds, setDocIds] = useState("");
  const [options, setOptions] = useState(defaultOptions);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const visibleDocuments = documents.filter(isOfficialDatasetDocument);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runQuestionTask({
      qid,
      mode,
      question,
      options,
      answerFormat: "single",
      docIds: mode === "A" ? docIds.split(",").map((item) => item.trim()).filter(Boolean) : undefined,
    });
  }

  return (
    <AppShell title="题目执行页" description="运行 A 榜或 B 榜任务，触发规则召回、证据筛选和逐选项推理。">
      <section className="grid gap-6 xl:grid-cols-[1.25fr_0.9fr]">
        <form onSubmit={handleSubmit} className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">题号</span>
              <input value={qid} onChange={(event) => setQid(event.target.value)} className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white" />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">模式</span>
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as "A" | "B")}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white"
              >
                <option value="A">A 榜：已知文档</option>
                <option value="B">B 榜：盲检索</option>
              </select>
            </label>
          </div>

          <label className="mt-4 block">
            <span className="mb-2 block text-sm text-slate-300">题干</span>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={4} className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white" />
          </label>

          {options.map((option, index) => (
            <label key={index} className="mt-4 block">
              <span className="mb-2 block text-sm text-slate-300">选项 {String.fromCharCode(65 + index)}</span>
              <input
                value={option}
                onChange={(event) => setOptions(options.map((item, optionIndex) => (optionIndex === index ? event.target.value : item)))}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white"
              />
            </label>
          ))}

          {mode === "A" ? (
            <label className="mt-4 block">
              <span className="mb-2 block text-sm text-slate-300">文档 ID 列表</span>
              <input
                value={docIds}
                onChange={(event) => setDocIds(event.target.value)}
                placeholder="使用逗号分隔，留空表示手动填写"
                className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white"
              />
            </label>
          ) : null}

          <button type="submit" disabled={loading} className="mt-6 rounded-2xl bg-cyan-400 px-5 py-3 text-sm font-medium text-slate-950 hover:bg-cyan-300 disabled:bg-cyan-700">
            {loading ? "执行中..." : "运行任务"}
          </button>
          {error ? <p className="mt-3 text-sm text-rose-300">{error}</p> : null}
        </form>

        <section className="rounded-3xl border border-white/10 bg-slate-900/60 p-6">
          <h3 className="text-lg font-semibold text-white">可用文档提示</h3>
          <div className="mt-5 space-y-3">
            {visibleDocuments.map((document) => (
              <article key={document.documentId} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-sm font-medium text-white">{document.title}</p>
                <p className="mt-2 text-xs text-slate-400">documentId: {document.documentId}</p>
                <p className="mt-1 text-xs text-slate-500">状态：{document.status}</p>
              </article>
            ))}
            {visibleDocuments.length === 0 ? <p className="text-sm text-slate-400">暂无正式题库文档提示；如需自定义测试，可继续手动填写文档 ID。</p> : null}
          </div>
        </section>
      </section>
    </AppShell>
  );
}
