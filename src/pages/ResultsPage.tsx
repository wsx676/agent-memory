import { useEffect, useMemo } from "react";

import AppShell from "@/components/layout/AppShell";
import { useAppStore } from "@/store/useAppStore";
import { isOfficialDatasetDocument, shouldShowOfficialResult } from "@/utils/documentScope";

export default function ResultsPage() {
  const { documents, results, refreshDocuments, refreshResults } = useAppStore();

  useEffect(() => {
    void Promise.all([refreshDocuments(), refreshResults()]);
  }, [refreshDocuments, refreshResults]);

  const officialDocumentIds = useMemo(
    () => new Set(documents.filter(isOfficialDatasetDocument).map((document) => document.documentId)),
    [documents],
  );
  const visibleResults = useMemo(
    () => results.filter((result) => shouldShowOfficialResult(result, officialDocumentIds)),
    [results, officialDocumentIds],
  );

  return (
    <AppShell title="结果中心" description="查看题目答案、证据链、Token 统计和运行日志，确认提交产物是否完整。">
      <section className="space-y-6">
        {visibleResults.map((result) => {
          const visibleEvidence = result.evidence.filter((item) => officialDocumentIds.has(item.docId));

          return (
            <article key={result.taskId} className="rounded-3xl border border-white/10 bg-white/5 p-6">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-400">任务 {result.taskId}</p>
                  <h3 className="mt-3 text-2xl font-semibold text-white">
                    {result.qid} <span className="text-cyan-200">{"=>"} {result.answer}</span>
                  </h3>
                </div>
                <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-50">
                  Tokens: {result.tokenUsage.totalTokens}
                </div>
              </div>

              <div className="mt-6 grid gap-4 xl:grid-cols-[1.3fr_1fr]">
                <div className="space-y-3">
                  {visibleEvidence.map((item, index) => (
                    <div key={`${item.docId}-${index}`} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <p className="text-sm text-cyan-100">
                        {item.docId} / 第 {item.pageNo} 页 {item.clauseNo ? `/ ${item.clauseNo}` : ""}
                      </p>
                      <p className="mt-3 text-sm leading-6 text-slate-200">{item.quotedText}</p>
                      <p className="mt-3 text-xs leading-5 text-slate-400">{item.reasoning}</p>
                    </div>
                  ))}
                  {visibleEvidence.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 p-4 text-sm text-slate-400">
                      当前结果未命中正式题库证据片段，已在展示层隐藏本地测试文档。
                    </div>
                  ) : null}
                </div>

                <div className="space-y-4">
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <h4 className="text-sm font-medium text-white">Qwen 主链</h4>
                    <div className="mt-4 grid gap-2 text-xs text-slate-300">
                      <div className="rounded-xl bg-white/5 px-3 py-2">启用状态: {result.llmTrace.enabled ? "已启用" : "未启用"}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">是否尝试调用: {result.llmTrace.attempted ? "是" : "否"}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">实际执行链路: {result.llmTrace.used}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">请求模型: {result.llmTrace.requestedModel}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">实际模型: {result.llmTrace.actualModel}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">是否 fallback: {result.llmTrace.fallback ? "是" : "否"}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">fallback 原因: {result.llmTrace.fallbackReason}</div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">
                        Tokens: {result.llmTrace.promptTokens} / {result.llmTrace.completionTokens} / {result.llmTrace.totalTokens}
                      </div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">
                        缺失配置: {result.llmTrace.missingSettings.length ? result.llmTrace.missingSettings.join(", ") : "none"}
                      </div>
                      <div className="rounded-xl bg-white/5 px-3 py-2">错误信息: {result.llmTrace.error || "none"}</div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <h4 className="text-sm font-medium text-white">运行日志</h4>
                    <div className="mt-4 space-y-2 text-xs text-slate-300">
                      {result.logs.map((log) => (
                        <div key={log} className="rounded-xl bg-white/5 px-3 py-2">
                          {log}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </article>
          );
        })}
        {visibleResults.length === 0 ? <p className="text-sm text-slate-400">尚无正式题库相关结果，请先运行正式数据集任务。</p> : null}
      </section>
    </AppShell>
  );
}
