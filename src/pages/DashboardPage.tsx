import { useEffect } from "react";
import { Activity, Database, FileCheck2, Package } from "lucide-react";

import AppShell from "@/components/layout/AppShell";
import { useAppStore } from "@/store/useAppStore";

const cards = [
  { key: "documents", label: "文档总量", icon: Database },
  { key: "processedDocuments", label: "已预处理文档", icon: FileCheck2 },
  { key: "tasks", label: "执行任务数", icon: Activity },
  { key: "results", label: "结果条目数", icon: Package },
] as const;

export default function DashboardPage() {
  const { dashboard, refreshDashboard } = useAppStore();

  useEffect(() => {
    void refreshDashboard();
  }, [refreshDashboard]);

  return (
    <AppShell title="项目控制台" description="查看环境概况、任务负载与当前 MVP 是否具备交付条件。">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map(({ key, label, icon: Icon }) => (
          <article key={key} className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">{label}</span>
              <Icon className="h-4 w-4 text-cyan-200" />
            </div>
            <p className="mt-6 text-4xl font-semibold text-white">{dashboard?.[key] ?? "--"}</p>
          </article>
        ))}
      </section>

      <section className="mt-6 grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <article className="rounded-3xl border border-white/10 bg-slate-900/60 p-6">
          <h3 className="text-lg font-semibold text-white">全流程开发状态</h3>
          <div className="mt-5 grid gap-3">
            {[
              "仓库初始化与 develop 分支已建立",
              "前后端工程骨架已生成",
              "MVP API、页面路由、质量门禁正在落地",
              "下一步进入联调、测试与部署文档整理",
            ].map((item) => (
              <div key={item} className="rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm text-slate-200">
                {item}
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-100/70">交付判断</p>
          <p className="mt-4 text-2xl font-semibold text-white">{dashboard?.deliveryReady ? "可生成提交产物" : "待补齐交付文件"}</p>
          <p className="mt-3 text-sm leading-6 text-cyan-50/80">
            当前控制台会持续汇总 `answer.csv`、`evidence.json`、日志与文档状态，便于后续进入联调和预部署阶段。
          </p>
        </article>
      </section>
    </AppShell>
  );
}
