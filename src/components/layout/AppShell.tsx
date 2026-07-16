import { Link, useLocation } from "react-router-dom";
import { FileText, FolderKanban, Gauge, PackageCheck, ScrollText, ShieldCheck } from "lucide-react";
import { useEffect, type ReactNode } from "react";

const navItems = [
  { to: "/", label: "控制台", icon: Gauge },
  { to: "/documents", label: "文档资产", icon: FolderKanban },
  { to: "/tasks", label: "题目执行", icon: ScrollText },
  { to: "/results", label: "结果中心", icon: FileText },
  { to: "/quality", label: "质量面板", icon: ShieldCheck },
  { to: "/delivery", label: "交付中心", icon: PackageCheck },
];

type AppShellProps = {
  title: string;
  description: string;
  children: ReactNode;
};

export default function AppShell({ title, description, children }: AppShellProps) {
  const location = useLocation();

  useEffect(() => {
    document.title = `${title} | 金融长文档智能阅读理解系统`;
  }, [title]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-[1600px] gap-6 px-6 py-6">
        <aside className="hidden w-72 flex-col rounded-[28px] border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-cyan-950/20 lg:flex">
          <div className="mb-8 space-y-3">
            <span className="inline-flex rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs tracking-[0.3em] text-cyan-200">
              FINANCE AGENT MVP
            </span>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-white">金融长文档智能阅读理解系统</h1>
              <p className="mt-2 text-sm leading-6 text-slate-400">围绕检索、证据、推理、交付的竞赛化控制台。</p>
            </div>
          </div>

          <nav className="space-y-2">
            {navItems.map(({ to, label, icon: Icon }) => {
              const active = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition ${
                    active
                      ? "bg-cyan-400/15 text-white shadow-lg shadow-cyan-900/30"
                      : "text-slate-400 hover:bg-white/5 hover:text-white"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="mt-auto rounded-3xl border border-white/10 bg-slate-950/70 p-4">
            <p className="text-xs uppercase tracking-[0.25em] text-slate-500">执行准则</p>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li>离线重预处理，在线轻上下文</li>
              <li>规则召回打底，Qwen 负责精筛与推理</li>
              <li>答案必须附着证据与 Token 统计</li>
            </ul>
          </div>
        </aside>

        <main className="flex-1 rounded-[32px] border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.12),_transparent_30%),linear-gradient(180deg,rgba(15,23,42,0.96),rgba(2,6,23,0.96))] p-6 shadow-2xl shadow-slate-950/40">
          <header className="mb-6 flex flex-col gap-3 border-b border-white/10 pb-6">
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/70">MVP 控制台</p>
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-white">{title}</h2>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">{description}</p>
              </div>
              <div className="flex gap-3 text-xs text-slate-400">
                <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-emerald-200">Git Flow</span>
                <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-2 text-cyan-100">CI 准备完成</span>
              </div>
            </div>
          </header>

          {children}
        </main>
      </div>
    </div>
  );
}
