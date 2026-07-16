import { useEffect, useState } from "react";

import AppShell from "@/components/layout/AppShell";
import { api } from "@/utils/api";

export default function QualityPage() {
  const [quality, setQuality] = useState<Record<string, string>>({});

  useEffect(() => {
    void api.getQuality().then(setQuality);
  }, []);

  return (
    <AppShell title="质量面板" description="集中展示代码规范、测试覆盖率、CI 质量门禁与后续联调验收目标。">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {Object.entries(quality).map(([key, value]) => (
          <article key={key} className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{key}</p>
            <p className="mt-4 text-2xl font-semibold text-white">{value}</p>
          </article>
        ))}
      </section>

      <section className="mt-6 rounded-3xl border border-white/10 bg-slate-900/60 p-6">
        <h3 className="text-lg font-semibold text-white">质量执行标准</h3>
        <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-300">
          <li>核心逻辑测试覆盖率不低于 80%。</li>
          <li>所有前端改动均需通过 ESLint、Prettier、TypeScript 检查。</li>
          <li>所有后端改动均需通过 Ruff、mypy、pytest。</li>
          <li>联调阶段需验证接口兼容性、数据流转正确性与端到端完整性。</li>
        </ul>
      </section>
    </AppShell>
  );
}
