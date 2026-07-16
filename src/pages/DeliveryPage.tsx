import { useEffect, useState } from "react";

import AppShell from "@/components/layout/AppShell";
import { api } from "@/utils/api";

type Artifact = {
  name: string;
  description: string;
  path: string;
};

export default function DeliveryPage() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  useEffect(() => {
    void api.getDelivery().then(setArtifacts);
  }, []);

  return (
    <AppShell title="交付中心" description="查看 MVP 当前已生成的文档、部署手册、交付报告和后续需要打包的正式产物。">
      <section className="grid gap-4 xl:grid-cols-2">
        {artifacts.map((artifact) => (
          <article key={artifact.name} className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{artifact.name}</p>
            <h3 className="mt-3 text-lg font-semibold text-white">{artifact.description}</h3>
            <p className="mt-3 text-sm text-slate-400">{artifact.path}</p>
          </article>
        ))}
      </section>

      <section className="mt-6 rounded-3xl border border-emerald-400/20 bg-emerald-400/10 p-6">
        <h3 className="text-lg font-semibold text-white">最终交付清单</h3>
        <p className="mt-3 text-sm leading-6 text-emerald-50/85">
          当前项目会产出 PRD、技术架构文档、可运行前后端工程、测试脚本、部署手册、开发过程文档、偏差说明与交付报告，为后续 MVP 验收做准备。
        </p>
      </section>
    </AppShell>
  );
}
