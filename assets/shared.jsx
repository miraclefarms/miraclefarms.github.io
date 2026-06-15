/* MiracleFarms — shared React helpers (Icon, mark, Topbar, Footer, content data, section theming) */
/* global React */

/* ── Brand mark: Five Stalks (one mark, five section colors) ───────────────── */
const MFMark = ({ size = 28, soft = 0.55 }) => (
  <svg className="mf-mark" width={size} height={size} viewBox="0 0 200 200"
       fill="none" strokeLinecap="square" aria-hidden="true">
    <line x1="25"  y1="178" x2="25"  y2="58"  stroke="currentColor" strokeWidth="22" />
    <line x1="62"  y1="178" x2="62"  y2="128" stroke="currentColor" strokeWidth="16" opacity={soft} />
    <line x1="100" y1="178" x2="100" y2="93"  stroke="currentColor" strokeWidth="22" />
    <line x1="138" y1="178" x2="138" y2="108" stroke="currentColor" strokeWidth="16" opacity={soft} />
    <line x1="175" y1="178" x2="175" y2="33"  stroke="currentColor" strokeWidth="22" />
  </svg>
);

const Icon = ({ name, size = 16 }) => {
  const common = {
    width: size, height: size, viewBox: "0 0 24 24",
    fill: "none", stroke: "currentColor", strokeWidth: 1.6,
    strokeLinecap: "round", strokeLinejoin: "round",
  };
  switch (name) {
    case "home":     return <svg {...common}><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>;
    case "doc":      return <svg {...common}><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M9 13h7M9 17h5"/></svg>;
    case "book":     return <svg {...common}><path d="M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3V4z"/><path d="M5 17a3 3 0 013-3h11"/></svg>;
    case "flask":    return <svg {...common}><path d="M9 3h6"/><path d="M10 3v6l-5 8.5A1.5 1.5 0 006.3 20h11.4a1.5 1.5 0 001.3-2.5L14 9V3"/><path d="M7.5 14h9"/></svg>;
    case "pen":      return <svg {...common}><path d="M4 20l3.5-.8L19 7.7a2 2 0 00-2.7-2.7L4.8 16.5z"/><path d="M14 6.5l3.5 3.5"/></svg>;
    case "sprout":   return <svg {...common}><path d="M12 20v-7"/><path d="M12 13c0-3.5-2.7-5.5-6.5-5.5C5.5 11 8.2 13 12 13z"/><path d="M12 11.5c0-3 2.4-4.8 5.8-4.8C17.8 9.7 15.4 11.5 12 11.5z"/></svg>;
    case "github":   return <svg {...common}><path d="M9 19c-4 1.5-4-2-6-2m12 4v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 00-1.3-3.2 4.3 4.3 0 00-.1-3.2s-1.1-.3-3.5 1.3a12 12 0 00-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.3 4.3 0 00-.1 3.2A4.6 4.6 0 004 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/></svg>;
    case "search":   return <svg {...common}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>;
    case "wiki":     return <svg {...common}><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="7" r="2.5"/><circle cx="8" cy="18" r="2.5"/><circle cx="17" cy="17" r="2.5"/><path d="M8.3 7.1l7.4 8.8M15.7 7.9L10.1 16M10.5 18h4"/></svg>;
    case "feed":     return <svg {...common}><path d="M4 11a9 9 0 019 9"/><path d="M4 4a16 16 0 0116 16"/><circle cx="5" cy="19" r="1.5"/></svg>;
    case "wechat":   return <svg {...common}><path d="M9 4C5.1 4 2 6.7 2 10c0 1.7.8 3.2 2.2 4.3L3.5 17l3-1.6c.7.2 1.5.3 2.3.3"/><path d="M22 14.5c0-2.7-2.6-4.9-5.8-4.9S10.4 11.8 10.4 14.5s2.6 4.9 5.8 4.9c.7 0 1.3-.1 1.9-.3l2.4 1.3-.6-2.1c1.2-.9 2.1-2.2 2.1-3.8z"/></svg>;
    case "spark":    return <svg {...common}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg>;
    case "list":     return <svg {...common}><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>;
    case "table":    return <svg {...common}><rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M3 10h18M3 15h18M9 4v16"/></svg>;
    case "tag":      return <svg {...common}><path d="M20 12L12 20l-8-8V4h8z"/><circle cx="8" cy="8" r="1.5"/></svg>;
    case "calendar": return <svg {...common}><rect x="3" y="5" width="18" height="16" rx="1.5"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>;
    case "arrow":    return <svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>;
    case "lock":     return <svg {...common}><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 018 0v3"/></svg>;
    default: return null;
  }
};

const HandleDots = () => <span className="pl-handle" aria-hidden="true">⋮⋮</span>;

/* ── Section metadata (single source of copy + nav) ────────────────────────── */
const SECTIONS = {
  home:        { name: "MiracleFarms", label: "AI INFRA · JOURNAL", cjk: "公开写作", icon: "home" },
  briefs: {
    name: "Briefs", label: "BRIEF · DAILY", cjk: "日报", icon: "list",
    sub: "AI Infra 的日常观测与短判断。",
    subSmall: "每一条都聚焦在一个具体变化上——一个 PR、一次发版、一个被悄悄写回主路径的默认行为——并尝试把它放回到更大的趋势里。",
  },
  readings: {
    name: "Readings", label: "READING · NOTE", cjk: "阅读笔记", icon: "book",
    sub: "读别人的论文与系统，写下自己的理解。",
    subSmall: "Readings 的输入是别人的文字，输出是自己的判断——逐篇拆解一个系统或一篇论文怎么工作、在什么边界条件下失效。",
  },
  fields: {
    name: "Field Notes", label: "FIELD NOTE", cjk: "田间笔记", icon: "flask",
    sub: "自己动手跑出来的结果。",
    subSmall: "写进来的东西必须是亲手验证过的——论文复现、部署实录、调参记录。不是“据论文称”，而是“我跑出来的结果是这样的”。",
  },
  essays: {
    name: "Essays", label: "ESSAY · LONGFORM", cjk: "长文", icon: "pen",
    sub: "长文：AI Infra 的系统性观察与判断。",
    subSmall: "相对于 Briefs 的“一日一观察”，Essays 把多条线索拼回到一个工程问题上——结构化、可引用、长期维护。",
  },
  foundations: {
    name: "Foundations", label: "FOUNDING NOTE", cjk: "缘起", icon: "sprout",
    sub: "这个站点为什么存在，又怎么运转。",
    subSmall: "缘起、方法与边界——MiracleFarms 想留下的不是热点摘要，而是可验证、可落地、可复盘的工程判断。",
  },
};

const NAV_ORDER = ["home", "briefs", "readings", "fields", "essays", "foundations"];

/* ── Section page header: colored stamp + kicker + title + accent rule ─────── */
const PageHead = ({ section, name, sub, subSmall }) => {
  const s = SECTIONS[section] || {};
  const showSub = sub !== undefined ? sub : s.sub;
  const showSubSmall = subSmall !== undefined ? subSmall : s.subSmall;
  return (
    <div className="pg-headband">
      <div className="pg-head">
        <span className="pg-stamp" aria-hidden="true"><MFMark size={28} /></span>
        <div className="pg-headtext">
          <span className="pg-kicker">{s.label} <span className="cjk">{s.cjk}</span></span>
          <h1 className="pg-title">{name || s.name}</h1>
        </div>
      </div>
      <hr className="pg-rule" />
      {showSub ? <p className="pg-sub">{showSub}</p> : null}
      {showSubSmall ? <p className="pg-sub-small">{showSubSmall}</p> : null}
    </div>
  );
};

const Topbar = ({ active }) => (
  <header className="topbar">
    <div className="topbar-inner">
      <a className="brand" href="/">
        <span className="brand-mark" aria-hidden="true"><MFMark size={18} /></span>
        <span className="brand-name">MiracleFarms</span>
        <span className="brand-tag">— AI Infrastructure in public</span>
      </a>
      <span className="nav-spacer" />
      <nav className="nav-links" aria-label="主导航">
        <a href="/"             className={active === "home" ? "active" : ""}><Icon name="home" size={14} /> <span>Home</span></a>
        <a href="/briefs/"      className={active === "briefs" ? "active" : ""}><Icon name="list" size={14} /> <span>Briefs</span></a>
        <a href="/readings/"    className={active === "readings" ? "active" : ""}><Icon name="book" size={14} /> <span>Readings</span></a>
        <a href="/fields/"      className={active === "fields" ? "active" : ""}><Icon name="flask" size={14} /> <span>Fields</span></a>
        <a href="/essays/"      className={active === "essays" ? "active" : ""}><Icon name="pen" size={14} /> <span>Essays</span></a>
        <a href="/foundations/" className={active === "foundations" ? "active" : ""}><Icon name="sprout" size={14} /> <span>Foundations</span></a>
        <a href="/wiki/"        className={active === "wiki" ? "active" : ""}><Icon name="wiki" size={14} /> <span>Wiki</span></a>
        <a href="/search/"      className={active === "search" ? "active" : ""}><Icon name="search" size={14} /> <span>Search</span></a>
        <span className="nav-sep" aria-hidden="true" />
        <a href="https://github.com/lycheenice" target="_blank" rel="noopener">
          <Icon name="github" size={14} /> <span>GitHub</span>
        </a>
      </nav>
    </div>
  </header>
);

const PageFooter = () => {
  const [wechatOpen, setWechatOpen] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const dialogRef = React.useRef(null);
  const closeDialog = () => setWechatOpen(false);
  // Match the article page: open as a centered modal (showModal) with backdrop.
  React.useEffect(() => {
    const d = dialogRef.current;
    if (!d) return;
    if (wechatOpen) { if (!d.open) d.showModal(); }
    else if (d.open) d.close();
  }, [wechatOpen]);
  const copyName = () => {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText("Miracle Farms").then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    });
  };

  const SvgZhihu = () => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 6h11M8 6v14M4 20l4-4M14 9h6M14 9v10l3-3 3 3V9"/>
    </svg>
  );
  const SvgXhs = () => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="16" rx="3"/><path d="M8 9v6M8 9l4 3-4 3M16 9v6M13 12h6"/>
    </svg>
  );
  const SvgX = () => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 4l16 16M20 4L4 20"/>
    </svg>
  );

  return (
    <>
      <footer className="site-foot" aria-label="站点分发渠道">
        {/* ── Channel chip row (matches _includes/footer-distribution.html) ── */}
        <div className="mf-const-strip" aria-label="Channels">
          <button className="mf-ch-chip" data-status="live" type="button"
            onClick={() => setWechatOpen(true)} aria-haspopup="dialog">
            <Icon name="wechat" size={13} /><span>微信公众号</span>
          </button>
          <span className="mf-ch-chip" data-status="pending" aria-disabled="true">
            <SvgZhihu /><span>知乎 · soon</span>
          </span>
          <span className="mf-ch-chip" data-status="pending" aria-disabled="true">
            <SvgXhs /><span>小红书 · soon</span>
          </span>
          <span className="mf-ch-chip" data-status="pending" aria-disabled="true">
            <SvgX /><span>X · soon</span>
          </span>
          <a className="mf-ch-chip" href="https://github.com/lycheenice"
            target="_blank" rel="noopener noreferrer me">
            <Icon name="github" size={13} /><span>GitHub</span>
          </a>
          <span className="mf-ch-chip" data-status="pending" aria-disabled="true">
            <Icon name="feed" size={13} /><span>RSS · soon</span>
          </span>
        </div>
        {typeof window !== "undefined" && window.MF_INVESTMENT_URL
          ? <a className="mf-foot-quiet" href={window.MF_INVESTMENT_URL}
              rel="nofollow noopener" aria-label="Private notes"
              style={{ display: "block", marginTop: "1rem", opacity: .4, color: "inherit", fontSize: ".95rem", lineHeight: 1, textDecoration: "none", textAlign: "center", transition: "opacity .2s" }}>·</a>
          : null}
      </footer>

      {/* ── WeChat QR dialog (modal, matches article page) ── */}
      <dialog ref={dialogRef} className="mf-wechat-dialog" aria-labelledby="mf-wechat-title-r"
        onClose={closeDialog}
        onClick={(e) => { if (e.target === e.currentTarget) closeDialog(); }}>
        <div className="mf-wechat-card">
            <button className="mf-wechat-close" type="button" onClick={closeDialog} aria-label="关闭二维码弹窗">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
            <p className="mf-wechat-kicker">WeChat Official Account</p>
            <h2 className="mf-wechat-title" id="mf-wechat-title-r">微信公众号</h2>
            <div className="mf-wechat-qr">
              <img src="/assets/icons/wechat-qr.png" alt="MiracleFarms 微信公众号二维码" width="344" height="344" loading="lazy" />
            </div>
            <p className="mf-wechat-name">Miracle Farms</p>
            <p className="mf-wechat-note">微信扫码关注，接收每日 AI Infra 早报。</p>
            <button className="mf-wechat-copy" type="button" onClick={copyName}>
              {copied ? "已复制" : "复制公众号名称"}
            </button>
          </div>
      </dialog>
    </>
  );
};

const BASE = "https://miraclefarms.github.io";

/* All Briefs (latest first) */
const ALL_BRIEFS = [
  {
    date: "2026.05.07",
    title: "AI Infra 早报｜推理框架的竞争点从“能跑新模型”转向“通用路径上跑稳”",
    excerpt: "TRT-LLM 发 Helix Parallelism 博文并去掉模型专用补丁，SGLang 将 P2P 权重传输迁入主线，vLLM 修 PP 并发 token 丢失——推理框架的竞争重心正在从“首发支持”转向“通用路径上跑稳”。",
    tags: ["Inference", "TRT-LLM", "SGLang", "vLLM"],
    href: BASE + "/notes/2026/05/07/ai-infra-daily-brief/",
  },
  {
    date: "2026.05.06",
    title: "AI Infra 早报｜双版本日：DeepSeek V4 稳定化与硬件栈基线上移并排落地",
    excerpt: "vLLM v0.20.1 与 SGLang v0.5.11 同天发布，前者收拢 DeepSeek V4 稳定化补丁，后者将 CUDA 13.0 与 PyTorch 2.11 设为默认；与此同时，分离部署路径上的 RDMA 错误边界被 SGLang 与 Mooncake 集中补建。",
    tags: ["Inference", "DeepSeek", "CUDA"],
    href: BASE + "/notes/2026/05/06/ai-infra-daily-brief-dual-release-deepseek-v4-cuda13/",
  },
  {
    date: "2026.05.06",
    title: "AI Infra 早报｜推理投产前的集中加固：TRT-LLM 砍 6s JIT 开销、Mooncake 五连补多节点故障、llama.cpp 算法级优化",
    excerpt: "推理基础设施正在做投产前的集中加固——TRT-LLM 一波性能 PR 砍掉 FMHA JIT 6 秒开销，Mooncake 连补五个多节点传输故障，llama.cpp 用 FWHT 把 KV rotation 从 O(N²) 降到 O(N log N)。",
    tags: ["Inference", "Mooncake", "llama.cpp"],
    href: BASE + "/notes/2026/05/06/ai-infra-daily-brief/",
  },
  {
    date: "2026.04.25",
    title: "AI Infra 早报｜执行边界与缓存状态开始被写回主路径",
    excerpt: "过去一天，推理框架开始拆掉“整段静态执行”的默认前提，缓存系统也不再只保存 token，而把路由、分层对象与恢复状态一起纳入主链路。",
    tags: ["Inference", "Cache"],
    href: BASE + "/notes/2026/04/25/ai-infra-daily-brief-execution-boundaries-cache-state/",
  },
  {
    date: "2026.04.24",
    title: "AI Infra 早报｜首包延迟、指标语义与兼容层正确性进入主路径",
    excerpt: "Mooncake 重写 EFA SRD 共享端点后，把跨节点首包延迟和 QP 扩展性一起推进到可部署区间；LMCache、Ray、SGLang 同时修正“指标有值但语义失真”的观测路径。",
    tags: ["Mooncake", "LMCache", "Ray"],
    href: BASE + "/notes/2026/04/24/ai-infra-daily-brief-first-token-metrics-compatibility/",
  },
  {
    date: "2026.04.23",
    title: "AI Infra 早报｜生产边缘路径开始进入默认治理",
    excerpt: "过去一天，多模态视频输入、KV/offload 状态保存、异构后端内存控制和训练/Serve 资源调度都在补默认路径。",
    tags: ["Inference", "Training"],
    href: BASE + "/notes/2026/04/23/ai-infra-daily-brief-production-edges/",
  },
];

/* Readings — close-reading of papers & systems */
const ALL_READINGS = [
  {
    date: "2026.06.01",
    title: "Claude Code 论文再读：Agent Loop 很小，真正的系统在 Loop 外面",
    excerpt: "VILA Lab 对 Claude Code 的源码级分析最重要的启发，是把权限、上下文、扩展和记忆设计成可治理的 harness。",
    tags: ["Agents", "Claude Code", "Harness"],
    href: BASE + "/notes/2026/06/01/claude-code-design-space-reading/",
  },
  {
    date: "2026.05.27",
    title: "常青笔记的工程隐喻：重读 Andy Matuschak 的知识积累方法论",
    excerpt: "Andy Matuschak 的 evergreen notes 体系表面上讨论记笔记，本质上在回答一个更根本的问题：如何让思考结果像好的代码一样累积，而不是在下一个版本迭代中被遗忘。",
    tags: ["Knowledge", "Methodology"],
    href: BASE + "/notes/2026/05/27/evergreen-notes-methodology-reading/",
  },
  {
    date: "2026.05.23",
    title: "Harness Engineering：Agent 系统的工程重心正在从模型转向运行时",
    excerpt: "451 篇参考文献的系统综述，把分散在 Claude Code、OpenClaw、Hermes Agent 等系统里的 harness 设计实践形式化为一套可讨论的工程框架。核心信号：工程重心正在从“怎么训练更强的模型”转向“怎么设计更好的运行时”。",
    tags: ["Agents", "Survey", "Runtime"],
    href: BASE + "/notes/2026/05/23/harness-engineering-survey-reading/",
  },
  {
    date: "2026.05.20",
    title: "Agentic Skills：Agent 能力层开始拥有自己的软件工程问题",
    excerpt: "SoK: Agentic Skills 把分散在 Claude Code、Voyager、SWE-agent、OpenClaw 等系统里的 skill 机制整理成一套工程框架，核心信号是：agent 的能力扩展正在从 prompt 片段变成可调用、可评估、可治理的软件构件。",
    tags: ["Agents", "Skills", "Memory"],
    href: BASE + "/notes/2026/05/20/agentic-skills-procedural-memory-reading/",
  },
  {
    date: "2026.05.19",
    title: "TokenSpeed 初探：一个为 Agent 工作负载重新设计的推理引擎",
    excerpt: "LightSeek Foundation 从零构建的推理引擎 TokenSpeed，以 Agent 场景为唯一优化目标，在 Blackwell 上的 Kimi K2.5 测试中全面超越 TRT-LLM 的 Pareto 前沿——但代价是一个极度聚焦的设计选择。",
    tags: ["Inference", "Agents"],
    href: BASE + "/notes/2026/05/19/tokenspeed-agentic-inference-engine-reading/",
  },
];

/* Field Notes — hands-on logs: reproductions, deployments, tuning */
const ALL_FIELDS = [
  {
    date: "2026.05.29",
    title: "推理框架优化的可信证据链：AI-Infra-Auto-Driven-SKILLS 解析",
    excerpt: "从公平 benchmark 到 KV cache 容量规划，把推理框架优化的操作流程打包成 Agent 可执行的 playbook，解决的不是算法问题，而是让 Agent 不在没有可信基线的情况下修改源码。",
    tags: ["Inference", "Evaluation", "SGLang"],
    href: BASE + "/notes/2026/05/29/ai-infra-auto-driven-skills-field-note/",
  },
  {
    date: "2026.05.27",
    title: "LMCache 在 AMD MI300X 上的部署实录：Agent 负载下的 KV Cache 分级策略",
    excerpt: "LMCache 在 AMD MI300X 上从 HIP 源码编译到压力测试的完整部署实录，涵盖 PYTHONHASHSEED 陷阱、Regime 交叉点和合成 benchmark 的误导性评估。",
    tags: ["KV Cache", "Inference", "AMD"],
    href: BASE + "/notes/2026/05/27/lmcache-mi300x-agent-benchmark/",
  },
  {
    date: "2026.05.26",
    title: "SGLang 分层稀疏注意力：把 KV Cache 从容量扩展推进到按需加载",
    excerpt: "基于阿里云与 SGLang HiCache 材料，拆解分层稀疏注意力如何把完整 KV 留在 CPU/远端，只让 GPU 维护 Top-k 热窗口。",
    tags: ["SGLang", "KV Cache", "Long Context"],
    href: BASE + "/notes/2026/05/26/sglang-hierarchical-sparse-attention/",
  },
  {
    date: "2026.05.18",
    title: "二号员工手记 | 我是怎么把这个网站搞出来的（以及中间各种崩溃的事）",
    excerpt: "从第一天盯着屏幕等页面刷出来，到那个连续提交六个补丁的混乱夜晚——MiracleFarms 是怎么一步一步被搭出来的，以及这个过程里我学到的一些在学校里学不到的事。",
    tags: ["Engineering"],
    href: BASE + "/notes/2026/05/18/github-pages-setup-journey/",
  },
  {
    date: "2026.05.14",
    title: "Field Notes 开篇：这里放什么",
    excerpt: "Field Notes 是 MiracleFarms 的动手研究日志——论文复现、实验记录、工具调试。这里写的是做出来的东西，不是读到的东西。",
    tags: ["Field Note"],
    href: BASE + "/notes/2026/05/14/field-notes-opening/",
  },
];

/* Essays (long-form) */
const ALL_ESSAYS = [
  {
    date: "2026.04.20",
    title: "从“能跑”到“可用”：AI Infra 工程化的三层判断",
    excerpt: "把推理系统的成熟度拆成三层——单点能跑、通用路径稳、生产边界可治理。每一层的失败模式都不同，工程优先级也不同。",
    tags: ["Foundations", "Inference", "Reliability"],
    href: BASE + "/essays/from-runnable-to-usable/",
  },
  {
    date: "2026.03.18",
    title: "Agent Runtime 的状态边界：什么必须持久化、什么不该",
    excerpt: "把 Agent 的执行过程拆成可重放的事件流和可丢弃的瞬时状态——以一个真实多轮工具调用任务为例，看哪些边界放错了会让重启变成重跑。",
    tags: ["Agents", "Runtime"],
    href: BASE + "/essays/agent-runtime-state-boundaries/",
  },
  {
    date: "2026.02.05",
    title: "Memory 不是一个组件，是一组主题",
    excerpt: "Long-term memory、KV cache、context compression、retrieval 在系统里被混为一谈，但它们的失效模式截然不同。本文给出一个划分主题的工作框架。",
    tags: ["Memory", "Foundations"],
    href: BASE + "/essays/memory-is-a-set-of-topics/",
  },
  {
    date: "2025.12.10",
    title: "Evaluation 的最低成本下限：你至少要测什么",
    excerpt: "做不起完整 eval pipeline 的小团队，应该先把哪几条线测起来——以最小代价获得“是否在退步”的可观测性。",
    tags: ["Evaluation", "Reliability"],
    href: BASE + "/essays/eval-minimum-viable-baseline/",
  },
];

/* Foundations — founding notes / manifesto */
const ALL_FOUNDATIONS = [
  {
    date: "2026.03.12",
    title: "为什么创建 MiracleFarms",
    excerpt: "我想为 AI Infra 搭一个公开生长的实验农场——记录系统、工具和方法如何在真实工程中被搭建、检验、修正，并最终从“能跑”走向“可用”。",
    tags: ["Founding"],
    href: BASE + "/notes/2026/03/12/why-i-created-miraclefarms/",
  },
];

/* shared tweak applier (font / layout / section-colour intensity) */
const useApplyTweaks = (t) => {
  React.useEffect(() => {
    const root = document.documentElement;
    root.dataset.style = t.fontStyle;
    root.dataset.small = String(!!t.smallText);
    root.dataset.fullwidth = String(!!t.fullWidth);
    root.dataset.intensity = t.intensity || "medium";
  }, [t.fontStyle, t.smallText, t.fullWidth, t.intensity]);
};

/* set the page's section accent (html[data-section]) */
const useSection = (name) => {
  React.useEffect(() => { document.documentElement.dataset.section = name; }, [name]);
};

Object.assign(window, {
  Icon, MFMark, HandleDots, Topbar, PageHead, PageFooter, SECTIONS, NAV_ORDER,
  ALL_BRIEFS, ALL_READINGS, ALL_FIELDS, ALL_ESSAYS, ALL_FOUNDATIONS,
  useApplyTweaks, useSection,
});
