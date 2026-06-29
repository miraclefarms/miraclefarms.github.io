/* MiracleFarms — Foundations index App (source). Compiled to /assets/js/pages/foundations.js
   by _ci/build-jsx.sh. Reads live data from window.ALL_FOUNDATIONS (injected per page). */
/* global React, ReactDOM, TweaksPanel, useTweaks, TweakSection, TweakRadio, TweakToggle,
          Icon, MFMark, HandleDots, Topbar, PageHead, PageFooter, ALL_FOUNDATIONS, useApplyTweaks, useSection */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "fontStyle": "serif",
  "smallText": false,
  "fullWidth": false,
  "intensity": "medium"
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  useSection("foundations");
  useApplyTweaks(t);

  return (
    <>
      <Topbar active="foundations" />
      <main className="page">
        <div className="pg-meta">
          <a href="/">MiracleFarms</a>
          <span className="dot">›</span>
          <span>Foundations</span>
        </div>

        <PageHead section="foundations" />

        <aside className="callout">
          <span className="ico" aria-hidden="true">🌱</span>
          <div>
            <p>
              <strong>Less hype, more systems.</strong>　这里不是新闻聚合页，也不是自动摘要流水线。我们更关心：一个系统为什么这样设计、它在什么边界条件下会失效，以及一项新能力落到工程实践里究竟意味着什么。
            </p>
          </div>
        </aside>

        <h6 className="h2-sub">Sections</h6>
        <h2 className="h2">四个板块，四个问题</h2>
        <p className="muted">
          MiracleFarms 的内容分成四层，每一层回答一个不同的问题，也对应一种颜色。
        </p>
        <dl className="props">
          <dt><Icon name="list" size={13} /> <a href="/briefs/">Briefs</a></dt>
          <dd>今天 AI Infra 生态里发生了什么值得记录的事？</dd>
          <dt><Icon name="book" size={13} /> <a href="/readings/">Readings</a></dt>
          <dd>这篇论文、这个系统是怎么工作的，边界在哪里？</dd>
          <dt><Icon name="flask" size={13} /> <a href="/fields/">Fields</a></dt>
          <dd>我自己跑出来的结果是什么？（亲手验证，不是“据论文称”）</dd>
          <dt><Icon name="pen" size={13} /> <a href="/essays/">Essays</a></dt>
          <dd>这件事从更长的视角看，意味着什么？</dd>
        </dl>

        <hr className="divider" />

        <h6 className="h2-sub">Method</h6>
        <h2 className="h2">方法与边界</h2>
        <p>
          Agent 辅助检索与整理，人负责判断与发布。所有结论以可复现的链接、commit、日期为锚点；不做未经验证的断言，不做营销腔总结。
        </p>
        <p className="muted">
          “Farms” 用复数，是因为这里关注的不是某一个单点成果，而是一组持续生长的系统、工具和主题。我们希望留下的，不是热点摘要，而是可验证、可落地、可复盘的工程判断。
        </p>

        <details className="toggle">
          <summary>为什么叫 MiracleFarms</summary>
          <div className="toggle-body">
            <p>把 AI Infrastructure 当成一片需要长期耕作的农场——播种、观察、修剪、复盘，而不是一次性发版。Miracle 不是指奇迹会自动发生，而是指持续投入之后，系统真的会从“能跑”长成“可用”。</p>
          </div>
        </details>

        <hr className="divider" />

        <h6 className="h2-sub">Founding notes</h6>
        <h2 className="h2">缘起</h2>
        <div className="pagelist" role="list">
          {ALL_FOUNDATIONS.map((f, i) => (
            <a key={i} className="pagelink" href={f.href} role="listitem">
              <HandleDots />
              <span className="pl-ico" aria-hidden="true"><Icon name="sprout" size={16} /></span>
              <div className="pl-body">
                <div className="pl-title">{f.title}{f.locked && <span className="pl-lock" aria-label="需要密码" title="需要密码阅读"><Icon name="lock" size={11} /></span>}{f.hasEn && <span className="pl-lang" title="中英双语" aria-label="提供中英双语版本"><i>中</i><i>EN</i></span>}</div>
                <div className="pl-excerpt">{f.excerpt}</div>
                <div className="pl-meta">
                  {f.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
                </div>
              </div>
              <div className="pl-date">{f.date}</div>
            </a>
          ))}
        </div>

        <dl className="props" style={{ marginTop: 28 }}>
          <dt><Icon name="spark" size={13} /> 作者</dt>
          <dd>MiracleFarms · Lychee &amp; Ethan</dd>
          <dt><Icon name="feed" size={13} /> 更新</dt>
          <dd>持续迭代，长期维护。</dd>
          <dt><Icon name="wechat" size={13} /> 公众号</dt>
          <dd>
            <a href="https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=MzAxNzg4NDc4MQ==" target="_blank" rel="noopener" style={{ textDecoration: "underline", textDecorationColor: "var(--rule)", textUnderlineOffset: "3px" }}>
              MiracleFarms
            </a>
          </dd>
        </dl>

      </main>
      <PageFooter />

      <TweaksPanel title="Tweaks">
        <TweakSection label="板块色强度 Section colour">
          <TweakRadio
            value={t.intensity}
            onChange={(v) => setTweak("intensity", v)}
            options={[
              { value: "clean",      label: "克制" },
              { value: "medium",     label: "中度" },
              { value: "expressive", label: "明显" },
            ]}
          />
        </TweakSection>
        <TweakSection label="字体 Font style">
          <TweakRadio
            value={t.fontStyle}
            onChange={(v) => setTweak("fontStyle", v)}
            options={[
              { value: "default", label: "Default" },
              { value: "serif",   label: "Serif" },
              { value: "mono",    label: "Mono" },
            ]}
          />
        </TweakSection>
        <TweakSection label="排版 Layout">
          <TweakToggle label="Small text 小字号" value={t.smallText} onChange={(v) => setTweak("smallText", v)} />
          <TweakToggle label="Full width 全宽页面" value={t.fullWidth} onChange={(v) => setTweak("fullWidth", v)} />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
