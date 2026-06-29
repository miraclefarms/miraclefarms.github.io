/* MiracleFarms — Essays index App (source). Compiled to /assets/js/pages/essays.js
   by _ci/build-jsx.sh. Reads live data from window.ALL_ESSAYS (injected per page). */
/* global React, ReactDOM, TweaksPanel, useTweaks, TweakSection, TweakRadio, TweakToggle,
          Icon, MFMark, HandleDots, Topbar, PageHead, PageFooter, ALL_ESSAYS, useApplyTweaks, useSection */

const { useState, useMemo } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "fontStyle": "serif",
  "smallText": false,
  "fullWidth": false,
  "intensity": "medium"
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  useSection("essays");
  useApplyTweaks(t);

  const [activeTag, setActiveTag] = useState(null);

  const tagCounts = useMemo(() => {
    const m = new Map();
    for (const e of ALL_ESSAYS) for (const tag of e.tags) m.set(tag, (m.get(tag) || 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const filtered = useMemo(
    () => activeTag ? ALL_ESSAYS.filter(e => e.tags.includes(activeTag)) : ALL_ESSAYS,
    [activeTag]
  );

  // Group by year
  const byYear = useMemo(() => {
    const groups = {};
    for (const e of filtered) {
      const y = e.date.slice(0, 4);
      (groups[y] = groups[y] || []).push(e);
    }
    return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
  }, [filtered]);

  return (
    <>
      <Topbar active="essays" />
      <main className="page">
        <div className="pg-meta">
          <a href="/">MiracleFarms</a>
          <span className="dot">›</span>
          <span>Essays</span>
        </div>

        <PageHead section="essays" />

        <aside className="callout">
          <span className="ico" aria-hidden="true">📚</span>
          <div>
            <p>
              共 <strong>{ALL_ESSAYS.length}</strong> 篇 · 主题覆盖 <strong>Inference / Agents / Memory / Evaluation / Reliability</strong>。每一篇都尝试在一个具体边界上给出可复盘的判断，而不是综述。
            </p>
          </div>
        </aside>

        {/* Filter bar (no view switcher — essays are read in order) */}
        <div className="filterbar" role="toolbar" aria-label="筛选">
          <span className="lbl">Filter:</span>
          <button
            className={"fchip" + (activeTag === null ? " on" : "")}
            onClick={() => setActiveTag(null)}
            type="button"
          >全部 <span className="count">{ALL_ESSAYS.length}</span></button>
          {tagCounts.map(([tag, n]) => (
            <button
              key={tag}
              className={"fchip" + (activeTag === tag ? " on" : "")}
              onClick={() => setActiveTag(activeTag === tag ? null : tag)}
              type="button"
            >{tag} <span className="count">{n}</span></button>
          ))}
        </div>

        {filtered.length === 0 ? (
          <div className="empty">没有匹配的 essay。</div>
        ) : (
          byYear.map(([yr, items]) => (
            <section key={yr}>
              <div className="year-row">
                <span className="yr">{yr}</span>
                <span className="ln" />
                <span>{items.length} 篇</span>
              </div>
              <div className="pagelist" role="list">
                {items.map((e, i) => (
                  <a key={i} className="pagelink" href={e.href} role="listitem">
                    <HandleDots />
                    <span className="pl-ico" aria-hidden="true"><Icon name="pen" size={16} /></span>
                    <div className="pl-body">
                      <div className="pl-title">{e.title}{e.locked && <span className="pl-lock" aria-label="需要密码" title="需要密码阅读"><Icon name="lock" size={11} /></span>}{e.hasEn && <span className="pl-lang" title="中英双语" aria-label="提供中英双语版本"><i>中</i><i>EN</i></span>}</div>
                      <div className="pl-excerpt">{e.excerpt}</div>
                      <div className="pl-meta">
                        {e.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
                      </div>
                    </div>
                    <div className="pl-date">{e.date}</div>
                  </a>
                ))}
              </div>
            </section>
          ))
        )}

        <hr className="divider" />

        <h6 className="h2-sub">Reading note</h6>
        <h2 className="h2">关于 Essays 的写作</h2>
        <p className="muted">
          Essays 的更新频率明显低于 Briefs——它们通常需要数周到数月的观察沉淀，并经过一次以上的重写。我们倾向于把已经被多个 brief 反复触及的主题，整理成一篇可以长期被引用的 essay。
        </p>

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
