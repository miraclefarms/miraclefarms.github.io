/* MiracleFarms — Fields index App (source). Compiled to /assets/js/pages/fields.js
   by _ci/build-jsx.sh. Reads live data from window.ALL_FIELDS (injected per page). */
/* global React, ReactDOM, TweaksPanel, useTweaks, TweakSection, TweakRadio, TweakToggle,
          Icon, MFMark, HandleDots, Topbar, PageHead, PageFooter, ALL_FIELDS, useApplyTweaks, useSection */

const { useState, useMemo } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "fontStyle": "serif",
  "smallText": false,
  "fullWidth": false,
  "intensity": "medium"
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  useSection("fields");
  useApplyTweaks(t);

  const [activeTag, setActiveTag] = useState(null);

  const tagCounts = useMemo(() => {
    const m = new Map();
    for (const f of ALL_FIELDS) for (const tag of f.tags) m.set(tag, (m.get(tag) || 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const filtered = useMemo(
    () => activeTag ? ALL_FIELDS.filter(f => f.tags.includes(activeTag)) : ALL_FIELDS,
    [activeTag]
  );

  const byYear = useMemo(() => {
    const groups = {};
    for (const f of filtered) {
      const y = f.date.slice(0, 4);
      (groups[y] = groups[y] || []).push(f);
    }
    return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
  }, [filtered]);

  return (
    <>
      <Topbar active="fields" />
      <main className="page">
        <div className="pg-meta">
          <a href="/">MiracleFarms</a>
          <span className="dot">›</span>
          <span>Field Notes</span>
        </div>

        <PageHead section="fields" />

        <aside className="callout">
          <span className="ico" aria-hidden="true">🧪</span>
          <div>
            <p>
              共 <strong>{ALL_FIELDS.length}</strong> 篇 · 核心约束只有一条：写进来的东西必须是<strong>亲手验证过的</strong>。不是“据论文称”，不是“理论上应该”，而是“我跑出来的结果是这样的”。
            </p>
          </div>
        </aside>

        <div className="filterbar" role="toolbar" aria-label="筛选">
          <span className="lbl">Filter:</span>
          <button
            className={"fchip" + (activeTag === null ? " on" : "")}
            onClick={() => setActiveTag(null)}
            type="button"
          >全部 <span className="count">{ALL_FIELDS.length}</span></button>
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
          <div className="empty">没有匹配的 field note。</div>
        ) : (
          byYear.map(([yr, items]) => (
            <section key={yr}>
              <div className="year-row">
                <span className="yr">{yr}</span>
                <span className="ln" />
                <span>{items.length} 篇</span>
              </div>
              <div className="pagelist" role="list">
                {items.map((f, i) => (
                  <a key={i} className="pagelink" href={f.href} role="listitem">
                    <HandleDots />
                    <span className="pl-ico" aria-hidden="true"><Icon name="flask" size={16} /></span>
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
            </section>
          ))
        )}

        <hr className="divider" />

        <h6 className="h2-sub">What goes here</h6>
        <h2 className="h2">这里会放什么</h2>
        <p className="muted">
          论文复现、部署实录、调参记录、踩坑笔记——核心是“可复现”，每一条结论都尽量带上环境、版本与命令。Fields 的更新不追快，只追<strong>能被别人照着跑一遍</strong>。
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
