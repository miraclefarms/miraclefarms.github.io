/* MiracleFarms — Readings index App (source). Compiled to /assets/js/pages/readings.js
   by _ci/build-jsx.sh. Reads live data from window.ALL_READINGS (injected per page). */
/* global React, ReactDOM, TweaksPanel, useTweaks, TweakSection, TweakRadio, TweakToggle,
          Icon, MFMark, HandleDots, Topbar, PageHead, PageFooter, ALL_READINGS, useApplyTweaks, useSection */

const { useState, useMemo } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "fontStyle": "serif",
  "smallText": false,
  "fullWidth": false,
  "intensity": "medium"
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  useSection("readings");
  useApplyTweaks(t);

  const [activeTag, setActiveTag] = useState(null);

  const tagCounts = useMemo(() => {
    const m = new Map();
    for (const r of ALL_READINGS) for (const tag of r.tags) m.set(tag, (m.get(tag) || 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const filtered = useMemo(
    () => activeTag ? ALL_READINGS.filter(r => r.tags.includes(activeTag)) : ALL_READINGS,
    [activeTag]
  );

  const byYear = useMemo(() => {
    const groups = {};
    for (const r of filtered) {
      const y = r.date.slice(0, 4);
      (groups[y] = groups[y] || []).push(r);
    }
    return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
  }, [filtered]);

  return (
    <>
      <Topbar active="readings" />
      <main className="page">
        <div className="pg-meta">
          <a href="/">MiracleFarms</a>
          <span className="dot">›</span>
          <span>Readings</span>
        </div>

        <PageHead section="readings" />

        <aside className="callout">
          <span className="ico" aria-hidden="true">📖</span>
          <div>
            <p>
              共 <strong>{ALL_READINGS.length}</strong> 篇 · Reading 的输入是别人的文字，输出是自己的判断。我们倾向于挑那些<strong>能改变工程判断</strong>的论文或系统精读，而不是综述式罗列。
            </p>
          </div>
        </aside>

        <div className="filterbar" role="toolbar" aria-label="筛选">
          <span className="lbl">Filter:</span>
          <button
            className={"fchip" + (activeTag === null ? " on" : "")}
            onClick={() => setActiveTag(null)}
            type="button"
          >全部 <span className="count">{ALL_READINGS.length}</span></button>
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
          <div className="empty">没有匹配的 reading。</div>
        ) : (
          byYear.map(([yr, items]) => (
            <section key={yr}>
              <div className="year-row">
                <span className="yr">{yr}</span>
                <span className="ln" />
                <span>{items.length} 篇</span>
              </div>
              <div className="pagelist" role="list">
                {items.map((r, i) => (
                  <a key={i} className="pagelink" href={r.href} role="listitem">
                    <HandleDots />
                    <span className="pl-ico" aria-hidden="true"><Icon name="book" size={16} /></span>
                    <div className="pl-body">
                      <div className="pl-title">{r.title}{r.locked && <span className="pl-lock" aria-label="需要密码" title="需要密码阅读"><Icon name="lock" size={11} /></span>}{r.hasEn && <span className="pl-lang" title="中英双语" aria-label="提供中英双语版本"><i>中</i><i>EN</i></span>}</div>
                      <div className="pl-excerpt">{r.excerpt}</div>
                      <div className="pl-meta">
                        {r.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
                      </div>
                    </div>
                    <div className="pl-date">{r.date}</div>
                  </a>
                ))}
              </div>
            </section>
          ))
        )}

        <hr className="divider" />

        <h6 className="h2-sub">Reading vs Field</h6>
        <h2 className="h2">Readings 与 Fields 的分工</h2>
        <p className="muted">
          Readings 和 Fields 看起来最近，方向却相反：Reading 的输入是别人的文字，输出是自己的理解；Field Note 的输入是一个问题或一篇论文，输出是自己的数据。Readings 是消化，<a href="/fields/" style={{ textDecoration: "underline", textDecorationColor: "var(--rule)", textUnderlineOffset: "3px" }}>Fields</a> 是生产。
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
