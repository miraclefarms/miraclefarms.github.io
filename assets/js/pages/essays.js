(() => {
  // _jsx/pages/essays.jsx
  var { useState, useMemo } = React;
  var TWEAK_DEFAULTS = (
    /*EDITMODE-BEGIN*/
    {
      "fontStyle": "serif",
      "smallText": false,
      "fullWidth": false,
      "intensity": "medium"
    }
  );
  function App() {
    const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
    useSection("essays");
    useApplyTweaks(t);
    const [activeTag, setActiveTag] = useState(null);
    const tagCounts = useMemo(() => {
      const m = /* @__PURE__ */ new Map();
      for (const e of ALL_ESSAYS) for (const tag of e.tags) m.set(tag, (m.get(tag) || 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    }, []);
    const filtered = useMemo(
      () => activeTag ? ALL_ESSAYS.filter((e) => e.tags.includes(activeTag)) : ALL_ESSAYS,
      [activeTag]
    );
    const byYear = useMemo(() => {
      const groups = {};
      for (const e of filtered) {
        const y = e.date.slice(0, 4);
        (groups[y] = groups[y] || []).push(e);
      }
      return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
    }, [filtered]);
    return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(Topbar, { active: "essays" }), /* @__PURE__ */ React.createElement("main", { className: "page" }, /* @__PURE__ */ React.createElement("div", { className: "pg-meta" }, /* @__PURE__ */ React.createElement("a", { href: "/" }, "MiracleFarms"), /* @__PURE__ */ React.createElement("span", { className: "dot" }, "\u203A"), /* @__PURE__ */ React.createElement("span", null, "Essays")), /* @__PURE__ */ React.createElement(PageHead, { section: "essays" }), /* @__PURE__ */ React.createElement("aside", { className: "callout" }, /* @__PURE__ */ React.createElement("span", { className: "ico", "aria-hidden": "true" }, "\u{1F4DA}"), /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("p", null, "\u5171 ", /* @__PURE__ */ React.createElement("strong", null, ALL_ESSAYS.length), " \u7BC7 \xB7 \u4E3B\u9898\u8986\u76D6 ", /* @__PURE__ */ React.createElement("strong", null, "Inference / Agents / Memory / Evaluation / Reliability"), "\u3002\u6BCF\u4E00\u7BC7\u90FD\u5C1D\u8BD5\u5728\u4E00\u4E2A\u5177\u4F53\u8FB9\u754C\u4E0A\u7ED9\u51FA\u53EF\u590D\u76D8\u7684\u5224\u65AD\uFF0C\u800C\u4E0D\u662F\u7EFC\u8FF0\u3002"))), /* @__PURE__ */ React.createElement("div", { className: "filterbar", role: "toolbar", "aria-label": "\u7B5B\u9009" }, /* @__PURE__ */ React.createElement("span", { className: "lbl" }, "Filter:"), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "fchip" + (activeTag === null ? " on" : ""),
        onClick: () => setActiveTag(null),
        type: "button"
      },
      "\u5168\u90E8 ",
      /* @__PURE__ */ React.createElement("span", { className: "count" }, ALL_ESSAYS.length)
    ), tagCounts.map(([tag, n]) => /* @__PURE__ */ React.createElement(
      "button",
      {
        key: tag,
        className: "fchip" + (activeTag === tag ? " on" : ""),
        onClick: () => setActiveTag(activeTag === tag ? null : tag),
        type: "button"
      },
      tag,
      " ",
      /* @__PURE__ */ React.createElement("span", { className: "count" }, n)
    ))), filtered.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "empty" }, "\u6CA1\u6709\u5339\u914D\u7684 essay\u3002") : byYear.map(([yr, items]) => /* @__PURE__ */ React.createElement("section", { key: yr }, /* @__PURE__ */ React.createElement("div", { className: "year-row" }, /* @__PURE__ */ React.createElement("span", { className: "yr" }, yr), /* @__PURE__ */ React.createElement("span", { className: "ln" }), /* @__PURE__ */ React.createElement("span", null, items.length, " \u7BC7")), /* @__PURE__ */ React.createElement("div", { className: "pagelist", role: "list" }, items.map((e, i) => /* @__PURE__ */ React.createElement("a", { key: i, className: "pagelink", href: e.href, role: "listitem" }, /* @__PURE__ */ React.createElement(HandleDots, null), /* @__PURE__ */ React.createElement("span", { className: "pl-ico", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(Icon, { name: "pen", size: 16 })), /* @__PURE__ */ React.createElement("div", { className: "pl-body" }, /* @__PURE__ */ React.createElement("div", { className: "pl-title" }, e.title, e.locked && /* @__PURE__ */ React.createElement("span", { className: "pl-lock", "aria-label": "\u9700\u8981\u5BC6\u7801", title: "\u9700\u8981\u5BC6\u7801\u9605\u8BFB" }, /* @__PURE__ */ React.createElement(Icon, { name: "lock", size: 11 })), e.hasEn && /* @__PURE__ */ React.createElement("span", { className: "pl-lang", title: "\u4E2D\u82F1\u53CC\u8BED", "aria-label": "\u63D0\u4F9B\u4E2D\u82F1\u53CC\u8BED\u7248\u672C" }, /* @__PURE__ */ React.createElement("i", null, "\u4E2D"), /* @__PURE__ */ React.createElement("i", null, "EN"))), /* @__PURE__ */ React.createElement("div", { className: "pl-excerpt" }, e.excerpt), /* @__PURE__ */ React.createElement("div", { className: "pl-meta" }, e.tags.map((tag) => /* @__PURE__ */ React.createElement("span", { key: tag, className: "tag" }, tag)))), /* @__PURE__ */ React.createElement("div", { className: "pl-date" }, e.date)))))), /* @__PURE__ */ React.createElement("hr", { className: "divider" }), /* @__PURE__ */ React.createElement("h6", { className: "h2-sub" }, "Reading note"), /* @__PURE__ */ React.createElement("h2", { className: "h2" }, "\u5173\u4E8E Essays \u7684\u5199\u4F5C"), /* @__PURE__ */ React.createElement("p", { className: "muted" }, "Essays \u7684\u66F4\u65B0\u9891\u7387\u660E\u663E\u4F4E\u4E8E Briefs\u2014\u2014\u5B83\u4EEC\u901A\u5E38\u9700\u8981\u6570\u5468\u5230\u6570\u6708\u7684\u89C2\u5BDF\u6C89\u6DC0\uFF0C\u5E76\u7ECF\u8FC7\u4E00\u6B21\u4EE5\u4E0A\u7684\u91CD\u5199\u3002\u6211\u4EEC\u503E\u5411\u4E8E\u628A\u5DF2\u7ECF\u88AB\u591A\u4E2A brief \u53CD\u590D\u89E6\u53CA\u7684\u4E3B\u9898\uFF0C\u6574\u7406\u6210\u4E00\u7BC7\u53EF\u4EE5\u957F\u671F\u88AB\u5F15\u7528\u7684 essay\u3002")), /* @__PURE__ */ React.createElement(PageFooter, null), /* @__PURE__ */ React.createElement(TweaksPanel, { title: "Tweaks" }, /* @__PURE__ */ React.createElement(TweakSection, { label: "\u677F\u5757\u8272\u5F3A\u5EA6 Section colour" }, /* @__PURE__ */ React.createElement(
      TweakRadio,
      {
        value: t.intensity,
        onChange: (v) => setTweak("intensity", v),
        options: [
          { value: "clean", label: "\u514B\u5236" },
          { value: "medium", label: "\u4E2D\u5EA6" },
          { value: "expressive", label: "\u660E\u663E" }
        ]
      }
    )), /* @__PURE__ */ React.createElement(TweakSection, { label: "\u5B57\u4F53 Font style" }, /* @__PURE__ */ React.createElement(
      TweakRadio,
      {
        value: t.fontStyle,
        onChange: (v) => setTweak("fontStyle", v),
        options: [
          { value: "default", label: "Default" },
          { value: "serif", label: "Serif" },
          { value: "mono", label: "Mono" }
        ]
      }
    )), /* @__PURE__ */ React.createElement(TweakSection, { label: "\u6392\u7248 Layout" }, /* @__PURE__ */ React.createElement(TweakToggle, { label: "Small text \u5C0F\u5B57\u53F7", value: t.smallText, onChange: (v) => setTweak("smallText", v) }), /* @__PURE__ */ React.createElement(TweakToggle, { label: "Full width \u5168\u5BBD\u9875\u9762", value: t.fullWidth, onChange: (v) => setTweak("fullWidth", v) }))));
  }
  ReactDOM.createRoot(document.getElementById("root")).render(/* @__PURE__ */ React.createElement(App, null));
})();
