(() => {
  // _jsx/pages/fields.jsx
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
    useSection("fields");
    useApplyTweaks(t);
    const [activeTag, setActiveTag] = useState(null);
    const tagCounts = useMemo(() => {
      const m = /* @__PURE__ */ new Map();
      for (const f of ALL_FIELDS) for (const tag of f.tags) m.set(tag, (m.get(tag) || 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    }, []);
    const filtered = useMemo(
      () => activeTag ? ALL_FIELDS.filter((f) => f.tags.includes(activeTag)) : ALL_FIELDS,
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
    return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(Topbar, { active: "fields" }), /* @__PURE__ */ React.createElement("main", { className: "page" }, /* @__PURE__ */ React.createElement("div", { className: "pg-meta" }, /* @__PURE__ */ React.createElement("a", { href: "/" }, "MiracleFarms"), /* @__PURE__ */ React.createElement("span", { className: "dot" }, "\u203A"), /* @__PURE__ */ React.createElement("span", null, "Field Notes")), /* @__PURE__ */ React.createElement(PageHead, { section: "fields" }), /* @__PURE__ */ React.createElement("aside", { className: "callout" }, /* @__PURE__ */ React.createElement("span", { className: "ico", "aria-hidden": "true" }, "\u{1F9EA}"), /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("p", null, "\u5171 ", /* @__PURE__ */ React.createElement("strong", null, ALL_FIELDS.length), " \u7BC7 \xB7 \u6838\u5FC3\u7EA6\u675F\u53EA\u6709\u4E00\u6761\uFF1A\u5199\u8FDB\u6765\u7684\u4E1C\u897F\u5FC5\u987B\u662F", /* @__PURE__ */ React.createElement("strong", null, "\u4EB2\u624B\u9A8C\u8BC1\u8FC7\u7684"), "\u3002\u4E0D\u662F\u201C\u636E\u8BBA\u6587\u79F0\u201D\uFF0C\u4E0D\u662F\u201C\u7406\u8BBA\u4E0A\u5E94\u8BE5\u201D\uFF0C\u800C\u662F\u201C\u6211\u8DD1\u51FA\u6765\u7684\u7ED3\u679C\u662F\u8FD9\u6837\u7684\u201D\u3002"))), /* @__PURE__ */ React.createElement("div", { className: "filterbar", role: "toolbar", "aria-label": "\u7B5B\u9009" }, /* @__PURE__ */ React.createElement("span", { className: "lbl" }, "Filter:"), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "fchip" + (activeTag === null ? " on" : ""),
        onClick: () => setActiveTag(null),
        type: "button"
      },
      "\u5168\u90E8 ",
      /* @__PURE__ */ React.createElement("span", { className: "count" }, ALL_FIELDS.length)
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
    ))), filtered.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "empty" }, "\u6CA1\u6709\u5339\u914D\u7684 field note\u3002") : byYear.map(([yr, items]) => /* @__PURE__ */ React.createElement("section", { key: yr }, /* @__PURE__ */ React.createElement("div", { className: "year-row" }, /* @__PURE__ */ React.createElement("span", { className: "yr" }, yr), /* @__PURE__ */ React.createElement("span", { className: "ln" }), /* @__PURE__ */ React.createElement("span", null, items.length, " \u7BC7")), /* @__PURE__ */ React.createElement("div", { className: "pagelist", role: "list" }, items.map((f, i) => /* @__PURE__ */ React.createElement("a", { key: i, className: "pagelink", href: f.href, role: "listitem" }, /* @__PURE__ */ React.createElement(HandleDots, null), /* @__PURE__ */ React.createElement("span", { className: "pl-ico", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(Icon, { name: "flask", size: 16 })), /* @__PURE__ */ React.createElement("div", { className: "pl-body" }, /* @__PURE__ */ React.createElement("div", { className: "pl-title" }, f.title, f.locked && /* @__PURE__ */ React.createElement("span", { className: "pl-lock", "aria-label": "\u9700\u8981\u5BC6\u7801", title: "\u9700\u8981\u5BC6\u7801\u9605\u8BFB" }, /* @__PURE__ */ React.createElement(Icon, { name: "lock", size: 11 })), f.hasEn && /* @__PURE__ */ React.createElement("span", { className: "pl-lang", title: "\u4E2D\u82F1\u53CC\u8BED", "aria-label": "\u63D0\u4F9B\u4E2D\u82F1\u53CC\u8BED\u7248\u672C" }, /* @__PURE__ */ React.createElement("i", null, "\u4E2D"), /* @__PURE__ */ React.createElement("i", null, "EN"))), /* @__PURE__ */ React.createElement("div", { className: "pl-excerpt" }, f.excerpt), /* @__PURE__ */ React.createElement("div", { className: "pl-meta" }, f.tags.map((tag) => /* @__PURE__ */ React.createElement("span", { key: tag, className: "tag" }, tag)))), /* @__PURE__ */ React.createElement("div", { className: "pl-date" }, f.date)))))), /* @__PURE__ */ React.createElement("hr", { className: "divider" }), /* @__PURE__ */ React.createElement("h6", { className: "h2-sub" }, "What goes here"), /* @__PURE__ */ React.createElement("h2", { className: "h2" }, "\u8FD9\u91CC\u4F1A\u653E\u4EC0\u4E48"), /* @__PURE__ */ React.createElement("p", { className: "muted" }, "\u8BBA\u6587\u590D\u73B0\u3001\u90E8\u7F72\u5B9E\u5F55\u3001\u8C03\u53C2\u8BB0\u5F55\u3001\u8E29\u5751\u7B14\u8BB0\u2014\u2014\u6838\u5FC3\u662F\u201C\u53EF\u590D\u73B0\u201D\uFF0C\u6BCF\u4E00\u6761\u7ED3\u8BBA\u90FD\u5C3D\u91CF\u5E26\u4E0A\u73AF\u5883\u3001\u7248\u672C\u4E0E\u547D\u4EE4\u3002Fields \u7684\u66F4\u65B0\u4E0D\u8FFD\u5FEB\uFF0C\u53EA\u8FFD", /* @__PURE__ */ React.createElement("strong", null, "\u80FD\u88AB\u522B\u4EBA\u7167\u7740\u8DD1\u4E00\u904D"), "\u3002")), /* @__PURE__ */ React.createElement(PageFooter, null), /* @__PURE__ */ React.createElement(TweaksPanel, { title: "Tweaks" }, /* @__PURE__ */ React.createElement(TweakSection, { label: "\u677F\u5757\u8272\u5F3A\u5EA6 Section colour" }, /* @__PURE__ */ React.createElement(
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
