(() => {
  // _jsx/pages/readings.jsx
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
    useSection("readings");
    useApplyTweaks(t);
    const [activeTag, setActiveTag] = useState(null);
    const tagCounts = useMemo(() => {
      const m = /* @__PURE__ */ new Map();
      for (const r of ALL_READINGS) for (const tag of r.tags) m.set(tag, (m.get(tag) || 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    }, []);
    const filtered = useMemo(
      () => activeTag ? ALL_READINGS.filter((r) => r.tags.includes(activeTag)) : ALL_READINGS,
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
    return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(Topbar, { active: "readings" }), /* @__PURE__ */ React.createElement("main", { className: "page" }, /* @__PURE__ */ React.createElement("div", { className: "pg-meta" }, /* @__PURE__ */ React.createElement("a", { href: "/" }, "MiracleFarms"), /* @__PURE__ */ React.createElement("span", { className: "dot" }, "\u203A"), /* @__PURE__ */ React.createElement("span", null, "Readings")), /* @__PURE__ */ React.createElement(PageHead, { section: "readings" }), /* @__PURE__ */ React.createElement("aside", { className: "callout" }, /* @__PURE__ */ React.createElement("span", { className: "ico", "aria-hidden": "true" }, "\u{1F4D6}"), /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("p", null, "\u5171 ", /* @__PURE__ */ React.createElement("strong", null, ALL_READINGS.length), " \u7BC7 \xB7 Reading \u7684\u8F93\u5165\u662F\u522B\u4EBA\u7684\u6587\u5B57\uFF0C\u8F93\u51FA\u662F\u81EA\u5DF1\u7684\u5224\u65AD\u3002\u6211\u4EEC\u503E\u5411\u4E8E\u6311\u90A3\u4E9B", /* @__PURE__ */ React.createElement("strong", null, "\u80FD\u6539\u53D8\u5DE5\u7A0B\u5224\u65AD"), "\u7684\u8BBA\u6587\u6216\u7CFB\u7EDF\u7CBE\u8BFB\uFF0C\u800C\u4E0D\u662F\u7EFC\u8FF0\u5F0F\u7F57\u5217\u3002"))), /* @__PURE__ */ React.createElement("div", { className: "filterbar", role: "toolbar", "aria-label": "\u7B5B\u9009" }, /* @__PURE__ */ React.createElement("span", { className: "lbl" }, "Filter:"), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "fchip" + (activeTag === null ? " on" : ""),
        onClick: () => setActiveTag(null),
        type: "button"
      },
      "\u5168\u90E8 ",
      /* @__PURE__ */ React.createElement("span", { className: "count" }, ALL_READINGS.length)
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
    ))), filtered.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "empty" }, "\u6CA1\u6709\u5339\u914D\u7684 reading\u3002") : byYear.map(([yr, items]) => /* @__PURE__ */ React.createElement("section", { key: yr }, /* @__PURE__ */ React.createElement("div", { className: "year-row" }, /* @__PURE__ */ React.createElement("span", { className: "yr" }, yr), /* @__PURE__ */ React.createElement("span", { className: "ln" }), /* @__PURE__ */ React.createElement("span", null, items.length, " \u7BC7")), /* @__PURE__ */ React.createElement("div", { className: "pagelist", role: "list" }, items.map((r, i) => /* @__PURE__ */ React.createElement("a", { key: i, className: "pagelink", href: r.href, role: "listitem" }, /* @__PURE__ */ React.createElement(HandleDots, null), /* @__PURE__ */ React.createElement("span", { className: "pl-ico", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(Icon, { name: "book", size: 16 })), /* @__PURE__ */ React.createElement("div", { className: "pl-body" }, /* @__PURE__ */ React.createElement("div", { className: "pl-title" }, r.title, r.locked && /* @__PURE__ */ React.createElement("span", { className: "pl-lock", "aria-label": "\u9700\u8981\u5BC6\u7801", title: "\u9700\u8981\u5BC6\u7801\u9605\u8BFB" }, /* @__PURE__ */ React.createElement(Icon, { name: "lock", size: 11 })), r.hasEn && /* @__PURE__ */ React.createElement("span", { className: "pl-lang", title: "\u4E2D\u82F1\u53CC\u8BED", "aria-label": "\u63D0\u4F9B\u4E2D\u82F1\u53CC\u8BED\u7248\u672C" }, /* @__PURE__ */ React.createElement("i", null, "\u4E2D"), /* @__PURE__ */ React.createElement("i", null, "EN"))), /* @__PURE__ */ React.createElement("div", { className: "pl-excerpt" }, r.excerpt), /* @__PURE__ */ React.createElement("div", { className: "pl-meta" }, r.tags.map((tag) => /* @__PURE__ */ React.createElement("span", { key: tag, className: "tag" }, tag)))), /* @__PURE__ */ React.createElement("div", { className: "pl-date" }, r.date)))))), /* @__PURE__ */ React.createElement("hr", { className: "divider" }), /* @__PURE__ */ React.createElement("h6", { className: "h2-sub" }, "Reading vs Field"), /* @__PURE__ */ React.createElement("h2", { className: "h2" }, "Readings \u4E0E Fields \u7684\u5206\u5DE5"), /* @__PURE__ */ React.createElement("p", { className: "muted" }, "Readings \u548C Fields \u770B\u8D77\u6765\u6700\u8FD1\uFF0C\u65B9\u5411\u5374\u76F8\u53CD\uFF1AReading \u7684\u8F93\u5165\u662F\u522B\u4EBA\u7684\u6587\u5B57\uFF0C\u8F93\u51FA\u662F\u81EA\u5DF1\u7684\u7406\u89E3\uFF1BField Note \u7684\u8F93\u5165\u662F\u4E00\u4E2A\u95EE\u9898\u6216\u4E00\u7BC7\u8BBA\u6587\uFF0C\u8F93\u51FA\u662F\u81EA\u5DF1\u7684\u6570\u636E\u3002Readings \u662F\u6D88\u5316\uFF0C", /* @__PURE__ */ React.createElement("a", { href: "/fields/", style: { textDecoration: "underline", textDecorationColor: "var(--rule)", textUnderlineOffset: "3px" } }, "Fields"), " \u662F\u751F\u4EA7\u3002")), /* @__PURE__ */ React.createElement(PageFooter, null), /* @__PURE__ */ React.createElement(TweaksPanel, { title: "Tweaks" }, /* @__PURE__ */ React.createElement(TweakSection, { label: "\u677F\u5757\u8272\u5F3A\u5EA6 Section colour" }, /* @__PURE__ */ React.createElement(
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
