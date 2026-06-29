(() => {
  // _jsx/pages/briefs.jsx
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
  function yearOf(dateStr) {
    return dateStr.slice(0, 4);
  }
  function App() {
    const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
    useSection("briefs");
    useApplyTweaks(t);
    const [view, setView] = useState("list");
    const [activeTag, setActiveTag] = useState(null);
    const tagCounts = useMemo(() => {
      const m = /* @__PURE__ */ new Map();
      for (const b of ALL_BRIEFS) for (const tag of b.tags) m.set(tag, (m.get(tag) || 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    }, []);
    const filtered = useMemo(
      () => activeTag ? ALL_BRIEFS.filter((b) => b.tags.includes(activeTag)) : ALL_BRIEFS,
      [activeTag]
    );
    const byYear = useMemo(() => {
      const groups = {};
      for (const b of filtered) {
        const y = yearOf(b.date);
        (groups[y] = groups[y] || []).push(b);
      }
      return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
    }, [filtered]);
    return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(Topbar, { active: "briefs" }), /* @__PURE__ */ React.createElement("main", { className: "page" }, /* @__PURE__ */ React.createElement("div", { className: "pg-meta" }, /* @__PURE__ */ React.createElement("a", { href: "/" }, "MiracleFarms"), /* @__PURE__ */ React.createElement("span", { className: "dot" }, "\u203A"), /* @__PURE__ */ React.createElement("span", null, "Briefs")), /* @__PURE__ */ React.createElement(PageHead, { section: "briefs" }), /* @__PURE__ */ React.createElement("aside", { className: "callout" }, /* @__PURE__ */ React.createElement("span", { className: "ico", "aria-hidden": "true" }, "\u{1F5D2}\uFE0F"), /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("p", null, "\u5171 ", /* @__PURE__ */ React.createElement("strong", null, ALL_BRIEFS.length), " \u6761", ALL_BRIEFS.length > 0 ? /* @__PURE__ */ React.createElement(React.Fragment, null, " \xB7 \u6700\u65B0 ", /* @__PURE__ */ React.createElement("strong", null, ALL_BRIEFS[0].date)) : null, " \xB7 \u6309\u65E5\u671F\u5012\u5E8F\u6392\u5217\u3002\u53EF\u6309\u6807\u7B7E\u7B5B\u9009\uFF0C\u6216\u5207\u6362 Table \u89C6\u56FE\u3002"))), /* @__PURE__ */ React.createElement("div", { className: "filterbar", role: "toolbar", "aria-label": "\u7B5B\u9009\u4E0E\u89C6\u56FE" }, /* @__PURE__ */ React.createElement("span", { className: "lbl" }, "Filter:"), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "fchip" + (activeTag === null ? " on" : ""),
        onClick: () => setActiveTag(null),
        type: "button"
      },
      "\u5168\u90E8 ",
      /* @__PURE__ */ React.createElement("span", { className: "count" }, ALL_BRIEFS.length)
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
    )), /* @__PURE__ */ React.createElement("span", { style: { flex: 1 } }), /* @__PURE__ */ React.createElement("span", { className: "viewtabs", role: "tablist", "aria-label": "\u89C6\u56FE\u5207\u6362" }, /* @__PURE__ */ React.createElement("button", { className: view === "list" ? "on" : "", onClick: () => setView("list"), type: "button" }, /* @__PURE__ */ React.createElement(Icon, { name: "list", size: 12 }), " List"), /* @__PURE__ */ React.createElement("button", { className: view === "table" ? "on" : "", onClick: () => setView("table"), type: "button" }, /* @__PURE__ */ React.createElement(Icon, { name: "table", size: 12 }), " Table"))), filtered.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "empty" }, "\u6CA1\u6709\u5339\u914D\u7684 brief\u3002\u8BD5\u8BD5\u6E05\u9664\u7B5B\u9009\u3002") : view === "list" ? byYear.map(([yr, items]) => /* @__PURE__ */ React.createElement("section", { key: yr }, /* @__PURE__ */ React.createElement("div", { className: "year-row" }, /* @__PURE__ */ React.createElement("span", { className: "yr" }, yr), /* @__PURE__ */ React.createElement("span", { className: "ln" }), /* @__PURE__ */ React.createElement("span", null, items.length, " \u6761")), /* @__PURE__ */ React.createElement("div", { className: "pagelist", role: "list" }, items.map((b, i) => /* @__PURE__ */ React.createElement("a", { key: i, className: "pagelink", href: b.href, role: "listitem" }, /* @__PURE__ */ React.createElement(HandleDots, null), /* @__PURE__ */ React.createElement("span", { className: "pl-ico", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(Icon, { name: "doc", size: 16 })), /* @__PURE__ */ React.createElement("div", { className: "pl-body" }, /* @__PURE__ */ React.createElement("div", { className: "pl-title" }, b.title, b.locked && /* @__PURE__ */ React.createElement("span", { className: "pl-lock", "aria-label": "\u9700\u8981\u5BC6\u7801", title: "\u9700\u8981\u5BC6\u7801\u9605\u8BFB" }, /* @__PURE__ */ React.createElement(Icon, { name: "lock", size: 11 })), b.hasEn && /* @__PURE__ */ React.createElement("span", { className: "pl-lang", title: "\u4E2D\u82F1\u53CC\u8BED", "aria-label": "\u63D0\u4F9B\u4E2D\u82F1\u53CC\u8BED\u7248\u672C" }, /* @__PURE__ */ React.createElement("i", null, "\u4E2D"), /* @__PURE__ */ React.createElement("i", null, "EN"))), /* @__PURE__ */ React.createElement("div", { className: "pl-excerpt" }, b.excerpt), /* @__PURE__ */ React.createElement("div", { className: "pl-meta" }, b.tags.map((tag) => /* @__PURE__ */ React.createElement("span", { key: tag, className: "tag" }, tag)))), /* @__PURE__ */ React.createElement("div", { className: "pl-date" }, b.date)))))) : /* @__PURE__ */ React.createElement("table", { className: "ntable", role: "table" }, /* @__PURE__ */ React.createElement("thead", null, /* @__PURE__ */ React.createElement("tr", null, /* @__PURE__ */ React.createElement("th", { className: "col-date", style: { width: "120px" } }, /* @__PURE__ */ React.createElement("span", { className: "col-ico" }, /* @__PURE__ */ React.createElement(Icon, { name: "calendar", size: 12 })), "Date"), /* @__PURE__ */ React.createElement("th", { className: "col-title" }, /* @__PURE__ */ React.createElement("span", { className: "col-ico" }, /* @__PURE__ */ React.createElement(Icon, { name: "doc", size: 12 })), "Title"), /* @__PURE__ */ React.createElement("th", { className: "col-tags", style: { width: "240px" } }, /* @__PURE__ */ React.createElement("span", { className: "col-ico" }, /* @__PURE__ */ React.createElement(Icon, { name: "tag", size: 12 })), "Tags"))), /* @__PURE__ */ React.createElement("tbody", null, filtered.map((b, i) => /* @__PURE__ */ React.createElement("tr", { key: i }, /* @__PURE__ */ React.createElement("td", { className: "col-date" }, b.date), /* @__PURE__ */ React.createElement("td", { className: "col-title" }, /* @__PURE__ */ React.createElement("a", { href: b.href }, b.title, b.hasEn && /* @__PURE__ */ React.createElement("span", { className: "pl-lang", title: "\u4E2D\u82F1\u53CC\u8BED", "aria-label": "\u63D0\u4F9B\u4E2D\u82F1\u53CC\u8BED\u7248\u672C" }, /* @__PURE__ */ React.createElement("i", null, "\u4E2D"), /* @__PURE__ */ React.createElement("i", null, "EN")))), /* @__PURE__ */ React.createElement("td", { className: "col-tags" }, b.tags.map((tag) => /* @__PURE__ */ React.createElement("span", { key: tag, className: "tag" }, tag)))))))), /* @__PURE__ */ React.createElement(PageFooter, null), /* @__PURE__ */ React.createElement(TweaksPanel, { title: "Tweaks" }, /* @__PURE__ */ React.createElement(TweakSection, { label: "\u677F\u5757\u8272\u5F3A\u5EA6 Section colour" }, /* @__PURE__ */ React.createElement(
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
