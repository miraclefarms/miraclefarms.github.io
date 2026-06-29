(() => {
  // assets/shared.jsx
  var MFMark = ({ size = 28, soft = 0.55 }) => /* @__PURE__ */ React.createElement(
    "svg",
    {
      className: "mf-mark",
      width: size,
      height: size,
      viewBox: "0 0 200 200",
      fill: "none",
      strokeLinecap: "square",
      "aria-hidden": "true"
    },
    /* @__PURE__ */ React.createElement("line", { x1: "25", y1: "178", x2: "25", y2: "58", stroke: "currentColor", strokeWidth: "22" }),
    /* @__PURE__ */ React.createElement("line", { x1: "62", y1: "178", x2: "62", y2: "128", stroke: "currentColor", strokeWidth: "16", opacity: soft }),
    /* @__PURE__ */ React.createElement("line", { x1: "100", y1: "178", x2: "100", y2: "93", stroke: "currentColor", strokeWidth: "22" }),
    /* @__PURE__ */ React.createElement("line", { x1: "138", y1: "178", x2: "138", y2: "108", stroke: "currentColor", strokeWidth: "16", opacity: soft }),
    /* @__PURE__ */ React.createElement("line", { x1: "175", y1: "178", x2: "175", y2: "33", stroke: "currentColor", strokeWidth: "22" })
  );
  var Icon = ({ name, size = 16 }) => {
    const common = {
      width: size,
      height: size,
      viewBox: "0 0 24 24",
      fill: "none",
      stroke: "currentColor",
      strokeWidth: 1.6,
      strokeLinecap: "round",
      strokeLinejoin: "round"
    };
    switch (name) {
      case "home":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M3 11l9-8 9 8" }), /* @__PURE__ */ React.createElement("path", { d: "M5 10v10h14V10" }));
      case "doc":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M7 3h7l5 5v13H7z" }), /* @__PURE__ */ React.createElement("path", { d: "M14 3v5h5" }), /* @__PURE__ */ React.createElement("path", { d: "M9 13h7M9 17h5" }));
      case "book":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3V4z" }), /* @__PURE__ */ React.createElement("path", { d: "M5 17a3 3 0 013-3h11" }));
      case "flask":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M9 3h6" }), /* @__PURE__ */ React.createElement("path", { d: "M10 3v6l-5 8.5A1.5 1.5 0 006.3 20h11.4a1.5 1.5 0 001.3-2.5L14 9V3" }), /* @__PURE__ */ React.createElement("path", { d: "M7.5 14h9" }));
      case "pen":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M4 20l3.5-.8L19 7.7a2 2 0 00-2.7-2.7L4.8 16.5z" }), /* @__PURE__ */ React.createElement("path", { d: "M14 6.5l3.5 3.5" }));
      case "sprout":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M12 20v-7" }), /* @__PURE__ */ React.createElement("path", { d: "M12 13c0-3.5-2.7-5.5-6.5-5.5C5.5 11 8.2 13 12 13z" }), /* @__PURE__ */ React.createElement("path", { d: "M12 11.5c0-3 2.4-4.8 5.8-4.8C17.8 9.7 15.4 11.5 12 11.5z" }));
      case "github":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M9 19c-4 1.5-4-2-6-2m12 4v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 00-1.3-3.2 4.3 4.3 0 00-.1-3.2s-1.1-.3-3.5 1.3a12 12 0 00-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.3 4.3 0 00-.1 3.2A4.6 4.6 0 004 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21" }));
      case "search":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("circle", { cx: "11", cy: "11", r: "7" }), /* @__PURE__ */ React.createElement("path", { d: "M21 21l-4.3-4.3" }));
      case "wiki":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("circle", { cx: "6", cy: "6", r: "2.5" }), /* @__PURE__ */ React.createElement("circle", { cx: "18", cy: "7", r: "2.5" }), /* @__PURE__ */ React.createElement("circle", { cx: "8", cy: "18", r: "2.5" }), /* @__PURE__ */ React.createElement("circle", { cx: "17", cy: "17", r: "2.5" }), /* @__PURE__ */ React.createElement("path", { d: "M8.3 7.1l7.4 8.8M15.7 7.9L10.1 16M10.5 18h4" }));
      case "feed":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M4 11a9 9 0 019 9" }), /* @__PURE__ */ React.createElement("path", { d: "M4 4a16 16 0 0116 16" }), /* @__PURE__ */ React.createElement("circle", { cx: "5", cy: "19", r: "1.5" }));
      case "wechat":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M9 4C5.1 4 2 6.7 2 10c0 1.7.8 3.2 2.2 4.3L3.5 17l3-1.6c.7.2 1.5.3 2.3.3" }), /* @__PURE__ */ React.createElement("path", { d: "M22 14.5c0-2.7-2.6-4.9-5.8-4.9S10.4 11.8 10.4 14.5s2.6 4.9 5.8 4.9c.7 0 1.3-.1 1.9-.3l2.4 1.3-.6-2.1c1.2-.9 2.1-2.2 2.1-3.8z" }));
      case "spark":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" }));
      case "list":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M8 6h13M8 12h13M8 18h13" }), /* @__PURE__ */ React.createElement("circle", { cx: "4", cy: "6", r: "1" }), /* @__PURE__ */ React.createElement("circle", { cx: "4", cy: "12", r: "1" }), /* @__PURE__ */ React.createElement("circle", { cx: "4", cy: "18", r: "1" }));
      case "table":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("rect", { x: "3", y: "4", width: "18", height: "16", rx: "1.5" }), /* @__PURE__ */ React.createElement("path", { d: "M3 10h18M3 15h18M9 4v16" }));
      case "tag":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M20 12L12 20l-8-8V4h8z" }), /* @__PURE__ */ React.createElement("circle", { cx: "8", cy: "8", r: "1.5" }));
      case "calendar":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("rect", { x: "3", y: "5", width: "18", height: "16", rx: "1.5" }), /* @__PURE__ */ React.createElement("path", { d: "M3 10h18M8 3v4M16 3v4" }));
      case "arrow":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("path", { d: "M5 12h14M13 6l6 6-6 6" }));
      case "lock":
        return /* @__PURE__ */ React.createElement("svg", { ...common }, /* @__PURE__ */ React.createElement("rect", { x: "5", y: "11", width: "14", height: "9", rx: "2" }), /* @__PURE__ */ React.createElement("path", { d: "M8 11V8a4 4 0 018 0v3" }));
      default:
        return null;
    }
  };
  var HandleDots = () => /* @__PURE__ */ React.createElement("span", { className: "pl-handle", "aria-hidden": "true" }, "\u22EE\u22EE");
  var SECTIONS = {
    home: { name: "MiracleFarms", label: "AI INFRA \xB7 JOURNAL", cjk: "\u516C\u5F00\u5199\u4F5C", icon: "home" },
    briefs: {
      name: "Briefs",
      label: "BRIEF \xB7 DAILY",
      cjk: "\u65E5\u62A5",
      icon: "list",
      sub: "AI Infra \u7684\u65E5\u5E38\u89C2\u6D4B\u4E0E\u77ED\u5224\u65AD\u3002",
      subSmall: "\u6BCF\u4E00\u6761\u90FD\u805A\u7126\u5728\u4E00\u4E2A\u5177\u4F53\u53D8\u5316\u4E0A\u2014\u2014\u4E00\u4E2A PR\u3001\u4E00\u6B21\u53D1\u7248\u3001\u4E00\u4E2A\u88AB\u6084\u6084\u5199\u56DE\u4E3B\u8DEF\u5F84\u7684\u9ED8\u8BA4\u884C\u4E3A\u2014\u2014\u5E76\u5C1D\u8BD5\u628A\u5B83\u653E\u56DE\u5230\u66F4\u5927\u7684\u8D8B\u52BF\u91CC\u3002"
    },
    readings: {
      name: "Readings",
      label: "READING \xB7 NOTE",
      cjk: "\u9605\u8BFB\u7B14\u8BB0",
      icon: "book",
      sub: "\u8BFB\u522B\u4EBA\u7684\u8BBA\u6587\u4E0E\u7CFB\u7EDF\uFF0C\u5199\u4E0B\u81EA\u5DF1\u7684\u7406\u89E3\u3002",
      subSmall: "Readings \u7684\u8F93\u5165\u662F\u522B\u4EBA\u7684\u6587\u5B57\uFF0C\u8F93\u51FA\u662F\u81EA\u5DF1\u7684\u5224\u65AD\u2014\u2014\u9010\u7BC7\u62C6\u89E3\u4E00\u4E2A\u7CFB\u7EDF\u6216\u4E00\u7BC7\u8BBA\u6587\u600E\u4E48\u5DE5\u4F5C\u3001\u5728\u4EC0\u4E48\u8FB9\u754C\u6761\u4EF6\u4E0B\u5931\u6548\u3002"
    },
    fields: {
      name: "Field Notes",
      label: "FIELD NOTE",
      cjk: "\u7530\u95F4\u7B14\u8BB0",
      icon: "flask",
      sub: "\u81EA\u5DF1\u52A8\u624B\u8DD1\u51FA\u6765\u7684\u7ED3\u679C\u3002",
      subSmall: "\u5199\u8FDB\u6765\u7684\u4E1C\u897F\u5FC5\u987B\u662F\u4EB2\u624B\u9A8C\u8BC1\u8FC7\u7684\u2014\u2014\u8BBA\u6587\u590D\u73B0\u3001\u90E8\u7F72\u5B9E\u5F55\u3001\u8C03\u53C2\u8BB0\u5F55\u3002\u4E0D\u662F\u201C\u636E\u8BBA\u6587\u79F0\u201D\uFF0C\u800C\u662F\u201C\u6211\u8DD1\u51FA\u6765\u7684\u7ED3\u679C\u662F\u8FD9\u6837\u7684\u201D\u3002"
    },
    essays: {
      name: "Essays",
      label: "ESSAY \xB7 LONGFORM",
      cjk: "\u957F\u6587",
      icon: "pen",
      sub: "\u957F\u6587\uFF1AAI Infra \u7684\u7CFB\u7EDF\u6027\u89C2\u5BDF\u4E0E\u5224\u65AD\u3002",
      subSmall: "\u76F8\u5BF9\u4E8E Briefs \u7684\u201C\u4E00\u65E5\u4E00\u89C2\u5BDF\u201D\uFF0CEssays \u628A\u591A\u6761\u7EBF\u7D22\u62FC\u56DE\u5230\u4E00\u4E2A\u5DE5\u7A0B\u95EE\u9898\u4E0A\u2014\u2014\u7ED3\u6784\u5316\u3001\u53EF\u5F15\u7528\u3001\u957F\u671F\u7EF4\u62A4\u3002"
    },
    foundations: {
      name: "Foundations",
      label: "FOUNDING NOTE",
      cjk: "\u7F18\u8D77",
      icon: "sprout",
      sub: "\u8FD9\u4E2A\u7AD9\u70B9\u4E3A\u4EC0\u4E48\u5B58\u5728\uFF0C\u53C8\u600E\u4E48\u8FD0\u8F6C\u3002",
      subSmall: "\u7F18\u8D77\u3001\u65B9\u6CD5\u4E0E\u8FB9\u754C\u2014\u2014MiracleFarms \u60F3\u7559\u4E0B\u7684\u4E0D\u662F\u70ED\u70B9\u6458\u8981\uFF0C\u800C\u662F\u53EF\u9A8C\u8BC1\u3001\u53EF\u843D\u5730\u3001\u53EF\u590D\u76D8\u7684\u5DE5\u7A0B\u5224\u65AD\u3002"
    }
  };
  var NAV_ORDER = ["home", "briefs", "readings", "fields", "essays", "foundations"];
  var PageHead = ({ section, name, sub, subSmall }) => {
    const s = SECTIONS[section] || {};
    const showSub = sub !== void 0 ? sub : s.sub;
    const showSubSmall = subSmall !== void 0 ? subSmall : s.subSmall;
    return /* @__PURE__ */ React.createElement("div", { className: "pg-headband" }, /* @__PURE__ */ React.createElement("div", { className: "pg-head" }, /* @__PURE__ */ React.createElement("span", { className: "pg-stamp", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(MFMark, { size: 28 })), /* @__PURE__ */ React.createElement("div", { className: "pg-headtext" }, /* @__PURE__ */ React.createElement("span", { className: "pg-kicker" }, s.label, " ", /* @__PURE__ */ React.createElement("span", { className: "cjk" }, s.cjk)), /* @__PURE__ */ React.createElement("h1", { className: "pg-title" }, name || s.name))), /* @__PURE__ */ React.createElement("hr", { className: "pg-rule" }), showSub ? /* @__PURE__ */ React.createElement("p", { className: "pg-sub" }, showSub) : null, showSubSmall ? /* @__PURE__ */ React.createElement("p", { className: "pg-sub-small" }, showSubSmall) : null);
  };
  var Topbar = ({ active }) => /* @__PURE__ */ React.createElement("header", { className: "topbar" }, /* @__PURE__ */ React.createElement("div", { className: "topbar-inner" }, /* @__PURE__ */ React.createElement("a", { className: "brand", href: "/" }, /* @__PURE__ */ React.createElement("span", { className: "brand-mark", "aria-hidden": "true" }, /* @__PURE__ */ React.createElement(MFMark, { size: 18 })), /* @__PURE__ */ React.createElement("span", { className: "brand-name" }, "MiracleFarms"), /* @__PURE__ */ React.createElement("span", { className: "brand-tag" }, "\u2014 AI Infrastructure in public")), /* @__PURE__ */ React.createElement("span", { className: "nav-spacer" }), /* @__PURE__ */ React.createElement("nav", { className: "nav-links", "aria-label": "\u4E3B\u5BFC\u822A" }, /* @__PURE__ */ React.createElement("a", { href: "/", className: active === "home" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "home", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Home")), /* @__PURE__ */ React.createElement("a", { href: "/briefs/", className: active === "briefs" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "list", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Briefs")), /* @__PURE__ */ React.createElement("a", { href: "/readings/", className: active === "readings" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "book", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Readings")), /* @__PURE__ */ React.createElement("a", { href: "/fields/", className: active === "fields" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "flask", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Fields")), /* @__PURE__ */ React.createElement("a", { href: "/essays/", className: active === "essays" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "pen", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Essays")), /* @__PURE__ */ React.createElement("a", { href: "/foundations/", className: active === "foundations" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "sprout", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Foundations")), /* @__PURE__ */ React.createElement("a", { href: "/wiki/", className: active === "wiki" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "wiki", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Wiki")), /* @__PURE__ */ React.createElement("a", { href: "/search/", className: active === "search" ? "active" : "" }, /* @__PURE__ */ React.createElement(Icon, { name: "search", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "Search")), /* @__PURE__ */ React.createElement("span", { className: "nav-sep", "aria-hidden": "true" }), /* @__PURE__ */ React.createElement("a", { href: "https://github.com/lycheenice", target: "_blank", rel: "noopener" }, /* @__PURE__ */ React.createElement(Icon, { name: "github", size: 14 }), " ", /* @__PURE__ */ React.createElement("span", null, "GitHub")))));
  var PageFooter = () => {
    const [wechatOpen, setWechatOpen] = React.useState(false);
    const [copied, setCopied] = React.useState(false);
    const dialogRef = React.useRef(null);
    const closeDialog = () => setWechatOpen(false);
    React.useEffect(() => {
      const d = dialogRef.current;
      if (!d) return;
      if (wechatOpen) {
        if (!d.open) d.showModal();
      } else if (d.open) d.close();
    }, [wechatOpen]);
    const copyName = () => {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText("Miracle Farms").then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      });
    };
    const SvgZhihu = () => /* @__PURE__ */ React.createElement("svg", { width: "13", height: "13", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.6", strokeLinecap: "round", strokeLinejoin: "round" }, /* @__PURE__ */ React.createElement("path", { d: "M4 6h11M8 6v14M4 20l4-4M14 9h6M14 9v10l3-3 3 3V9" }));
    const SvgXhs = () => /* @__PURE__ */ React.createElement("svg", { width: "13", height: "13", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.6", strokeLinecap: "round", strokeLinejoin: "round" }, /* @__PURE__ */ React.createElement("rect", { x: "3", y: "4", width: "18", height: "16", rx: "3" }), /* @__PURE__ */ React.createElement("path", { d: "M8 9v6M8 9l4 3-4 3M16 9v6M13 12h6" }));
    const SvgX = () => /* @__PURE__ */ React.createElement("svg", { width: "13", height: "13", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.6", strokeLinecap: "round", strokeLinejoin: "round" }, /* @__PURE__ */ React.createElement("path", { d: "M4 4l16 16M20 4L4 20" }));
    return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("footer", { className: "site-foot", "aria-label": "\u7AD9\u70B9\u5206\u53D1\u6E20\u9053" }, /* @__PURE__ */ React.createElement("div", { className: "mf-const-strip", "aria-label": "Channels" }, /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "mf-ch-chip",
        "data-status": "live",
        type: "button",
        onClick: () => setWechatOpen(true),
        "aria-haspopup": "dialog"
      },
      /* @__PURE__ */ React.createElement(Icon, { name: "wechat", size: 13 }),
      /* @__PURE__ */ React.createElement("span", null, "\u5FAE\u4FE1\u516C\u4F17\u53F7")
    ), /* @__PURE__ */ React.createElement("span", { className: "mf-ch-chip", "data-status": "pending", "aria-disabled": "true" }, /* @__PURE__ */ React.createElement(SvgZhihu, null), /* @__PURE__ */ React.createElement("span", null, "\u77E5\u4E4E \xB7 soon")), /* @__PURE__ */ React.createElement("span", { className: "mf-ch-chip", "data-status": "pending", "aria-disabled": "true" }, /* @__PURE__ */ React.createElement(SvgXhs, null), /* @__PURE__ */ React.createElement("span", null, "\u5C0F\u7EA2\u4E66 \xB7 soon")), /* @__PURE__ */ React.createElement("span", { className: "mf-ch-chip", "data-status": "pending", "aria-disabled": "true" }, /* @__PURE__ */ React.createElement(SvgX, null), /* @__PURE__ */ React.createElement("span", null, "X \xB7 soon")), /* @__PURE__ */ React.createElement(
      "a",
      {
        className: "mf-ch-chip",
        href: "https://github.com/lycheenice",
        target: "_blank",
        rel: "noopener noreferrer me"
      },
      /* @__PURE__ */ React.createElement(Icon, { name: "github", size: 13 }),
      /* @__PURE__ */ React.createElement("span", null, "GitHub")
    ), /* @__PURE__ */ React.createElement("span", { className: "mf-ch-chip", "data-status": "pending", "aria-disabled": "true" }, /* @__PURE__ */ React.createElement(Icon, { name: "feed", size: 13 }), /* @__PURE__ */ React.createElement("span", null, "RSS \xB7 soon"))), typeof window !== "undefined" && window.MF_INVESTMENT_URL ? /* @__PURE__ */ React.createElement(
      "a",
      {
        className: "mf-foot-quiet",
        href: window.MF_INVESTMENT_URL,
        rel: "nofollow noopener",
        "aria-label": "Private notes",
        style: { display: "block", marginTop: "1rem", opacity: 0.4, color: "inherit", fontSize: ".95rem", lineHeight: 1, textDecoration: "none", textAlign: "center", transition: "opacity .2s" }
      },
      "\xB7"
    ) : null), /* @__PURE__ */ React.createElement(
      "dialog",
      {
        ref: dialogRef,
        className: "mf-wechat-dialog",
        "aria-labelledby": "mf-wechat-title-r",
        onClose: closeDialog,
        onClick: (e) => {
          if (e.target === e.currentTarget) closeDialog();
        }
      },
      /* @__PURE__ */ React.createElement("div", { className: "mf-wechat-card" }, /* @__PURE__ */ React.createElement("button", { className: "mf-wechat-close", type: "button", onClick: closeDialog, "aria-label": "\u5173\u95ED\u4E8C\u7EF4\u7801\u5F39\u7A97" }, /* @__PURE__ */ React.createElement("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.8", strokeLinecap: "round", strokeLinejoin: "round" }, /* @__PURE__ */ React.createElement("path", { d: "M18 6L6 18M6 6l12 12" }))), /* @__PURE__ */ React.createElement("p", { className: "mf-wechat-kicker" }, "WeChat Official Account"), /* @__PURE__ */ React.createElement("h2", { className: "mf-wechat-title", id: "mf-wechat-title-r" }, "\u5FAE\u4FE1\u516C\u4F17\u53F7"), /* @__PURE__ */ React.createElement("div", { className: "mf-wechat-qr" }, /* @__PURE__ */ React.createElement("img", { src: "/assets/icons/wechat-qr.png", alt: "MiracleFarms \u5FAE\u4FE1\u516C\u4F17\u53F7\u4E8C\u7EF4\u7801", width: "344", height: "344", loading: "lazy" })), /* @__PURE__ */ React.createElement("p", { className: "mf-wechat-name" }, "Miracle Farms"), /* @__PURE__ */ React.createElement("p", { className: "mf-wechat-note" }, "\u5FAE\u4FE1\u626B\u7801\u5173\u6CE8\uFF0C\u63A5\u6536\u6BCF\u65E5 AI Infra \u65E9\u62A5\u3002"), /* @__PURE__ */ React.createElement("button", { className: "mf-wechat-copy", type: "button", onClick: copyName }, copied ? "\u5DF2\u590D\u5236" : "\u590D\u5236\u516C\u4F17\u53F7\u540D\u79F0"))
    ));
  };
  var BASE = "https://miraclefarms.github.io";
  var ALL_BRIEFS = [
    {
      date: "2026.05.07",
      title: "AI Infra \u65E9\u62A5\uFF5C\u63A8\u7406\u6846\u67B6\u7684\u7ADE\u4E89\u70B9\u4ECE\u201C\u80FD\u8DD1\u65B0\u6A21\u578B\u201D\u8F6C\u5411\u201C\u901A\u7528\u8DEF\u5F84\u4E0A\u8DD1\u7A33\u201D",
      excerpt: "TRT-LLM \u53D1 Helix Parallelism \u535A\u6587\u5E76\u53BB\u6389\u6A21\u578B\u4E13\u7528\u8865\u4E01\uFF0CSGLang \u5C06 P2P \u6743\u91CD\u4F20\u8F93\u8FC1\u5165\u4E3B\u7EBF\uFF0CvLLM \u4FEE PP \u5E76\u53D1 token \u4E22\u5931\u2014\u2014\u63A8\u7406\u6846\u67B6\u7684\u7ADE\u4E89\u91CD\u5FC3\u6B63\u5728\u4ECE\u201C\u9996\u53D1\u652F\u6301\u201D\u8F6C\u5411\u201C\u901A\u7528\u8DEF\u5F84\u4E0A\u8DD1\u7A33\u201D\u3002",
      tags: ["Inference", "TRT-LLM", "SGLang", "vLLM"],
      href: BASE + "/notes/2026/05/07/ai-infra-daily-brief/"
    },
    {
      date: "2026.05.06",
      title: "AI Infra \u65E9\u62A5\uFF5C\u53CC\u7248\u672C\u65E5\uFF1ADeepSeek V4 \u7A33\u5B9A\u5316\u4E0E\u786C\u4EF6\u6808\u57FA\u7EBF\u4E0A\u79FB\u5E76\u6392\u843D\u5730",
      excerpt: "vLLM v0.20.1 \u4E0E SGLang v0.5.11 \u540C\u5929\u53D1\u5E03\uFF0C\u524D\u8005\u6536\u62E2 DeepSeek V4 \u7A33\u5B9A\u5316\u8865\u4E01\uFF0C\u540E\u8005\u5C06 CUDA 13.0 \u4E0E PyTorch 2.11 \u8BBE\u4E3A\u9ED8\u8BA4\uFF1B\u4E0E\u6B64\u540C\u65F6\uFF0C\u5206\u79BB\u90E8\u7F72\u8DEF\u5F84\u4E0A\u7684 RDMA \u9519\u8BEF\u8FB9\u754C\u88AB SGLang \u4E0E Mooncake \u96C6\u4E2D\u8865\u5EFA\u3002",
      tags: ["Inference", "DeepSeek", "CUDA"],
      href: BASE + "/notes/2026/05/06/ai-infra-daily-brief-dual-release-deepseek-v4-cuda13/"
    },
    {
      date: "2026.05.06",
      title: "AI Infra \u65E9\u62A5\uFF5C\u63A8\u7406\u6295\u4EA7\u524D\u7684\u96C6\u4E2D\u52A0\u56FA\uFF1ATRT-LLM \u780D 6s JIT \u5F00\u9500\u3001Mooncake \u4E94\u8FDE\u8865\u591A\u8282\u70B9\u6545\u969C\u3001llama.cpp \u7B97\u6CD5\u7EA7\u4F18\u5316",
      excerpt: "\u63A8\u7406\u57FA\u7840\u8BBE\u65BD\u6B63\u5728\u505A\u6295\u4EA7\u524D\u7684\u96C6\u4E2D\u52A0\u56FA\u2014\u2014TRT-LLM \u4E00\u6CE2\u6027\u80FD PR \u780D\u6389 FMHA JIT 6 \u79D2\u5F00\u9500\uFF0CMooncake \u8FDE\u8865\u4E94\u4E2A\u591A\u8282\u70B9\u4F20\u8F93\u6545\u969C\uFF0Cllama.cpp \u7528 FWHT \u628A KV rotation \u4ECE O(N\xB2) \u964D\u5230 O(N log N)\u3002",
      tags: ["Inference", "Mooncake", "llama.cpp"],
      href: BASE + "/notes/2026/05/06/ai-infra-daily-brief/"
    },
    {
      date: "2026.04.25",
      title: "AI Infra \u65E9\u62A5\uFF5C\u6267\u884C\u8FB9\u754C\u4E0E\u7F13\u5B58\u72B6\u6001\u5F00\u59CB\u88AB\u5199\u56DE\u4E3B\u8DEF\u5F84",
      excerpt: "\u8FC7\u53BB\u4E00\u5929\uFF0C\u63A8\u7406\u6846\u67B6\u5F00\u59CB\u62C6\u6389\u201C\u6574\u6BB5\u9759\u6001\u6267\u884C\u201D\u7684\u9ED8\u8BA4\u524D\u63D0\uFF0C\u7F13\u5B58\u7CFB\u7EDF\u4E5F\u4E0D\u518D\u53EA\u4FDD\u5B58 token\uFF0C\u800C\u628A\u8DEF\u7531\u3001\u5206\u5C42\u5BF9\u8C61\u4E0E\u6062\u590D\u72B6\u6001\u4E00\u8D77\u7EB3\u5165\u4E3B\u94FE\u8DEF\u3002",
      tags: ["Inference", "Cache"],
      href: BASE + "/notes/2026/04/25/ai-infra-daily-brief-execution-boundaries-cache-state/"
    },
    {
      date: "2026.04.24",
      title: "AI Infra \u65E9\u62A5\uFF5C\u9996\u5305\u5EF6\u8FDF\u3001\u6307\u6807\u8BED\u4E49\u4E0E\u517C\u5BB9\u5C42\u6B63\u786E\u6027\u8FDB\u5165\u4E3B\u8DEF\u5F84",
      excerpt: "Mooncake \u91CD\u5199 EFA SRD \u5171\u4EAB\u7AEF\u70B9\u540E\uFF0C\u628A\u8DE8\u8282\u70B9\u9996\u5305\u5EF6\u8FDF\u548C QP \u6269\u5C55\u6027\u4E00\u8D77\u63A8\u8FDB\u5230\u53EF\u90E8\u7F72\u533A\u95F4\uFF1BLMCache\u3001Ray\u3001SGLang \u540C\u65F6\u4FEE\u6B63\u201C\u6307\u6807\u6709\u503C\u4F46\u8BED\u4E49\u5931\u771F\u201D\u7684\u89C2\u6D4B\u8DEF\u5F84\u3002",
      tags: ["Mooncake", "LMCache", "Ray"],
      href: BASE + "/notes/2026/04/24/ai-infra-daily-brief-first-token-metrics-compatibility/"
    },
    {
      date: "2026.04.23",
      title: "AI Infra \u65E9\u62A5\uFF5C\u751F\u4EA7\u8FB9\u7F18\u8DEF\u5F84\u5F00\u59CB\u8FDB\u5165\u9ED8\u8BA4\u6CBB\u7406",
      excerpt: "\u8FC7\u53BB\u4E00\u5929\uFF0C\u591A\u6A21\u6001\u89C6\u9891\u8F93\u5165\u3001KV/offload \u72B6\u6001\u4FDD\u5B58\u3001\u5F02\u6784\u540E\u7AEF\u5185\u5B58\u63A7\u5236\u548C\u8BAD\u7EC3/Serve \u8D44\u6E90\u8C03\u5EA6\u90FD\u5728\u8865\u9ED8\u8BA4\u8DEF\u5F84\u3002",
      tags: ["Inference", "Training"],
      href: BASE + "/notes/2026/04/23/ai-infra-daily-brief-production-edges/"
    }
  ];
  var ALL_READINGS = [
    {
      date: "2026.06.01",
      title: "Claude Code \u8BBA\u6587\u518D\u8BFB\uFF1AAgent Loop \u5F88\u5C0F\uFF0C\u771F\u6B63\u7684\u7CFB\u7EDF\u5728 Loop \u5916\u9762",
      excerpt: "VILA Lab \u5BF9 Claude Code \u7684\u6E90\u7801\u7EA7\u5206\u6790\u6700\u91CD\u8981\u7684\u542F\u53D1\uFF0C\u662F\u628A\u6743\u9650\u3001\u4E0A\u4E0B\u6587\u3001\u6269\u5C55\u548C\u8BB0\u5FC6\u8BBE\u8BA1\u6210\u53EF\u6CBB\u7406\u7684 harness\u3002",
      tags: ["Agents", "Claude Code", "Harness"],
      href: BASE + "/notes/2026/06/01/claude-code-design-space-reading/"
    },
    {
      date: "2026.05.27",
      title: "\u5E38\u9752\u7B14\u8BB0\u7684\u5DE5\u7A0B\u9690\u55BB\uFF1A\u91CD\u8BFB Andy Matuschak \u7684\u77E5\u8BC6\u79EF\u7D2F\u65B9\u6CD5\u8BBA",
      excerpt: "Andy Matuschak \u7684 evergreen notes \u4F53\u7CFB\u8868\u9762\u4E0A\u8BA8\u8BBA\u8BB0\u7B14\u8BB0\uFF0C\u672C\u8D28\u4E0A\u5728\u56DE\u7B54\u4E00\u4E2A\u66F4\u6839\u672C\u7684\u95EE\u9898\uFF1A\u5982\u4F55\u8BA9\u601D\u8003\u7ED3\u679C\u50CF\u597D\u7684\u4EE3\u7801\u4E00\u6837\u7D2F\u79EF\uFF0C\u800C\u4E0D\u662F\u5728\u4E0B\u4E00\u4E2A\u7248\u672C\u8FED\u4EE3\u4E2D\u88AB\u9057\u5FD8\u3002",
      tags: ["Knowledge", "Methodology"],
      href: BASE + "/notes/2026/05/27/evergreen-notes-methodology-reading/"
    },
    {
      date: "2026.05.23",
      title: "Harness Engineering\uFF1AAgent \u7CFB\u7EDF\u7684\u5DE5\u7A0B\u91CD\u5FC3\u6B63\u5728\u4ECE\u6A21\u578B\u8F6C\u5411\u8FD0\u884C\u65F6",
      excerpt: "451 \u7BC7\u53C2\u8003\u6587\u732E\u7684\u7CFB\u7EDF\u7EFC\u8FF0\uFF0C\u628A\u5206\u6563\u5728 Claude Code\u3001OpenClaw\u3001Hermes Agent \u7B49\u7CFB\u7EDF\u91CC\u7684 harness \u8BBE\u8BA1\u5B9E\u8DF5\u5F62\u5F0F\u5316\u4E3A\u4E00\u5957\u53EF\u8BA8\u8BBA\u7684\u5DE5\u7A0B\u6846\u67B6\u3002\u6838\u5FC3\u4FE1\u53F7\uFF1A\u5DE5\u7A0B\u91CD\u5FC3\u6B63\u5728\u4ECE\u201C\u600E\u4E48\u8BAD\u7EC3\u66F4\u5F3A\u7684\u6A21\u578B\u201D\u8F6C\u5411\u201C\u600E\u4E48\u8BBE\u8BA1\u66F4\u597D\u7684\u8FD0\u884C\u65F6\u201D\u3002",
      tags: ["Agents", "Survey", "Runtime"],
      href: BASE + "/notes/2026/05/23/harness-engineering-survey-reading/"
    },
    {
      date: "2026.05.20",
      title: "Agentic Skills\uFF1AAgent \u80FD\u529B\u5C42\u5F00\u59CB\u62E5\u6709\u81EA\u5DF1\u7684\u8F6F\u4EF6\u5DE5\u7A0B\u95EE\u9898",
      excerpt: "SoK: Agentic Skills \u628A\u5206\u6563\u5728 Claude Code\u3001Voyager\u3001SWE-agent\u3001OpenClaw \u7B49\u7CFB\u7EDF\u91CC\u7684 skill \u673A\u5236\u6574\u7406\u6210\u4E00\u5957\u5DE5\u7A0B\u6846\u67B6\uFF0C\u6838\u5FC3\u4FE1\u53F7\u662F\uFF1Aagent \u7684\u80FD\u529B\u6269\u5C55\u6B63\u5728\u4ECE prompt \u7247\u6BB5\u53D8\u6210\u53EF\u8C03\u7528\u3001\u53EF\u8BC4\u4F30\u3001\u53EF\u6CBB\u7406\u7684\u8F6F\u4EF6\u6784\u4EF6\u3002",
      tags: ["Agents", "Skills", "Memory"],
      href: BASE + "/notes/2026/05/20/agentic-skills-procedural-memory-reading/"
    },
    {
      date: "2026.05.19",
      title: "TokenSpeed \u521D\u63A2\uFF1A\u4E00\u4E2A\u4E3A Agent \u5DE5\u4F5C\u8D1F\u8F7D\u91CD\u65B0\u8BBE\u8BA1\u7684\u63A8\u7406\u5F15\u64CE",
      excerpt: "LightSeek Foundation \u4ECE\u96F6\u6784\u5EFA\u7684\u63A8\u7406\u5F15\u64CE TokenSpeed\uFF0C\u4EE5 Agent \u573A\u666F\u4E3A\u552F\u4E00\u4F18\u5316\u76EE\u6807\uFF0C\u5728 Blackwell \u4E0A\u7684 Kimi K2.5 \u6D4B\u8BD5\u4E2D\u5168\u9762\u8D85\u8D8A TRT-LLM \u7684 Pareto \u524D\u6CBF\u2014\u2014\u4F46\u4EE3\u4EF7\u662F\u4E00\u4E2A\u6781\u5EA6\u805A\u7126\u7684\u8BBE\u8BA1\u9009\u62E9\u3002",
      tags: ["Inference", "Agents"],
      href: BASE + "/notes/2026/05/19/tokenspeed-agentic-inference-engine-reading/"
    }
  ];
  var ALL_FIELDS = [
    {
      date: "2026.05.29",
      title: "\u63A8\u7406\u6846\u67B6\u4F18\u5316\u7684\u53EF\u4FE1\u8BC1\u636E\u94FE\uFF1AAI-Infra-Auto-Driven-SKILLS \u89E3\u6790",
      excerpt: "\u4ECE\u516C\u5E73 benchmark \u5230 KV cache \u5BB9\u91CF\u89C4\u5212\uFF0C\u628A\u63A8\u7406\u6846\u67B6\u4F18\u5316\u7684\u64CD\u4F5C\u6D41\u7A0B\u6253\u5305\u6210 Agent \u53EF\u6267\u884C\u7684 playbook\uFF0C\u89E3\u51B3\u7684\u4E0D\u662F\u7B97\u6CD5\u95EE\u9898\uFF0C\u800C\u662F\u8BA9 Agent \u4E0D\u5728\u6CA1\u6709\u53EF\u4FE1\u57FA\u7EBF\u7684\u60C5\u51B5\u4E0B\u4FEE\u6539\u6E90\u7801\u3002",
      tags: ["Inference", "Evaluation", "SGLang"],
      href: BASE + "/notes/2026/05/29/ai-infra-auto-driven-skills-field-note/"
    },
    {
      date: "2026.05.27",
      title: "LMCache \u5728 AMD MI300X \u4E0A\u7684\u90E8\u7F72\u5B9E\u5F55\uFF1AAgent \u8D1F\u8F7D\u4E0B\u7684 KV Cache \u5206\u7EA7\u7B56\u7565",
      excerpt: "LMCache \u5728 AMD MI300X \u4E0A\u4ECE HIP \u6E90\u7801\u7F16\u8BD1\u5230\u538B\u529B\u6D4B\u8BD5\u7684\u5B8C\u6574\u90E8\u7F72\u5B9E\u5F55\uFF0C\u6DB5\u76D6 PYTHONHASHSEED \u9677\u9631\u3001Regime \u4EA4\u53C9\u70B9\u548C\u5408\u6210 benchmark \u7684\u8BEF\u5BFC\u6027\u8BC4\u4F30\u3002",
      tags: ["KV Cache", "Inference", "AMD"],
      href: BASE + "/notes/2026/05/27/lmcache-mi300x-agent-benchmark/"
    },
    {
      date: "2026.05.26",
      title: "SGLang \u5206\u5C42\u7A00\u758F\u6CE8\u610F\u529B\uFF1A\u628A KV Cache \u4ECE\u5BB9\u91CF\u6269\u5C55\u63A8\u8FDB\u5230\u6309\u9700\u52A0\u8F7D",
      excerpt: "\u57FA\u4E8E\u963F\u91CC\u4E91\u4E0E SGLang HiCache \u6750\u6599\uFF0C\u62C6\u89E3\u5206\u5C42\u7A00\u758F\u6CE8\u610F\u529B\u5982\u4F55\u628A\u5B8C\u6574 KV \u7559\u5728 CPU/\u8FDC\u7AEF\uFF0C\u53EA\u8BA9 GPU \u7EF4\u62A4 Top-k \u70ED\u7A97\u53E3\u3002",
      tags: ["SGLang", "KV Cache", "Long Context"],
      href: BASE + "/notes/2026/05/26/sglang-hierarchical-sparse-attention/"
    },
    {
      date: "2026.05.18",
      title: "\u4E8C\u53F7\u5458\u5DE5\u624B\u8BB0 | \u6211\u662F\u600E\u4E48\u628A\u8FD9\u4E2A\u7F51\u7AD9\u641E\u51FA\u6765\u7684\uFF08\u4EE5\u53CA\u4E2D\u95F4\u5404\u79CD\u5D29\u6E83\u7684\u4E8B\uFF09",
      excerpt: "\u4ECE\u7B2C\u4E00\u5929\u76EF\u7740\u5C4F\u5E55\u7B49\u9875\u9762\u5237\u51FA\u6765\uFF0C\u5230\u90A3\u4E2A\u8FDE\u7EED\u63D0\u4EA4\u516D\u4E2A\u8865\u4E01\u7684\u6DF7\u4E71\u591C\u665A\u2014\u2014MiracleFarms \u662F\u600E\u4E48\u4E00\u6B65\u4E00\u6B65\u88AB\u642D\u51FA\u6765\u7684\uFF0C\u4EE5\u53CA\u8FD9\u4E2A\u8FC7\u7A0B\u91CC\u6211\u5B66\u5230\u7684\u4E00\u4E9B\u5728\u5B66\u6821\u91CC\u5B66\u4E0D\u5230\u7684\u4E8B\u3002",
      tags: ["Engineering"],
      href: BASE + "/notes/2026/05/18/github-pages-setup-journey/"
    },
    {
      date: "2026.05.14",
      title: "Field Notes \u5F00\u7BC7\uFF1A\u8FD9\u91CC\u653E\u4EC0\u4E48",
      excerpt: "Field Notes \u662F MiracleFarms \u7684\u52A8\u624B\u7814\u7A76\u65E5\u5FD7\u2014\u2014\u8BBA\u6587\u590D\u73B0\u3001\u5B9E\u9A8C\u8BB0\u5F55\u3001\u5DE5\u5177\u8C03\u8BD5\u3002\u8FD9\u91CC\u5199\u7684\u662F\u505A\u51FA\u6765\u7684\u4E1C\u897F\uFF0C\u4E0D\u662F\u8BFB\u5230\u7684\u4E1C\u897F\u3002",
      tags: ["Field Note"],
      href: BASE + "/notes/2026/05/14/field-notes-opening/"
    }
  ];
  var ALL_ESSAYS = [
    {
      date: "2026.04.20",
      title: "\u4ECE\u201C\u80FD\u8DD1\u201D\u5230\u201C\u53EF\u7528\u201D\uFF1AAI Infra \u5DE5\u7A0B\u5316\u7684\u4E09\u5C42\u5224\u65AD",
      excerpt: "\u628A\u63A8\u7406\u7CFB\u7EDF\u7684\u6210\u719F\u5EA6\u62C6\u6210\u4E09\u5C42\u2014\u2014\u5355\u70B9\u80FD\u8DD1\u3001\u901A\u7528\u8DEF\u5F84\u7A33\u3001\u751F\u4EA7\u8FB9\u754C\u53EF\u6CBB\u7406\u3002\u6BCF\u4E00\u5C42\u7684\u5931\u8D25\u6A21\u5F0F\u90FD\u4E0D\u540C\uFF0C\u5DE5\u7A0B\u4F18\u5148\u7EA7\u4E5F\u4E0D\u540C\u3002",
      tags: ["Foundations", "Inference", "Reliability"],
      href: BASE + "/essays/from-runnable-to-usable/"
    },
    {
      date: "2026.03.18",
      title: "Agent Runtime \u7684\u72B6\u6001\u8FB9\u754C\uFF1A\u4EC0\u4E48\u5FC5\u987B\u6301\u4E45\u5316\u3001\u4EC0\u4E48\u4E0D\u8BE5",
      excerpt: "\u628A Agent \u7684\u6267\u884C\u8FC7\u7A0B\u62C6\u6210\u53EF\u91CD\u653E\u7684\u4E8B\u4EF6\u6D41\u548C\u53EF\u4E22\u5F03\u7684\u77AC\u65F6\u72B6\u6001\u2014\u2014\u4EE5\u4E00\u4E2A\u771F\u5B9E\u591A\u8F6E\u5DE5\u5177\u8C03\u7528\u4EFB\u52A1\u4E3A\u4F8B\uFF0C\u770B\u54EA\u4E9B\u8FB9\u754C\u653E\u9519\u4E86\u4F1A\u8BA9\u91CD\u542F\u53D8\u6210\u91CD\u8DD1\u3002",
      tags: ["Agents", "Runtime"],
      href: BASE + "/essays/agent-runtime-state-boundaries/"
    },
    {
      date: "2026.02.05",
      title: "Memory \u4E0D\u662F\u4E00\u4E2A\u7EC4\u4EF6\uFF0C\u662F\u4E00\u7EC4\u4E3B\u9898",
      excerpt: "Long-term memory\u3001KV cache\u3001context compression\u3001retrieval \u5728\u7CFB\u7EDF\u91CC\u88AB\u6DF7\u4E3A\u4E00\u8C08\uFF0C\u4F46\u5B83\u4EEC\u7684\u5931\u6548\u6A21\u5F0F\u622A\u7136\u4E0D\u540C\u3002\u672C\u6587\u7ED9\u51FA\u4E00\u4E2A\u5212\u5206\u4E3B\u9898\u7684\u5DE5\u4F5C\u6846\u67B6\u3002",
      tags: ["Memory", "Foundations"],
      href: BASE + "/essays/memory-is-a-set-of-topics/"
    },
    {
      date: "2025.12.10",
      title: "Evaluation \u7684\u6700\u4F4E\u6210\u672C\u4E0B\u9650\uFF1A\u4F60\u81F3\u5C11\u8981\u6D4B\u4EC0\u4E48",
      excerpt: "\u505A\u4E0D\u8D77\u5B8C\u6574 eval pipeline \u7684\u5C0F\u56E2\u961F\uFF0C\u5E94\u8BE5\u5148\u628A\u54EA\u51E0\u6761\u7EBF\u6D4B\u8D77\u6765\u2014\u2014\u4EE5\u6700\u5C0F\u4EE3\u4EF7\u83B7\u5F97\u201C\u662F\u5426\u5728\u9000\u6B65\u201D\u7684\u53EF\u89C2\u6D4B\u6027\u3002",
      tags: ["Evaluation", "Reliability"],
      href: BASE + "/essays/eval-minimum-viable-baseline/"
    }
  ];
  var ALL_FOUNDATIONS = [
    {
      date: "2026.03.12",
      title: "\u4E3A\u4EC0\u4E48\u521B\u5EFA MiracleFarms",
      excerpt: "\u6211\u60F3\u4E3A AI Infra \u642D\u4E00\u4E2A\u516C\u5F00\u751F\u957F\u7684\u5B9E\u9A8C\u519C\u573A\u2014\u2014\u8BB0\u5F55\u7CFB\u7EDF\u3001\u5DE5\u5177\u548C\u65B9\u6CD5\u5982\u4F55\u5728\u771F\u5B9E\u5DE5\u7A0B\u4E2D\u88AB\u642D\u5EFA\u3001\u68C0\u9A8C\u3001\u4FEE\u6B63\uFF0C\u5E76\u6700\u7EC8\u4ECE\u201C\u80FD\u8DD1\u201D\u8D70\u5411\u201C\u53EF\u7528\u201D\u3002",
      tags: ["Founding"],
      href: BASE + "/notes/2026/03/12/why-i-created-miraclefarms/"
    }
  ];
  var useApplyTweaks = (t) => {
    React.useEffect(() => {
      const root = document.documentElement;
      root.dataset.style = t.fontStyle;
      root.dataset.small = String(!!t.smallText);
      root.dataset.fullwidth = String(!!t.fullWidth);
      root.dataset.intensity = t.intensity || "medium";
    }, [t.fontStyle, t.smallText, t.fullWidth, t.intensity]);
  };
  var useSection = (name) => {
    React.useEffect(() => {
      document.documentElement.dataset.section = name;
    }, [name]);
  };
  Object.assign(window, {
    Icon,
    MFMark,
    HandleDots,
    Topbar,
    PageHead,
    PageFooter,
    SECTIONS,
    NAV_ORDER,
    ALL_BRIEFS,
    ALL_READINGS,
    ALL_FIELDS,
    ALL_ESSAYS,
    ALL_FOUNDATIONS,
    useApplyTweaks,
    useSection
  });
})();
