---
layout: default
title: Privacy
description: MiracleFarms 的隐私与站点统计说明。
permalink: /privacy/
---

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Privacy</span>
  </div>

  <div class="pg-icon" aria-hidden="true">◎</div>
  <h1 class="pg-title">隐私与统计</h1>
  <p class="pg-sub">MiracleFarms 只做聚合级站点统计，不建立个人画像。</p>
  <p class="pg-sub-small">
    统计的目的，是理解哪些文章被阅读、读者大致从哪里来，以及哪些入口和设备需要优化。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">i</span>
    <div>
      <p>
        仓库默认配置关闭统计功能。部署时启用后，生产环境会加载 Cloudflare Web Analytics；本地预览不会发送统计事件。
      </p>
    </div>
  </aside>

  <hr class="divider">

  <p class="h2-sub">Scope</p>
  <h2 class="h2">会统计什么</h2>
  <p class="muted">
    统计数据包括页面路径、来源页面、国家或地区、设备类型、浏览器、操作系统、页面性能与访问时间等聚合信息。
  </p>

  <p class="h2-sub">Boundaries</p>
  <h2 class="h2">不会做什么</h2>
  <p class="muted">
    MiracleFarms 不在站点代码中保存原始 IP，不写入 cookie，不使用 localStorage 做跨页面追踪，不采集姓名、邮箱等主动身份信息，也不把统计数据用于广告定向。
  </p>

  <p class="h2-sub">Choice</p>
  <h2 class="h2">Do Not Track</h2>
  <p class="muted">
    如果浏览器启用了 Do Not Track，站点不会注入统计 beacon。你也可以使用浏览器的内容拦截、脚本拦截或隐私插件阻止第三方统计请求。
  </p>

  <footer class="pg-footer">
    <span>Privacy by default. · Metrics for maintenance, not profiling.</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>
