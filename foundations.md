---
layout: default
title: Foundations
description: MiracleFarms 的站点起源、方法论与基础文本。
permalink: /foundations/
---

{% assign founding_notes = site.posts | where: 'kind', 'founding-note' | sort: 'date' | reverse %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Foundations</span>
  </div>

  <div class="pg-icon" aria-hidden="true">🏡</div>
  <h1 class="pg-title">起点与方法</h1>
  <p class="pg-sub">MiracleFarms 的站点起源、写作方法与基础文本。</p>
  <p class="pg-sub-small">
    这里收录 MiracleFarms 的基础文本：站点为何建立、如何写作、长期关注什么，以及这套公开研究型写作方法背后的基本假设。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">🌾</span>
    <div>
      <p>
        共 <strong>{{ founding_notes.size }}</strong> 篇基础文本 · 适合第一次来到站点，或想理解它为何这样组织内容的读者。
      </p>
    </div>
  </aside>

  {% if founding_notes.size == 0 %}
  <div class="empty">暂无基础文本。</div>
  {% else %}
  <div class="pagelist" role="list">
    {% for post in founding_notes %}
    <a class="pagelink" href="{{ post.url | relative_url }}" role="listitem">
      <span class="pl-handle" aria-hidden="true">⋮⋮</span>
      <span class="pl-ico" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3V4z"/><path d="M5 17a3 3 0 013-3h11"/></svg>
      </span>
      <div class="pl-body">
        <div class="pl-title">{{ post.title }}</div>
        {% if post.intro %}
        <div class="pl-excerpt">{{ post.intro | truncate: 140 }}</div>
        {% endif %}
        <div class="pl-meta">
          <span class="tag">{{ post.category | default: 'Founding Note' }}</span>
        </div>
      </div>
      <div class="pl-date">{{ post.date | date: "%Y.%m.%d" }}</div>
    </a>
    {% endfor %}
  </div>
  {% endif %}

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>
