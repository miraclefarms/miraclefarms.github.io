---
layout: default
title: Essays
description: MiracleFarms Essays — AI Infrastructure 的长文与系统性文章。
permalink: /essays/
---

{% assign essays = site.posts | where: 'kind', 'essay' | sort: 'date' | reverse %}
{% assign field_notes = site.posts | where: 'kind', 'field-note' | sort: 'date' | reverse %}
{% assign all_posts = essays | concat: field_notes | sort: 'date' | reverse %}

{% comment %}kind values present: essay, field-note{% endcomment %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Essays</span>
  </div>

  <div class="pg-icon" aria-hidden="true">📓</div>
  <h1 class="pg-title">Essays</h1>
  <p class="pg-sub">长文：AI Infra 的系统性观察与判断。</p>
  <p class="pg-sub-small">
    相对于 Briefs 的"一日一观察"，Essays 把多条线索拼回到一个工程问题上——结构化、可引用、长期维护。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">📚</span>
    <div>
      <p>
        共 <strong>{{ all_posts.size }}</strong> 篇 · 主题覆盖 <strong>Inference / Agents / Memory / Evaluation / Reliability</strong>。每一篇都尝试在一个具体边界上给出可复盘的判断，而不是综述。
      </p>
    </div>
  </aside>

  <div class="filterbar" role="toolbar" aria-label="筛选" id="essays-filterbar">
    <span class="lbl">Filter:</span>
    <button class="fchip on" data-kind="" type="button">
      全部 <span class="count">{{ all_posts.size }}</span>
    </button>
    <button class="fchip" data-kind="essay" type="button">
      Essay <span class="count">{{ essays.size }}</span>
    </button>
    {% if field_notes.size > 0 %}
    <button class="fchip" data-kind="field-note" type="button">
      Field Note <span class="count">{{ field_notes.size }}</span>
    </button>
    {% endif %}
  </div>

  <div id="essays-list-view">
    {% assign prev_year = "" %}
    {% for post in all_posts %}
      {% assign yr = post.date | date: "%Y" %}
      {% if yr != prev_year %}
        {% unless forloop.first %}</div></section>{% endunless %}
        <section class="essay-year-group" data-year="{{ yr }}">
        <div class="year-row">
          <span class="yr">{{ yr }}</span>
          <span class="ln"></span>
          <span>{% assign yr_posts = all_posts | where_exp: "p", "p.date contains yr" %}{{ yr_posts.size }} 篇</span>
        </div>
        <div class="pagelist" role="list">
        {% assign prev_year = yr %}
      {% endif %}
      <a class="pagelink essay-item" href="{{ post.url | relative_url }}" role="listitem"
         data-kind="{{ post.kind }}">
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
            <span class="tag-kind">{{ post.category | default: post.kind }}</span>
            {% for tag in post.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
          </div>
        </div>
        <div class="pl-date">{{ post.date | date: "%Y.%m.%d" }}</div>
      </a>
    {% endfor %}
    </div></section>
  </div>

  <div id="essays-empty" class="empty" style="display:none">没有匹配的 essay。</div>

  <hr class="divider">

  <p class="h2-sub">Reading note</p>
  <h2 class="h2">关于 Essays 的写作</h2>
  <p class="muted">
    Essays 的更新频率明显低于 Briefs——它们通常需要数周到数月的观察沉淀，并经过一次以上的重写。我们倾向于把已经被多个 brief 反复触及的主题，整理成一篇可以长期被引用的 essay。
  </p>

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>

<script>
(function() {
  var chips    = document.querySelectorAll('#essays-filterbar .fchip');
  var listView = document.getElementById('essays-list-view');
  var emptyMsg = document.getElementById('essays-empty');
  var activeTag = '';

  function applyFilter() {
    var items      = listView.querySelectorAll('.essay-item');
    var yearGroups = listView.querySelectorAll('.essay-year-group');
    var visible = 0;

    items.forEach(function(el) {
      var show = !activeTag || el.dataset.kind === activeTag;
      el.style.display = show ? '' : 'none';
      if (show) visible++;
    });

    yearGroups.forEach(function(group) {
      var hasVisible = Array.from(group.querySelectorAll('.essay-item'))
        .some(function(el) { return el.style.display !== 'none'; });
      group.style.display = hasVisible ? '' : 'none';
    });

    emptyMsg.style.display = visible === 0 ? '' : 'none';
  }

  chips.forEach(function(chip) {
    chip.addEventListener('click', function() {
      activeTag = this.dataset.kind;
      chips.forEach(function(c) { c.classList.toggle('on', c.dataset.kind === activeTag); });
      applyFilter();
    });
  });
})();
</script>
