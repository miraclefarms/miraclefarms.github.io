---
layout: default
title: Readings
description: MiracleFarms Readings — AI Infrastructure 论文泛读与技术概览。
permalink: /readings/
---

{% assign readings = site.posts | where: 'kind', 'reading' | sort: 'date' | reverse %}

{% assign all_tag_str = "" %}
{% for post in readings %}
  {% for tag in post.tags %}
    {% assign all_tag_str = all_tag_str | append: "," | append: tag %}
  {% endfor %}
{% endfor %}
{% assign tag_list = all_tag_str | split: "," | uniq | sort %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Readings</span>
  </div>

  <div class="pg-icon" aria-hidden="true">🌿</div>
  <h1 class="pg-title">Readings</h1>
  <p class="pg-sub">论文泛读与技术概览。</p>
  <p class="pg-sub-small">
    读别人写的东西，整理成自己能用的知识——论文解读、系统概览、基准测试分析。比 Brief 深，比 Essay 轻。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">🔬</span>
    <div>
      <p>
        共 <strong>{{ readings.size }}</strong> 篇 · 覆盖 <strong>Inference / KV Cache / Agents / Evaluation</strong>。每篇聚焦一篇论文或一个系统，读完能知道它是什么、为什么重要、值不值得深挖。
      </p>
    </div>
  </aside>

  <div class="filterbar" role="toolbar" aria-label="筛选与视图" id="readings-filterbar">
    <span class="lbl">Filter:</span>
    <button class="fchip on" data-tag="" type="button">
      全部 <span class="count">{{ readings.size }}</span>
    </button>
    {% for tag in tag_list %}
    {% if tag != "" %}
    {% assign t_count = readings | where_exp: "p", "p.tags contains tag" | size %}
    <button class="fchip" data-tag="{{ tag }}" type="button">
      {{ tag }} <span class="count">{{ t_count }}</span>
    </button>
    {% endif %}
    {% endfor %}
    <span style="flex:1"></span>
    <span class="viewtabs" role="tablist">
      <button class="on" data-view="list" type="button">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>
        List
      </button>
      <button data-view="table" type="button">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M3 10h18M3 15h18M9 4v16"/></svg>
        Table
      </button>
    </span>
  </div>

  <div id="readings-list-view">
    {% assign prev_year = "" %}
    {% for post in readings %}
      {% assign yr = post.date | date: "%Y" %}
      {% if yr != prev_year %}
        {% unless forloop.first %}</div></section>{% endunless %}
        {% assign yr_count = 0 %}
        {% for p in readings %}
          {% assign pyr = p.date | date: "%Y" %}
          {% if pyr == yr %}{% assign yr_count = yr_count | plus: 1 %}{% endif %}
        {% endfor %}
        <section class="reading-year-group" data-year="{{ yr }}">
        <div class="year-row">
          <span class="yr">{{ yr }}</span>
          <span class="ln"></span>
          <span class="yr-count" data-total="{{ yr_count }}">{{ yr_count }} 篇</span>
        </div>
        <div class="pagelist" role="list">
        {% assign prev_year = yr %}
      {% endif %}
      <a class="pagelink reading-item" href="{{ post.url | relative_url }}" role="listitem"
         data-tags="{{ post.tags | join: ',' }}">
        <span class="pl-handle" aria-hidden="true">⋮⋮</span>
        <span class="pl-ico" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
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

  <div id="readings-table-view" style="display:none">
    <table class="ntable" role="table">
      <thead>
        <tr>
          <th class="col-date" style="width:120px">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="1.5"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>
            </span>Date
          </th>
          <th class="col-title">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
            </span>Title
          </th>
          <th class="col-tags" style="width:200px">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12L12 20l-8-8V4h8z"/><circle cx="8" cy="8" r="1.5"/></svg>
            </span>Tags
          </th>
        </tr>
      </thead>
      <tbody>
        {% for post in readings %}
        <tr class="reading-table-row" data-tags="{{ post.tags | join: ',' }}">
          <td class="col-date">{{ post.date | date: "%Y.%m.%d" }}</td>
          <td class="col-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></td>
          <td class="col-tags">
            {% for tag in post.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div id="readings-empty" class="empty" style="display:none">没有匹配的 reading。</div>

  <hr class="divider">

  <p class="h2-sub">Reading note</p>
  <h2 class="h2">关于 Readings 的定位</h2>
  <p class="muted">
    Readings 是 Brief 和 Essay 之间的中间层。Brief 捕捉"今天发生了什么"，Essay 给出"这件事意味着什么"，而 Readings 回答"这篇论文/这个系统是怎么工作的"。输入是别人写的东西，输出是自己能用的地图。
  </p>

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>

<script>
(function() {
  var chips      = document.querySelectorAll('#readings-filterbar .fchip');
  var viewBtns   = document.querySelectorAll('#readings-filterbar .viewtabs button');
  var listView   = document.getElementById('readings-list-view');
  var tableView  = document.getElementById('readings-table-view');
  var emptyMsg   = document.getElementById('readings-empty');
  var activeTag  = '';
  var activeView = 'list';

  function matchesTag(el) {
    if (!activeTag) return true;
    var tags = el.dataset.tags ? el.dataset.tags.split(',') : [];
    return tags.indexOf(activeTag) !== -1;
  }

  function applyFilter() {
    var listItems  = listView.querySelectorAll('.reading-item');
    var tableRows  = tableView.querySelectorAll('.reading-table-row');
    var yearGroups = listView.querySelectorAll('.reading-year-group');
    var visible = 0;

    listItems.forEach(function(el) {
      var show = matchesTag(el);
      el.style.display = show ? '' : 'none';
      if (show) visible++;
    });
    tableRows.forEach(function(el) {
      el.style.display = matchesTag(el) ? '' : 'none';
    });
    yearGroups.forEach(function(group) {
      var groupItems = Array.from(group.querySelectorAll('.reading-item'));
      var groupVisible = groupItems.filter(function(el) { return el.style.display !== 'none'; });
      group.style.display = groupVisible.length > 0 ? '' : 'none';
      var countEl = group.querySelector('.yr-count');
      if (countEl) {
        var total = countEl.dataset.total;
        countEl.textContent = activeTag
          ? groupVisible.length + ' / ' + total + ' 篇'
          : total + ' 篇';
      }
    });
    emptyMsg.style.display = visible === 0 ? '' : 'none';
  }

  chips.forEach(function(chip) {
    chip.addEventListener('click', function() {
      activeTag = this.dataset.tag;
      chips.forEach(function(c) { c.classList.toggle('on', c.dataset.tag === activeTag); });
      applyFilter();
    });
  });

  viewBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      activeView = this.dataset.view;
      viewBtns.forEach(function(b) { b.classList.toggle('on', b.dataset.view === activeView); });
      listView.style.display  = activeView === 'list'  ? '' : 'none';
      tableView.style.display = activeView === 'table' ? '' : 'none';
    });
  });
})();
</script>
