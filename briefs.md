---
layout: default
title: Briefs
description: MiracleFarms Briefs — AI Infra 早报与短文集合，按日期与主题归档。
permalink: /briefs/
---

{% assign briefs = site.posts | where: 'kind', 'brief' | sort: 'date' | reverse %}
{% assign all_tag_str = "" %}
{% for post in briefs %}
  {% for tag in post.tags %}
    {% assign all_tag_str = all_tag_str | append: "," | append: tag %}
  {% endfor %}
{% endfor %}
{% assign tag_list = all_tag_str | split: "," | uniq | sort %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Briefs</span>
  </div>

  <div class="pg-icon" aria-hidden="true">📰</div>
  <h1 class="pg-title">Briefs</h1>
  <p class="pg-sub">AI Infra 的日常观测与短判断。</p>
  <p class="pg-sub-small">
    每一条都聚焦在一个具体变化上——一个 PR、一次发版、一个被悄悄写回主路径的默认行为——并尝试把它放回到更大的趋势里。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">🗒️</span>
    <div>
      <p>
        共 <strong>{{ briefs.size }}</strong> 条 · 最新 <strong>{{ briefs.first.date | date: "%Y.%m.%d" }}</strong> · 按日期倒序排列。
      </p>
    </div>
  </aside>

  <div class="filterbar" role="toolbar" aria-label="筛选与视图" id="briefs-filterbar">
    <span class="lbl">Filter:</span>
    <button class="fchip on" data-tag="" type="button">
      全部 <span class="count">{{ briefs.size }}</span>
    </button>
    {% for tag in tag_list %}
    {% if tag != "" %}
    {% assign t_count = briefs | where_exp: "p", "p.tags contains tag" | size %}
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

  <div id="briefs-list-view">
    {% assign prev_year = "" %}
    {% for post in briefs %}
      {% assign yr = post.date | date: "%Y" %}
      {% if yr != prev_year %}
        {% unless forloop.first %}</div></section>{% endunless %}
        <section class="brief-year-group" data-year="{{ yr }}">
        <div class="year-row">
          <span class="yr">{{ yr }}</span>
          <span class="ln"></span>
          <span>{% assign yr_posts = briefs | where_exp: "p", "p.date contains yr" %}{{ yr_posts.size }} 条</span>
        </div>
        <div class="pagelist" role="list">
        {% assign prev_year = yr %}
      {% endif %}
      <a class="pagelink brief-item" href="{{ post.url | relative_url }}" role="listitem"
         data-tags="{{ post.tags | join: ',' }}" data-year="{{ yr }}">
        <span class="pl-handle" aria-hidden="true">⋮⋮</span>
        <span class="pl-ico" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M9 13h7M9 17h5"/></svg>
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

  <div id="briefs-table-view" style="display:none">
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
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M9 13h7M9 17h5"/></svg>
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
        {% for post in briefs %}
        <tr class="brief-table-row" data-tags="{{ post.tags | join: ',' }}">
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

  <div id="briefs-empty" class="empty" style="display:none">没有匹配的 brief。试试清除筛选。</div>

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>

<script>
(function() {
  var chips      = document.querySelectorAll('#briefs-filterbar .fchip');
  var viewBtns   = document.querySelectorAll('#briefs-filterbar .viewtabs button');
  var listView   = document.getElementById('briefs-list-view');
  var tableView  = document.getElementById('briefs-table-view');
  var emptyMsg   = document.getElementById('briefs-empty');
  var activeTag  = '';
  var activeView = 'list';

  function matchesTag(el) {
    if (!activeTag) return true;
    var tags = el.dataset.tags ? el.dataset.tags.split(',') : [];
    return tags.indexOf(activeTag) !== -1;
  }

  function applyFilter() {
    var listItems  = listView.querySelectorAll('.brief-item');
    var tableRows  = tableView.querySelectorAll('.brief-table-row');
    var yearGroups = listView.querySelectorAll('.brief-year-group');
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
      var hasVisible = Array.from(group.querySelectorAll('.brief-item'))
        .some(function(el) { return el.style.display !== 'none'; });
      group.style.display = hasVisible ? '' : 'none';
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
