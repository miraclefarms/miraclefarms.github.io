---
layout: default
title: Fields
description: MiracleFarms Fields — AI Infra 实验与论文复现的动手研究日志。
permalink: /fields/
---

{% assign field_notes = site.posts | where: 'kind', 'field-note' | sort: 'date' | reverse %}

{% assign all_tag_str = "" %}
{% for post in field_notes %}
  {% for tag in post.tags %}
    {% assign all_tag_str = all_tag_str | append: "," | append: tag %}
  {% endfor %}
{% endfor %}
{% assign tag_list = all_tag_str | split: "," | uniq | sort %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Fields</span>
  </div>

  <div class="pg-icon" aria-hidden="true">🔬</div>
  <h1 class="pg-title">Fields</h1>
  <p class="pg-sub">实验与复现：动手做的 AI Infra 研究。</p>
  <p class="pg-sub-small">
    Readings 是读别人写的东西；Fields 是自己动手做的东西——论文复现、工具基准测试、调试记录。一手数据，不是综述。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">🔬</span>
    <div>
      <p>
        共 <strong>{{ field_notes.size }}</strong> 篇 · 覆盖 <strong>Inference / KV Cache / Agents / Evaluation</strong>。每篇聚焦一次实际动手的研究记录，可复现、可引用。
      </p>
    </div>
  </aside>

  <div class="filterbar" role="toolbar" aria-label="筛选与视图" id="fields-filterbar">
    <span class="lbl">Filter:</span>
    <button class="fchip on" data-tag="" type="button">
      全部 <span class="count">{{ field_notes.size }}</span>
    </button>
    {% for tag in tag_list %}
    {% if tag != "" %}
    {% assign t_count = field_notes | where_exp: "p", "p.tags contains tag" | size %}
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

  <div id="fields-list-view">
    {% assign prev_year = "" %}
    {% for post in field_notes %}
      {% assign yr = post.date | date: "%Y" %}
      {% if yr != prev_year %}
        {% unless forloop.first %}</div></section>{% endunless %}
        {% assign yr_count = 0 %}
        {% for p in field_notes %}
          {% assign pyr = p.date | date: "%Y" %}
          {% if pyr == yr %}{% assign yr_count = yr_count | plus: 1 %}{% endif %}
        {% endfor %}
        <section class="field-year-group" data-year="{{ yr }}">
        <div class="year-row">
          <span class="yr">{{ yr }}</span>
          <span class="ln"></span>
          <span class="yr-count" data-total="{{ yr_count }}">{{ yr_count }} 篇</span>
        </div>
        <div class="pagelist" role="list">
        {% assign prev_year = yr %}
      {% endif %}
      <a class="pagelink field-item" href="{{ post.url | relative_url }}" role="listitem"
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

  <div id="fields-table-view" style="display:none">
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
        {% for post in field_notes %}
        <tr class="field-table-row" data-tags="{{ post.tags | join: ',' }}">
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

  <div id="fields-empty" class="empty" style="display:none">没有匹配的 field note。</div>

  <hr class="divider">

  <p class="h2-sub">Field note</p>
  <h2 class="h2">关于 Fields 的定位</h2>
  <p class="muted">
    Fields 是 MiracleFarms 里最靠近原始数据的地方。Brief 捕捉"今天发生了什么"，Reading 消化"别人怎么写的"，Essay 给出"这意味着什么"——而 Fields 回答"我自己跑出来的结果是什么"。实验记录、复现笔记、基准测试、工具调试：这里写的东西都经过了亲手验证。
  </p>

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>

<script>
(function() {
  var chips      = document.querySelectorAll('#fields-filterbar .fchip');
  var viewBtns   = document.querySelectorAll('#fields-filterbar .viewtabs button');
  var listView   = document.getElementById('fields-list-view');
  var tableView  = document.getElementById('fields-table-view');
  var emptyMsg   = document.getElementById('fields-empty');
  var activeTag  = '';
  var activeView = 'list';

  function matchesTag(el) {
    if (!activeTag) return true;
    var tags = el.dataset.tags ? el.dataset.tags.split(',') : [];
    return tags.indexOf(activeTag) !== -1;
  }

  function applyFilter() {
    var listItems  = listView.querySelectorAll('.field-item');
    var tableRows  = tableView.querySelectorAll('.field-table-row');
    var yearGroups = listView.querySelectorAll('.field-year-group');
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
      var groupItems = Array.from(group.querySelectorAll('.field-item'));
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
