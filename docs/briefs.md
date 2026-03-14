---
layout: default
title: Briefs - MiracleFarms
description: AI Infrastructure 简报栏目
permalink: /briefs/
---

<section class="section">
  <div class="section-heading">
    <p class="eyebrow">Briefs</p>
    <h2>简报栏目</h2>
    <p class="section-intro">快速浏览与获取高信号更新。每篇简报尽量回答一个值得继续追下去的问题。</p>
  </div>

  <div class="post-list refined-post-list">
    {% assign briefs = site.posts | where: 'kind', 'brief' | sort: 'date' | reverse %}
    {% for post in briefs %}
    <article class="post-card refined-post-card post-kind-brief">
      <div class="post-card-topline">
        <span class="post-category">Brief</span>
        <span class="post-card-meta">{{ post.date | date: "%Y.%m.%d" }}</span>
      </div>
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      <p>{{ post.intro }}</p>
      <a class="text-link" href="{{ post.url | relative_url }}">阅读全文 →</a>
    </article>
    {% endfor %}
  </div>
</section>
