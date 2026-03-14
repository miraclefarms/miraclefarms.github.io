---
layout: default
title: Essays - MiracleFarms
description: AI Infrastructure 深度长文
permalink: /essays/
---

<section class="section">
  <div class="section-heading">
    <p class="eyebrow">Essays</p>
    <h2>深度长文</h2>
    <p class="section-intro">适合结构化慢读与系统理解。每篇长文尝试拆开一个常被简化的系统问题。</p>
  </div>

  <div class="post-list refined-post-list">
    {% assign essays = site.posts | where: 'kind', 'essay' | sort: 'date' | reverse %}
    {% for post in essays %}
    <article class="post-card refined-post-card post-kind-essay">
      <div class="post-card-topline">
        <span class="post-category">Essay</span>
        <span class="post-card-meta">{{ post.date | date: "%Y.%m.%d" }}</span>
      </div>
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      <p>{{ post.intro }}</p>
      <a class="text-link" href="{{ post.url | relative_url }}">阅读全文 →</a>
    </article>
    {% endfor %}
  </div>
</section>
