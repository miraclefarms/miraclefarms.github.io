# 字数与阅读时长统计 — 设计文档

**日期：** 2026-05-09  
**状态：** 已批准，待实现

---

## 需求

在每篇 essay / reading 文章的 `post-meta-row` 中，展示文章字数和预计阅读时长。brief / field-note / founding-note 不显示。

显示格式：`约 3,200 字 · 8 分钟`

---

## 方案选择

采用 **JavaScript 客户端计算**（方案 B），放弃 Jekyll Liquid `number_of_words`（对中文按空格切分，严重低估）。

理由：博客内容为中英混排技术文章，CJK 字符与英文单词需分别按不同速度计算阅读时间，Liquid 无法做到，客户端正则计数最准确。

---

## 实现细节

### 文件改动

| 文件 | 改动 |
|------|------|
| `_layouts/post.html` | 在 `post-meta-row` 末尾添加条件 chip + JS 计数逻辑 |
| `assets/css/site.css` | 添加 `.post-word-count-chip` 样式 |

### HTML（`_layouts/post.html`）

在 `post-meta-row` 关闭标签前插入：

```html
{% if page.kind == 'essay' or page.kind == 'reading' %}
<span id="post-word-count" class="post-word-count-chip"></span>
{% endif %}
```

### JavaScript（`_layouts/post.html` 的 `<script>` 块顶部）

```js
var wcEl = document.getElementById('post-word-count');
if (wcEl) {
  var body = document.getElementById('post-body-content');
  var text = body ? body.innerText : '';
  var cjk  = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
  var eng  = (text.replace(/[一-鿿㐀-䶿]/g, ' ')
                   .match(/\b[a-zA-Z0-9]+\b/g) || []).length;
  var mins = Math.max(1, Math.round(cjk / 400 + eng / 150));
  var total = cjk + eng;
  wcEl.textContent = '约 ' + total.toLocaleString('zh-CN') + ' 字 · ' + mins + ' 分钟';
}
```

**参数：**
- CJK 阅读速度：400 字/分钟（技术内容）
- 英文阅读速度：150 词/分钟（技术内容）
- 最短显示：1 分钟

### CSS（`assets/css/site.css`）

紧接在 `.post-updated-chip` 规则之后添加：

```css
.post-word-count-chip {
  display: inline-flex; align-items: center;
  color: var(--ink-faint);
  font-size: 13px; font-family: var(--font-default);
}
```

不加 chip 背景色，与日期/作者保持视觉一致，不抢眼。

---

## 边界情况

- `post-body-content` 不存在（非 post 页面）：`body` 为 null，`text` 为空字符串，`total` 为 0，`wcEl` 仍为空字符串，不显示任何内容。
- 极短文章（total < 某阈值）：`mins` 最小为 1，不会显示"0 分钟"。
- 纯英文 essay：CJK count 为 0，仅按英文词数计算。

---

## 不在范围内

- brief / field-note / founding-note 不显示字数
- 不做服务端预渲染（不修改 front matter 或 `_data/`）
- 不做"进度条"或滚动时更新的动态阅读进度
