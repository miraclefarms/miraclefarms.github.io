# 微信公众号格式规范

微信公众号版不是新的文章类型，而是同一篇 brief / essay 的渠道改写版。

但这里有一个必须显式写出来的分叉：

- `brief -> 微信公众号` 仍然沿用日报路径，默认是短段落、低配图密度、日报配色，**正文保留日期行 `📅 YYYY-MM-DD`**。
- `essay -> 微信公众号技术长文` 虽然复用大部分调研、引用整理与素材筛选流程，但**必须走独立长文路径**：保留 GitHub.io 版 essay 的正文配图，写入 `wechat_variant: essay-longform`，并使用经典蓝主题；**技术长文不写日期行**，标题下方直接跟引导语 blockquote。

不要把技术长文误写成只有题图的 brief 模板。写作路径不分叉，后面的发布脚本就很容易走偏。

## 题图模板

- 微信公众号题图提示词不再直接内嵌在文章正文里，默认改由版本化模板文件统一管理：
  - `scripts/config/wechat-cover-prompt-templates.json`
- **Brief 默认模板**：`daily-morning-paper-v1`，会把文章内容简报和日期交给一张带折痕、照片、引语和版式设计的日报桌面照片；模板只使用内容简报作为语义来源，禁止把比例、版式或提示词元信息印成报纸正文
- 当前默认锁定这一套模板，除非 front matter 显式指定其他 `wechat_cover_prompt_template`
- **Essay 长文默认模板**：`book-on-desk-v1`，会把文章内容简报交给一本展开平放在桌面上的科技图书，页面包含与文章主题相关的排版设计、图表和引语，不体现具体时间
- 模板文件除了 prompt 文案，也负责声明生成参数（例如 `aspectRatio`）；发布脚本会直接读取它，避免“模板写竖图、接口还在发横图”的上下游错位
- Essay 长文如需使用 `book-on-desk-v1`，在 front matter 中写入 `wechat_cover_prompt_template: book-on-desk-v1`
- 如果未来确实需要按文章覆盖模板，再额外引入 front matter 字段；在那之前，写作者默认依赖模板文件即可

## 目标

- 保留 GitHub.io 版本的核心判断
- 改写成适合公众号阅读和编辑器粘贴的 Markdown
- 正文主体不使用超链接语法，但参考资料中保留完整 URL，方便编辑与读者回查
- `essay -> 微信公众号技术长文` 时，保留与 GitHub.io 版 essay 相同的正文配图，而不光是题图

## 输出路径

- 保存到：`/Users/lychee/mycode/miraclefarms.github.io/docs/wechat/YYYY-MM-DD-slug-wechat.md`
- 如果目录不存在，按需创建

---

## 文章结构模板（Brief 类型）

```markdown
---
title: 今日焦点：{核心主题描述}
wechat_variant: brief
intro: {一句话摘要，40-80 字，用于公众号摘要栏}
---

**📅 YYYY-MM-DD**

> {引导语：结合当天热点点明为什么现在值得读，不超过 90 字}

---

## {分类一}

**{条目标题}[N]** - {正文描述：当前问题 → 预期效果，结尾可打 **[持续更新]** 标签}

**{条目标题}[N]** 与 **{条目标题}[N]** - {合并描述多个相关条目}，属于 **[持续更新]**。

---

## {分类二}

...

---

> 一句话结论：**{全文最核心的判断，一句话，不超过 60 字}**

---

## 参考

[1] {条目标题}：{完整 URL}

[2] {条目标题}：{完整 URL}
```

### Brief 标题与导语

- **Brief 的标题写在 front matter 的 `title` 字段里**，正文 body 禁止重复标题，不写 `# 今日焦点...` 或其他 H1。这样草稿箱标题和正文不会出现同一个大标题。
- 题图下方的导语不是 intro 的压缩版，也不是标题改写。它必须结合当天热点，抓住读者已经关心的模型、框架或硬件窗口，例如 DeepSeek、GPT-OSS、Blackwell、新模型首发、新硬件量产、跨框架同日就位。
- 导语要回答“为什么今天值得读 / 为什么读者现在需要关心”：把技术变化和现实压力连起来，例如“新模型上线窗口正在压缩，今天这些 PR 决定 NVFP4 能不能从实验格式进入默认服务”。
- 避免“X 正在成为关键趋势”这类抽象句；优先使用具体名字、具体压力和具体后果。

### H2 分类名约定

Brief 常用分类（按需选用，不必全部出现）：

- `$推理侧$`
- `$训练侧$`
- `$生产部署侧$`
- `$应用侧$`
- `$工具链$`

H2 直接写分类名，**不加** `$` 包裹，**不使用**中文数字编号（`一、二、三、`是 GitHub.io 版的规范，公众号版不沿用）。

---

## 文章结构模板（Essay 类型）

Essay 的微信公众号版本是技术长文，不沿用 brief 的日报模板，**不写日期行**。这里要显式告诉发布链路：这是篇长文。

```markdown
---
author: Ethan
intro: {一句话摘要，40-80 字，用于公众号摘要}
wechat_variant: essay-longform
wechat_cover_prompt_template: book-on-desk-v1
source_url: https://miraclefarms.github.io/notes/YYYY/MM/DD/{slug}/
---
# {描述性中文标题}

> {引导语：一句话点明文章的核心判断，不超过 80 字}

---

## {第一节标题}

{正文段落}

![图 1 描述](../../assets/{post-slug}/fig-1-architecture.png)

*图 1：这张图解释正文里的哪个关键机制。*

{继续分析}

## {第二节标题}

{正文段落}

![图 2 描述](../../assets/{post-slug}/fig-2-benchmark-throughput.png)

*图 2：这张图支撑的是哪条实验或架构判断。*

{继续分析}

---

> 一句话结论：**{全文最核心的判断，一句话，不超过 60 字}**

---

## 参考

[1] {资料标题}：{完整 URL}

[2] {资料标题}：{完整 URL}
```

### Essay 长文的专用要求

- front matter 必须写 `wechat_variant: essay-longform`
- front matter 建议写 `author` 与 `intro`，让草稿箱摘要可直接复用
- **front matter 必须写 `source_url`**，填入 GitHub.io 对应文章的完整 URL（格式：`https://miraclefarms.github.io/notes/YYYY/MM/DD/{slug}/`）。发布脚本会把它写入草稿的 `content_source_url`，即公众号文章底部的"阅读原文"链接
- 题图提示词从 `scripts/config/wechat-cover-prompt-templates.json` 读取
- Essay 微信公众号技术长文默认使用 `book-on-desk-v1` 模板（16:9 横版，展开的科技图书平放在桌面上，页面内容与文章主题相关，不体现具体时间）；在 front matter 中写入 `wechat_cover_prompt_template: book-on-desk-v1`
- 题图的图片比例等生成参数也由同一模板文件声明，发布脚本按模板读取，不要在别处再硬编码一套
- **保留与 GitHub.io 版 essay 使用同一组配图**；如果 GitHub.io 版用了 3 张关键图，微信公众号技术长文也应保留这 3 张，而不是只留题图
- 正文配图优先复用 repo 内的本地图片路径，推荐写成相对 `docs/wechat/` 文件的路径，例如 `../../assets/{post-slug}/fig-1-architecture.png`
- **正文配图必须为 PNG / JPG / WEBP 格式**。微信 `uploadimg` 接口不接受 SVG。如果 GitHub.io 版原图是 SVG，必须先转成 PNG 再在微信版里引用。转换方法：`node -e "require('sharp')('input.svg').png().toFile('output.png', (e,i)=>console.log(e||i))"`（`sharp` 已在 repo 的 npm 依赖中）
- 不要把 GitHub.io 站点的公开 URL 当成默认做法；技术长文路径应优先交给发布脚本处理本地图片上传
- Essay 的微信公众号技术长文使用**经典蓝**主题，不沿用 brief 的翡翠绿

---

## 正文规则

- 不使用 Markdown 链接，不使用 HTML anchor
- 正文 body 禁止重复标题：Brief / Essay 都不在正文里写 H1，标题只放在 front matter `title` 或公众号草稿标题中
- GitHub.io 版里的 `[[N]](url)` 或 `<a href="url">[N]</a>`，统一改写成纯文本引用 `[N]`
- 引用号直接跟在相关内容后即可，不要求放进粗体标题内
- 图可以保留；图注写法与 GitHub.io 版一致
- `brief -> 微信公众号` 与 `essay -> 微信公众号技术长文` 都允许题图，但 essay 长文不能停在题图层面
- 如果文章正文里没有显式题图 Markdown，发布脚本会根据模板文件自动生成并插入题图；这就是当前的默认路径

### Brief 条目写法

每个条目用一段写完：

```markdown
**{条目标题}[N]** - {当前问题是什么 → 这次变化做了什么 → 预期效果}，属于 **[持续更新]**。
```

- 相关联的两三个条目可以合并在一段，用“前者 / 后者”或“这组变化”连接
- 正文内容比 GitHub.io 版更口语、更短句，但不要写成营销文案
- 每节控制在 2-4 段，段落不要过长

### Essay 段落写法

- 维持 essay 的论证顺序，不要拆成日报式条目列表
- 每个 H2 下通常 2-5 段，图片应紧跟第一次深入分析该机制 / 实验的位置
- 可以适度收短句子，但不要把技术长文改成营销摘要
- 一篇 essay 的微信公众号技术长文通常保留 2-4 张图；如果 GitHub.io 版某张图明显承担关键论证，就不应在微信版里消失
- **不要使用 Markdown bullet list（`- ` 或 `* ` 列表）**。微信公众号编辑器对标准列表的渲染不稳定，可能出现缩进丢失、圆点符号缺失或整体格式错乱。需要并列展开的内容一律改用连续段落：使用"第一点……""第二点……""第三点……"的衔接句式，或用"首先……其次……最后……"的过渡语，将内容嵌入段落流中

### 持续更新标签

如果该条目在此前日报已出现过、今天是继续跟进，结尾打：`属于 **[持续更新]**。`

---

## 参考资料处理

章节名固定为 `## 参考`（不用 `## 参考资料` 或 `## 参考来源`）。

格式：`[N] {条目标题}：{完整 URL}`

- 冒号后直接接 URL，不加括号，不加来源类型说明
- 标题与 GitHub.io 版保持一致

示例：

```markdown
## 参考

[1] vLLM 在 MRV2 中引入 probabilistic rejection sampling：https://github.com/vllm-project/vllm/pull/35461

[2] SGLang 接入 Elastic NIXL-EP 通信路径：https://github.com/sgl-project/sglang/pull/19248

[3] KServe 为不受支持的 scaling 组合增加显式校验：https://github.com/kserve/kserve/pull/5212
```

---

## 改写节奏

- 比 GitHub.io 版更短句、更口语，但不要写成营销文案
- 开头两个 blockquote 之后直接进入正文，不再重复铺背景
- 如果原文有过密的引用或过长的技术铺垫，公众号版可以适当收束，只保留最支持判断的部分
- 结尾 `> 一句话结论：` blockquote 是必须有的，浓缩全天最重要的一个判断
- Essay 技术长文要保住“论证 + 配图 + 结论”的完整结构，不要因为迁移渠道就退化成纯摘要

---

## 自检

- 正文主体没有 Markdown 链接、HTML 链接
- URL 只出现在 `## 参考` 章节，格式为 `[N] 标题：URL`
- 没有把 URL 混进标题或图注
- Brief 的 H2 直接写分类名，没有 `$` 包裹，没有用中文数字编号
- Brief 正文 body 没有 H1，大标题只在 front matter `title` 中出现
- Brief 题图下方导语结合了当天热点，并回答了“为什么读者现在需要关心”
- 结尾有 `> 一句话结论：**...**` blockquote
- 改写后仍然保留原文的核心判断，而不是只剩新闻摘要
- 正文没有使用 Markdown bullet list（`- ` 或 `* `），并列内容已改用连续段落展开
- 如果是 essay 的微信公众号技术长文，是否已经写入 `wechat_variant: essay-longform`
- 如果是 essay 的微信公众号技术长文，front matter 是否写入了 `source_url`（GitHub.io 原文链接）？发布脚本依赖它生成"阅读原文"按钮
- 如果是 essay 的微信公众号技术长文，是否保留了与 GitHub.io 版 essay 相同的正文配图，而不光是题图
- 如果是 essay 的微信公众号技术长文，正文配图是否全部为 PNG / JPG / WEBP？SVG 需要用 `sharp` 先转成 PNG
- 如果是 essay 的微信公众号技术长文，是否明确走经典蓝主题
- 如果是 essay 的微信公众号技术长文，**标题下方没有日期行 `📅`**
