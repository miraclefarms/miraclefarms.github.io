# MiracleFarms 首页改写计划（v1）

## 目标

本轮首页改写的目标不是重新定义站点，而是把已有内容组织得更清楚：

1. 让第一次进入的人在 3 秒内知道这是一个做什么的站点
2. 让首页阅读路径更连续，减少重复入口
3. 让每个内容入口都回答“为什么值得点进去”
4. 把方法说明与品牌隐喻分层放置，避免首屏过虚

## 当前首页的主要问题

基于 2026-03-14 当前版本，首页已经完成了“去重 latest feature”的收敛，但仍有以下问题：

1. **首页主结构仍然偏内容类型分栏**
   - 用户需要先理解 Brief / Essay / Foundation 的差异，才能开始浏览
   - 这更像归档页逻辑，不像首页逻辑

2. **首屏已经比之前清楚，但仍缺少更明确的站点定义**
   - 现在能看出是 AI Infra 写作站
   - 但“为什么值得继续读”还不够集中

3. **首页缺少一个明确的 Start Here 区块**
   - 没有给新访客一个低认知负担的入口选择

4. **Latest Writing / Briefs 的信息组织虽已收敛，但仍可继续提升**
   - 首页更适合展示“最近值得读的 3 篇”
   - 而不是继续强调内容类型分仓

5. **How We Work 缺失**
   - agent-assisted 是站点特征之一
   - 但目前首页没有把这种方法论转化为读者可感知的信任机制

6. **Farms 隐喻解释的位置仍需收敛**
   - 首页应该感受到这种气质
   - 但完整解释应放到 About 区块，而不是首屏展开

## 改写原则

### 原则 1：首屏先定义站点，再补气质

首屏顺序应是：
- 品牌名
- 一句站点定义
- 一句研究范围
- 一句轻量隐喻
- 明确 CTA

### 原则 2：首页按阅读路径组织，不按仓库结构组织

首页的逻辑应该是：
- 我是谁
- 你从哪里开始
- 最近有什么值得读
- 我长期研究什么
- 我如何工作
- MiracleFarms 这个名字是什么意思

而不是：
- Briefs
- Essays
- Foundations

### 原则 3：每个卡片都要回答“为什么值得点”

因此内容卡片至少需要：
- 标题
- 标签
- 日期
- 一句短摘要（两行以内）
- 清晰动作按钮

### 原则 4：少讲宏大愿景，多讲研究对象、方法与判断

首页重点应放在：
- 研究什么
- 怎么研究
- 最近写了什么
- 为什么值得读

## 目标首页结构

### 1. Hero

建议文案方向：A / C 中间态。

目标：
- 直接说明这是一个公开记录 AI Infrastructure 的研究写作站
- 补充 inference systems、agent runtime、memory、evaluation、engineering 等范围
- 农场隐喻只轻点，不展开解释

建议 CTA：
- 从这里开始
- 阅读最新
- 进入 GitHub

### 2. Start Here（三入口卡片）

模块标题：`从这里开始`

三张卡片：

1. **Why MiracleFarms**
   - 说明站点起点、名字和方法
2. **Latest Brief**
   - 给出一篇最近的 brief
3. **Deep Dive**
   - 给出一篇代表性长文

作用：
- 替代首页当前按类型分栏的第一入口
- 降低新读者的选择成本

### 3. Recent Updates

模块标题：`最近更新`

策略：
- 首页只展示 3 篇
- 组合建议：1 篇 Brief + 1 篇 Essay + 1 篇 Foundation

要求：
- 每篇都要有一句“为什么值得读”的短摘要
- tag 直接标明 Brief / Essay / Foundation

### 4. Topic Map

目标：
- 从“说明文字”改为“可导航的主题地图”
- 强调长期跟踪，而非热点堆叠

建议主题：
- Inference Systems
- Training & Alignment
- Agent Infrastructure
- Evaluation & Reliability

### 5. How We Work

目标：
- 把 agent-assisted 从设定变成方法说明
- 增加读者信任

建议四步：
- Research
- Drafting
- Verification
- Judgment

核心表述：
- Agent 帮助检索、整理、比较与草拟
- 判断、取舍与最终发布由人完成

### 6. About MiracleFarms

目标：
- 收纳 Farms 隐喻解释
- 缩短首页上的背景叙事

### 7. Footer tagline

建议方向：
- `Less hype, more systems.`
- `公开生长，而不是一次性完成。`

## 增量实施顺序

### Step 1（本轮优先）
- 重写 Hero
- 把首页第一大块从“内容入口分栏”改为“Start Here”
- 把内容展示改成“最近更新”三卡片

### Step 2
- 重写 Topic Map
- 新增 How We Work 区块

### Step 3
- 收敛 About 文案
- 调整导航文字与页尾收束语

### Step 4（可选）
- 如果后续需要，再把归档页（briefs / essays / foundations）文案继续统一成同一叙事风格

## 与当前站点的一致性要求

改写时保留以下不变：
- MiracleFarms 的基本世界观
- AI Infrastructure 研究写作站的定位
- 农场作为长期观察 / 迭代 / 公开生长的隐喻
- 站点整体的克制、研究型气质

不做的事：
- 不把首页改成过度品牌广告页
- 不把首页写成抽象 manifesto
- 不在首屏展开过多 AGI 愿景叙事
