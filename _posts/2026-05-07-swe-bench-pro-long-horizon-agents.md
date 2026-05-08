---
title: SWE-Bench Pro：代码智能体的长周期工程门槛
date: 2026-05-07 12:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: SWE-Bench Pro 用企业级长周期任务重新划定代码智能体评测边界：公开集仍低于45%，商业集低于20%，瓶颈落在多文件理解、工具使用与验证设计。
tags: [Agents, Evaluation]
---

代码智能体评测正在进入一个更接近真实工程的阶段。SWE-Bench Pro 的核心价值，是把“修一个 GitHub issue”推进到“在陌生代码库里完成长周期、多文件、可验证的工程变更”<a href="https://arxiv.org/pdf/2509.16941v2">[1]</a>。这个变化会同时拷问模型能力、agent scaffold、测试环境和数据污染控制。

这篇论文最该被记住的判断是：当前代码智能体的短板已经很难只用“模型会不会写代码”解释。公开集上，Claude Sonnet 4.5、Claude Sonnet 4、GPT-5 high 的 Pass@1 仍分别只有 43.6%、42.7%、41.8%；商业集上，最高的 Claude Opus 4.1 只有 17.8%。当任务跨越数个文件、数十到数百行补丁、不同语言生态和业务约束时，失败往往来自路径选择、工具使用、上下文管理和验证接口，而不只是函数级代码生成。

## 一、SWE-Bench Pro 在改写评测对象

SWE-Bench Pro 继承了 SWE-Bench 的 issue resolution 设定：给模型一个代码库和任务描述，让 agent 生成 patch，再用测试判断是否解决问题<a href="https://arxiv.org/abs/2310.06770">[5]</a>。但它对任务来源和任务形态做了两层重新定义。

第一层是污染控制。论文把 1,865 个问题拆成三组：731 个公开问题来自 11 个 copyleft 许可证仓库，858 个 held-out 问题来自另外 12 个仓库并保持私有，276 个 commercial 问题来自 18 个创业公司的私有代码库。公开集已经发布到 Hugging Face，数据字段包含 `patch`、`test_patch`、`problem_statement`、`requirements`、`interface`、`dockerhub_tag` 等；这让外部研究者看到的不只是 leaderboard，还有复现实验所需的任务结构<a href="https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro">[3]</a>。

第二层是任务复杂度。SWE-Bench Verified 中有相当比例的任务只需要一两行修改，SWE-Bench Pro 则主动排除了 1 到 10 行的 trivial edits。论文给出的参考补丁平均 107.4 行代码、跨 4.1 个文件；Figure 1 里的分布也很直观，SWE-Bench Verified 的修改量集中在个位数行，SWE-Bench Pro 的 public set 则明显右移到几十行甚至百行以上。

![SWE-Bench Verified 与 SWE-Bench Pro 的补丁规模对比](/assets/swe-bench-pro-long-horizon-agents/fig-1-patch-complexity.png)

*图 1：SWE-Bench Pro 把任务规模从少量单文件补丁推进到多文件、几十到上百行修改。评测难度的提升首先来自工程跨度，题目描述长度只是表层现象。来源：arXiv HTML Figure 1。*

这张图背后的含义很直接：如果 benchmark 主要由小补丁组成，模型只要定位到正确函数，就有机会靠局部模式补全拿分；当修改分散到多个文件，agent 必须理解调用链、接口约束、测试意图和项目约定。此时，模型生成能力只是执行链路的一段，前面的 repo navigation 和后面的 verification loop 同样决定结果。

## 二、人工增强把“规格含糊”从主要变量中拿掉

SWE-Bench Pro 的另一个重要设计，是把每个任务拆成 problem statement、requirements 和 interface。Problem statement 来自原始 commit、PR、issue 等材料，并被改写成 issue 风格；requirements 补充测试所期待的行为；interface 则在函数名、类名、路径等测试敏感位置显式约束模型输出<a href="https://arxiv.org/html/2509.16941v2">[2]</a>。

这个设计容易被误读成“给模型开后门”。更准确的理解是，作者想把评测重心从“猜用户到底要什么”转移到“给定足够规格后能不能完成工程实现”。真实软件工程当然包含需求澄清，但在 benchmark 里，如果测试只接受某个接口名，而题目没有说明这个接口名，模型提交一个语义可行但测试不匹配的方案就会变成 false negative。SWE-Bench Pro 用人工 requirements 和 interface 收窄解空间，是为了让测试更像验证器，减少隐藏规则带来的误判。

Figure 2 展示了 public set 的文件数与任务类型分布。大头落在 1 到 9 个文件之间，但 10 到 19 个文件、20 个以上文件的任务也存在；类别上，major bug、data bug、refactoring enhancement、code quality enhancement、UI/UX、integration、security、performance 等并存。这种分布比单一 Python 后端 bug 修复更接近实际团队 backlog。

![SWE-Bench Pro public set 的文件数和任务类别分布](/assets/swe-bench-pro-long-horizon-agents/fig-2-task-distribution.png)

*图 2：public set 同时覆盖 bug、feature、refactor、security、UI/UX、performance 等任务类型。它考察的是 agent 在不同工程语境中迁移处理流程的能力。来源：arXiv HTML Figure 2。*

人工增强的效果在消融实验里很明显。作者在 50 turn、2 美元成本上限的分析设置下比较了两种输入：完整的 problem statement + requirements + interface，以及只给 problem statement。GPT-5 high 从 25.9% 掉到 8.4%，Claude Opus 4.1 从 22.7% 掉到 8.2%。这个差距说明，当前 agent 在规格缺口面前很容易走向“形式上合理、测试上失败”的解法；也说明 benchmark 的上限并不只由模型决定，任务描述如何把测试期望转成可执行约束，同样是评测系统的一部分。

这里也有边界。SWE-Bench Pro 默认设置不主要测需求澄清能力，也不测 agent 能不能主动向人类追问。它更像一个“规格已补齐的工程实现”评测。对于企业内部真实流程，这个假设并不总成立；但对模型和 scaffold 横向比较而言，这种处理减少了大量非目标噪声。

## 三、公开集分数高了，商业集把落差重新拉开

论文 v2 的公开集结果显示，头部模型已经能在 public set 上接近 40% 到 45%。Claude Sonnet 4.5 为 43.6%，Claude Sonnet 4 为 42.7%，GPT-5 high 为 41.8%，Claude Haiku 4.5 也达到 39.5%。Kimi K2 Instruct 和 GPT-OSS 120B 分别是 27.7% 与 16.2%。

商业集给出了另一幅图景。Claude Opus 4.1 为 17.8%，GPT-5 high 为 15.7%，GPT-5 medium 为 14.9%，Gemini 2.5 Pro Preview 为 10.1%，Claude Sonnet 4 为 9.1%，GPT-4o 只有 3.6%。这个落差比公开集排名本身更有解释力：当代码库来自私有创业公司，业务规则、项目约定、依赖环境和隐含上下文都更接近真实工作场景，agent 的可迁移性马上下降。

论文使用统一的 SWE-Agent scaffold，并把模型最大轮数设为 50；开源模型由 vLLM 在 8 张 H100 上托管，尽量把基础设施变量收拢到相同框架内。GitHub 仓库也给出了复现实验的路径：生成 patch、用 `gather_patches.py` 汇总预测，再通过 `swe_bench_pro_eval.py` 在 Docker 环境里执行评测<a href="https://github.com/scaleapi/SWE-bench_Pro-os">[4]</a>。这对 AI Infra 读者很关键，因为 benchmark 不再只是数据集，评测 harness 本身已经变成一个工程产品。

Figure 3 把结果进一步拆到文件数、LOC、语言和仓库维度。文件数增加后，resolve rate 明显下滑；当任务涉及 10 个以上文件或 500 行以上修改，头部模型还能保持一定命中，小模型和开源模型则接近归零。语言维度上，Go 和 Python 相对更容易，JavaScript 与 TypeScript 波动更大；仓库维度上，同一模型在不同 repo 的表现可以从个位数到 50% 以上。

![模型表现随文件数、LOC、语言和仓库变化](/assets/swe-bench-pro-long-horizon-agents/fig-3-model-breakdown.png)

*图 3：文件数和修改规模增加会迅速放大模型差距；语言和仓库差异说明 agent 能力并非单一代码生成能力，而是受项目结构、工具链、测试设计和文档质量共同影响。来源：arXiv HTML Figure 3。*

这组拆分给模型训练和 agent 框架设计都提出了更细的问题。训练侧不能只堆更多单文件 patch，必须覆盖跨文件依赖、项目特定接口和多语言构建系统。scaffold 侧则要重视代码检索、增量验证、错误恢复和上下文压缩。一个能写出正确局部 diff 的模型，如果总是找错文件、漏掉调用方、读文件读到上下文溢出，同样无法在 SWE-Bench Pro 上稳定拿分。

## 四、失败模式说明瓶颈已经转向执行链路

SWE-Bench Pro 的 failure mode analysis 使用 GPT-5 作为 judge，对失败轨迹最后 20 轮进行分类。这个方法本身有自动评审误差，但它提供了一个有用视角：不同模型失败的形态差异很大。

Claude Opus 4.1 的失败里，提交 patch 的比例高达 74.2%。在这些已提交失败中，wrong solution 占 50.3%，syntax error 占 31.3%。这类失败更像“走到了终点，但补丁语义或可运行性有问题”。GPT-5 high 则呈现相反形态，未提交失败占 72.8%，其中 tool-use 占 96.4%。这说明它在某些轨迹中可能卡在探索或工具交互阶段，无法把中间状态稳定推进到可提交 patch。

Claude Sonnet 4 的未提交失败里，long context 占 57.4%，stuck in loop 占 33.9%。这组数字对长周期 agent 尤其有启发：上下文窗口不只是容量问题，也是行为稳定性问题。模型在多轮工具调用后会遗忘目标、重复读文件、围绕同一错误打转，最终耗尽 turn budget。Gemini 2.5 Pro Preview 和 Qwen3 32B 的失败则更多落在 syntax、tool-use、incorrect file 等执行细节上，反映出模型能力和工具接口耦合还不够稳。

从基础设施角度看，这些失败模式指向三类改进。第一，repo navigation 需要更强的结构化索引，避免 agent 用纯文本 grep 和反复阅读承担全部定位工作。第二，工具调用要更可恢复，例如 patch 应用失败、测试命令超时、依赖安装失败时，系统应提供结构化错误和可继续路径。第三，上下文管理要从“截断历史”升级为任务状态机，保留目标、已读文件、假设、失败测试和待验证点，减少原始 shell 输出对模型工作记忆的污染。

## 五、边界：它还覆盖不了完整软件工程

SWE-Bench Pro 很有价值，但边界需要写清楚。

首先，语言覆盖仍有限。论文明确提到 public set 主要覆盖 Python、JavaScript、TypeScript 和 Go，Java、C++、Rust 等大型企业工程常见语言不足。这会影响对系统级、性能敏感、强类型大型项目的判断。

其次，测试仍是主要验证器。fail2pass 和 pass2pass 可以给出清晰的自动分数，但真实工程还有代码可维护性、性能回归、安全风险、架构一致性和 code review 可接受性。一个 patch 通过测试，不代表它就是团队愿意合并的方案；一个 patch 没通过特定测试，也可能是接口选择不同导致的误判。

第三，commercial set 的私有性带来了必要但真实的复现限制。私有代码能显著降低污染并贴近企业任务，但外部研究者无法完整审计任务和环境。这个取舍可以理解，却也意味着公开 leaderboard 的分数需要和公开可复现实验分开阅读。

最后，默认输入里的 requirements 和 interface 降低了需求含糊度。它适合评估“规格明确后的实现能力”，对需求澄清、产品判断、人机协作和代码评审闭环覆盖不足。下一代评测如果要更靠近真实团队流程，可能需要把 issue discussion、review comments、设计文档和多轮人类反馈纳入任务轨迹。

## 六、结论

SWE-Bench Pro 把代码智能体评测的焦点从“能不能修一个公开 issue”推向“能不能在长周期工程环境里稳定交付 patch”。公开集低于 45%、商业集低于 20% 的结果，说明当前头部模型已经具备相当强的局部实现能力，但距离自治软件工程仍有明显差距。差距主要落在工程链路：找对位置、保持目标、使用工具、跨文件修改、处理测试和在有限上下文里维持任务状态。

对模型团队来说，这篇论文提示训练数据要覆盖更长、更脏、更跨文件的真实变更。对 agent 框架团队来说，分数提升未必只来自更强模型，也可能来自更好的文件索引、测试调度、错误恢复和上下文压缩。对评测基础设施来说，SWE-Bench Pro 的真正贡献是把 benchmark、环境、数据污染控制和人工验证放在同一套系统里讨论。

它没有回答“AI agent 什么时候能替代工程师”这种宏大问题。它给出的更具体答案是：如果任务需要专业工程师花数小时到数天完成，当前 agent 还远没有稳定通过。但现在我们终于有了一个更接近这个问题的尺子。

---

## 参考资料

[1] [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? arXiv PDF v2](https://arxiv.org/pdf/2509.16941v2)

[2] [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? arXiv HTML v2](https://arxiv.org/html/2509.16941v2)

[3] [ScaleAI/SWE-bench_Pro dataset on Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)

[4] [scaleapi/SWE-bench_Pro-os GitHub repository](https://github.com/scaleapi/SWE-bench_Pro-os)

[5] [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
