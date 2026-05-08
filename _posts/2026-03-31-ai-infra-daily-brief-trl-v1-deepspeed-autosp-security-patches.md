---
title: AI Infra 早报｜安全补丁潮中的里程碑——TRL v1 正式发布，训练框架迎来新起点
date: 2026-03-31 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: TRL v1.0 正式发布，VESPO/DPPO 和 vLLM 自蒸馏推理路径一并落地；DeepSpeed v0.18.9 将 AutoSP compiler-based 序列并行和 Muon ZeRO3 支持打包入主干。vLLM 修补 batch runner SSRF 漏洞和 RTX50/DGX Spark NVFP4 崩溃，OpenClaw 补上 exec 审批 shell carrier 绕过并推进 flow registry 架构。里程碑落地与安全债收尾并行，架构层在安静打地基。
tags: [Training, Inference, Quantization]
---

今天是一个里程碑落地与安全补丁并行的工作日。训练侧有两件值得记录的事：TRL 完成 v1.0 正式版合并，DeepSpeed 发布 v0.18.9 并将 AutoSP 合入主干。推理侧以安全修复为主，vLLM 修补了一个潜藏已久的 SSRF 漏洞，并解决了新一代 NVIDIA GPU 上的 NVFP4 崩溃问题。应用侧，OpenClaw 补上 exec 审批机制的一个 shell carrier 绕过漏洞，同日开始搭建任务编排的 flow registry 架构层。

## 一、TRL v1.0：后训练算法库走到里程碑

Hugging Face TRL 今日完成 v1.0 正式版合并 [[6]](https://github.com/huggingface/trl/pull/5409)。从 v0.29.1 到 v1.0 这一跳，最重要的是 VESPO 和 DPPO 两个新 RL 算法正式稳定。

VESPO（Variational Sequence-Level Soft Policy Optimization）的核心贡献是系统性解决 off-policy RL 的策略陈旧性问题。传统 GRPO 用 token-level clipping，GSPO 做 sequence-length normalization，但两者都是启发式的。VESPO 从 variational 框架推导出一个 Gamma 加权核，对极端序列级重要性权重做平滑抑制，且不引入长度偏差——用 `loss_type="vespo"` 一行即可切换。DPPO 则用散度约束替代 PPO 的 clip 机制，提供更有原则的 trust-region 更新，目前是实验性 trainer。

今日单独合并的 PR #5388 [[6]](https://github.com/huggingface/trl/pull/5409) 对自蒸馏 trainer 来说意义相当：SDPO/SDFT 现在可以通过 `use_vllm=True` 将 rollout 生成切换到 vLLM 引擎，告别纯 Transformers generate 路径，并附带完整的 TP 配置和权重同步逻辑。reward function 也新增了 extra metrics logging 接口，中间信号可以直接上报 wandb，不再需要写自定义 callback。

## 二、DeepSpeed v0.18.9：AutoSP 走出实验室

DeepSpeed v0.18.9 [[7]](https://github.com/microsoft/DeepSpeed/releases/tag/v0.18.9) 最重要的变更是将 AutoSP（Automatic Sequence Parallelism）从独立实验性项目合入主干。

AutoSP 是一个 compiler optimization pass，在 `torch.compile()` 生成的 Torch IR FX graph 上直接操作：它自动识别 input/label/position_id 张量，将它们沿 sequence 维度分片，并在 attention 计算周围插入 Ulysses-style all-to-all 集合通信，整个过程不产生 graph break。用户只需在入口调用 `prepare_autosp_inputs()` 标注输入张量的 sequence 维度，编译器接管后续的分片推导和通信插入。

这意味着长上下文训练的序列并行配置成本大幅下降——不再需要手动改模型代码来处理序列分割和通信，编译器的 IR 级操作可以对现有模型透明地完成这件事。同版本还新增了 Muon 优化器对 ZeRO Stage 3 的支持 [[7]](https://github.com/microsoft/DeepSpeed/releases/tag/v0.18.9)，以及 AutoTP 对 HuggingFace `tp_plan` 接口的接入，进一步拓宽自动张量并行的模型覆盖面。

## 三、vLLM：两个值得关注的安全与稳定性修复

### SSRF 漏洞

batch transcription/translation 请求的 `file_url` 字段原本直接传入 `aiohttp.ClientSession().get()`，完全没有 hostname 校验 [[1]](https://github.com/vllm-project/vllm/pull/38482)。在线服务路径的 `MediaConnector` 已经有 `--allowed-media-domains` allowlist 保护，但 batch runner 的 `download_bytes_from_url` 函数没有复用这一机制，攻击者可以通过构造恶意 batch JSON 让 vLLM 进程向 `169.254.169.254` 等内网元数据端点发起请求。此次补丁补上了这个漏洞，file_url 的 hostname 现在必须通过 allowlist 校验才能发起请求。

### NVFP4 在新一代 GPU 上崩溃

RTX 5090（SM120）和 DGX Spark（SM121）这两类 SM12x GPU 在运行 NVFP4 量化模型时触发 `cudaErrorIllegalInstruction` [[2]](https://github.com/vllm-project/vllm/pull/38423)。根因有三个独立来源：CUTLASS v4.2.2 缺少 SM12x 的 NVFP4 tile constraints；FlashInfer 0.6.6 的 MoE 后端同样使用了缺陷版本的 CUTLASS；`cutlass_scaled_mm_supports_fp4()` 的检测逻辑只看 CUDA 运行时版本，不看 SM-specific kernel 是否真的编译进去了，在 `ENABLE_NVFP4_SM100` 的 build 下会误报为"SM12x 可用"，随后在 dispatch 阶段触发 SM100 PTX 被 JIT 交叉编译到 SM120 产生的非法指令。此 PR 三个根因一次全修。

## 四、边缘推理：llama.cpp 的 Adreno OpenCL 后端

llama.cpp b8589 [[3]](https://github.com/ggml-org/llama.cpp/releases/tag/b8589) 为高通 Adreno GPU 新增 q4_K 量化 OpenCL GEMM/GEMV kernel，覆盖 Snapdragon 系列移动 SoC 的 GPU 推理加速路径。同时修复了老设备的编译 bug 和 X Elite 平台的 fp16 denorm 处理。这是继此前 CUDA/ROCm/Vulkan/Metal 之后，llama.cpp 对边缘硬件后端版图的又一次填充。同日还有 b8586 修复了 CUDA CUB argsort 在 `nrows % block_size == 0` 时的未初始化值 bug [[3]](https://github.com/ggml-org/llama.cpp/releases/tag/b8589)。

## 五、OpenClaw：安全加固与任务编排架构

今日 OpenClaw 有两类值得记录的变更，方向截然不同但同样重要。

安全侧，PR #57871 [[10]](https://github.com/openclaw/openclaw/pull/57871) 修补了 exec 审批机制的一个 shell carrier 绕过问题。之前 allowlist 对 `sh -lc '$0 "$@"' ...` 这类 positional carrier 形式有特殊处理路径，错误地将 carrier 携带的目标当成审批主体；未被识别的 dispatch-capable carrier（如 xargs）可以借此扩展已审批指令的执行范围。修复方式是完全移除 carrier target matching 逻辑，配套加入了覆盖 xargs 等 carrier 的回归测试。Gateway 同日修复了 config 文件读取时的 shell 插值问题 [[13]](https://github.com/openclaw/openclaw/pull/57921)，防止 `$` 字符被意外展开。Matrix 协议的一次性 CLI 发送路径也补上了 E2EE 恢复 [[14]](https://github.com/openclaw/openclaw/pull/57936)。

架构侧，两个连续 PR 完成了 flow registry 的基础建设。PR #57865 [[11]](https://github.com/openclaw/openclaw/pull/57865) 引入 SQLite-backed flow registry，task record 支持 `parentFlowId` 关联；PR #57874 [[12]](https://github.com/openclaw/openclaw/pull/57874) 为 detached ACP run 和 subagent run 自动创建单任务父 flow，task 状态变化和终结信号通过 flow owner 路由回发起方 session。这两个 PR 是 OpenClaw 从"单次 task dispatch"走向分层任务编排的架构起点，当前版本为 scaffold，用户可见行为暂未变化，但它为后续 multi-step orchestration 打下了存储和路由层的基础。

## 六、其他值得关注的变更

TensorRT-LLM 推进至 v1.3.0rc10 [[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/12570)，Eagle speculative decoding 在 FC 前加入 Normalization 以改善草稿质量，并修复了 gRPC request manager 中 `include_stop_token_in_output` 参数未被透传的问题。

SGLang 将 AMD 侧的 MoRI 组件升至 v0.1.0 首个正式 tag [[5]](https://github.com/sgl-project/sglang/pull/21673)，从 commit hash 锁定改为 tagged release，意味着 AMD 路由优化组件进入规范化发布节奏。

Mooncake Transfer Engine 完成 TENT Auto_Connect 的默认化重构 [[8]](https://github.com/kvcache-ai/Mooncake/pull/1758)，移除 `sync` 参数，统一到 `TransferAsync` 接口，在昇腾 Atlas 平台上完成验证，降低了 Ascend 侧传输引擎的接入成本。

Ray Serve 修复了链式 `DeploymentResponse` 在上游 actor 死亡时请求永久 hang 的问题 [[9]](https://github.com/ray-project/ray/pull/62147)，之前 `ActorDiedError` 在 chained response 的 await 路径中没有被正确传播，导致请求无声无息地 hang 住而非快速失败。

---

整体来看，今天是一个"补足"的工作日：TRL 和 DeepSpeed 分别补足了算法多样性和长上下文并行能力上的关键短板；vLLM 和 OpenClaw 各自补足了已知安全漏洞；OpenClaw 的 flow registry 则在补足 agent 任务编排所需的架构基础。没有颠覆性的新方向，但每一件事都指向实际的问题，并给出了完整的解答。

## 参考来源

[1] [vLLM SSRF in batch runner download_bytes_from_url](https://github.com/vllm-project/vllm/pull/38482)

[2] [vLLM NVFP4 bugfix for DGX Spark and RTX50](https://github.com/vllm-project/vllm/pull/38423)

[3] [llama.cpp b8589 Adreno OpenCL q4_K kernel](https://github.com/ggml-org/llama.cpp/releases/tag/b8589)

[4] [TensorRT-LLM v1.3.0rc10 Eagle Norm before FC](https://github.com/NVIDIA/TensorRT-LLM/pull/12570)

[5] [SGLang AMD MoRI v0.1.0](https://github.com/sgl-project/sglang/pull/21673)

[6] [TRL v1.0 正式版 Release PR](https://github.com/huggingface/trl/pull/5409)

[7] [DeepSpeed v0.18.9 release（含 AutoSP）](https://github.com/microsoft/DeepSpeed/releases/tag/v0.18.9)

[8] [Mooncake TENT Auto_Connect 默认化](https://github.com/kvcache-ai/Mooncake/pull/1758)

[9] [Ray Serve 修复链式 DeploymentResponse ActorDiedError hang](https://github.com/ray-project/ray/pull/62147)

[10] [OpenClaw exec 审批 shell carrier 绕过修复](https://github.com/openclaw/openclaw/pull/57871)

[11] [OpenClaw Tasks flow registry scaffold](https://github.com/openclaw/openclaw/pull/57865)

[12] [OpenClaw Tasks one-task flow routing](https://github.com/openclaw/openclaw/pull/57874)

[13] [OpenClaw Gateway config shell interpolation 修复](https://github.com/openclaw/openclaw/pull/57921)

[14] [OpenClaw Matrix E2EE 一次性发送修复](https://github.com/openclaw/openclaw/pull/57936)
