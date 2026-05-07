# Repo Scope

默认追踪以下仓库；如果用户给出明确 repo 列表，用用户列表覆盖。

- `vllm-project/vllm`
- `sgl-project/sglang`
- `NVIDIA/TensorRT-LLM`
- `NVIDIA/Megatron-LM`
- `ggml-org/llama.cpp`
- `kvcache-ai/Mooncake`
- `LMCache/LMCache`
- `huggingface/trl`
- `microsoft/DeepSpeed`
- `ray-project/ray`
- `openclaw/openclaw`

## 时间窗口

- 默认回看最近三天
- 以 Asia/Shanghai 日期为准组织“今日早报”

## 优先级提示

优先看：

- merged PR
- release
- runtime / serving / kernel / scheduler / KV cache / MoE / EP / quantization / deployment 相关改动
- 官方博客、设计文档、架构说明

谨慎纳入：

- docs-only
- CI-only
- tests-only
- 没有进入主分支的讨论性材料
