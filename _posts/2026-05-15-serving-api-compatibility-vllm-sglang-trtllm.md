---
title: vLLM、SGLang 与 TensorRT-LLM 的服务 API 兼容层设计
date: 2026-05-15 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: 基于三家框架当前源码与官方 API 文档，梳理 OpenAI、Anthropic 两类接口差异，以及 vLLM、SGLang、TensorRT-LLM 的兼容实现边界。
tags: [vLLM, SGLang, TRT-LLM, Inference]
---

> **版本声明**：本文分析基于 vLLM commit `95cfe102a5ff`、SGLang commit `3f7e538b2ffa`、TensorRT-LLM commit `da9ce581162d`（均为 2026-05-15 查询到的 main HEAD）；除非特别说明，以下描述均基于此版本。

把一个推理框架说成“OpenAI compatible”很容易，真正麻烦的是它能不能承接上层应用已经依赖的对话状态、工具调用、流式事件和错误语义。`/v1/chat/completions` 能跑通只解决了第一层问题。Claude Code、OpenAI Responses、带工具的 agent loop、带图片的 tool result 进入同一条链路后，兼容层就不再是几个 FastAPI route，而是一个协议翻译器。

这篇笔记只看服务端接口，不展开 KV cache、调度器和 attention backend。核心判断是：vLLM 和 SGLang 都把 OpenAI Chat Completions 当成内部协议中枢，再在边界上适配 Anthropic Messages；TensorRT-LLM 的内建 server 主要沿 OpenAI 家族推进，Chat、Completions 和 Responses 都在内建路由里，Anthropic Messages 目前需要外部网关或应用层适配。这个差异会直接影响“能不能把 Claude 客户端指到本地推理服务”这种看似简单的部署动作。

## 一、两套 API 的差异不在 URL，而在对象模型

OpenAI 的服务接口现在分成三条常见路径：旧式 `Completions`、仍被大量生态使用的 `Chat Completions`，以及官方推荐给新项目的 `Responses`。OpenAI 文档把 Responses 定义为新的 API primitive，并明确说 Chat Completions 仍受支持，但新项目推荐使用 Responses<a href="https://platform.openai.com/docs/guides/responses-vs-chat-completions">[1]</a>。这对自托管框架有一个现实含义：只兼容 `/v1/chat/completions` 已经不够覆盖新的 agent SDK、工具调用和多轮状态链路。

Chat Completions 的形态最像传统 RPC：请求里有 `model` 和 `messages`，每条 message 有 `role` 和 `content`，响应是 `choices[]`，每个 choice 里给出一个 assistant message。流式响应也延续这个结构，只是把完整 `message` 换成增量 `delta`。OpenAI 的 streaming 指南明确说 Chat Completions stream 里返回的是 `delta` 字段，可以包含 role token、content token 或空内容<a href="https://platform.openai.com/docs/guides/streaming-responses">[2]</a>。这套设计非常适合作为本地推理框架的通用入口，因为大多数模型最终都要把一组消息渲染成一个 prompt，再把 token 流解码回文本。

Responses API 的抽象层更高。它把 `input`、`instructions`、`tools`、`previous_response_id`、`reasoning`、`text.format` 放进同一个对象里，响应输出也从“一个 assistant message”扩展成一组 typed output items。官方 reference 里，`previous_response_id` 用来串起多轮状态，`tools` 是一组允许模型调用的工具定义，`stream` 则通过 SSE 发出带类型的 response events<a href="https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses">[3]</a>。换句话说，Responses 相比 Chat Completions 的关键变化，是把 agent loop 的一部分状态管理和事件模型上移到了服务端接口。

Anthropic Messages API 的方向不同。Claude API 的主入口是 `POST /v1/messages`，并且有独立的 `POST /v1/messages/count_tokens`；请求需要 `anthropic-version` 这样的版本头<a href="https://platform.claude.com/docs/en/api/overview">[4]</a>。system prompt 放在顶层 `system` 字段，不占用 `messages` 里的 `system` role；输出 content 使用 content block 数组，而非单个字符串。工具调用由 assistant content 里的 `tool_use` block 表示，不使用 OpenAI 的 `tool_calls` 数组；工具结果由下一轮 user message 里的 `tool_result` block 带回。

流式协议的差异更明显。Anthropic 的 stream 是一个状态机：先发 `message_start`，然后每个 content block 依次经历 `content_block_start`、多个 `content_block_delta`、`content_block_stop`，最后发 `message_delta` 和 `message_stop`<a href="https://platform.claude.com/docs/en/build-with-claude/streaming">[5]</a>。工具参数流使用 `input_json_delta.partial_json`，不同于 OpenAI Chat 的 `tool_calls[].function.arguments` delta。这解释了为什么 Anthropic-compatible server 不能只把 URL 改成 `/v1/messages`，它必须重建一套 SSE 事件序列。

工具语义同样有边界。Anthropic 文档把工具分成 client tools 和 server tools：client tools 由应用执行，Claude 返回 `tool_use` block，应用把 `tool_result` 发回；server tools 由 Anthropic 侧执行，结果直接进入响应<a href="https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview">[6]</a>。OpenAI Responses 也支持 built-in tools 和自定义 function tools，但输出 item、工具结果回填、并行工具调用的字段形态不同。对自托管框架来说，“支持 tools”必须拆开看：支持工具 schema、支持模型生成工具调用、支持流式工具参数、支持服务端执行内置工具，是四件事。

## 二、vLLM：OpenAI Chat 是主干，Anthropic 是边界转换

vLLM 的内建 server 明确注册了 OpenAI Chat、Responses、Completions 和 Anthropic Messages 四类 route。`register_generate_api_routers()` 先挂 `/v1/chat/completions`，再挂 `/v1/responses`、`/v1/completions`，最后挂 Anthropic router<a href="https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/generate/api_router.py#L19-L42">[7]</a>。官方 serving 文档也把 vLLM 的 HTTP server 描述为实现 OpenAI Completions、Chat API 和 Responses API，并提示可以用官方 OpenAI Python client 调用<a href="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/">[8]</a>。

Anthropic 兼容层的实现很直接：`AnthropicServingMessages` 继承自 `OpenAIServingChat`，收到 `/v1/messages` 后先把 Anthropic request 转成 `ChatCompletionRequest`，然后调用同一条 `create_chat_completion()` 路径，最后把 OpenAI Chat response 或 stream 转回 Anthropic Messages 形态<a href="https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/anthropic/serving.py#L121-L457">[9]</a>。官方 API 文档页也把这个类标成 “Handler for Anthropic Messages API requests”，并写明它会转换成 OpenAI request 再调用底层 chat completion endpoint<a href="https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/">[10]</a>。

这条转换链里有几个细节能看出 vLLM 的兼容边界。顶层 `system` 被插入成 OpenAI `{"role": "system"}` message；Anthropic image block 被转成 OpenAI `image_url` content part；`tool_use` 被转成 OpenAI `tool_calls[].function`，`tool_result` 被转成 `role: "tool"` 的 message；`tool_choice: "any"` 被映射成 OpenAI 的 `required`；`stop_sequences` 被映射成 OpenAI `stop`。这些映射足以让 Claude-style 客户端走到本地模型，但语义依赖模型 chat template 和 tool parser。模型如果本身没有学会对应工具调用格式，server 层只能搬运结构，不能保证行为。

Responses 在 vLLM 里是独立 handler，不只是把 Responses 请求粗暴改写成 Chat。`ResponsesRequest` 按官方字段顺序建模，包含 `input`、`instructions`、`previous_response_id`、`tools`、`reasoning`、`stream` 等字段，同时也加入了 vLLM 自己的扩展，如 `cache_salt`、`priority`、`kv_transfer_params` 和 `chat_template_kwargs`<a href="https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/responses/protocol.py#L136-L286">[11]</a>。`OpenAIServingResponses` 会处理 `previous_response_id`、构造 messages 和 engine inputs，再进入 engine client 生成<a href="https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/responses/serving.py#L325-L470">[12]</a>。

实际部署时，vLLM 的优势是兼容面宽：OpenAI SDK、Anthropic SDK、Responses 客户端都有内建入口。风险点也清楚：OpenAI Responses 里的 hosted built-in tools、server-side prompt cache、OpenAI 账户级 metadata 不可能在本地完全复刻；Anthropic 的 thinking block、tool_result 图片、streaming tool 参数虽然有结构映射，最后仍要落到模型模板和解析器上。把 vLLM 当作“协议模拟器”可以，把它当作“OpenAI/Anthropic 云端语义完整复刻”就会踩坑。

## 三、SGLang：HTTP server 直接挂多协议，Gateway 继续扩 OpenAI 面

SGLang 的 Python HTTP server 同样把 OpenAI-compatible API 当作主入口。`http_server.py` 直接注册 `/v1/completions`、`/v1/chat/completions`、`/v1/embeddings`、`/v1/audio/transcriptions`、`/v1/models`、`/v1/responses`、`/v1/rerank` 等端点<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L1483-L1709">[13]</a>。官方文档也把 OpenAI-compatible endpoint 描述成从 OpenAI 切到自托管模型时减少客户端改动的入口<a href="https://sgl-project-sglang-93.mintlify.app/backend/openai-compatible-api">[14]</a>。

Anthropic 路径在 SGLang 里也不是独立 engine。启动时，server 先构造 `openai_serving_chat`，再用它初始化 `AnthropicServing`<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L320-L350">[15]</a>。`/v1/messages` 和 `/v1/messages/count_tokens` 只是把请求交给 `anthropic_serving`<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L1758-L1778">[16]</a>。

`AnthropicServing` 的文件头已经把设计写清楚：把 Anthropic request 转成 OpenAI ChatCompletion format，委托给 `OpenAIServingChat`，再把响应转回 Anthropic format<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/anthropic/serving.py#L1-L64">[17]</a>。实现上，顶层 `system` 被拼成 OpenAI system message，图片转成 `image_url`，`tool_use` 转成 OpenAI function tool call，user 侧 `tool_result` 转成 OpenAI tool message。非流式请求会调用 OpenAI chat handler 的 `_convert_to_internal_request()` 和 `_handle_non_streaming_request()`；流式请求先拿 OpenAI chat stream，再逐块转成 Anthropic 的 `message_start`、`content_block_delta`、`message_delta`、`message_stop` 事件<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/anthropic/serving.py#L320-L650">[18]</a>。

Responses API 在 SGLang 里已经有单独 handler。`OpenAIServingResponses.create_responses()` 会处理 `previous_response_id`、选择 Harmony 或普通路径构造 messages 和 engine prompts，然后生成 `GenerateReqInput` 交给 tokenizer manager；stream、background、store、tool server 逻辑也在这个 handler 内处理<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/openai/serving_responses.py#L162-L370">[19]</a>。这说明 SGLang 的 Responses 已经嵌入自己的生成、工具和 history 管理路径，定位高于纯转发 proxy。

SGL Model Gateway 又给 SGLang 增加了一层路由能力。README 里列出 OpenAI-compatible `/v1/chat/completions`、`/v1/responses`、`/v1/conversations`、`/v1/embeddings`、`/v1/rerank`、`/v1/classify` 等端点，并区分 HTTP router、gRPC router 和 OpenAI router<a href="https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/sgl-model-gateway/README.md#L451-L483">[20]</a>。这和 Python HTTP server 的定位不同：Python server 是单实例推理服务入口，Model Gateway 更像多 worker、多 backend 的 control/data plane。写应用适配时，要先确认自己对接的是 `sglang serve` 还是 `sgl-model-gateway`，两者的兼容面和状态存储位置不完全一样。

## 四、TensorRT-LLM：内建 server 走 OpenAI 家族，Anthropic 留给外层

TensorRT-LLM 的 `trtllm-serve` 文档说得很明确：它启动 OpenAI-compatible server，支持 `/v1/models`、`/v1/completions`、`/v1/chat/completions`，并提供 `/health`、`/metrics`、`/version`；文档后面又给出 Chat、Completions 和 Responses API 的调用示例<a href="https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html">[21]</a>。源码 route list 也对应这个判断：`OpenAIServer.register_routes()` 注册 `/v1/completions`、`/v1/chat/completions`、`/v1/responses`、`GET/DELETE /v1/responses/{response_id}`，但在当前 `tensorrt_llm/serve` 内建 server 路由中没有 `/v1/messages`<a href="https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L584-L653">[22]</a>。

Chat 路径的设计和另外两家类似：`openai_chat()` 读取 `ChatCompletionRequest`，把 messages 解析成 conversation，套 chat template，构造 sampling params，再调用 generator。工具 schema 会转成 template 所需的 tool dict，strict tool 会进入 constrained decoding；多模态 message 通过 `parse_chat_messages_coroutines()` 处理<a href="https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L1083-L1185">[23]</a>。它关注的是把 OpenAI Chat payload 高效落到 TRT-LLM executor，而不是同时承担多 provider 协议翻译。

Responses 路径则通过 `responses_utils` 做预处理和后处理。`openai_responses()` 会处理 `previous_response_id`、conversation store、Harmony 模式、streaming processor，然后把预处理出的 input tokens 和 sampling params 交给 `generator.generate_async()`<a href="https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L1664-L1785">[24]</a>。这个设计更接近 vLLM 和 SGLang 的 Responses handler：Responses 有自己的状态和事件后处理，不只是 Chat endpoint 的别名。

Anthropic 兼容是 TensorRT-LLM 当前内建 server 的明显边界。要让 Anthropic SDK、Claude Code 或只认 `/v1/messages` 的客户端调用 TRT-LLM，工程上通常要在前面加一层网关：把 Anthropic Messages 请求转成 OpenAI Chat 或 Responses，把 stream 事件从 OpenAI delta 改写成 Anthropic content block 事件，再把 tool_use/tool_result 做双向映射。这一层可以由应用自己写，也可以交给 LiteLLM、OpenRouter 类 proxy，或企业内部 gateway。TRT-LLM 负责模型执行路径，协议多样性留给上游，这个取舍和 NVIDIA 生态里 Triton、NIM、外部网关分层的习惯一致。

## 五、兼容性对照：三家都支持 OpenAI Chat，分歧从 Responses 和 Anthropic 开始

把三家框架放在同一张表里，能看出兼容工作的真实边界：

| 能力 | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|--------------|
| `/v1/chat/completions` | 内建支持；OpenAI Chat 是主 serving 路径 | 内建支持；Python server 与 Model Gateway 都覆盖 | 内建支持；`trtllm-serve` 主路径 |
| `/v1/completions` | 内建支持 | 内建支持 | 内建支持 |
| `/v1/responses` | 内建支持；独立 Responses handler | 内建支持；Python handler + Gateway 扩展 | 内建支持；独立 Responses 预处理/后处理 |
| `/v1/messages` | 内建支持；转 OpenAI Chat | 内建支持；转 OpenAI Chat | 当前内建 server 未注册 |
| `/v1/messages/count_tokens` | 内建支持 | 内建支持 | 当前内建 server 未注册 |
| OpenAI SDK 直接调用 | 成熟 | 成熟 | 成熟 |
| Anthropic SDK 直接调用 | 可行，但依赖具体字段覆盖 | 可行，但依赖具体字段覆盖 | 需要外部适配 |
| 工具调用 | 依赖 tool parser、chat template、模型格式 | 依赖 tool parser、Harmony/模板和 tool server | 依赖 tool parser、模板和 executor 后处理 |

这张表的部署含义很实际。已有应用如果只使用 OpenAI Chat Completions，三家都能作为替代后端；如果应用用了 OpenAI Responses 的 `previous_response_id`、background、typed streaming events 或 built-in tools，需要逐项压测，不能只看 endpoint 是否存在；如果应用基于 Anthropic Messages，vLLM 和 SGLang 可以直接试，TensorRT-LLM 要先放一个协议网关。

最容易被忽略的是工具流式事件。OpenAI Chat 的工具参数在 `tool_calls[].function.arguments` 里增量出现，Anthropic 的工具参数在 `content_block_delta` 的 `input_json_delta.partial_json` 里出现。vLLM 和 SGLang 都在 Anthropic adapter 里维护 content block 状态，把 OpenAI chat stream 改写成 Anthropic stream；这层状态机写错，客户端遇到的故障就会从文本延迟升级成工具调用 JSON 无法恢复。Claude-style agent 客户端通常对这个特别敏感。

另一个边界是 system/developer role。OpenAI Chat 已经支持 `developer` message；Anthropic Messages 仍然把 system instruction 放在顶层 `system`。自托管框架在 Anthropic -> OpenAI 转换时通常会把 top-level system 变成 OpenAI `system` role，而不是 `developer` role。对于普通开源 chat template 这往往是可行的；对于严格区分 instruction hierarchy 的模型，角色降级可能改变行为。这个问题源自协议抽象和模型模板之间缺少统一标准。

## 六、落地建议：把“兼容”拆成四个测试

第一层测试是基础 schema。用官方 OpenAI SDK 打 `/v1/chat/completions`，用官方 Anthropic SDK 打 `/v1/messages`，确认非流式文本、流式文本、错误响应和 request id 都符合客户端预期。很多问题会在这一层暴露出来，例如 base URL 是否多写了 `/v1`，模型名是否和客户端硬编码校验冲突，或者服务端是否忽略了某些 OpenAI 参数。

第二层测试是 tool schema 和 tool result round trip。OpenAI 侧测 `tools`、`tool_choice`、assistant `tool_calls`、下一轮 `role: "tool"`；Anthropic 侧测 `tools.input_schema`、assistant `tool_use`、下一轮 user `tool_result`。对 vLLM 和 SGLang，这正是 adapter 的关键路径。对 TensorRT-LLM，如果前面加网关，网关要负责这组双向转换。

第三层测试是 streaming event。不要只看 SDK 能否打印文本，要记录原始 SSE：OpenAI Chat 应该是 `data: {...delta...}` 加最终 `[DONE]`，OpenAI Responses 应该是 typed events，Anthropic Messages 应该有 `message_start`、content block 事件、`message_delta`、`message_stop`。工具参数流式输出要单独测，因为它比纯文本更容易暴露状态机问题。

第四层测试是状态。OpenAI Responses 的 `previous_response_id`、Anthropic 的 stateless multi-turn messages、框架自己的 conversation store，是三种不同状态来源。vLLM、SGLang 和 TensorRT-LLM 都在 Responses handler 里实现了一部分本地 store 或 history 逻辑，但这不等价于 OpenAI 云端完整语义。生产里应把状态归属写清楚：是客户端每轮带全量 history，还是 server 保存 response，还是外部 gateway 管理 conversation。

## 七、结论

三家框架的接口设计已经从“提供 OpenAI Chat 兼容端点”进入到“模拟多 provider 协议层”的阶段。vLLM 和 SGLang 的共同路线是把 OpenAI Chat 作为内部中间表示，让 Anthropic Messages 在边界转换；Responses 则逐步成为更高层的独立入口，承接工具、状态和 typed streaming。TensorRT-LLM 的路线更收敛，内建 server 围绕 OpenAI 家族接口展开，把 Anthropic 这种 provider-specific 兼容留给外层。

做选型时，问题不该问“这个框架支不支持 OpenAI API”，而应该问：我的客户端依赖的是 Chat、Responses 还是 Anthropic Messages？工具调用是否需要流式？状态存在客户端、server 还是网关？这些问题回答完，兼容层该放在框架内、应用内还是独立 gateway，基本就清楚了。

---

## 参考资料

[1] [OpenAI: Migrate to the Responses API](https://platform.openai.com/docs/guides/responses-vs-chat-completions)

[2] [OpenAI: Streaming API responses](https://platform.openai.com/docs/guides/streaming-responses)

[3] [OpenAI API Reference: Create a model response](https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses)

[4] [Claude API Overview](https://platform.claude.com/docs/en/api/overview)

[5] [Claude API Docs: Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)

[6] [Claude API Docs: Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

[7] [vLLM source: generate API router registration](https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/generate/api_router.py#L19-L42)

[8] [vLLM Docs: OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

[9] [vLLM source: Anthropic Messages serving adapter](https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/anthropic/serving.py#L121-L457)

[10] [vLLM API Docs: Anthropic serving handler](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/)

[11] [vLLM source: Responses protocol](https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/responses/protocol.py#L136-L286)

[12] [vLLM source: Responses serving](https://github.com/vllm-project/vllm/blob/95cfe102a5ffd6bc10c2e897a2d3f3fd3fb250db/vllm/entrypoints/openai/responses/serving.py#L325-L470)

[13] [SGLang source: OpenAI-compatible routes](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L1483-L1709)

[14] [SGLang Docs: OpenAI Compatible API](https://sgl-project-sglang-93.mintlify.app/backend/openai-compatible-api)

[15] [SGLang source: AnthropicServing initialization](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L320-L350)

[16] [SGLang source: Anthropic routes](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/http_server.py#L1758-L1778)

[17] [SGLang source: Anthropic serving adapter overview](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/anthropic/serving.py#L1-L64)

[18] [SGLang source: Anthropic stream conversion](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/anthropic/serving.py#L320-L650)

[19] [SGLang source: OpenAI Responses serving](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/python/sglang/srt/entrypoints/openai/serving_responses.py#L162-L370)

[20] [SGLang source: SGL Model Gateway README](https://github.com/sgl-project/sglang/blob/3f7e538b2ffabb6ed7cfa39d9c97095e50b23e40/sgl-model-gateway/README.md#L451-L483)

[21] [TensorRT-LLM Docs: trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)

[22] [TensorRT-LLM source: OpenAI server route registration](https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L584-L653)

[23] [TensorRT-LLM source: OpenAI Chat handler](https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L1083-L1185)

[24] [TensorRT-LLM source: OpenAI Responses handler](https://github.com/NVIDIA/TensorRT-LLM/blob/da9ce581162d158183b867c782452c74e7e61983/tensorrt_llm/serve/openai_server.py#L1664-L1785)

### 版本对齐信息

| 依赖 | 版本/Commit | 日期 |
|------|-------------|------|
| vLLM | `95cfe102a5ff`（main HEAD） | 2026-05-15 |
| SGLang | `3f7e538b2ffa`（main HEAD） | 2026-05-15 |
| TensorRT-LLM | `da9ce581162d`（main HEAD） | 2026-05-15 |
| OpenAI API 文档 | 官方文档，查询于 2026-05-15 | 2026-05-15 |
| Claude API 文档 | 官方文档，查询于 2026-05-15 | 2026-05-15 |
