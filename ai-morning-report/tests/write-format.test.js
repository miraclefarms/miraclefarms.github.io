const test = require('node:test');
const assert = require('node:assert/strict');

const {
  ensureBriefTags,
  normalizeReferenceSourceSpacing,
} = require('../src/stages/03-write');

test('ensureBriefTags inserts canonical topic tags when the AI omits them', () => {
  const content = `---
title: AI Infra 早报｜量化配置进入可组合规格
date: 2026-05-14 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 今天关注 NVFP4、KV offload、推测解码与 MLA kernel。
---

vLLM 合入 NVFP4 quantization，KV offload 转向分层管理。

## 一、推测解码

custom proposer 让 speculative decoding 继续前进，MLA attention kernel 也在同步优化。
`;

  const normalized = ensureBriefTags(content);

  assert.match(
    normalized,
    /^tags: \[Quantization, KV Cache, Speculative Decoding, Attention\]$/m,
  );
});

test('normalizeReferenceSourceSpacing keeps each brief reference visually separated', () => {
  const content = `## 参考来源

[1] [vLLM PR](https://github.com/vllm-project/vllm/pull/1)
[2] [SGLang PR](https://github.com/sgl-project/sglang/pull/2)
[3] [TRT-LLM PR](https://github.com/NVIDIA/TensorRT-LLM/pull/3)
`;

  const normalized = normalizeReferenceSourceSpacing(content);

  assert.match(
    normalized,
    /\[1\] \[vLLM PR\]\([^)]+\)\n\n\[2\] \[SGLang PR\]\([^)]+\)\n\n\[3\] \[TRT-LLM PR\]\([^)]+\)/,
  );
});
