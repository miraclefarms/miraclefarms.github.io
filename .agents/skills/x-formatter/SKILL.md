---
name: x-formatter
description: >
  Convert published MiracleFarms GitHub.io posts into English X posts or threads tuned for AI Infra discourse.
  Use when generating X/Twitter distribution copy from an existing MiracleFarms brief, essay,
  reading note, or field note. Outputs JSON for automated publishing pipelines.
---

# MiracleFarms X Formatter

Turn a GitHub.io article into an English X post or thread optimized for AI infrastructure audiences.

## Input

The caller provides:

- The publication date.
- The canonical GitHub.io URL.
- The complete GitHub.io markdown article, including front matter and body.

Treat the GitHub.io article as the source of truth. Do not add facts, numbers, or claims not present in the article.

---

## Format Decision

Choose the format before writing. Match format to article kind:

| Article kind | Default format | Override |
|---|---|---|
| `brief` (daily dispatch) | 5-7 post thread | drop to 4 if content is thin |
| `essay` (deep analysis) | 4-6 post thread | single long post if the essay has one sharp thesis |
| `reading` (paper overview) | 3-4 post thread | single post if paper has one viral stat |
| `field-note` | 1-2 post single | short thread if multiple connected observations |

**Thread vs single post rule:** Use a thread when there are 2+ distinct mechanisms, numbers, or implications worth unpacking. Use a single post when one sentence captures everything important.

---

## Output Contract

Return JSON only: a JSON array of strings.

- Each string is one post.
- Each string must be at most **260 characters** (buffer below platform 280 limit).
- The **first string** must contain the canonical GitHub.io URL *and* stand alone as a complete thought. URL goes in post 1 to avoid the 30-50% reach penalty of linking mid-thread.
- Do not wrap the JSON in Markdown fences.
- Do not include commentary before or after the JSON.
- Image attachment: if a benchmark chart or diagram in the article would make post 1's claim concrete, append `[attach: assets/{slug}/filename.png]` on a new line at the end of that string. The calling script strips this before posting.

---

## AI Infra Content Patterns (Reference)

High-performing AI infra posts on X cluster around these themes. Bias toward whichever is present in the article:

**Hot topics (2025-2026):**
- Disaggregated prefill/decode (production deployments at Meta, LinkedIn, Mistral)
- KV cache reuse and offload (Mooncake, LMCache, RadixAttention)
- MoE model serving (DeepSeek-V3/R1, Qwen, Gemma MoE)
- Speculative decoding improvements
- Blackwell FP4/NVFP4 inference cost reduction
- Agent serving — long context, multi-step, remote coding agents
- vLLM vs SGLang performance competition
- Prefix caching, cross-request KV reuse

**Style anchors for AI infra on X:**
- @karpathy: 1-3 sentence conceptual insights, confident present tense
- @dylan522p: hardware economics, precise numbers, specific comparisons
- @lmsysorg: benchmark-first, throughput/latency numbers front-loaded

---

## Style Guide

**Voice:** A sharp engineer writing between meetings. Analytical, direct, no hype.

**Lead with the most concrete thing:** a number, a before/after ratio, a counter-intuitive claim. Never lead with "A new paper shows..." or "Researchers have found..."

**Technical nouns to keep verbatim:** inference, KV cache, prefill/decode, disaggregation, speculative decoding, routing, Blackwell, H100, CUDA, FP4/FP8, MoE, TTFT, TPS, throughput, latency, quantization, prefix caching, LoRA, serving.

**Compression rules:**
- Cut filler: "This is significant because...", "It's worth noting that..."
- Use `→` for before/after or cause/effect
- Use line breaks between ideas within a single post (improves readability at phone scale)
- Present tense, active voice

**Tone by article kind:**
- Brief: punchy, signal-dense, 1 key number per post
- Essay: analytical, explains *why*, not just *what*
- Reading/Paper: leads with the finding, second post explains the mechanism

**Emoji:** none by default. Never decorative.

**Hashtags:** 0 is default. Use 1-2 only if a tag is actively searched (e.g. #vLLM, #LLMInference). Never generic (#AI, #MachineLearning).

**Final post:** End with a question or implication that invites engineers to reply. Replies carry 27× the algorithmic weight of likes.

---

## Thread Structure

### Daily Brief → thread

```
Post 1: One-sentence thesis + URL.
         Lead with today's most important shift — a number or a named change.

Post 2–4: One concrete technical development per post.
           What changed → what it enables → why it matters to engineers.
           Include specific numbers where the article has them.

Post 5: The broader infrastructure implication.
         "Taken together, this means..." — what should an infra engineer do differently?

Post 6 (optional): An open question or unresolved tension that invites replies.
```

### Essay → thread

```
Post 1: The core question or thesis + URL.
         State what the essay is arguing, not just what it's about.

Post 2–3: Mechanism and evidence.
           Key insight + the data or experiment that validates it.

Post 4: Boundary conditions or tradeoffs.
         When does this break? What does it sacrifice?

Post 5: Concrete implication for engineers or investors.
         "If you're running X workload, this means Y."
```

### Paper / Reading → thread

```
Post 1: The one result worth remembering + URL.
         Lead with the number: "3.1× throughput on Llama-70B with PD disaggregation."

Post 2: The mechanism in plain terms.
         Why does it work? One sentence per step.

Post 3: Who should care and when.
         "Matters most for chat-heavy MoE workloads, less so for batch inference."

Post 4 (optional): Open engineering question or implication.
```

---

## Image Attachment Priority

When attaching an image via `[attach: ...]`, prefer:
1. Throughput/latency comparison charts
2. Architecture diagrams showing the key component placement
3. Terminal output proving a benchmark number

Never attach stock photos, decorative images, or abstract graphics.

---

## Quality Bar

Read back each post at phone-scroll speed. Check:

1. Does post 1 make someone stop scrolling without reading the thread?
2. Does every middle post add *new* information — not a restatement?
3. Does the final post leave engineers with a concrete action or open question?
4. Can a non-Chinese-speaking infra engineer read this fully without confusion?
5. Does it sound like a person wrote it, not a translation?

---

## Self-Check

- [ ] Output is valid JSON array, no Markdown fences
- [ ] Every string ≤ 260 characters
- [ ] Post 1 contains the canonical URL and stands alone
- [ ] No Chinese text anywhere
- [ ] No marketing language ("revolutionary", "groundbreaking", "exciting")
- [ ] No external links except the canonical URL in post 1
- [ ] Technical nouns preserved exactly (KV cache, not "key-value cache")
- [ ] Emoji: 0 (default) or ≤ 1 if intentional
- [ ] Hashtags: 0 (default) or ≤ 2 if justified
- [ ] Each middle post adds new information — no restating
- [ ] Final post invites reply or states implication
