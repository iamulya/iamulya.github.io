---
title: "vLLM Deep Dive Part 3: Architectures — 60+ Models, What Actually Makes Them Different, and the 2026 Frontier"
date: "2026-07-12 14:00:00 +0100"
categories: [AI Infrastructure, Deep Dives]
tags: [vLLM Deep Dive Series, LLM Serving]
mermaid: true
image:
  path: /assets/img/ai-infrastructure.png
  alt: "vLLM Deep Dive Series — Part 3: Architectures"
---

> This article is Part 3 of the [vLLM Deep Dive Series](/tags/vllm-deep-dive-series). [Part 1](/posts/vllm-deep-dive-part-1) covered the core engine. [Part 2](/posts/vllm-deep-dive-part-2) covered scaling.
{: .prompt-info }

vLLM supports over 60 model architectures. That's not 60 models — it's 60 fundamentally different neural network designs, each requiring dedicated implementation code. This post explains why that matters, groups architectures by what actually makes them different, covers the full serving feature set, and dives into the cutting-edge 2026 additions.

## Why Architecture Count Matters

Each entry in vLLM's supported models list represents a different model **architecture** — a different arrangement of neural network layers. vLLM needs architecture-specific code because the forward pass, attention mechanism, KV cache layout, and tensor shapes differ between architectures.

Think of it like car engines: a V6, inline-4, and rotary engine all produce horsepower, but you can't swap parts between them. Similarly, Llama's attention is different from DeepSeek's Multi-Latent Attention, which is different from Mamba's state-space model.

Each architecture requires vLLM to implement:

1. **Model weights loading** — different layer names, tensor shapes, parameter mappings
2. **Attention computation** — MHA vs MQA vs GQA vs MLA vs Linear vs SSM all have different math
3. **KV cache layout** — different numbers of KV heads, different dimensions, MLA compresses into latent space, linear attention/SSM use fixed-size state instead
4. **Parallelism strategy** — MoE models need expert parallelism, dense models don't
5. **Position encoding** — RoPE vs ALiBi vs absolute vs YaRN

> When a new model (like DeepSeek V4) invents a new attention mechanism, someone has to write the vLLM-specific implementation. This is why model support isn't instant.
{: .prompt-warning }

## The Four Dimensions of Architecture

Every model architecture is a combination of choices across four dimensions:

| Dimension | Description | Variants |
|---|---|---|
| **Attention Type** | How the model attends to previous tokens | **MHA** (Multi-Head): each head has full K,V — lots of memory. **MQA** (Multi-Query): single shared K,V head — minimal KV cache (PaLM, StarCoder). **GQA** (Grouped-Query): K,V shared across head groups — between MQA and MHA (Llama 2+, Mistral). **MLA** (Multi-Latent): compress K,V into latent space — DeepSeek's innovation. **Sliding Window**: attend only to last N tokens (Mistral). **Linear**: replace softmax with kernel feature maps — O(n) time, basis for RWKV/RetNet. **Cross**: Q from one sequence, K,V from another — encoder-decoder and multimodal fusion (Whisper, vision-language models). **SSM** (State-Space Model): no attention matrix, linear time (Mamba). |
| **Feed-Forward** | How the MLP processes each token | **Dense**: every parameter used for every token (Llama, Gemma). **MoE** (Mixture-of-Experts): router picks 2-8 experts per token — huge total params but small active set (DeepSeek V3, Mixtral). |
| **Positional Encoding** | How the model knows token position | **RoPE** (Rotary): most modern models. **ALiBi** (Attention Linear Biases): Falcon. **Absolute**: learned embeddings (GPT-2). **YaRN/NTK**: extended RoPE for longer contexts. |
| **Normalization** | Where/how layer norms are applied | **Pre-norm**: normalize before attention (most modern). **Post-norm**: after (GPT-2 era). **RMSNorm**: simpler, faster variant (Llama). |

## Architecture Family Groups

### Group 1: Standard Dense Transformers (GQA + RoPE + RMSNorm)

The most common architecture. These models differ mainly in training data, size, and hyperparameters — not in fundamental structure.

| Architecture | What Makes It Distinct |
|---|---|
| **LlamaForCausalLM** | The reference architecture. GQA, RoPE, RMSNorm, SwiGLU MLP. Almost everything else is a variant of this. |
| **MistralForCausalLM** | Llama-like + sliding window attention (attend to last 4096 tokens only). Saves memory for long contexts. |
| **GemmaForCausalLM** | Google's Llama-like. GeGLU activation and different head dimensions. |
| **Qwen2ForCausalLM** | Alibaba's Llama-like. Slight MLP structure and vocabulary differences. |
| **PhiForCausalLM** | Microsoft's small-but-capable models. Llama-like with partial attention. |
| **Yi / Baichuan / InternLM / Aquila** | Chinese Llama variants with different training data and tokenizers. |
| **Falcon / StableLM / OLMo / Granite** | Other Llama-like variants from various organizations. |

### Group 2: Mixture-of-Experts (MoE)

Same transformer structure, but with MoE instead of dense MLP.

| Architecture | What Makes It Distinct |
|---|---|
| **MixtralForCausalLM** | 8 experts, top-2 routing. First popular open MoE. |
| **DeepseekV2/V3/V4** | Multi-Latent Attention (MLA) + MoE with 256 experts. MLA compresses KV cache by projecting K,V into a low-rank latent space — dramatically less memory. V3+ introduces Native Sparse Attention (NSA) — hardware-aligned sparse attention that combines coarse-grained compression with fine-grained token selection, trained natively rather than applied post-hoc. |
| **Qwen3MoeForCausalLM** | Qwen with MoE. |
| **DbrxForCausalLM** | Databricks MoE with fine-grained experts. |

### Group 3: State-Space Models (SSM), Linear Attention, and Hybrids

Fundamentally different from transformers. These architectures achieve O(n) time complexity by avoiding the n×n attention matrix entirely — either through state-space models, linear attention, or retention mechanisms.

| Architecture | What Makes It Distinct |
|---|---|
| **MambaForCausalLM** | Pure SSM. Uses selective state-space model with input-dependent gating — O(n) time, fixed-size state instead of KV cache. Very memory-efficient for long contexts. Downside: lower quality on some recall-heavy tasks. |
| **RWKV** | Linear attention variant. Replaces softmax attention with a linear recurrence (WKV operator) that can run as an RNN at inference — O(1) per token, no KV cache. Now at RWKV-7 ("Goose"). |
| **RetNet** | Microsoft's retention mechanism — multi-scale exponential decay that supports parallel (training), recurrent (O(1) inference), and chunkwise modes. Mathematically distinct from both SSM and standard linear attention. |
| **JambaForCausalLM** | Hybrid: alternates SSM and attention layers. SSM's efficiency for most layers, attention's quality for critical ones. |
| **NemotronHForCausalLM** | NVIDIA's hybrid Mamba+Attention with different layer ratios. |

### Group 4: Diffusion Language Model

| Architecture | What Makes It Distinct |
|---|---|
| **DiffusionGemma** | Instead of generating tokens one-by-one (autoregressive), generates all tokens simultaneously and iteratively refines them — like image diffusion but for text. Potentially much faster. Very new (June 2026). |

### Group 5: Older Architectures

| Architecture | What Makes It Distinct |
|---|---|
| **GPT2LMHeadModel** | Original decoder. Absolute positional embeddings, post-norm, MHA. Limited to ~1024 context. |
| **GPTNeoXForCausalLM** | Rotary embeddings, parallel attention+MLP computation. |
| **GPTJForCausalLM** | EleutherAI's early large open model. |
| **GPTBigCodeForCausalLM** | Multi-Query Attention (MQA — single shared K,V head for all query heads). Used by StarCoder. Early evidence that KV head reduction works, paving the way for GQA. |

## Multimodal Models

vLLM supports vision+language, audio, and video models natively through the same serving infrastructure:

| Architecture | Example Models | Modalities |
|---|---|---|
| **LlavaForConditionalGeneration** | LLaVA 1.5, LLaVA-NeXT | Image + Text |
| **Gemma4ForConditionalGeneration** | Gemma 4 | Image + Text + Video |
| **Qwen2VLForConditionalGeneration** | Qwen2-VL, Qwen2.5-VL | Image + Text + Video |
| **InternVLForConditionalGeneration** | InternVL, InternVL2 | Image + Text |
| **PaliGemmaForConditionalGeneration** | PaliGemma | Image + Text |
| **Phi3VForCausalLM** | Phi-3-Vision, Phi-3.5-Vision | Image + Text |
| **MiniCPMV** | MiniCPM-V | Image + Text |
| **Pixtral** | Pixtral (Mistral) | Image + Text |
| **NemotronOmni** | Nemotron 3 Nano Omni | Image + Text + Audio |

Speech-to-text is supported via **Whisper** with an OpenAI-compatible transcription API.

## Embedding, Classification & Reward Models

vLLM isn't just for generation. It also serves:

| Type | Architectures |
|---|---|
| **Embeddings** | MistralModel, LlamaModel, Qwen2Model, GteModel, NomicBertModel |
| **Token Classification** | NER, POS tagging |
| **Sequence Classification** | Sentiment, toxicity detection |
| **Reward Models** | RLHF reward scoring |
| **Cross-encoder Scoring** | Reranking for RAG pipelines |

## Serving Features

### OpenAI-Compatible API

Drop-in replacement. No client code changes required.

- `/v1/chat/completions` — Chat
- `/v1/completions` — Text completions
- `/v1/embeddings` — Embeddings
- `/v1/models` — Model listing
- OpenAI Responses API support (new)
- Generative scoring API

### Tool Calling

- OpenAI-compatible `tools` parameter
- Parallel tool calls
- Forced tool use (`tool_choice: "required"`)
- MCP (Model Context Protocol) integration
- xLAM tool calling models

### Structured Output

- JSON mode (`response_format: { type: "json_object" }`)
- JSON Schema enforcement (guaranteed valid JSON matching schema)
- Regex-guided generation
- Grammar-guided decoding (CFG)
- Backends: outlines / xgrammar

### Reasoning

- Chain-of-thought with streaming
- OpenAI reasoning format (`reasoning_content` field)
- Reasoning + tool calls combined

### LoRA (Low-Rank Adaptation)

Serve **multiple fine-tuned variants** on a single base model. The base model weights are shared — only the adapter deltas are loaded per request. Hot-swap adapters per request at serving time.

### Observability

- **Prometheus** metrics — TTFT, tokens/sec, queue depth, KV cache usage
- **OpenTelemetry** tracing
- **Grafana** dashboards (pre-built templates)
- KV cache event tracking

### Integrations

Production deployment guides for Docker, Kubernetes, Helm, SageMaker, SkyPilot, RunPod, and Modal. First-class integrations with Claude Code, OpenAI Codex, LangChain, and LlamaIndex.

## Cutting-Edge Features

### Semantic Router (VSR)

Intelligent request routing *before* requests hit the model. Session-aware agentic routing maintains context across multi-turn sessions. Vision encoder integration for multimodal routing. Production-grade stateful routing (v0.3 "Themis").

### Disaggregated KV Cache

Multiple implementations for externalizing the KV cache:

| System | Description |
|---|---|
| **PegaFlow** (Novita AI) | External KV cache as standalone Rust process |
| **Mooncake Store** | Distributed KV cache for agentic workloads (3.8x throughput) |
| **LMCache** | KV cache sharing across vLLM instances |
| **FlexKV** | Flexible KV cache connector |

### Reinforcement Learning Integration

- Native RL APIs for weight transfer between training and inference
- **vime** — RL post-training framework on vLLM
- **VeRL-Omni** — RL for multimodal/diffusion models
- RLHF rollout engine with NCCL, IPC, HTTP transports

### DGX Spark / GB10

- Blackwell sm_121 architecture support
- Unified memory
- NVFP4 quantization
- Blog post with benchmarks and configuration

### Model Runner V2

Ground-up rewrite of the execution core. The new `ModelState` abstraction enables **non-autoregressive models** like DiffusionGemma — the first time vLLM can serve models that don't generate tokens one-by-one. Modular, extensible, no API changes.

## What vLLM Does NOT Do

| Feature | Status |
|---|---|
| Apple Silicon / MLX | ❌ Not supported |
| Built-in auth / RBAC | ❌ Use nginx/envoy |
| Built-in UI / dashboard | ❌ Use Grafana + Prometheus |
| Model training (full) | ❌ Inference-only |
| Cost tracking / billing | ❌ Use external tools |
| Multi-tenant isolation | ❌ Single-model per instance |

## Series Summary

Across these three posts, we've covered the full vLLM stack:

| Post | What It Covers |
|---|---|
| [Part 1: The Engine](/posts/vllm-deep-dive-part-1) | PagedAttention, continuous batching, chunked prefill, automatic prefix caching, CUDA graphs, torch.compile, quantization, attention kernels |
| [Part 2: Scaling](/posts/vllm-deep-dive-part-2) | Speculative decoding, tensor/pipeline/data/expert/context parallelism, disaggregated serving, hardware support |
| **Part 3: Architectures** (this post) | 60+ model architectures, serving features, multimodal/embedding support, 2026 cutting-edge features |

vLLM isn't just an inference server — it's the operating system for production LLM serving. Understanding its internals is the difference between "my model runs" and "my model runs well."
