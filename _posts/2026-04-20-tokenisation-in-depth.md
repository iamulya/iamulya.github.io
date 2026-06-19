---
layout: post
title: "Tokenisation in Depth: BPE, SentencePiece, Vocabularies, and Why Tokens Are Not Words"
date: 2026-04-20 09:00:00 +0000
categories: [Generative AI in Depth]
tags: [LLM, Tokenisation, bpe, sentencepiece, tiktoken, Vocabulary, Tokenizer, Generative AI in Depth]
description: "A deep dive into how text becomes token IDs: BPE, WordPiece, SentencePiece, tiktoken, vocabulary size trade-offs, byte fallback, and why tokenisation decisions have lasting consequences."
math: true
mermaid: true
image:
  path: /assets/img/generative-ai-in-depth.png
  alt: "Generative AI in Depth — A Technical Deep Dive Series"
---

> This article is **Part 1 of 15** in the [Generative AI in Depth](/categories/generative-ai-in-depth/) series.
{: .prompt-info }


Every LLM article talks about tokens. Few explain what a token actually is, how it's chosen, and why the tokeniser is one of the most consequential architectural decisions a model team makes — baked in at the start and essentially impossible to change later.

This article covers the full tokenisation stack: what problem tokenisers solve, how Byte Pair Encoding works step by step, how modern tokenisers like tiktoken and SentencePiece are built, the trade-offs in vocabulary size, byte-level fallback, and the surprising ways tokenisation shapes model behaviour.

---

## The Problem: Text Is Not Tokens

Neural networks operate on fixed-size numerical vectors, not raw text. The tokeniser's job is to convert a string of Unicode characters into a sequence of integer IDs that the model can process.

The simplest approaches don't work well:

**Character-level tokenisation:** Split text into individual characters. Vocabulary is small (~200 unique characters for Unicode subset), but every word requires many tokens — "hello" is 5 tokens. This makes sequences very long and attention expensive (attention is O(T²) in sequence length T). Long sequences also make it harder for the model to learn long-range dependencies.

**Word-level tokenisation:** Split on whitespace and punctuation. Vocabulary size explodes (English alone has > 170K common words; add proper nouns, URLs, code, and it's unbounded). Unknown words ("outta-vocabulary" or OOV) must be replaced by a `<UNK>` token, losing information. Also doesn't work across languages with no spaces (Chinese, Japanese, Thai).

**Subword tokenisation:** The solution used by all modern LLMs. Splits words into frequently-occurring subword units: "unhappiness" → ["un", "happiness"], "tokenization" → ["token", "ization"]. Common words are one token; rare words are multiple shorter tokens. Vocabulary size is bounded and manageable (typically 32K–256K), sequences are shorter than character-level, and OOV is nearly impossible.

---

## Byte Pair Encoding (BPE)

BPE (originally a text compression algorithm, adapted for NLP by Sennrich et al., 2016) is the tokenisation algorithm behind GPT-2, GPT-3, GPT-4 (tiktoken), Llama, Mistral, and most modern LLMs.

### Building a BPE Vocabulary

**Input:** A large text corpus and a target vocabulary size $V$ (e.g., 50,000 tokens).

**Step 1 — Start with a character vocabulary:**

Begin with a vocabulary containing every unique character (or byte) in the training corpus. Pre-tokenise the text into words (split on whitespace), and represent each word as a sequence of characters with a special end-of-word marker:

```
"low"   → ["l", "o", "w", "</w>"]
"lower" → ["l", "o", "w", "e", "r", "</w>"]
"new"   → ["n", "e", "w", "</w>"]
"newer" → ["n", "e", "w", "e", "r", "</w>"]
```

**Step 2 — Count adjacent pair frequencies:**

Count how often every adjacent pair of symbols appears across the entire corpus:

```
("l", "o"):    45,231 occurrences
("o", "w"):    38,102 occurrences
("l", "o") followed by ("o", "w"): → pair ("lo") appears 45,231 times
...
```

**Step 3 — Merge the most frequent pair:**

Take the most frequent pair, merge it into a new symbol, and add it to the vocabulary:

```
Most frequent: ("l", "o") → merge into "lo"
New vocabulary: {..., "l", "o", "lo", ...}

Words now:
"low"   → ["lo", "w", "</w>"]
"lower" → ["lo", "w", "e", "r", "</w>"]
```

**Step 4 — Repeat until vocabulary size $V$ is reached:**

Recount pair frequencies with the new vocabulary and merge again. Each merge step adds one token to the vocabulary. After $V$ merges, the vocabulary is complete.

**Example — partial trace on a small corpus:**

```
Initial:  {"l":8, "o":8, "w":6, "e":4, "r":2, "n":2, "d":1, ...}

Iter 1: merge ("e","r") → "er"   [freq=4]
Iter 2: merge ("l","o") → "lo"   [freq=8]  
Iter 3: merge ("lo","w") → "low" [freq=6]
Iter 4: merge ("low","er") → "lower" [freq=2]
...
```

Common words and common subwords emerge naturally from frequency analysis. The algorithm is **greedy and deterministic** — given the same corpus and target vocabulary size, it always produces the same vocabulary.

### Encoding at Inference Time

At inference time, encoding a new string uses the **merge rules learned during training**, applied in order. Given the input "lower":

1. Start: `["l", "o", "w", "e", "r"]`
2. Apply merge rule ("l","o")→"lo": `["lo", "w", "e", "r"]`
3. Apply merge rule ("lo","w")→"low": `["low", "e", "r"]`
4. Apply merge rule ("e","r")→"er": `["low", "er"]`
5. Result: `["low", "er"]` → token IDs `[3201, 81]` (hypothetical)

The encoding is **deterministic**: the same string always produces the same token sequence.

---

## Byte-Level BPE

Standard BPE starts from character-level units. **Byte-level BPE** starts from the 256 possible byte values instead.

This is a crucial difference:

- **Character-level BPE**: vocabulary must cover all Unicode characters (>140,000). Characters outside the vocabulary cause OOV. Different scripts (Latin, Cyrillic, CJK, Arabic) need separate handling.
- **Byte-level BPE**: any Unicode string can be UTF-8 encoded into bytes. Since the vocabulary covers all 256 byte values, **any text can always be tokenised** — no OOV is possible.

Byte-level BPE is used by GPT-2, GPT-3, GPT-4, Llama, Mistral, and most modern models. The 256 byte tokens form the "floor" of the vocabulary; all higher tokens are merges of byte sequences.

**Example — "café" in byte-level BPE:**

The character "é" in UTF-8 is two bytes: `0xC3 0xA9`. If "é" is rare enough that the BPE merges haven't combined `0xC3 0xA9` into a single token:

```
"café" → UTF-8 bytes → [0x63, 0x61, 0x66, 0xC3, 0xA9]
       → apply BPE merges → ["ca", "f", "é"]  (if "é" merge exists)
       → or                → ["ca", "f", "Ã©"]  (byte representation, if no merge)
```

This byte-level fallback means the model can tokenise any language, emoji, or unusual character — at the cost of multiple tokens per character for rare scripts.

---

## WordPiece and Unigram Language Model

Two other subword algorithms are used by major models:

### WordPiece (BERT, DistilBERT, mBERT)

Similar to BPE but the merge criterion is different. Instead of merging the most frequent pair, WordPiece merges the pair that **maximises the training data likelihood** under the current vocabulary. This means merging pairs that carry the most mutual information, not just the most common pairs.

WordPiece also uses a `##` prefix to indicate continuation tokens: "unhappiness" → `["un", "##happy", "##ness"]`. This differs from BPE where continuation is implied by the absence of a space prefix.

### Unigram Language Model (SentencePiece)

Rather than bottom-up merging, Unigram starts with a large candidate vocabulary and prunes it. The algorithm:

1. Start with all possible substrings up to some maximum length
2. Train a unigram language model (each token has an independent probability)
3. Remove the tokens whose removal causes the smallest increase in corpus encoding loss
4. Repeat until the target vocabulary size is reached

Unigram tokenisation produces **probabilistic** tokenisations — the same string might be tokenised differently based on context. During training, this regularisation effect can improve generalisation. During inference, the most probable tokenisation is used.

---

## SentencePiece

SentencePiece (Kudo and Richardson, 2018) is the tokeniser library used by T5, LLaMA 1 and 2, Gemma 1 and 2, and many other models. It's not a tokenisation algorithm itself — it's a **library that implements BPE or Unigram** with some additional properties:

**1. Language-agnostic pre-tokenisation:**
Standard BPE pre-tokenises on whitespace, which doesn't work for Japanese, Chinese, or Thai (no word boundaries). SentencePiece treats the input as a raw stream of Unicode characters, with whitespace treated as a regular character (represented as `▁`). This makes it truly language-agnostic.

```
"Hello world" → ["▁Hello", "▁world"]
"こんにちは"   → ["▁こん", "に", "ちは"]  (or finer splits)
```

**2. Byte fallback:**
For unknown characters, SentencePiece can fall back to individual bytes (represented as `<0xXX>`), similar to byte-level BPE.

**3. Self-contained model:**
The entire tokeniser (vocabulary + merge rules) is stored in a single `.model` file. Loading it requires no external dependencies.

**SentencePiece vs tiktoken:**

| Property | SentencePiece | tiktoken |
|---|---|---|
| Algorithm | BPE or Unigram | BPE (byte-level) |
| Pre-tokenisation | Whitespace-agnostic (▁) | Regex-based |
| Implementation | C++ library, Python bindings | Rust, fast Python bindings |
| Speed | Moderate | Very fast (~5–10× SentencePiece) |
| Used by | LLaMA 1/2, Gemma, T5, Mistral* | GPT-2/3/4, Claude |

*Mistral models switched to tiktoken-compatible BPE in later versions.

---

## tiktoken

tiktoken is OpenAI's tokeniser library, used for GPT-2 through GPT-4 and most Claude models. It uses **byte-level BPE with regex pre-tokenisation**.

### The Regex Pre-tokenisation Step

Before BPE merges are applied, tiktoken splits the input with a regex pattern. For GPT-4 (cl100k_base):

```python
PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

This splits on:
- Contractions: `'s`, `'t`, `'ll`, `'ve`, `'re`, `'d`, `'m`
- Words (with optional leading space)
- Numbers (up to 3 digits per token)
- Punctuation with optional trailing whitespace
- Whitespace sequences

The key effect: **BPE merges never cross word boundaries**. The regex ensures "token" and " token" are tokenised independently — they may get different token IDs even though they differ only in a leading space.

This pre-tokenisation also means numbers like "12345" are split into `["12", "34", "5"]` (3-digit blocks). This is why LLMs famously struggle with arithmetic — "42" and "42" in different positions may or may not be the same token, and multi-digit numbers are split across tokens.

### tiktoken Vocabularies

| Name | Model | Vocabulary size |
|---|---|---|
| `gpt2` | GPT-2 | 50,257 |
| `r50k_base` | GPT-3 (text-davinci-001) | 50,281 |
| `p50k_base` | GPT-3 (Codex, text-davinci-002) | 50,281 |
| `cl100k_base` | GPT-3.5, GPT-4, Claude | 100,277 |
| `o200k_base` | GPT-4o | 200,019 |

The jump from 50K to 100K vocabularies approximately halves the sequence length for most inputs — the same text takes fewer tokens. The jump to 200K in GPT-4o further reduces token count, particularly for non-English languages.

---

## Non-English and Multilingual Text

Tokenisation is not language-neutral. Most widely-used tokenisers were trained on English-dominant corpora, and the consequences for non-English users are significant.

### Why Scripts Differ

The root of the problem is **UTF-8 encoding**. Unicode characters do not all occupy the same number of bytes:

| Script | Bytes per character (UTF-8) | Examples |
|---|---|---|
| ASCII (English, basic punctuation) | 1 | A–Z, 0–9, `!`, `.` |
| Latin extended, Cyrillic, Greek, Hebrew | 2 | é, ü, ñ, Ж, ψ, א |
| CJK (Chinese, Japanese, Korean), Arabic, Thai | 3 | 你, の, 한, ب, ก |
| Emoji, historic scripts | 4 | 😀, 𐍈 |

BPE starts from bytes. A BPE tokeniser trained on English-heavy data has seen billions of ASCII byte sequences and far fewer 3-byte CJK sequences. Merges accumulate aggressively for common English subwords but rarely fire for CJK characters — so each CJK character often remains 1–3 tokens on its own.

**Token fertility** (characters per token) is the practical result:

```
English:  ~4 characters per token   (baseline)
French:   ~4.2 chars/token
German:   ~4.1 chars/token
Russian:  ~3.1 chars/token          (2-byte Cyrillic; fewer merges)
Arabic:   ~2.3 chars/token          (3-byte, right-to-left)
Chinese:  ~1.5 chars/token          (3-byte CJK; minimal BPE merges)
Japanese: ~1.5 chars/token
Thai:     ~1.1 chars/token          (3-byte; no spaces between words)
```

For the same semantic content, a Chinese prompt uses roughly 2–3× as many tokens as an English one. This means:
- **Shorter effective context window** per "thought" — a 128K-token context holds far less Chinese text than English
- **Higher API cost** — providers charge per token
- **Worse model performance** — models trained on 80%+ English data have seen each non-English concept fewer times, and with noisier tokenisation

### Scripts Without Word Boundaries

Standard BPE pre-tokenises on whitespace: it splits the text into words first, then applies BPE within each word. This breaks for **Chinese**, **Japanese**, **Thai**, **Lao**, **Burmese (Myanmar script)**, and **Khmer**, which have no spaces between words. A BPE tokeniser that pre-tokenises on whitespace would receive entire sentences as single "words" — producing enormous, inefficient tokenisations.

**How each algorithm handles this:**

| Tokeniser | Approach |
|---|---|
| Byte-level BPE (tiktoken) | Regex pre-tokenisation matches Unicode letter blocks (`\p{L}+`) — CJK characters match as individual letter sequences, so BPE operates character by character within them |
| SentencePiece | Treats the full input as a raw Unicode stream with no pre-tokenisation split — handles CJK, Thai, and any other script uniformly |
| Original character-level BPE | Fails: whitespace split produces pathological tokenisations for CJK |

SentencePiece's approach is the most principled: by representing whitespace as `▁` and treating it as a regular character, it makes no assumptions about word boundaries at all. This is why it was chosen for multilingual models (LLaMA 1/2, mT5, Gemma).

### Right-to-Left Languages

Arabic, Hebrew, Urdu, and Farsi are written right-to-left. **Tokenisation itself is unaffected** — BPE and SentencePiece both operate on the underlying byte or Unicode stream, which is a left-to-right sequence regardless of visual rendering direction. The token IDs produced are correct; the rendering direction is a UI concern.

What *does* matter for Arabic and Hebrew: both scripts use 3-byte UTF-8 characters, combined with frequent diacritics and gemination markers that are additional characters. Arabic's connected script also means that the visual shape of a character depends on its neighbours — but this is transparent to the tokeniser, which operates on code points.

### Byte Fallback: The Last Resort

When a character is rare enough that BPE has no merge that covers it, it falls back to individual bytes. In byte-level BPE (tiktoken), this is automatic — every character decomposes to bytes if no higher merge applies. In SentencePiece with byte fallback enabled, rare characters are represented as `<0xXX>` tokens.

What this looks like in practice for an obscure character like `‌` (zero-width non-joiner, U+200C, used in Arabic/Persian):

```
U+200C → UTF-8: [0xE2, 0x80, 0x8C]
→ No merge exists → 3 separate byte tokens
```

The model sees 3 tokens where a human sees nothing visible. These byte-token sequences are in the model's vocabulary and can be generated, but the model has seen them rarely during training — their embeddings are typically weak.

### Code Switching

Many real-world prompts mix languages — a French user asking a question, then quoting an English error message; a Japanese developer writing Japanese prose with English function names. This is called **code switching**.

Byte-level tokenisers handle this transparently: they tokenise each language independently within the same sequence. The challenge is at the model level — the model must have seen sufficient training data in each language, and the embedding space must connect concepts across languages. Tokenisation is necessary but not sufficient for multilingual capability.

### How Modern Models Address the Problem

The vocabulary expansion trend of the last two years is largely driven by multilingual demands:

| Model | Vocabulary | Key multilingual change |
|---|---|---|
| LLaMA 2 | 32,000 | Primarily English BPE |
| LLaMA 3 | 128,256 | 4× expansion; added many CJK, Cyrillic, Arabic tokens |
| GPT-4 | 100,277 | Doubled from GPT-3's ~50K; better non-English coverage |
| GPT-4o | 200,019 | Further expansion; better CJK and non-Latin efficiency |
| Gemma 2 | 256,128 | Very large; broad multilingual coverage |
| Qwen 2.5 | 151,643 | Optimised for Chinese (Alibaba's primary language) |

More vocabulary tokens for a script means more BPE merges have been learned for it, which means shorter token sequences (better fertility) and more training exposure per concept. LLaMA 3's jump from 32K to 128K specifically targeted the CJK and Cyrillic fertility gap — the same Chinese sentence that took ~3 tokens/character in LLaMA 2 takes ~2 tokens/character in LLaMA 3.

The practical rule: **for non-English applications, prefer models with larger vocabularies trained on multilingual data, and always test token counts on your actual language before estimating context usage or API costs.**

---

## Vocabulary Size Trade-Offs

Choosing the vocabulary size $V$ involves real trade-offs:

### Sequence Length

For a fixed piece of text, a larger vocabulary produces shorter sequences (more text per token). Shorter sequences mean:
- Lower memory cost (KV cache scales with sequence length T)
- Faster inference (attention is O(T²))
- Better long-range dependency learning (fewer steps between distant concepts)

### Embedding Table Size

The token embedding table has shape $[V, d_{\text{model}}]$. For Llama 3 (d=4096, V=128,256):

$$V \times d = 128256 \times 4096 \approx 500M \text{ parameters}$$

That's about 2 GB in BF16 — a non-trivial fraction of the total model. A vocabulary of 256K × 4096 would be 4 GB just for embeddings.

### Model Capacity

With more vocabulary tokens, the model has more "entries" to specialise. But each token needs to appear enough times during pre-training for the model to learn a useful embedding. Very large vocabularies may have underrepresented tokens with poor embeddings.

### Common vocabulary sizes in practice:

| Model | Vocabulary size | Notes |
|---|---|---|
| GPT-2 | 50,257 | Baseline BPE |
| LLaMA 1 | 32,000 | SentencePiece, relatively small |
| LLaMA 2 | 32,000 | Same as LLaMA 1 |
| LLaMA 3 | 128,256 | Major expansion for multilingual/code |
| Mistral 7B v0.1 | 32,000 | SentencePiece |
| Mistral v0.3+ | 32,768 | tiktoken-compatible |
| Gemma 1/2 | 256,128 | Very large vocabulary |
| GPT-4 | 100,277 | cl100k_base |
| GPT-4o | 200,019 | o200k_base |
| Qwen 2.5 | 151,643 | tiktoken-based |

### Tokenisation Efficiency by Language

With a vocabulary primarily trained on English text, tokenisation efficiency varies dramatically by language:

```
English:    ~4 characters per token  (baseline)
Spanish:    ~4.3 chars/token
German:     ~4.1 chars/token  (compound words split more)
French:     ~4.2 chars/token
Russian:    ~3.1 chars/token  (Cyrillic, 2-byte UTF-8 characters)
Chinese:    ~1.5 chars/token  (3-byte UTF-8, few merges)
Arabic:     ~2.3 chars/token
Japanese:   ~1.5 chars/token
Thai:       ~1.1 chars/token
```

For the same semantic content, a Chinese or Thai user uses 2–3× as many tokens as an English user. This means:
- Higher inference cost per "thought"
- Shorter effective context window for non-English languages
- Models trained on English-dominant data may have worse multilingual performance partly because the tokenisation is less efficient

LLaMA 3's vocabulary expansion to 128K (from LLaMA 2's 32K) specifically targeted this — adding many more tokens for non-Latin scripts.

---

## Special Tokens

Beyond regular vocabulary tokens, tokenisers include **special tokens** with specific roles:

| Token type | Examples | Purpose |
|---|---|---|
| Beginning of sequence | `<bos>`, `<s>`, `\|begin_of_text\|` | Marks start of a sequence |
| End of sequence | `<eos>`, `</s>`, `\|eot_id\|` | Model stops generating |
| Padding | `<pad>` | Pads batches to equal length |
| Unknown | `<unk>` | Fallback for OOV (rare in byte-level BPE) |
| Role markers | `\|system\|`, `\|user\|`, `\|assistant\|` | Chat template structure |
| Thinking | `<think>`, `</think>` | Reasoning model scratchpad |

Special tokens are typically added to the vocabulary as new entries (not derived from BPE merges). The tokeniser has dedicated handling to insert them at the right positions.

**Chat templates and special tokens** are inseparable — the prompt formatter inserts special tokens, and the model was trained to associate specific special tokens with specific behaviours. Using the wrong chat template (e.g., formatting a Llama 3 prompt with Llama 2's special tokens) produces poor results.

> **Chat template mismatch degrades quality silently.** The model won't error — it'll generate text, just worse text. A model formatted with the wrong chat template may: ignore system prompts, repeat user messages in the response, fail to stop at the right place, or produce incoherent multi-turn conversations. Always use `tokenizer.apply_chat_template()` in HuggingFace or the framework's built-in chat formatting — never hand-craft special token sequences by guessing.
{: .prompt-warning }

---

## How Tokenisation Shapes Model Behaviour

Tokenisation is not just a preprocessing step — it has deep effects on what the model can and cannot do:

### Arithmetic

Numbers like `12345` tokenise as `["123", "45"]` in cl100k_base — the split is chosen for compression, not mathematical structure. This means place-value columns (ones, tens, hundreds…) don't align with token boundaries, so the model can't perform column arithmetic the way a human does with pencil and paper.

> **Token boundaries break place-value alignment.** `"12345 + 67890"` tokenises as `["123", "45", " +", " 678", "90"]` — the ones digit of `12345` is buried inside `"45"` and the ones digit of `67890` is buried inside `"90"`. To add them, the model would need to extract sub-token characters and align across arbitrary boundaries, which attention can't do natively. Large models appear to solve arithmetic through memorised pattern-matching, not column arithmetic — which is why they fail on very large numbers or unusual formats they haven't seen before. **Chain-of-thought** helps because writing out intermediate steps externalises the alignment onto the token stream, letting the model approximate column arithmetic one token at a time. Digit-by-digit tokenisation (one token per digit) eliminates this problem at the cost of longer sequences.
{: .prompt-info }

### Code

Programming languages have different tokenisation efficiency than natural language. Python's indentation (spaces) tokenises as multiple tokens per indent level. In GPT-2 (p50k_base), 4 spaces of indentation = 1 token; in cl100k_base, it's also 1 token. But some languages and patterns tokenise very inefficiently: whitespace-heavy formats like YAML or Markdown tables can cost 2-3× more tokens than equivalent code.

### Tokenisation Boundary Effects

The model learns statistics from the training data's token boundaries. Common words have single tokens; the model sees them atomically. Rare words are split; the model must compose the meaning from subwords. "GPU" might tokenise as `["G", "PU"]` in a small vocabulary model — the model has to infer it's a compound from context, rather than having learned a single embedding for the concept.

### Character-Level Blindness

The same root cause that breaks arithmetic — tokens are the atomic unit, not characters — also breaks character-level reasoning:

- **Counting letters:** "How many r's in strawberry?" In cl100k_base, `strawberry` tokenises as `["straw", "berry"]`. The model sees two tokens, not nine characters. It must somehow *recall* that `"straw"` contains one 'r' and `"berry"` contains two — knowledge that was incidentally absorbed during pretraining, not computed at inference time. Early GPT-4 reliably answered "2" instead of "3" on this exact word.
- **Spotting a letter:** "Does 'necessary' contain a double letter?" The model has to reconstruct character-level composition from a token it processes holistically.
- **Rhyming and spelling:** Words that sound similar may tokenise completely differently; words that look similar on the page may share no token structure.
- **Anagrams:** Reordering characters within or across tokens is invisible at the token level.

> **The token is the smallest unit the model processes — it has no direct access to the characters inside it.** Character-level facts ("straw" contains 'r' at position 3) are only available to the model if they were captured statistically during pretraining through contexts like spelling exercises, crosswords, or linguistically annotated text. This is why frontier models improved at letter-counting over successive versions — not because the architecture changed, but because later training data included more character-level reasoning examples. Chain-of-thought helps here for the same reason as arithmetic: "s-t-r-a-w-b-e-r-r-y" spelled out explicitly puts each character in its own token, restoring direct access.
{: .prompt-info }

### The "Assistant" Capitalisation Problem

In some vocabularies, "assistant" and "Assistant" are different tokens. If the training data always uses "Assistant" (capital A) in system prompts and the inference code uses "assistant" (lowercase), the model may not trigger the right behaviour. Tokenisation is case-sensitive.

### Tokenisation Artifacts

Certain token sequences can produce unexpected model behaviour:

- **The "SolidGoldMagikarp" phenomenon**: tokens that appeared frequently in training data with unusual context can cause models to produce gibberish or unexpected outputs when prompted directly. The token `" SolidGoldMagikarp"` (a Reddit username that appeared often in training data) famously caused GPT-2 to output nonsense.
- **Glitch tokens**: tokens with embeddings that weren't well-trained (e.g., tokens from compressed web data) can destabilise model outputs.

---

## Tokenisation at Inference vs Training

A critical consistency requirement: **the tokeniser used at inference must exactly match the tokeniser used during training**. If they differ:

- Different token splits produce different input IDs
- The model sees inputs it was never trained to handle
- Outputs degrade silently — no error, just worse results

Model cards and releases always specify the tokeniser as part of the model's specification. Serving frameworks (vLLM, SGLang, etc.) load the tokeniser from the model's HuggingFace repo automatically.

---

## Counting Tokens: The Practical Impact

Tokens matter for:

**Context windows:** The 128K context of Llama 3 is 128K *tokens*, not characters or words. A 100-page document at ~500 words/page, ~4 chars/word, ~4 chars/token ≈ 500K characters ≈ 125K tokens — nearly fills the window.

**API billing:** OpenAI, Anthropic, Google all charge per token. For a fixed budget, tokenisation efficiency determines how much text you can process. Non-English languages cost 2–3× more per semantic unit.

**Batch sizes during training:** Packing sequences into fixed-length windows (e.g., 4096 tokens per batch item) is standard. Longer documents need truncation; shorter documents can be packed together.

---

## Key Takeaways

- Tokenisers convert text to integer sequences using subword algorithms that balance vocabulary size, sequence length, and coverage
- **BPE** iteratively merges the most frequent adjacent byte/character pairs — the algorithm behind GPT, LLaMA, Mistral, and most modern models
- **Byte-level BPE** (tiktoken, LLaMA 3) guarantees OOV-free tokenisation by starting from 256 byte values — any Unicode string is always encodable
- **SentencePiece** (LLaMA 1/2, Gemma) handles language-agnostic tokenisation by treating whitespace as a regular character (▁)
- Larger vocabularies → shorter sequences → faster, cheaper inference; but larger embedding tables and sparser coverage per token
- Non-English languages are typically 2–3× less token-efficient than English with English-trained vocabularies; modern models (LLaMA 3, GPT-4o) have expanded vocabularies to address this
- Tokenisation shapes model capabilities: arithmetic, code handling, and multilingual performance all trace partly to tokenisation decisions
- The tokeniser is baked into the model at training time — changing it later requires retraining from scratch

> **You cannot swap or upgrade the tokenizer after training.** If you want a different tokenizer (e.g., more multilingual coverage, a larger vocabulary, different special tokens), you must retrain from scratch on the new tokenizer — there is no "tokenizer migration" path. When evaluating base models, check the tokenizer's vocabulary size and coverage for your target language/domain *before* committing to it for fine-tuning.
{: .prompt-warning }

---

## Further Reading

- [Inside LLM Inference](/posts/decoder-forward-pass-dimensions) — what happens after tokenisation (the forward pass)
- [Attention Mechanisms and KV Cache](/posts/attention-mechanisms-and-kv-architectures) — how token sequences are processed
- [The Memory Math](/posts/llm-memory-math) — sequence length (in tokens) drives memory and cost
- [LLM Evaluation in Depth](/posts/llm-evaluation-in-depth) — how tokenisation affects benchmark scores
- BPE paper: Sennrich et al., [arXiv 1508.07909](https://arxiv.org/abs/1508.07909)
- SentencePiece paper: Kudo and Richardson, [arXiv 1808.06226](https://arxiv.org/abs/1808.06226)
