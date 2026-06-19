---
layout: post
title: "LLM Evaluation in Depth: Benchmarks, Contamination, and What Actually Matters"
date: 2026-06-06 09:00:00 +0000
categories: [Generative AI in Depth]
tags: [LLM, Evaluation, Benchmarks, mmlu, humaneval, Contamination, Evals, Elo, ARC-AGI, GPQA, HLE, SWE-bench, Generative AI in Depth]
description: "A critical look at how LLMs are evaluated in 2025: MMLU-Pro, GPQA, HLE, AIME, SWE-bench, BFCL, ARC-AGI, Chatbot Arena Elo, benchmark contamination, and how to evaluate your own fine-tuned model."
math: true
mermaid: true
image:
  path: /assets/img/generative-ai-in-depth.png
  alt: "Generative AI in Depth — A Technical Deep Dive Series"
---

> This article is **Part 14 of 15** in the [Generative AI in Depth](/categories/generative-ai-in-depth/) series.
{: .prompt-info }


Every LLM release announces new state-of-the-art benchmark scores. Most of these scores are misleading in ways that require understanding to interpret. This article explains how LLM evaluation actually works — what benchmarks measure, why they fail, how contamination corrupts them, and what better alternatives exist.

---

## The Evaluation Problem

Evaluating a language model is fundamentally different from evaluating most ML systems:

- **There is no single correct output**: For "Write a haiku about autumn", infinitely many responses are valid. You can't compute accuracy against a ground truth.
- **The output space is combinatorially large**: Unlike classification (pick from N classes), generation produces sequences from a vocabulary of tens of thousands, up to thousands of tokens long.
- **Quality is multidimensional**: A model can be accurate, fluent, helpful, safe, well-calibrated, and fast — these are all different and partially independent.
- **The evaluator is often another LLM**: Many modern evaluation methods use a strong frontier model as the judge. This introduces two distinct problems: *circularity*, when the same model is used to evaluate its own outputs; and *stylistic bias*, where any LLM judge systematically favours responses that resemble its own training distribution — even when judging a completely different model.

The field has converged on several families of evaluation approaches, each with genuine trade-offs.

---

## The Saturation Problem

Before examining individual benchmarks, it helps to understand why the field is constantly replacing them. The pattern repeats:

1. A benchmark is released, and models score 40–60%.
2. Over one to two years, frontier models reach 85–90%.
3. The benchmark becomes useless for discrimination — the gap between "good" and "great" models collapses into statistical noise.
4. A harder benchmark is released, and the cycle restarts.

MMLU (2021) is at ~91–95% for frontier models. HumanEval (2021) is above 95%. ARC-Challenge (2018) is essentially saturated. This is why evaluation research has accelerated: benchmarks have a shelf life.

---

## Agentic and AGI-Oriented Benchmarks

The most important shift in LLM evaluation over the past two years is the move from measuring what a model *knows* to measuring what it can reliably *do* — across multiple steps, with tools, in dynamic environments. These benchmarks are where the frontier is moving.

### SWE-bench Verified

SWE-bench (Jimenez et al., 2023) uses real GitHub issues from open-source Python repositories. A model must produce a git patch that fixes the issue, verified by running the repository's test suite. This is qualitatively different from knowledge benchmarks: it requires understanding an existing codebase, multi-file reasoning, reading error messages, and producing valid diff syntax.

**SWE-bench Verified** is a human-curated subset where each issue has been independently verified to be solvable and unambiguous.

The best scores come from **agentic scaffolds** (iterative file editing, test-running loops, bash execution) rather than single-shot generation. The benchmark has become a key measure of real software engineering ability, and scores advance rapidly as agent frameworks improve — the leaderboard at [swebench.com](https://swebench.com) is the authoritative source for current numbers.

### BFCL (Berkeley Function-Calling Leaderboard)

BFCL (Yan et al., 2024) evaluates a model's ability to call tools correctly — the core capability required for any LLM-based agent. It covers:

- **Simple function calling:** one function available, one call needed
- **Multiple function calling:** several functions provided, model must select the right one
- **Parallel function calling:** the model must make multiple API calls simultaneously from one query
- **Function relevance detection:** the model must recognise when no available function is appropriate

BFCL V3 added multi-turn, multi-step evaluation — sequences where the model must track conversation state, adapt to API errors, and chain multiple calls correctly. This is much closer to how agents are actually used in production.

A model may perform well on single-call scenarios but degrade significantly on parallel or chained calls, which is what BFCL V3 reveals.

### τ-bench (TAU-bench)

τ-bench (Yao et al., 2024) evaluates agents in dynamic, multi-turn conversations where both a **user** and **domain-specific API tools** are present. The agent must follow domain policies (e.g., airline booking rules), call the correct APIs in the correct order, and handle user clarifications — all while keeping track of a changing database state.

The benchmark uses a simulated user (another LLM) and checks the final database state against a target state.

**Key finding from the paper:** At publication time (2024), even state-of-the-art function-calling agents succeeded on fewer than 50% of tasks, and reliability degraded sharply under repeated trials (pass^8 < 25% in retail tasks). This points to a systematic problem: LLM agents are brittle — they can solve a task once, but performing it reliably across varied users is much harder.

### GAIA (General AI Assistants)

GAIA (Mialon et al., 2023) evaluates agents on real-world questions that require tool use (web browsing, calculator, code execution, file reading) and multi-step planning. Questions are categorised by difficulty:

- **Level 1:** Straightforward, single tool use
- **Level 2:** Multi-step reasoning with 2–3 tool calls
- **Level 3:** Complex, requiring 5+ tool calls and synthesis across multiple sources

Human performance is ~92% — far above early model baselines. Level 3 remains very hard for all current systems. GAIA is notable because it requires the integration of tools, long-horizon planning, and factual accuracy simultaneously — all things that fail in production agents. Current scores are maintained on the [GAIA Hugging Face leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

### ARC-AGI (Abstraction and Reasoning Corpus)

ARC-AGI (Chollet, 2019) is philosophically different from all other benchmarks. It does **not** test knowledge. Instead, it tests **fluid intelligence**: the ability to identify abstract rules from a handful of input-output grid examples and apply them to a new input.

The series has evolved significantly:
- **ARC-AGI-1** (2019–2024): passive pattern induction from grid examples. Pure LLMs score in single digits; the best reasoning systems with high compute reached ~75%. Human average: ~64%.
- **ARC-AGI-2** (2025): substantially harder. Leading reasoning systems at launch scored in single-digit percentages. Human panels (2+ humans per task) solved 100% of tasks.
- **ARC-AGI-3** (2026): now the active leaderboard benchmark. Unlike ARC-AGI-1/2 which test passive fluid intelligence, ARC-AGI-3 challenges AI **agents** to adapt on the fly to novel interactive environments — a significant increase in difficulty.

Every task across all versions is designed to be:
- Solvable by most humans in a few minutes with no specialist knowledge
- Resistant to memorisation (patterns are novel per task)
- Hard for pure LLMs (which rely on pattern-matching pre-training data)

The core insight: models that dominate knowledge-heavy benchmarks like GPQA still struggle on ARC-AGI tasks that any adult human solves in minutes. The capability gap reveals what scaling alone has not provided: efficient, sample-efficient generalisation from small amounts of novel evidence.

Current scores for all three ARC-AGI versions are at [arcprize.org/leaderboard](https://arcprize.org/leaderboard).

> ARC-AGI scores should always be quoted with the efficiency metric (cost per task). The leaderboard plots both dimensions simultaneously — a high score achieved at thousands of dollars per task is not the same as a high score achieved at human-comparable cost.

---

## Chatbot Arena and Elo Ratings

### The Elo System for LLMs

Chatbot Arena (LMSYS, 2023) uses real humans to evaluate models via pairwise comparison. Users interact with two anonymous models simultaneously and vote for which they prefer.

**The intuition:** Every model starts with the same score. When model A beats model B, A's score goes up and B's goes down. The key insight is that the adjustment depends on how *surprising* the result was — beating a much stronger opponent gives you a big boost; beating a much weaker one barely moves your score at all. Over thousands of comparisons, this self-corrects: models end up with scores that accurately reflect their relative quality, regardless of who happened to face whom.

Formally, the Elo rating system converts pairwise win rates into a single scalar ranking:

$$\text{Expected score}(A \text{ vs } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

After each match, ratings are updated:
$$R_A' = R_A + K(S_A - E_A)$$

where $S_A = 1$ (win), $0.5$ (tie), $0$ (loss); $E_A$ is the expected score; $K$ is the learning rate.

**Why Elo is valuable:**
- Human preference is the closest available proxy for real-world chat quality
- Real users, real queries — not cherry-picked benchmark prompts
- Covers all model dimensions simultaneously (accuracy, helpfulness, safety, personality)
- Harder to game than automated benchmarks, because gaming Elo requires gaming actual human preferences at scale

**Limitations:**
- High variance: thousands of battles per model are needed for stable estimates
- Selection bias: the Arena user population (largely technical, English-speaking) is not representative of all users
- Preferences reflect the user population's values, which may not match your specific use case
- "Agreeable" models that never push back or admit uncertainty may win Elo votes while being worse for tasks requiring precision

> Arena Elo scores shift every week as new model versions are released and accumulate votes. Always check [lmarena.ai](https://lmarena.ai) for current rankings.
{: .prompt-info }

---

## Knowledge and Reasoning Benchmarks

### MMLU (Massive Multitask Language Understanding)

MMLU (Hendrycks et al., 2021) is the most widely cited LLM benchmark. It consists of 57 subjects spanning STEM, humanities, social sciences, and professional domains. Each question is 4-choice multiple choice.

**The problem:** Frontier models now score 91–95%, exceeding the human expert ceiling of ~89–90%. The benchmark can no longer distinguish between top models, and its 4-choice format inflates scores (random guessing scores 25%).

**MMLU is now primarily useful for:**
- Tracking older or smaller models
- Longitudinal comparisons across years
- Detecting gross regressions in a fine-tuning pipeline

For discriminating frontier models against each other, MMLU is the wrong tool.

### MMLU-Pro

MMLU-Pro (Wang et al., 2024) directly addresses MMLU's saturation by:
- Expanding from 4 to **10 choices** per question, reducing the guessing floor from 25% to 10%
- Replacing trivia-style questions with reasoning-heavy problems
- Removing low-quality questions from the original MMLU

**Effect:** Frontier models that scored 90%+ on MMLU score 60–80% on MMLU-Pro — a 16–33 percentage point drop according to the paper. This spread is large enough to meaningfully separate models.

**Additional benefit:** Score sensitivity to prompt format drops from 4–5 percentage points on MMLU to ~2 pp on MMLU-Pro. Chain-of-thought reasoning now helps more than direct answering (unlike MMLU, where CoT was optional), confirming the benchmark is actually testing reasoning rather than pattern recognition.

> **For frontier model comparisons, prefer MMLU-Pro over MMLU.** The 10-choice format, reduced contamination, and greater score spread make it far more informative. When reading third-party comparisons, check whether they used MMLU or MMLU-Pro — the scores are not interchangeable.
{: .prompt-warning }

### GPQA (Graduate-Level Google-Proof Q&A)

GPQA (Rein et al., 2023) contains 448 questions written by domain experts across biology, chemistry, and physics, designed to be difficult enough that even PhDs cannot easily look up the answer.

Human expert accuracy (PhD students in the relevant domain): ~65–70%.

The "Diamond" subset is harder and remains one of the more discriminating non-task-specific benchmarks available — top reasoning models now score above human expert baselines on it, but meaningful spread between model tiers persists. Unlike MMLU, GPQA questions require deep domain reasoning that is harder to memorise, though contamination risk increases as the dataset ages. Current scores are reported in model technical reports and tracked on [livebench.ai](https://livebench.ai).

### MATH and AIME

**MATH** (Hendrycks et al., 2021) contains 12,500 competition-level mathematics problems across algebra, geometry, number theory, and calculus. It is now largely saturated for frontier reasoning models, which exceed 90% on the full set.

**AIME** (American Invitational Mathematics Examination) is a harder target: 15 free-response integer problems (answer range 0–999), drawn from an actual high-school mathematics competition held annually. There is no multiple-choice guessing advantage.

AIME is now the primary discriminator for mathematical reasoning. Non-reasoning models score in the 10–20% range; top reasoning models with extended compute reach well above 85%. The gap between the two model classes is dramatic and unambiguous — larger than on almost any other benchmark. Current scores are tracked at [Chatbot Arena's math leaderboard](https://lmarena.ai) and in individual model technical reports.

> **Important caveat:** AIME results depend heavily on the number of samples and the compute budget. A model's score with high compute (many reasoning traces + best-of-k selection) can be dramatically higher than its single-attempt pass@1. Always check whether a published AIME score is pass@1, pass@k, or majority@k before comparing models.
{: .prompt-warning }

### HLE (Humanity's Last Exam)

HLE (Phan et al., 2025) is the hardest publicly released closed-form academic benchmark to date. Published in *Nature* in January 2026, it contains 2,500 questions across mathematics, the natural sciences, and humanities, all written by domain experts. The key design constraint: each question has a single unambiguous correct answer that **cannot be quickly retrieved via internet search**.

At launch, frontier models scored in the **low single digits** — even the strongest models of the time were wrong on over 90% of questions. As reasoning models have improved, scores have risen, but HLE remains one of the few benchmarks where frontier models are demonstrably far from ceiling. Current scores are tracked at [lastexam.ai](https://lastexam.ai) (which also maintains the live HLE-Rolling variant with continuously updated questions).

HLE is specifically designed to resist the saturation problem. Its subject-matter-expert authorship makes contamination difficult, and the breadth of subjects (covering niche graduate and doctoral-level content across dozens of fields) means no model can specialise for it easily.

---

## Code Benchmarks

### HumanEval

HumanEval (Chen et al., 2021) is OpenAI's benchmark for coding ability. It contains 164 handwritten Python programming problems evaluated by functional execution (pass all tests = correct).

**pass@k metric:**

$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where $n$ is the number of samples generated, $c$ is the number that pass all tests, and $k$ is the number of attempts allowed.

**Current state:** Frontier models score 95%+ on pass@1. HumanEval is saturated. It remains useful for testing small or quantised models, but is not appropriate for frontier comparisons.

**Core limitations:**
- Only 164 problems — high statistical variance
- All Python, function-completion format — doesn't test debugging, multi-file reasoning, or code review
- Near-certain contamination for any model trained on the open web post-2021

### LiveCodeBench

LiveCodeBench (Jain et al., 2024) directly addresses HumanEval's saturation and contamination problems. It continuously pulls new competitive programming problems from Codeforces, LeetCode, and AtCoder **after** a specified cutoff date. Since problems post-date the model's training cutoff, they cannot have been memorised.

Problems are harder than HumanEval — they require multi-step algorithmic reasoning, not just function completion — and the rolling-update design means the benchmark stays current as models improve.

For frontier code model comparisons, LiveCodeBench is the correct benchmark to cite.

### SWE-bench Verified

SWE-bench (Jimenez et al., 2023) uses real GitHub issues from open-source Python repositories. A model must produce a git patch that fixes the issue, verified by running the repository's test suite. This is qualitatively different from HumanEval: it requires understanding an existing codebase, multi-file reasoning, reading error messages, and producing valid diff syntax.

**SWE-bench Verified** is a human-curated subset where each issue has been independently verified to be solvable and unambiguous.

The best scores come from **agentic scaffolds** (iterative file editing, test-running loops, bash execution) rather than single-shot generation. The benchmark has become a key measure of real software engineering ability, and scores advance rapidly as agent frameworks improve — the leaderboard at [swebench.com](https://swebench.com) is the authoritative source for current numbers.

---

## Instruction Following

### IFEval (Instruction Following Evaluation)

IFEval (Zhou et al., 2023) evaluates a model's ability to follow **verifiable formatting constraints** — the kind of constraints that appear in real system prompts but are easy to check programmatically:

- "Write your response in more than 400 words"
- "Mention the word 'sustainable' at least three times"
- "Do not use any bullet points"
- "Wrap your response in JSON with the key 'answer'"

This is evaluated automatically — no judge model, no subjectivity. A response either satisfies the constraint or it does not.

IFEval tests a capability that is distinct from knowledge or reasoning: the model's ability to maintain format discipline across multi-turn conversations and complex system prompts. Models that score well on MMLU-Pro can still fail IFEval, because instruction following is a trained behaviour separate from knowledge recall.

IFEval is now a standard component of the Hugging Face Open LLM Leaderboard v2 precisely because it tests something orthogonal to the knowledge benchmarks.

---

## Factual Accuracy

### SimpleQA

SimpleQA (Wei et al., 2024) is OpenAI's benchmark for **short, verifiable factual questions** — the kind of question that has a single correct answer that can be checked against authoritative sources:

- "What year was the Hubble Space Telescope launched?"
- "Who wrote *The Road*?"

The surprise: frontier models perform surprisingly poorly on this benchmark. Models that score 90%+ on MMLU often score 60–75% on SimpleQA, because factual accuracy on specific, checkable claims is different from pattern-matched test performance. SimpleQA also measures **hallucination rate** — cases where the model confidently gives a wrong answer instead of saying it doesn't know.

SimpleQA is useful as a calibration check: a high SimpleQA score indicates a model that is grounded in specific facts, not just pattern-completion over training data.

---

## Contamination-Resistant Benchmarks

### LiveBench

LiveBench (White et al., 2024) is designed from the ground up to be resistant to both contamination and the biases of LLM-as-judge:

1. Questions are **continuously updated from recent sources** — competition problems from the past month, recent arXiv papers, current news events — so models trained before the question-release date cannot have seen them.
2. Answers are scored against **objective ground truth** rather than by a judge model.
3. The benchmark spans math, coding, reasoning, language, instruction following, and data analysis.

The rolling-update design means LiveBench stays discriminating as models improve, unlike static benchmarks that saturate. As of its launch, top models scored below 70% — far below the saturation threshold.

---

## Chat and Human Evaluation

### MT-Bench

MT-Bench (Zheng et al., 2023) evaluates multi-turn conversation quality across 80 questions in 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities). Each conversation has 2 turns.

**Evaluation:** GPT-4 rates each response on a 1–10 scale.

**Problem:** Using GPT-4 as a judge transfers GPT-4's biases — it prefers longer responses, its own hedging style, and formats similar to its own outputs. Models fine-tuned to score well on MT-Bench may game the GPT-4 judge without being genuinely better assistants.

MT-Bench is now superseded by Arena-Hard (a harder, curated subset of real arena questions) for most frontier comparisons.

### AlpacaEval

AlpacaEval compares model outputs against a reference model using GPT-4 as judge. **AlpacaEval 2.0 length-controlled (LC)** attempts to correct for the systematic bias toward longer responses by measuring win rate at similar response lengths. It correlates reasonably well with Chatbot Arena rankings and is faster to run than full human evaluation.

---

## Benchmark Contamination

The most serious systemic problem in LLM evaluation: **test set contamination**.

### What Is Contamination?

Modern LLMs are trained on web-scale data — Common Crawl, GitHub, Wikipedia, academic papers, and more. Many evaluation benchmarks are published on the web. If training data contains benchmark questions and answers, the model has "seen the test" — its performance measures memorisation, not generalisation.

**Forms of contamination:**
1. **Direct**: The exact question-answer pair appears in training data
2. **Indirect**: A related text (study guide, answer key, discussion post) appears in training data
3. **Temporal**: A benchmark released before the training cutoff has been scraped

### Estimating Contamination

**n-gram overlap:** Check if $k$-grams from benchmark questions appear in training data. GPT-4's technical report used 50-gram overlap. Limitation: paraphrased contamination doesn't show up in n-gram checks.

**Canonical example:** MMLU was released in 2021. LLaMA 3's technical report acknowledged MMLU contamination and reported decontaminated scores approximately 5–7 points lower than unadjusted scores.

> **When comparing models on MMLU, treat any model trained post-2021 as potentially contaminated.** A claimed 91% may genuinely be 84–86% on a clean test set. When the gap between two models is under 3 percentage points, contamination alone can explain the difference. Use MMLU-Pro, GPQA, LiveBench, or HLE for comparisons where contamination levels are unknown.
{: .prompt-warning }

**Performance cliff at cutoff date:** If performance on time-indexed questions drops sharply for questions that post-date the training cutoff, the pre-cutoff scores were likely contaminated.

### The Contamination Arms Race

In response to contamination:
- **LiveBench**: monthly new questions from recent competitive math, arXiv papers, current events
- **LiveCodeBench**: new competitive programming problems from contests held after the training cutoff
- **HLE**: subject-matter-expert questions on niche graduate-level topics, written specifically to resist retrieval
- **ARC-AGI**: tasks are generated per-evaluation with novel abstract patterns, making memorisation structurally impossible

The field is moving toward dynamic or continuously updated benchmarks, but adoption is slow because historical comparability matters to researchers.

---

## How to Evaluate Your Own Fine-Tuned Model

Standard benchmarks measure general base capabilities. When evaluating a fine-tuned model for a specific use case, the approach is different.

### Task-Specific Automated Metrics

**For extraction/classification:**
- Precision, recall, F1 on held-out labelled examples
- Confusion matrix analysis to understand error types, not just aggregate error rates

**For generation (summarisation, translation):**
- ROUGE (n-gram overlap): fast, widely used, imperfect — doesn't capture meaning
- BERTScore: semantic similarity using embeddings — better than n-gram metrics
- Perplexity on held-out domain data: lower is better

**For code generation:**
- pass@k on a held-out test set drawn from your actual codebase
- Execution-based evaluation: does the code run and produce correct output?

**For instruction following:**
- Adapt IFEval's format-verification approach to your specific system prompt constraints
- Measure **compliance rate** separately from **quality** — a model can comply with all format rules while still giving poor answers

### LLM-as-Judge

Use a strong frontier model as the judge:

```
System: You are an expert evaluator.
Rate the following response on accuracy (1-5), completeness (1-5), and conciseness (1-5).

Prompt: [user question]
Response: [model output]

Provide your ratings in JSON: {"accuracy": ..., "completeness": ..., "conciseness": ...}
```

**Calibration is mandatory:** Before using LLM-as-judge at scale, validate it against human-labelled examples. Check that judge scores correlate with human preferences (Spearman ρ > 0.7 is acceptable; > 0.85 is good).

> **Do not trust LLM-as-judge scores without human calibration.** LLM judges systematically prefer responses that resemble their own training distribution. Run the judge on 50–100 human-labelled examples from your domain and compute the Spearman correlation. If ρ < 0.65, either choose a different judge model or revise your rubric.
{: .prompt-tip }

**Positional bias:** LLM judges favour the first response when evaluating A vs B. Use randomised presentation order and average over both orderings.

### Human Evaluation

The gold standard, but expensive and slow. Best practices:

1. **Representative sample**: 200–500 examples drawn from your actual use-case distribution, not hand-picked examples
2. **Multiple raters**: at least 2–3 per example; report inter-rater agreement (Cohen's κ)
3. **Blind evaluation**: raters should not know which model produced which response
4. **Pairwise preferred**: "A or B?" is more reliable than absolute rating (1–10)
5. **Specific rubrics**: define what "good" means — accuracy, tone, length, format compliance — before annotators begin

### The Evaluation Pyramid

For production systems, combine methods in layers:

```
                    ┌──────────────────┐
                    │   Human eval     │ ← ground truth, expensive
                    │   (N=200–500)    │
                    └────────┬─────────┘
                    ┌────────┴─────────┐
                    │  LLM-as-judge    │ ← fast, scalable, biased
                    │  (N=2000–5000)   │
                    └────────┬─────────┘
                    ┌────────┴─────────┐
                    │  Automated       │ ← instant, narrow
                    │  metrics         │
                    │  (N=all)         │
                    └──────────────────┘
```

Run automated metrics on every model version (CI/CD). Run LLM-as-judge on weekly snapshots. Run human evaluation on major releases.

---

## What Benchmarks Miss

Even the best benchmarks fail to capture:

**Calibration:** Does the model know what it doesn't know? A model that answers every question confidently but is wrong 30% of the time is more dangerous than one that says "I'm not sure" in those cases. Expected Calibration Error (ECE) and Brier Score are almost never reported on LLM leaderboards.

**Reliability over repeated trials:** ARC-AGI's pass^k metric and τ-bench's pass^8 surface this. A model that solves a task correctly 60% of the time is fundamentally unreliable for production. Single-trial benchmark scores hide this.

**Robustness to prompting:** The same question phrased differently can shift scores by 5–10 percentage points. MMLU-Pro reduces this to ~2 pp, but production systems run on specific prompts that may not match any benchmark's phrasing.

**Long-context faithfulness:** Most benchmarks use short contexts. Production RAG pipelines and coding assistants work with 50k–200k token contexts. RULER and similar long-context evaluations exist but are rarely included in headline leaderboards.

**Latency and cost:** A model that scores 95% on GPQA but takes 30 seconds to respond is often less useful than an 85% model that responds in 2 seconds. Benchmarks almost never report these, though ARC-AGI-2 is a notable exception in explicitly reporting cost-per-task alongside accuracy.

**Edge cases and failure modes:** A model's average accuracy across a benchmark tells you nothing about how it fails. Production failures are almost always in the tail — unusual inputs, adversarial prompts, very long chains of reasoning — not the average case.

---

## Key Takeaways

| Benchmark | What it measures | Status |
|---|---|---|
| MMLU | General knowledge, 57 subjects | Saturated for frontier models; use MMLU-Pro instead |
| MMLU-Pro | Reasoning-heavy, 10-choice version of MMLU | Current standard for knowledge/reasoning |
| GPQA Diamond | Graduate-level expert science | Still discriminating; contamination-resistant |
| HLE | Frontier expert knowledge, ungooglable | Hardest closed-form benchmark; most frontier-safe |
| AIME | Competition mathematics | Primary discriminator for math/reasoning models |
| HumanEval | Python function completion | Saturated; use LiveCodeBench instead |
| LiveCodeBench | Contamination-resistant competitive coding | Current standard for coding |
| SWE-bench Verified | Real GitHub issue fixing | Best proxy for real software engineering |
| IFEval | Format instruction following | Tests orthogonal capability; often overlooked |
| SimpleQA | Short verifiable factual questions | Measures hallucination and factual grounding |
| LiveBench | Contamination-resistant multi-domain | Rolling updates keep it current |
| Chatbot Arena Elo | Human preference, pairwise voting | Closest ground truth for chat quality |
| BFCL | Tool/function calling accuracy | Key for agent evaluation |
| τ-bench | Multi-turn agentic task completion | Tests agent reliability, not just capability |
| ARC-AGI | Fluid intelligence, novel pattern induction | Tests what scaling hasn't solved |

**For your own fine-tune:** use task-specific automated metrics + LLM-as-judge calibrated against human labels + periodic human evaluation on representative examples. Benchmarks miss calibration, reliability, robustness, and cost — all of which matter in production.

---

## Further Reading

- [Fine-Tuning and Adaptation](/posts/finetuning-and-adaptation) — how to train the model you'll need to evaluate
- [LLM Serving in Depth](/posts/llm-serving-in-depth) — production considerations beyond eval scores
- MMLU: Hendrycks et al., [arXiv 2009.03300](https://arxiv.org/abs/2009.03300)
- MMLU-Pro: Wang et al., [arXiv 2406.01574](https://arxiv.org/abs/2406.01574)
- HumanEval: Chen et al., [arXiv 2107.03374](https://arxiv.org/abs/2107.03374)
- HLE: Phan et al., [arXiv 2501.14249](https://arxiv.org/abs/2501.14249)
- LiveBench: White et al., [arXiv 2406.19314](https://arxiv.org/abs/2406.19314)
- Chatbot Arena: Zheng et al., [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- GPQA: Rein et al., [arXiv 2311.12022](https://arxiv.org/abs/2311.12022)
- SWE-bench: Jimenez et al., [arXiv 2310.06770](https://arxiv.org/abs/2310.06770)
- BFCL: Yan et al., [gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/leaderboard.html)
- τ-bench: Yao et al., [arXiv 2406.12045](https://arxiv.org/abs/2406.12045)
- IFEval: Zhou et al., [arXiv 2311.07911](https://arxiv.org/abs/2311.07911)
- ARC-AGI: Chollet, [arcprize.org](https://arcprize.org/arc-agi)
