---
title: A Deep Dive into Andrey Karpathy's Nanochat
date: 2025-10-28 09:00:00 +0100
categories: [Gen AI, LLM Training]
tags: [Gen AI, LLM Training, LLM Pre-Training, LLM Post-Training, Nanochat, Andrey Karpathy]
image:
  path: /assets/img/nanochat.png
  alt: "A Deep Dive into Andrey Karpathy's Nanochat"
---

The **nanochat** project demonstrates the possibility of achieving functional conversational AI at a relatively micro-scale using modern hardware and hyper-efficient engineering.

The baseline model, often referred to as the "$100 tier" achieved via the speedrun.sh script, is a micro-model featuring **1.9 billion parameters**. The training regimen for this tier is highly optimized, targeting approximately four hours of computation time on a single NVIDIA 8xH100 GPU node, costing around $24 per hour. This training uses roughly **38 billion tokens** from an open dataset.

While this micro-model is described as a "little ChatGPT clone" capable of basic conversation, writing stories, poems, and answering simple questions, the designer notes that it significantly outperforms models like GPT-2 from 2019 but falls dramatically short of contemporary commercial offerings, such as GPT-5.1. The flagship production model, identified as "d32," possesses **32 layers** in its Transformer architecture and also uses 1.9 billion parameters, but requires a much longer training commitment, totaling approximately 33 hours and costing around $800 to reach a higher level of coherence and capability.

## **Foundational Architecture**

### **Minimalist Engineering and Language Choices**

The design philosophy behind nanochat emphasizes accessibility and minimal dependencies. The entire project is intentionally clean and hackable, allowing users to understand and tamper with every part of the process.

The codebase is predominantly written in Python, leveraging the PyTorch framework for its deep learning computations. However, the engineering design strategically delegates performance-critical tasks to more efficient compiled languages. While Python serves as the high-level language directing the workflow, the lower-level numerical operations occur efficiently within libraries written in C, CUDA, or other optimized languages for the target hardware. Crucially, the component responsible for tokenizer training, which is a computationally demanding process over vast text datasets, is implemented in **Rust** to maximize speed and ensure reliable performance during the build phase.

### **Model Tiers and Scaling Economics**

The architecture is a standard Transformer model, differentiated by depth and training budget. The "d32" designation for the flagship model indicates it employs **32 layers** in its neural network.

The high efficiency of the $100 tier (4 hours) is contingent upon using highly optimized modern hardware (the 8xH100 node) and the streamlined, dependency-lite code. The effective cost-efficiency of nanochat demonstrates that by pairing highly efficient systems code with specialized, powerful cloud compute, developers can radically minimize the time and cost associated with obtaining a functional LLM prototype.

The following table details the two primary development tiers:

Nanochat Model Tiers and Resource Requirements

| Model Tier | Parameter Count (Approx.) | Layer Depth (d) | Training Cost (Approx.) | Training Time (8xH100) | Data Tokens |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Speedrun ($100) | 1.9 Billion | (Fewer than 32, e.g., 20\) | \~$100 | \~4 Hours | 38 Billion |
| d32 Flagship | 1.9 Billion | 32 | \~$800 | \~33 Hours | 38 Billion (Likely higher token count) |

The successful deployment of a conversational model within the constrained $100 budget highlights a critical shift in LLM accessibility. This high leverage is achieved because nanochat ensures maximal utilization of the expensive GPU resources, turning what was once a multi-day effort into an afternoon project and radically lowering the barrier to entry for experimentation.

## **III. Data Preparation and Efficient Tokenization System**

### **Rust-Based BPE Tokenization for Performance**

Tokenization is the critical initial step, converting raw text into numerical tokens that the model can process. nanochat utilizes the **Byte Pair Encoding (BPE)** algorithm, which intelligently creates an efficient vocabulary by iteratively merging the most frequently occurring adjacent byte sequences. This generates subword units capable of representing both common words and handling rare words gracefully.

For efficiency, the design adopts a deliberate choice: a **Rust-based BPE implementation (RustBPETokenizer)** is specifically employed for the training phase. The rationale for using Rust stems from the fact that tokenizer training involves extensive, computationally demanding operations over large text datasets. Rust's performance and reliability ensure this initial build phase is executed as quickly as possible. While the training uses Rust BPE, the architecture permits a dual approach, with subsequent inference phases potentially leveraging optimized libraries such as OpenAI's tiktoken for continuous, high-speed execution.

### **Data Sources: Pretraining vs. Conversational Adaptation**

The training relies on a multi-modal data strategy that partitions data into distinct training phases to impart different knowledge and skills.

1. **Pretraining Data:** The foundational knowledge of the model is acquired during pretraining on massive web text. This phase uses text from many webpages within the **FineWeb-EDU dataset**. Specifically, the training targets a re-packaged, fully shuffled shard of the [karpathy/fineweb-edu-100b-shuffle sample-100B](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) version. For the baseline $100 tier, the training involves learning from approximately 38 billion tokens.  
2. **Midtraining/SFT Data:** These later stages focus on adapting the model from a passive text predictor into an active, reasoning conversational agent. Key datasets include:  
   * **[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk):** Provides a large volume of general conversational context, totaling 460K rows in the default mixture, teaching basic dialogue patterns.  
   * **[MMLU (Multiple-Choice Data)](https://huggingface.co/datasets/cais/mmlu):** Includes 100K rows of problems drawn from standardized multiple-choice quizzes, essential for adapting the model to reasoning benchmarks.  
   * **[GSM8K (Math/Tool Use Data)](https://huggingface.co/datasets/openai/gsm8k):** Consists of 8K rows focused on teaching simple mathematics and the invocation of the external calculator tool.

### **Structuring Dialogue with Special Tokens**

For nanochat to effectively manage multi-turn conversations and respond correctly as an assistant, it must be explicitly taught the structure of dialogue. This is achieved through the use of special tokens, following the established format used by large-scale commercial models.

These critical tokens delineate roles and context: 

```xml 
<|bos|> (Beginning of Sequence), <|user_start|>, <|user_end|>, and <|assistant_start|>. 
```

It is notable that these conversational tokens are absent during the base pretraining phase, which focuses purely on general language prediction from internet documents. They are deliberately introduced during the Midtraining and Supervised Fine-Tuning (SFT) stages. This introduction is a sophisticated pedagogical maneuver: by inserting these structured tokens and fine-tuning the model specifically on dialogue data, nanochat rapidly shifts the model's objective from passive text compression to active, role-aware response generation. This step is indispensable for transforming the base language predictor into a competent chatbot.

## **The Multi-Stage Training Protocol: Mastering Chat Behavior**

The nanochat training pipeline is a sequential, three-stage process, managed by dedicated scripts ([base_train.py](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py), [mid_train.py](https://github.com/karpathy/nanochat/blob/master/scripts/mid_train.py), [chat_sft.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py)). This structured curriculum is the primary factor enabling micro-models to achieve useful conversational abilities.

### **Stage 1: Base Model Pretraining (Knowledge Acquisition)**

This is the most computationally intensive phase, where the model gains core knowledge by being trained to compress vast quantities of web text through the task of next-token prediction.

For an example model size of approximately 560 million parameters, the training process aims to consume *11.2B* tokens, adhering to Chinchilla scaling law recommendations. Given that each optimization step processes approximately *0.5M* tokens (32 rows of 2048 tokens across 8 GPUs), this requires roughly *21,400 iterations* and an estimated computational burden of 4e19 FLOPs. This phase typically requires about 3 - 4 hours of dedicated compute time.

### **Stage 2: Midtraining (Adaptation for Structure and Tools)**

The midtraining stage is short, taking only about eight minutes, yet it provides the maximum pedagogical leverage. The rationale for this stage is that micro-models cannot reliably acquire critical structural and reasoning capabilities simply from random internet data; these skills must be explicitly taught.

Midtraining adapts the model in two crucial ways:

1. **Multiple Choice Understanding:** It teaches the model the specific algorithm required for multiple-choice quizzes (i.e., associating a choice with a letter, A, B, C, D, and emitting only the correct choice). This skill is vital because many common model evaluations, such as MMLU, rely on this quiz format.  
2. **Tool Use Integration:** The model is taught to invoke external functions, such as the Python interpreter, for complex calculations or code generation. This capability is enabled by embedding the required commands within special, recognized tokens, specifically: 

```xml 
<|python_start|> and <|python_end|> 
```

This mechanism allows the model to tackle problems requiring calculation, such as those found in the GSM8K dataset.

The fact that this brief, 8-minute phase contributes significantly to performance metrics (resulting in a 2x MMLU score increase compared to a pretrain-only model) demonstrates a strategic optimization of the training curriculum. The data mixture is engineered to rapidly bootstrap crucial structural and tool-use skills that would otherwise demand orders of magnitude more generalized pretraining data to emerge organically, directly conserving the limited $100 compute budget.

### **Stage 3: Supervised Fine-Tuning (SFT) and Identity Injection**

The SFT stage is dedicated to refining the model's behavior, alignment, and conversational style. It uses curated conversational data where explicit human instructions and correct assistant answers are provided.

A major functional component introduced in nanochat is the straightforward injection of arbitrary identity and specific capabilities. Developers can generate synthetic conversations—for instance, 1000 multi-turn conversations can be created in minutes using an API from a larger LLM. This custom synthetic data, which can teach the model self-awareness or specific new skills (such as letter counting), is simply added into the data mixtures for both midtraining and SFT. This functionality provides a highly accessible "creativity knob" for customizing the nanochat instance to a developer’s precise needs, demystifying the specialized process of alignment that commercial models often employ.

## **Optimization and Distribution**

Despite its billing as a "minimal" project, nanochat incorporates highly advanced systems engineering choices to ensure efficiency and effective utilization of the high-end hardware, particularly concerning parameter optimization and distributed training.

### **Hybrid Optimization Strategy (Muon \+ AdamW)**

Training large Transformer models requires robust optimization algorithms. Nanochat employs a sophisticated **hybrid optimization strategy**, recognizing that different types of parameters respond best to different optimization methods based on their geometric structure.

The Hybrid Optimization Strategy

| Parameter Group | Dimensions | Optimizer Used | Justification/Benefit |
| :---- | :---- | :---- | :---- |
| Attention/MLP Weight Matrices (Linear layers) | 2D | Muon Optimizer | Provides essential orthogonalization properties for 2D matrices, ensuring stability and efficient gradient flow. |
| Embeddings and Language Model Head | 1D | Distributed AdamW | Leverages adaptive learning rates, maintaining separate update rates for each parameter based on gradient statistics. |

The **Muon optimizer** is specifically designed for the two-dimensional weight matrices found in the attention and Multi-Layer Perceptron (MLP) layers, which form the bulk of the Transformer's structure. For the one-dimensional parameters, such as the token embeddings and the language model head, **AdamW (Adam with decoupled weight decay)** is employed, as its adaptive learning rates prove more effective for these groups.

### **Distributed Training and Memory Efficiency**

The training is executed efficiently across the powerful 8xH100 GPU node using torchrun for distributed processing. To manage the substantial memory requirements of training a 1.9 billion parameter model, especially the intermediate optimizer states, nanochat implements a distributed version of AdamW that follows **ZeRO-2 principles (Zero Redundancy Optimizer Stage 2)**.

ZeRO-2 works by sharding the optimizer states (momentum and variance vectors) across all eight available GPUs. This technique is critical, as it reduces the memory footprint consumed by the optimizer states by a factor equivalent to the number of GPUs. The implementation of such a technically sophisticated, memory-efficient distributed optimization system within a minimal codebase underscores that performance engineering was not compromised for simplicity. This approach ensures maximal utilization of the expensive compute node, which is necessary to hit the ambitious 4-hour, $100 training target.

## **Inference Mechanics and Real-Time Efficiency**

Efficient inference is paramount for deploying a real-time conversational agent. nanochat integrates advanced techniques, notably Key-Value Caching and optimized sampling strategies, to minimize latency and maintain generation quality.

### **Key-Value (KV) Caching Deep Dive**

In the process of autoregressive generation, where the model generates one token at a time based on all previous tokens, a standard, straightforward transformer implementation would redundantly re-calculate the self-attention mechanism over the entire growing sequence for every new token. This repetition leads to high computational cost and rapidly increasing latency as the generated sequence lengthens.

KV caching resolves this compute overlap by storing the intermediate states of the attention layers during inference. Specifically, in the transformer attention mechanism, each token generates Query (Q), Key (K), and Value (V) vectors. During subsequent steps, the Keys and Values calculated for all previous tokens (both the initial prompt and already generated tokens) are stored and reused, eliminating redundant computation.

In nanochat, this critical KV cache is architecturally defined as a 6-dimensional tensor with the shape:

cache_shape = (L, 2, B, H, T, D)

where *L* is the number of transformer layers; *2* is the separate storage dimension for Keys and Values; *B* is the batch size; *H* is the number of attention heads; *T* is the maximum sequence (context) length; and *D* is the head dimension. This highly specific architectural implementation minimizes inference latency by substituting expensive repeated matrix multiplications with fast memory lookups, ensuring that the model remains fast even when generating long responses.

### **Generation and Sampling Strategies**

Token generation requires careful management of randomness to balance coherence (a deterministic output) with creativity (variability). Nanochat achieves this balance using a combination of Temperature and Top-K sampling.

The generation process follows a defined flow:

1. The transformer outputs logits (raw prediction scores) across the entire vocabulary.  
2. **Top-K Filtering:** The tokens are sorted by logit value, and only the top *K* most likely tokens are retained, restricting the pool of potential candidates.  
3. **Temperature Scaling:** A temperature value *T* is applied to the logits of these retained *K* tokens. A higher temperature (up to 2.0) flattens the probability distribution, increasing the likelihood of selecting less probable tokens and resulting in greater variability; a lower temperature (down to 0.0) sharpens the distribution, leading to more deterministic and focused output.  
4. The probabilities are renormalized to sum to 1.0.  
5. The final token is sampled from this restricted, scaled distribution.

The integration of an optimized KV cache architecture and standardized sampling strategies demonstrates a commitment to designing not just a research model, but a complete blueprint for deploying an efficient, low-latency, real-time LLM service.


## **Conclusion**

The nanochat project successfully delivers a complete, deployable LLM solution optimized for minimal compute constraints. It achieved its goal of creating "the best ChatGPT $100 can buy" by demonstrating profound efficiency across every stage of the pipeline: utilizing high-performance Rust for tokenization, adopting a strategic hybrid optimization approach (Muon + AdamW) with ZeRO-2 for memory efficiency, and implementing low-latency inference mechanics like the 6D KV cache. The project serves as a comprehensive, end-to-end blueprint for contemporary LLM development, illustrating how transformers, tokenization, and fine-tuning must interact cohesively to form a functional, real-world system.