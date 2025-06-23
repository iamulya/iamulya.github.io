---
title: "Chapter 8 - Multi-Agent Orchestration Patterns"
date: "2025-06-23 14:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

So far, we have explored the essential building blocks of the Agents SDK: `Agent`, `Runner`, `Tool`, and `Handoff`. We've built individual agents that can perform tasks and delegate control. Now, we will elevate our perspective from single agents to entire systems. **Orchestration** is the art of defining the flow of control and data between multiple agents to solve a complex problem.

The Agents SDK is intentionally unopinionated about how you orchestrate. Its Python-first design gives you the freedom to implement any pattern you can imagine. This chapter will explore three fundamental and powerful orchestration patterns that serve as the foundation for almost any agentic application you might build:

1.  **Deterministic Flows (Chaining):** Creating predictable, step-by-step pipelines.
2.  **LLM as a Judge (Critique and Refinement):** Using agents to iteratively improve quality.
3.  **Parallelization (Fan-Out/Fan-In):** Executing multiple agents concurrently for speed and diversity.

Mastering these patterns will allow you to move from building individual AI workers to designing a collaborative AI workforce.

## Deterministic Flows (Chaining)

The simplest way to orchestrate multiple agents is to chain them together in a fixed sequence. The output of one agent becomes the input for the next. This pattern is ideal when a complex task can be broken down into a series of predictable, dependent steps. It offers maximum control and predictability at the cost of the flexibility that comes from letting the LLM decide the next step.

Let's build a simple content creation pipeline. A `BrainstormerAgent` will generate ideas for a blog post, and then a `WriterAgent` will take the chosen idea and write the post. The flow is controlled entirely by our Python code.

```python
#
# A deterministic chain: Brainstormer -> Writer
#

from pydantic import BaseModel, Field
from agents import Agent, Runner, trace
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

class BlogIdeas(BaseModel):
    ideas: list[str] = Field(description="A list of three creative blog post titles.")

def main():

    # 1. Define the first agent in the chain
    brainstormer_agent = Agent(
        name="Brainstormer",
        instructions="You are an expert idea generator. Generate creative blog post titles based on the user's topic.",
        model=DEFAULT_LLM,
        output_type=BlogIdeas # We use a structured output
    )

    # 2. Define the second agent in the chain
    writer_agent = Agent(
        name="Writer",
        instructions="You are a professional writer. Write a short, engaging blog post (2-3 paragraphs) based on the provided title.",
        model=DEFAULT_LLM 
    )

    # --- Code-driven Orchestration ---
    with trace("Chained Blog Writing Workflow"):
        # Step 1: Run the brainstormer
        print("--- Running Brainstormer Agent ---")
        topic = "the future of renewable energy"
        brainstorm_result = Runner.run_sync(brainstormer_agent, f"Topic: {topic}")

        # Extract the structured output
        ideas_output: BlogIdeas = brainstorm_result.final_output
        print(f"Generated Ideas: {ideas_output.ideas}")

        # Choose an idea (can be done by user or logic)
        chosen_title = ideas_output.ideas[0]
        print(f"
Selected Title: '{chosen_title}'")

        # Step 2: Run the writer with the output of the first agent
        print("
--- Running Writer Agent ---")
        write_result = Runner.run_sync(
            writer_agent,
            f"Please write a blog post titled: '{chosen_title}'"
        )

        print("
--- Final Blog Post ---")
        print(write_result.final_output)

if __name__ == "__main__":
    main()
```

This pattern is simple, robust, and easy to debug. The flow of control is explicit in your code. You can chain as many agents as necessary, creating sophisticated pipelines for tasks like report generation (research -> outline -> draft -> format) or data processing.

![*A deterministic chain where the output of one agent is the input to the next.*](/assets/img/2025-06-23-multi-agent-orchestration-patterns/figure-1.png)


## LLM as a Judge (Critique and Refinement)

A hallmark of intelligence is the ability to self-correct. The "LLM as a Judge" pattern formalizes this into an iterative loop. One agent (the "generator") produces work, and a second agent (the "judge" or "evaluator") critiques it. This process repeats until the judge is satisfied with the quality. This is a powerful technique for dramatically improving the quality and reliability of LLM outputs.

Let's build a system where a `CoderAgent` writes a Python function, and a `ReviewerAgent` checks it for correctness and provides feedback until the code is approved.

```python
#
# An iterative refinement loop using a "judge" agent.
#
from typing import Literal
from pydantic import BaseModel, Field
from agents import Agent, Runner, trace, TResponseInputItem
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

class CodeEvaluation(BaseModel):
    score: Literal["pass", "fail"] = Field(description="Does the code meet the requirements and seem correct?")
    feedback: str = Field(description="If the score is 'fail', provide concise, actionable feedback for how to fix the code.")

def main():

    # 1. The Generator Agent
    coder_agent = Agent(
        name="Python Coder",
        instructions="You are a skilled Python developer. Write a single Python function to solve the user's request. Do not write any explanations, just the code block.",
        model=DEFAULT_LLM
    )

    # 2. The Judge Agent with a structured output
    reviewer_agent = Agent(
        name="Code Reviewer",
        instructions="You are a senior code reviewer. Evaluate the provided Python function based on the original request. Check for correctness and style. Provide a 'pass' or 'fail' score and feedback.",
        output_type=CodeEvaluation,
        model=DEFAULT_LLM
    )

    # --- Orchestration Loop ---
    task = "a function that takes a list of strings and returns a new list with all strings converted to uppercase."
    # We use a list to build the conversation history for the coder agent
    conversation_history: list[TResponseInputItem] = [{"role": "user", "content": task}]
    max_revisions = 3

    with trace("LLM as a Judge: Code Review"):
        for i in range(max_revisions):
            print(f"
--- Attempt {i + 1} ---")

            # Step A: Generate the code
            print("Coder Agent is generating code...")
            coder_result = Runner.run_sync(coder_agent, conversation_history)
            generated_code = coder_result.final_output
            print(f"Generated Code:
```python
{generated_code}
```")

            # Add the generated code to its own history
            conversation_history = coder_result.to_input_list()

            # Step B: Judge the code
            print("
Reviewer Agent is evaluating...")
            review_input = f"Original Request: {task}

Code to Review:
```python
{generated_code}
```"
            reviewer_result = Runner.run_sync(reviewer_agent, review_input)

            evaluation: CodeEvaluation = reviewer_result.final_output
            print(f"Reviewer Score: {evaluation.score.upper()}")

            # Step C: Decide whether to loop or break
            if evaluation.score == "pass":
                print("Code passed review! Final code is ready.")
                break
            else:
                print(f"Reviewer Feedback: {evaluation.feedback}")
                # Add the feedback to the coder's conversation history for the next attempt
                feedback_for_coder = f"A reviewer has provided feedback on your last attempt. Please fix it. Feedback: {evaluation.feedback}"
                conversation_history.append({"role": "user", "content": feedback_for_coder})
        else:
            print("
Max revisions reached. The code did not pass the review.")

if __name__ == "__main__":
    main()
```


>  Separating Conversation Histories
> 
> In the "LLM as a Judge" example, note that we maintain two separate contexts. The `coder_agent` has a `conversation_history` that evolves with feedback. The `reviewer_agent`, however, is called with fresh input each time (`review_input`). This is a deliberate design choice. The reviewer should provide an objective assessment of the code against the *original* request, without being biased by the back-and-forth of the revision process.
{: .prompt-info }

This pattern is incredibly versatile. It can be used for improving essays, validating plans, checking data for inconsistencies, and much more.

![*A critique and refinement loop using a generator and a judge.*](/assets/img/2025-06-23-multi-agent-orchestration-patterns/figure-2.png)


## Parallelization (Fan-Out/Fan-In)

Many tasks don't need to be sequential. When you can perform several sub-tasks independently, running them in parallel can dramatically reduce latency. This is often called a "fan-out/fan-in" pattern. Python's `asyncio` library, particularly `asyncio.gather`, makes this pattern straightforward to implement with the Agents SDK.

Common use cases include:

*   **Parallel Research:** Kicking off multiple web searches for different aspects of a topic simultaneously.
*   **Multiple Drafts:** Asking three agents to write three different versions of an email, then using a fourth agent to pick the best one or merge them.
*   **Multi-Perspective Analysis:** Querying agents with different personas (e.g., an optimist, a pessimist, a legal expert) to get a well-rounded view of a situation.

Let's implement the multi-perspective analysis pattern.

```python
 #
# A fan-out/fan-in pattern for parallel execution.
#
import asyncio
from agents import Agent, Runner, trace
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

async def main():

    # 1. Define the parallel specialist agents (the "Fan-Out")
    optimist_agent = Agent(
        name="Optimist",
        instructions="You are an eternal optimist. You see the best in every situation and focus only on the positive outcomes.",
        model=DEFAULT_LLM
    )
    pessimist_agent = Agent(
        name="Pessimist",
        instructions="You are a deep pessimist. You see the worst in every situation and focus only on the risks and negative outcomes.",
        model=DEFAULT_LLM
    )
    realist_agent = Agent(
        name="Realist",
        instructions="You are a balanced realist. You weigh the pros and cons objectively.",
        model=DEFAULT_LLM
    )

    # 2. Define the synthesizer agent (the "Fan-In")
    synthesizer_agent = Agent(
        name="Synthesizer",
        instructions="You have been given three perspectives on a topic: one optimistic, one pessimistic, and one realistic. Your job is to synthesize these into a single, balanced, final answer.",
        model=DEFAULT_LLM
    )

    # --- Orchestration ---
    topic = "the impact of AI on the job market for software developers."
    with trace("Parallel Perspective Analysis"):
        print(f"Analyzing topic: {topic}
")

        # Step A: Fan-Out - Run all three persona agents in parallel
        optimist_task = Runner.run(optimist_agent, topic)
        pessimist_task = Runner.run(pessimist_agent, topic)
        realist_task = Runner.run(realist_agent, topic)

        results = await asyncio.gather(optimist_task, pessimist_task, realist_task)
        optimist_view, pessimist_view, realist_view = [res.final_output for res in results]

        print("--- Perspectives Gathered ---")
        print(f"Optimist says: {optimist_view}
")
        print(f"Pessimist says: {pessimist_view}
")
        print(f"Realist says: {realist_view}
")

        # Step B: Fan-In - Feed the parallel results into the synthesizer
        synthesis_input = f"""
        Topic: {topic}

        Optimistic Perspective:
        {optimist_view}

        Pessimistic Perspective:
        {pessimist_view}

        Realistic Perspective:
        {realist_view}

        Please synthesize these into a final, balanced analysis.
        """
        final_result = await Runner.run(synthesizer_agent, synthesis_input)

        print("--- Final Synthesized Report ---")
        print(final_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

![*A fan-out/fan-in parallelization pattern.*](/assets/img/2025-06-23-multi-agent-orchestration-patterns/figure-3.png)


## Chapter Summary

Orchestration is where the true power of an agentic system emerges. In this chapter, we moved beyond single agents and learned how to weave them into complex, purposeful workflows. We explored three fundamental patterns: the predictable **Deterministic Chain**, the quality-improving **LLM as a Judge** loop, and the fast and diverse **Parallelization** pattern.

These patterns are not mutually exclusive; the most sophisticated applications often blend them. You might have a deterministic chain where one of the steps is an iterative review loop, or a parallel fan-out where each parallel branch is itself a multi-step chain. Because the SDK is built on simple Python primitives, you have the full power of the language at your disposal to combine these patterns in any way you see fit.

With a solid grasp of how to build and orchestrate agents, we are now ready to tackle the crucial topics of safety and reliability. In the [next chapter](https://iamulya.one/posts/guardrails-for-safety-and-control), we will introduce **Guardrails**, the SDK's mechanism for validating agent inputs and outputs to prevent errors and malicious use.
