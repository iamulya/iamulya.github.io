---
title: "Chapter 1 - An Introduction to Agentic AI"
date: "2025-06-23 07:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Welcome to the start of your journey into building sophisticated AI systems. In this chapter, we will lay the foundational groundwork for everything that follows. We'll begin by exploring the conceptual shift from simple, single-shot Large Language Model (LLM) calls to dynamic, multi-step "agentic" workflows. We will then introduce the OpenAI Agents SDK, discuss its core design principles, and provide a high-level overview of the key components you'll be mastering throughout this book.

## The Paradigm Shift: From LLM Calls to Agentic Workflows

For the past few years, the dominant pattern for interacting with Large Language Models has been the single-shot, request-response cycle. A developer sends a prompt to an API endpoint and receives a text completion. This model is incredibly powerful for tasks like summarization, translation, and simple Q&A.

Let's look at a standard, non-agentic approach to a simple task: getting a weather-themed haiku using a Gemini model via `litellm`.

```python
#
# A traditional, non-agentic LLM call
#

import asyncio
import litellm

async def get_haiku(topic: str):
    """Generates a haiku using a direct LLM call."""

    messages = [{
        "role": "user",
        "content": f"Write a haiku about {topic}."
    }]

    # Direct call to the model
    response = await litellm.acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content

async def main():
    haiku = await get_haiku("a rainy day in Tokyo")
    print(haiku)

if __name__ == "__main__":
    asyncio.run(main())
```

This works perfectly for its intended purpose. However, its limitations become apparent when we consider more complex tasks. What if we wanted to:

1.  **Get the *actual* weather first**, then write the haiku?
2.  **Validate** that the output is, in fact, a haiku?
3.  **Delegate** to a "Poetry Specialist" model if the topic is complex?
4.  **Remember** the user's location for future requests?

Fulfilling these requirements with the single-shot pattern requires you, the developer, to write significant orchestration code. You would need to manage state, chain API calls, parse outputs, and implement conditional logic—all outside the context of the LLM's reasoning capabilities.

This is where the agentic paradigm comes in. An **agent** is not just a model; it's a persistent process powered by an LLM that can:

*   **Reason:** Decompose a high-level goal into smaller, executable steps.
*   **Use Tools:** Interact with external systems (APIs, databases, files) to gather information or perform actions.
*   **Maintain Context:** Remember previous interactions to inform future decisions.
*   **Delegate:** Hand off tasks to other specialized agents.

The OpenAI Agents SDK provides the lightweight framework to build these processes. Let's re-implement our haiku example using the SDK.

```python
#
# The same task, accomplished with the Agents SDK
#

import asyncio
from agents import Agent, Runner

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

async def main():
    # Define an Agent: an LLM configured with instructions
    haiku_agent = Agent(
        name="Haiku Poet",
        instructions="You are a poetic assistant who only responds in haikus (5-7-5 syllables).",
        model=DEFAULT_LLM # Using the 'litellm/' prefix
    )

    # Use the Runner to execute the agent
    result = await Runner.run(
      starting_agent=haiku_agent,
      input="Write a haiku about a rainy day in Tokyo."
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

At first glance, this might seem like more code for the same result. But the power lies in what's happening under the hood. The `Runner.run()` call initiates an **agent loop**, a process that can continue over multiple turns, call tools, and even hand off to other agents. We've moved from a simple function call to a robust, extensible workflow engine.


> The Agent Loop is Key
> 
> The fundamental difference between a direct LLM call and using the Agents SDK is the **agent loop**. The `Runner` class manages this loop, automatically handling the cycle of:
> 
> 1.  Calling the LLM with the current conversation history and available tools.
> 2.  Parsing the LLM's response.
> 3.  Executing any tool calls the LLM requested.
> 4.  Appending the tool results to the conversation history.
> 5.  Repeating until a final answer is generated.
> 
> This loop is the core of agentic behavior, and the SDK manages it for you.
{: .prompt-info }

## Why the OpenAI Agents SDK?

The world of AI frameworks is vast. The OpenAI Agents SDK differentiates itself through a set of clear design principles aimed at developer productivity and power.

*   **Lightweight and Python-first:** The SDK introduces very few new abstractions. Instead of learning a complex set of framework-specific concepts, you orchestrate agents using standard Python features like `if/else` statements, `for` loops, and `asyncio.gather`. The core primitives you need to learn are `Agent` and `Runner`. This philosophy dramatically shortens the learning curve and makes your code more transparent and debuggable.

*   **Extensible and Provider-Agnostic:** While developed at OpenAI, the SDK is built to be model-agnostic. As seen in our example, using models from Google, Anthropic, or any of the 100+ providers supported by LiteLLM is as simple as changing a model string. The SDK's components—from models to tracing—are designed around interfaces, allowing you to plug in your own implementations for maximum control.

*   **Powerful Primitives:** The SDK focuses on a small but potent set of building blocks. By combining Agents, Tools, and Handoffs, you can model incredibly complex workflows, from simple tool-using bots to sophisticated multi-agent systems where different agents collaborate to solve a problem. It provides the "just enough" functionality to be immediately useful without becoming bloated.


> Not Just for OpenAI Models
> 
> The name "OpenAI Agents SDK" reflects its origin, but its capability extends far beyond. The integration with `litellm` and the underlying `ModelProvider` interface make it a versatile tool for any developer, regardless of their preferred LLM provider. Throughout this book, we will use Gemini models in our primary examples to underscore this flexibility.
{: .prompt-info }

## Core Primitives at a Glance

This entire book is dedicated to exploring the SDK's components in detail, but let's take a high-level look at the five key primitives that form the foundation of any application you build.

1.  **Agent:** The central building block. It's an LLM configured with a name, instructions (a system prompt), and a set of available `tools` and `handoffs`. Think of it as a blueprint for a specialized AI worker.

2.  **Tool:** An action an Agent can take. This is typically a Python function exposed to the LLM, allowing it to interact with the outside world, fetch data, or perform calculations.

3.  **Handoff:** A specialized type of tool that allows one agent to delegate the entire task to another, more specialized agent. This is the primary mechanism for creating multi-agent systems.

4.  **Runner:** The engine that executes the agentic workflow. You provide it with a starting agent and an initial input, and the `Runner` manages the entire multi-turn agent loop until a final result is achieved.

5.  **Tracing:** A built-in observability system that records every step of your agent's execution—every LLM call, every tool use, every handoff. This is an indispensable tool for debugging, monitoring, and improving your agents.

These components interact in a predictable flow, which we can visualize as follows:

![*High-level architecture of the Agents SDK.*](/assets/img/2025-06-23-introduction-to-agentic-ai/figure-1.png)


In the [next chapter](https://iamulya.one/posts/your-first-agent), we will dive straight into the code, setting up your environment and building a functional, tool-using agent from scratch. Let's get started.
