---
title: "Chapter 3 - Beyond OpenAI: Using Any Model Provider"
date: "2025-06-23 09:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

A core design principle of the OpenAI Agents SDK is extensibility. While it works seamlessly with OpenAI's state-of-the-art models, the framework is fundamentally model-agnostic. You are not locked into a single provider. This flexibility allows you to choose the best LLM for the task at hand, balancing cost, speed, and capability across your agentic system.

This chapter will guide you through the mechanisms that enable this flexibility. We'll start with the easiest method: using the built-in `LiteLLM` integration to access over 100 different models. We'll then discuss common compatibility issues and how to handle them, and finally, briefly touch on how you could build a provider for a completely custom model.

## The `Model` and `ModelProvider` Interfaces

The SDK's provider-agnosticism is made possible by two key abstract classes defined in `src/agents/models/interface.py`:

*   **`Model`**: This is the interface that every model implementation must adhere to. It has two primary methods: `get_response()` for non-streaming calls and `stream_response()` for streaming. Any class that correctly implements this interface can be used by the `Runner`.

*   **`ModelProvider`**: This is a factory class. Its job is to take a model name as a string (e.g., `"gpt-4o"`) and return a corresponding `Model` instance.

The SDK's default `ModelProvider` is the `MultiProvider`, which can route requests to different providers based on a prefix. This is the magic that makes using various models so simple.

## Using Any Model via LiteLLM

The easiest and most powerful way to use non-OpenAI models is through the SDK's built-in integration with **LiteLLM**. LiteLLM is a library that provides a standardized interface for calling a vast array of LLM providers, including Google, Anthropic, Cohere, Mistral, and many more.

To use it, you simply prefix your model string with `litellm/`. The SDK's `MultiProvider` will automatically detect this prefix and route the request through the `LitellmModel` implementation.

Let's build a multi-agent workflow that uses both a Gemini and a Claude model in the same run.

```python
from agents import Agent, Runner
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

def main():

    # 1. Define a specialist agent using Google's Gemini
    code_explainer_agent = Agent(
        name="Code Explainer",
        instructions="You are an expert at explaining complex code in simple terms.",
        model=DEFAULT_LLM,
        handoff_description="Use for explaining code."
    )

    # 2. Define another specialist using Anthropic's Claude
    poem_writer_agent = Agent(
        name="Poet",
        instructions="You are a poet who writes beautiful poems about any topic.",
        model="litellm/anthropic/claude-3-haiku-20240307",
        handoff_description="Use for writing poetry."
    )

    # 3. The Triage agent can be from any provider
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Analyze the user's request and hand off to the appropriate specialist agent.",
        model=DEFAULT_LLM,
        handoffs=[code_explainer_agent, poem_writer_agent]
    )

    print("--- Running Code Explanation Workflow ---")
    code_result = Runner.run_sync(
        triage_agent,
        "Can you explain what this Python list comprehension does: `[x*x for x in range(10)]`?"
    )
    print(f"Final response from: {code_result.last_agent.name}")
    print(f"Response: {code_result.final_output}")

    print("
--- Running Poetry Workflow ---")
    poem_result = Runner.run_sync(
        triage_agent,
        "Write me a short poem about the moon."
    )
    print(f"Final response from: {poem_result.last_agent.name}")
    print(f"Response: {poem_result.final_output}")


if __name__ == "__main__":
    main()
```
This example seamlessly orchestrates agents powered by two different model providers. The `MultiProvider` handles the routing automatically based on the model string prefix.


> The `litellm` library is designed to find API keys from environment variables automatically. The convention is `<PROVIDER>_API_KEY`.
> 
> - For Google models, it looks for `GOOGLE_API_KEY`.
> - For Anthropic, it looks for `ANTHROPIC_API_KEY`.
> - For Cohere, `COHERE_API_KEY`, and so on.
> 
> Before running code that uses multiple providers, ensure you have set all the necessary environment variables. You can find the specific variable names in the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).
{: .prompt-info }

## Handling Provider Incompatibilities

While LiteLLM provides a unified *calling* interface, it cannot magically make all providers support the same *features*. When you venture beyond OpenAI models, you must be aware of potential capability gaps.

### Structured Outputs (`output_type`)

The `Agent`'s `output_type` parameter is a powerful feature that relies on the model's ability to adhere to a specific JSON schema. This is often called "JSON Mode" or "Tool Calling" by model providers. While support is growing, it is not universal.

**Symptom:** You might encounter a `BadRequestError` from the model provider, often mentioning that `response_format` or `json_schema` is an unsupported parameter.

```
# Fictional error message
BadRequestError: Error code: 400 - 
{
    'error': {
                'message': "'response_format' is not a valid parameter for this model."
            }
}
```

**Solution:**

1.  **Check Provider Docs:** First, check if your chosen model supports a constrained JSON output mode.
2.  **Prompt-Based JSON:** If it doesn't, you must remove the `output_type` from your `Agent` definition. Your only recourse is to engineer your `instructions` to explicitly ask for a JSON response.

```python
#
# Forcing JSON via prompt when structured output is not supported.
#
import json
from pydantic import BaseModel, ValidationError
from agents import Agent, Runner

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

class UserProfile(BaseModel):
    name: str
    age: int

def main():
    # Note: We do NOT set output_type on the agent
    json_agent = Agent(
        name="JSON Extractor",
        # We bake the schema and instructions into the prompt itself
        instructions=f"""
        You are a JSON extraction expert. Extract the user's name and age from their message.
        You MUST respond with only a single, valid JSON object that conforms to this Pydantic schema:

        ```json
        {json.dumps(UserProfile.model_json_schema(), indent=2)}
        ```
        """,
        model=DEFAULT_LLM
    )

    # After running, you would need to manually parse the `result.final_output` string
    result = Runner.run_sync(json_agent, "Hi, I'm Bob and I'm 32 years old.")
    try:
        # Sanitize the output to extract only the JSON part
        final_result = result.final_output

        # Find the start and end indices of the JSON object
        json_start = final_result.find('{')
        json_end = final_result.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the LLM output.")

        clean_json_str = final_result[json_start:json_end]

        print(f"Cleaned JSON String for Parsing:
{clean_json_str}
")

        # Now parse the cleaned string
        profile = UserProfile.model_validate_json(clean_json_str)

        print("Successfully parsed UserProfile!")
        print(f"Name: {profile.name}, Age: {profile.age}")
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        print(f"Failed to parse LLM output: {e}")

if __name__ == "__main__":
    main()
```


> ## Prompt-Based JSON is Brittle
> 
> Relying on prompting to get structured data is significantly less reliable than using a model's native JSON mode. The model may fail to produce valid JSON, add extra conversational text, or ignore the format entirely. This will lead to parsing errors in your application code. Whenever possible, choose a model that natively supports structured outputs if your application depends on them.
{: .prompt-info }

### Parallel Tool Calling

The `ModelSettings` parameter `parallel_tool_calls=True` is a powerful optimization that allows an LLM to request multiple tool calls in a single turn. This is a feature primarily associated with newer OpenAI models.

**Symptom:** Many other models do not support parallel calls. They will either ignore the setting and call tools sequentially, or they may produce an error.

**Solution:** When using non-OpenAI models, it is safest to assume that parallel tool calls are not supported. Do not set `parallel_tool_calls=True`. Design your agent's logic to work with sequential tool calls. If a task requires information from multiple tools, the agent will simply perform them one after another across multiple turns in the agent loop.

### Explicit Model Instantiation

For finer control, instead of relying on the `litellm/` prefix, you can directly instantiate the `LitellmModel` class from the `agents.extensions.models` module. This allows you, for example, to pass an API key directly if you prefer not to use environment variables.

```python
import asyncio
import os
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

def main():
    api_key = os.getenv("MY_SECRET_STORE_ANTHROPIC_KEY") # Or get it from wherever

    # Instantiate the model directly, passing the key
    claude_model = LitellmModel(
        model="anthropic/claude-3-haiku-20240307",
        api_key=api_key
    )

    agent = Agent(
        name="Direct Model Agent",
        instructions="You are a helpful assistant.",
        model=claude_model # Pass the model instance
    )

    result = Runner.run_sync(agent, "Hi there!")
    print(result.final_output)

if __name__ == "__main__":
    main()
```

## Chapter Summary

In this chapter, we shattered the boundaries of a single provider and unlocked the full, model-agnostic potential of the Agents SDK. We saw how the `Model` and `ModelProvider` interfaces provide a clean abstraction and how the built-in `LiteLLM` integration makes using over 100 different models as simple as changing a string.

Crucially, we also addressed the practical challenges of a diverse ecosystem. You now understand the common feature disparities—especially around structured outputs and parallel tool calling—and have strategies to handle them. You are no longer just an OpenAI agent developer; you are an AI application developer, capable of selecting the right tool, and the right LLM, for any job.

With this comprehensive understanding of the SDK's core features, from agents and tools to orchestration and multi-provider support, you are now fully equipped to build complex, real-world agentic applications. In the [next chapter](https://iamulya.one/posts/anatomy-of-an-agent), we will dive deeper into the `Agent` class itself, exploring its full range of configuration options, including context management, typed outputs, and lifecycle hooks.
