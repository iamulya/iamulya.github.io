---
title: "Chapter 2 - Your First Agent: Quickstart and the Runner"
date: "2025-06-23 08:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

In the [previous chapter](https://iamulya.one/posts/introduction-to-agentic-ai), we introduced the high-level concepts behind agentic AI. Now, it's time to translate that theory into practice. This chapter is your hands-on guide to building and executing your first functional agent. We will begin with a foundational "Hello, World" example and then incrementally add complexity by introducing tools. Along the way, we'll dissect the `Runner` class and the **Agent Loop**—the core engine that brings your agents to life. By the end of this chapter, you will not only have a running agent but also a clear mental model of the entire execution flow.


> If you have already followed the steps in Appendix A to run the code examples (highly recommended), you don't necessarily need to follow all of the following steps. However, it's a good idea to glance over these sections before jumping to the "Hello, Agent!" section. 
{: .prompt-info }

## Environment Setup

Before we write any code, let's ensure your development environment is correctly configured. This is a one-time setup for your project.

First, create and activate a Python virtual environment. This isolates your project's dependencies from other Python projects on your system.

```bash
# Create the project directory
mkdir my-agent-project
cd my-agent-project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
```

Next, install the necessary packages. We'll need `openai-agents` for the SDK itself and `litellm` to interact with Google's Gemini models.

```bash
pip install "openai-agents[litellm]"
```

Finally, you must provide an API key for the model provider you intend to use. Since we are using Gemini in these examples (Read the Preface to understand why), you will need to get a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it as an environment variable.

```bash
export GOOGLE_API_KEY="your-api-key-here"
```


>  API Keys are Essential
> 
> The SDK needs credentials to communicate with LLM providers. If the appropriate environment variable (e.g., `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) is not set, `litellm` will raise an authentication error when the `Runner` attempts to call the model. Always ensure your keys are correctly configured in your terminal session or environment management tool.
{: .prompt-info }

## Hello, Agent!

With our environment ready, let's build the simplest possible agent. An agent, at its core, is an LLM configured with a set of instructions. The two fundamental classes we'll use are `Agent` and `Runner`.

-   **`Agent`**: This class defines the agent's identity and capabilities. We'll give it a `name`, a set of `instructions` (its system prompt), and tell it which `model` to use.
-   **`Runner`**: This is the execution engine. We'll use `Runner.run_sync()` for this first example, which is a convenient synchronous method that runs the agent and waits for the final result.

Here is the complete code:

```python
#
# A basic agent that responds to a prompt.
#

from agents import Agent, Runner
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

def main():
    # 1. Define the Agent
    polite_agent = Agent(
        name="Polite Assistant",
        instructions="You are a helpful and polite assistant. You always answer in a clear and friendly tone.",
        model=DEFAULT_LLM
    )

    # 2. Define the input
    user_input = "What is the primary function of a CPU in a computer?"

    # 3. Use the Runner to execute the agent
    print(f"User: {user_input}")
    result = Runner.run_sync(polite_agent, user_input)

    # 4. Print the final output
    print(f"
Agent: {result.final_output}")

if __name__ == "__main__":
    main()
```

In this example, the `Runner` takes our `polite_agent` and the `user_input`, sends them to the Gemini model, and receives a single, complete response. This single-turn interaction is the most basic form of the **Agent Loop**.

![*Sequence diagram of a simple, single-turn Agent Loop.*](/assets/img/2025-06-23-your-first-agent/figure-1.png)


The `result` object returned by the `Runner` is an instance of `RunResult`. For now, we are only interested in the `final_output` property, which contains the agent's text response. We'll explore the other properties of `RunResult` later in this chapter.

## Empowering Agents with Tools

Our first agent was essentially a wrapper around a single LLM call. The true power of agents is unlocked when they can interact with the outside world. We enable this by giving them **tools**.

In the Agents SDK, any Python function can be turned into a tool using the `@function_tool` decorator. The SDK automatically inspects the function's signature and docstring to create a schema that the LLM can understand and use.

Let's create a simple tool and give it to a new agent.

```python
#
# An agent that can use a tool to get information.
#

import random
from agents import Agent, Runner, function_tool

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# --- Tool Definition ---
@function_tool
def get_stock_price(symbol: str) -> float:
    """
    Gets the current stock price for a given ticker symbol.

    Args:
        symbol: The stock ticker symbol (e.g., "GOOGL", "AAPL").
    """
    print(f"[Tool Executed]: Getting stock price for {symbol}")
    # In a real application, this would call a financial data API.
    # Here, we'll just return a random price for demonstration.
    return round(random.uniform(100, 1000), 2)

# --- Agent Definition ---
def main():

    # Define an agent and provide it with the tool
    financial_agent = Agent(
        name="Financial Assistant",
        instructions="You are a financial assistant. Use your tools to answer questions. Be concise.",
        model=DEFAULT_LLM,
        tools=[get_stock_price] # The tool is passed in a list
    )

    # Use the Runner to execute the agent
    result = Runner.run_sync(
        financial_agent,
        "What is the current price of Google's stock?"
    )

    print(f"
Final Answer:
{result.final_output}")

if __name__ == "__main__":
    main()
```

When you run this code, you'll see the following output:

```
[Tool Executed]: Getting stock price for GOOGL

Final Answer:
The current price of GOOGL is $...
```

Notice the line `[Tool Executed]` appears *before* the final answer. This is a critical insight into how the Agent Loop works with tools. The agent didn't know the price of Google's stock. It reasoned that it needed to use the `get_stock_price` tool, the SDK executed that tool, and then the agent used the tool's output to formulate its final response.

## Unpacking the Multi-Turn Agent Loop

The previous example involved a multi-turn interaction with the LLM, all managed seamlessly by the `Runner`. Let's break down the sequence of events:

1.  **Turn 1: Reasoning and Tool Selection**
    *   The `Runner` sends the initial prompt ("What is the current price of Google's stock?") and the schema of the `get_stock_price` tool to the Gemini model.
    *   The model analyzes the prompt. It recognizes that it cannot answer the question directly but has a tool that can.
    *   Instead of returning a final text answer, the LLM returns a special message indicating a **tool call**. It specifies that it wants to call the `get_stock_price` tool with the argument `symbol="GOOGL"`.

2.  **SDK Action: Tool Execution**
    *   The Agents SDK intercepts this tool call message.
    *   It looks up the `get_stock_price` function in the agent's `tools` list.
    *   It executes the Python function `get_stock_price("GOOGL")`.
    *   The function returns a float (e.g., `178.45`).

3.  **Turn 2: Synthesizing the Final Answer**
    *   The `Runner` automatically appends the result of the tool call to the conversation history. The history now contains the original user prompt and a new message indicating that the `get_stock_price` tool was called and returned `178.45`.
    *   It sends this entire updated history back to the Gemini model.
    *   The model now has all the information it needs. It sees the original question and the data from the tool. It uses this context to generate a natural language response: "The current price of GOOGL is $178.45."
    *   This time, the model's response is a final answer, not a tool call. The Agent Loop has completed its objective and terminates.

This multi-turn process is the essence of how agents reason and act.

![*Sequence diagram of a multi-turn Agent Loop with a tool call.*](/assets/img/2025-06-23-your-first-agent/figure-2.png)


## Async and Streaming

While `run_sync()` is convenient for simple scripts, most real-world applications (like web servers) are asynchronous. The SDK is async-native and provides two other methods on the `Runner`:

*   **`Runner.run()`**: The asynchronous equivalent of `run_sync()`. You should `await` its result. It's the preferred method for any application using `asyncio`.
*   **`Runner.run_streamed()`**: This method also runs asynchronously but returns a `RunResultStreaming` object immediately. You can then iterate over this object's events to get real-time updates as the LLM generates tokens or calls tools. This is ideal for chatbot UIs where you want to display the response as it's being generated.

We will cover streaming in depth in a later chapter, but here’s a quick preview:

```python
#
# A quick look at streaming
#

# ... (agent and tool definitions from before) ...

async def stream_example():
    # ...
    financial_agent = Agent(
        # ... same definition
    )
    result = Runner.run_streamed( # Note: run_streamed
        financial_agent,
        "What is the current price of Google's stock?"
    )

    print("Agent's Final Answer: ", end="")
    # The stream_events() iterator yields different event types.
    # We filter for text deltas to print the response token-by-token.
    async for event in result.stream_events():
        if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
            print(event.data.delta, end="", flush=True)

    print() # for a final newline

# ...
```

## The `RunResult` Object

The `Runner` returns a rich `RunResult` object containing a complete record of the agent's execution. While `final_output` is often all you need, exploring the other properties is crucial for debugging and building complex logic.

*   `result.new_items`: A list of all new events generated during the run, including `MessageOutputItem`, `ToolCallItem`, and `ToolCallOutputItem`. This gives you a structured history of what happened.
*   `result.raw_responses`: The raw, unprocessed response objects from the LLM provider.
*   `result.last_agent`: A reference to the agent that produced the final output. This is especially important in multi-agent workflows with handoffs.
*   `result.to_input_list()`: A convenience method that formats the entire run history into a list suitable for being the input to the *next* `Runner.run()` call, which is how you build conversational memory.

Let's inspect it:

```python
#
# Inspecting the RunResult object
#

# ... (agent and tool definitions from before) ...

def inspect_result():
    # ...
    financial_agent = Agent(
        # ... same definition
    )
    result = Runner.run_sync(
        financial_agent,
        "What is the current price of Google's stock?"
    )

    print("--- Final Output ---")
    print(result.final_output)

    print("
--- Last Agent ---")
    print(f"The last agent to run was: {result.last_agent.name}")

    print("
--- New Items (Structured History) ---")
    for item in result.new_items:
        print(f" - Type: {item.type}, Agent: {item.agent.name}")
        if item.type == 'tool_call_output_item':
            print(f"   Output: {item.output}")

# Expected Output:
#
# --- Final Output ---
# The current price for GOOGL is $...
#
# --- Last Agent ---
# The last agent to run was: Financial Assistant
#
# --- New Items (Structured History) ---
#  - Type: tool_call_item, Agent: Financial Assistant
#  - Type: tool_call_output_item, Agent: Financial Assistant
#    Output: ...
#  - Type: message_output_item, Agent: Financial Assistant
```


> The `result.to_input_list()` method is the key to creating conversational agents. By taking the result of one run and using it as the basis for the input to the next, you build up a conversation history that the agent can refer to.
> 
> ```python
> # Turn 1
> result1 = Runner.run_sync(agent, "My name is Alex.")
> 
> # Turn 2
> next_turn_input = result1.to_input_list()
> next_turn_input.append({"role": "user", "content": "What is my name?"})
> result2 = Runner.run_sync(agent, next_turn_input)
> # The agent will correctly answer "Your name is Alex."
> ```
{: .prompt-info }

## Chapter Summary

In this chapter, we took our first practical steps with the OpenAI Agents SDK. We set up our environment, defined a simple agent, and used the `Runner` to execute it. We then empowered our agent with a custom Python tool, which unveiled the multi-turn **Agent Loop** that the SDK manages on our behalf. We've seen how the agent can reason, select a tool, and use its output to generate a final answer. Finally, we briefly looked at asynchronous execution, streaming, and the detailed `RunResult` object.

You now have a solid mental model of the fundamental execution flow. In the [next chapter](https://iamulya.one/posts/using-any-model-provider), we'll explore the ecosystem beyond OpenAI, showing you how to integrate over 100 different models from various providers into your agentic workflows.
