---
title: "Chapter 10 - Tracing, Debugging, and Visualization"
date: "2025-06-23 16:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

As agentic workflows grow in complexity, understanding *what* they did and *why* they did it becomes increasingly challenging. A single user prompt can trigger a cascade of LLM calls, tool executions, and handoffs across multiple agents. Simply looking at the final output isn't enough to debug issues or optimize performance. To address this, the Agents SDK comes with a powerful, built-in **Tracing** system.

A trace is a comprehensive, structured log of a single end-to-end workflow. It captures every significant event, providing a "glass box" view into your agent's execution. This chapter will cover how tracing works automatically, how to create custom traces and events, how to control sensitive data, and how to visualize your agent architectures.


> ## Tracing is On by Default
> 
> The SDK is designed for observability from the ground up. Tracing is enabled by default and requires no special configuration to get started, provided you have a valid OpenAI API key set (as traces are uploaded to the OpenAI platform). If you are not using OpenAI for tracing, you will need to configure the SDK accordingly, as we'll discuss later.
> {: .prompt-info }

## The Anatomy of a Trace: Traces and Spans

The tracing system is built on two core concepts:

*   **Trace:** Represents a single, complete workflow. For example, one run of `Runner.run_sync()` constitutes one trace. A trace has a `workflow_name` (e.g., "Customer Support Bot"), an optional `group_id` (e.g., a conversation ID to link multiple traces), and a unique `trace_id`.
*   **Span:** Represents a single, timed operation *within* a trace. Each span has a start time, an end time, a type, and associated metadata. Spans can be nested, creating a hierarchical view of the execution.

The SDK automatically creates spans for all major operations:

*   `agent_span`: An entire turn for a specific agent.
*   `generation_span`: A call to an LLM.
*   `function_span`: A call to a `@function_tool`.
*   `handoff_span`: An agent-to-agent handoff.
*   `guardrail_span`: The execution of a guardrail check.

When you run an agent, the `Runner` automatically wraps the entire execution in a `trace()` with the default name "Agent workflow". You can then view this trace in the [OpenAI Traces dashboard](https://platform.openai.com/traces).

Let's run a simple tool-using agent and examine the trace it produces.

```python
from agents import Agent, Runner, function_tool, RunConfig
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

@function_tool
def get_user_city() -> str:
    """Retrieves the user's city from a database."""
    print("[Tool Executed]: get_user_city()")
    return "San Francisco"

def main():

    weather_agent = Agent(
        name="Weather Assistant",
        instructions="You are a helpful assistant. Use your tools to find the user's city and then tell them it's always sunny there.",
        model=DEFAULT_LLM,
        tools=[get_user_city]
    )

    # Use RunConfig to give our trace a meaningful name
    run_config = RunConfig(workflow_name="Simple Weather Lookup")
    result = Runner.run_sync(
        weather_agent,
        "What's the weather like where I am?",
        run_config=run_config
    )
    print(f"
Final Answer: {result.final_output}")

if __name__ == "__main__":
    main()
```

After running this code, a link to the trace will be printed in your terminal. Opening it reveals a timeline visualization.

![*A conceptual representation of the trace for the weather agent.*](/assets/img/2025-06-23-tracing-debugging-and-visualization/figure-1.png)

This structured, hierarchical view is invaluable. You can see exactly how long each step took, what data was passed between them, and how the agent made its decisions.


> By default, trace data is securely exported to the OpenAI platform for visualization in the Traces dashboard. This export process requires authentication via an OpenAI API key.
> 
> If you are exclusively using other model providers (like Google's Gemini) and do not have an OpenAI key, you have two main options:
> 
> 1.  **Disable Tracing:** Set the environment variable `OPENAI_AGENTS_DISABLE_TRACING=1` or pass `RunConfig(tracing_disabled=True)` to the `Runner`.
> 2.  **Use a Custom Trace Processor:** The SDK allows you to redirect trace data to other observability platforms (like Weights & Biases, Arize, etc.). This is an advanced topic covered in the documentation.
> 
> {: .prompt-info }

## Higher-Level Traces

The `Runner` automatically creates a trace for each run. However, sometimes a single logical "workflow" involves multiple calls to `Runner.run()`. For example, in a conversational loop, each user turn triggers a new run. To group these related runs into a single, cohesive trace, you can wrap your code in a `trace()` context manager.

```python
import asyncio
from agents import Agent, Runner, trace, TResponseInputItem
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

async def main():

    concise_agent = Agent(
        name="Concise Agent",
        instructions="You are extremely concise. Respond in 20 words or less.",
        model=DEFAULT_LLM,
    )

    # Wrap multiple Runner calls in a single trace
    with trace(workflow_name="Two-Turn Conversation", group_id="convo-123"):
        print("--- Turn 1 ---")
        # Turn 1
        result1 = await Runner.run(
            concise_agent, "Explain the concept of photosynthesis."
        )
        print(f"Agent: {result1.final_output}")

        # Prepare input for the next turn
        turn2_input: list[TResponseInputItem] = result1.to_input_list()
        turn2_input.append({
            "role": "user",
            "content": "Now explain it as if I were five years old."
        })

        print("
--- Turn 2 ---")
        # Turn 2
        result2 = await Runner.run(concise_agent, turn2_input)
        print(f"Agent: {result2.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

When you view the trace for this execution, you will see both runs nested under the "Two-Turn Conversation" trace, providing a complete history of the interaction. The `group_id` is especially useful for linking all traces associated with a single user session or conversation thread.

## Custom Spans for Deeper Insight

The automatically generated spans are incredibly useful, but sometimes you want to trace parts of your own application logic that exist outside the agent loop. For this, the SDK provides the `custom_span()` context manager.

Let's imagine our tool doesn't just return a random number but performs a series of complex, potentially slow operations. We can wrap these operations in custom spans to profile them.

```python
import asyncio
import time
from agents import function_tool, custom_span

@function_tool
async def complex_data_fetch(query: str) -> str:
    """A tool that simulates a complex, multi-step data fetching process."""
    print("[Tool]: Starting complex data fetch...")

    # A custom span for the first part of our internal logic
    with custom_span("Database Query", data={"query": query, "db": "primary_users"}):
        await asyncio.sleep(0.5) # Simulate DB latency
        print("[Tool]: ...database query complete.")

    # A custom span for the second part
    with custom_span("Data Processing", data={"step": "aggregation"}):
        await asyncio.sleep(0.3) # Simulate processing time
        print("[Tool]: ...data processing complete.")

    return f"Processed data for query: {query}"

# ... agent and runner setup would go here ...
```

When you view the trace now, inside the `function_span` for `complex_data_fetch`, you will find two nested `custom_span`s: "Database Query" and "Data Processing". 

## Controlling Sensitive Data in Traces

Traces can contain potentially sensitive information. By default, the `generation_span` captures the full input and output of the LLM, and the `function_span` captures the arguments and return value of your tools.

The `RunConfig` object provides `trace_include_sensitive_data` to control this. When set to `False`, the SDK will still create the spans, but it will omit the potentially sensitive `input` and `output` fields from the trace data.

```python
from agents import Agent, Runner, RunConfig, function_tool
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

@function_tool
def process_user_data(user_id: str, personal_note: str) -> str:
    """Processes sensitive user data."""
    return "Data processed successfully."

def main():
    sensitive_agent = Agent(
        name="Processor",
        instructions="Process the data.",
        model=DEFAULT_LLM,
        tools=[process_user_data]
    )

    # Create a RunConfig to disable sensitive data logging
    secure_run_config = RunConfig(
        workflow_name="Secure Data Processing",
        trace_include_sensitive_data=False
    )

    # The trace for this run will not contain the LLM prompts or tool arguments.
    Runner.run_sync(
        sensitive_agent,
        "Process user 'user-abc-123' with note 'This is a very secret note.'",
        run_config=secure_run_config
    )
    print("Run complete. Check the trace to confirm data was omitted.")

if __name__ == "__main__":
    main()
```

When you inspect the trace generated by this run, the `generation_span` and `function_span` will be present, but their `input` and `output` fields in the span metadata will be empty.

## Visualizing Agent Architectures

While traces are essential for understanding a *single execution*, they don't give you a high-level picture of your system's *architecture*. How do your agents connect? Which tools can they use? Which handoffs are possible?

For this, the SDK provides a visualization utility. This feature requires the `graphviz` optional dependency.

```bash
pip install "openai-agents[viz]"
```

The `draw_graph` function from `agents.extensions.visualization` inspects your top-level agent and recursively maps out all its tools and handoffs to generate a graph.

```python
from agents import Agent, function_tool
from agents.extensions.visualization import draw_graph
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# --- Define a complex agent architecture ---
@function_tool
def general_knowledge_qa(question: str) -> str:
    """Answers general knowledge questions."""
    return "The answer is 42."

billing_agent = Agent(name="Billing Agent", instructions="...")
shipping_agent = Agent(name="Shipping Agent", instructions="...")

support_agent = Agent(
    name="Support Agent",
    instructions="I help with support tickets.",
    tools=[general_knowledge_qa],
    model=DEFAULT_LLM,
    handoffs=[shipping_agent]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="I route users to the correct department.",
    model=DEFAULT_LLM,
    handoffs=[support_agent, billing_agent]
)

# --- Generate and save the visualization ---
graph = draw_graph(triage_agent, filename="triage_system_architecture")
print("Graph saved to triage_system_architecture.png")
```

This code generates a PNG file visualizing your entire agent system, making it easy to understand the possible flows of control and capabilities at a glance.

![*The architecture of our sample triage system.*](/assets/img/2025-06-23-tracing-debugging-and-visualization/figure-2.png)


## Chapter Summary

Observability is not an afterthought; it's a core requirement for building and maintaining robust agentic systems. In this chapter, we explored the SDK's powerful, built-in tracing capabilities. We learned how traces and spans provide a detailed, hierarchical view of every agent execution automatically. We then saw how to create higher-level traces to group multiple runs and how to use custom spans to instrument our own code. We also covered the critical features for controlling sensitive data in traces and for visualizing the high-level architecture of our agent systems.

With this knowledge, you are no longer flying blind. You have the tools to debug, profile, and deeply understand the behavior of your agents. In the [next chapter](https://iamulya.one/posts/customer-service-bot), we will put it all together in a case study: a multi-agent system designed to handle customer service inquiries for an airline..
