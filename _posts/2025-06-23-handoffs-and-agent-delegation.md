---
title: "Chapter 7 - Handoffs and Agent Delegation"
date: "2025-06-23 13:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

In the [last chapter](https://iamulya.one/posts/mcp), we empowered our agents by giving them tools, allowing them to delegate specific, function-like tasks. This "agent-as-tool" pattern is powerful but follows a distinct request-response model: the orchestrator agent calls a tool, waits for a result, and then continues its own reasoning process.

But what if you don't want to just delegate a task? What if you want to transfer the *entire responsibility* of the conversation to another, more specialized agent? This is the role of a **Handoff**.

A handoff is a complete transfer of control. When an agent performs a handoff, its own execution stops, and the new agent takes over the conversation, inheriting the history and continuing the agent loop. It's the SDK's equivalent of a customer service representative saying, "I can't help with that, let me transfer you to our billing department." This chapter explores this powerful mechanism for creating sophisticated, multi-agent systems.

## Creating a Basic Handoff

The primary way to enable handoffs is through the `handoffs` parameter on an `Agent` instance. This parameter takes a list of other `Agent` objects that the current agent is allowed to delegate to.

Under the hood, the SDK automatically creates a special tool for each potential handoff. If you have an agent named `"Billing Department"`, the SDK creates a tool called `transfer_to_billing_department` and makes it available to the LLM. When the LLM decides to call this tool, the SDK intercepts it and executes the handoff protocol.

Let's build a simple triage system where a frontline agent determines the user's language and hands off to the appropriate specialist.

```python
from agents import Agent, Runner, trace
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

def main():

    # 1. Define the specialist agents
    spanish_agent = Agent(
        name="Spanish Specialist",
        instructions="You are a helpful assistant who communicates only in Spanish.",
        model=DEFAULT_LLM
    )

    english_agent = Agent(
        name="English Specialist",
        instructions="You are a helpful assistant who communicates only in English.",
        model=DEFAULT_LLM
    )

    # 2. Define the Triage Agent with its handoff targets
    triage_agent = Agent(
        name="Triage Agent",
        instructions="You are a language routing agent. Based on the user's language, hand off to the correct specialist. Do not answer the question yourself.",
        model=DEFAULT_LLM,
        handoffs=[spanish_agent, english_agent] # List of possible handoff targets
    )

    # 3. Run the workflow with a Spanish query
    with trace("Language Handoff Workflow"):
        result = Runner.run_sync(
            triage_agent,
            "Hola, ¿cuál es la capital de México?"
        )

    print(f"Final Response came from: {result.last_agent.name}")
    print(f"Agent says: {result.final_output}")

if __name__ == "__main__":
    main()

# Expected Output:
#
# Final Response came from: Spanish Specialist
# Agent says: ¡Hola! La capital de México es la Ciudad de México.
```

In this example, the `Triage Agent` never answers the question. Its LLM recognizes the input is Spanish and calls the automatically generated `transfer_to_spanish_specialist` tool. The SDK then stops the `Triage Agent` and starts a new agent loop with the `Spanish Specialist`, which then generates the final answer.

![*Sequence diagram of a basic handoff.*](/assets/img/2025-06-23-handoffs-and-agent-delegation/figure-1.png)


## Customizing Handoffs with the `handoff()` Helper

Passing an agent directly into the `handoffs` list is convenient, but for more control, you should use the `handoff()` helper function. This function creates a `Handoff` object that lets you customize several aspects of the delegation process.

*   `tool_name_override` & `tool_description_override`: By default, the tool name and description are generated from the target agent's `name` and `handoff_description`. These parameters allow you to provide more specific names and instructions to the LLM, which can significantly improve routing accuracy.
*   `on_handoff`: A callback function that executes at the moment of handoff. This is invaluable for side effects like logging the transfer, triggering a notification, or pre-fetching data that the next agent will need.

Let's create a more explicit handoff to a billing agent, logging the event.

```python
from agents import Agent, Runner, handoff, RunContextWrapper
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Define the on_handoff callback
def log_billing_transfer(ctx: RunContextWrapper) -> None:
    """This function will be called when the handoff occurs."""
    print("[AUDIT LOG]: Transferring user to the billing department.")

# 2. Define the target agent
billing_agent = Agent(
    name="Billing Department",
    instructions="You help users with their billing inquiries.",
    model=DEFAULT_LLM,
    # We can add a description that the orchestrator will see
    handoff_description="Use for questions about invoices, payments, or subscriptions."
)

# 3. Create the handoff using the helper
custom_handoff = handoff(
    agent=billing_agent,
    on_handoff=log_billing_transfer,
    tool_name_override="goto_billing_specialist",
    tool_description_override="Transfer the user to the billing department for any questions related to payments."
)

# 4. Configure the triage agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a support router.",
    model=DEFAULT_LLM,
    handoffs=[custom_handoff]
)

def main():
    Runner.run_sync(triage_agent, "I have a question about my last invoice.")

if __name__ == "__main__":
    main()

# Expected Output:
#
# [AUDIT LOG]: Transferring user to the billing department.
# (Followed by the billing agent's response)
```


> ## Descriptions Drive Routing
> 
> The most important factor in the LLM's decision to hand off is the quality of the tool description. Be explicit. Instead of "Billing agent," a description like "Use for all questions about invoices, payment methods, subscription status, and refunds" gives the model a much clearer signal for when to use the handoff.
{: .prompt-info }

## Passing Structured Data During Handoffs

Sometimes, you need to pass specific, structured information to the agent taking over. For example, when escalating a support ticket, you might want to enforce that a reason for the escalation is provided. You can achieve this by specifying an `input_type` on your `handoff` object.

When `input_type` is set to a Pydantic model, the SDK instructs the LLM to provide a JSON object matching that model's schema as an argument to the handoff tool. This validated data is then passed to your `on_handoff` callback.

```python
from pydantic import BaseModel, Field
from agents import Agent, Runner, handoff, RunContextWrapper
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Define the data schema for the handoff
class EscalationData(BaseModel):
    reason: str = Field(description="A brief explanation for why the escalation is necessary.")
    ticket_id: str = Field(description="The customer's support ticket ID.")

# 2. The on_handoff callback now accepts the validated data
def process_escalation(ctx: RunContextWrapper, data: EscalationData) -> None:
    print("
--- Escalation Protocol Initiated ---")
    print(f"Escalating ticket {data.ticket_id}")
    print(f"Reason: {data.reason}")
    print("------------------------------------")
    # In a real app, you might now create a Jira ticket or alert a human.

# 3. Define the agents and the typed handoff
human_support_agent = Agent(name="Human Support Team", instructions="...")
triage_agent = Agent(
    name="Triage Agent",
    instructions="You are an AI assistant. If you cannot resolve the issue, escalate to a human.",
    model=DEFAULT_LLM,
    handoffs=[
        handoff(
            agent=human_support_agent,
            on_handoff=process_escalation,
            input_type=EscalationData # Specify the input schema
        )
    ]
)

def main():

    Runner.run_sync(triage_agent, "My ticket is T-12345. I'm really frustrated, I've asked three times about my refund and the issue is still not solved. I need to speak to a person.")

if __name__ == "__main__":
    main()
```

When this runs, the Triage Agent's LLM is forced to generate a JSON object like `{"reason": "Customer is frustrated about an unresolved refund.", "ticket_id": "T-12345"}`. This data is validated and passed to `process_escalation`, ensuring your workflow receives the structured data it needs.

## Filtering the Conversation with `input_filter`

By default, the agent receiving a handoff sees the *entire* conversation history up to that point. This is often desirable, but can sometimes be problematic. The history might contain confusing tool calls or irrelevant chatter that could distract the specialist agent.

The `input_filter` parameter on the `handoff()` function gives you complete control over the context that gets passed along. It's a function that receives the current state of the conversation (as a `HandoffInputData` object) and must return a new, modified `HandoffInputData` object.

The Agents SDK provides some convenient pre-built filters in `agents.extensions.handoff_filters`. A very common one is `remove_all_tools`, which strips all `ToolCallItem` and `ToolCallOutputItem` messages from the history.

Let's see it in action.

```python
from agents import Agent, Runner, handoff, function_tool
from agents.extensions import handoff_filters
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

@function_tool
def check_system_status() -> str:
    """Checks the internal system status."""
    return "All systems are operational."

# Specialist agent that should not be concerned with system status checks.
faq_agent = Agent(
    name="FAQ Agent",
    instructions="Answer questions concisely. Do not comment on your tools or previous turns.",
    model=DEFAULT_LLM
)

# Triage agent has a tool and a handoff with a filter
triage_agent = Agent(
    name="Triage Agent",
    instructions="Use your tool to check status, or handoff to the FAQ agent.",
    model=DEFAULT_LLM,
    tools=[check_system_status],
    handoffs=[
        handoff(
            agent=faq_agent,
            input_filter=handoff_filters.remove_all_tools # The filter
        )
    ]
)

def main():

    # 1. First, call the tool on the triage agent
    result1 = Runner.run_sync(triage_agent, "First, check the system status.")
    print("--- Turn 1 Complete. Tool call was made. ---
")

    # 2. Now, ask a question that triggers the handoff
    result2 = Runner.run_sync(
        triage_agent,
        result1.to_input_list() + [{"role": "user", "content": "What is an LLM?"}]
    )

    print(f"FAQ Agent Final Answer: {result2.final_output}")
    print("
--- Final conversation history for FAQ Agent ---")
    # We inspect the history the final agent saw.
    # The tool call items from turn 1 will be missing.
    for item in result2.to_input_list():
      if item.get("tool_calls") or item.get("type") == "function_call_output":
          assert False, "Tool item was found in history, but should have been filtered!"
    print("History verified: No tool items found.")

if __name__ == "__main__":
    main()
```

![*An `input_filter` modifying the context before passing it to the next agent.*](/assets/img/2025-06-23-handoffs-and-agent-delegation/figure-2.png)



> ## Be Careful with Context Removal
> 
> While powerful, `input_filter` should be used with care. Aggressively removing history can cause the next agent to lose crucial context, leading it to ask for information the user has already provided. The `remove_all_tools` filter is generally safe, but custom filters that remove user or assistant messages should be tested thoroughly.
{: .prompt-info }

## Chapter Summary

In this chapter, we explored handoffs, the SDK's primary mechanism for delegating control between agents. We learned how to create simple handoffs and how to customize their behavior, data requirements, and the context they inherit using the `handoff()` helper function. We contrasted this pattern with the "agent-as-tool" approach from the previous chapter, highlighting that handoffs are about a permanent transfer of conversational responsibility.

You now understand the two main paradigms for multi-agent interaction: delegation with `as_tool()` and delegation with `handoff()`. In the next part of the book, we will move on to advanced orchestration concepts, showing you how to combine these primitives into robust, real-world patterns for building complex AI applications.
