---
title: "Chapter 4 - Anatomy of an Agent"
date: "2025-06-23 10:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

The `Agent` is the fundamental blueprint for your AI workers; mastering its configuration is the key to building powerful and predictable systems. In this chapter, we will dissect the `Agent` class, exploring how to define its core behavior, exert fine-grained control over the model, manage state with context, enforce structured outputs, and observe its operations with lifecycle hooks.

## Core Agent Configuration

At its most basic, an `Agent` is defined by its identity and the model that powers it. Let's look at the essential parameters.

*   `name`: A human-readable string to identify the agent. This is crucial for debugging and is used extensively in tracing to show which agent performed which action.
*   `instructions`: The system prompt. This is the most critical piece of configuration, as it defines the agent's persona, goals, constraints, and how it should behave.
*   `model`: A string identifying the LLM to use. The Agents SDK uses `litellm` under the hood, allowing you to specify models from over 100 providers using the format `"litellm/<provider>/<model_name>"`.
*   `model_settings`: An optional `ModelSettings` object that allows you to fine-tune the LLM's generation parameters, such as `temperature`, `top_p`, and `max_tokens`.

Let's create an agent with specific model settings to make its creative writing more focused.

```python
#
# Configuring an agent with specific model settings.
#

from agents import Agent, Runner, ModelSettings
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

def main():

    # Define the Agent with custom ModelSettings
    storyteller_agent = Agent(
        name="Creative Storyteller",
        instructions="You are a master storyteller who writes compelling, short opening paragraphs for fantasy novels.",
        model=DEFAULT_LLM,
        model_settings=ModelSettings(
            temperature=0.8, # Increase creativity
            max_tokens=250   # Limit the length of the response
        )
    )

    # Use the Runner to execute the agent
    result = Runner.run_sync(
        storyteller_agent,
        "A lone knight approaching a dragon's lair."
    )

    print(result.final_output)

if __name__ == "__main__":
    main()
```


>  Crafting Effective Instructions
> 
> The `instructions` parameter is your primary tool for prompt engineering. A good system prompt should be clear, specific, and provide context. Consider including:
> 
> *   **Persona:** "You are a senior financial analyst."
> *   **Goal:** "Your goal is to provide a concise summary of the provided text."
> *   **Constraints:** "You must not use technical jargon. Always respond in under 100 words."
> *   **Formatting:** "The output must be a markdown-formatted list."
{: .prompt-info }

## Advanced Model Control

Beyond basic temperature and token limits, the `ModelSettings` object provides powerful parameters to precisely control the LLM's behavior, especially regarding tool use and reasoning.

*   **`tool_choice`**: This parameter guides or dictates the model's tool-using behavior. It can be:
    *   `"auto"` (default): The model decides whether to call a tool.
    *   `"required"`: The model *must* call one of its available tools.
    *   `"none"`: The model *must not* call any tools.
    *   A string with a tool's name (e.g., `"get_weather"`): The model *must* call that specific tool.

*   **`parallel_tool_calls`**: A boolean that, if `True`, allows the model to request multiple tool calls in a single turn. This can dramatically improve performance by executing independent data-gathering steps concurrently.

*   **`reasoning`**: When supported by the model, this setting instructs the LLM to emit its "chain-of-thought" or reasoning steps as it works towards a decision. These steps are captured as `ReasoningItem` events in the `RunResult`.

Let's build a workflow orchestrator that uses all three of these settings to efficiently prepare a system for a user.

```python
import asyncio
from agents import Agent, Runner, function_tool, ModelSettings

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# --- Tool Definitions ---
@function_tool
async def setup_database(user_id: str) -> str:
    """Prepares the user's database."""
    print(f"[Tool]: Setting up database for {user_id}...")
    await asyncio.sleep(1) # Simulate network latency
    print(f"[Tool]: Database ready for {user_id}.")
    return "Database connection is ready."

@function_tool
async def authenticate_user(user_id: str) -> str:
    """Authenticates the user and retrieves their permissions."""
    print(f"[Tool]: Authenticating {user_id}...")
    await asyncio.sleep(0.5) # Simulate authentication call
    print(f"[Tool]: {user_id} is authenticated as 'admin'.")
    return "User is authenticated with 'admin' role."

# --- Agent Definition ---
async def main():

    orchestrator_agent = Agent(
        name="Workflow Orchestrator",
        instructions="You are a system orchestrator. Your job is to prepare the system for the user by calling all necessary setup tools.",
        model=DEFAULT_LLM, # A model that supports parallel calls
        tools=[setup_database, authenticate_user],
        model_settings=ModelSettings(
            tool_choice="required",            # Must call tools on the first turn
            parallel_tool_calls=True,          # Encourage concurrent tool calls
        )
    )

    # --- Run the Orchestration ---
    result = await Runner.run(
        orchestrator_agent,
        "Prepare the system for user 'alex-456'."
    )

    print("
--- Final Output ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

When you run this, you'll see the print statements from both tools interleave, confirming they were executed concurrently. You'll also see the model's reasoning printed before the final output.


>  `parallel_tool_calls` is Provider-Dependent
> 
> The ability to request multiple tools in one turn is an advanced feature. While newer OpenAI models (`gpt-4o`, etc.) excel at this, support among other providers varies. 
> 
> However, if you use a model that does not support it, the SDK will not fail. Instead, the model will likely fall back to calling the tools sequentially, one per turn, which will be less efficient. Always test this behavior with your chosen model.
{: .prompt-info }

![*Sequence diagram for parallel tool calls.*](/assets/img/2025-06-23-anatomy-of-an-agent/figure-1.png)



> Forcing tool use with `tool_choice="required"` is powerful but dangerous. If you don't manage it carefully, you can create an infinite loop where the agent calls a tool, gets the result, and is then forced to call a tool again.
> 
> To prevent this, the SDK has a built-in safety feature: the `Agent`'s `reset_tool_choice` parameter, which defaults to `True`. After an agent turn in which a tool is used, the `Runner` will automatically reset the `tool_choice` setting to `"auto"` for the next turn, allowing the LLM to generate a final response instead of being forced to call another tool. You can disable this by setting `agent.reset_tool_choice = False` if you have a specific use case that requires continuous forced tool use.
{: .prompt-info }

## Dynamic Instructions

While static instructions are powerful, some workflows require the agent's behavior to adapt based on the current situation. You can achieve this by passing a callable (a function) to the `instructions` parameter instead of a string. This function will be executed just before the LLM is called, giving you a chance to generate a system prompt dynamically.

The function receives a `RunContextWrapper` object, which gives it access to the application's state. Let's create a "moody" agent whose instructions change based on a value in our context.

```python
from dataclasses import dataclass
from typing import Literal
from agents import Agent, Runner, RunContextWrapper

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Define a context class to hold our state
@dataclass
class AgentContext:
    mood: Literal["happy", "grumpy", "poetic"]

# 2. Define the dynamic instructions function
def get_moody_instructions(
    run_context: RunContextWrapper[AgentContext],
    agent: Agent[AgentContext]
) -> str:
    """Generates instructions based on the mood in the context."""
    mood = run_context.context.mood
    if mood == "happy":
        return "You are a cheerful and optimistic assistant. Use lots of exclamation points!"
    elif mood == "grumpy":
        return "You are a grumpy assistant who answers questions reluctantly. Keep it short."
    else: # poetic
        return "You answer all questions in the form of a rhyming couplet."

# 3. Create the agent with the dynamic instructions function
moody_agent = Agent[AgentContext](
    name="Moody Assistant",
    instructions=get_moody_instructions, # Pass the function, not its result
    model=DEFAULT_LLM
)

def main():

    # Run the agent with different moods
    for mood in ["happy", "grumpy", "poetic"]:
        print(f"--- Running agent in a {mood.upper()} mood ---")
        context = AgentContext(mood=mood)
        result = Runner.run_sync(
            moody_agent,
            "What is the capital of France?",
            context=context # Pass the context to the runner
        )
        print(f"Agent: {result.final_output}
")

if __name__ == "__main__":
    main()
```

## Managing State with Context

By default, agents are stateless. Each `Runner.run` call is independent. However, most real-world applications require state—a way to remember user information, track conversation history, or provide dependencies like database connections. The SDK manages this through a powerful and type-safe **Context** system.

The system is generic, denoted by `Agent[TContext]`. You can define any Python class to serve as your context, `TContext`, and the SDK ensures it's available wherever you need it.

The context is passed to tools and hooks inside a `RunContextWrapper`. This wrapper contains two key properties:

*   `wrapper.context`: Your custom context object that you passed to the `Runner`.
*   `wrapper.usage`: A `Usage` object that tracks token counts and API requests for the run so far.

```python
from dataclasses import dataclass, field
from agents import Agent, Runner, RunContextWrapper, function_tool

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Define a context class to hold session data
@dataclass
class UserSession:
    user_id: str
    items_in_cart: list[str] = field(default_factory=list)

# 2. Define a tool that interacts with the context
@function_tool
def add_to_cart(
    ctx: RunContextWrapper[UserSession], # Receives the context wrapper
    item_name: str
) -> str:
    """Adds an item to the user's shopping cart."""
    ctx.context.items_in_cart.append(item_name)
    return f"Successfully added '{item_name}' to the cart."

# 3. Define the agent, specifying its context type
shopping_agent = Agent[UserSession](
    name="Shopping Assistant",
    instructions="Help the user with their shopping. Use your tools to manage the cart.",
    model=DEFAULT_LLM,
    tools=[add_to_cart]
)

def main():
    session_context = UserSession(user_id="alex-123")
    Runner.run_sync(
        shopping_agent,
        "Please add a 'laptop' to my cart.",
        context=session_context
    )
    print(f"Cart contents from context: {session_context.items_in_cart}")

if __name__ == "__main__":
    main()
```

>  Context is Local, Not Sent to the LLM
> 
> It's critical to understand that the `context` object itself is **never** sent to the LLM. It is a local Python object for your code's use only. If you need the LLM to be aware of information from the context (like a user's name), you must explicitly include it in the `instructions` or have a tool return that information into the conversation history.
{: .prompt-info }

## Typed Outputs for Structured Data

Often, you don't want a free-form text response from an agent; you want structured data that your application can easily work with. The `output_type` parameter on the `Agent` class lets you specify a Pydantic `BaseModel` or `dataclass` that the agent's final output must conform to.

The SDK will automatically:

1.  Generate a JSON schema from your Pydantic model.
2.  Instruct the LLM to respond with JSON matching that schema.
3.  Parse and validate the LLM's JSON output, returning a Python object of your specified type.

```python
import asyncio
import os
from pydantic import BaseModel, Field
from agents import Agent, Runner

class CalendarEvent(BaseModel):
    event_name: str = Field(description="The name or title of the event.")
    date: str = Field(description="The date of the event, in YYYY-MM-DD format.")

event_extractor_agent = Agent(
    name="Event Extractor",
    instructions="Extract the event details from the user's message.",
    model="litellm/gemini/gemini-1.5-flash-latest",
    output_type=CalendarEvent
)
# ...
```


> By default, the SDK generates JSON schemas in **"strict mode"**. This enforces a subset of the JSON Schema standard that guarantees the LLM's output will be valid JSON. This means some Pydantic features, like `Union` types or dictionaries with non-string keys, are not supported in `output_type`. If you must use a non-strict schema, you can configure it by wrapping your type in `AgentOutputSchema(MyType, strict_json_schema=False)`.
{: .prompt-info }

## Observing the Agent with Lifecycle Hooks

For logging, monitoring, or executing custom logic at specific moments, the SDK provides a system of **hooks**. You can attach a custom hooks class to your `Agent` to be notified of key events in its lifecycle. Create a class that inherits from `AgentHooks` and override the methods you're interested in, such as `on_start`, `on_end`, `on_tool_start`, and `on_tool_end`.

```python
from agents import AgentHooks, Tool, RunContextWrapper

class MyAgentObserver(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        print(f"Event: Agent {agent.name} is starting.")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        print(f"Event: Agent {agent.name} finished tool '{tool.name}'.")

# Attach the hooks to an agent
hooked_agent = Agent(
    name="My Agent",
    instructions="...",
    hooks=MyAgentObserver()
)
```
These hooks give you precise control points to inject your own logic into the agent's execution flow without modifying the core SDK `Runner`.

## Chapter Summary

In this chapter, we moved from theory to practice, deeply exploring the `Agent` class—the primary blueprint for our AI workers. We've learned how to configure its core properties, how to exert fine-grained control over model behavior with settings like `tool_choice` and `parallel_tool_calls`, and how to make agents adaptive using dynamic instructions and stateful context. We also saw how to get reliable, structured data using typed outputs and how to observe an agent's internal lifecycle using hooks.

You are now equipped with a detailed understanding of how to define a single, powerful agent. In the [next chapter](https://iamulya.one/posts/empowering-agents), we will expand our agent's capabilities significantly by diving deep into the world of **Tools**, enabling our agents to read files, search the web, and interact with any API you can imagine.
