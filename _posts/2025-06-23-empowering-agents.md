---
title: "Chapter 5 - Empowering Agents with Tools"
date: "2025-06-23 11:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

In the previous chapters, we defined what an agent is and explored the mechanics of the `Runner` and the agent loop. We saw that an agent powered by a Large Language Model (LLM) is an excellent reasoner. However, its knowledge is frozen at the time of its training, and it has no ability to interact with the outside world. To build truly useful applications, we must give our agents the ability to perform actions. In the Agents SDK, we do this by providing them with **Tools**.

A tool is an external capability that an agent can invoke to gather information, manipulate data, or trigger an action in another system. This chapter will be a deep dive into the three primary ways to equip your agents with tools: creating custom **Function Tools** from your Python code, leveraging provider-hosted services like **Hosted Tools**, and an advanced pattern where you can use other **Agents as Tools**.

## Function Tools: Your Code as a Capability

The most common and flexible way to create a tool is by using the `@function_tool` decorator. This powerful feature allows you to take almost any Python function and safely expose it to an LLM. The SDK handles the complex work of creating a machine-readable schema that the LLM can use to understand what the tool does, what arguments it requires, and how to call it.

Let's start with the simplest possible example: a tool that adds two numbers.

```python
from agents import Agent, Runner, function_tool
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Define a regular Python function and decorate it
@function_tool
def add(a: int, b: int) -> int:
    """
    Calculates the sum of two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    print(f"[Tool Executed]: add(a={a}, b={b})")
    return a + b

def main():

    # 2. Create an agent and give it the tool
    calculator_agent = Agent(
        name="Calculator Agent",
        instructions="You are a calculator. Use your tools to perform calculations.",
        model=DEFAULT_LLM,
        tools=[add] # Pass the decorated function directly
    )

    # 3. Run the agent with a prompt that requires the tool
    result = Runner.run_sync(
        calculator_agent,
        "What is 115 plus 237?"
    )

    print(f"
Final Answer: {result.final_output}")


if __name__ == "__main__":
    main()

# Expected Output:
#
# [Tool Executed]: add(a=115, b=237)
#
# Final Answer: The sum of 115 and 237 is 352.
```

When the `Runner` executes `calculator_agent`, the LLM receives not only the user's prompt but also a description of the `add` tool. The model reasons that it should call this tool with `a=115` and `b=237`. The SDK executes the call, gets the result `352`, and feeds it back into the agent loop. The LLM then uses this result to formulate the final answer.

### How Function Tools Work Under the Hood

The `@function_tool` decorator is doing a lot of heavy lifting for you. Here's a breakdown of the process:

1.  **Signature Inspection:** It uses Python's built-in `inspect` module to analyze your function's signature (`a: int, b: int`) and determine the names and types of its parameters.
2.  **Docstring Parsing:** It uses the `griffe` library to parse your function's docstring, extracting the overall tool description ("Calculates the sum...") and the description for each argument.
3.  **Schema Generation:** It dynamically builds a Pydantic model representing the function's arguments. This model is then converted into a JSON schema.
4.  **LLM Presentation:** This final JSON schema is what's presented to the LLM, giving it a structured understanding of the tool's capabilities.

![*The process of turning a Python function into a schema for the LLM.*](/assets/img/2025-06-23-empowering-agents/figure-1.png)



>  Rich Type Support
> 
> The function tool schema generation supports more than just primitive types like `int` and `str`. You can use Pydantic models, `dataclass`es, or `TypedDict`s as arguments, and the SDK will generate a nested JSON schema accordingly. This allows you to pass complex, structured data to your tools.
{: .prompt-info }

Let's see an example with a Pydantic model as an input argument.

```python
from agents import Agent, Runner, function_tool
from pydantic import BaseModel

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()


class Location(BaseModel):
    city: str
    country: str

@function_tool
def get_forecast(location: Location) -> str:
    """Provides a weather forecast for a specific location."""
    print(f"[Tool Executed]: get_forecast for {location.city}, {location.country}")
    return f"The weather in {location.city} is expected to be sunny and pleasant."

weather_agent = Agent(
    name="Weather Agent",
    instructions="Provide weather forecasts using your tools.",
    model=DEFAULT_LLM,
    tools=[get_forecast]
)

def main():
    result = Runner.run_sync(weather_agent, "What's the weather like in Paris, France?")
    print(f"
Final Answer: {result.final_output}")


if __name__ == "__main__":
    main()
```
The LLM will correctly generate a JSON object like `{"location": {"city": "Paris", "country": "France"}}` to call this tool.

### Handling Tool Errors Gracefully

Real-world tools can fail. Network requests time out, APIs return errors, or inputs might be invalid. By default, if a tool raises an unhandled exception, the entire agent run will crash. The SDK provides the `failure_error_function` parameter in the `@function_tool` decorator to manage this.

You can provide a callable that receives the context and the exception, and its return value (a string) will be sent back to the LLM as the tool's output. This allows the agent to reason about the failure and potentially retry or inform the user.

```python
from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings
from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()


def custom_error_handler(ctx: RunContextWrapper, error: Exception) -> str:
    """A custom function to format error messages for the LLM."""
    print(f"[Error Handler]: Caught a {type(error).__name__}: {error}")
    return f"The tool failed with an error. Please check your inputs. Error: {error}"

@function_tool(failure_error_function=custom_error_handler)
def divide(a: float, b: float) -> float:
    """Divides the first number by the second number."""
    print(f"[Tool Executed]: Dividing {a} by {b}")
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def main():

    error_handling_agent = Agent(
        name="Error Handling Agent",
        instructions="Perform the division using the divide tool. If an error occurs, explain it to the user.",
        model=DEFAULT_LLM,
        tools=[divide],
        model_settings=ModelSettings(
            tool_choice="required",            # Must call tools on the first turn
        )
    )

    # This will trigger the failure_error_function
    result = Runner.run_sync(error_handling_agent, "Can you divide 10 by 0?")

    print(f"
Final Answer: {result.final_output}")

if __name__ == "__main__":
    main()

# Expected Output:
#
# [Error Handler]: Caught a ValueError: Cannot divide by zero.
#
# Final Answer: I'm sorry, but I cannot perform that calculation. The tool failed with an error because you cannot divide by zero. Please provide a non-zero divisor.
```


> The string returned by your `failure_error_function` is crucial. A clear, descriptive error message gives the LLM the best chance to understand what went wrong. It might then decide to re-call the tool with corrected parameters or explain the problem to the end-user, leading to a much more robust application. Passing `failure_error_function=None` will cause the original exception to be raised, halting the run.
{: .prompt-info }

## Hosted Tools: Provider-Managed Capabilities

While function tools offer maximum flexibility, some LLM providers offer their own powerful, tightly integrated tools that run on their infrastructure. The Agents SDK provides simple wrapper classes for these "hosted tools."

Examples include:

*   `WebSearchTool`: Enables the agent to perform web searches.
*   `CodeInterpreterTool`: Provides a sandboxed environment for running code.
*   `FileSearchTool`: Allows the agent to retrieve information from vector stores you've created.


> **This is a critical point:** Hosted tools are generally specific to the provider that offers them. The tools listed above (`WebSearchTool`, `CodeInterpreterTool`, etc.) are designed for and supported by the **OpenAI Responses API**. They will **not** work when using Gemini or other third-party models via LiteLLM.
> 
> Attempting to use an unsupported hosted tool with a model provider will result in an error. Always check your provider's documentation for their supported tool-use capabilities. Because this book focuses on Gemini, we will not provide a runnable example for these tools, but show the syntax for completeness.
{: .prompt-info }

Here's how you would theoretically use the `WebSearchTool` if you were using an OpenAI model:

```python
#
# NOTE: This example is for demonstration purposes and will NOT
# work with the Gemini models used elsewhere in this chapter.
# It requires an OpenAI model compatible with the Responses API.
#

from agents import Agent, WebSearchTool, OpenAIResponsesModel

# This agent would need to use an OpenAI-specific model class
search_agent = Agent(
    name="Web Searcher",
    instructions="You are a helpful research assistant.",
    model=OpenAIResponsesModel(model="gpt-4o"), # Using an OpenAI model
    tools=[
        WebSearchTool()
    ]
)

# result = Runner.run_sync(
#   search_agent,
#   "What's the latest news about NASA's Artemis program?"
# )
```

## Agents as Tools: Creating a Team of Specialists

One of the most powerful architectural patterns in the Agents SDK is the ability to use one agent as a tool for another. This allows you to create a "manager" or "orchestrator" agent that can delegate complex sub-tasks to a team of specialized agents.

This is different from a handoff (which we'll cover in the next chapter). In a handoff, control is transferred completely. When using an agent as a tool, the orchestrator agent calls the specialist, gets a result back, and then decides what to do next. This is perfect for tasks that can be broken down into parallelizable or sequential sub-problems.

Every `Agent` instance has an `.as_tool()` method that converts it into a `FunctionTool`.

Let's build a translation orchestrator. It will receive a request to translate text into multiple languages and will call specialized agents for each language in parallel.

```python
from agents import Agent, Runner, trace

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

def main():

    # 1. Define Specialist Agents
    spanish_translator = Agent(
        name="Spanish Translator",
        instructions="You are an expert translator. Translate the user's text into Spanish.",
        model=DEFAULT_LLM,
    )

    french_translator = Agent(
        name="French Translator",
        instructions="You are an expert translator. Translate the user's text into French.",
        model=DEFAULT_LLM,
    )

    # 2. Define the Orchestrator and provide the specialists as tools
    orchestrator = Agent(
        name="Translation Orchestrator",
        instructions="You are a project manager for translations. Use your tools to fulfill the user's request. Call all relevant tools.",
        model=DEFAULT_LLM,
        tools=[
            spanish_translator.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Use this to translate a given text into Spanish."
            ),
            french_translator.as_tool(
                tool_name="translate_to_french",
                tool_description="Use this to translate a given text into French."
            ),
        ]
    )

    # 3. Run the orchestration
    with trace("Multi-language Translation Trace"):
        result = Runner.run_sync(
            orchestrator,
            "Please translate the phrase 'hello world' into Spanish and French."
        )

    print(f"Orchestrator's Final Report:
{result.final_output}")


if __name__ == "__main__":
    main()

# Expected Output:
#
# Orchestrator's Final Report:
# The translation of "hello world" into Spanish is "hola mundo" and into French is "bonjour le monde".
```

In this workflow, the orchestrator agent receives the request and sees two tools available: `translate_to_spanish` and `translate_to_french`. It correctly reasons that it should call both. The SDK executes these two sub-agent runs, collects their `final_output`, and passes them back to the orchestrator, which then synthesizes the final report.

![*An orchestrator agent calling two specialist agents as tools.*](/assets/img/2025-06-23-empowering-agents/figure-2.png)


You can even customize what data the sub-agent returns using the `custom_output_extractor` argument in `.as_tool()`. This lets you, for example, return a specific field from a sub-agent's structured output instead of the entire text response.

## Chapter Summary

In this chapter, we fully explored how to give agents capabilities through tools. We started with the versatile `@function_tool` decorator, learning how it turns any Python function into a robust, schema-aware tool for an LLM. We covered handling complex inputs, accessing state via context, and gracefully managing errors. We also discussed the concept of provider-hosted tools and the critical importance of checking for compatibility. Finally, we unlocked a powerful architectural pattern: using entire agents as tools, enabling sophisticated orchestration and delegation.

You are now equipped to build agents that can do more than just talk—they can *act*. In the [next chapter](https://iamulya.one/posts/mcp), we will look at the powerful *Model Context Protocol*.
