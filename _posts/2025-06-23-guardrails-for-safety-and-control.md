---
title: "Chapter 9 - Guardrails for Safety and Control"
date: "2025-06-23 15:00:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

As we've assembled agents into complex, orchestrated workflows, a new challenge emerges: reliability. How do we ensure our agents stay on task, don't respond to inappropriate queries, and don't produce unsafe or incorrect output? While prompt engineering is our first line of defense, we need a more robust, programmatic way to enforce policies. This is the purpose of **Guardrails**.

A guardrail is a specialized check that runs *in parallel* to an agent's main execution loop. It acts as an independent validator, observing an agent's input or output and making a judgment based on a set of rules. If a rule is violated, the guardrail can trigger a "tripwire," immediately halting the agent's execution and raising an exception. This allows you to build safer, more predictable systems and prevent your agents from going off the rails.

This chapter will explore the two types of guardrails—Input and Output—and demonstrate how to implement them to enforce application-specific policies.

## The Tripwire Concept: Failing Fast

The core mechanism that makes guardrails effective is the **tripwire**. When you define a guardrail, you implement a function that returns a `GuardrailFunctionOutput` object. This object has a critical boolean property: `tripwire_triggered`.

If this property is `True`, the `Runner` immediately stops processing and raises an `InputGuardrailTripwireTriggered` or `OutputGuardrailTripwireTriggered` exception. This "fail-fast" behavior is crucial for efficiency. Imagine your main agent uses a powerful but slow and expensive model. Your guardrail, however, can be a much smaller, faster model. The guardrail can analyze an incoming user request and reject it *before* the expensive model is ever invoked, saving both time and money.

![*An input guardrail halting execution before the main agent runs.*](/assets/img/2025-06-23-guardrails-for-safety-and-control/figure-1.png)


## Input Guardrails: Validating the User's Request

Input guardrails are your application's bouncers. They run against the initial user input when an agent workflow begins, ensuring the request is valid before any significant processing occurs. They only run if the agent is the *first* agent in an execution chain.

A common use case is topic enforcement. Let's design a system where our main agent is a highly specialized code-writing assistant. We want to prevent users from asking it general knowledge questions.

First, we design a simple "judge" agent whose only job is to classify the user's intent.

```python
#
# A guardrail agent to classify user intent.
#
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    is_coding_question: bool = Field(description="Is the user asking a question related to programming, code, or software development?")
    reasoning: str = Field(description="A brief justification for the classification.")

# This agent is small and fast, perfect for a guardrail.
guardrail_agent = Agent(
    name="Intent Classifier",
    instructions="Classify the user's request. Is it a coding-related question?",
    model="litellm/gemini/gemini-1.5-flash-latest",
    output_type=IntentClassification
)
```

Now, we implement the guardrail function itself. We decorate it with `@input_guardrail` and have it run our `guardrail_agent`. The `tripwire_triggered` value will be `True` if the request is *not* a coding question.

```python
#
# The input guardrail implementation.
#
from agents import input_guardrail, GuardrailFunctionOutput, RunContextWrapper, TResponseInputItem

@input_guardrail
async def topic_check_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This guardrail checks if the input is a coding question."""
    print("[Guardrail]: Checking if the request is on-topic...")

    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    classification: IntentClassification = result.final_output

    # The tripwire is triggered if the topic is NOT a coding question.
    should_trip = not classification.is_coding_question

    if should_trip:
        print("[Guardrail]: Tripwire triggered! Request is off-topic.")

    return GuardrailFunctionOutput(
        output_info=classification.reasoning, # We can pass along info for logging
        tripwire_triggered=should_trip
    )
```

Finally, we attach this guardrail to our main agent and build a simple run loop to demonstrate how the exception is caught.

```python
#
# Main application logic with error handling.
#
import asyncio
import os
from agents import Agent, Runner, InputGuardrailTripwireTriggered

def main():

    # Main agent is more powerful and specialized.
    coding_agent = Agent(
        name="Python Expert",
        instructions="You are an expert Python developer who provides concise, accurate code solutions.",
        model=DEFAULT_LLM,
        input_guardrails=[topic_check_guardrail] # Attach the guardrail
    )

    print("--- Test Case 1: On-topic request ---")
    try:
        on_topic_result = Runner.run_sync(
            coding_agent,
            "How do I sort a dictionary by its values in Python?"
        )
        print(f"Agent Response: {on_topic_result.final_output}")
    except InputGuardrailTripwireTriggered as e:
        print(f"This should not happen. Guardrail tripped unexpectedly: {e.guardrail_result.output.output_info}")


    print("
--- Test Case 2: Off-topic request ---")
    try:
        off_topic_result = Runner.run_sync(
            coding_agent,
            "What was the main cause of the fall of the Roman Empire?"
        )
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail correctly tripped!")
        print(f"Reason from guardrail: {e.guardrail_result.output.output_info}")
        print("Main agent was never called, saving time and resources.")

if __name__ == "__main__":
    main()
```


>  Guardrails are for Safety, Not Logic
> 
> It might be tempting to use guardrails to implement primary application logic (e.g., "if the user says 'billing', trip the wire and then manually route to the billing agent"). This is an anti-pattern.
> 
> Guardrails are for validating inputs and outputs against *safety policies*. The core logic of *what to do* should be handled by the agent itself through its `instructions` and `tools`/`handoffs`. Let the agent reason about the next step; use guardrails to ensure it does so safely.
{: .prompt-info }

## Output Guardrails: Validating the Agent's Response

While input guardrails protect the agent from bad user requests, output guardrails protect the user from bad agent responses. They run only on the `final_output` of the *last* agent in a chain, just before the result is returned from the `Runner`.

This is the perfect place to implement checks for:

*   **Content Safety:** Ensuring the response contains no hate speech, violence, etc.
*   **PII (Personally Identifiable Information) Redaction:** Checking that the agent hasn't accidentally leaked sensitive data.
*   **Response Quality:** Validating that the response adheres to a specific format or quality bar.

Let's build a guardrail that checks if an agent's response contains a (fictional) sensitive project codename.

```python
#
# An output guardrail to check for sensitive words.
#
from pydantic import BaseModel
from agents import Agent, Runner, output_guardrail, GuardrailFunctionOutput, RunContextWrapper, OutputGuardrailTripwireTriggered

from tinib00k.utils import DEFAULT_LLM, load_and_check_keys
load_and_check_keys()

# 1. Main agent's output type
class AgentResponse(BaseModel):
    response: str

# 2. Guardrail's output type
class PIIAnalysis(BaseModel):
    contains_pii: bool
    reasoning: str

# 3. Guardrail agent
pii_checker_agent = Agent(
    name="PII Checker",
    instructions="Analyze the text. Does it contain the secret project codename 'Bluebird'?",
    output_type=PIIAnalysis,
    model=DEFAULT_LLM
)

# 4. The output guardrail function. Note the type hint for `output`.
@output_guardrail
async def pii_check_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: AgentResponse # Type hint matches main agent's output_type
) -> GuardrailFunctionOutput:
    """Checks the agent's final response for the sensitive codename."""
    print("[Guardrail]: Checking final output for sensitive data...")
    result = await Runner.run(pii_checker_agent, output.response, context=ctx.context)
    analysis: PIIAnalysis = result.final_output

    should_trip = analysis.contains_pii
    if should_trip:
        print("[Guardrail]: Tripwire triggered! Sensitive data detected.")

    return GuardrailFunctionOutput(
        output_info=analysis.reasoning,
        tripwire_triggered=should_trip
    )

# 5. The main agent, with the output guardrail attached
main_agent = Agent(
    name="Internal Comms Agent",
    instructions="You are writing an internal company update. The secret project is codenamed 'Project Bluebird'. Announce that its launch has been successful.",
    output_type=AgentResponse,
    model=DEFAULT_LLM,
    output_guardrails=[pii_check_guardrail]
)

def main():

    try:
        Runner.run_sync(main_agent, "Write the internal announcement.")
    except OutputGuardrailTripwireTriggered as e:
        print("
Output guardrail successfully caught the sensitive information.")
        print("The response was blocked before it could be sent to the user.")
        print(f"Reason: {e.guardrail_result.output.output_info}")

if __name__ == "__main__":
    main()
```
This demonstrates a complete safety check. The `main_agent` is instructed to use the sensitive term, but the `pii_check_guardrail` inspects its final output and triggers the tripwire, preventing the leak.


>  Guardrail Scope: First and Last Agents
> 
> An important implementation detail to remember:
> 
> *   An agent's `input_guardrails` will only execute if it is the **first** agent being run by the `Runner`.
> *   An agent's `output_guardrails` will only execute if it is the **last** agent to produce a `final_output`.
> 
> This design aligns with their intended purpose: validating the initial entry and final exit points of a workflow.
{: .prompt-info }

## Chapter Summary

In this chapter, we introduced Guardrails, the SDK's primary mechanism for ensuring the safety and reliability of your agentic applications. We learned about the "fail-fast" **tripwire** concept, which allows for efficient validation by halting execution when a policy is violated. We built a practical **Input Guardrail** to enforce topic constraints on a user's request and an **Output Guardrail** to prevent the leakage of sensitive information in an agent's final response.

By strategically placing these checks at the boundaries of your workflows, you can build applications that are not only powerful but also robust, predictable, and safe. You now have all the core components in your toolkit: agents for reasoning, tools for action, handoffs for delegation, patterns for orchestration, and guardrails for safety.

In the [next chapter](https://iamulya.one/posts/tracing-debugging-and-visualization), we will shift our focus from building to observing. We'll explore the SDK's built-in **Tracing** capabilities, an indispensable feature for debugging, visualizing, and understanding the complex, often non-deterministic behavior of your multi-agent systems.
