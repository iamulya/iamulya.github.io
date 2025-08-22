---
title: Chapter 4 - Crafting Your First LlmAgent 
date: "2025-08-22 09:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Previously we surveyed the fundamental components of the Agent Development Kit. Now, it's time to put that knowledge into practice by building and dissecting our first `LlmAgent`. The `LlmAgent` (often aliased as `Agent` for convenience via `from google.adk.agents import Agent`) is the workhorse for creating agents that leverage the power of Large Language Models. This chapter will guide you through its anatomy, how to provide instructions, understand its interaction with LLMs, and customize its behavior using configurations and callbacks.

## Anatomy of an `LlmAgent`

Let's start by revisiting the basic structure of an `LlmAgent` definition. At its simplest, an `LlmAgent` requires a `name` and a `model`.

```python
from google.adk.agents import Agent # LlmAgent is aliased as Agent

# A very minimal LlmAgent
basic_responder_agent = Agent(
    name="basic_responder",
    model="gemini-2.0-flash" # Specifies the LLM to use
)
```

While this agent will function, it lacks specific guidance. Let's look at the key parameters you'll typically use:

- **`name: str`**: (Required) A unique string identifier for the agent. This name is used in logs, traces, and when one agent needs to refer to another (e.g., for transfer). It must be a valid Python identifier and cannot be "user".
- **`model: Union[str, BaseLlm]`**: (Required) Specifies the Large Language Model the agent will use. This can be:
    - A string identifier for a model known to the `LLMRegistry` (e.g., `"gemini-2.0-flash"`).
    - An instance of a `BaseLlm` subclass (e.g., `LiteLlm(model="openai/gpt-4")`).
- **`instruction: Union[str, InstructionProvider]`**: (Optional, but highly recommended) This is the system prompt or the primary set of instructions given to the LLM to guide its behavior, personality, and task execution. It can be a static string or a callable (`InstructionProvider`) that dynamically generates the instruction based on the current context.
- **`description: str`**: (Optional, but important for multi-agent systems) A natural language description of what this agent does. This helps other agents (or an orchestrating LLM) decide if this agent is suitable for a given task.

```python
from google.adk.agents import Agent

polite_translator_agent = Agent(
    name="polite_translator",
    model="gemini-2.0-flash",
    instruction="You are a polite translator. Translate the user's text into French. If the text is already in French, politely inform the user.",
    description="Translates English text to French politely."
)
```


> ## Best Practice: Meaningful Agent Names and Descriptions
> 
> Choose a name that is a good programmatic identifier. The description is crucial for the LLM (and potentially other agents or developers) to understand the agent's purpose. Make it concise but comprehensive. For example, instead of "Agent that does translations," use "Translates user input from English to French, handling polite phrasings."
> {: .prompt-info }

## Working with Instructions: Static vs. Dynamic

The `instruction` parameter is fundamental to shaping your agent's behavior.

**Static Instructions:**

The simplest way is to provide a static string, as shown in the `polite_translator_agent` example. This instruction is sent to the LLM as part of the system prompt with every request.

**Dynamic Instructions with `InstructionProvider`:**

Sometimes, you need the agent's guiding instructions to change based on the current state of the conversation or external factors. For this, ADK allows you to pass a callable (a function or a method) as the `instruction`. This callable is an `InstructionProvider`.

An `InstructionProvider` is a function that takes a `ReadonlyContext` object as input and returns a string (the instruction) or an awaitable that resolves to a string. The `ReadonlyContext` gives you access to the current invocation ID, agent name, and session state (read-only).


> ## Dynamic Instructions for Adaptive Behavior
> 
> InstructionProvider functions are powerful for making agents adapt to changing contexts (e.g., user roles, time of day, specific data in the session state). Use them when an agent's core directive needs to be flexible rather than static.
> {: .prompt-info }


> ## Complexity in Dynamic Instructions
> 
> While powerful, overly complex logic within an `InstructionProvider` can make the agent's behavior harder to predict and debug. Aim for clarity and test these functions thoroughly. Remember, the instruction ultimately guides the LLM, so ensure it's coherent and unambiguous.
> {: .prompt-info }

Following is an example code where the agent greets you depending on the time of day by using an `InstructionProvider`. 

```python
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext 
from datetime import datetime
from building_intelligent_agents.utils import load_environment_variables, DEFAULT_LLM

load_environment_variables()

def get_time_based_greeting_instruction(context: ReadonlyContext) -> str:
    current_hour = datetime.now().hour
    user_name = context.state.get("user:user_name", "there") 
    if 5 <= current_hour < 12: greeting_time = "morning"
    elif 12 <= current_hour < 18: greeting_time = "afternoon"
    else: greeting_time = "evening"
    return f"You are a cheerful assistant. Greet the user '{user_name}' and wish them a good {greeting_time}. Then, ask how you can help."

dynamic_greeter_agent = Agent(
    name="dynamic_greeter", model=DEFAULT_LLM,
    instruction=get_time_based_greeting_instruction, 
    description="Greets the user dynamically based on the time of day and their name."
)

if __name__ == "__main__":
    from google.adk.runners import InMemoryRunner
    from google.genai.types import Content, Part
    runner = InMemoryRunner(agent=dynamic_greeter_agent, app_name="DynamicApp")
    user_id = "jane_doe"; session_id = "session_greet_jane"; initial_state = {"user:user_name": "Jane"}
    runner.session_service._create_session_impl(app_name="DynamicApp", user_id=user_id, session_id=session_id, state=initial_state)
    user_message = Content(parts=[Part(text="Hello")])
    print(f"Running dynamic_greeter_agent for user: {user_id}...")
    for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
    print()
```

In this example, `get_time_based_greeting_instruction` accesses the session state (e.g., `user:user_name`) via the `ReadonlyContext` to personalize the instruction. The instruction sent to the LLM will change depending on when the agent is run and what's in the session state.


> ## A Note on State Injection in Instructions
> 
> By default, ADK attempts to inject values from the session state into your static string instructions if they contain placeholders like `{my_variable}` or `{user:user_name}`. However, when you use an `InstructionProvider` function, this automatic state injection is **bypassed** for the instruction string returned by your provider. This is because your provider function already has access to the `ReadonlyContext` and can explicitly fetch and format any state variables it needs, offering more control.
> {: .prompt-info }

## Understanding LLM Flows: The Default `SingleFlow`

When an `LlmAgent` is invoked, its interaction with the LLM and tools is managed by an **LLM Flow** (`google.adk.flows.llm_flows.BaseLlmFlow`). The default flow for a standalone `LlmAgent` (one not explicitly configured for complex multi-agent transfers) is the `SingleFlow`.

The `SingleFlow` essentially does the following in a loop:

1. **Prepares `LlmRequest`**: Gathers history, instructions, and tool declarations.
2. **Calls LLM**: Sends the request to the LLM.
3. **Processes `LlmResponse`**:
    - If the LLM returns text, it's yielded as an `Event`. This might be the final answer.
    - If the LLM requests a tool call, the `SingleFlow` executes the tool.
    - The tool's response is then packaged and sent back to the LLM in the next iteration of the loop (often for summarization or to inform the next step).
4. This loop continues until the LLM provides a final text response without requesting further tool calls, or an error occurs, or a callback/tool signals to end the invocation.

![*Diagram: Simplified conceptual loop of the `SingleFlow`.*](/assets/img/2025-08-22-crafting-your-first-llmagent/figure-1.png)


We will explore more complex flows like `AutoFlow` (which enables agent-to-agent transfers) in the multi-agent systems part of the book. For now, understanding that `SingleFlow` handles the turn-by-turn conversation and tool use is sufficient.


> ## Flows Abstract LLM Interaction Patterns
> 
> LLM Flows like SingleFlow (and AutoFlow later) encapsulate common patterns of interacting with an LLM, including preparing requests, handling tool calls, and processing responses. Understanding that a flow manages this loop helps you focus on the agent's specific instructions and tools.
> {: .prompt-info }

## LLM Interaction: `LlmRequest` and `LlmResponse`

These two Pydantic models are central to how ADK agents communicate with LLMs:

- **`google.adk.models.LlmRequest`**: Encapsulates everything an agent sends *to* the LLM.
    - `model: Optional[str]`: The target model name.
    - `contents: list[types.Content]`: The conversation history, including previous user messages, agent responses, and tool call/response pairs.
    - `config: Optional[types.GenerateContentConfig]`: Contains:
        - `system_instruction: Optional[str]`: The compiled system prompt.
        - `tools: Optional[list[types.Tool]]`: Declarations of available tools.
        - `temperature`, `top_p`, `max_output_tokens`, `safety_settings`, etc.
        - `response_schema`: If you expect structured JSON output from the LLM.
    - `tools_dict: dict[str, BaseTool]`: A mapping of tool names to their `BaseTool` instances (used internally by ADK).
- **`google.adk.models.LlmResponse`**: Encapsulates what the agent receives *from* the LLM.
    - `content: Optional[types.Content]`: The primary output from the LLM. This `Content` object can contain:
        - `Part(text="...")`: Plain text response.
        - `Part(function_call=FunctionCall(...))`: A request by the LLM to call a specific tool.
        - `Part(inline_data=Blob(...))`: For multimodal responses (e.g., image generation, though less common for agent text responses).
    - `partial: Optional[bool]`: True if this is part of a streaming text response.
    - `usage_metadata: Optional[types.GenerateContentResponseUsageMetadata]`: Information about token counts.
    - `error_code: Optional[str]`, `error_message: Optional[str]`: If an error occurred during the LLM call.

The `Runner` and the `LLM Flow` handle the construction of `LlmRequest` and the conversion of the raw LLM output into an `LlmResponse` and subsequently into `Event` objects.

## Customizing Model Behavior: `generate_content_config`

The `LlmAgent` allows you to pass a `generate_content_config` argument, which should be an instance of `google.genai.types.GenerateContentConfig`. This object lets you control various aspects of the LLM's generation process.

Following is an agent that helps with creative writing, with safety settings turned down using the `generate_content_config` argument.

```python
from google.adk.agents import Agent
from google.genai.types import GenerateContentConfig, SafetySetting, HarmCategory, HarmBlockThreshold
from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM, DEFAULT_REASONING_LLM

load_environment_variables()

custom_safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
]

creative_writer_agent = Agent(
    name="creative_writer", model=DEFAULT_REASONING_LLM, 
    instruction="You are a creative writer. Write a short, imaginative story based on the user's prompt.",
    description="Generates short creative stories.",
    generate_content_config=GenerateContentConfig(
        temperature=0.9, top_p=0.95, top_k=40, max_output_tokens=1024,
        safety_settings=custom_safety_settings)
)

if __name__ == "__main__":
    from google.adk.runners import InMemoryRunner
    from google.genai.types import Content, Part

    current_session_id = "creative_story_session"
    current_user_id = "creative_writer_user"
    runner = InMemoryRunner(agent=creative_writer_agent, app_name="CreativeApp")
    
    create_session(runner, current_session_id, current_user_id)

    user_prompt = Content(parts=[Part(text="A brave squirrel on a quest to find the legendary golden acorn.")], role="user")
    print("Creative Writer Story:")
    
    for event in runner.run(user_id=current_user_id, session_id=current_session_id, new_message=user_prompt):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
    print()
```

Key fields in `GenerateContentConfig`:

- `temperature: float`: Controls randomness. Lower values are more deterministic; higher values are more creative/random.
- `top_p: float`, `top_k: int`: Control token sampling strategies.
- `max_output_tokens: int`: Maximum number of tokens to generate.
- `stop_sequences: list[str]`: Sequences of strings that, if generated, will cause the LLM to stop.
- `safety_settings: list[SafetySetting]`: Configure content safety filters (e.g., for harassment, hate speech).
- `response_mime_type` & `response_schema`: Used when you expect structured JSON output from the LLM (covered in detail later).


> ## `GenerateContentConfig` within ADK
> 
> - **Don't set `system_instruction` here directly.** Use the `LlmAgent.instruction` parameter.
> - **Don't set `tools` here directly.** Use the `LlmAgent.tools` parameter.
> - **Don't set `thinking_config` here directly.** Use the `LlmAgent.planner` parameter with a `BuiltInPlanner`.
> 
> ADK manages these specific fields through its own dedicated agent parameters to ensure proper integration with its flows and tool handling mechanisms.
> {: .prompt-info }


> ## Best Practice: Tune temperature for Desired Output
> 
> The temperature setting in GenerateContentConfig is one of the most impactful for controlling LLM output.
> 
> - Low temperature (e.g., 0.0-0.3): More focused, deterministic, good for factual recall or consistent formatting.
> - High temperature (e.g., 0.7-1.0): More creative, diverse, good for brainstorming or story generation.
>     
> Experiment to find the right balance for your agent's task.
> {: .prompt-info }
    

## Callbacks for Fine-Grained Control

ADK provides several callback points within the `LlmAgent` lifecycle, allowing you to inject custom logic before or after key operations. These callbacks receive a `CallbackContext` object.

- **`before_agent_callback: Optional[BeforeAgentCallback]`**
    - Called before the agent's main `_run_async_impl` or `_run_live_impl` logic begins.
    - Receives: `CallbackContext`.
    - Can return: `Optional[types.Content]`. If content is returned, the agent's normal run is skipped, and this content is yielded as the agent's response. Useful for pre-emptive handling or validation.
- **`after_agent_callback: Optional[AfterAgentCallback]`**
    - Called after the agent's main run logic completes but before the final event is solidified.
    - Receives: `CallbackContext`.
    - Can return: `Optional[types.Content]`. If content is returned, it overrides any response the agent might have generated internally and is yielded as the final agent response. Useful for post-processing or adding boilerplate.
- **`before_model_callback: Optional[BeforeModelCallback]`**
    - Called just before the `LlmRequest` is sent to the LLM.
    - Receives: `CallbackContext`, `LlmRequest` (mutable).
    - Can return: `Optional[LlmResponse]`. If an `LlmResponse` is returned, the actual call to the LLM is skipped, and this response is used instead. Useful for caching, request modification, or mocking LLM calls.
- **`after_model_callback: Optional[AfterModelCallback]`**
    - Called immediately after receiving the `LlmResponse` from the LLM.
    - Receives: `CallbackContext`, `LlmResponse` (mutable).
    - Can return: `Optional[LlmResponse]`. If an `LlmResponse` is returned, it replaces the original LLM response. Useful for response modification, validation, or logging.
    


> ## Callbacks for Monitoring and Modification
> 
> - Use before_model_callback to inspect or modify the exact prompt being sent to the LLM, or to implement caching.
> - Use after_model_callback to inspect or modify the raw LLM response before ADK processes it further (e.g., for tool calls). This is also a good place to log token usage from response.usage_metadata.
> - before_agent_callback is great for input validation or pre-emptive responses based on session state.
> 
> All callback types can be a single callable or a list of callables. If a list, they are executed in order until one returns a non-None value (which then short-circuits further callbacks in that list).
> {: .prompt-info }

The following agent uses callbacks for logging and even to block certain users using a custom `before_agent_callback`.

```python
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part
from typing import Optional 
import logging

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM

load_environment_variables()

def my_before_agent_cb(callback_context: CallbackContext) -> Optional[Content]:
    logger.info(f"[{callback_context.agent_name}] In before_agent_callback. Invocation ID: {callback_context.invocation_id}")
    if "block_user" in callback_context.state.get("user:flags", []):
        logger.warning(f"[{callback_context.agent_name}] User is blocked. Skipping agent run.")
        return Content(parts=[Part(text="I'm sorry, I cannot process your request at this time.")])
    return None

async def my_before_model_cb(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    logger.info(f"[{callback_context.agent_name}] In before_model_callback. Modifying request.")
    if llm_request.contents and llm_request.contents[-1].role == "user":
        llm_request.contents[-1].parts[0].text = f"Consider this: {llm_request.contents[-1].parts[0].text}"
    return None

def my_after_model_cb(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    logger.info(f"[{callback_context.agent_name}] In after_model_callback.")
    if llm_response.content and llm_response.content.parts and llm_response.content.parts[0].text:
        llm_response.content.parts[0].text += " (Processed by after_model_callback)"
    llm_response.custom_metadata = {"source": "after_model_cb_modification"}
    return llm_response

callback_demo_agent = Agent(
    name="callback_demo_agent", model=DEFAULT_LLM,
    instruction="You are an echo agent. Repeat the user's message.",
    description="Demonstrates agent and model callbacks.",
    before_agent_callback=my_before_agent_cb,
    before_model_callback=my_before_model_cb,
    after_model_callback=my_after_model_cb
)

if __name__ == "__main__":
    from google.adk.runners import InMemoryRunner
    runner = InMemoryRunner(agent=callback_demo_agent, app_name="CallbackApp")

    user_id="cb_user"
    session_id="s_normal"
    
    create_session(runner, session_id, user_id)
    
    print("
--- Scenario 1: Normal Run ---")
    user_message1 = Content(parts=[Part(text="Hello ADK!")])
    for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message1):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
            if event.custom_metadata: print(f" [Metadata: {event.custom_metadata}]", end="")
    print()

    print("
--- Scenario 2: User Blocked ---")
    user_message2 = Content(parts=[Part(text="Another message.")])
    blocked_user_session_id = "s_blocked"
    runner.session_service._create_session_impl(
        app_name="CallbackApp", user_id="cb_user_blocked",
        session_id=blocked_user_session_id, state={"user:flags": ["block_user"]}
    )
    for event in runner.run(user_id="cb_user_blocked", session_id=blocked_user_session_id, new_message=user_message2):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
    print()
```

Running this script will show log messages from the callbacks and block a user that has certain flags in the state `"user:flags": ["block_user"]`.

```jsx
Creating session: s_normal for user: cb_user on app: CallbackApp
Session created successfully.

--- Scenario 1: Normal Run ---
INFO:__main__:[callback_demo_agent] In before_agent_callback. Invocation ID: e-ce7ed1b6-979a-4a25-bc96-049ae6e20240
INFO:__main__:[callback_demo_agent] In before_model_callback. Modifying request.
INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.0-flash, backend: GoogleLLMVariant.GEMINI_API, stream: False
INFO:google_adk.google.adk.models.google_llm:
LLM Request:
-----------------------------------------------------------
System Instruction:
You are an echo agent. Repeat the user's message.

You are an agent. Your internal name is "callback_demo_agent".

 The description about you is "Demonstrates agent and model callbacks."
-----------------------------------------------------------
Contents:
{"parts":[{"text":"Handle the requests as specified in the System Instruction."}],"role":"user"}
-----------------------------------------------------------
Functions:

-----------------------------------------------------------

INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent "HTTP/1.1 200 OK"
INFO:google_adk.google.adk.models.google_llm:
LLM Response:
-----------------------------------------------------------
Text:
Handle the requests as specified in the System Instruction.

-----------------------------------------------------------
Function calls:

-----------------------------------------------------------
Raw response:
{"candidates":[{"content":{"parts":[{"text":"Handle the requests as specified in the System Instruction.
"}],"role":"model"},"finish_reason":"STOP","avg_logprobs":5.087303262288597e-7}],"model_version":"gemini-2.0-flash","usage_metadata":{"candidates_token_count":11,"candidates_tokens_details":[{"modality":"TEXT","token_count":11}],"prompt_token_count":54,"prompt_tokens_details":[{"modality":"TEXT","token_count":54}],"total_token_count":65},"automatic_function_calling_history":[]}
-----------------------------------------------------------

INFO:__main__:[callback_demo_agent] In after_model_callback.
Handle the requests as specified in the System Instruction.
 (Processed by after_model_callback) [Metadata: {'source': 'after_model_cb_modification'}]

--- Scenario 2: User Blocked ---
INFO:__main__:[callback_demo_agent] In before_agent_callback. Invocation ID: e-77011598-d1b1-4dc8-98c8-95de79e87738
WARNING:__main__:[callback_demo_agent] User is blocked. Skipping agent run.
I'm sorry, I cannot process your request at this time.
```

Here is a sequence diagram for how the request/response flow is modified.

![*Diagram: Flow of execution with agent and model callbacks.*](/assets/img/2025-08-22-crafting-your-first-llmagent/figure-2.png)



> ## Best Practice: Keep Callbacks Focused
> 
> Callbacks should ideally perform a single, well-defined task (e.g., logging, a specific modification, a validation check). This keeps them maintainable and easier to understand within the overall agent flow. Avoid putting overly complex business logic directly into callbacks if it can be part of the agent's primary logic or a tool.
> {: .prompt-info }


> ## Mutable Objects in Callbacks
> 
> Be mindful when modifying objects like LlmRequest or LlmResponse within callbacks. Changes made will affect the subsequent processing. This is powerful but requires care to avoid unintended side effects. Always log what you're changing for easier debugging.
> {: .prompt-info }

**What's Next?**

We've now covered the essentials of creating and configuring a single `LlmAgent`. While this agent can respond based on its instructions and LLM, its true power is unlocked when it can interact with the outside world. Next, we'll learn how to give our agents the ability to perform actions by defining and using custom Python tools.
