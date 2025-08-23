---
title: "Chapter 11 - Orchestrating Agent Behavior: Flows and Planners"
date: "2025-08-22 13:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

So far, we've built `LlmAgent` instances that can understand user input, generate text, and use tools. However, for more complex tasks, agents often need to follow a multi-step process, make intermediate decisions, and potentially even create and execute plans. ADK provides **LLM Flows** and **Planners** to manage and orchestrate these more sophisticated behaviors.

This chapter explores how LLM Flows define the interaction loop with the LLM and tools, and how Planners can be integrated to enable agents to create and follow explicit plans.

## Understanding LLM Flows (`BaseLlmFlow`)

An **LLM Flow** (`google.adk.flows.llm_flows.BaseLlmFlow`) is an internal ADK component that dictates the sequence of operations an `LlmAgent` performs during a single turn (or step) of interaction. It manages the loop of:

1. Preparing the `LlmRequest` (gathering history, instructions, tools).
2. Calling the LLM.
3. Processing the `LlmResponse` (handling text, tool calls, errors).
4. Deciding if further interaction with the LLM is needed (e.g., after a tool call) or if a final response has been reached.

You typically don't instantiate `BaseLlmFlow` directly. Instead, `LlmAgent` instances are internally assigned a flow type based on their configuration.

**Key Responsibilities of an LLM Flow:**

- Invoking **LLM Flow Processors** (see Section 11.2) to modify `LlmRequest` before sending it to the LLM.
- Calling the `BaseLlm` instance's `generate_content_async` method.
- Invoking **LLM Flow Processors** to modify the `LlmResponse` after receiving it from the LLM.
- Handling tool calls requested by the LLM, including invoking the appropriate `BaseTool` and managing the tool response.
- Orchestrating agent-to-agent transfers (in more advanced flows like `AutoFlow`).

**Default Flows:**

- **`SingleFlow` (`google.adk.flows.llm_flows.single_flow.SingleFlow`):**
    - This is the default flow for an `LlmAgent` that is not configured for complex multi-agent transfers (i.e., an agent that primarily interacts with tools and the user directly).
    - It handles the basic loop of LLM calls and tool executions until a final textual response is generated.
- **`AutoFlow` (`google.adk.flows.llm_flows.auto_flow.AutoFlow`):**
    - Inherits from `SingleFlow` and adds capabilities for **agent-to-agent transfer**.
    - It automatically includes the necessary logic and internal tools (like `transfer_to_agent`) to allow the LLM to decide to delegate a task to another registered sub-agent or its parent agent.
    - An `LlmAgent` typically uses `AutoFlow` if it has `sub_agents` defined or if its `disallow_transfer_to_parent` / `disallow_transfer_to_peers` flags are not set to completely restrict transfers.
    
    ```python
    from google.adk.agents import Agent
    
    # This agent will implicitly use SingleFlow because it has no sub_agents
    # and default transfer settings allow it to consider only itself if no parent.
    simple_tool_user_agent = Agent(
        name="simple_tool_user",
        model="gemini-1.5-flash-latest",
        instruction="Use tools if needed to answer questions.",
        # tools=[some_tool_instance] # (Assume a tool is added)
    )
    
    # This agent, by having a sub_agent, would likely use AutoFlow by default.
    # orchestrator_agent = Agent(
    #     name="orchestrator",
    #     model="gemini-1.5-pro-latest",
    #     instruction="Coordinate tasks. You can delegate to sub_agents.",
    #     sub_agents=[simple_tool_user_agent]
    # )
    
    ```
    

The specific flow an agent uses is determined internally by ADK based on the agent's configuration (e.g., presence of sub-agents, transfer disallow flags).


> ## Flows Encapsulate Interaction Logic
> 
> LLM Flows are a powerful abstraction within ADK. They separate the how of LLM interaction (the loop, tool handling, processor invocation) from the what (the agent's specific instructions, tools, and model). This allows ADK to evolve its interaction patterns without requiring changes to your core agent definitions.
> {: .prompt-info }

## LLM Flow Processors: Customizing the Request-Response Cycle

LLM Flows are made highly extensible through **LLM Flow Processors**. These are small, focused classes that can intercept and modify the `LlmRequest` before it's sent to the LLM, or the `LlmResponse` after it's received.

ADK uses a series of built-in processors to implement core functionalities. Each `BaseLlmFlow` (like `SingleFlow` and `AutoFlow`) has a list of `request_processors` and `response_processors`.

**Key Built-in Request Processors (executed in order):**

- **`basic.request_processor`**: Sets up fundamental `LlmRequest` fields like the model name and generation config from the agent's attributes.
- **`auth_preprocessor.request_processor`**: Handles authentication-related information, particularly for resuming tool calls that required user authentication.
- **`instructions.request_processor`**: Populates the `system_instruction` in the `LlmRequest` from the agent's `instruction` and `global_instruction`, including performing state injection.
- **`identity.request_processor`**: Adds a default instruction telling the LLM its name and description.
- **`contents.request_processor`**: Builds the `contents` (conversation history) for the `LlmRequest` by filtering and formatting events from the session.
- **`_nl_planning.request_processor` (for Planners)**: If a planner is active, this adds planning-specific instructions.
- **`_code_execution.request_processor` (for Code Executors)**: If a code executor is active (and not `BuiltInCodeExecutor`), this might preprocess data files or convert previous code execution parts to text.
- **`agent_transfer.request_processor` (for `AutoFlow`)**: Adds instructions and tool declarations related to agent-to-agent transfer.
- *(Tool-specific `process_llm_request` methods)*: Each tool added to an agent also gets a chance to process the `LlmRequest` (e.g., to add its `FunctionDeclaration`).

**Key Built-in Response Processors:**

- **`_nl_planning.response_processor`**: If a planner is active, this processes the LLM's response for planning-related artifacts (e.g., extracting thoughts, updating plan state).
- **`_code_execution.response_processor`**: If a code executor is active, this extracts code from the LLM's response, invokes the executor, and prepares the execution result to be sent back to the LLM.

![*Diagram: Role of LLM Flow Processors in the request-response cycle.*](/assets/img/2025-08-22-orchestrating-agent-behavior-flows-and-planners/figure-1.png)


While you typically don't write these processors yourself unless deeply customizing ADK, understanding their existence and order helps in debugging and predicting agent behavior. For example, knowing that `instructions.request_processor` runs before tools add their declarations means your agent's main instruction can refer to tools that will be declared later in the processing chain.

## Introduction to Planners (`BasePlanner`)

For tasks requiring multiple steps, complex reasoning, or dynamic adaptation based on intermediate results, simply prompting an LLM might not be sufficient. **Planners** provide a mechanism for agents to create and follow an explicit plan of action.

- **`google.adk.planners.BasePlanner`**: The abstract base class for all planners.
    - **`build_planning_instruction(readonly_context, llm_request) -> Optional[str]`**: This method is called by `_nl_planning.request_processor`. It should return a string containing instructions for the LLM on how to generate a plan, think step-by-step, or structure its reasoning. This instruction is appended to the main system prompt.
    - **`process_planning_response(callback_context, response_parts) -> Optional[List[types.Part]]`**: This method is called by `_nl_planning.response_processor`. It receives the raw `Part`s from the LLM's response and can process them to:
        - Extract "thoughts," "plans," or "actions" identified by special tags.
        - Modify session state (via `callback_context.state`) to store the current plan, completed steps, or observations.
        - Potentially alter the `response_parts` before they are further processed (e.g., for tool calls or final output).

An `LlmAgent` is equipped with a planner by assigning an instance of a `BasePlanner` subclass to its `planner` attribute.

## `BuiltInPlanner`: Utilizing Model's Native Planning

Some advanced LLMs (particularly newer Gemini models) have built-in "thinking" or "planning" capabilities that can be enabled through specific configurations.

- **`google.adk.planners.BuiltInPlanner`**: This planner is used to activate and configure such native model planning features.
    - It takes a `thinking_config: types.ThinkingConfig` argument during initialization.
    - Its `build_planning_instruction` method is a no-op (it doesn't add extra text instructions, as the `thinking_config` handles it).
    - Its `apply_thinking_config(llm_request)` method (called by `_nl_planning.request_processor`) adds the `self.thinking_config` to the `LlmRequest.config`.
    - Its `process_planning_response` method is also typically a no-op, as the model's built-in planning usually structures its output (including "thoughts" or intermediate steps if configured) in a way ADK can already handle.

```python
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner # Key import
from google.genai.types import ThinkingConfig # For ThinkingConfig
from google.adk.tools import FunctionTool # Example tool
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_REASONING_LLM
load_environment_variables()

def get_product_price(product_id: str) -> dict:
    """Gets the price for a given product ID."""
    prices = {"prod123": 29.99, "prod456": 49.50}
    if product_id in prices:
        return {"product_id": product_id, "price": prices[product_id]}
    return {"error": "Product not found"}

price_tool = FunctionTool(func=get_product_price)

# Configure the built-in thinking
product_thinking_config = ThinkingConfig(include_thoughts=True)

# Create the planner
builtin_item_planner = BuiltInPlanner(thinking_config=product_thinking_config)

agent_with_builtin_planner = Agent(
    name="smart_shopper_builtin",
    model=DEFAULT_REASONING_LLM, # Ensure this model supports ThinkingConfig
    instruction="You are an assistant helping a user find product prices. Think step-by-step and use tools.",
    tools=[price_tool],
    planner=builtin_item_planner # Assign the planner
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=agent_with_builtin_planner, app_name="BuiltInPlanApp")
    session_id = "s_builtinplan"
    user_id = "plan_user"
    create_session(runner, user_id=user_id, session_id=session_id)

    prompt = "What's the price of product prod123 and then prod456?"

    print(f"YOU: {prompt}")
    user_message = Content(parts=[Part(text=prompt)], role="user")  # User message to the agent
    print("ASSISTANT: ", end="", flush=True)

    async def main():
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
                    elif hasattr(part, 'thought') and part.thought: # Check if part is a thought
                        print(f"\n  [THOUGHT]: {part.text.strip() if part.text else 'No text in thought'}\n  ", end="")
                    # Also print tool calls/responses for clarity if they appear as separate events
                    elif part.function_call:
                         print(f"\n  [TOOL CALL]: {part.function_call.name}({part.function_call.args})\n  ", end="")
                    elif part.function_response:
                         print(f"\n  [TOOL RESPONSE to {part.function_response.name}]: {part.function_response.response}\n  ", end="")
        print()

    import asyncio
    asyncio.run(main())
```


> ## `BuiltInPlanner` for Simplicity with Capable Models
> 
> If your target LLM offers robust built-in planning or "thinking" capabilities, `BuiltInPlanner` is often the easiest way to leverage them. You configure the desired `ThinkingConfig` and ADK handles passing it to the model. The model then internally structures its intermediate reasoning steps and tool calls.
> {: .prompt-info }


> ## Model Support for `ThinkingConfig`
> 
> Not all models support `ThinkingConfig`, or they may support different modes and options. Always consult the documentation for your specific model version to understand its planning capabilities and the correct `ThinkingConfig` parameters. Using unsupported configurations can lead to errors or unexpected behavior.
> {: .prompt-info }

## `PlanReActPlanner`: Implementing the ReAct (Reason+Act) Pattern

The ReAct (Reason + Act) paradigm is a popular prompting strategy for enabling LLMs to solve complex tasks by iteratively:

1. **Reasoning** about the current state and the overall goal.
2. Deciding on an **Action** (often a tool call) to take next.
3. Making an **Observation** (the result of the action).
4. Repeating the process, incorporating the observation into new reasoning.

The `google.adk.planners.PlanReActPlanner` helps implement a variation of this by instructing the LLM to explicitly structure its output with planning, reasoning, and action tags.

**How it Works with ADK:**

- **`build_planning_instruction(...)`**: Injects a detailed prompt into the `LlmRequest` instructing the LLM to:
    - First, generate an overall plan under a `/*PLANNING*/` tag.
    - Then, for each step, provide reasoning under `/*REASONING*/`, followed by an action (tool call) under `/*ACTION*/`.
    - If a plan needs revision based on observations, use `/*REPLANNING*/`.
    - Finally, provide the answer under `/*FINAL_ANSWER*/`.
- **`process_planning_response(...)`**:
    - Parses the LLM's response for these tags.
    - Parts tagged as `/*PLANNING*/`, `/*REASONING*/`, `/*ACTION*/` (if it's text before a tool call), or `/*REPLANNING*/` are marked as "thoughts" (by setting `part.thought = True`). This means they are typically logged in the trace but not directly shown to the end-user as part of the conversational response (unless the final answer itself is empty and only thoughts were produced).
    - Tool calls (which the LLM should place after an `/*ACTION*/` tag or reasoning) are extracted and executed.
    - Content under `/*FINAL_ANSWER*/` is treated as the direct response to the user.
    - It can also update session state (e.g., `state['current_plan']`, `state['last_observation']`) via the `callback_context`.

```python
from google.adk.agents import Agent
from google.adk.planners import PlanReActPlanner # Key import
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions.state import State # For state manipulation
from google.adk.agents.callback_context import CallbackContext # For planner state
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_REASONING_LLM
load_environment_variables()

# Dummy tools for demonstration
def search_knowledge_base(query: str, tool_context: CallbackContext) -> str:
    """Searches the company knowledge base for a query."""
    tool_context.state[State.TEMP_PREFIX + "last_search_query"] = query # Example of using temp state
    if "policy" in query.lower():
        return "Found document: 'HR001: Work From Home Policy - Employees can work remotely twice a week with manager approval.'"
    if "onboarding" in query.lower():
        return "Found document: 'IT005: New Employee Onboarding Checklist - Includes setting up accounts and mandatory training.'"
    return "No relevant documents found for your query."

def request_manager_approval(employee_id: str, reason: str) -> str:
    """Sends a request to the employee's manager for approval."""
    return f"Approval request sent for employee {employee_id} regarding: {reason}. Status: PENDING."

search_tool = FunctionTool(func=search_knowledge_base)
approval_tool = FunctionTool(func=request_manager_approval)

# Create the PlanReActPlanner
react_planner = PlanReActPlanner()

# Create the agent
hr_assistant_react = Agent(
    name="hr_assistant_react",
    model=DEFAULT_REASONING_LLM, # Needs a strong reasoning model
    instruction="You are an HR assistant. Follow the Plan-Reason-Act-Observe cycle. First, create a plan. Then, for each step, provide reasoning, take an action (use a tool if necessary), and then reason about the observation to proceed or replan. Conclude with a final answer.",
    tools=[search_tool, approval_tool],
    planner=react_planner # Assign the planner
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=hr_assistant_react, app_name="ReActHrApp")
    session_id = "s_react"
    user_id = "hr_user"
    create_session(runner, user_id=user_id, session_id=session_id)

    # This prompt requires multiple steps and decisions
    prompt = "Employee emp456 wants to work from home full-time. What's the process?"

    print(f"YOU: {prompt}")
    user_message = Content(parts=[Part(text=prompt)], role="user")  # User message to the agent
    print("HR_ASSISTANT_REACT (Follow trace in Dev UI for full ReAct flow):
")

    full_response_parts = []
    async def main():
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            # The ReAct planner often produces "thought" parts that are not meant for direct display.
            # The final answer is typically marked or is the last textual part after planning/actions.
            # For this CLI example, we'll just print all non-thought text.
            # In the Dev UI, thoughts are clearly distinguished in the Trace.
            if event.content and event.content.parts:
                for part in event.content.parts:
                    is_thought = hasattr(part, 'thought') and part.thought
                    if part.text and not is_thought:
                        print(part.text, end="")
                        full_response_parts.append(part.text)
                    elif part.text and is_thought: # Optionally print thoughts for CLI demo
                        print(f"
  [THOUGHT/PLAN]:
  {part.text.strip()}
  ", end="")
                    elif part.function_call :
                         print(f"
  [TOOL CALL]: {part.function_call.name}({part.function_call.args})
  ", end="")
                    elif part.function_response:
                         print(f"
  [TOOL RESPONSE to {part.function_response.name}]: {part.function_response.response}
  ", end="")
        print("
--- Combined Agent Response ---")
        print("".join(full_response_parts))

    import asyncio
    asyncio.run(main())
```

**Expected Interaction (Conceptual Trace with `PlanReActPlanner`):**

1. **User:** "Employee emp456 wants to work from home full-time. What's the process?"
2. **LLM (Prompted by `PlanReActPlanner`):**
    
    ```jsx
    [THOUGHT/PLAN]:
      /*PLANNING*/

    1. Search the knowledge base for work from home full-time process or similar terms to find relevant policies or procedures.

    2. Based on the search results, describe the process to the user.
    
    /*REASONING*/
    The user is asking about the process for working from home full-time. 
    The most appropriate tool to find this information is `search_knowledge_base`. I will start by searching for a broad query and then refine if needed.
    
    /*ACTION*/
      
      [TOOL CALL]: search_knowledge_base({'query': 'work from home full-time process'})
    ```
    
3. **ADK:** Executes `search_knowledge_base`. **Event:** `tool_call`
4. **Tool Response:** "Found document: 'HR001: Work From Home Policy...twice a week...manager approval.'"
5. **ADK:** Sends tool response back. **Event:** `tool_response`
6. **LLM:**
    
    ```jsx
    /*REASONING*/
    The initial search for "work from home full-time process" did not yield any results. 
    This suggests that the exact phrase might not be in the knowledge base, 
    or the knowledge base does not contain specific documents 
    for "full-time" remote work. I will try a more general query like 
    "work from home policy" to see if there's a general remote work policy 
    that might cover full-time arrangements.
    
    /*ACTION*/
      
      [TOOL CALL]: search_knowledge_base({'query': 'work from home policy'})
    
    ```
    
7. **ADK:** Executes `search_knowledge_base`. **Event:** `tool_call`
8. **ADK:** Sends tool response back. **Event:** `tool_response`
9. **LLM:**
    
    ```jsx
    [TOOL RESPONSE to search_knowledge_base]: 
    {'result': "Found document: 'HR001: Work From Home Policy - Employees can work remotely twice a week with manager approval.'"}
      
      [THOUGHT/PLAN]:
      /*REASONING*/
    The search for "work from home policy" returned a document titled "HR001: Work From Home Policy - Employees can work remotely twice a week with manager approval." This document indicates that employees can work remotely twice a week with manager approval. However, the user's request is for *full-time* work from home. The current policy seems to only allow for partial remote work. Therefore, the knowledge base does not seem to have information about full-time work from home. I will inform the user about the existing policy and that full-time work from home is not explicitly covered by the found policy.
    
    /*FINAL_ANSWER*/
      
    According to the current "HR001: Work From Home Policy", employees are allowed to work remotely twice a week with manager approval. There is no explicit process or policy found in the knowledge base regarding full-time work from home.
--- Combined Agent Response
    
    ```
    
10. **ADK:** Yields final text. **Event:** `text`

The parts tagged with `/*PLANNING*/` and `/*REASONING*/` would be marked as `part.thought = True` by the `PlanReActPlanner`'s `process_planning_response` method and typically not shown directly to the user but logged in the trace.


> ## Best Practice: Use PlanReActPlanner for Explicit Step-by-Step Reasoning
> 
> PlanReActPlanner is excellent when you want the LLM to explicitly show its work and follow a structured problem-solving approach. It makes the agent's reasoning process more transparent and debuggable by inspecting the tagged thoughts and actions in the trace. It's particularly good for tasks that naturally break down into sequential steps involving tool use and observation.
> {: .prompt-info }


> ## Prompt Verbosity and LLM Adherence with PlanReActPlanner
> 
> - The instructions injected by `PlanReActPlanner` are quite detailed and add to the prompt length.
> - The effectiveness heavily relies on the LLM's ability to consistently follow the tagging format (`/*PLANNING*/`, `/*ACTION*/`, etc.). Stronger reasoning models tend to perform better. Less capable models might struggle with the format or skip steps.
> - You might need to fine-tune the agent's main instruction to reinforce the ReAct pattern if the LLM deviates.
> {: .prompt-info }

## Practical Example: Research Assistant Agent with a Planner

Let's combine concepts: an agent that uses a planner and search tools to research a topic. We can choose either `BuiltInPlanner` (if the model supports it well) or `PlanReActPlanner`. For this example, let's assume we use `PlanReActPlanner` for more explicit step-by-step output for demonstration.

```python
from google.adk.agents import Agent
from google.adk.planners import PlanReActPlanner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search # Using built-in search
from google.adk.tools import FunctionTool
from google.adk.tools.load_web_page import load_web_page 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_REASONING_LLM, DEFAULT_LLM
load_environment_variables()

web_page_loader_tool = FunctionTool(func=load_web_page)
research_planner = PlanReActPlanner()

search_agent = Agent(
    name="search_agent",
    model=DEFAULT_LLM, # Needs good reasoning and tool use
    instruction="You are a research assistant. Use the google search tool to find relevant information and URLs. ",
    tools=[google_search])

research_assistant = Agent(
    name="research_assistant_planner",
    model=DEFAULT_REASONING_LLM, # Needs good reasoning and tool use
    instruction="You are a research assistant. Create a plan to answer the user's research query. "
                "Use search to find relevant information and URLs. "
                "Use load_web_page to get content from specific URLs if needed. "
                "Synthesize the information and provide a comprehensive answer. "
                "Follow the Plan-Reason-Act-Observe cycle meticulously.",
    tools=[AgentTool(search_agent), web_page_loader_tool],
    planner=research_planner
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=research_assistant, app_name="ResearchPlannerApp")
    session_id = "s_research_plan"
    user_id = "research_user"
    create_session(runner, user_id=user_id, session_id=session_id)
    prompt = "What are the main challenges and opportunities in quantum computing today?"

    print(f"YOU: {prompt}")
    user_message = Content(parts=[Part(text=prompt)], role="user")  # User message to the agent
    print("RESEARCH_ASSISTANT (Follow trace in Dev UI for full ReAct flow):
")

    async def main():
        # Using a set to avoid printing duplicate thought/plan sections if they are repeated
        printed_thoughts = set()
        full_final_answer_parts = []

        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    is_thought = hasattr(part, 'thought') and part.thought
                    if part.text and not is_thought:
                        print(part.text, end="", flush=True)
                        full_final_answer_parts.append(part.text)
                    elif part.text and is_thought:
                        thought_text = part.text.strip()
                        if thought_text not in printed_thoughts:
                            print(f"
  [THOUGHT/PLAN]:
  {thought_text}
  ", end="", flush=True)
                            printed_thoughts.add(thought_text)
                    elif part.function_call:
                         print(f"
  [TOOL CALL]: {part.function_call.name}({part.function_call.args})
  ", end="", flush=True)
                    elif part.function_response:
                         print(f"
  [TOOL RESPONSE to {part.function_response.name}]: (Content might be long, check trace)
  ", end="", flush=True)
        print("

--- Research Assistant's Combined Final Answer ---")
        print("".join(full_final_answer_parts).strip())

    asyncio.run(main())
```


> ## Using `google_search` along with other tools in the same agent can result in error!
> 
> If you try to use `google_search` along with any other tool, you will get the following error: 'Tool use with function calling is unsupported’. That is the reason why a separate agent had to be used just for the search feature instead of just adding `google_search` as a normal tool in the tool list for research_assistant instead of as an `AgentTool` (an agent that acts as a tool)
> {: .prompt-info }

Running this example (especially with `adk web .` to see the trace) would demonstrate the agent:

1. Formulating a plan (e.g., search for "quantum computing challenges", search for "quantum computing opportunities", summarize findings).
2. Executing searches using `google_search`.
3. Potentially calling `load_web_page` if it identifies specific URLs from search results.
4. Reasoning about the retrieved information.
5. Finally, compiling the `/*FINAL_ANSWER*/`.

**What's Next?**

We've now explored how LLM Flows manage the basic interaction loop and how Planners (`BuiltInPlanner` and `PlanReActPlanner`) can endow agents with more sophisticated, multi-step reasoning capabilities. This ability to plan and adapt is key to tackling complex tasks.

In the next part of the book, "Part 3: Building and Managing Multi-Agent Systems," we'll expand beyond single agents and learn how to design, implement, and orchestrate systems where multiple specialized agents collaborate to achieve even more complex goals.
