---
title: Chapter 13 - Implementing Orchestration Logic with Shell Agents 
date: "2025-08-22 14:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Previously we designed multi-agent architectures and discussed how an orchestrator `LlmAgent` could manage the flow between specialized sub-agents using agent transfers. While you can certainly implement all orchestration logic within a primary `LlmAgent`'s instructions, ADK provides helpful "shell" agents—`SequentialAgent`, `ParallelAgent`, and `LoopAgent`—that offer pre-built structures for common orchestration patterns.

These shell agents are subclasses of `BaseAgent` but are not typically `LlmAgent`s themselves. They don't directly interact with an LLM for their *own* decision-making. Instead, they execute their defined list of `sub_agents` according to a specific pattern (sequentially, in parallel, or in a loop). This can simplify your design by offloading common orchestration control flow to these dedicated components.

## `SequentialAgent`: For Pipeline-Style Workflows

The `google.adk.agents.SequentialAgent` executes its `sub_agents` one after the other, in the order they are provided in its `sub_agents` list. The entire conversation history and state from one sub-agent's execution are available to the next sub-agent in the sequence.

**When to Use:**

- For tasks that naturally break down into a fixed series of steps.
- When the output of one agent is the direct input or context for the next.
- Examples:
    - Data processing pipelines (Ingest -> Clean -> Analyze -> Report).
    - A user interaction flow where the agent must first gather information, then verify it, then perform an action.

**Defining a `SequentialAgent`:**

```python
from google.adk.agents import Agent, SequentialAgent # Import SequentialAgent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Define Sub-Agents for the Pipeline ---

def gather_user_data(name: str, email: str, tool_context: ToolContext) -> dict:
    """Gathers user's name and email and stores it in session state."""
    tool_context.state["user_name_collected"] = name
    tool_context.state["user_email_collected"] = email
    return {"status": "success", "message": f"Data for {name} collected."}

gather_data_tool = FunctionTool(func=gather_user_data)

data_collection_agent = Agent(
    name="data_collector",
    model=DEFAULT_LLM,
    instruction="Your goal is to collect the user's name and email. Use the 'gather_user_data' tool. Ask for name and email if not provided in the initial query.",
    description="Collects user name and email.",
    tools=[gather_data_tool]
)

def validate_email_format(email: str, tool_context: ToolContext) -> dict:
    """Validates the format of an email address."""
    # Simple regex for basic email validation
    import re
    if re.match(r'[\w.-]+@[\w.-]+.\w+', email):
        tool_context.state["email_validated"] = True
        return {"is_valid": True, "email": email}
    else:
        tool_context.state["email_validated"] = False
        return {"is_valid": False, "email": email, "error": "Invalid email format."}

validate_email_tool = FunctionTool(func=validate_email_format)

email_validation_agent = Agent(
    name="email_validator",
    model=DEFAULT_LLM,
    instruction="You will receive an email address, possibly from session state. Your task is to validate its format using the 'validate_email_format' tool. "
                "The email to validate will be in `state['user_email_collected']`. "
                "Confirm the validation result.",
    description="Validates email format.",
    tools=[validate_email_tool]
)
# Note: This agent is instructed to look for state.
# The SequentialAgent makes the state from previous agents available.


def send_welcome_email(tool_context: ToolContext) -> str:
    """Simulates sending a welcome email if validation passed."""
    if tool_context.state.get("email_validated") and tool_context.state.get("user_name_collected"):
        name = tool_context.state["user_name_collected"]
        email = tool_context.state["user_email_collected"]
        # In a real app, this would use an email API
        return f"Welcome email conceptually sent to {name} at {email}."
    return "Could not send welcome email: email not validated or name missing."

send_email_tool = FunctionTool(func=send_welcome_email)

welcome_email_agent = Agent(
    name="welcome_emailer",
    model=DEFAULT_LLM,
    instruction="If the email has been validated (check `state['email_validated']`), "
                "use the 'send_welcome_email' tool to send a welcome message. "
                "Confirm the action.",
    description="Sends a welcome email.",
    tools=[send_email_tool]
)

# --- Define the SequentialAgent ---
user_onboarding_pipeline = SequentialAgent(
    name="user_onboarding_orchestrator",
    description="A pipeline to onboard new users: collect data, validate email, send welcome.",
    sub_agents=[
        data_collection_agent,
        email_validation_agent,
        welcome_email_agent
    ]
)

# --- Running the SequentialAgent ---
if __name__ == "__main__":
    runner = InMemoryRunner(agent=user_onboarding_pipeline, app_name="OnboardingApp")
    session_id = "s_onboard_seq"
    user_id = "new_user_seq"
    create_session(runner, user_id=user_id, session_id=session_id)

    # The initial message will be processed by data_collection_agent first.
    # Then email_validation_agent, then welcome_email_agent.
    initial_prompt = "Onboard user Alice with email alice@example.com"
    print(f"YOU: {initial_prompt}")
    user_message = Content(parts=[Part(text=initial_prompt)], role="user")  # User message to the agent

    async def main():
        final_agent_responses = []
        # The SequentialAgent itself doesn't produce direct textual output from an LLM.
        # It yields events from its sub_agents. We look for the final text from the *last* sub_agent.
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            print(f"  EVENT from [{event.author}]:")
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"    Text: {part.text.strip()}")
                        if event.author == welcome_email_agent.name and not event.get_function_calls() and not event.get_function_responses():
                            final_agent_responses.append(part.text.strip())
                    elif part.function_call:
                        print(f"    Tool Call: {part.function_call.name}({part.function_call.args})")
                    elif part.function_response:
                        print(f"    Tool Response for {part.function_response.name}: {part.function_response.response}")

        print("
--- Final Output from Pipeline (last agent's text response) ---")
        if final_agent_responses:
            print("SEQUENTIAL_PIPELINE: " + " ".join(final_agent_responses))
        else:
            print("SEQUENTIAL_PIPELINE: No final text output from the last agent.")

        # Inspect final state
        final_session = await runner.session_service.get_session(
            app_name="OnboardingApp", user_id=user_id, session_id=session_id
        )
        print(f"
Final session state: {final_session.state}")

    asyncio.run(main())
```

**Execution Flow:**

1. `user_onboarding_pipeline` (the `SequentialAgent`) receives the initial user message.
2. It first invokes `data_collection_agent.run_async()`. This agent will likely call its `gather_user_data` tool, and the state (`user_name_collected`, `user_email_collected`) will be updated.
3. Once `data_collection_agent` completes its turn (e.g., after its LLM confirms data collection), `user_onboarding_pipeline` invokes `email_validation_agent.run_async()`. This agent reads `state['user_email_collected']`, calls its `validate_email_format` tool, and updates `state['email_validated']`.
4. Finally, `welcome_email_agent.run_async()` is invoked. It checks `state['email_validated']` and calls `send_welcome_email`.
5. The events yielded by the `SequentialAgent` will be the collective events from all its sub-agents, in the order they were executed.

![*Diagram: Execution flow of a `SequentialAgent`.*](/assets/img/2025-08-22-implementing-orchestration-logic-with-shell-agents/figure-1.png)



> ## Best Practice: Design Sub-Agents for Clear Handoffs
> 
> When using SequentialAgent, ensure each sub-agent clearly defines its expected inputs (often from session state set by previous agents) and its outputs (again, often by updating session state for subsequent agents). Clear instructions within each LlmAgent are key to this.
> {: .prompt-info }


> ## SequentialAgent for Deterministic Workflows
> 
> SequentialAgent is excellent for workflows where the order of operations is fixed and known. It provides a simpler alternative to coding complex sequential logic within a single orchestrator LLM's instructions.
> {: .prompt-info }

## `ParallelAgent`: For Concurrent Task Execution or Ensemble Methods

The `google.adk.agents.ParallelAgent` executes all its `sub_agents` concurrently. It waits for all of them to complete and then effectively merges their outputs or makes them available. Each sub-agent in a `ParallelAgent` typically operates in an isolated manner regarding its direct LLM interactions for the current turn, though they all share the same initial `InvocationContext` (and thus the same session state and history up to that point).

**When to Use:**

- When you need multiple perspectives on the same problem (e.g., different agents trying different approaches).
- To perform independent sub-tasks simultaneously.
- For ensemble methods where results from multiple agents are later aggregated or voted upon.

**Defining a `ParallelAgent`:**

```python
from google.adk.agents import Agent, ParallelAgent # Import ParallelAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Define Sub-Agents for Parallel Execution ---
sentiment_analyzer_agent = Agent(
    name="sentiment_analyzer",
    model=DEFAULT_LLM,
    instruction="Analyze the sentiment of the provided text. Output only 'positive', 'negative', or 'neutral'.",
    description="Analyzes text sentiment."
    # output_key="sentiment_analysis_result" # LlmAgent can save its output to state
)
# For LlmAgent output_key to work as expected with ParallelAgent,
# each agent's output needs to be distinguishable.
# We'll set it in the orchestrator for this example.

keyword_extractor_agent = Agent(
    name="keyword_extractor",
    model=DEFAULT_LLM,
    instruction="Extract up to 3 main keywords from the provided text. Output as a comma-separated list.",
    description="Extracts keywords from text."
    # output_key="keyword_extraction_result"
)

# --- Define the ParallelAgent ---
text_analysis_parallel_tasks = ParallelAgent(
    name="parallel_text_analyzer",
    description="Performs sentiment analysis and keyword extraction in parallel.",
    sub_agents=[
        sentiment_analyzer_agent,
        keyword_extractor_agent
    ]
)

# --- Orchestrator to use the ParallelAgent and combine results ---
class AnalysisOrchestrator(Agent): # Custom agent to manage parallel results
    async def _run_async_impl(self, ctx):
        # First, run the parallel tasks. The ParallelAgent itself doesn't have an LLM.
        # It yields events from its children. We need to collect those.
        # For this demo, we'll assume the user's initial text is the input for parallel tasks.

        # Re-invoke the ParallelAgent. This is a bit manual for this example.
        # A more robust way would be to make ParallelAgent a tool for the orchestrator
        # or use a different flow.

        print("  Orchestrator: Starting parallel analysis...")
        all_parallel_events = []
        async for event in text_analysis_parallel_tasks.run_async(ctx):
            all_parallel_events.append(event)
            # Optionally yield these to show progress
            # yield event

        # Now, extract results from the state (assuming sub-agents saved them)
        # This requires sub-agents to be designed to save their distinct outputs to state.
        # For simplicity, we'll assume the LLM for the orchestrator will synthesize
        # from the combined history of the parallel agents.
        # The ParallelAgent sets a distinct `branch` for each sub-agent's events.

        # Create a summary prompt for the orchestrator's LLM
        history_summary = "Parallel analysis performed. Results from sub-agents:
"
        sentiment_result_text = "Sentiment: Not found."
        keywords_result_text = "Keywords: Not found."

        for event in all_parallel_events:
            if event.author == sentiment_analyzer_agent.name and event.content and event.content.parts[0].text:
                if not event.get_function_calls() and not event.get_function_responses(): # Final text
                    sentiment_result_text = f"Sentiment Analysis by '{event.author}' (branch '{event.branch}'): {event.content.parts[0].text.strip()}"
            elif event.author == keyword_extractor_agent.name and event.content and event.content.parts[0].text:
                 if not event.get_function_calls() and not event.get_function_responses(): # Final text
                    keywords_result_text = f"Keyword Extraction by '{event.author}' (branch '{event.branch}'): {event.content.parts[0].text.strip()}"

        combined_results_text = f"{sentiment_result_text}
{keywords_result_text}"
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name, # Orchestrator's name
            content=Content(parts=[Part(text=f"Combined analysis results:
{combined_results_text}")])
        )

analysis_orchestrator = AnalysisOrchestrator( # Using our custom LlmAgent-like class
    name="analysis_orchestrator",
    # instruction="You received analysis results. Summarize them for the user.",
    description="Orchestrates parallel text analysis and presents combined results.",
    sub_agents=[text_analysis_parallel_tasks] # Conceptually, it USES the parallel agent
)

# --- Running the ParallelAgent (via orchestrator) ---
if __name__ == "__main__":
    runner = InMemoryRunner(agent=analysis_orchestrator, app_name="ParallelAnalysisApp")
    session_id = "s_parallel_an"
    user_id = "parallel_user"
    create_session(runner, user_id=user_id, session_id=session_id)

    review_text = "This ADK framework is incredibly powerful and flexible, making agent development a breeze! Highly recommended."
    print(f"YOU: Analyze this text: {review_text}")
    user_message = Content(parts=[Part(text=review_text)], role="user")  # User message to the agent

    async def main():
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.author == analysis_orchestrator.name and event.content and event.content.parts:
                print(f"ORCHESTRATOR: {event.content.parts[0].text.strip()}")


    asyncio.run(main())
```

**Execution Flow:**

1. `analysis_orchestrator` is invoked.
2. Its `_run_async_impl` explicitly calls `text_analysis_parallel_tasks.run_async()`.
3. `ParallelAgent` (`text_analysis_parallel_tasks`) launches `sentiment_analyzer_agent.run_async()` and `keyword_extractor_agent.run_async()` concurrently (using `asyncio.gather` or similar internally).
4. Events from both sub-agents are yielded by the `ParallelAgent` as they become available. A crucial feature is that the `ParallelAgent` sets a unique `branch` attribute on the `Event` objects generated by each of its sub-agents (e.g., `event.branch = "parallel_text_analyzer.sentiment_analyzer"`). This allows a subsequent agent or logic to distinguish which sub-agent produced which event.
5. The `analysis_orchestrator` collects these events and then manually constructs a summary. A more sophisticated orchestrator might feed these branched events back into its own LLM to synthesize a combined result.

![*Diagram: Execution flow of a `ParallelAgent` managed by an orchestrator.*](/assets/img/2025-08-22-implementing-orchestration-logic-with-shell-agents/figure-2.png)



> ## `event.branch` for Disambiguating Parallel Outputs
> 
> When events are yielded from a ParallelAgent, the event.branch attribute is automatically set by ADK to indicate which sub-agent generated that event (e.g., "parent_agent_name.sub_agent_name"). This is crucial for any subsequent logic or agent that needs to process or combine the results from the parallel tasks correctly.
> {: .prompt-info }


> ## Resource Consumption with ParallelAgent
> 
> Running many complex sub-agents in parallel can consume significant resources (CPU, memory, LLM API quotas). Be mindful of the number and complexity of agents launched concurrently.
> {: .prompt-info }

## `LoopAgent`: For Iterative Tasks or Retries

The `google.adk.agents.LoopAgent` executes its list of `sub_agents` (often just one) repeatedly until a maximum number of iterations is reached or one of its sub-agents signals an "escalation" (exit condition).

**When to Use:**

- For tasks that require iterative refinement (e.g., an agent tries to generate something, another agent critiques it, and the first agent refines it based on critique, repeating several times).
- To implement retry mechanisms for fallible operations.
- For polling or waiting for a condition to be met.

**Defining a `LoopAgent` and the `exit_loop` Tool:**

The `LoopAgent` itself doesn't have an LLM. The decision to exit the loop is typically made by  calling the `google.adk.tools.exit_loop` tool.

```python
from google.adk.agents import Agent, LoopAgent # Import LoopAgent
from google.adk.tools import FunctionTool, ToolContext, exit_loop # Import exit_loop
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Define Sub-Agent for the Loop ---
# This agent will try to "refine" a draft.
# In a real scenario, it might get feedback from another agent or the user.

def check_draft_quality(draft: str, tool_context: ToolContext) -> dict:
    """Simulates checking draft quality. Exits loop if 'final' is in draft."""
    iteration = tool_context.state.get("loop_iteration", 0) + 1
    tool_context.state["loop_iteration"] = iteration
    tool_context.state["current_draft"] = draft  # Update the current draft in state

    if "final" in draft.lower() or iteration >= 3: # Exit condition
        print(f"    [QualityCheckTool] Draft meets criteria or max iterations ({iteration}) reached. Signaling exit.")
        exit_loop(tool_context) # Call the exit_loop tool
        return {"quality": "good", "feedback": "Looks good!" if "final" in draft.lower() else "Max iterations reached.", "action": "exit"}
    else:
        print(f"    [QualityCheckTool] Draft needs more work (iteration {iteration}).")
        return {"quality": "poor", "feedback": "Needs more detail about ADK benefits.", "action": "refine"}

quality_check_tool = FunctionTool(func=check_draft_quality)

# The sub-agent that does the work and decides to exit
drafting_agent = Agent(
    name="draft_refiner",
    model=DEFAULT_LLM,
    instruction="You are a document drafter. You will be given a topic and previous draft (if any, from `state['current_draft']`). "
                "Generate or refine the draft. Then, use the 'check_draft_quality' tool to assess it. "
                "The tool will provide feedback. If the tool signals to exit, your job is done for this iteration. "
                "If it signals to refine, use the feedback to improve the draft in your next thought process.",
    description="Drafts and refines documents iteratively.",
    tools=[quality_check_tool]
)

# --- Define the LoopAgent ---
iterative_refinement_loop = LoopAgent(
    name="document_refinement_loop",
    description="Iteratively refines a document until quality criteria are met.",
    sub_agents=[drafting_agent],
    max_iterations=5 # A safeguard, the tool will exit sooner
)

# --- Running the LoopAgent ---
if __name__ == "__main__":
    runner = InMemoryRunner(agent=iterative_refinement_loop, app_name="LoopRefineApp")
    session_id = "s_loop_refine"
    user_id = "loop_user"

    initial_prompt = "Draft a short paragraph about the benefits of ADK. Initial draft: 'ADK is a toolkit.'"
    print(f"YOU: {initial_prompt}")
    user_message = Content(parts=[Part(text=initial_prompt)], role="user")  # User message to the agent

    # Initialize state for the first draft
    initial_state = {"current_draft": "ADK is a toolkit."}
    create_session(runner, user_id=user_id, session_id=session_id, state=initial_state.copy())

    async def main():
        final_agent_responses = []
        # LoopAgent yields events from its sub_agents.
        # The `escalate` action from `exit_loop` will stop the LoopAgent.
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
            # For LoopAgent, it's often useful to pass initial state for the loop
            # We can modify the runner to set this, or have an outer orchestrator.
            # For simplicity, let's assume drafting_agent is smart enough or has it in context.
            # A better way for LoopAgent might be an initial context-setting agent before the loop.
            # For this example, we'll have the drafting_agent look for 'current_draft' in state.
        ):
            current_session = await runner.session_service.get_session(
                app_name="LoopRefineApp", user_id=user_id, session_id=session_id
            )
            print(f"  EVENT from [{event.author}] (Loop Iteration {current_session.state.get('loop_iteration', 0)}):") # HACK: Peeking into state
            if event.actions and event.actions.escalate:
                print("    ESCALATE signal received. Loop will terminate.")

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        print(f"    Tool Call: {part.function_call.name}({part.function_call.args})")
                    elif part.function_response:
                        print(f"    Tool Response for {part.function_response.name}: {part.function_response.response}")

        print("
--- Final Output from Loop (last substantive text from sub-agent) ---")

        print(f"
Final session state: {current_session.state}")

    asyncio.run(main())
```

**Execution Flow:**

1. `iterative_refinement_loop` (the `LoopAgent`) starts.
2. It invokes `drafting_agent.run_async()`.
3. `drafting_agent` generates/refines a draft (possibly using `state['current_draft']`) and then calls `check_draft_quality(draft=..., tool_context=...)`.
4. `check_draft_quality` logic runs:
    - If exit condition met: it calls `exit_loop(tool_context)`. 

    This sets `tool_context.actions.escalate = True`.
    
    - The `Event` yielded by `drafting_agent` (containing the `tool_response` from `check_draft_quality`) will have this `escalate` flag.
5. `LoopAgent` inspects the event. If `event.actions.escalate` is `True`, the loop terminates.
6. If `event.actions.escalate` is `False` (or not set) and `max_iterations` (if set on `LoopAgent`) is not reached, the `LoopAgent` goes back to step 2, re-invoking `drafting_agent`. `drafting_agent` would then use the feedback from `check_draft_quality` (which it received as a tool response in its previous turn) to improve the draft.

![*Diagram: Execution flow of a `LoopAgent` with a sub-agent that can signal loop termination.*](/assets/img/2025-08-22-implementing-orchestration-logic-with-shell-agents/figure-3.png)



> ## Best Practice: Clear Exit Conditions for Loops
> 
> Ensure the sub-agent(s) within a LoopAgent have clear logic and instructions for when to call exit_loop. Also, always set a reasonable max_iterations on the LoopAgent itself as a safeguard against infinite loops, especially during development.
> {: .prompt-info }


> ## Iterative Refinement and Polling
> 
> LoopAgent combined with stateful sub-agents is powerful for tasks involving iterative refinement (like the drafting example) or for scenarios where an agent needs to poll an external system until a certain status is achieved.
> {: .prompt-info }

**What's Next?**

We've now explored ADK's shell agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`), which provide structured ways to implement common multi-agent orchestration patterns without needing to encode all the control flow logic within a single orchestrator `LlmAgent`'s instructions. These tools, combined with the agent transfer mechanisms discussed previously, give you a rich toolkit for building complex, collaborative multi-agent systems.

Next we'll touch upon even more advanced multi-agent concepts like the Agent-to-Agent (A2A) protocol and the ability to use LangGraph inside ADK using `LangGraphAgent`, for scenarios requiring highly complex, stateful, and potentially distributed agent interactions.
