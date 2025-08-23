---
title: Chapter 12 - Designing Multi-Agent Architectures 
date: "2025-08-22 13:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Up to this point, our focus has been primarily on building and empowering individual `LlmAgent` instances. While single, highly capable agents can achieve a lot, many complex real-world problems benefit from a **Multi-Agent System (MAS)** approach. A MAS involves multiple autonomous agents interacting with each other, or an orchestrator, to achieve a common goal or solve a problem that might be too large or diverse for a single agent.

ADK is designed with multi-agent systems in mind, providing mechanisms for agents to delegate tasks and coordinate their efforts. This chapter explores the principles of designing such systems, how ADK facilitates agent communication through transfers, and common architectural patterns.

## Principles of Multi-Agent Systems

Why use multiple agents instead of one super-agent?

- **Modularity and Specialization:** Each agent can be designed to be an expert in a specific domain or task (e.g., a research agent, a data analysis agent, a user communication agent). This makes individual agents simpler to develop, test, and maintain.
- **Scalability:** Different agents can potentially run independently, allowing for better resource utilization and scalability.
- **Robustness:** If one specialized agent encounters an issue, other parts of the system might still be able to function or adapt.
- **Complexity Management:** Breaking down a large, complex problem into sub-problems that individual agents can tackle makes the overall system design more manageable.
- **Reusability:** Specialized agents can be reused across different multi-agent applications.

**Key Design Considerations for MAS:**

- **Agent Roles and Responsibilities:** Clearly define what each agent is responsible for.
- **Communication Protocol:** How will agents exchange information and delegate tasks? In ADK, this is primarily through "agent transfer."
- **Coordination Strategy:** How will the overall workflow be managed? Will there be a central orchestrator, a pipeline, or a more decentralized approach?
- **Knowledge Sharing:** How will agents share context or results? (e.g., via session state, artifacts, or direct message passing if applicable).
- **Conflict Resolution:** (More advanced) If agents have conflicting goals or information, how are these resolved?

## Defining Parent Agents and Sub-Agents in ADK

ADK implements hierarchical multi-agent systems by allowing an `LlmAgent` to have `sub_agents`.

```python
from google.adk.agents import Agent # LlmAgent

# Define specialized sub-agents
research_sub_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="You are a research specialist. Given a topic, find relevant information using available search tools.",
    description="Finds information on a given topic.",
    # tools=[google_search] 
)

writer_sub_agent = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="You are a skilled writer. Given information, synthesize it into a coherent summary or report.",
    description="Writes summaries or reports based on provided information."
)

# Define the parent/orchestrator agent
report_orchestrator_agent = Agent(
    name="report_orchestrator",
    model="gemini-2.5-flash-preview-05-20", # A more capable model for orchestration
    instruction="You are an orchestrator for creating research reports. "
                "First, use the 'researcher' agent to gather information on the user's topic. "
                "Then, pass the researcher's findings to the 'writer' agent to create a final report. "
                "Manage the flow and present the final report to the user.",
    description="Orchestrates research and writing of reports.",
    sub_agents=[research_sub_agent, writer_sub_agent] # Key: Assigning sub-agents
)

# The root_agent for the runner would be report_orchestrator_agent
# from google.adk.runners import InMemoryRunner
# if __name__ == "__main__":
#     runner = InMemoryRunner(agent=report_orchestrator_agent, app_name="ReportMAS")
    # ... runner logic ...

```

**Key Points:**

- An `LlmAgent` can be assigned a list of other `BaseAgent` (including `LlmAgent`) instances to its `sub_agents` parameter.
- This creates a parent-child relationship. `report_orchestrator_agent` is the parent of `research_sub_agent` and `writer_sub_agent`.
- The parent agent can then be instructed to delegate tasks to its sub-agents.
- Each sub-agent can have its own model, instructions, tools, and even its own sub-agents, allowing for nested hierarchies.


> ## Description is Key for Delegation
> 
> When a parent agent (or its LLM) decides whether to delegate a task to a sub-agent, it heavily relies on the description of the sub-agents. Write clear, concise, and accurate descriptions that highlight each sub-agent's unique capabilities and when it should be invoked.
> {: .prompt-info }

## Agent Communication: The Role of Agent Transfer

In ADK, the primary mechanism for "communication" or delegation between `LlmAgent` instances in a hierarchy is **Agent Transfer**. This is not direct message passing in the traditional sense but rather a transfer of control.

**The `transfer_to_agent` Tool:**

- ADK provides an internal tool conceptually named `transfer_to_agent`.
- When an `LlmAgent` is configured with sub-agents (or can transfer to its parent), its LLM Flow (typically `AutoFlow`) makes this `transfer_to_agent` tool implicitly available to the LLM.
- The `transfer_to_agent` tool takes one main argument: `agent_name` (the name of the target agent).
- When the orchestrating LLM decides to delegate a task, it generates a function call to `transfer_to_agent(agent_name="target_agent_name")`.

**How `AutoFlow` Facilitates Transfers:**

- The `google.adk.flows.llm_flows.auto_flow.AutoFlow` is automatically used by `LlmAgent`s that are part of a potential transfer chain (e.g., they have sub-agents, or `disallow_transfer_to_parent` is `False`).
- `AutoFlow` includes an `agent_transfer.request_processor`. This processor:
    1. Identifies potential target agents (sub-agents, parent, peers based on agent config).
    2. Appends instructions to the `LlmRequest` informing the LLM about these available agents and their descriptions.
    3. Ensures the `transfer_to_agent` tool declaration is included in the `LlmRequest`.
- When the orchestrating LLM calls `transfer_to_agent`, the `AutoFlow` intercepts this:
    1. It yields an `Event` indicating the transfer request.
    2. The `Runner` then identifies the target agent.
    3. The `Runner` invokes the `run_async` method of the target agent, passing the current `InvocationContext`. The user's original message (or a refined message from the orchestrator) becomes the input for the target agent.
    4. The conversation effectively continues with the target agent now active.

**Example of LLM "Deciding" to Transfer (Conceptual LLM Output):**

Imagine `report_orchestrator_agent` receives the prompt "Research the impact of AI on healthcare."
Its LLM might internally reason and then output a function call:

```
/* LLM output for report_orchestrator_agent */
I need to gather information first. The 'researcher' agent is best for this.
<tool_code>
transfer_to_agent(agent_name="researcher", user_query="Impact of AI on healthcare")
</tool_code>

```

*(Note: The `user_query` or similar parameter to pass context during transfer is a conceptual addition that ADK handles by ensuring the `InvocationContext` carries the necessary user input forward.)*

ADK processes this:

1. The `report_orchestrator_agent`'s turn effectively ends with this transfer request.
2. The `Runner` now invokes `research_sub_agent.run_async(...)` with the context (including the user query).
3. `research_sub_agent` now runs its own LLM Flow, uses its tools (e.g., search), and generates its response.

**Returning from a Sub-Agent:**
How does control return to the parent/orchestrator?

- **Implicitly via `SingleFlow` in Sub-Agent:** If the sub-agent uses `SingleFlow` (i.e., it's not designed to further delegate deeply), once it produces a final textual response, its `run_async` completes. The `Runner`, which was awaiting this, then typically looks for the next agent to run. If the previous active agent was an orchestrator that can continue, the `Runner` might reactivate it.
- **Explicit Transfer Back:** The sub-agent could also be instructed to call `transfer_to_agent(agent_name="report_orchestrator")` when its task is done.
- **Orchestrator's Plan:** The orchestrator's initial instructions might explicitly state the sequence: "1. Call researcher. 2. *Then, I (orchestrator) will take the output and call writer*." In this case, after the researcher finishes, the orchestrator's LLM is prompted again with the researcher's output in the history.

![*Diagram: Agent transfer sequence from an Orchestrator to a Researcher sub-agent.*](/assets/img/2025-08-22-designing-multi-agent-architectures/figure-1.png)



> ## Transfer Loops and Deadlocks
> 
> Poorly designed instructions or agent descriptions can lead to agents endlessly transferring tasks back and forth or getting stuck. Ensure:
> 
> - Each agent has a clear, distinct responsibility.
> - Transfer conditions are well-defined in the orchestrator's instructions.
> - Sub-agents have a clear way to signal task completion (either by providing a final answer or by explicitly transferring back if designed to do so).
> - ADK's `max_llm_calls` in `RunConfig` can act as a failsafe against runaway loops.
> {: .prompt-info }

## Common Multi-Agent Patterns

ADK's flexible agent definition and transfer mechanism support various common MAS patterns:

**1. Hierarchical (Coordinator-Worker / Orchestrator-Specialist):**

- **Structure:** A parent "coordinator" or "orchestrator" agent breaks down a complex task and delegates sub-tasks to specialized "worker" or "specialist" sub-agents.
- **Example:** The `report_orchestrator_agent` (coordinator) using `research_sub_agent` and `writer_sub_agent` (workers).
- **ADK Implementation:** Parent `LlmAgent` with `sub_agents` list. Parent's instructions focus on delegation logic. Sub-agents focus on their specific tasks.

**2. Sequential (Pipeline):**

- **Structure:** Data or a task flows through a sequence of agents, each performing a specific transformation or step. The output of one agent becomes the input for the next.
- **Example:**
    1. `DataIngestionAgent`: Fetches raw data.
    2. `DataCleaningAgent`: Cleans and preprocesses the data.
    3. `AnalysisAgent`: Performs analysis on the cleaned data.
    4. `ReportGenerationAgent`: Formats the analysis into a report.
- **ADK Implementation:**
    - Can be implemented with a master orchestrator agent that calls sub-agents in sequence.
    - Alternatively, ADK provides `google.adk.agents.SequentialAgent` (a `BaseAgent` subclass, not an `LlmAgent`) which explicitly runs its `sub_agents` one after the other.

![*Diagram: Sequential/Pipeline multi-agent pattern.*](/assets/img/2025-08-22-designing-multi-agent-architectures/figure-2.png)


**3. Parallel (Ensemble / Competing Experts):**

- **Structure:** Multiple agents work on the same task or sub-problem concurrently. Their results might be:
    - Combined or synthesized by another agent.
    - Voted upon.
    - The "best" result selected based on some criteria.
- **Example:** Three different `SummarizationAgent`s using slightly different instructions or models process the same document. An `EvaluationAgent` then picks the best summary.
- **ADK Implementation:**
    - An orchestrator can be programmed to invoke multiple sub-agents conceptually in parallel (though true parallelism depends on how `asyncio` schedules their `run_async` calls).
    - ADK provides `google.adk.agents.ParallelAgent` (a `BaseAgent` subclass) for explicitly running sub-agents in parallel and gathering their distinct outputs.

![*Diagram: Parallel/Ensemble multi-agent pattern.*](/assets/img/2025-08-22-designing-multi-agent-architectures/figure-3.png)



> ## Best Practice: Start with Simple Patterns
> 
> When designing your first MAS, start with simpler patterns like a clear hierarchy or a short pipeline. As you gain experience, you can explore more complex coordination strategies. Clearly defining each agent's API (its description and how it expects input/provides output) is crucial.
> {: .prompt-info }

## Case Study: Designing a Research Assistant MAS

Let's outline the design for a multi-agent research assistant that takes a user's topic, researches it, and produces a structured summary.

**Agents:**

1. **`QueryUnderstandingAgent` (Sub-Agent)**
    - **Model:** `gemini-2.0-flash`
    - **Description:** "Refines and clarifies the user's research topic, breaking it down into specific searchable questions if necessary."
    - **Instruction:** "Analyze the user's research topic. If it's broad, decompose it into 2-3 specific questions. If it's already specific, confirm understanding. Output the specific question(s)."
    - **Output:** JSON object with a list of questions.
2. **`WebSearchAgent` (Sub-Agent)**
    - **Model:** `gemini-2.0-flash` (good for search integration)
    - **Description:** "Given specific research questions, uses Google Search to find relevant web pages and extracts key information snippets from them."
    - **Instruction:** "For each input question, perform targeted Google searches. From the top 3-5 results for each, extract concise, relevant snippets of information. If a URL seems highly relevant, note it down for potential deeper content extraction. Compile all snippets per question."
    - **Tools:** `google_search`
    - **Output:** JSON object mapping original questions to lists of information snippets and potentially promising URLs.
3. **`InformationSynthesizerAgent` (Sub-Agent)**
    - **Model:** `gemini-2.5-flash-preview-05-20` (good for summarization and writing)
    - **Description:** "Takes a collection of research snippets and URLs, synthesizes the information, and writes a structured summary or report."
    - **Instruction:** "You will receive research findings (text snippets, possibly full page content if `load_web_page` was used by a prior agent). Synthesize this information into a coherent, structured summary answering the original user topic. Ensure proper attribution if sources are clear. Structure with headings and bullet points."
    - **(Optional Tool):** If `WebSearchAgent` only provides URLs, this agent might use `FunctionTool(load_web_page)`.
    - **Output:** The final textual report.
4. **`MainResearchOrchestrator` (Root Agent)**
    - **Model:** `gemini-2.0-flash` (for good orchestration)
    - **Description:** "Manages the end-to-end research process for a user's topic."
    - **Instruction:**
        
    You are the main orchestrator for a research task.
    1. Receive the user's research topic.
    2. Delegate to 'QueryUnderstandingAgent' to refine/clarify the topic into specific questions.
    3. Take the questions from 'QueryUnderstandingAgent' and delegate to 'WebSearchAgent' to gather information.
    4. Take the findings from 'WebSearchAgent' and delegate to 'InformationSynthesizerAgent' to compile the final report.
    5. Present the final report to the user. Manage the flow and data transfer between agents.
    
    - **Sub-Agents:** `[QueryUnderstandingAgent, WebSearchAgent, InformationSynthesizerAgent]`

This case study illustrates how breaking down a complex task (research) into specialized agent roles can lead to a more structured and potentially more effective system. The orchestrator manages the high-level flow, while sub-agents handle their expert tasks. Communication happens via the orchestrator transferring control and context (implicitly through session history/state, or explicitly by instructing agents to output data for the next agent).


> ## State Management for Inter-Agent Communication
> 
> In MAS like the research assistant, session state (tool_context.state or callback_context.state) becomes crucial for passing information between agents. For example:
> 
> - `QueryUnderstandingAgent` could save its output (list of questions) to `state['refined_questions']`.
> - `MainResearchOrchestrator`, in its next turn, reads `state['refined_questions']` and uses it as input for `WebSearchAgent`.
> 
> Use clear, agreed-upon state keys.
> {: .prompt-info }



**What's Next?**

Designing a multi-agent architecture is the first step. Next, we'll look at ADK's specific shell agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) that provide pre-built structures for common orchestration patterns, simplifying the implementation of some of the designs discussed here.
