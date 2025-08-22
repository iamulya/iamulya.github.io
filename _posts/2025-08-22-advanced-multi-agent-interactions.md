---
title: Chapter 14 - Advanced Multi-Agent Interactions: LangGraphAgent and A2A 
date: "2025-08-22 14:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

In the previous chapters, we've explored building multi-agent systems using ADK's hierarchical structures, agent transfers, and shell orchestrators like `SequentialAgent`, `ParallelAgent`, and `LoopAgent`. These are powerful for many common coordination patterns. However, for scenarios requiring highly complex, stateful, cyclical, or even distributed agent interactions, ADK provides avenues for integrating with more specialized concepts and frameworks.

This chapter offers a glimpse into two such advanced areas:

1. **`LangGraphAgent`**: Integrating ADK agents with LangGraph, a library for building stateful, multi-agent applications as cyclical graphs.
2. **Agent-to-Agent (A2A) Communication Protocol**: A brief overview of a standardized protocol for more direct, remote communication between agents, often in distributed settings.

It's important to note that a deep dive into LangGraph itself or the full A2A protocol specification is beyond the scope of this book. Instead, this chapter aims to show how ADK can interface with these concepts, enabling you to leverage their strengths within your ADK projects.

## `LangGraphAgent`: Building Stateful Agent Applications as Graphs

[LangGraph](https://langchain-ai.github.io/langgraph/) (from the creators of Langchain) is a library for building robust and stateful multi-agent applications by defining them as cyclical graphs. In LangGraph, agents (or more generally, any callable function or runnable) are represented as "nodes," and the "edges" define how state transitions from one node to another. This allows for complex loops, conditional branching, and human-in-the-loop interactions.

ADK provides the `google.adk.agents.LangGraphAgent` as an experimental bridge. This allows you to define a LangGraph `CompiledGraph` and wrap it as an ADK agent. The ADK `Runner` can then invoke this `LangGraphAgent`, passing it the current conversation history, and the LangGraph graph will execute its defined flow.

**Key Concepts of LangGraph relevant to ADK Integration:**

- **State Graph (`StatefulGraph`, `CompiledGraph`):** The core of a LangGraph application. You define a state schema (often a Pydantic model or a TypedDict) that is passed between nodes.
- **Nodes:** Python functions or Langchain Runnables that operate on the state and can modify it.
- **Edges:** Define the transitions between nodes, often conditionally based on the current state or the output of a node.
- **Checkpointer:** LangGraph can integrate with checkpointers (like `LangChainRedis` or custom ones) to persist the graph's state, allowing for resumable and long-running agent interactions.

**Using `LangGraphAgent` in ADK:**

1. **Define your LangGraph graph:** This involves setting up your state, nodes, and edges using LangGraph's API.
2. **Compile the graph:** Obtain a `CompiledGraph` instance.
3. **Wrap it with `LangGraphAgent`:** Instantiate `LangGraphAgent`, providing the compiled graph and an optional ADK instruction.
4. **Use it like any other ADK agent:** Add it to an orchestrator's `sub_agents` list or use it as a root agent.

```python
# This is a conceptual example. A full, runnable LangGraph example
# can be quite involved and depends on specific LangGraph setup.
# Ensure 'langgraph' and 'langchain_core' are installed: pip install langgraph langchain_core

from google.adk.agents import Agent as AdkAgent
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Conceptual LangGraph Setup (Illustrative) ---
try:
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver # Simple in-memory checkpointer

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
        # Add other state variables your graph needs

    # Define a clear flag for tool requests within AIMessage content
    TOOL_REQUEST_FLAG = "[TOOL_REQUEST_FLAG]"

    def llm_node(state: AgentState):
        # Check the type of the last message to decide behavior
        last_message = state['messages'][-1]
        response_content = ""

        if isinstance(last_message, HumanMessage):
            # User's turn
            user_input = last_message.content
            if "tool" in user_input.lower(): # User mentions "tool"
                # LLM decides to call a tool
                response_content = f"Okay, I understand you want to use a tool for '{user_input}'. Requesting the tool. {TOOL_REQUEST_FLAG}"
            else:
                response_content = f"LangGraph responding to user: '{user_input}'"
        elif isinstance(last_message, AIMessage): # This AIMessage is from tool_node (or a previous llm_node turn)
            # LLM processes the tool's output or continues conversation
            # Ensure this response does NOT contain TOOL_REQUEST_FLAG unless a new tool is needed
            ai_input_content = last_message.content
            response_content = f"Processed previous step. Last AI message was: '{ai_input_content}'. Continuing..."
        else: # Should not happen in this simple graph (e.g. SystemMessage is last)
            response_content = "LangGraph: Unsure how to proceed with the current message state."
        
        print(f"DEBUG: llm_node producing: '{response_content}'")
        return {"messages": [AIMessage(content=response_content)]}

    def tool_node(state: AgentState):
        # This node would execute a tool based on LLM's request
        # The AIMessage from tool_node should reflect the tool's output
        tool_output_content = "LangGraph tool_node executed conceptually. Result: Tool_ABC_Data."
        print(f"DEBUG: tool_node producing: '{tool_output_content}'")
        return {"messages": [AIMessage(content=tool_output_content)]}

    def should_call_tool(state: AgentState):
        # Check the last AI message for the specific flag
        if state['messages'] and isinstance(state['messages'][-1], AIMessage):
            last_ai_content = state['messages'][-1].content
            print(f"DEBUG: should_call_tool checking AI content: '{last_ai_content}'")
            if TOOL_REQUEST_FLAG in last_ai_content:
                print("DEBUG: should_call_tool -> routing to tool_executor")
                return "tool_executor"
        print("DEBUG: should_call_tool -> routing to END")
        return END

    # Create a conceptual graph
    builder = StateGraph(AgentState)
    builder.add_node("llm_entry_point", llm_node)
    builder.add_node("tool_executor", tool_node)

    builder.set_entry_point("llm_entry_point")
    builder.add_conditional_edges(
        "llm_entry_point",
        should_call_tool,
        {"tool_executor": "tool_executor", END: END}
    )
    builder.add_edge("tool_executor", "llm_entry_point") # Loop back after tool use

    memory = MemorySaver()
    runnable_graph = builder.compile(checkpointer=memory)
    LANGGRAPH_SETUP_SUCCESS = True
    print("Conceptual LangGraph graph compiled.")

except ImportError:
    print("LangGraph or LangChain components not found. pip install langgraph langchain_core")
    LANGGRAPH_SETUP_SUCCESS = False
    runnable_graph = None
except Exception as e:
    print(f"Error during LangGraph setup: {e}")
    LANGGRAPH_SETUP_SUCCESS = False
    runnable_graph = None

# --- ADK Agent Definition ---
langgraph_adk_agent = None
orchestrator = None
if LANGGRAPH_SETUP_SUCCESS and runnable_graph:
    langgraph_adk_agent = LangGraphAgent(
        name="my_langgraph_powered_agent",
        graph=runnable_graph, # Pass the compiled LangGraph graph
        instruction="This agent is powered by a LangGraph state machine. Interact normally." # ADK-level instruction
    )
    print("LangGraphAgent initialized.")

    orchestrator = AdkAgent(
            name="main_orchestrator",
            model=DEFAULT_LLM, # Orchestrator's own model
            instruction="Delegate all tasks to my_langgraph_powered_agent.",
            sub_agents=[langgraph_adk_agent]
        )

# --- Running the ADK LangGraphAgent ---
if __name__ == "__main__":
    runner = InMemoryRunner(agent=orchestrator, app_name="LangGraphADKApp")
    session_id = "s_langgraph_adk" # ADK session ID for the orchestrator
    user_id = "lg_user"
    create_session(runner, user_id=user_id, session_id=session_id)

    prompts = [
        "Hello LangGraph agent, tell me about yourself.",
        "use the tool.", # To trigger the tool_node path
        "What happened after the tool use?"
    ]

    async def main():
        for i, prompt_text in enumerate(prompts):
            print(f"
--- Turn {i+1} ---")
            print(f"YOU: {prompt_text}")
            user_message_adk = Content(parts=[Part(text=prompt_text)], role="user")

            print("ORCHESTRATOR/LANGGRAPH_AGENT: ", end="", flush=True)
            # The orchestrator will likely decide to transfer to langgraph_adk_agent
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id, 
                new_message=user_message_adk
            ):
                for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print()

    asyncio.run(main())
```

**Execution Flow with `LangGraphAgent`:**

1. An ADK `Runner` invokes an orchestrator agent, or directly the `LangGraphAgent`.
2. If an orchestrator is used, it transfers control to the `LangGraphAgent` via `transfer_to_agent`.
3. The `LangGraphAgent._run_async_impl` method is called. It converts the ADK `Session.events` (conversation history) into a format LangGraph expects (typically a list of `langchain_core.messages.BaseMessage`).
4. It invokes the `langgraph_compiled_graph.invoke()` method with this history and a `RunnableConfig` that includes a `thread_id` (derived from ADK's `session_id`) for stateful, multi-turn conversations managed by the LangGraph checkpointer.
5. The LangGraph graph executes its defined flow of nodes and edges.
6. The final state of the LangGraph execution (usually the last message(s) in its state) is then converted back into an ADK `Event` (with text content) and yielded.

![*Diagram: Conceptual flow of ADK interacting with a `LangGraphAgent`*](/assets/img/2025-08-22-advanced-multi-agent-interactions/figure-1.png)



> ## Stateful and Cyclical Logic with LangGraph
> 
> LangGraphAgent shines when you need to model agent interactions that are not strictly hierarchical or sequential but involve cycles, complex conditional logic, or require robust state persistence across many turns. LangGraph's checkpointer mechanism is particularly useful for long-running, resumable agent processes.
> {: .prompt-info }


> ## Experimental Integration and Complexity
> 
> - The `LangGraphAgent` integration is marked as somewhat experimental in ADK. Its API or behavior might evolve.
> - Building and debugging LangGraph applications themselves can be complex. You'll need a good understanding of LangGraph's concepts (state, nodes, edges, checkpointers etc.).
> - Ensure that the state schema used in your LangGraph graph and the message formats are compatible with how `LangGraphAgent` converts ADK history and expects output.
> {: .prompt-info }


> ## Best Practice: Use LangGraph for "Inner Loop" Complexity
> 
> Consider using LangGraphAgent to encapsulate a particularly complex part of your overall agent system. An ADK orchestrator agent could delegate to a LangGraphAgent for a sub-task that benefits from LangGraph's cyclical graph capabilities, while the broader multi-agent system is still managed using ADK's hierarchical patterns.
> {: .prompt-info }

## Agent-to-Agent (A2A) Communication Protocol

Previously we introduced the concept of multi-agent systems where ADK agents can transfer control to one another within the same application. However, the vision for AI agent collaboration extends far beyond a single process or framework. The **Agent-to-Agent (A2A) Communication Protocol** is an open standard, initiated by Google, designed to enable seamless communication and interoperability between independent AI agent systems, even if they are built on different frameworks, by different companies, and running on separate servers.

This section provides a detailed overview of the A2A protocol, its core concepts, and how it facilitates inter-agent collaboration. While ADK doesn't provide a full A2A client or server implementation out-of-the-box for *remote* communication, understanding A2A is crucial if you plan to:

- Have your ADK agents interact with external agents that are A2A-compliant.
- Expose your ADK agent system as an A2A-compliant service for other agents to consume.


> ## A2A is an Evolving Specification
> 
> The A2A protocol is still under development and evolving and this chapter depicts its status in May / June 2025. Its adoption and tooling are growing but may not be as mature as some intra-framework communication methods. Implementing a full A2A server around your ADK agent requires careful consideration of the A2A spec and robust error handling, serialization, and network communication logic.
> {: .prompt-info }

### Why A2A? The Need for Agent Interoperability

As AI agents become more specialized and powerful, the ability for them to collaborate is essential for tackling complex, multi-faceted problems. A2A aims to:

- **Break Down Silos:** Connect agents across diverse ecosystems and technology stacks.
- **Enable Complex Collaboration:** Allow specialized agents to delegate tasks, share context, and work together on goals that a single agent cannot achieve alone.
- **Promote Open Standards:** Foster a community-driven approach to agent communication, encouraging innovation and broad adoption.
- **Preserve Opacity:** Allow agents to collaborate based on declared capabilities without needing to expose their internal state, proprietary logic, or specific tool implementations. This enhances security and protects intellectual property.

A2A is about agents communicating *as peers*, managing tasks, and exchanging rich information, distinct from how an agent might use a more traditional, stateless tool via a simple API call.

### Core A2A Concepts and Terminology

The A2A protocol is built around several key concepts, detailed in its [official specification](https://google-a2a.github.io/A2A/specification/):

- **A2A Client & A2A Server (Remote Agent):**
    - An **A2A Client** (which could be another agent or an application) initiates requests.
    - An **A2A Server** (the "Remote Agent") exposes an A2A-compliant HTTP endpoint to process these requests.
- **Agent Card (`AgentCard`):**
    - A JSON metadata document, typically published by an A2A Server at a well-known URI (e.g., `/.well-known/agent.json`).
    - It's the agent's "digital business card," describing its:
        - `name`, `description`, `provider`, `version`.
        - `url`: The base HTTP(S) endpoint for its A2A service.
        - `capabilities`: Supported A2A features (e.g., `streaming`, `pushNotifications`).
        - `securitySchemes` & `security`: How clients should authenticate (e.g., OAuth2, API Key).
        - `defaultInputModes` & `defaultOutputModes`: Default [Media Types](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types) (e.g., "text/plain", "application/json", "image/png") it accepts and produces.
        - `skills`: An array of `AgentSkill` objects, detailing specific capabilities (ID, name, description, tags, examples, skill-specific I/O modes).
        - `supportsAuthenticatedExtendedCard`: A boolean indicating if a more detailed card is available after authentication.
- **Task (`Task`):**
    - The fundamental unit of work. When a client sends a message (e.g., a user request), the A2A Server typically creates a `Task` to track its processing.
    - Each task has a unique `id` (server-generated) and a `contextId` (server-generated, to group related tasks in a conversation).
    - It maintains a `status` (`TaskStatus` object) which includes the current `TaskState`.
- **Task State (`TaskState` Enum):**
    - Defines the lifecycle of a task: `submitted`, `working`, `input-required` (agent needs more info from client), `auth-required` (agent needs secondary auth from user), `completed`, `failed`, `canceled`, `rejected`, `unknown`.
- **Message (`Message`):**
    - A communication turn between the client and server, with a `role` ("user" or "agent").
    - Contains `parts` (the actual content) and various IDs (`messageId`, `taskId`, `contextId`, `referenceTaskIds`).
- **Part (`Part` Union Type):**
    - The smallest unit of content within a `Message` or `Artifact`. Can be:
        - `TextPart`: For textual content (`text`, `mimeType`).
        - `FilePart`: For files, referenced either by `bytes` (base64 encoded) or a `uri`, along with `name` and `mimeType`.
        - `DataPart`: For structured JSON data (`data` object, `mimeType`).
- **Artifact (`Artifact`):**
    - A tangible output generated by the agent for a task (e.g., a report document, a generated image, structured data results).
    - Has an `artifactId`, `name`, `description`, and contains `parts`.
- **Transport and Format:**
    - Communication **MUST** be over HTTP(S).
    - Payloads use **JSON-RPC 2.0**.
    - `Content-Type` for JSON-RPC is `application/json`.

![*Diagram: Interactions in A2A*](/assets/img/2025-08-22-advanced-multi-agent-interactions/figure-2.png)




> ## Agent Cards for Capability Discovery
> 
> The AgentCard is central to A2A. It allows an A2A Client to dynamically understand what a remote agent can do (skills), what kind of data it expects (inputModes), what it produces (outputModes), and how to securely connect to it (url, securitySchemes). Well-defined Agent Cards are crucial for effective agent discovery and interoperability.
> {: .prompt-info }

### Key A2A RPC Methods (High-Level)

The A2A specification defines several JSON-RPC methods. Here are the main ones:

- **`message/send`**:
    - Client sends a `Message` (e.g., user prompt) to initiate or continue a task.
    - Server typically responds with a `Task` object representing the work, or directly with a `Message` if the interaction is very short and stateless.
    - Suitable for synchronous interactions or when client-side polling for task status is acceptable.
- **`message/stream`**:
    - Client sends a `Message` and subscribes to real-time updates for the associated task via Server-Sent Events (SSE).
    - The server must have `capabilities.streaming: true` in its Agent Card.
    - The SSE stream delivers `SendStreamingMessageResponse` events containing `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`, or new `Message` objects from the agent. The final event in a sequence is marked `final: true`.
    - **Example Events in Stream:** Task submitted -> Task working -> Artifact chunk 1 -> Artifact chunk 2 (lastChunk) -> Task completed (final).
- **`tasks/get`**:
    - Client retrieves the current state of a specific `Task` using its `id`.
    - Response is a `Task` object.
- **`tasks/cancel`**:
    - Client requests cancellation of an ongoing `Task`.
    - Response is the updated `Task` object (hopefully with state `canceled`).
- **`tasks/pushNotificationConfig/set` & `tasks/pushNotificationConfig/get`**:
    - Manages configurations for push notifications. The client provides a webhook URL (`PushNotificationConfig`) where the A2A Server can POST updates for a long-running task.
    - The server must have `capabilities.pushNotifications: true`.
- **`tasks/resubscribe`**:
    - Allows a client to reconnect to an SSE stream for an ongoing task if the previous connection was interrupted.
- **`agent/authenticatedExtendedCard` (HTTP GET, not JSON-RPC):**
    - If `AgentCard.supportsAuthenticatedExtendedCard` is true, this endpoint (e.g., `{AgentCard.url}/../agent/authenticatedExtendedCard`) provides a potentially more detailed Agent Card after client authentication.

### Interaction Patterns

A2A supports several interaction patterns:

1. **Synchronous Request/Response (using `message/send`):**
    - Client sends request.
    - Server processes quickly and returns final `Task` (in terminal state) or `Message`.
    - Simple, but not suitable for long-running tasks.
2. **Asynchronous Polling (using `message/send` then `tasks/get`):**
    - Client sends request with `message/send`.
    - Server responds quickly with a `Task` in `submitted` or `working` state.
    - Client periodically calls `tasks/get` with the task ID to check for status updates until a terminal state is reached.
3. **Streaming with Server-Sent Events (using `message/stream`):**
    - Client sends request with `message/stream`.
    - Server keeps HTTP connection open and pushes updates (status, messages, artifact chunks) as SSE events.
    - Provides real-time feedback.
    
    ```{mermaid}
%%| fig-width: 50%
%%| fig-cap: "*Diagram: Simplified A2A Streaming with SSE.*"
    sequenceDiagram
        participant Client
        participant Server
    
        Client->>Server: message/stream (Request_ID_123, Message: "Generate report")
        Server-->>Client: HTTP 200 OK (Content-Type: text/event-stream)
        Server-->>Client: SSE Event (data: {jsonrpc:"2.0", id:123, result:{taskId:"T1", status:{state:"working"}, kind:"status-update", final:false}})
        Server-->>Client: SSE Event (data: {jsonrpc:"2.0", id:123, result:{taskId:"T1", artifact:{artifactId:"A1", parts:[{kind:"text", text:"Section 1..."}]}, kind:"artifact-update", append:false, lastChunk:false}})
        Server-->>Client: SSE Event (data: {jsonrpc:"2.0", id:123, result:{taskId:"T1", artifact:{artifactId:"A1", parts:[{kind:"text", text:"Section 2..."}]}, kind:"artifact-update", append:true, lastChunk:true}})
        Server-->>Client: SSE Event (data: {jsonrpc:"2.0", id:123, result:{taskId:"T1", status:{state:"completed"}, kind:"status-update", final:true}})
        Note over Server: Connection Closes
    
    ```
    
4. **Push Notifications:**
    - Client initiates task and provides a webhook URL.
    - Client can disconnect.
    - When task state changes significantly (e.g., completion), A2A Server POSTs a notification to the client's webhook.
    - Client's webhook service receives notification, then typically calls `tasks/get` to fetch full task details.


> ## Choosing the Right Interaction Pattern
> 
> - Use `message/send` with polling (`tasks/get`) for tasks that are somewhat long but where the client can afford to check periodically.
> - Prefer `message/stream` for interactive experiences requiring real-time updates or incremental results display.
> - Use push notifications for very long-running tasks (minutes/hours/days) or when clients (like mobile apps or serverless functions) cannot maintain persistent connections.
> {: .prompt-info }

### Authentication, Authorization, and Security

A2A emphasizes leveraging standard web security:

- **Transport Security:** HTTPS is mandatory for production.
- **Authentication:** Declared in `AgentCard.securitySchemes`. Credentials (API keys, Bearer tokens from OAuth/OIDC) are passed in HTTP headers. A2A Servers MUST authenticate every request. The A2A protocol itself does not handle credential acquisition (this is out-of-band).
- **Authorization:** Server-side responsibility based on the authenticated client identity and defined policies.
- **Push Notification Security:** Requires careful validation of webhook URLs by the server (to prevent SSRF) and strong authentication of notifications by the client's webhook receiver. The spec suggests mechanisms for the server to authenticate to the client's webhook.
- **Input Validation:** Servers must validate all RPC parameters and message/artifact content.


> ## Security is a Shared Responsibility in A2A
> 
> While the A2A protocol provides fields in the Agent Card to declare security requirements (e.g., "this API requires OAuth2 with 'read:pets' scope"), the actual implementation of acquiring tokens, validating them, and enforcing authorization policies rests with the A2A Client and A2A Server implementers, using standard web security libraries and practices.
> {: .prompt-info }

### Relationship to MCP (Model Context Protocol)

- **MCP:** Focuses on how an AI model/agent connects to and uses **tools, APIs, and resources** (like function calling).
- **A2A:** Focuses on how independent **AI agents communicate and collaborate as peers**.

An A2A Server agent might internally use MCP to interact with its own set of tools to fulfill a task requested by an A2A Client.

### ADK and A2A Integration

Currently, ADK's primary mode of multi-agent interaction is "agent transfer" within a single `Runner` process. However, ADK is designed with an eye towards broader interoperability.

- **Exposing an ADK Agent as an A2A Server:**
The `google-a2a/A2A` repository includes an [example](https://github.com/google-a2a/A2A/tree/main/samples/python/agents/google_adk) demonstrating how to wrap an ADK agent system (with its `Runner`) within a web server (e.g., using FastAPI or Starlette, similar to how `adk web` works) to expose an A2A-compliant HTTP endpoint.
    1. Define an `AgentCard` for your ADK agent system.
    2. Implement an `AgentExecutor` (from the `a2a-sdk`) that translates incoming A2A `message/send` or `message/stream` requests into calls to your ADK `runner.run_async()`.
    3. Map the ADK `Event` stream back into A2A `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`, or final `Task` objects to be sent via SSE or as JSON-RPC responses.
    4. This allows external A2A clients to interact with your ADK agent system.

- **ADK Agent as an A2A Client :**
You could build a custom ADK `BaseTool` (or `FunctionTool`) that acts as an A2A client. This tool would:
    1. Take parameters like the target A2A agent's URL (or a way to discover its Agent Card) and the user's request.
    2. Use an HTTP client (like `aiohttp` for async) and potentially the `a2a-sdk` client library to send A2A requests to the remote agent.
    3. Process the A2A response (which might be a `Task` ID for polling, or an SSE stream to consume) and return a summary or result to the calling ADK agent.

```python
# --- Conceptual ADK Tool as A2A Client ---
from google.adk.tools import BaseTool, ToolContext
from google.genai.types import FunctionDeclaration, Schema, Type
import httpx # or aiohttp
# Potentially use a2a-sdk client if available and suitable
# from a2a import A2AClient, MessageSendParams, Message, TextPart (from a2a-sdk)

class RemoteA2AAgentTool(BaseTool):
    def __init__(self, name: str, description: str, target_agent_card_url: str):
        super().__init__(name=name, description=description)
        self.target_agent_card_url = target_agent_card_url
        # In a real tool, you might fetch and parse the AgentCard here
        # to understand the target_agent_url and capabilities.
        self.target_agent_api_endpoint = "extracted_or_known_from_card" # Placeholder

    def _get_declaration(self) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=Schema(type=Type.OBJECT, properties={
                "user_prompt": Schema(type=Type.STRING, description="The prompt to send to the remote A2A agent.")
            }, required=["user_prompt"])
        )

    async def run_async(self, args: dict[str, any], tool_context: ToolContext) -> dict:
        user_prompt = args.get("user_prompt")
        if not user_prompt:
            return {"error": "User prompt is required."}

        # Construct A2A message/send payload
        a2a_request_payload = {
            "jsonrpc": "2.0",
            "id": tool_context.function_call_id or "adk_a2a_call",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": user_prompt}],
                    "messageId": f"adk-msg-{tool_context.invocation_id}"
                }
                # Add contextId, taskId if continuing an existing A2A task
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.target_agent_api_endpoint, json=a2a_request_payload, timeout=30.0)
                response.raise_for_status() # Raise an exception for bad status codes
                a2a_response = response.json()
                # Process a2a_response (check for errors, extract result/task_id)
                # This is highly dependent on the A2A server's response structure (Task vs Message directly)
                return {"status": "success", "remote_agent_response": a2a_response.get("result", {})}
        except httpx.HTTPStatusError as e:
            return {"error": f"A2A HTTP error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"error": f"A2A request failed: {str(e)}"}

# Usage:
# remote_invoice_agent_tool = RemoteA2AAgentTool(
# name="query_remote_invoice_agent",
# description="Sends a query to the remote A2A-compliant invoicing agent.",
# target_agent_card_url="<https://finance.example.com/.well-known/agent.json>"
# )
# my_adk_agent = Agent(..., tools=[remote_invoice_agent_tool])

```


> ## A2A for True Inter-Framework Collaboration
> 
> If your goal is to have an ADK agent interact with an agent built in LangGraph, CrewAI, Semantic Kernel, or any other framework that can expose or consume an A2A interface, then understanding and implementing A2A (either as a server wrapper for your ADK agent or a client tool within ADK) is the path forward.
> {: .prompt-info }

**What's Next?**

This chapter has provided a high-level introduction to advanced multi-agent interaction patterns, showing how ADK can interface with powerful concepts like LangGraph and the A2A protocol. These approaches enable the development of highly sophisticated, stateful, and potentially distributed agent systems.

Next, we will put it all together in a more real-life example of a multi-agent systems.
