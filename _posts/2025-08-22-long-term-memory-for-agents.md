---
title: Chapter 19 - Long-Term Memory for Agents 
date: "2025-08-22 17:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

While session state provides agents with memory within a single conversation, and artifacts allow them to work with files, true intelligence often requires the ability to learn and recall information across *many* conversations and over extended periods. This is the domain of **Long-Term Memory**. ADK provides a `MemoryService` abstraction to equip agents with this capability, allowing them to store and retrieve knowledge gleaned from past interactions.

This chapter introduces the `BaseMemoryService` interface, its implementations (`InMemoryMemoryService` and `VertexAiRagMemoryService`), and the tools agents use to interact with this persistent knowledge base.

## The `BaseMemoryService` Interface and `MemoryEntry`

The `google.adk.memory.BaseMemoryService` is an abstract class defining the contract for how agents store and retrieve long-term memories.

**Key Abstract Methods of `BaseMemoryService`:**

- **`add_session_to_memory(session: Session)`**: Ingests an entire `Session` object (or relevant parts of it) into the memory store. This is how the agent "learns" from past conversations. The service implementation decides how to chunk, embed, and index this session data.
- **`search_memory(app_name: str, user_id: str, query: str) -> SearchMemoryResponse`**: Searches the memory for information relevant to the given `query`, scoped to the `app_name` and `user_id`.

**`google.adk.memory.SearchMemoryResponse`**:

- A Pydantic model that wraps the results of a memory search.
- Its primary attribute is `memories: list[MemoryEntry]`.

**`google.adk.memory.MemoryEntry`**:

- A Pydantic model representing a single piece of information retrieved from memory.
- Key attributes:
    - `content: types.Content`: The actual content of the memory (e.g., a user's statement, an agent's important response).
    - `author: Optional[str]`: Who originally authored this piece of information.
    - `timestamp: Optional[str]`: An ISO 8601 formatted string indicating when the original content occurred.

The `Runner` is typically configured with a `MemoryService` instance, making it accessible to agents and tools via their contexts.

![*Diagram: `BaseMemoryService` interface and its concrete implementations.*](/assets/img/2025-08-22-long-term-memory-for-agents/figure-1.png)


## Strategies for Populating and Retrieving Memory

- **Populating Memory (`add_session_to_memory`):**
    - **When to call:** Typically, you might call `add_session_to_memory` at the end of a user session, or perhaps after significant milestones within a long-running session. This allows the agent to "commit" the learnings from that interaction to its long-term store.
    - **What to store:** The `MemoryService` implementation decides how to process the `Session` object. It might extract key user statements, agent summaries, important facts identified, or even entire conversational turns. For services like `VertexAiRagMemoryService`, it often involves converting text into vector embeddings for semantic search.
- **Retrieving Memory (`search_memory`):**
    - **Active Retrieval:** An agent can explicitly decide to search its memory using a tool like `LoadMemoryTool` (see Section 18.5). The LLM formulates a query based on the current conversation to find relevant past information.
    - **Proactive Retrieval:** The `PreloadMemoryTool` (see Section 18.5) automatically searches the memory based on the current user's input *before* the main LLM call for the turn. Relevant retrieved memories are then injected into the LLM's prompt as context.

## `InMemoryMemoryService`

The `google.adk.memory.InMemoryMemoryService` provides a very basic, keyword-based memory implementation that stores session events directly in memory.

- **Pros:**
    - No external dependencies, easy to use for local development and testing.
    - Automatically used by `InMemoryRunner` if no other memory service is provided.
- **Cons:**
    - Memory is lost when the Python process ends.
    - Search is a simple keyword match on the text parts of events, not semantic.
    - Can become slow and memory-intensive with very large histories.
- **How it works:**
    - `add_session_to_memory`: Stores a reference to the list of `Event` objects from the session, keyed by `app_name/user_id` and `session_id`.
    - `search_memory`: Iterates through all stored events for the user. It splits the query and event text into words and returns events where any query word (case-insensitive) matches any word in the event's text.

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext # For saving to memory & searching
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.adk.events.event import Event
from google.adk.memory import InMemoryMemoryService # Explicit import for clarity
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

# --- Agent and Tools ---
async def save_current_session_to_long_term_memory(tool_context: ToolContext) -> str:
    """Explicitly saves the current session's events to long-term memory."""
    if not tool_context._invocation_context.memory_service:
        return "Error: Memory service is not available."

    # Get the full current session object from the context
    current_session = tool_context._invocation_context.session
    await tool_context._invocation_context.memory_service.add_session_to_memory(current_session)
    return f"Session {current_session.id} has been committed to long-term memory."

save_to_memory_tool = FunctionTool(save_current_session_to_long_term_memory)

async def recall_information(query: str, tool_context: ToolContext) -> dict:
    """Searches long-term memory for information related to the query."""
    if not tool_context._invocation_context.memory_service:
        return {"error": "Memory service is not available."}

    search_response = await tool_context.search_memory(query)

    if not search_response.memories:
        return {"found_memories": 0, "results": "No relevant memories found."}

    # For InMemoryMemoryService, results are raw MemoryEntry objects. Let's format them.
    formatted_results = []
    for mem_entry in search_response.memories[:3]: # Limit for brevity
        text_parts = [p.text for p in mem_entry.content.parts if p.text]
        formatted_results.append(
            f"Author: {mem_entry.author}, Timestamp: {mem_entry.timestamp}, Content: {' '.join(text_parts)}"
        )
    return {"found_memories": len(search_response.memories), "results": formatted_results}

recall_tool = FunctionTool(recall_information)

memory_interactive_agent = Agent(
    name="memory_keeper",
    model=DEFAULT_LLM,
    instruction=(
        "You have access to a shared, long-term memory that persists across our conversations (sessions). "
        "Use 'save_current_session_to_long_term_memory' to save important facts. "
        "When asked to recall information, especially if it seems like something discussed previously (even in a different session), "
        "use 'recall_information' with a relevant search query to check this long-term memory."
    ),
    tools=[save_to_memory_tool, recall_tool]
)

if __name__ == "__main__":
    # Explicitly create an InMemoryMemoryService instance to interact with
    memory_service = InMemoryMemoryService()

    # Initialize runner with our specific memory_service
    runner = InMemoryRunner(agent=memory_interactive_agent, app_name="MemoryDemoApp")
    runner.memory_service = memory_service # Override the default one from InMemoryRunner

    user_id = "user_mem_test"
    session_id_1 = "memory_session_1"
    session_id_2 = "memory_session_2"
    create_session(runner, user_id=user_id, session_id=session_id_1)  # Create first session
    create_session(runner, user_id=user_id, session_id=session_id_2)  # Create second session

    async def run_turn(uid, sid, prompt_text, expected_tool_call=None):
        print(f"
--- Running for User: {uid}, Session: {sid} ---")
        print(f"YOU: {prompt_text}")
        user_message = Content(parts=[Part(text=prompt_text)], role="user")
        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(user_id=uid, session_id=sid, new_message=user_message):
            if event.author == memory_interactive_agent.name and event.content and event.content.parts and not event.get_function_calls() and not event.get_function_responses(): 
                print(event.content.parts[0].text.strip())
            # For debugging, show tool calls
            if event.get_function_calls():
                for fc in event.get_function_calls():
                    print(f"
  [TOOL CALL by {event.author}]: {fc.name}({fc.args})")
                    if expected_tool_call and fc.name == expected_tool_call:
                        print(f"    (Matches expected tool call: {expected_tool_call})")
            if event.get_function_responses():
                 for fr in event.get_function_responses():
                    print(f"
  [TOOL RESPONSE to {fr.name}]: {fr.response}")

    async def main():
        # Turn 1 (Session 1): Store some information
        await run_turn(user_id, session_id_1, "Please remember that my favorite hobby is hiking and save this session.",
                       expected_tool_call="save_current_session_to_long_term_memory")

        # Turn 2 (Session 1): Recall information
        await run_turn(user_id, session_id_1, "What did I tell you about my favorite hobby?",
                       expected_tool_call="recall_information")

        # Turn 3 (New Session - Session 2): Try to recall information from Session 1
        await run_turn(user_id, session_id_2, "Do you remember my favorite hobby?",
                       expected_tool_call="recall_information")

        # Verify memory content (InMemoryMemoryService specific inspection)
        search_res = await memory_service.search_memory(app_name="MemoryDemoApp", user_id=user_id, query="hobby")
        print(f"
[DEBUG] Direct memory search for 'hobby' found {len(search_res.memories)} entries.")
        for entry in search_res.memories:
            print(f"  - Author: {entry.author}, Content: {entry.content.parts[0].text if entry.content.parts else ''}")

    asyncio.run(main())
```


> ## InMemoryMemoryService is for Prototyping Only
> 
> Due to its simple keyword matching and lack of persistence, InMemoryMemoryService should not be used for production applications requiring reliable long-term memory. It's primarily for understanding the mechanics and for basic local testing.
> {: .prompt-info }

## `VertexAiRagMemoryService`: Leveraging Vertex AI RAG

For robust, scalable, and semantically rich long-term memory, ADK provides `VertexAiRagMemoryService`. This service integrates with **Vertex AI RAG (Retrieval-Augmented Generation)**, which allows you to build a corpus of your data (including past conversations) and perform semantic searches over it.

**Prerequisites:**

- Google Cloud Project with Vertex AI API enabled.
- Vertex AI RAG set up:
    - A RAG Corpus created.
    - Files (representing session data) uploaded to this corpus. `VertexAiRagMemoryService` handles formatting session data into files and uploading them.
- Authentication configured for ADK to access Vertex AI (e.g., `gcloud auth application-default login` or a service account with roles like "Vertex AI User" and permissions for RAG operations).
- `google-cloud-aiplatform` library installed with RAG support (usually included with recent versions or via `google-cloud-aiplatform[rag]`).

**Initialization:**

```python
from google.adk.memory import VertexAiRagMemoryService
import os

GCP_PROJECT_RAG = os.getenv("GOOGLE_CLOUD_PROJECT")
# Vertex AI RAG Corpus Name/ID or full resource name
# e.g., "my_adk_sessions_corpus" or
# "projects/<PROJECT_ID>/locations/<LOCATION>/ragCorpora/<CORPUS_ID>"
RAG_CORPUS_ID = os.getenv("ADK_RAG_CORPUS_ID")
# Optional location for the RAG corpus if not part of the full resource name
RAG_CORPUS_LOCATION = os.getenv("ADK_RAG_CORPUS_LOCATION", "us-central1")

if GCP_PROJECT_RAG and RAG_CORPUS_ID:
    try:
        # Construct the full corpus resource name if only ID is given
        full_rag_corpus_name = RAG_CORPUS_ID
        if not RAG_CORPUS_ID.startswith("projects/"):
            full_rag_corpus_name = f"projects/{GCP_PROJECT_RAG}/locations/{RAG_CORPUS_LOCATION}/ragCorpora/{RAG_CORPUS_ID}"

        rag_memory_service = VertexAiRagMemoryService(
            rag_corpus=full_rag_corpus_name,
            similarity_top_k=5, # Retrieve top 5 most similar chunks
            # vector_distance_threshold=0.7 # Optional: filter by similarity score
        )
        print(f"VertexAiRagMemoryService initialized with corpus: {full_rag_corpus_name}")

        # This instance can now be passed to a Runner
        # from google.adk.runners import Runner
        # my_agent_for_rag = ...
        # runner_with_rag_mem = Runner(
        #     app_name="MyRagApp", # This app_name is used in file display names in RAG
        #     agent=my_agent_for_rag,
        #     session_service=...,
        #     memory_service=rag_memory_service
        # )
    except ImportError:
        print("Vertex AI SDK with RAG support not found. Ensure 'google-cloud-aiplatform[rag]' is installed or similar.")
    except Exception as e:
        print(f"Failed to initialize VertexAiRagMemoryService: {e}")
        print("Ensure RAG Corpus exists, Vertex AI API is enabled, and auth is correct.")
else:
    print("GOOGLE_CLOUD_PROJECT or ADK_RAG_CORPUS_ID environment variables not set. Skipping VertexAiRagMemoryService setup.")
```

**How `VertexAiRagMemoryService` Works:**

- **`add_session_to_memory(session)`**:
    1. Converts the `session.events` into a structured text format (JSON lines, each representing an event with author, timestamp, and text).
    2. Saves this structured text to a temporary local file.
    3. Uploads this temporary file to the configured Vertex AI RAG Corpus using `vertexai.preview.rag.upload_file()`. The file in RAG is typically given a display name like `{app_name}.{user_id}.{session_id}` for identification.
    4. Vertex AI RAG then automatically chunks, embeds, and indexes the content of this file.
- **`search_memory(app_name, user_id, query)`**:
    1. Calls `vertexai.preview.rag.retrieval_query()` with the user's `query` and the configured RAG resources.
    2. Vertex AI RAG performs a semantic search across the indexed documents in the corpus.
        - *(Currently, filtering by `user_id` or `app_name` within the RAG query itself might be limited. The service primarily searches the whole corpus it's pointed to. Post-retrieval filtering based on `context.source_display_name` might be needed if strict user-scoping is critical directly from this tool without LLM re-filtering.)*
    3. The retrieved `Contexts` (chunks of text from the ingested session files) are parsed back into `MemoryEntry` objects. The service attempts to reconstruct event-like structures if the ingested format (JSON lines) is found.

![*Diagram: Interaction flow for `VertexAiRagMemoryService`.*](/assets/img/2025-08-22-long-term-memory-for-agents/figure-2.png)



> ## Best Practice: Semantic Search with VertexAiRagMemoryService
> 
> For intelligent recall based on meaning rather than just keywords, VertexAiRagMemoryService is the way to go. It allows your agent to find relevant past conversations even if the exact phrasing isn't used in the current query. This is crucial for building truly knowledgeable and context-aware long-term memory.
> {: .prompt-info }


> ## RAG Corpus Setup and Costs
> 
> - Setting up a Vertex AI RAG Corpus and ensuring your service account has the right permissions (e.g., "Vertex AI User", "Storage Object User" for the underlying GCS bucket of the RAG corpus) is essential.
> - Vertex AI RAG and its underlying services (like Vector Search, GCS) incur costs. Monitor your usage and understand the pricing model.
> - Ingestion into RAG is asynchronous. There might be a delay between calling `add_session_to_memory` and the data being fully searchable.
> {: .prompt-info }

## Tools for Memory Interaction: `LoadMemoryTool` and `PreloadMemoryTool`

As briefly introduced previously and demonstrated in the `InMemoryMemoryService` example, ADK provides two key tools for agents to interact with the configured `MemoryService`:

- **`google.adk.tools.load_memory_tool` (LoadMemoryTool)**:
    - A standard `FunctionTool` that an LLM can explicitly call.
    - Its declaration asks for a `query: str`.
    - When called, its `run_async` method (the `load_memory` function) invokes `tool_context.search_memory(query)`.
    - The `SearchMemoryResponse` (containing a list of `MemoryEntry` objects) is returned to the LLM. The LLM then uses this information.
    - **Use Case:** When the agent, based on its reasoning, identifies a specific need to look up past information.
- **`google.adk.tools.preload_memory_tool` (PreloadMemoryTool)**:
    - This is a "request processor" `BaseTool` (its `process_llm_request` method does the work).
    - It **automatically** runs before each main LLM call in a turn.
    - It takes the current `tool_context.user_content` (the user's latest message) as an implicit query to `tool_context.search_memory()`.
    - If relevant memories are found, their text content is formatted and **prepended to the `LlmRequest`'s system instruction** under a `<PAST_CONVERSATIONS>` tag.
    - The LLM receives this past context proactively, without needing to ask for it.
    - **Use Case:** To give the LLM relevant background from previous conversations automatically, making it seem more contextually aware from the start of its reasoning process for the current turn.

```python
from google.adk.agents import Agent
from google.adk.tools import load_memory, preload_memory
from google.adk.runners import InMemoryRunner
from google.adk.events.event import Event
from google.adk.memory import InMemoryMemoryService
from google.genai.types import Content, Part
import time
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM, DEFAULT_REASONING_LLM
load_environment_variables()  # Load environment variables for ADK configuration

# Setup dummy memory as in the InMemoryMemoryService example
memory_service_instance = InMemoryMemoryService()
past_events_for_memory = [
    Event(author="user", timestamp=time.time()-7200, content=Content(parts=[Part(text="I'm planning a trip to Japan next year.")])),
    Event(author="planner_bot", timestamp=time.time()-7190, content=Content(parts=[Part(text="Okay, Japan trip noted for next year!")])),
    Event(author="user", timestamp=time.time()-3600, content=Content(parts=[Part(text="For my Japan trip, I'm interested in Kyoto.")])),
    Event(author="planner_bot", timestamp=time.time()-3590, content=Content(parts=[Part(text="Kyoto is a great choice for your Japan trip!")])),
]
memory_service_instance._session_events["MemoryToolsApp/mem_tools_user"] = {"past_trip_planning": past_events_for_memory}

# Agent using both preload and allowing explicit load
smart_planner_agent = Agent(
    name="smart_trip_planner",
    model=DEFAULT_REASONING_LLM, # Needs good reasoning
    instruction="You are a trip planning assistant. Past conversation snippets might be provided under <PAST_CONVERSATIONS> to help you. "
                "If you need more specific details from past conversations not automatically provided, use the 'load_memory' tool with a targeted query. "
                "Remember to be helpful and use retrieved information effectively.",
    tools=[
        preload_memory, # Will automatically add relevant context to system instruction
        load_memory     # Allows LLM to explicitly query memory
    ]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=smart_planner_agent, app_name="MemoryToolsApp")
    runner.memory_service = memory_service_instance # Inject our populated memory service

    user_id = "mem_tools_user"
    session_id = "current_trip_planning"
    create_session(runner, user_id=user_id, session_id=session_id)  # Create a session for the user

    async def run_planner_turn(prompt_text):
        print(f"
YOU: {prompt_text}")
        user_message = Content(parts=[Part(text=prompt_text)], role="user")  # User message to the agent
        print("SMART_TRIP_PLANNER: ", end="", flush=True)

        # The Dev UI Trace is invaluable here to see:
        # 1. What `preload_memory_tool` adds to the system instruction.
        # 2. If/when `load_memory_tool` is called by the LLM.
        # 3. The results of `load_memory_tool`.
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and not (hasattr(part, 'thought') and part.thought): # Don't print thoughts for cleaner CLI
                        print(part.text, end="", flush=True)
                    elif part.function_call:
                        print(f"
  [TOOL CALL by {event.author}]: {part.function_call.name}({part.function_call.args})", end="")
                    # We might not see tool responses directly if preload_memory_tool is effective
                    # or if load_memory_tool's output is summarized by LLM.
        print()

    async def main():
        # Query that should benefit from preload_memory_tool
        await run_planner_turn("I mentioned a country I want to visit. What was it and which city was I interested in there?")
        # Expected: PreloadMemoryTool finds "Japan" and "Kyoto". Agent answers directly.

        # Query that might require explicit load_memory if preload isn't specific enough
        # (or if we wanted to demonstrate explicit call)
        # await run_planner_turn("What were the exact first two things I told you about my trip planning?")
        # Expected (if LLM decides to use load_memory): LLM calls load_memory(query="first two things about trip planning")

    asyncio.run(main())
```


> ## Combining PreloadMemoryTool and LoadMemoryTool
> 
> For many applications, using both tools provides a good balance:
> 
> - `PreloadMemoryTool` gives the LLM immediate, relevant context for the current user query.
> - `LoadMemoryTool` allows the LLM to dig deeper or search for different aspects of the memory if the preloaded information isn't sufficient or if it needs to answer a very specific historical question.
> 
> Your agent's instruction should guide it on how and when to use `load_memory` if `preload_memory` is also active.
> {: .prompt-info }


> ## Querying Effectiveness of Memory Search
> 
> The quality of results from search_memory (and thus the effectiveness of both memory tools) heavily depends on:
> 
> - **For `InMemoryMemoryService`:** How well the keywords in the `query` match words in the stored events.
> - **For `VertexAiRagMemoryService`:** The quality of the embeddings, the semantic similarity between the `query` and the stored content, and the RAG configuration (chunking strategy, `similarity_top_k`, etc.).
> 
> Crafting effective queries (either by the LLM for `LoadMemoryTool` or by using the user's direct input for `PreloadMemoryTool`) is key.
> {: .prompt-info }

**What's Next?**

With session management, artifact handling, and now long-term memory, our ADK agents are becoming quite sophisticated in how they manage and utilize information. This ability to remember and learn from past interactions is a cornerstone of building truly intelligent and personalized agents.

Next we'll shift our focus to a critical aspect of the development lifecycle: how to measure and assess the quality, correctness, and effectiveness of the agents we build.
