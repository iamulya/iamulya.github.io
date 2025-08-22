---
title: Chapter 6 - Leveraging Pre-built and Specialized Tools 
date: "2025-08-22 10:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Previously we learned how to craft custom tools by wrapping Python functions with `FunctionTool`. This is incredibly powerful for bespoke tasks. However, ADK also provides a suite of pre-built and specialized tools designed to handle common functionalities that many agents require. Leveraging these tools can significantly speed up your development process and provide robust, tested integrations out-of-the-box.

This chapter will introduce you to some of the most useful pre-built tools in ADK, covering internet access, web page interaction, memory management, artifact handling, and user interaction.

## Internet Access: `GoogleSearchTool` and `VertexAiSearchTool`

Giving agents access to up-to-date information from the internet is a common requirement. ADK offers tools to integrate with Google Search and Vertex AI Search.

**`google.adk.tools.google_search` (GoogleSearchTool)**

This tool allows your agent to perform Google searches. You don't implement its `run_async` method; you simply add it to your agent's tool list, and the model, when appropriate, will invoke it and incorporate search results into its response generation.

```python
from google.adk.agents import Agent
from google.adk.tools import google_search # Import the pre-built tool
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from building_intelligent_agents.utils import create_session, load_environment_variables, DEFAULT_LLM

load_environment_variables()

search_savvy_agent = Agent(
    name="search_savvy_agent",
    model=DEFAULT_LLM, 
    instruction="You are a helpful research assistant. Use Google Search when you need to find current information or verify facts.",
    tools=[google_search] # Add the tool instance
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=search_savvy_agent, app_name="SearchApp")
    prompts = [
        "What is the latest news about the Mars rover Perseverance?",
        "Who won the latest Formula 1 race?",
        "What is the capital of France?" 
    ]

    session_id = "search_session"
    user_id = "search_user"

    create_session(runner, session_id, user_id)

    for prompt_text in prompts:
        print(f"\nYOU: {prompt_text}")
        user_message = Content(parts=[Part(text=prompt_text)], role="user")

        print("ASSISTANT: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
                    # With search, grounding metadata might be present
                    if event.grounding_metadata and event.grounding_metadata.web_search_queries:
                        print(f"\n  (Searched for: {event.grounding_metadata.web_search_queries})", end="")
        print()
```

When `search_savvy_agent` processes a query like "What is the latest news about the Mars rover Perseverance?", the model, recognizing the need for current information and seeing the `google_search` tool available, can internally trigger a search. The search results are then used by the model to formulate its answer. You might see `grounding_metadata` in the `Event` indicating the queries performed.


> ## Model-Integrated Search
> 
> The beauty of google_search with capable Gemini models (Gemini 2 and later) is its seamless integration. The model often handles the search, result interpretation, and citation implicitly. This can lead to more natural and well-grounded responses without explicit tool call/response steps visible in the event trace for the search itself, though grounding metadata should appear.
> {: .prompt-info }

**`google.adk.tools.VertexAiSearchTool`**

If you have a specific Vertex AI Search data store configured, you can use `VertexAiSearchTool` to ground your agent's responses in your own curated data.

```python
from google.adk.agents import Agent
from google.adk.tools import VertexAiSearchTool # Import the tool
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os

from building_intelligent_agents.utils import create_session, DEFAULT_LLM

# This requires a Google Cloud Project and a configured Vertex AI Search data store.
# Ensure your environment is authenticated for GCP (e.g., via `gcloud auth application-default login`)
# and the necessary APIs are enabled (Vertex AI API, Search API).

# Furthermore, upload the document from https://arxiv.org/abs/2505.24832 in a GCP bucket.
# Subsequently, create a data store using this bucket and wait until the document is imported into the data store

# Replace with your actual project, location, and data store ID
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") 

# Example Data Store ID format: projects/<PROJECT_ID>/locations/<LOCATION>/collections/default_collection/dataStores/<DATA_STORE_ID>
DATA_STORE_ID = os.getenv("VERTEX_AI_SEARCH_DATA_STORE_ID")

if not all([GCP_PROJECT, DATA_STORE_ID]):
    print("Error: GOOGLE_CLOUD_PROJECT and VERTEX_AI_SEARCH_DATA_STORE_ID environment variables must be set.")
    # exit() # Uncomment to make it a hard stop

# Initialize the tool with your data store ID
if DATA_STORE_ID:
    llm_knowledge_tool = VertexAiSearchTool(data_store_id=DATA_STORE_ID)

    llm_knowledge_agent = Agent(
        name="llm_knowledge_agent",
        model=DEFAULT_LLM, # Requires a model that supports Vertex AI Search tool
        instruction="You are an expert on LLM memories. Use the provided search tool to answer the user queries.",
        tools=[llm_knowledge_tool]
    )

    if __name__ == "__main__":
        if not DATA_STORE_ID:
            print("Skipping VertexAiSearchTool example as DATA_STORE_ID is not set.")
        else:
            runner = InMemoryRunner(agent=llm_knowledge_agent, app_name="LlmKnowledgeApp")
            session_id = "s_llm_knowledge"
            user_id = "emp123"
            create_session(runner, session_id, user_id)

            prompt = "What is double descent phenomenon"
            print(f"\nYOU: {prompt}")
            user_message = Content(parts=[Part(text=prompt)], role="user")
            print("ASSISTANT: ", end="", flush=True)
            for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
                    if event.grounding_metadata and event.grounding_metadata.retrieval_queries:
                         print(f"\n  (Retrieved from Vertex AI Search with queries: {event.grounding_metadata.retrieval_queries})", end="")
            print()
else:
    print("Skipping VertexAiSearchTool agent definition as DATA_STORE_ID is not set.")
    llm_knowledge_agent = None # Define it as None if not configured
```

This tool tells the Gemini model to use the specified Vertex AI Search data store for grounding its responses. The model handles the retrieval and incorporates the information.


> ## Configuration and Permissions for Search Tools
> 
> `VertexAiSearchTool`: Requires a correctly configured Vertex AI Search data store/engine and appropriate IAM permissions for the credentials ADK is using to access GCP (e.g., your user credentials via `gcloud auth application-default login`, or a service account if deployed). Make sure you are using the Vertex AI setup for this example, i.e. 
> 
> GOOGLE_GENAI_USE_VERTEXAI=1 
> 
> GOOGLE_CLOUD_PROJECT=your-project-id 
> 
> GOOGLE_CLOUD_LOCATION=location 
> 
> VERTEX_AI_SEARCH_DATA_STORE_ID=your-data-store-id
> 
> {: .prompt-info }

## Web Page Loading: `load_web_page` tool

Sometimes, an agent might get a URL from a search result or user input and need to fetch the content of that web page for further processing (e.g., summarization, information extraction).

The `google.adk.tools.load_web_page` is a regular Python function that can be wrapped with `FunctionTool`. It uses the `requests` and `BeautifulSoup4` (with `lxml` parser) libraries to fetch and extract text from a URL.

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.load_web_page import load_web_page # The function itself
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# Ensure you have the necessary libraries: pip install requests beautifulsoup4 lxml
# Create the FunctionTool
web_page_loader_tool = FunctionTool(func=load_web_page)
# You can customize name and description if needed:
# web_page_loader_tool.name = "fetch_web_content"
# web_page_loader_tool.description = "Fetches and extracts text content from a given URL."

browser_agent = Agent(
    name="web_browser_agent",
    model=DEFAULT_LLM,
    instruction="You are an assistant that can fetch content from web pages using the provided tool and then answer questions about it or summarize it.",
    tools=[web_page_loader_tool]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=browser_agent, app_name="BrowserApp")

    session_id = "browse_session"
    user_id = "browse_user"

    create_session(runner, session_id, user_id)

    prompts = [
        "Can you get the main text from [https://www.python.org/] and summarize it in one sentence?"
    ]

    for prompt_text in prompts:
        print(f"\nYOU: {prompt_text}")
        user_message = Content(parts=[Part(text=prompt_text)], role="user")
        print("ASSISTANT: ", end="", flush=True)

        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
        print()
```


> ## Combine Search and Page Loading
> 
> For robust web information retrieval, agents often benefit from a two-step process:
>  
> 1. Use a search tool (`google_search`) to find relevant URLs.
> 2. Use `load_web_page` (via `FunctionTool`) to fetch content from a promising URL identified in step 1.
> 
> This requires the agent's instruction to guide it through this multi-step reasoning.
> {: .prompt-info }


> ## Web Page Complexity and Size
> 
> - `load_web_page` extracts text using BeautifulSoup. It might struggle with highly dynamic JavaScript-heavy pages or non-HTML content.
> - Fetched web page content can be very long. This can lead to large prompts when the content is fed back to the LLM, potentially exceeding token limits or increasing costs. Consider strategies like instructing the LLM to request summarization of specific sections, or implementing chunking if you process the content yourself before sending it to the LLM.
> 
> You can however also easily more powerful browser automation tools like [browser-use](https://github.com/browser-use/browser-use) as a custom tool with ADK Agent.
> {: .prompt-info }

## Interacting with Agent Memory: `LoadMemoryTool`, `PreloadMemoryTool`

As discussed previously, ADK provides a `MemoryService` for agents to have long-term memory across sessions. Two specialized tools facilitate interaction with this service:

- **`google.adk.tools.load_memory_tool` (LoadMemoryTool)**:
    - Allows the agent to actively query its long-term memory.
    - The LLM formulates a `query` string, and this tool passes it to the `MemoryService.search_memory()` method.
    - The search results (a list of `MemoryEntry` objects) are returned to the LLM.
    - **When to use:** When the agent explicitly needs to recall past information relevant to the current query.
- **`google.adk.tools.preload_memory_tool` (PreloadMemoryTool)**:
    - This is a "request processor" tool. It doesn't have a `run_async` method that the LLM calls directly.
    - Instead, its `process_llm_request` method is invoked *before* the `LlmRequest` is sent to the LLM.
    - It automatically takes the current user's query, searches the memory via `tool_context.search_memory()`, and if relevant memories are found, it prepends them to the system instruction of the `LlmRequest` under a `<PAST_CONVERSATIONS>` tag.
    - **When to use:** To proactively provide relevant context from past conversations to the LLM at the beginning of each turn, without the LLM needing to explicitly ask for it.

```python
from google.adk.agents import Agent
from google.adk.tools import load_memory, preload_memory
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.adk.events.event import Event
from google.genai.types import Content, Part
import time

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Setup a MemoryService with some data ---
# In a real app, this would be a persistent service like VertexAiRagMemoryService
# For this example, we'll manually populate the InMemoryMemoryService.
memory_service_instance = InMemoryRunner(agent=Agent(name="dummy"), app_name="MemApp").memory_service

# Create a dummy past session to populate memory
past_session = Session(
    app_name="MemApp",
    user_id="memory_user",
    id="past_session_abc",
    events=[
        Event(author="user", timestamp=time.time()-3600, content=Content(parts=[Part(text="My favorite color is blue.")])),
        Event(author="mem_agent", timestamp=time.time()-3590, content=Content(parts=[Part(text="Okay, I'll remember that your favorite color is blue.")])),
        Event(author="user", timestamp=time.time()-1800, content=Content(parts=[Part(text="I live in Paris.")])),
        Event(author="mem_agent", timestamp=time.time()-1790, content=Content(parts=[Part(text="Noted, you live in Paris.")])),
    ]
)
# await memory_service_instance.add_session_to_memory(past_session) # Call in async context if not using InMemory

# For InMemoryMemoryService, direct manipulation for setup:
memory_service_instance._session_events[f"MemApp/memory_user"] = {"past_session_abc": past_session.events}


# --- Agent using PreloadMemoryTool ---
proactive_memory_agent = Agent(
    name="proactive_memory_agent",
    model=DEFAULT_LLM,
    instruction="You are a helpful assistant that remembers things about the user. Relevant past conversations will be provided to you automatically.",
    tools=[preload_memory] # Just add it, it works automatically
)

# --- Agent using LoadMemoryTool ---
reactive_memory_agent = Agent(
    name="reactive_memory_agent",
    model=DEFAULT_LLM,
    instruction="You are a helpful assistant. If you need to recall past information to answer a question, use the 'load_memory' tool with a relevant query.",
    tools=[load_memory]
)

root_agent = reactive_memory_agent

if __name__ == "__main__":
    print("--- Testing Proactive Memory Agent (with PreloadMemoryTool) ---")
    # The InMemoryRunner for proactive_memory_agent needs our populated memory_service_instance
    runner_proactive = InMemoryRunner(agent=proactive_memory_agent, app_name="MemApp")
    runner_proactive.memory_service = memory_service_instance # Inject our populated memory service

    session_id = "s_proactive"
    user_id = "memory_user"
    create_session(runner_proactive, session_id=session_id, user_id=user_id)

    prompt1 = "What do you know about me?"
    print(f"YOU: {prompt1}")
    user_message1 = Content(parts=[Part(text=prompt1)], role="user")
    print("PROACTIVE_AGENT: ", end="", flush=True)
    for event in runner_proactive.run(user_id=user_id, session_id=session_id, new_message=user_message1):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
    print()

    print("\n--- Testing Reactive Memory Agent (with LoadMemoryTool) ---")
    runner_reactive = InMemoryRunner(agent=reactive_memory_agent, app_name="MemApp")
    runner_reactive.memory_service = memory_service_instance # Inject our populated memory service

    reactive_session_id = "s_reactive"
    create_session(runner_reactive, session_id=reactive_session_id, user_id=user_id)

    prompt2 = "Remind me of my favorite color."
    print(f"YOU: {prompt2}")
    user_message2 = Content(parts=[Part(text=prompt2)], role="user")
    print("REACTIVE_AGENT: ", end="", flush=True)
    for event in runner_reactive.run(user_id=user_id, session_id=reactive_session_id, new_message=user_message2):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(part.text, end="")
    print()
```


> ## PreloadMemoryTool for Seamless Context
> 
> PreloadMemoryTool is excellent for providing always-on, relevant context from past interactions. It makes the agent seem more naturally aware of history without requiring explicit "search my memory" instructions from the user or the LLM.
> {: .prompt-info }


> ## PreloadMemoryTool Prompt Length
> 
> If the memory search for PreloadMemoryTool returns a lot of text, it can significantly increase the length of the system instruction and thus the overall prompt sent to the LLM. Be mindful of token limits and potential increases in latency or cost. The underlying MemoryService implementation (e.g., similarity_top_k in VertexAiRagMemoryService) can help control how much information is retrieved.
> {: .prompt-info }

## Managing Agent Artifacts: `LoadArtifactsTool`

Agents might generate or need to refer to files (artifacts) like images, documents, or data files.

- **`google.adk.tools.load_artifacts_tool` (LoadArtifactsTool)**:
    - This is another "request processor" tool.
    - Its `process_llm_request` method checks the `ArtifactService` for available artifacts in the current session.
    - If artifacts exist, it appends an instruction to the `LlmRequest` informing the LLM about the available artifact names and instructing it to use the `load_artifacts` function (which is a pseudo-function name this tool handles) if it needs to access their content.
    - When the LLM "calls" `load_artifacts(artifact_names=["file.txt"])`, this tool intercepts it. If the actual content is needed for the LLM to proceed (i.e., the LLM has just "called" it in the previous turn), the tool will load the specified artifacts from the `ArtifactService` and append them as `Content` objects (with `Part.inline_data`) to the `LlmRequest` for the *next* LLM call.
    - This allows a two-step process: 1. LLM becomes aware of artifacts. 2. LLM requests specific artifacts, and their content is then provided.

```python
from google.adk.agents import Agent
from google.adk.tools import load_artifacts # The pre-built tool
from google.adk.tools.tool_context import ToolContext # For a custom tool to SAVE artifacts
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import asyncio # For async save_artifact in tool

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# Custom tool to simulate saving an artifact
async def create_report_artifact(report_content: str, tool_context: ToolContext) -> dict:
    """Creates a report and saves it as an artifact."""
    filename = "summary_report.txt"
    artifact_part = Part(text=report_content)
    await tool_context.save_artifact(filename=filename, artifact=artifact_part)
    return {"status": "success", "filename_created": filename, "message": f"Report '{filename}' created and saved."}

report_creator_tool = FunctionTool(func=create_report_artifact)

artifact_handling_agent = Agent(
    name="artifact_manager",
    model=DEFAULT_LLM,
    instruction="You can create reports and then later refer to them. If a report is available, use load_artifacts to access its content if needed.",
    tools=[
        report_creator_tool,
        load_artifacts # Add the tool to make agent aware of artifacts
    ]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=artifact_handling_agent, app_name="ArtifactApp")
    session_id = "s_artifact_test"
    user_id="artifact_user"
    create_session(runner, session_id, user_id)

    async def main():
        # Turn 1: Create an artifact
        prompt1 = "Please create a short report about ADK."
        print(f"\nYOU: {prompt1}")
        user_message1 = Content(parts=[Part(text=prompt1)], role="user")
        print("AGENT: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message1):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(part.text, end="")
        print()
        # Expected: Agent calls create_report_artifact. Artifact "summary_report.txt" is saved.

        # Turn 2: Ask about the artifact
        prompt2 = "What was in the report you just created?"
        print(f"\nYOU: {prompt2}")
        user_message2 = Content(parts=[Part(text=prompt2)], role="user")
        print("AGENT: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message2):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(part.text, end="")
        print()

    asyncio.run(main())
```


> ## LoadArtifactsTool for Contextual File Access
> 
> This tool is particularly useful when an agent needs to refer back to files it (or another process/tool) previously created within the same session. It avoids cluttering the prompt with all file contents on every turn, only loading them when the LLM deems it necessary.
> {: .prompt-info }

## User Interaction: `GetUserChoiceTool`

Sometimes, an agent needs to present a few options to the user and get their selection before proceeding.

- **`google.adk.tools.get_user_choice_tool` (a `LongRunningFunctionTool`)**:
    - The **LLM generates the options** based on the conversation context and its understanding of the ambiguity. The option can also be provided in the LLM instructions in case the options are not dynamic.
    - The get_user_choice_tool itself just acts as a signal and a placeholder for this user interaction. Its Python code returning None and setting skip_summarization=True are conventions for the ADK framework to handle this special "wait for user input" state.
    - Your application's **UI layer** is responsible for:
        1. Detecting that the get_user_choice tool was called by the LLM.
        2. Extracting the options argument that the LLM provided to the tool.
        3. Displaying these options to the human user.
        4. Collecting the human user's selection.
        5. Sending the selection back to the agent as a tool response.

```python
import asyncio
from google.genai.types import Content, Part, FunctionResponse

from google.adk.agents import Agent
from google.adk.tools import get_user_choice
from google.adk.runners import InMemoryRunner

from building_intelligent_agents.utils import load_environment_variables, DEFAULT_LLM
load_environment_variables()

instruction = (
    "You are a helpful beverage assistant. "
    "First, ask the user to choose between 'coffee' and 'tea' using the get_user_choice tool. "
    "After the user makes a choice, confirm their selection by saying 'You chose [their choice]! Excellent!'"
)

beverage_assistant = Agent(
    name="beverage_assistant",
    model=DEFAULT_LLM,
    instruction=instruction,
    tools=[get_user_choice],
)

runner = InMemoryRunner(agent=beverage_assistant)


async def main():
    """Simulates a multi-turn conversation with the agent."""
    user_id = "test_user"
    session_id = "test_session_1234"

    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    # --- Turn 1: User starts the conversation ---
    print("--- Turn 1: User starts the conversation ---")
    user_message = "I'm thirsty, what are my options?"
    print(f"[user]: {user_message}
")

    function_call_id = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(parts=[Part(text=user_message)]),
    ):
        print(f"  [event from agent]: {event.author} -> {event.content.parts if event.content else 'No Content'}")
        if calls := event.get_function_calls():
            options = calls[0].args.get("options", [])
            function_call_id = calls[0].id
            print(
                f"  [system]: Agent wants to call get_user_choice with ID: {function_call_id}"
            )
            print(f"  [system]: The UI would now show the user these options: {options}
")

    if not function_call_id:
        print("Error: Agent did not call the get_user_choice tool as expected.")
        return

    # --- Turn 2: Simulate the user making a choice ---
    user_choice = "coffee"
    print(f"--- Turn 2: User chooses '{user_choice}' ---")

    # Create the function response object first, ensuring the ID is set
    func_resp = FunctionResponse(
        name="get_user_choice",
        response={"result": user_choice},
        id=function_call_id  # This correctly sets the ID
    )

    # Now create the Part from the FunctionResponse object
    response_part = Part(function_response=func_resp)

    # Create and add an empty text part to satisfy the Gemini API which expects a text Part for "user" Content
    empty_text_part = Part(text="")

    # Create the final content message to send to the agent
    function_response_content = Content(
        role="user", # Function responses are sent back as the 'user'
        parts=[empty_text_part,response_part],
    )

    print(
        f"[user function_response]: name='get_user_choice', response={'result': '{user_choice}'}\n"
    )

    # --- Turn 3: Agent processes the choice and responds ---
    print("--- Turn 3: Agent confirms the choice ---")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=function_response_content,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"[{event.author}]: {event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```


> ## Best Practice: Clear Instructions for get_user_choice
> 
> Since the user's choice comes in a subsequent turn, your agent's main instruction should guide the LLM on:
> 
> - How to phrase the question when presenting options.
> - How to recognize and process the user's choice from their next message.
> 
> This often involves a multi-turn reasoning capability in the LLM.
> {: .prompt-info }

## Controlling Agent Loops: `ExitLoopTool`

When using orchestrator agents like `LoopAgent` (which we'll cover in multi-agent systems), you need a way for a sub-agent within the loop to signal that the loop should terminate.

- **`google.adk.tools.exit_loop` (a Python function, typically wrapped by `FunctionTool`)**:
    - When this tool is called by an agent running inside a `LoopAgent`, it sets `tool_context.actions.escalate = True`.
    - The `LoopAgent` checks for this `escalate` flag in the events from its sub-agents. If `True`, the `LoopAgent` terminates its loop.

This tool is primarily relevant in the context of `LoopAgent`, so we'll see more practical examples when we discuss multi-agent orchestration.

```python
# Conceptual usage within a LoopAgent's sub-agent
from google.adk.tools import FunctionTool, exit_loop
from google.adk.agents import Agent, LoopAgent # LoopAgent to be detailed later

exit_tool = FunctionTool(exit_loop)

sub_agent_in_loop = Agent(
    name="looper_child",
    model="gemini-1.0-pro", # example
    instruction="Perform a task. If the task is complete or a certain condition is met, call the 'exit_loop' tool.",
    tools=[exit_tool]
)

# conceptual_loop_agent = LoopAgent(
#     name="main_looper",
#     sub_agents=[sub_agent_in_loop],
#     max_iterations=5 # Example
# )

```

**What's Next?**

By now, you should have a good grasp of both creating custom `FunctionTool`s and leveraging ADK's diverse set of pre-built tools. These tools significantly enhance what your agents can perceive and act upon. Next we'll shift our focus to integrating with a vast array of external REST APIs by learning how to use OpenAPI specifications and the `APIHubToolset`.
