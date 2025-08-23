---
title: Chapter 16 - ADK Runner and Runtime Configuration 
date: "2025-08-22 15:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

We've spent considerable time designing and building agents, equipping them with tools, and even orchestrating them into multi-agent systems. Now, we turn our attention to the engine that brings these agents to life: the **ADK Runner**. This chapter delves into the `Runner` class, its core execution methods, and how you can customize the runtime behavior of your agents using the `RunConfig` object for features like streaming, speech interaction, and more.

## The `Runner` Class in Depth

The `google.adk.runners.Runner` is the central component responsible for executing your ADK agents. It acts as the bridge between your application code (which initiates an agent interaction) and the agent system itself.

**Key Responsibilities of the `Runner`:**

1. **Session Management:** It interacts with a `BaseSessionService` to create, retrieve, and update agent sessions. This includes loading conversation history and state at the beginning of an interaction and saving new events and state changes.
2. **Agent Invocation:** It identifies the correct root agent to invoke (or the appropriate agent to resume a conversation with) and calls its `run_async` (or `run_live`) method.
3. **Context Creation:** It constructs the `InvocationContext` object, providing the agent with all necessary information (session, services, user input, run configuration).
4. **Event Streaming:** It consumes the asynchronous stream of `Event` objects yielded by the agent and makes them available to your application.
5. **Input Handling:** It processes new user messages, potentially saving input blobs as artifacts before passing them to the agent.

**Initializing a `Runner`:**
You typically initialize a `Runner` by providing:

- `app_name: str`: A name for your application, used for namespacing sessions.
- `agent: BaseAgent`: The root agent of your application.
- `session_service: BaseSessionService`: An instance of a session service (e.g., `InMemorySessionService`, `DatabaseSessionService`).
- `artifact_service: Optional[BaseArtifactService]`: (Optional) An instance of an artifact service.
- `memory_service: Optional[BaseMemoryService]`: (Optional) An instance of a memory service.

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService
from google.adk.memory import InMemoryMemoryService
import os

from building_intelligent_agents.utils import load_environment_variables, DEFAULT_LLM
load_environment_variables()

# Define a simple agent
my_root_agent = Agent(
    name="main_app_agent",
    model=DEFAULT_LLM,
    instruction="You are the main agent for this application."
)

# Option 1: Using all InMemory services (similar to InMemoryRunner)
in_memory_session_svc = InMemorySessionService()
in_memory_artifact_svc = InMemoryArtifactService()
in_memory_memory_svc = InMemoryMemoryService()

runner_in_memory = Runner(
    app_name="MyInMemoryApp",
    agent=my_root_agent,
    session_service=in_memory_session_svc,
    artifact_service=in_memory_artifact_svc,
    memory_service=in_memory_memory_svc
)
print(f"Runner initialized with InMemory services for app: {runner_in_memory.app_name}")

# Option 2: Using persistent services (conceptual, requires setup)
# Ensure GOOGLE_CLOUD_PROJECT and potentially other env vars are set for GCS/Database
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
GCS_BUCKET_NAME = os.getenv("ADK_ARTIFACT_GCS_BUCKET", "my-adk-artifacts-bucket")
# Example: "mysql+pymysql://user:pass@host/db" or "sqlite:///./my_adk_sessions.db"
DATABASE_URL = os.getenv("ADK_DATABASE_URL", "sqlite:///./adk_sessions_chapter15.db")

if GOOGLE_CLOUD_PROJECT and GCS_BUCKET_NAME and DATABASE_URL:
    try:
        db_session_svc = DatabaseSessionService(db_url=DATABASE_URL)
        gcs_artifact_svc = GcsArtifactService(bucket_name=GCS_BUCKET_NAME, project=GOOGLE_CLOUD_PROJECT)
        # memory_svc_persistent = VertexAiRagMemoryService(...) # Or other persistent memory

        runner_persistent = Runner(
            app_name="MyPersistentApp",
            agent=my_root_agent,
            session_service=db_session_svc,
            artifact_service=gcs_artifact_svc,
            # memory_service=memory_svc_persistent
            memory_service=InMemoryMemoryService() # Placeholder for simplicity
        )
        print(f"Runner initialized with persistent services for app: {runner_persistent.app_name}")
    except Exception as e:
        print(f"Could not initialize persistent Runner: {e}")
        print("Ensure Database, GCS bucket, and relevant SDKs/permissions are set up.")
else:
    print("Skipping persistent Runner setup due to missing env vars (GOOGLE_CLOUD_PROJECT, ADK_ARTIFACT_GCS_BUCKET, ADK_DATABASE_URL).")
```


> ## Decoupling Agent Logic from Persistence
> 
> The Runner's design, requiring explicit service instances, promotes loose coupling. Your core agent logic (LlmAgent definitions, tools) remains independent of how sessions, artifacts, or memory are stored. This makes it easy to switch from local in-memory development to production-grade persistent backends.
> {: .prompt-info }

**Core Execution Methods:**

- **`run_async(user_id, session_id, new_message, run_config=RunConfig()) -> AsyncGenerator[Event, None]`**:
    - This is the primary asynchronous method for executing an agent turn.
    - It retrieves or creates the session, appends the `new_message` (if any), constructs the `InvocationContext`, invokes the appropriate agent's `run_async` method, and yields the stream of `Event` objects generated by the agent.
    - It also handles persisting events and state changes via the `SessionService` for non-partial events.
- **`run(user_id, session_id, new_message, run_config=RunConfig()) -> Generator[Event, None, None]`**:
    - A synchronous wrapper around `run_async`. It's convenient for simple scripts and local testing where full `asyncio` orchestration isn't desired.
    - Internally, it runs `run_async` in a separate thread and uses a queue to yield events back to the synchronous caller.
    
    ```python
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner # InMemoryRunner is a pre-configured Runner
    from google.genai.types import Content, Part

    from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
    load_environment_variables()  # Load environment variables for ADK configuration

    greet_agent = Agent(name="greeter", model=DEFAULT_LLM, instruction="Greet the user warmly.")
    runner = InMemoryRunner(agent=greet_agent, app_name="GreetApp")

    user_msg = Content(parts=[Part(text="Hello there!")], role="user")  # User message to the agent

    async def use_run_async():
        print("
--- Using run_async ---")
        session_id = "s_async"  
        user_id = "async_user"
        create_session(runner, user_id=user_id, session_id=session_id)
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(f"Async Event from {event.author}: {event.content.parts[0].text.strip()}")

    def use_run_sync():
        print("
--- Using run (sync wrapper) ---")
        session_id = "s_sync"  
        user_id = "sync_user"
        create_session(runner, user_id=user_id, session_id=session_id)
        for event in runner.run(user_id="sync_user", session_id="s_sync", new_message=user_msg):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(f"Sync Event from {event.author}: {event.content.parts[0].text.strip()}")
    ```
    
- **`run_live(user_id, session_id, live_request_queue, run_config=RunConfig(), ...)`**:
    - An **experimental** method for bidirectional streaming interactions, typically involving audio input/output.
    - It takes a `LiveRequestQueue` for sending real-time data (like audio chunks) to the agent and yields `Event` objects as the agent processes and responds.
    - This is used for more advanced scenarios like voice bots. We'll touch upon this conceptually when discussing `RunConfig`.

![*Diagram: Detailed sequence of `Runner.run_async` invocation.*](/assets/img/2025-08-22-adk-runner-and-runtime-configuration/figure-1.png)


## `InMemoryRunner` for Local Development and Testing

As seen in many examples, `google.adk.runners.InMemoryRunner` is a subclass of `Runner` that simplifies setup for local development:

```python
from google.adk.runners import InMemoryRunner
from google.adk.agents import Agent

my_agent = Agent(name="test_agent", model="gemini-2.0-flash", instruction="Be brief.")
# InMemoryRunner automatically sets up InMemorySessionService, InMemoryArtifactService, InMemoryMemoryService
runner = InMemoryRunner(agent=my_agent, app_name="TestApp")

```

It's perfect for:

- Quickly testing agent logic.
- Running examples from this book.
- Unit tests where you don't need persistent state across test runs.


> ## Best Practice: Start with InMemoryRunner
> 
> For new projects or when learning ADK, InMemoryRunner is the easiest way to get started. You can focus on defining your agent's logic and tools without worrying about database or cloud storage setup. Transition to a Runner with persistent services when you need to save conversation history or state beyond a single execution of your script.
> {: .prompt-info }

## Understanding `InvocationContext` and its Lifecycle

The `google.adk.agents.invocation_context.InvocationContext` is a Pydantic model that acts as a carrier for all relevant information during a single agent invocation (one full processing turn for a `new_message`).

**Key Attributes of `InvocationContext`:**

- `invocation_id: str`: A unique ID for this specific turn.
- `session: Session`: The current `Session` object (including history and state).
- `agent: BaseAgent`: The *current* agent being executed within this invocation.
- `user_content: Optional[types.Content]`: The initial user message that triggered this invocation.
- `run_config: Optional[RunConfig]`: The runtime configuration for this invocation.
- References to `artifact_service`, `session_service`, `memory_service`.
- `live_request_queue: Optional[LiveRequestQueue]`: For `run_live`.
- `end_invocation: bool`: A flag that can be set by callbacks or tools to prematurely terminate the current invocation.
- `_invocation_cost_manager`: Tracks metrics like LLM calls.

The `Runner` creates an `InvocationContext` at the start of `run_async` or `run_live`. This same context object (or a copy with the `agent` attribute updated) is passed down if control transfers between agents in a multi-agent system. This ensures all agents in a single turn share the same session view, services, and run configuration.

## Customizing Runtime with `RunConfig`

The `google.adk.agents.run_config.RunConfig` Pydantic model allows you to customize various runtime behaviors of the agent when you call `runner.run_async` or `runner.run`.

```python
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner, RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai.types import SpeechConfig, Content, Part # For speech and transcription
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

# This example is conceptual for some features like speech, as they require
# actual audio input/output capabilities not easily shown in a text script.

story_agent = Agent(
    name="story_teller",
    model=DEFAULT_LLM,
    instruction="Tell a very short, one-sentence story."
)
runner = InMemoryRunner(agent=story_agent, app_name="ConfigDemoApp")
session_id = "s_user_rc"  # Session ID for the user
user_id = "user_rc"  # User ID for the session
create_session(runner, user_id=user_id, session_id=session_id)  # Create a session for the user

user_input_message = Content(parts=[Part(text="A story please.")], role="user")  # User's input message to the agent

async def demo_run_configs():
    # Scenario 1: Default RunConfig (no streaming)
    print("
--- Scenario 1: Default (No Streaming) ---")
    default_config = RunConfig()
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input_message, run_config=default_config):
        if event.content and event.content.parts[0].text: print(event.content.parts[0].text.strip())

    # Scenario 2: SSE Streaming
    print("
--- Scenario 2: SSE Streaming ---")
    sse_config = RunConfig(streaming_mode=StreamingMode.SSE)
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input_message, run_config=sse_config):
        if event.content and event.content.parts[0].text:
            print(event.content.parts[0].text, end="", flush=True) # Print chunks
    print()

    # Scenario 3: Limiting LLM Calls (Conceptual)
    # This agent doesn't make many calls, but shows the config
    print("
--- Scenario 3: Max LLM Calls (Conceptual) ---")
    # If agent tries more than 1 LLM call, LlmCallsLimitExceededError would be raised by InvocationContext
    # For this simple agent, it will likely make only 1 call.
    limit_config = RunConfig(max_llm_calls=1)
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input_message, run_config=limit_config):
            if event.content and event.content.parts[0].text: print(event.content.parts[0].text.strip())
    except Exception as e: # Catching generic Exception for demo
        print(f"  Caught expected error due to max_llm_calls: {type(e).__name__} - {e}")

    # Scenario 4: Input Blobs as Artifacts (Conceptual)
    print("
--- Scenario 4: Save Input Blobs as Artifacts (Conceptual) ---")
    artifact_config = RunConfig(save_input_blobs_as_artifacts=True)
    # If user_input_message contained a Part with inline_data (e.g., an image),
    # the Runner would save it to the ArtifactService before passing to agent.
    # The agent would see a text part like "Uploaded file: artifact_..."
    # This requires an ArtifactService to be configured with the Runner.
    # runner_with_artifacts = Runner(..., artifact_service=InMemoryArtifactService())
    # await runner_with_artifacts.run_async(..., run_config=artifact_config, new_message=message_with_blob)
    print("  (This would save input blobs to ArtifactService if message contained them and ArtifactService was active)")

    # Scenario 5: Compositional Function Calling (CFC) - Experimental for SSE
    # Requires a model supporting CFC (e.g., Gemini 2.0+ via LIVE API)
    # and BuiltInCodeExecutor or tools that benefit from it.
    # print("
--- Scenario 5: Compositional Function Calling (CFC) via SSE ---")
    # cfc_config = RunConfig(
    #     support_cfc=True,
    #     streaming_mode=StreamingMode.SSE # CFC currently implies SSE via LIVE API usage
    # )
    # An agent using BuiltInCodeExecutor or complex tools would benefit.
    # For this simple agent, it won't show much difference in output.
    # The underlying LLM call mechanism changes to use the LIVE API.
    # print("  (Agent would use LIVE API for potential CFC if tools/code exec were involved)")
    # async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input_message, run_config=cfc_config):
    #     if event.content and event.content.parts[0].text: print(event.content.parts[0].text.strip())

    # Scenario 6: Bidirectional Streaming (BIDI) with Speech & Transcription - Highly Conceptual for CLI
    # This is for `runner.run_live()` and requires actual audio streams.
    # print("
--- Scenario 6: BIDI Streaming with Speech & Transcription (Conceptual for CLI) ---")
    # bidi_config = RunConfig(
    #     streaming_mode=StreamingMode.BIDI,
    #     speech_config=SpeechConfig(
    #         # See google.genai.types.SpeechConfig for options
    #         # Example: engine="chirp_universal", language_codes=["en-US"]
    #     ),
    #     response_modalities=["AUDIO", "TEXT"], # Agent can respond with audio and/or text
    #     output_audio_transcription=AudioTranscriptionConfig(), # Get text of agent's audio
    #     input_audio_transcription=AudioTranscriptionConfig() # Transcribe user's audio input
    # )
    # print(f"  Configured for BIDI: {bidi_config.model_dump_json(indent=2, exclude_none=True)}")
    # To use this:
    # from google.adk.agents.live_request_queue import LiveRequestQueue
    # live_queue = LiveRequestQueue()
    # # In a real app, you'd feed audio chunks to live_queue.send_realtime(Blob(...))
    # async for event in runner.run_live(..., live_request_queue=live_queue, run_config=bidi_config):
    #     # Process events, which might include audio blobs or transcriptions
    # live_queue.close()
    print("  (Actual run_live with BIDI/speech needs real audio input/output handling)")

if __name__ == "__main__":
    asyncio.run(demo_run_configs())
```

**Key `RunConfig` Attributes:**

- **`streaming_mode: StreamingMode`**:
    - `StreamingMode.NONE` (default): Standard request/response.
    - `StreamingMode.SSE` (Server-Sent Events): Enables unidirectional streaming of text responses from the LLM. The `Runner` uses the LLM's streaming generation endpoint.
    - `StreamingMode.BIDI` (Bidirectional): **Experimental**. For live, two-way streaming, typically involving audio. This mode makes the `Runner` use the LLM's `connect()` method and `run_live()`.
- **`speech_config: Optional[types.SpeechConfig]`**: (For BIDI streaming) Configures speech-to-text (STT) and text-to-speech (TTS) engines, language codes, etc., if the agent is interacting via voice.
- **`response_modalities: Optional[list[str]]`**: (For BIDI streaming) Specifies what kind of output the agent can produce (e.g., `["AUDIO", "TEXT"]`).
- **`save_input_blobs_as_artifacts: bool`**: If `True`, any `Part` in the `new_message` that contains `inline_data` (e.g., an image or audio file uploaded by the user) will be automatically saved to the `ArtifactService` by the `Runner` before the agent processes the message. The `Part` in the message passed to the agent will be replaced with a text placeholder like "Uploaded file: artifact\_...".
- **`support_cfc: bool`**: **Experimental**. If `True` (and `streaming_mode` is `SSE`), ADK will attempt to use the LLM's LIVE API endpoint, which may enable Compositional Function Calling (CFC) for models that support it. CFC allows for more complex, nested, or parallel tool calls in a single LLM turn.
- **`output_audio_transcription: Optional[types.AudioTranscriptionConfig]`**: (For BIDI streaming with audio output) If set, requests a text transcription of the agent's spoken audio response.
- **`input_audio_transcription: Optional[types.AudioTranscriptionConfig]`**: (For BIDI streaming with audio input) If set, instructs the LLM (or ADK's internal transcriber if model doesn't support it directly on input) to provide a text transcription of the user's spoken audio.
- **`max_llm_calls: int`**: (Default: 500) A safeguard to prevent runaway loops or excessive LLM interactions within a single `runner.run_async()` invocation. If the number of calls to `llm.generate_content_async()` exceeds this limit, an `LlmCallsLimitExceededError` is raised. Set to `0` or negative to disable the limit.


> ## Best Practice: Use RunConfig for Runtime Flexibility
> 
> RunConfig allows you to change how an agent executes (e.g., streaming vs. non-streaming) without modifying the agent's core definition. This is useful for adapting the same agent logic to different interaction modalities or performance requirements.
> {: .prompt-info }


> ## Experimental Features in RunConfig
> 
> Features like StreamingMode.BIDI and support_cfc are often experimental and their behavior or API might change in future ADK versions. Always check the latest ADK documentation for the status of these features. BIDI streaming, in particular, requires significant application-side logic to handle actual audio input/output.
> {: .prompt-info }

**What's Next?**

We've now thoroughly explored the ADK `Runner` and how `RunConfig` allows for fine-grained control over agent execution. This knowledge is essential for moving your agents from simple local scripts to more robust and interactive applications. Next, we'll focus on "Session Management and State Persistence," diving deep into how ADK handles conversation history and state using different `SessionService` implementations.
