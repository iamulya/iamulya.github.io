---
title: Chapter 24 - Extending and Customizing ADK 
date: "2025-08-22 19:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

The Agent Development Kit (ADK) is designed to be a flexible and extensible framework. While its out-of-the-box components (`LlmAgent`, `InMemoryRunner`, various tools, and services) cover a wide range of use cases, there will be times when you need to customize its core behavior or integrate deeply with proprietary systems. ADK's architecture, built around abstract base classes and well-defined interfaces, makes such extensions possible.

This chapter explores how you can extend ADK by:

- Creating custom `BaseAgent` implementations for non-LLM or specialized logic.
- Developing custom `Runner` logic for unique execution environments.
- Implementing custom versions of `BaseSessionService`, `BaseArtifactService`, and `BaseMemoryService` to integrate with your preferred backend storage or knowledge systems.
- Contributing new, reusable tools or toolsets.
- Potentially adding support for new LLM providers (though this is more involved).

## Creating Custom `BaseAgent` Implementations

While `LlmAgent` is the workhorse for LLM-driven behavior, and shell agents like `SequentialAgent` handle common orchestration patterns, you might need an agent with entirely custom, non-LLM-based logic. This is where subclassing `google.adk.agents.BaseAgent` directly comes into play.

**When to Create a Custom `BaseAgent`:**

- **Rule-Based Agents:** For agents that follow a fixed set of rules or a deterministic state machine without LLM intervention.
- **Integration Agents:** Agents that primarily act as a bridge or facade to a specific legacy system or a non-LLM AI model (e.g., a classical machine learning model).
- **Specialized Orchestrators:** If `SequentialAgent`, `ParallelAgent`, or `LoopAgent` don't quite fit your desired orchestration pattern, you might implement a custom orchestrator agent.
- **Stateful Non-LLM Logic:** An agent that needs to manage complex internal state and transitions without each step being mediated by an LLM.

**Implementing `BaseAgent`:**
You need to override the `_run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]` method. This method contains the core logic of your agent.

```python
from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai.types import Content, Part
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class TrafficLightAgent(BaseAgent):
    """
    A simple rule-based agent that cycles through traffic light states.
    It uses session state to remember its current light.
    """
    def __init__(self, name: str = "traffic_light_controller", description: str = "Manages a traffic light state."):
        super().__init__(name=name, description=description)
        self.light_sequence = ["red", "red-amber", "green", "amber"]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] - Invocation ID: {ctx.invocation_id} - Running custom logic.")

        current_light = ctx.state.get("current_light_color", "red") # Get current or default

        try:
            current_index = self.light_sequence.index(current_light)
            next_index = (current_index + 1) % len(self.light_sequence)
            next_light = self.light_sequence[next_index]
        except ValueError: # Should not happen if state is managed well
            logger.warning(f"[{self.name}] Unknown light state '{current_light}', resetting to red.")
            next_light = "red"

        logger.info(f"[{self.name}] Current light: {current_light}, Next light: {next_light}")

        # Update the state
        actions = EventActions(state_delta={"current_light_color": next_light})

        # Yield an event with the new state
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=Content(parts=[Part(text=f"The traffic light is now: {next_light}")]),
            actions=actions
        )
        # This agent produces one event and finishes its turn.

# --- Example Usage ---
if __name__ == "__main__":
    from google.adk.runners import InMemoryRunner
    import asyncio

    traffic_agent = TrafficLightAgent()
    runner = InMemoryRunner(agent=traffic_agent, app_name="TrafficSystem")
    session_id = "traffic_session_1"

    async def simulate_traffic_cycles(num_cycles: int):
        # Create session with initial state (or let it default)
        await runner.session_service.create_session(
            app_name="TrafficSystem",
            user_id="system_user",
            session_id=session_id,
            state={"current_light_color": "amber"} # Start from amber for first cycle
        )

        for i in range(num_cycles * len(traffic_agent.light_sequence)):
            print(f"\n--- Cycle Trigger {i+1} ---")
            # For a rule-based agent, the input message content might be less important
            # or could be a generic "next_state" trigger.
            trigger_message = Content(parts=[Part(text="Advance light.")])

            async for event in runner.run_async(
                user_id="system_user",
                session_id=session_id,
                new_message=trigger_message
            ):
                if event.author == traffic_agent.name and event.content:
                    print(f"  AGENT [{event.author}]: {event.content.parts[0].text}")
                    current_session = await runner.session_service.get_session(
                        app_name="TrafficSystem", user_id="system_user", session_id=session_id
                    )
                    print(f"    Current actual state: {current_session.state.get('current_light_color')}")
            await asyncio.sleep(0.5) # Pause between cycles

    asyncio.run(simulate_traffic_cycles(2)) # Simulate 2 full light sequences

```

**Key considerations for custom `BaseAgent`s:**

- **State Management:** If your agent is stateful, use `ctx.state` to read and write state variables. Remember to package state changes into `EventActions(state_delta=...)` for the `Event` you yield.
- **Event Generation:** Your agent must `yield` `Event` objects. Even if it doesn't have textual output for the user, it might yield an event that only contains `actions` (e.g., to update state).
- **Asynchronous Nature:** The `_run_async_impl` method is a coroutine. Use `await` for any I/O-bound operations.
- **No LLM by Default:** `BaseAgent` doesn't automatically have an LLM. If you need LLM capabilities, you'd typically use or compose with an `LlmAgent`.


> ## Custom Agents for Deterministic Logic
> 
> BaseAgent subclasses are perfect for implementing parts of your system that require deterministic, rule-based logic that doesn't need the nuanced understanding (or the cost and latency) of an LLM for every step. They can seamlessly integrate into a larger multi-agent system managed by ADK.
> {: .prompt-info }

## Developing Custom `Runner` Logic

While `Runner` and `InMemoryRunner` cover most use cases, you might want to customize the top-level execution loop or how the `Runner` integrates with your specific application environment (e.g., a custom message queue, a different web framework).

To do this, you would typically subclass `google.adk.runners.Runner` and override methods like:

- `run_async(...)`: You could wrap the parent's `run_async`, adding custom setup or teardown logic around it, or even reimplement the core loop if you have very specific needs for how `InvocationContext` is created or how agents are selected.
- `_new_invocation_context(...)`: To customize how the `InvocationContext` is created.
- `_find_agent_to_run(...)`: To change the logic for determining which agent should handle the current request (e.g., based on custom routing rules).
- `_append_new_message_to_session(...)`: To modify how incoming messages are processed or stored before the agent sees them.

```python
from google.adk.runners import Runner
from google.adk.agents import InvocationContext, BaseAgent
from google.adk.sessions import BaseSessionService, Session
from google.genai.types import Content
from google.adk.events.event import Event
from typing import AsyncGenerator, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsEmittingRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_count = 0
        self.total_processing_time = 0.0

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: Content,
        # run_config: RunConfig = RunConfig(), # Keep consistent with parent
        **kwargs # To catch run_config if passed
    ) -> AsyncGenerator[Event, None]:
        import time
        start_time = time.monotonic()
        self.request_count += 1
        logger.info(f"[MetricsEmittingRunner] Starting run_async for request #{self.request_count}")

        # Call the parent's run_async
        async for event in super().run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
            **kwargs # Pass through run_config etc.
        ):
            yield event

        end_time = time.monotonic()
        processing_time = end_time - start_time
        self.total_processing_time += processing_time
        logger.info(f"[MetricsEmittingRunner] Finished run_async for request #{self.request_count}. Time taken: {processing_time:.4f}s")
        logger.info(f"[MetricsEmittingRunner] Total requests: {self.request_count}, Avg time: {self.total_processing_time / self.request_count:.4f}s")

# --- Example Usage ---
# from google.adk.agents import Agent
# from google.adk.sessions import InMemorySessionService
#
# if __name__ == "__main__":
#     import asyncio
#     logging.basicConfig(level=logging.INFO)
#
#     simple_agent = Agent(name="metric_agent", model="gemini-1.5-flash-latest", instruction="Be very quick.")
#     custom_runner = MetricsEmittingRunner(
#         app_name="MetricsApp",
#         agent=simple_agent,
#         session_service=InMemorySessionService()
#     )
#
#     async def run_it():
#         msg = Content(parts=[Part(text="Hello")])
#         async for _ in custom_runner.run_async(user_id="test", session_id="s1", new_message=msg):
#             pass # Just consume events
#         async for _ in custom_runner.run_async(user_id="test", session_id="s2", new_message=msg):
#             pass
#
#     asyncio.run(run_it())

```


> ## Complexity of Custom Runners
> 
> Overriding core Runner logic can be complex and might break expected ADK behaviors if not done carefully. Only do this if you have a strong understanding of ADK's internal execution flow and a compelling reason that cannot be addressed by agent logic, callbacks, or custom services.
> {: .prompt-info }

## Implementing Custom Services (`SessionService`, `ArtifactService`, `MemoryService`)

If ADK's built-in service implementations (`InMemory...`, `Database...`, `Gcs...`, `VertexAiRag...`) don't meet your needs (e.g., you want to use a NoSQL database for sessions, a different cloud storage for artifacts, or a proprietary vector database for memory), you can create your own by subclassing the respective `Base...Service` interface.

**Example: Custom Artifact Service (Conceptual - Storing to Local Filesystem)**

```python
from google.adk.artifacts import BaseArtifactService
from google.genai.types import Part, Blob
import os
import shutil # For deleting directories
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class FileSystemArtifactService(BaseArtifactService):
    def __init__(self, base_storage_path: str = "./adk_local_artifacts"):
        self.base_storage_path = os.path.abspath(base_storage_path)
        os.makedirs(self.base_storage_path, exist_ok=True)
        logger.info(f"FileSystemArtifactService initialized at: {self.base_storage_path}")

    def _get_artifact_path(self, app_name: str, user_id: str, session_id: str, filename: str, version: Optional[int] = None) -> str:
        # Simplified path for this example
        session_path = os.path.join(self.base_storage_path, app_name, user_id, session_id)
        if version is None: # Path to the directory containing all versions of the file
            return os.path.join(session_path, filename)
        else: # Path to a specific version file
            return os.path.join(session_path, filename, str(version))

    async def save_artifact(self, *, app_name: str, user_id: str, session_id: str, filename: str, artifact: Part) -> int:
        versions = await self.list_versions(app_name, user_id, session_id, filename)
        new_version = 0 if not versions else max(versions) + 1

        versioned_file_path = self._get_artifact_path(app_name, user_id, session_id, filename, new_version)
        os.makedirs(os.path.dirname(versioned_file_path), exist_ok=True)

        data_to_write = b""
        mime_type = "application/octet-stream" # Default
        if artifact.text:
            data_to_write = artifact.text.encode('utf-8')
            mime_type = "text/plain"
        elif artifact.inline_data:
            data_to_write = artifact.inline_data.data
            mime_type = artifact.inline_data.mime_type

        with open(versioned_file_path, "wb") as f:
            f.write(data_to_write)
        # Store mime_type alongside, e.g., in a .meta file or by naming convention
        with open(versioned_file_path + ".mimetype", "w") as f_meta:
            f_meta.write(mime_type)

        logger.info(f"Saved artifact '{filename}' version {new_version} to {versioned_file_path}")
        return new_version

    async def load_artifact(self, *, app_name: str, user_id: str, session_id: str, filename: str, version: Optional[int] = None) -> Optional[Part]:
        if version is None:
            versions = await self.list_versions(app_name, user_id, session_id, filename)
            if not versions:
                logger.warning(f"No versions found for artifact '{filename}' in session '{session_id}'.")
                return None
            version = max(versions)

        file_path = self._get_artifact_path(app_name, user_id, session_id, filename, version)
        meta_file_path = file_path + ".mimetype"

        if not os.path.exists(file_path) or not os.path.exists(meta_file_path):
            logger.warning(f"Artifact or metadata for '{filename}' version {version} not found at {file_path}.")
            return None

        with open(file_path, "rb") as f:
            data = f.read()
        with open(meta_file_path, "r") as f_meta:
            mime_type = f_meta.read().strip()

        logger.info(f"Loaded artifact '{filename}' version {version} from {file_path}")
        # For simplicity, assuming all are binary blobs. Text could be special-cased.
        return Part(inline_data=Blob(mime_type=mime_type, data=data))

    async def list_artifact_keys(self, *, app_name: str, user_id: str, session_id: str) -> List[str]:
        session_path = self._get_artifact_path(app_name, user_id, session_id, "") # Gets up to session_id dir
        if not os.path.isdir(session_path):
            return []
        # List directories within session_path, these are filenames
        return [name for name in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, name))]

    async def delete_artifact(self, *, app_name: str, user_id: str, session_id: str, filename: str) -> None:
        artifact_dir_path = self._get_artifact_path(app_name, user_id, session_id, filename)
        if os.path.isdir(artifact_dir_path):
            shutil.rmtree(artifact_dir_path)
            logger.info(f"Deleted artifact directory: {artifact_dir_path}")

    async def list_versions(self, *, app_name: str, user_id: str, session_id: str, filename: str) -> List[int]:
        artifact_dir_path = self._get_artifact_path(app_name, user_id, session_id, filename)
        if not os.path.isdir(artifact_dir_path):
            return []
        versions = []
        for item in os.listdir(artifact_dir_path):
            if item.endswith(".mimetype"): continue # Skip metadata files
            try:
                versions.append(int(item))
            except ValueError:
                logger.warning(f"Non-integer version found: {item} in {artifact_dir_path}")
        return sorted(versions)

# --- Example Usage (Conceptual) ---
# if __name__ == "__main__":
#     from google.adk.runners import Runner
#     from google.adk.sessions import InMemorySessionService
#
#     # fs_artifact_service = FileSystemArtifactService(base_storage_path="./my_agent_files")
#     # my_agent = ... # an agent that uses save_tool and load_tool 
#
#     # runner_with_fs_artifacts = Runner(
#     #     app_name="FileSystemArtifactDemo",
#     #     agent=my_agent,
#     #     session_service=InMemorySessionService(), # Could be DatabaseSessionService
#     #     artifact_service=fs_artifact_service
#     # )
#     # ... then run interactions ...

```


> ## Best Practice: Ensure Asynchronous Operations in Custom Services
> 
> All methods in BaseSessionService, BaseArtifactService, and BaseMemoryService are defined as async def. If your custom implementation involves I/O (network calls, disk access, database queries), ensure you use appropriate asynchronous libraries (e.g., aiohttp for HTTP, asyncpg for PostgreSQL, aiofiles for disk) to avoid blocking ADK's main event loop. For CPU-bound tasks within these methods, consider running them in a separate thread pool using asyncio.to_thread.
> {: .prompt-info }

## Contributing Custom Tools or Toolsets

If you develop a generic, reusable tool or a toolset for a popular API/service that isn't already covered by ADK, consider contributing it back to the ADK community!

**Guidelines for Sharable Tools/Toolsets:**

1. **Inherit from `BaseTool` or `BaseToolset`**.
2. **Clear Naming and Descriptions:** Ensure `tool.name` and `tool.description` (and parameter descriptions) are very clear for LLM consumption.
3. **Well-Defined `FunctionDeclaration`:** For `BaseTool`s (not `FunctionTool`), implement `_get_declaration()` to provide an accurate schema. For `FunctionTool`s, use precise Python type hints and docstrings.
4. **Robust Error Handling:** Tools should gracefully handle potential errors (e.g., API failures, invalid input) and return informative error messages or dictionaries.
5. **Idempotency (if applicable):** If a tool performs an action that has side effects, consider if it can be made idempotent (calling it multiple times with the same input produces the same result without further side effects).
6. **Security:** Follow security best practices, especially regarding input validation and credential handling if the tool makes external calls.
7. **Dependencies:** Clearly list any external Python library dependencies.
8. **Documentation and Examples:** Provide clear documentation and usage examples.
9. **Testing:** Include unit tests.

Refer to ADK's `CONTRIBUTING.md` on GitHub for specific guidelines on code style, testing, and the pull request process.

## Final Thoughts: The Journey of Agent Development

Building truly intelligent, robust, and useful AI agents is a journey, not a destination. The field is dynamic, with new models, techniques, and challenges emerging constantly. Frameworks like Google's ADK provide a solid foundation, emphasizing software engineering principles, modularity, and extensibility to help you navigate this journey.

By understanding the core concepts presented in this book, mastering the tools and techniques, and embracing an iterative development and evaluation cycle, you are well-equipped to:

- Design and implement agents that can reason, plan, and act.
- Empower agents with a diverse range of tools and knowledge sources.
- Orchestrate complex multi-agent systems.
- Deploy and manage your agents effectively.
- And, importantly, contribute to the ongoing evolution of this exciting field.

The future of AI agents is bright, and with tools like ADK, developers are at the forefront of shaping that future. I encourage you to experiment, build, share, and continue learning as this technology unfolds.

*Happy Agent Building!*
