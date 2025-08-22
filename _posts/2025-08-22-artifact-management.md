---
title: Chapter 18 - Artifact Management 
date: "2025-08-22 16:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

While session state is excellent for storing structured data and conversational context, agents often need to work with larger, more complex data like files – images, documents, spreadsheets, code outputs, or multimedia. ADK's **Artifact Management** system provides a dedicated mechanism for agents to save, load, and manage these files, known as "artifacts."

This chapter explores the `BaseArtifactService` interface, its common implementations (`InMemoryArtifactService` and `GcsArtifactService`), and how agents and tools can interact with artifacts to enrich their capabilities.

## The `BaseArtifactService` Interface

Similar to the `BaseSessionService`, the `google.adk.artifacts.BaseArtifactService` is an abstract base class defining the contract for storing and retrieving artifacts. This abstraction allows your agent logic to remain independent of the specific artifact storage backend.

**Key Abstract Methods of `BaseArtifactService`:**

- **`save_artifact(app_name, user_id, session_id, filename, artifact: types.Part) -> int`**: Saves an artifact (provided as a `google.genai.types.Part` object, typically containing `inline_data`) to the storage. It returns an integer representing the version of the saved artifact (starting from 0).
- **`load_artifact(app_name, user_id, session_id, filename, version=None) -> Optional[types.Part]`**: Loads a specific version of an artifact. If `version` is `None`, it loads the latest version. Returns a `types.Part` object or `None` if not found.
- **`list_artifact_keys(app_name, user_id, session_id) -> list[str]`**: Lists the filenames of all artifacts currently stored for a given session.
- **`delete_artifact(app_name, user_id, session_id, filename)`**: Deletes all versions of a named artifact.
- **`list_versions(app_name, user_id, session_id, filename) -> list[int]`**: Lists all available versions for a specific artifact file.

The `Runner` is typically configured with an `ArtifactService` instance, which then becomes accessible to agents and tools via their respective contexts (`CallbackContext`, `ToolContext`).

![*Diagram: `BaseArtifactService` interface and its concrete implementations.*](/assets/img/2025-08-22-artifact-management/figure-1.png)


## Use Cases for Artifacts

Artifacts are useful in various scenarios:

- **Storing LLM Outputs:** If an LLM generates an image, a piece of code, a Markdown document, or a structured data file (like CSV or JSON) that's too large or unsuitable for direct inclusion in a chat message.
- **Outputs from Code Execution:** Code run by a `CodeExecutor` (like `VertexAiCodeExecutor` or `ContainerCodeExecutor`) can produce files (plots, data files), which can be saved as artifacts.
- **User Uploads:** When a user uploads a file to the agent system, it can be stored as an artifact for the agent to process (see `RunConfig.save_input_blobs_as_artifacts`).
- **Intermediate Tool Outputs:** A tool might generate a complex file that another tool or a later agent turn needs to consume.
- **Persistent Records:** Storing generated reports, logs, or important documents related to a session.

## `InMemoryArtifactService`

The `google.adk.artifacts.InMemoryArtifactService` stores artifacts in a Python dictionary in memory.

- **Pros:**
    - No external setup required.
    - Fast for local development and testing.
    - Automatically used by `InMemoryRunner`.
- **Cons:**
    - Artifacts are lost when the Python process ends.
    - Not suitable for production or large artifacts due to memory constraints.

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part, Blob
import asyncio # For async tool function

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

# Tool to create and save an artifact
async def create_and_save_text_artifact(filename: str, content_text: str, tool_context: ToolContext) -> dict:
    """Creates a text artifact and saves it using the artifact service."""
    print(f"  [Tool] Creating artifact '{filename}' with content: '{content_text}'")
    # A Part can be created from text, bytes, or a Blob
    artifact_part = Part(text=content_text) # Or Part(inline_data=Blob(mime_type="text/plain", data=content_text.encode()))

    version = await tool_context.save_artifact(filename=filename, artifact=artifact_part)
    return {"filename_saved": filename, "version": version, "status": "success"}

save_tool = FunctionTool(func=create_and_save_text_artifact)

# Tool to load an artifact
async def load_text_artifact_content(filename: str, tool_context: ToolContext) -> dict:
    """Loads a text artifact and returns its content."""
    print(f"  [Tool] Attempting to load artifact '{filename}'...")
    artifact_part = await tool_context.load_artifact(filename=filename) # Loads latest version by default
    if artifact_part and artifact_part.text:
        return {"filename_loaded": filename, "content": artifact_part.text, "status": "success"}
    elif artifact_part and artifact_part.inline_data: # If saved as bytes
        return {"filename_loaded": filename, "content": artifact_part.inline_data.data.decode(), "status": "success"}
    return {"filename_loaded": filename, "error": "Artifact not found or not text.", "status": "failure"}

load_tool = FunctionTool(func=load_text_artifact_content)

artifact_agent = Agent(
    name="artifact_handler_agent",
    model=DEFAULT_LLM,
    instruction="You can create text files (artifacts) and later read them. "
                "Use 'create_and_save_text_artifact' to save content. "
                "Use 'load_text_artifact_content' to read content from a previously saved file.",
    tools=[save_tool, load_tool]
)

if __name__ == "__main__":
    # InMemoryRunner uses InMemoryArtifactService by default
    runner = InMemoryRunner(agent=artifact_agent, app_name="InMemoryArtifactApp")
    session_id = "s_artifact_mem"
    user_id = "mem_artifact_user"
    create_session(runner, user_id=user_id, session_id=session_id)  # Create a session for the user

    async def main():
        # Turn 1: Create an artifact
        prompt1 = "Please create a file named 'notes.txt' with the content 'ADK is great for building agents.'"
        print(f"\n--- Turn 1 --- \nYOU: {prompt1}")
        user_message1 = Content(parts=[Part(text=prompt1)], role="user")  # User message to the agent
        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(user_id="mem_artifact_user", session_id=session_id, new_message=user_message1):
            if event.content and event.content.parts[0].text and not event.get_function_calls():
                print(event.content.parts[0].text.strip())

        # Verify artifact was saved (InMemoryArtifactService specific inspection)
        # In a real app, the agent would confirm or you'd check via another tool/Dev UI.
        saved_artifacts = await runner.artifact_service.list_artifact_keys(
            app_name="InMemoryArtifactApp", user_id="mem_artifact_user", session_id=session_id
        )
        print(f"  [DEBUG] Artifacts in session '{session_id}': {saved_artifacts}")
        assert "notes.txt" in saved_artifacts

        # Turn 2: Load the artifact
        prompt2 = "Now, please read the content of 'notes.txt'."
        print(f"\n--- Turn 2 --- \nYOU: {prompt2}")
        user_message2 = Content(parts=[Part(text=prompt2)], role="user")  # User message to the agent
        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(user_id="mem_artifact_user", session_id=session_id, new_message=user_message2):
            if event.content and event.content.parts[0].text and not event.get_function_calls():
                print(event.content.parts[0].text.strip())

    asyncio.run(main())
```


> ## `types.Part` for Artifact Content
> 
> Artifacts are saved and loaded as google.genai.types.Part objects. This allows you to store various types of content:
> 
> - `Part(text="...")` for plain text.
> - `Part(inline_data=Blob(mime_type="image/png", data=image_bytes))` for binary data like images.
> - `Part(inline_data=Blob(mime_type="application/pdf", data=pdf_bytes))` for other file types.
> 
> The `mime_type` is important for correct interpretation when loaded.
> {: .prompt-info }

## `GcsArtifactService`: Storing Artifacts in Google Cloud Storage

For persistent and scalable artifact storage, ADK provides `google.adk.artifacts.GcsArtifactService`. This service stores artifacts as objects in a specified Google Cloud Storage (GCS) bucket.

**Prerequisites:**

- Google Cloud Project with billing enabled.
- Google Cloud Storage API enabled.
- A GCS bucket created.
- Authentication configured for ADK to access GCS (e.g., `gcloud auth application-default login` for local development, or a service account with "Storage Object Admin" or finer-grained permissions on the bucket when deployed).
- `google-cloud-storage` library installed (`pip install google-cloud-storage`).

**Initialization:**

```python
from google.adk.artifacts import GcsArtifactService
import os

GCS_BUCKET_NAME_FOR_ADK = os.getenv("ADK_ARTIFACT_GCS_BUCKET")

if GCS_BUCKET_NAME_FOR_ADK:
    try:
        gcs_artifact_service_instance = GcsArtifactService(
            bucket_name=GCS_BUCKET_NAME_FOR_ADK
        )
        print(f"GcsArtifactService initialized for bucket: {GCS_BUCKET_NAME_FOR_ADK}")

        # This instance can now be passed to a Runner:
        # from google.adk.runners import Runner
        # from google.adk.sessions import DatabaseSessionService # (Example)
        # my_agent = ...
        # db_url = "sqlite:///./my_gcs_app_sessions.db"
        # runner_with_gcs = Runner(
        #     app_name="MyGCSApp",
        #     agent=my_agent,
        #     session_service=DatabaseSessionService(db_url=db_url),
        #     artifact_service=gcs_artifact_service_instance
        # )
    except Exception as e:
        print(f"Failed to initialize GcsArtifactService: {e}")
        print("Ensure 'google-cloud-storage' is installed, GCS bucket exists, and auth is configured.")
else:
    print("ADK_ARTIFACT_GCS_BUCKET environment variable not set. Skipping GCS ArtifactService setup.")
```

**Object Naming Convention in GCS:**`GcsArtifactService` stores artifacts using a path structure within the bucket:
`{app_name}/{user_id}/{session_id}/{filename}/{version}`
For example: `MyChatApp/user123/sessionABC/report.pdf/0`

If a filename starts with `user:`, like `user:profile_picture.png`, it's stored under a user-global path:
`{app_name}/{user_id}/user/{filename_without_prefix}/{version}`
Example: `MyChatApp/user123/user/profile_picture.png/0` (This allows sharing an artifact across multiple sessions for the same user).

**Using `GcsArtifactService` with an Agent (Conceptual):**
The agent code itself (like `artifact_agent` in the `InMemoryArtifactService` example) doesn't need to change. The difference is how the `Runner` is initialized.

```python
# Conceptual usage with GCS
# ... (define artifact_agent, save_tool, load_tool as in InMemory example) ...
#
# if GCS_BUCKET_NAME_FOR_ADK and os.getenv("GOOGLE_CLOUD_PROJECT"):
#     gcs_service = GcsArtifactService(bucket_name=GCS_BUCKET_NAME_FOR_ADK)
#     # Using InMemorySessionService here for simplicity, could be DatabaseSessionService
#     persistent_runner = Runner(
#         app_name="PersistentArtifactApp",
#         agent=artifact_agent,
#         session_service=InMemorySessionService(), # Or DatabaseSessionService
#         artifact_service=gcs_service
#     )
#     # ... now use persistent_runner.run_async(...) ...
#     # Artifacts will be saved to/loaded from your GCS bucket.
# else:
#     print("Skipping GCS runner example.")

```


> ## Best Practice: GcsArtifactService for Production
> 
> For any application requiring persistent artifact storage, scalability, and integration with other Google Cloud services, GcsArtifactService is the recommended choice. GCS offers durability, versioning (though ADK handles its own version numbers in the path), and fine-grained access control.
> {: .prompt-info }


> ## GCS Permissions and Costs
> 
> - Ensure the service account or user credentials used by your ADK application have the necessary IAM permissions on the GCS bucket (e.g., `roles/storage.objectAdmin` for full control, or more restricted roles like `roles/storage.objectCreator` and `roles/storage.objectViewer`).
> - Storing large or numerous artifacts in GCS will incur costs. Monitor your usage.
> {: .prompt-info }

## Using the `LoadArtifactsTool`

The `google.adk.tools.load_artifacts_tool` (an instance of `LoadArtifactsTool`) provides a way for the LLM to become aware of and request the content of existing artifacts within the current session.

**How it Works:**

1. **Awareness Phase (Request Processor):**
    - Before an `LlmRequest` is sent, `LoadArtifactsTool.process_llm_request()` checks the `ArtifactService` (via `tool_context.list_artifacts()`) for available artifact filenames.
    - If artifacts exist, it appends a system instruction to the `LlmRequest` like:
        
        ```
        You have a list of artifacts: ["notes.txt", "image_plot.png"]
        When the user asks questions about any of the artifacts, you should call the
        `load_artifacts` function to load the artifact. Do not generate any text other
        than the function call.
        
        ```
        
    - It also adds the `load_artifacts` function declaration to the tools available to the LLM for that turn.
2. **LLM Requests Artifact(s):**
    - If the LLM decides it needs the content of, say, `"notes.txt"`, it will generate a function call: `load_artifacts(artifact_names=["notes.txt"])`.
3. **Content Injection Phase (Request Processor - Next Turn Preprocessing):**
    - When ADK processes this function call to `load_artifacts`:
        - The `LoadArtifactsTool.run_async()` method simply returns the `artifact_names` it received. This tool doesn't directly return the file content in its *own* execution result.
        - In the *subsequent* LLM turn's preprocessing phase, `process_llm_request()` detects that the previous LLM turn was a call to `load_artifacts`.
        - It then uses `tool_context.load_artifact()` to fetch the actual content of `"notes.txt"` from the `ArtifactService`.
        - This content (as a `types.Part` with `inline_data`) is then appended to the *new* `LlmRequest`'s `contents` list, often with a preceding text part like `Content(role='user', parts=[Part(text='Artifact notes.txt is:'), loaded_artifact_part])`.
4. **LLM Uses Content:** The LLM now receives the actual artifact content as part of its conversational history and can use it to answer the user's query.

```python
from google.adk.agents import Agent
from google.adk.tools import load_artifacts
from google.adk.tools import FunctionTool, ToolContext 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part, Blob
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

# Tool to create and save an artifact (same as before)
async def create_image_artifact(filename: str, tool_context: ToolContext) -> dict:
    """Creates a dummy PNG image artifact."""
    print(f"  [Tool] Creating dummy image artifact '{filename}'")
    # Dummy PNG (1x1 transparent pixel)
    dummy_png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    artifact_part = Part(inline_data=Blob(mime_type="image/png", data=dummy_png_data))
    version = await tool_context.save_artifact(filename=filename, artifact=artifact_part)
    return {"filename_saved": filename, "version": version, "status": "success"}

save_image_tool = FunctionTool(func=create_image_artifact)

# Agent using LoadArtifactsTool
artifact_viewer_agent = Agent(
    name="artifact_viewer",
    model=DEFAULT_LLM, 
    instruction="You can create image artifacts and later view them. "
                "If artifacts are listed as available, and the user asks about one, "
                "use the 'load_artifacts' function to get its content before describing it.",
    tools=[save_image_tool, load_artifacts] # Add load_artifacts_tool
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=artifact_viewer_agent, app_name="ArtifactViewerApp")
    session_id = "s_artifact_viewer"
    user_id = "viewer_user"
    create_session(runner, user_id=user_id, session_id=session_id)  # Create a session for the user

    async def main():
        # Turn 1: Create an image artifact
        prompt1 = "Please create a dummy image named 'logo.png'."
        print(f"\n--- Turn 1: Create Artifact --- \nYOU: {prompt1}")
        user_message1 = Content(parts=[Part(text=prompt1)], role="user")  # User message to the agent
        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message1):
            if event.content and event.content.parts[0].text and not event.get_function_calls():
                print(event.content.parts[0].text.strip())

        # Turn 2: Ask about the artifact. `load_artifacts_tool` will inform the LLM it exists.
        # LLM should then call `load_artifacts`.
        prompt2 = "Describe the 'logo.png' artifact you created."
        print(f"\n--- Turn 2: Ask to Describe Artifact --- \nYOU: {prompt2}")
        user_message2 = Content(parts=[Part(text=prompt2)], role="user")  # User message to the agent
        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message2):
            # For CLI, we'll just see the final answer.
            # Dev UI Trace would show:
            # - Sys Instruction: "Artifacts available: ['logo.png']..."
            # - LLM calls: load_artifacts(artifact_names=['logo.png'])
            # - Event: Tool response from load_artifacts (just echoes names)
            # - Next LLM call's history includes: User: Artifact logo.png is: <Part with inline_data>
            # - LLM final response: "The artifact 'logo.png' is an image file..."
            if event.content and event.content.parts[0].text and not event.get_function_calls():
                print(event.content.parts[0].text.strip())

        # Check the artifact service to see the "logo.png"
        # loaded_artifact_part = await runner.artifact_service.load_artifact("ArtifactViewerApp", "viewer_user", session_id, "logo.png")
        # assert loaded_artifact_part is not None
        # assert loaded_artifact_part.inline_data.mime_type == "image/png"

    asyncio.run(main())
```


> ## Two-Step Artifact Access with LoadArtifactsTool
> 
> The two-step nature of LoadArtifactsTool (awareness then content loading on demand) is efficient. It prevents large artifact contents from being added to the prompt history unnecessarily on every turn, only loading them when the LLM explicitly requests them after being made aware of their existence.
> {: .prompt-info }

## Saving User-Uploaded Files as Artifacts 

Previously we introduced `RunConfig`. One of its options, `save_input_blobs_as_artifacts: bool`, directly ties into artifact management.

If you initialize your `Runner.run_async()` call with `RunConfig(save_input_blobs_as_artifacts=True)`, and the `new_message: types.Content` contains any `Part` with `inline_data` (e.g., user uploads an image or a PDF), the `Runner` will:

1. Iterate through these parts before invoking the agent.
2. For each part with `inline_data`, it calls `artifact_service.save_artifact()`, generating a unique filename (e.g., `artifact_<invocation_id>_<part_index>`).
3. It then **replaces** that original `Part` in the `new_message` with a simple `Part(text=f"Uploaded file: artifact_.... It is saved into artifacts")`.
4. The agent then receives this modified message.

The agent can subsequently use `LoadArtifactsTool` (or a custom tool using `tool_context.load_artifact()`) to access the content of these automatically saved artifacts.

```python
from google.adk.agents import Agent
from google.adk.tools import load_artifacts # To access it later
from google.adk.runners import InMemoryRunner, RunConfig # Import RunConfig
from google.genai.types import Content, Part, Blob
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

upload_processor_agent = Agent(
    name="upload_processor",
    model=DEFAULT_LLM,
    instruction="You will receive information about uploaded files. If asked about an uploaded file, use 'load_artifacts' to get its content and then describe it.",
    tools=[load_artifacts]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=upload_processor_agent, app_name="UploadApp")
    session_id = "s_upload_test"
    user_id = "upload_user"
    create_session(runner, user_id=user_id, session_id=session_id)  # Create a session for the user

    # Create a message with an inline_data part (simulating a user upload)
    dummy_text_data = "This is the content of my uploaded file."
    user_uploaded_file_part = Part(inline_data=Blob(mime_type="text/plain", data=dummy_text_data.encode()))
    user_query_part = Part(text="I've uploaded a file. Can you tell me what it is?")
    message_with_upload = Content(parts=[user_query_part, user_uploaded_file_part])

    # Configure RunConfig to save blobs
    config_save_blobs = RunConfig(save_input_blobs_as_artifacts=True)

    async def main():
        print(f"\n--- Running with save_input_blobs_as_artifacts=True ---")
        print(f"Original user message parts: {len(message_with_upload.parts)}")
        print(f"YOU (conceptually with upload): {user_query_part.text}")

        print("AGENT: ", end="", flush=True)
        async for event in runner.run_async(
            user_id="upload_user",
            session_id=session_id,
            new_message=message_with_upload,
            run_config=config_save_blobs
        ):
            # The agent will first see "Uploaded file: artifact_..."
            # Then it should call load_artifacts for that artifact name.
            # Then it will respond based on the content.
            if event.content and event.content.parts[0].text and not event.get_function_calls():
                print(event.content.parts[0].text.strip())

        # Verify the artifact was saved by the Runner
        artifacts = await runner.artifact_service.list_artifact_keys(app_name="UploadApp", user_id=user_id, session_id=session_id)
        print(f"  [DEBUG] Artifacts created by Runner: {artifacts}")
        assert len(artifacts) == 1

        # Verify content
        # loaded_artifact = await runner.artifact_service.load_artifact("UploadApp", "upload_user", session_id, artifacts[0])
        # assert loaded_artifact.inline_data.data.decode() == dummy_text_data

    asyncio.run(main())
```


> ## Best Practice: save_input_blobs_as_artifacts for User Files
> 
> This RunConfig option is the standard way to handle file uploads from users in ADK. It cleanly separates the act of receiving and storing the file from the agent's logic for processing it, promoting modularity.
> {: .prompt-info }

**What's Next?**

We've now explored how ADK manages files through its Artifact Service, enabling agents to work with diverse data types beyond simple text. This capability is crucial for many real-world applications. In the following chapter, "Long-Term Memory for Agents," we will look at how agents can retain knowledge and context across different sessions using ADK's Memory Service, moving beyond the transient nature of session state and artifacts for truly persistent learning and recall.
