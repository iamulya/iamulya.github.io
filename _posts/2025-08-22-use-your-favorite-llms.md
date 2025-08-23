---
title: Chapter 10 - Use your favorite LLMs 
date: "2025-08-22 12:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

At the heart of every `LlmAgent` in the Agent Development Kit is a Large Language Model (LLM). The agent's ability to understand, reason, generate text, and decide on actions is fundamentally driven by its interactions with this LLM. This chapter delves into how ADK abstracts and manages these interactions, covering the `BaseLlm` interface, specific model integrations, the structure of LLM requests and responses, and how to configure LLM behavior for optimal agent performance.

## The `BaseLlm` Interface and the LLM Registry

To support a variety of LLMs, ADK defines a common interface: `google.adk.models.BaseLlm`. All specific LLM integrations in ADK inherit from this abstract base class.

**Key Methods of `BaseLlm`:**

- **`model: str`**: An attribute storing the specific model identifier (e.g., `"gemini-2.0-flash"`).
- **`supported_models() -> list[str]` (classmethod)**: Returns a list of regular expressions that match the model names supported by this LLM implementation. This is used by the `LLMRegistry`.
- **`generate_content_async(llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]`**: The primary asynchronous method for sending a request to the LLM and receiving one or more `LlmResponse` objects. If `stream` is `True`, it yields partial responses.
- **`connect(llm_request: LlmRequest) -> BaseLlmConnection`**: An experimental method for establishing a live, bidirectional streaming connection to the LLM, returning a `BaseLlmConnection` object.

**`google.adk.models.LLMRegistry`**:
ADK maintains a global registry that maps model name patterns (defined by `supported_models()`) to their corresponding `BaseLlm` subclasses. When you create an `LlmAgent` and provide a model name as a string (e.g., `Agent(model="gemini-2.0-flash", ...)`), ADK uses this registry to:

1. Find the `BaseLlm` class that supports the given model name.
2. Instantiate that class (e.g., `Gemini(model="gemini-2.0-flash")`).

This allows ADK to seamlessly support different LLM providers and models without requiring the agent developer to always instantiate the specific LLM client class.

## Broad Model Support with `LiteLlm`

For maximum flexibility with a wide range of LLM providers (OpenAI, Azure OpenAI, Anthropic, Cohere, Hugging Face, etc.), local models via Ollama, and even self-hosted endpoints, ADK offers the `google.adk.models.LiteLlm` class. This class is a wrapper around the popular [LiteLLM library](https://litellm.ai/), which acts as a unified interface to over 100 LLM APIs.

**Prerequisites:**

- `litellm` Python library installed (`pip install litellm` or via `google-adk[extensions]`).
- Appropriate API keys for the desired cloud LLM provider set as environment variables (e.g., `OPENAI_API_KEY` for OpenAI models).
- For Ollama, ensure Ollama is installed and running with the desired models pulled.
- For self-hosted endpoints, ensure your model serving solution (like TGI or vLLM) is running and accessible.

```python
from google.adk.agents import Agent
# Ensure litellm is installed: pip install google-adk[extensions] or pip install litellm
try:
    from google.adk.models.lite_llm import LiteLlm # Key import
    LITELLM_AVAILABLE = True
except ImportError:
    print("LiteLLM library not found. Please install it ('pip install litellm').")
    LITELLM_AVAILABLE = False

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

openai_agent = None
if LITELLM_AVAILABLE:
    # Requires OPENAI_API_KEY environment variable to be set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. LiteLLM OpenAI example may fail.")

    try:
        # Specify the model using litellm's naming convention (e.g., "openai/gpt-4o")
        openai_llm_instance = LiteLlm(model="openai/gpt-4o")# For Azure OpenAI: "azure/your-deployment-name"
                                                          # For Cohere: "cohere/command-r"
                                                          # etc.

        openai_agent = Agent(
            name="openai_gpt_assistant",
            model=openai_llm_instance,
            instruction="You are a helpful assistant powered by an OpenAI model via LiteLLM."
        )
        print("OpenAI GPT agent (via LiteLLM) initialized.")
    except Exception as e:
        print(f"Error initializing LiteLlm agent: {e}")
else:
    print("Skipping LiteLLM example as the library is not available.")


if __name__ == "__main__":
    if openai_agent:
        runner = InMemoryRunner(agent=openai_agent, app_name="LiteLLM_OpenAI_App")
        session_id = "s_openai"
        user_id = "openai_user"
        create_session(runner, user_id=user_id, session_id=session_id)

        user_message = Content(parts=[Part(text="Write a short poem about Python programming.")], role="user")  
        print("\nOpenAI GPT Agent (via LiteLLM):")
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(part.text, end="")
        print()
```

`LiteLlm` translates ADK requests to the format `litellm` expects and converts `litellm`'s responses back to ADK's `LlmResponse`.


> ## Best Practice: Consistent Model Naming with LiteLLM
> 
> Refer to the litellm documentation for the correct model name strings for various providers (e.g., "openai/gpt-4", "azure/my-deployment", "huggingface/meta-llama/Llama-2-7b-chat-hf"). Ensure the necessary API keys are set as environment variables as per litellm's requirements.
> {: .prompt-info }

### Using Local Models with Ollama via `LiteLlm`

[Ollama](https://ollama.ai/) allows you to run open-source large language models locally. LiteLLM provides seamless integration with a running Ollama instance.

**Prerequisites for Ollama:**

1. Install Ollama from [ollama.ai](https://ollama.ai/).
2. Pull the models you want to use (e.g., `ollama pull llama3`, `ollama pull mistral`).
3. Ensure the Ollama server is running (it usually starts automatically after installation). By default, it serves on `http://localhost:11434`.

```python
from google.adk.agents import Agent
try:
    from google.adk.models.lite_llm import LiteLlm
    LITELLM_AVAILABLE = True
except ImportError: 
    LITELLM_AVAILABLE = False

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import requests # To check if Ollama is running

ollama_agent = None
if LITELLM_AVAILABLE:
    # Check if Ollama server is accessible
    ollama_running = False
    try:
        # Default Ollama API endpoint
        response = requests.get("<http://localhost:11434>")
        if response.status_code == 200 and "Ollama is running" in response.text:
            ollama_running = True
            print("Ollama server detected.")
        else:
            print("Ollama server found but not responding as expected.")
    except requests.exceptions.ConnectionError:
        print("Ollama server not found at <http://localhost:11434>. Please ensure Ollama is installed and running.")

    if ollama_running:
        try:
            # For Ollama, prefix the model name with "ollama/"
            # This assumes you have 'llama3' model pulled via `ollama pull llama3`
            ollama_llm_instance = LiteLlm(model="ollama/llama3")
            # You can also specify the full API base if Ollama runs elsewhere:
            # ollama_llm_instance = LiteLlm(model="ollama/llama3", api_base="<http://my-ollama-server:11434>")

            ollama_agent = Agent(
                name="local_llama3_assistant",
                model=ollama_llm_instance,
                instruction="You are a helpful assistant running locally via Ollama and Llama 3."
            )
            print("Ollama Llama3 agent (via LiteLLM) initialized.")
        except Exception as e:
            print(f"Error initializing LiteLlm agent for Ollama: {e}")
            print("Ensure you have pulled the model (e.g., 'ollama pull llama3').")
else:
    print("Skipping LiteLLM Ollama example as LiteLLM library is not available.")

if __name__ == "__main__":
    if ollama_agent:
        runner = InMemoryRunner(agent=ollama_agent, app_name="LiteLLM_Ollama_App")
        user_message = Content(parts=[Part(text="Why is the sky blue? Explain briefly.")])
        print("\nLocal Llama3 Agent (via LiteLLM and Ollama):")
        for event in runner.run(user_id="ollama_user", session_id="s_ollama", new_message=user_message):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(event.content.parts[0].text, end="")
        print()
    else:
        print("Ollama agent not run due to setup issues.")
```


> ## Local Development and Experimentation with Ollama
> 
> - **Cost-effective development:** No API costs for local model inference.
> - **Offline capabilities:** Run agents without internet access (once models are downloaded).
> - **Privacy:** Data doesn't leave your machine for inference.
> - **Rapid experimentation:** Quickly test different open-source models.
> {: .prompt-info }


> ## Ollama Model Performance
> 
> Not all models pulled via Ollama might support all features LiteLLM or ADK expect (e.g., complex tool calling might be less reliable with some smaller local models).
> {: .prompt-info }

### Using Self-Hosted Endpoints via `LiteLlm`

Many organizations deploy open-source models using serving solutions like Text Generation Interface (TGI) from Hugging Face, vLLM, or custom FastAPI servers. `LiteLlm` can connect to these self-hosted OpenAI-compatible API endpoints.

To do this, you typically need to provide the `api_base` for your self-hosted endpoint and often set a dummy API key (as LiteLLM might expect one even if your endpoint doesn't use it).

```python
from google.adk.agents import Agent
try:
    from google.adk.models.lite_llm import LiteLlm
    LITELLM_AVAILABLE = True
except ImportError: 
    LITELLM_AVAILABLE = False

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os
import requests

from building_intelligent_agents.utils import create_session

# Assume you have a self-hosted LLM (e.g., using TGI or vLLM)
# serving an OpenAI-compatible API at this base URL.
SELF_HOSTED_API_BASE = os.getenv("MY_SELF_HOSTED_LLM_API_BASE", "<http://localhost:8000/v1>") # Example for local TGI
# The model name here is often arbitrary for self-hosted endpoints but might be
# used by LiteLLM or the endpoint itself for routing if it serves multiple models.
# For TGI/vLLM, it's often just the "model" you want to hit at that endpoint.
SELF_HOSTED_MODEL_NAME = os.getenv("MY_SELF_HOSTED_LLM_MODEL_NAME", "custom/my-model")

self_hosted_agent = None
if LITELLM_AVAILABLE:
    # Check if the self-hosted endpoint is accessible
    endpoint_running = False
    try:
        # A simple check - a real check might query a /health or /v1/models endpoint
        response = requests.get(SELF_HOSTED_API_BASE.replace("/v1", "/health") if "/v1" in SELF_HOSTED_API_BASE else SELF_HOSTED_API_BASE) # Common health check
        if response.status_code == 200:
            endpoint_running = True
            print(f"Self-hosted endpoint detected at {SELF_HOSTED_API_BASE}.")
        else:
            print(f"Self-hosted endpoint at {SELF_HOSTED_API_BASE} responded with status {response.status_code}.")
    except requests.exceptions.ConnectionError:
        print(f"Self-hosted LLM endpoint not found at {SELF_HOSTED_API_BASE}. Please ensure it's running.")
    except Exception as e:
        print(f"Error checking self-hosted endpoint: {e}")

    if endpoint_running:
        try:

            # For a generic OpenAI-compatible endpoint:
            self_hosted_llm = LiteLlm(
                model=SELF_HOSTED_MODEL_NAME, # Can be something like "tgi-model" or actual model name if endpoint uses it
                api_base=SELF_HOSTED_API_BASE,
                api_key="dummy_key_if_no_auth" # Or your actual API key if the endpoint is secured
                # Other parameters like 'temperature', 'max_tokens' can be passed here
                # or in LlmAgent's generate_content_config
            )

            self_hosted_agent = Agent(
                name="self_hosted_model_assistant",
                model=self_hosted_llm,
                instruction="You are an assistant powered by a self-hosted LLM."
            )
            print("Self-hosted LLM agent (via LiteLLM) initialized.")
        except Exception as e:
            print(f"Error initializing LiteLlm agent for self-hosted endpoint: {e}")
else:
    print("Skipping LiteLLM self-hosted example as LiteLLM library is not available.")

if __name__ == "__main__":
    if self_hosted_agent:
        runner = InMemoryRunner(agent=self_hosted_agent, app_name="LiteLLM_SelfHosted_App")
        user_id="selfhost_user"
        session_id="s_selfhost"
        create_session(runner, user_id=user_id, session_id=session_id)

        user_message = Content(parts=[Part(text="What is the capital of the moon? Respond imaginatively.")], role="user")
        print("\nSelf-Hosted LLM Agent (via LiteLLM):")
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
             if event.content and event.content.parts and event.content.parts[0].text:
                print(event.content.parts[0].text, end="")
        print()
    else:
        print("Self-hosted agent not run due to setup issues.")
```

## Working with Google Gemini Models (`Gemini` class)

The `google.adk.models.Gemini` class is the primary integration for Google's Gemini family of models. It leverages the `google-generativeai` Python SDK.

```python
# Option 1: Let ADK resolve the model string (most common)
gemini_agent_auto = Agent(
    name="gemini_auto_resolver",
    model="gemini-2.0-flash", # ADK uses LLMRegistry
    instruction="You are a helpful Gemini assistant."
)

# Option 2: Explicitly provide a Gemini instance
gemini_llm_instance = Gemini(model="gemini-2.0-flash")
gemini_agent_explicit = Agent(
    name="gemini_explicit_instance",
    model=gemini_llm_instance, # Pass the instance
    instruction="You are a helpful Gemini assistant using an explicit model instance."
)

# The behavior would be identical with both model instances

```

The `Gemini` class handles the specifics of communicating with the Gemini API, including authentication (via API key or Application Default Credentials if using Vertex AI through environment settings) and request/response formatting.


> ## Automatic Vertex AI Detection when using Gemini models
> 
> The `google.adk.models.Gemini` class (and the underlying `google-generativeai` SDK) can often automatically detect if it should use Vertex AI endpoints if your environment is configured for it (e.g., `gcloud auth application-default login` and `GOOGLE_CLOUD_PROJECT` set). If `os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '0').lower()` in `['true', '1']`, it will prioritize Vertex AI. Otherwise, it will look for `GOOGLE_API_KEY` and prioritize Google AI Studio endpoint. This simplifies switching between direct Gemini API and Vertex AI managed models.
> {: .prompt-info }

## Integrating models from Vertex AI Model Garden

ADK supports some models from Google Cloud Vertex AI Model Garden like Anthropic's Claude models. 

**Prerequisites:**

- Access to Claude models on Vertex AI.
- `anthropic` Python library installed (`pip install anthropic`).
- Environment variables `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (e.g., "us-central1") must be set.

```python
from google.adk.agents import Agent
from google.adk.models.anthropic_llm import Claude # Import Claude
from google.adk.models.registry import LLMRegistry
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# Register Claude model with the LLMRegistry
# This is necessary to ensure the ADK can find and use the Claude model.
LLMRegistry.register(Claude)

# Requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION to be set
# and access to Claude models on Vertex AI for the project.
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") # e.g., "us-central1"
CLAUDE_MODEL_NAME = "claude-sonnet-4@20250514" # Example Claude model on Vertex AI

claude_agent = None
if GCP_PROJECT and GCP_LOCATION:
    try:
        claude_agent = Agent(
            name="claude_assistant",
            model=CLAUDE_MODEL_NAME,
            instruction="You are a helpful and thoughtful assistant powered by Claude. Provide comprehensive answers."
        )
        print(f"Claude agent initialized with model {CLAUDE_MODEL_NAME} on Vertex AI.")
    except ImportError:
        print("Anthropic SDK not found. Please install it ('pip install anthropic google-cloud-aiplatform').")
    except Exception as e:
        print(f"Could not initialize Claude model on Vertex AI: {e}")
        print("Ensure your project has access and GOOGLE_CLOUD_PROJECT/LOCATION are set.")
else:
    print("GOOGLE_CLOUD_PROJECT and/or GOOGLE_CLOUD_LOCATION not set. Skipping Claude example.")


if __name__ == "__main__":
    if claude_agent:
        runner = InMemoryRunner(agent=claude_agent, app_name="ClaudeApp")
        session_id = "s_claude"
        user_id = "claude_user"
        create_session(runner, user_id=user_id, session_id=session_id)

        user_message = Content(parts=[Part(text="Explain the concept of emergent properties in complex systems.")], role="user")
        print("
Claude Agent:")
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(part.text, end="")
        print()
```


> ## Model Naming for Claude on Vertex AI
> 
> When using Claude models via Vertex AI, the model name string needs to be the specific identifier Vertex AI uses (e.g., "claude-sonnet-4@20250514"). Check the Vertex AI documentation for the correct model IDs.
> {: .prompt-info }

## Configuring LLM Requests (`LlmRequest`)

The `google.adk.models.LlmRequest` Pydantic model is the standardized way ADK structures data sent *to* an LLM. The LLM Flow (e.g., `SingleFlow`) in an `LlmAgent` is responsible for constructing this object.

Key components of `LlmRequest`:

- **`model: Optional[str]`**: The specific model string.
- **`contents: list[types.Content]`**: This is the conversation history. It's a list of `google.genai.types.Content` objects. Each `Content` object has a `role` (`"user"` or `"model"`) and `parts` (a list of `Part` objects, which can be text, function calls, function responses, or inline data).
    - The history is ordered chronologically.
    - For models that support alternating user/model turns, ADK ensures this structure.
- **`config: Optional[types.GenerateContentConfig]`**: As discussed previously, this holds:
    - `system_instruction: Optional[str]`: The compiled system prompt (agent instruction + global instruction + tool-provided instructions like from `PreloadMemoryTool`).
    - `tools: Optional[list[types.Tool]]`: A list of `types.Tool` objects, where each `Tool` contains `FunctionDeclaration`s for the tools available to the LLM.
    - Generation parameters like `temperature`, `max_output_tokens`, `safety_settings`.
    - `response_schema`: If structured JSON output is expected.
- **`live_connect_config: types.LiveConnectConfig`**: Configuration specific to live, bidirectional streaming (speech config, response modalities, etc.).
- **`tools_dict: dict[str, BaseTool]`**: (Internal to ADK) A mapping of tool names to their actual `BaseTool` instances, used by the LLM Flow to execute the correct tool when the LLM requests a function call.

The LLM Flow and various request processors (like `instructions.py`, `contents.py`, `functions.py`) work together to populate these fields based on the agent's definition, the session history, and available tools.

## Interpreting LLM Responses (`LlmResponse`)

The `google.adk.models.LlmResponse` Pydantic model standardizes the data received *from* an LLM.

Key components of `LlmResponse`:

- **`content: Optional[types.Content]`**: The main payload from the LLM.
    - If the LLM provides text, it will be in `content.parts[0].text`.
    - If the LLM requests a tool call, it will be in `content.parts[0].function_call`.
    - The `content.role` will typically be `"model"`.
- **`partial: Optional[bool]`**: If `True`, this `LlmResponse` represents a chunk of a streaming text response. The full text is an accumulation of these partial responses.
- **`usage_metadata: Optional[types.GenerateContentResponseUsageMetadata]`**:
    - `prompt_token_count: int`
    - `candidates_token_count: int`
    - `total_token_count: int`
    This is crucial for monitoring costs and understanding model usage.
- **`grounding_metadata: Optional[types.GroundingMetadata]`**: If the LLM used grounding (e.g., via `GoogleSearchTool` or `VertexAiSearchTool`), this contains information about the search queries performed and sources used.
- **`error_code: Optional[str]`, `error_message: Optional[str]`**: If the LLM call failed.
- **`turn_complete: Optional[bool]`**: For live streaming, indicates if the model has finished its turn.
- **`interrupted: Optional[bool]`**: For live streaming, indicates if the user interrupted the model.
- **`custom_metadata: Optional[dict[str, Any]]`**: A field you can use (e.g., in an `after_model_callback`) to attach your own arbitrary, JSON-serializable data to an `LlmResponse`.

The LLM Flow converts these `LlmResponse` objects into ADK `Event` objects, which are then yielded by the `Runner`.


> ## Inspecting LlmResponse in Callbacks and Traces
> 
> - The `after_model_callback` in an `LlmAgent` receives the raw `LlmResponse`. This is an excellent place to log detailed information like `usage_metadata`, inspect `grounding_metadata`, or modify the response before ADK processes it further.
> - The Dev UI's Trace view will show the details of `LlmRequest`s and `LlmResponse`s for each LLM interaction, which is invaluable for debugging.
> {: .prompt-info }

## Streaming Responses

ADK supports streaming responses from LLMs that offer this capability. When an `LlmAgent` is run with a `RunConfig` that enables streaming (e.g., `RunConfig(streaming_mode=StreamingMode.SSE)`), the `BaseLlm` implementation's `generate_content_async` method is called with `stream=True`.

- Instead of a single `LlmResponse`, the method yields an `AsyncGenerator[LlmResponse, None]`.
- Each yielded `LlmResponse` typically contains a chunk of the text, and its `partial` attribute will be `True`.
- The final `LlmResponse` in the stream will have `partial=False` (or it might be a response that only contains a function call, or the `finish_reason` from the underlying SDK indicates completion).
- The ADK `Runner` yields these partial `Event` objects, allowing your application to display text to the user as it's being generated.

```python
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner, RunConfig # Import RunConfig
from google.adk.agents.run_config import StreamingMode # Import StreamingMode
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

streaming_demo_agent = Agent(
    name="streaming_writer",
    model=DEFAULT_LLM,
    instruction="Write a very short story (10-15 sentences) about a curious cat."
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=streaming_demo_agent, app_name="StreamingApp")
    session_id = "s_streaming"
    user_id = "streaming_user"
    create_session(runner, user_id=user_id, session_id=session_id)

    user_message = Content(parts=[Part(text="Tell me a story.")], role="user") # Prompt for the agent

    async def run_with_streaming():
        print("Streaming Agent Response:")
        # Configure the runner to use SSE (Server-Sent Events) streaming
        run_config = RunConfig(streaming_mode=StreamingMode.SSE)

        async for event in runner.run_async( # Use run_async for streaming
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
            run_config=run_config # Pass the run_config
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True) # Print chunks as they arrive
        print("\n--- End of Stream ---")

    asyncio.run(run_with_streaming())
```

When you run this, you'll see the story appear word by word or sentence by sentence, rather than all at once.


> ## Best Practice: Use Streaming for Better UX
> 
> For conversational agents, streaming responses significantly improves the user experience by providing immediate feedback instead of making the user wait for the entire response to be generated. Enable it via RunConfig when calling runner.run_async.
> {: .prompt-info }

**What's Next?**

We've now covered how ADK interacts with various LLMs, from request construction to response interpretation and streaming. This understanding is vital as LLMs are the "brains" of our `LlmAgent`s. Next we'll explore how to orchestrate more complex agent behaviors using ADK's LLM Flows and Planners, enabling agents to create and follow multi-step plans to achieve their goals.
