---
title: Chapter 21 - Deploying ADK Agents 
date: "2025-08-22 18:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

You've designed, built, tooled, and rigorously evaluated your ADK agent. Now it's time to take it from your local development environment and make it accessible to users or other systems. This chapter explores strategies and tools for deploying ADK agents, with a primary focus on using the ADK CLI to deploy to Vertex AI Agent Engine, Google Cloud Run, and also touching upon general principles for other deployment targets.

## Packaging Your Agent for Deployment

Before deploying, you need to ensure your agent project is self-contained and includes all necessary dependencies.

**Key Files and Considerations:**

1. **`agent.py` (or your main agent module):** This file must define the `root_agent` instance that will be served.
2. **`requirements.txt` OR `pyproject.toml`:** A file listing all Python dependencies for your project.
    - Ensure it includes `google-adk` and any libraries your custom tools or agents use (e.g., `requests`, `pandas`, `google-cloud-storage`, `sqlalchemy`, `psycopg2-binary` if using `DatabaseSessionService` with PostgreSQL).
3. **Tool Dependencies:** If your agent uses tools that rely on external files (e.g., local data files for a custom RAG tool, OpenAPI spec files if not embedded), ensure these are included in your deployment package or accessible at runtime.
4. **Environment Variables:** Identify all environment variables your agent or its tools require (e.g., `GOOGLE_API_KEY`, database connection strings, GCS bucket names, API keys for external services). These will need to be configured in the deployment environment.
5. **Dockerfile (for containerized deployments):** This is essential for Cloud Run and many other modern deployment platforms. The ADK CLI can help generate a basic one.

**Example Project Structure for Deployment:**

```
my_adk_app/
├── agent.py                                     # Defines root_agent
├── requirements.txt OR pyproject.toml           # Python dependencies
├── Dockerfile                                   # For containerization
├── tools/                                       # Directory for custom tool modules (optional)
│   └── my_custom_tool.py
└── data/                                        # Optional: Data files tools might need
    └── knowledge_base.csv

```


> ## Best Practice: Minimal requirements.txt/pyproject.toml
> 
> Aim for a minimal requirements.txt/pyproject.toml that only includes packages truly needed at runtime. Avoid including development-only dependencies (like pytest, pylint, pyink) in your deployment image to keep it lean and reduce potential vulnerabilities. Use dependency groups in pyproject.toml (if using uv or poetry) to manage dev vs. runtime dependencies.
> {: .prompt-info }

## Deployment Options

ADK agents, being Python applications, can be deployed in various ways:

1. **Vertex AI Agent Engine (Enterprise Grade):**
    - Google's enterprise platform for hosting, managing, and scaling sophisticated AI agents.
    - ADK provides direct support for deploying to Vertex AI Agent Engine via `adk deploy agent_engine`.
2. **Google Cloud Run (Recommended for Serverless):**
    - A fully managed serverless platform that automatically scales your containerized applications up or down, even to zero.
    - ADK provides direct support for deploying to Cloud Run via `adk deploy cloud_run`.
    - Ideal for stateless or stateful (with external backends like Cloud SQL, GCS) web-facing agents.
3. **Other Serverless platforms (e.g., AWS Lambda, Azure Functions):**
    - Your ADK agents can also easily run on any other serverless platforms.
4. **Virtual Machines (e.g., Google Compute Engine, AWS EC2):**
    - Traditional IaaS approach. You manage the OS, Python runtime, and your ADK application process. Offers more control but requires more management.
5. **Kubernetes (e.g., Google Kubernetes Engine - GKE):**
    - For complex, microservice-based agent architectures requiring fine-grained orchestration and scaling control.

Let's start with the deployment to Vertex AI Agent Engine.

## Deploying to Vertex AI Agent Engine

:::{.callout-important}
The support for Vertex AI Agent Engine is currently classified as experimental/in preview and thus can sometimes lead to unpredictable behavior  
:::

Okay, let's create a simple ADK agent and then outline the steps to deploy it to Vertex AI Agent Engine using the `adk deploy` command.

We'll create a directory structure like this:

```
my_simple_echo_agent/
├── .env
├── __init__.py
├── agent.py
├── test_agent.py
└── agent_engine_app.py
```

1.  **`my_simple_echo_agent/.env`**:
    This file will configure the agent to use Vertex AI for its LLM backend (even though our echo agent won't actually call an LLM, this is good practice for ADK agents intended for Vertex AI).

    ```bash
    GOOGLE_GENAI_USE_VERTEXAI=1
    ```

2.  **`my_simple_echo_agent/__init__.py`**:
    This makes the directory a Python package and exposes the `root_agent`.

    ```python
    from .agent import root_agent
    ```

3.  **`my_simple_echo_agent/agent.py`**:
    This defines our simple agent. Instead of an LLM call, we'll use a `before_agent_callback` to construct the response.

    ```python
    from google.adk.agents import LlmAgent # LlmAgent is an alias for Agent
    from google.adk.agents.callback_context import CallbackContext
    from google.genai import types

    def echo_user_input(callback_context: CallbackContext) -> types.Content:
        """
        A simple callback that takes the user's input and echoes it back.
        """
        user_message = ""
        invocation_ctx = getattr(callback_context, '_invocation_context', None)
        if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
            if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
                user_message = invocation_ctx.user_content.parts[0].text

        response_text = f"Echo Agent says: You sent '{user_message}'"
        return types.Content(parts=[types.Part(text=response_text)])

    root_agent = LlmAgent(
        name="simple_echo_agent",
        # Even though we use a callback, a model is usually expected.
        # For Vertex AI, it will try to use this model based on .env settings.
        model="gemini-2.0-flash",
        instruction="You are an echo agent. You repeat what the user says with a prefix.",
        description="A simple agent that echoes user input.",
        before_agent_callback=echo_user_input
    )
    ```
    *Note: The `before_agent_callback` intercepts the flow before any LLM call would happen.*

4.  **`my_simple_echo_agent/agent_engine_app.py`**:
    This file is required by Vertex AI Agent Engine to define the `AdkApp`.

    ```python
    from agent import root_agent # Imports root_agent from agent.py
    from vertexai.preview.reasoning_engines import AdkApp

    # enable_tracing can be True or False depending on your needs
    adk_app = AdkApp(
      agent=root_agent,
      enable_tracing=True,
    )
    ```
>     __CALLOUT_NOTE_START__
>     You could also use `adk create my_simple_echo_agent --model="gemini-2.0-flash" --project="your-gcp-project-id" --region="your-gcp-region"` and then modify `agent.py` to add the callback and remove any default tools if you prefer.
>     :::

**Step 2: Deploy to Vertex AI Agent Engine**

Prerequisites:

1.  **Install `google-adk`**:

    ```bash
    pip install google-adk
    ```

2.  **Google Cloud SDK (`gcloud`)**: Ensure you have `gcloud` installed, configured, and authenticated with a user or service account that has permissions to:
    *   Create/manage Vertex AI Agent Engine.
    *   Read/write to the GCS staging bucket.
    *   Enable necessary APIs (e.g., Vertex AI API, Cloud Build API, Artifact Registry API if a new container image is built).

3.  **GCP Project, Region, and Staging Bucket**:
    *   Have a GCP Project ID.
    *   Choose a supported Vertex AI region (e.g., `us-central1`).
    *   Have a GCS bucket in the same region for staging deployment artifacts. (e.g., `gs://your-adk-staging-bucket`).

Deployment Command:

Navigate to the directory containing `my_simple_echo_agent` (i.e., the parent directory).
Then run the `adk deploy agent_engine` command:

```bash
adk deploy agent_engine \
  --project="your-gcp-project-id" \
  --region="your-gcp-region" \
  --staging_bucket="gs://your-adk-staging-bucket" \
  --adk_app="agent_engine_app" \
  --trace_to_cloud \
  my_simple_echo_agent
```

Let's break down the command:

- `adk deploy agent_engine`: Specifies that we are deploying to Vertex AI Agent Engine.
- `--project="your-gcp-project-id"`: Your Google Cloud Project ID.
- `--region="your-gcp-region"`: The GCP region where you want to deploy (e.g., `us-central1`).
- `--staging_bucket="gs://your-adk-staging-bucket"`: The GCS bucket for storing temporary deployment files. **Ensure this bucket exists and the deploying identity has write permissions.**
- `--adk_app="agent_engine_app"`: This tells ADK to look for `agent_engine_app.py` (and the `adk_app` object within it) in your agent's directory. This corresponds to the `agent_engine_app.py` file we created.
- `--trace_to_cloud`: (Optional) Enables Cloud Trace for telemetry, which is useful for debugging.
- `my_simple_echo_agent`: This is the last argument, specifying the path to your agent's directory.

**What happens during deployment:**

1.  The ADK CLI packages your agent code (from `my_simple_echo_agent/`).
2.  It creates a temporary directory.
3.  It copies your agent code into this temp directory.
4.  It ensures an `agent_engine_app.py` (or the file specified by `--adk_app`) exists and creates one using a template if necessary (in our case, we provided it).
5.  It may create a default `requirements.txt` if one isn't present.
6.  It uses the Vertex AI SDK's `agent_engines.create()` method to deploy your application. This typically involves:
    -  Building a container image (if not using a pre-built one).
    -  Pushing the image to Artifact Registry.
    -  Creating a Agent Engine resource on Vertex AI.


**After Deployment:**

- You'll get a resource name for your deployed Agent Engine in the terminal output.
- You can then find and manage your Agent Engine in the Google Cloud Console under Vertex AI -> Agent Engine.
- You can interact with it using the Vertex AI SDK or by calling its HTTP endpoint (if configured).

**Testing:**
Now you can test your agent programatically using Vertex AI SDK

```python
from google.cloud import aiplatform
from vertexai.preview import reasoning_engines

# --- Configuration ---
PROJECT_ID = "<your-gcp-project-id>"  # Replace with your Project ID
LOCATION = "<your-gcp-region>"      # Replace with the region of your Agent Engine
RESOURCE_NUMERIC_ID = "<numeric_id_from_full_resource_name>" # The numeric ID part from the full resource name
REASONING_ENGINE_ID = "projects/{}/locations/{}/reasoningEngines/{}".format(
    PROJECT_ID,
    LOCATION,
    RESOURCE_NUMERIC_ID 
)

# --- Initialize Vertex AI SDK ---
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# --- Get the Reasoning Engine instance ---
# This represents your deployed ADK application
try:
    engine = reasoning_engines.ReasoningEngine(REASONING_ENGINE_ID)
    print(f"Successfully connected to Reasoning Engine: {engine.resource_name}")
except Exception as e:
    print(f"Error connecting to Reasoning Engine: {e}")
    exit()


APP_NAME_FOR_QUERY = "my_simple_echo_agent" # This should match your agent's folder name
USER_ID = "test-user-sdk"

async def test_agent():

    try:
        remote_session = engine.create_session(user_id=USER_ID, app_name=APP_NAME_FOR_QUERY)

        for event_dict in engine.stream_query(
            user_id=USER_ID,
            session_id=remote_session["id"],
            message="whats the weather in new york",
        ):
                # You can deserialize these back into ADK Event objects if needed,
                # but for simple display, printing the dict is fine.
                print(f"  Event Author: {event_dict.get('author')}")
                if event_dict.get("content") and event_dict["content"].get("parts"):
                    for part in event_dict["content"]["parts"]:
                        if part.get("text"):
                            print(f"    Text: {part['text']}")
                        if part.get("functionCall"):
                            print(f"    Function Call: {part['functionCall']['name']}({part['functionCall']['args']})")
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
```

This provides a complete, albeit simple, end-to-end example of creating an ADK agent and deploying it to Vertex AI Agent Engine. The key is the `agent_engine_app.py` file that bridges your ADK agent with the `AdkApp` class expected by the Agent Engine.

## Deploying to Google Cloud Run

The `adk deploy cloud_run` command significantly simplifies deploying your ADK agent as a containerized application on Cloud Run.

**Prerequisites for Cloud Run Deployment:**

1. **Google Cloud SDK (`gcloud` CLI) installed and authenticated:**
    
    ```bash
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project YOUR_GCP_PROJECT_ID
    
    ```
    
2. **APIs Enabled in your GCP Project:**
    - Cloud Run API (`run.googleapis.com`)
    - Artifact Registry API (`artifactregistry.googleapis.com`) (to store your Docker images)
    - Cloud Build API (`cloudbuild.googleapis.com`) (used by `gcloud run deploy` to build the image)
    You can enable these via the Cloud Console or `gcloud services enable ...`.
3. **Docker installed locally** (if you want `gcloud run deploy` to build the image locally first, though Cloud Build can also build remotely).
4. **Permissions:** Your authenticated user or service account needs roles like "Cloud Run Admin" (`roles/run.admin`), "Artifact Registry Writer" (`roles/artifactregistry.writer`), and "Cloud Build Editor" (`roles/cloudbuild.buildEditor`) or "Service Account User" on the Cloud Build service account.

**The `adk deploy cloud_run` command:**

Basic Syntax: `adk deploy cloud_run <agent_module_path> --service-name <your-service-name> --region <gcp-region> [options]`

**Key Steps Performed by the Command (Simplified):**

1. **Generates a Dockerfile:** If one doesn't exist in your project, it creates a basic Dockerfile suitable for running an ADK agent (using Uvicorn and FastAPI to serve the agent).
2. **Packages Application:** Prepares your agent code and dependencies.
3. **Invokes `gcloud run deploy`:** This `gcloud` command handles:
    - Building the Docker image (either locally using your Docker or remotely using Cloud Build).
    - Pushing the image to Google Artifact Registry.
    - Deploying the image as a new Cloud Run service or updating an existing one.
    - Configuring the service with specified settings (region, environment variables, etc.).

**Example Deployment:**

Let's deploy the `my_simple_echo_agent` we created earlier for Vertex AI Agent Engine deployment.

```bash
# Replace with your desired service name and region
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
SERVICE_NAME="simple-echo-service"
APP_NAME="simple-echo-app"
GCP_REGION="us-central1" # e.g., us-central1, europe-west1
AGENT_PATH="./my_simple_echo_agent"

adk deploy cloud_run \
--project=$GOOGLE_CLOUD_PROJECT \
--region=$GOOGLE_CLOUD_LOCATION \
--service_name=$SERVICE_NAME \
--app_name=$APP_NAME \
--with_ui \
$AGENT_PATH
```
- The command will prompt you for several confirmations if it's the first time or if APIs need enabling.
- It will then build the image, push it to Artifact Registry, and deploy the service.
- Upon successful deployment, it will output the **Service URL**.


> To deploy with the Web UI, include the `--with_ui` flag in your `adk deploy cloud_run` command. Use this option only in development or testing environments, as the Web UI is **not intended for production use**. Ideally you would have your own frontend which will communicate with the ADK agent through the API server (You can use the `adk api_server` command to run the ADK agent backend), which we will discuss in the next secion.
> {: .prompt-info }

Once deployed, you'll get a URL like `https://simple-echo-service-xxxxxx.region.a.run.app` where you can **access the Web UI**. This provides the easiest way to test your deployment.

![Cloud Run demo](echo.png)

You can also interact with this deployed agent by sending HTTP POST requests to its `/run_sse` endpoint as well.

**Example using `curl`:**

```bash
export TOKEN=$(gcloud auth print-identity-token)
export APP_URL=<your-service-url> #something like `https://simple-echo-service-xxxxxx.region.a.run.app`

curl -X POST -H "Authorization: Bearer $TOKEN" \
    $APP_URL/apps/$APP_NAME/users/echo_user/sessions/session_echo \
    -H "Content-Type: application/json" \
    -d '{"state": {"preferred_language": "English"}'

curl -X POST -H "Authorization: Bearer $TOKEN" \
    $APP_URL/run_sse \
    -H "Content-Type: application/json" \
    -d '{
    "app_name": $APP_NAME,
    "user_id": "echo_user",
    "session_id": "session_echo",
    "new_message": {
        "role": "user",
        "parts": [{
        "text": "Who are you?"
        }]
    },
    "streaming": false
    }'

```


> ## Best Practice: Use Secret Manager for Sensitive Data
> 
> Do not pass API keys or database passwords directly as environment variables in the deploy command for production. Instead:
>  
> 1. Store secrets in [Google Cloud Secret Manager](https://cloud.google.com/secret-manager).
> 2. Grant the Cloud Run service's runtime service account permission to access these secrets (e.g., "Secret Manager Secret Accessor" role).
> 3. In your `agent.py` or initialization code, fetch these secrets at startup using the Secret Manager client libraries.
> 
> The `adk deploy cloud_run` command has options (`-set-secrets`) to help integrate with Secret Manager.
> {: .prompt-info }

**Using Persistent Services with Cloud Run:**
If your agent uses `DatabaseSessionService`, `GcsArtifactService`, or `VertexAiRagMemoryService`:

- **Database:**
    - Use Google Cloud SQL (PostgreSQL or MySQL).
    - Ensure your Cloud Run service can connect to your Cloud SQL instance (e.g., by configuring a [Cloud SQL connector](https://cloud.google.com/sql/docs/mysql/connect-run) or VPC Serverless Access Connector).
    - Pass the database connection string as an environment variable (preferably via Secret Manager).
- **GCS for Artifacts:**
    - The Cloud Run service's runtime service account needs permissions to read/write to the specified GCS bucket.
    - Pass the bucket name as an environment variable.
- **Vertex AI RAG for Memory:**
    - The runtime service account needs permissions for Vertex AI and RAG operations.
    - Pass the RAG Corpus ID as an environment variable.


> ## `adk_version` in `adk deploy cloud_run`
> 
> The adk deploy cloud_run command allows specifying `--adk_version desired_version`. By default, it uses the version of ADK you have installed locally when generating the Dockerfile. If you need to pin the ADK version in your deployed container to a specific release for stability, use this option.
> {: .prompt-info }

## Other Deployment targets

You can use the `adk api_server` command to run the ADK agent backend *anywhere*. The `adk api_server` command is specifically designed to run the ADK backend as a **standalone FastAPI application**, without serving the ADK Web UI. This facilitates a decoupled architecture where your frontend (which could be the ADK Web UI hosted separately, a custom web application, or any other client) interacts with the ADK backend via its defined API endpoints.

**How `adk api_server` Enables Backend-Frontend Separation:**

1.  **Dedicated Backend Service:**
    When you run `adk api_server <AGENTS_DIR> [OPTIONS...]`, it starts a Uvicorn server hosting a FastAPI application. This application exposes RESTful API endpoints for managing and interacting with your ADK agents.
    *   It loads agents from the specified `<AGENTS_DIR>`.
    *   It handles session management (in-memory, database, or Vertex AI).
    *   It processes agent runs, tool executions, and artifact management.
    *   **Crucially, it does *not* serve any static frontend files.** This is the key difference from the `adk web` command, which serves both the backend API and the ADK Web UI from the same origin.

2.  **Independent Frontend:**
    Your frontend application (e.g., a React, Angular, Vue.js app, or even a separate instance of the ADK Web UI deployed elsewhere) will run on its own server and origin (e.g., `http://localhost:3000` for local development, or a different domain in production).

3.  **API-Based Communication:**
    The frontend communicates with the `adk api_server` backend by making HTTP requests (GET, POST, DELETE, WebSocket for live mode) to the backend's API endpoints (e.g., `/list-apps`, `/apps/{app_name}/users/{user_id}/sessions`, `/run_sse`, `/run_live`, etc.).

**Avoiding CORS (Cross-Origin Resource Sharing) Issues:**

When your frontend (e.g., at `http://localhost:3000`) tries to make requests to your backend API server (e.g., at `http://localhost:8000`), the browser's Same-Origin Policy (SOP) comes into play. If the origins are different, the browser will block the request unless the server explicitly allows it via CORS headers.

The `adk api_server` command and the underlying FastAPI application provide a mechanism to handle this:

1.  **`--allow_origins` Option:**
    The most important option for managing CORS is `--allow_origins`. This option is passed directly to FastAPI's `CORSMiddleware`.
    - You can specify one or more origins that are permitted to make requests to your ADK API server.
    - For example, if your frontend development server runs on `http://localhost:3000`, you would run the API server like this:

        ```bash
        adk api_server /path/to/your/agents_dir --port 8000 --host 127.0.0.1 --allow_origins http://localhost:3000
        ```

    - In a production environment, you would replace `http://localhost:3000` with the actual domain of your deployed frontend application (e.g., `https://my-frontend-app.com`).
    - You can specify multiple origins by repeating the option: `--allow_origins http://localhost:3000 --allow_origins https://dev.example.com`.


> ## Importance of `--allow_origins`
> 
> The `--allow_origins` flag is essential for a decoupled frontend/backend setup. Without correctly configuring it, your frontend application will be unable to communicate with the `adk api_server` due to browser security restrictions (CORS errors).
> {: .prompt-info }

**Example Scenario:**

- **Backend (ADK API Server):**

    ```bash
    adk api_server ./my_agents \
        --host 0.0.0.0 \
        --port 8000 \
        --allow_origins "http://localhost:3000" "https://my-custom-frontend.com" \
        --session_db_url "sqlite:///./adk_sessions.db"
    ```

    This starts the ADK backend on port 8000, listening on all network interfaces, allowing requests from your local frontend development server and your production frontend.

- **Frontend (e.g., a React app running on `localhost:3000`):**

    Your frontend code would make API calls to `http://localhost:8000` (or `https://your-backend-domain.com` in production). For instance, you can create a session like this:

    ```javascript
    // Example frontend JavaScript
    fetch('http://localhost:8000/apps/my_chat_app/users/user123/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: { initial_context: "Welcome!" } })
    })
    .then(response => response.json())
    .then(data => console.log('Session created:', data))
    .catch(error => console.error('Error creating session:', error));
    ```
    
    Because `http://localhost:3000` is in `--allow_origins`, the browser will permit this request.

**Key Command-Line Options for `adk api_server`:**

- `AGENTS_DIR` (argument): Path to the directory containing your agent definitions.
- `--host`: The network interface to bind the server to (default: `127.0.0.1`). Use `0.0.0.0` to make it accessible from other machines on your network.
- `--port`: The port number for the server (default: `8000`).
- `--session_db_url`: For persistent session storage (e.g., `sqlite:///./sessions.db` or a PostgreSQL/MySQL URL). If not provided, sessions are in-memory.
- `--artifact_storage_uri`: For persistent artifact storage (e.g., `gs://your-bucket-name`). If not provided, artifacts are in-memory.
- `--allow_origins`: As discussed, crucial for enabling cross-origin requests from your frontend.
- `--trace_to_cloud`: Enables exporting traces to Google Cloud Trace.
- `--reload`: Enables auto-reloading the server on code changes (useful for development, default is `True`).


> ## Difference from `adk web`
> 
> The `adk web` command starts a similar FastAPI backend but *also* serves the ADK's built-in web UI from the same origin. This is convenient for local development and testing within a single process. `adk api_server`, on the other hand, *only* runs the API, expecting the frontend to be served independently. This is the standard approach for building scalable and maintainable web applications.
> {: .prompt-info }

By using `adk api_server` and correctly configuring `--allow_origins`, you can effectively separate your ADK backend logic from your frontend presentation layer, allowing for independent development, deployment, and scaling of both components while managing browser security policies for cross-origin communication.

**What's Next?**

Successfully deploying your ADK agent makes it accessible and useful. We've covered the powerful `adk deploy cloud_run` command and touched upon principles for other deployment scenarios. The subsequent chapters will focus on "Advanced Topics and Best Practices," starting with telemetry, logging, and effective debugging strategies for your ADK agents.
