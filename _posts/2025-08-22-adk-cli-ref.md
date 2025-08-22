---
title: Appendix B - ADK CLI Reference
date: "2025-08-22 20:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

The Agent Development Kit (ADK) Command Line Interface (CLI) provides a suite of tools to help you create, run, evaluate, and deploy your AI agents. This reference details the available commands and their options.

**Global Options:**

-   `--help`: Show help message for any command and exit.
-   `--version`: Show the version of the ADK and exit.

------------------------------------------------------------------------

### `adk create`

Creates a new agent application in the current folder with a prepopulated template.

**Usage:**

``` bash
adk create [OPTIONS] APP_NAME
```

**Arguments:**

-   `APP_NAME` (STR, Required): The name of the application folder to be created for the agent's source code.

**Options:**

-   `--model TEXT`: (Optional) The model to be used for the root agent (e.g., `gemini-2.0-flash-001`). If not provided, you will be prompted.
-   `--api_key TEXT`: (Optional) The API Key needed to access the model, e.g., Google AI API Key. If not provided, you might be prompted depending on the chosen model/backend.
-   `--project TEXT`: (Optional) The Google Cloud Project ID for using Vertex AI as the backend. If not provided, you might be prompted or it might try to infer from `gcloud` config.
-   `--region TEXT`: (Optional) The Google Cloud Region for using Vertex AI as the backend. If not provided, you might be prompted or it might try to infer from `gcloud` config.

**Example:**

``` bash
adk create my_search_agent --model gemini-2.0-flash-001 --api_key YOUR_API_KEY
adk create my_vertex_agent --project my-gcp-project --region us-central1
```

------------------------------------------------------------------------

### `adk run`

Runs an interactive CLI for a specified agent, allowing you to test and interact with it locally.

**Usage:**

``` bash
adk run [OPTIONS] AGENT
```

**Arguments:**

-   `AGENT` (PATH, Required): The path to the agent's source code folder.

**Options:**

-   `--save_session`: (FLAG, Optional, Default: False) If set, saves the session to a JSON file on exit.
-   `--session_id TEXT`: (Optional) The session ID to use when saving the session if `--save_session` is true. If not set, you will be prompted.
-   `--replay PATH`: (Optional, Exclusive with `--resume`) The path to a JSON file containing the initial state of the session and user queries. A new session will be created using this state, and user queries will be run against it. Interaction is not possible after replay.
-   `--resume PATH`: (Optional, Exclusive with `--replay`) The path to a JSON file containing a previously saved session (using `--save_session`). The previous session will be re-displayed, and you can continue to interact with the agent.

**Example:**

``` bash
adk run path/to/my_agent
adk run path/to/my_agent --save_session --session_id my_test_run
adk run path/to/my_agent --replay path/to/input_queries.json
```

------------------------------------------------------------------------

### `adk eval`

Evaluates an agent using one or more evaluation sets.

**Usage:**

``` bash
adk eval [OPTIONS] AGENT_MODULE_FILE_PATH [EVAL_SET_FILE_PATH]...
```

**Arguments:**

-   `AGENT_MODULE_FILE_PATH` (PATH, Required): The path to the agent's source code folder (specifically, the directory containing `__init__.py` which imports `agent.py` where `root_agent` is defined).
-   `EVAL_SET_FILE_PATH...` (PATH, Required, Multiple): One or more paths to evaluation set JSON files (`.evalset.json`).
    -   To run specific evals from a set: `path/to/file.evalset.json:eval_id1,eval_id2`

**Options:**

-   `--config_file_path PATH`: (Optional) The path to a JSON configuration file for evaluation criteria.
-   `--print_detailed_results`: (FLAG, Optional, Default: False) If set, prints detailed results for each evaluation case to the console.

**Example:**

``` bash
adk eval path/to/my_agent path/to/my_agent/eval_set_01.evalset.json
adk eval path/to/my_agent path/to/set1.evalset.json:case_a,case_b path/to/set2.evalset.json
```

------------------------------------------------------------------------

### `adk web`

Starts a FastAPI server with a Web UI for developing and testing agents.

**Usage:**

``` bash
adk web [OPTIONS] [AGENTS_DIR]
```

**Arguments:**

-   `AGENTS_DIR` (PATH, Optional, Default: Current Directory): The directory containing agent application folders. Each sub-directory is treated as a single agent application.

**Options:**

-   `--session_db_url TEXT`: (Optional) The database URL to store session data.
    -   Examples: `agentengine://<agent_engine_resource_id>`, `sqlite:///./my_sessions.db`
    -   Refer to SQLAlchemy documentation for more DB URL formats.
-   `--artifact_storage_uri TEXT`: (Optional) The URI for artifact storage.
    -   Example: `gs://<bucket_name>` for Google Cloud Storage.
-   `--host TEXT`: (Optional, Default: `127.0.0.1`) The host address to bind the server to.
-   `--port INTEGER`: (Optional, Default: `8000`) The port number for the server.
-   `--allow_origins TEXT`: (Optional, Multiple) Additional origins to allow for CORS. Can be specified multiple times.
-   `--log_level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: (Optional, Default: `INFO`) Set the logging level.
-   `--trace_to_cloud`: (FLAG, Optional, Default: False) If set, enables Cloud Trace for telemetry.
-   `--reload / --no-reload`: (FLAG, Optional, Default: True) Enable/disable auto-reloading of the server on code changes.

**Example:**

``` bash
adk web path/to/my_agents_directory --port 8080
adk web --session_db_url sqlite:///./adk_sessions.db
```

------------------------------------------------------------------------

### `adk api_server`

Starts a FastAPI server providing API endpoints for agents (without the Web UI).

**Usage:**

``` bash
adk api_server [OPTIONS] [AGENTS_DIR]
```

**Arguments:**

-   `AGENTS_DIR` (PATH, Optional, Default: Current Directory): The directory containing agent application folders.

**Options:**

-   (Same options as `adk web`: `--session_db_url`, `--artifact_storage_uri`, `--host`, `--port`, `--allow_origins`, `--log_level`, `--trace_to_cloud`, `--reload / --no-reload`)

**Example:**

``` bash
adk api_server path/to/my_agents_directory --port 8001
```

------------------------------------------------------------------------

### `adk deploy`

A group of commands for deploying agents to hosted environments.

**Usage:**

``` bash
adk deploy [COMMAND] [OPTIONS]
```

#### `adk deploy cloud_run`

Deploys an agent application to Google Cloud Run.

**Usage:**

``` bash
adk deploy cloud_run [OPTIONS] AGENT
```

**Arguments:**

-   `AGENT` (PATH, Required): The path to the agent's source code folder.

**Options:**

-   `--project TEXT`: (Required) Google Cloud project ID to deploy the agent. If absent, uses the default from `gcloud` config.
-   `--region TEXT`: (Required) Google Cloud region to deploy the agent. If absent, `gcloud run deploy` will prompt.
-   `--service_name TEXT`: (Optional, Default: `adk-default-service-name`) The service name to use in Cloud Run.
-   `--app_name TEXT`: (Optional, Default: Agent folder name) The application name for the ADK API server.
-   `--port INTEGER`: (Optional, Default: `8000`) The port for the ADK API server.
-   `--trace_to_cloud`: (FLAG, Optional, Default: False) If set, enables Cloud Trace.
-   `--with_ui`: (FLAG, Optional, Default: False) If set, deploys the ADK Web UI alongside the API server.
-   `--temp_folder PATH`: (Optional, Default: System temp directory with timestamp) Temporary folder for generated Cloud Run source files.
-   `--verbosity [debug|info|warning|error|critical]`: (Optional, Default: `WARNING`) Overrides the default verbosity level for `gcloud run deploy`.
-   `--session_db_url TEXT`: (Optional) The database URL for session storage in Cloud Run.
-   `--artifact_storage_uri TEXT`: (Optional) The URI for artifact storage in Cloud Run (e.g., `gs://<bucket_name>`).
-   `--adk_version TEXT`: (Optional, Default: Current ADK version) The ADK version to install in the Cloud Run container.

**Example:**

``` bash
adk deploy cloud_run path/to/my_agent --project my-gcp-project --region us-central1 --service_name my-agent-service
```

#### `adk deploy agent_engine`

Deploys an agent application to Vertex AI Agent Engine.

**Usage:**

``` bash
adk deploy agent_engine [OPTIONS] AGENT
```

**Arguments:**

-   `AGENT` (PATH, Required): The path to the agent's source code folder.

**Options:**

-   `--project TEXT`: (Required) Google Cloud project ID.
-   `--region TEXT`: (Required) Google Cloud region.
-   `--staging_bucket TEXT`: (Required) GCS bucket for staging deployment artifacts (e.g., `gs://my-staging-bucket`).
-   `--trace_to_cloud`: (FLAG, Optional, Default: False) If set, enables Cloud Trace for Agent Engine.
-   `--adk_app TEXT`: (Optional, Default: `agent_engine_app`) Python file name (without `.py`) for defining the ADK application.
-   `--temp_folder PATH`: (Optional, Default: System temp directory with timestamp) Temporary folder for generated Agent Engine source files. Contents will be replaced if the folder exists.
-   `--env_file PATH`: (Optional, Default: `.env` in agent folder) Path to the `.env` file for environment variables.
-   `--requirements_file PATH`: (Optional, Default: `requirements.txt` in agent folder) Path to the `requirements.txt` file.

**Example:**

``` bash
adk deploy agent_engine path/to/my_agent --project my-gcp-project --region us-central1 --staging_bucket gs://my-agent-engine-staging
```
