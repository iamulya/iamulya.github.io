---
title: Appendix C - ADK Troubleshooting Guide
date: "2025-08-22 21:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

This guide provides solutions and tips for common issues encountered when working with the Google Agent Development Kit (ADK) for Python.

------------------------------------------------------------------------

### Agent Definition and Configuration Issues

**Issue: `ValueError: Agent name cannot be 'user'` or `ValueError: Found invalid agent name ... Agent name must be a valid identifier`**

-   **Cause:** The `name` attribute of an `Agent` or `BaseAgent` is either set to the reserved word "user" or contains invalid characters (e.g., spaces, hyphens).
-   **Solution:**
    -   Choose a different name if it's "user".
    -   Ensure the agent name is a valid Python identifier (starts with a letter or underscore, followed by letters, numbers, or underscores).

**Issue: `ValueError: Agent '...' already has a parent agent...`**

-   **Cause:** You are trying to add the same `Agent` instance as a `sub_agent` to multiple parent agents, or multiple times to the same parent. An agent can only have one parent.
-   **Solution:** If you need the same agent logic in multiple places, create separate instances of that agent, even if their configuration is identical (ensure they have unique `name`s).

**Issue: `ValueError: No model found for 'agent_name'` (when `model` is not set on an `LlmAgent`)**

-   **Cause:** An `LlmAgent` (or an agent in its hierarchy) does not have a `model` specified, and there's no `model` defined in any of its ancestor agents.
-   **Solution:**
    -   Specify the `model` (e.g., `"gemini-2.0-flash"`) directly in the `LlmAgent` definition.
    -   Alternatively, ensure that a parent agent in its hierarchy has a `model` defined, which will then be inherited.

**Issue: `ValueError: Invalid config for agent ...: output_schema cannot co-exist with agent transfer configurations.` or `...if output_schema is set, sub_agents must be empty...` or `...tools must be empty.`**

-   **Cause:** When an `LlmAgent` has an `output_schema` defined, it's expected to produce structured output directly and should not delegate to other agents or use tools.
-   **Solution:** If you define an `output_schema`:
    -   Set `disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True`.
    -   Ensure `sub_agents` list is empty.
    -   Ensure `tools` list is empty.

------------------------------------------------------------------------

### Runtime and Execution Issues

**Issue: API Key / Authentication errors (e.g., 401, 403, permission denied)**

-   **Cause:**
    -   Missing or incorrect API key (e.g., `GOOGLE_API_KEY` for Gemini API via Google AI Studio).
    -   Incorrect Google Cloud project or location for Vertex AI.
    -   Insufficient permissions for the service account or user credentials.
    -   OAuth consent not granted or token expired for tools requiring user authorization.
-   **Solution:**
    1.  **Environment Variables:**
        -   For Gemini API (Google AI Studio): Ensure `GOOGLE_API_KEY` is set in your `.env` file or environment.
        -   For Vertex AI: Ensure `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are correctly set. Ensure `GOOGLE_GENAI_USE_VERTEXAI=1`.
        -   Load `.env` files correctly: `python-dotenv` is used. ADK CLI tools (`adk web`, `adk run`) attempt to load `.env` from the agent's directory or parent directories.
    2.  **Credentials:**
        -   **Vertex AI:** Ensure Application Default Credentials (ADC) are set up correctly (`gcloud auth application-default login`). If running on GCP services (Cloud Run, GCE), ensure the service account has the necessary IAM roles (e.g., "Vertex AI User", "Service Account User" for impersonation if needed).
        -   **OAuth for Tools (e.g., Google API Toolset):** Ensure your OAuth client ID and secret are correctly configured. For the `adk web` UI, the OAuth flow should guide the user. For programmatic use, ensure tokens are handled correctly.
    3.  **Permissions:** Verify that the authenticated principal (user or service account) has the required permissions for the Google Cloud services or APIs being accessed (e.g., BigQuery Data Viewer, Gmail API access).

**Issue: `adk web` or `adk api_server` fails to start or shows errors.**

-   **Cause:**
    -   Port already in use.
    -   Incorrect `AGENTS_DIR` path.
    -   Syntax errors or import errors in agent definition files (`agent.py`, `__init__.py`).
    -   Database connection string issues (`--session_db_url`).
-   **Solution:**
    1.  **Port Conflict:** Try a different port using the `--port` option.
    2.  **Path:** Double-check the `AGENTS_DIR` path. It should be the directory *containing* your agent application folders.
    3.  **Agent Code:** Check the console output for Python tracebacks. Fix any errors in your agent code. The `adk web` UI will also try to display these errors.
    4.  **Database URL:**
        -   For SQLite: Ensure the path to the `.db` file is correct and writable (e.g., `sqlite:///./my_sessions.db`).
        -   For other databases: Verify the connection string, hostname, credentials, and that the necessary database drivers are installed (e.g., `psycopg2-binary` for PostgreSQL).
        -   For Agent Engine: Ensure the Agent Engine resource ID is correct (`agentengine://<resource_id>`).
    5.  **Logging:** Increase log verbosity with `--log_level DEBUG` to get more detailed error messages.

**Issue: Agent doesn't use tools as expected or makes incorrect tool calls.**

-   **Cause:**
    -   Tool not correctly added to the `agent.tools` list.
    -   Tool description is unclear or misleading for the LLM.
    -   Tool's input schema (function declaration) is incorrect or ambiguous.
    -   The LLM's instruction doesn't sufficiently guide it to use the tool.
-   **Solution:**
    1.  **Verify Tool Registration:** Ensure the tool instance or callable is in the `LlmAgent(tools=[...])` list.
    2.  **Improve Tool Description:** Make the `description` of your `BaseTool` or the docstring of your `FunctionTool` very clear about what the tool does, when it should be used, and what its parameters mean.
    3.  **Check Schema:** For `FunctionTool`, ensure type hints are accurate. ADK attempts to infer the schema. For custom `BaseTool`s, ensure `_get_declaration()` returns a correct `types.FunctionDeclaration`.
    4.  **Prompt Engineering:** Refine the agent's `instruction` to better guide the LLM on when and how to use the available tools. Few-shot examples (using `ExampleTool`) can be very effective.
    5.  **Debug with `adk web` UI:** The "Trace" view can show the LLM's reasoning for tool calls and the arguments it tried to use.

**Issue: `LlmCallsLimitExceededError: Max number of llm calls limit of 'N' exceeded`**

-   **Cause:** The agent is making more LLM calls in a single invocation (user turn) than allowed by `RunConfig(max_llm_calls=N)`. This can happen with complex tool use, planning, or agent loops.
-   **Solution:**
    1.  **Increase Limit:** If the complexity is expected, increase `max_llm_calls` in the `RunConfig` passed to the `Runner`.
    2.  **Optimize Agent Logic:** Review the agent's design. Can tool use be more efficient? Is there an infinite loop?
    3.  **Improve Tool Responses:** Ensure tools return concise, useful information to avoid unnecessary follow-up LLM calls for summarization or clarification.

------------------------------------------------------------------------

### CLI Tool Specific Issues

**Issue: `adk create` produces an agent that doesn't run (e.g., API key errors).**

-   **Cause:** The backend configuration (Google AI API Key or Vertex AI project/region) selected or provided during `adk create` was incorrect or not fully set up.
-   **Solution:**
    1.  Open the generated `.env` file in the new agent's directory.
    2.  Verify and correct the `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_GENAI_USE_VERTEXAI` variables according to your setup.
    3.  Ensure the corresponding credentials/services are active and have permissions.

**Issue: `adk deploy cloud_run` fails during `gcloud run deploy` step.**

-   **Cause:**
    -   `gcloud` CLI not authenticated or configured.
    -   Insufficient permissions to deploy to Cloud Run or access related services (e.g., Artifact Registry if using `--source`).
    -   Docker build failures (if there are issues in `Dockerfile` or `requirements.txt`).
    -   Quota limits in Google Cloud.
-   **Solution:**
    1.  **gcloud Auth:** Run `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`.
    2.  **Permissions:** Ensure your account or service account has "Cloud Run Admin", "Service Account User" (if deploying as a service account), and "Storage Admin" (for Artifact Registry) roles.
    3.  **Docker Build:** Examine the output of the `gcloud run deploy` command (use `--verbosity debug` for more details) for Docker build errors. Test building the Docker image locally first if issues persist.
    4.  **Check Cloud Build Logs:** The deployment uses Cloud Build. Check the build logs in the Google Cloud Console for detailed error messages.

**Issue: `adk eval` shows "ModuleNotFoundError: Eval module is not installed..."**

-   **Cause:** The optional dependencies required for evaluation are not installed.
-   **Solution:** Install the ADK with the `eval` extra: `bash     pip install "google-adk[eval]"`

------------------------------------------------------------------------

### General Debugging Tips

1.  **Logging:**
    -   ADK uses Python's standard `logging` module. By default, logs might go to stderr or a temporary file (e.g., when using `adk run`).
    -   For `adk web` and `adk api_server`, use the `--log_level DEBUG` option for more verbose output.
    -   In your custom agent or tool code, add `logger.debug(...)` or `logger.info(...)` statements to trace execution. `python     import logging     logger = logging.getLogger('google_adk.' + __name__) # Or your_module_name     logger.debug("My custom debug message: %s", my_variable)`
2.  **ADK Web UI:**
    -   The **Chat** tab allows direct interaction.
    -   The **Trace** tab is invaluable for understanding the LLM's internal "thoughts," tool calls, parameters, and responses. It shows the sequence of events in an invocation.
    -   The **Eval** tab can help run and compare evaluation sets directly within the UI.
3.  **Simplify:**
    -   If a complex multi-agent system or a toolset isn't working, try to isolate the problematic component.
    -   Test individual agents or tools with a simple `InMemoryRunner` and direct `run_async` calls in a Python script.
4.  **Check `.env` Files:**
    -   Ensure `.env` files are in the correct location (usually within the agent's specific directory, e.g., `my_agents_dir/my_specific_agent/.env`).
    -   Verify that environment variables are correctly named and have the right values.
    -   Remember that `.env` files are loaded when the agent module is first imported. If you change `.env` while `adk web --reload` is running, the server should restart and pick up changes. For `adk run`, you'll need to restart the command.
5.  **Read Error Messages Carefully:**
    -   Python tracebacks and ADK-specific error messages often point directly to the problem. Pay attention to `ValueError`, `TypeError`, and `AttributeError` messages.
6.  **Consult Documentation and Examples:**
    -   The official ADK documentation (<https://google.github.io/adk-docs/>) and sample repositories (<https://github.com/google/adk-samples>) are good resources.

------------------------------------------------------------------------

If you encounter an issue not covered here, or if the suggested solutions don't work, consider opening an issue on the ADK GitHub repository, providing as much detail as possible (ADK version, Python version, code snippets, error messages, and steps to reproduce).
