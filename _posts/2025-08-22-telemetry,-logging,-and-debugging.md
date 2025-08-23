---
title: Chapter 22 - Telemetry, Logging, and Debugging 
date: "2025-08-22 18:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Building sophisticated AI agents involves more than just writing functional code; it requires understanding their behavior, diagnosing issues, and monitoring their performance. The Agent Development Kit (ADK) provides several mechanisms for gaining insights into your agent's operations through telemetry, structured logging, and powerful debugging tools, primarily centered around the ADK Development UI.

This chapter explores how ADK integrates with OpenTelemetry for tracing, effective logging strategies, and practical techniques for debugging your agents.

## Understanding ADK Telemetry: OpenTelemetry Integration

ADK is instrumented with **OpenTelemetry (OTel)**, a vendor-neutral, open-source observability framework for collecting telemetry data (traces, metrics, and logs). This integration focuses on capturing high-level agent activities that are not typically visible to lower-level SDKs (like the `google-generativeai` SDK, which might have its own OTel instrumentation for raw API calls).

**What ADK Traces:**
ADK's telemetry, accessible via `google.adk.telemetry.tracer` (an OpenTelemetry `Tracer` instance), primarily creates **spans** for significant operations within an agent's lifecycle. A span represents a unit of work or an operation. Key operations traced by ADK include:

- **Overall Invocation:** The entire processing of a user's turn (`runner.run_async`).
- **Agent Run:** Each time an agent (`BaseAgent.run_async`) is invoked.
- **LLM Call:** The act of sending a request to and receiving a response from an LLM (captures `LlmRequest` and `LlmResponse` summaries).
- **Tool Call:** When an agent decides to use a tool.
- **Tool Response:** The result returned by a tool.
- **Data Sent to Live LLM Connection:** When using `run_live`.

**How to Use/View This Telemetry:**

1. **ADK Development UI (Trace View):** This is the **primary and easiest way** to visualize ADK's telemetry for local development. The hierarchical trace view is a direct visual representation of these OpenTelemetry spans and their relationships. Each collapsible item in that trace corresponds to a span.
2. **Cloud Trace:** If you have set `--trace_to_cloud` during your deployment using `adk deploy` or `adk api_server` command, your tracing data will be available in Cloud Trace in Google Cloud. Viewing Traces in Google Cloud Console:
- Navigate to the Google Cloud Console.
- In the navigation menu, go to Operations > Trace > Trace list.
- You can then search and filter traces. Traces from ADK will typically be associated with a service name (often derived from your app_name when deployed or a default like "fastapi" for local adk web/api_server).
- Clicking on a trace will show you a waterfall diagram of the spans, their durations, and the attributes associated with each span. This visual representation is extremely helpful for understanding the sequence and timing of operations.
2. **Configuring an OpenTelemetry Exporter (Advanced/Production):**
For production monitoring or more advanced analysis, you can configure an OpenTelemetry SDK with an exporter to send this trace data to an observability backend (e.g., Google Cloud Trace, Jaeger, Zipkin, Prometheus).
    

> ## Dev UI Trace View IS OpenTelemetry
> 
> The hierarchical trace view you see in the ADK Dev UI is powered by ADK's internal OpenTelemetry instrumentation. The Dev UI sets up an in-memory OTel exporter and a custom processor to render these spans visually. This means you're already benefiting from OTel when using the Dev UI.
> {: .prompt-info }

## Effective Logging Strategies for ADK Agents

While OpenTelemetry provides structured traces, traditional logging remains essential for detailed debugging and operational monitoring. Python's built-in `logging` module is fully compatible with ADK.

**Where to Add Logging:**

- **Agent Initialization:** Log key configuration parameters when an agent or toolset is initialized.
- **`InstructionProvider`s:** Log the dynamically generated instruction.
- **Callbacks (`before/after_agent_callback`, `before/after_model_callback`, `before/after_tool_callback`):** These are excellent places to log:
    - The data received (e.g., `LlmRequest`, `LlmResponse`, tool arguments, tool response).
    - Any modifications made by the callback.
    - The decision to skip an operation (e.g., if `before_model_callback` returns an `LlmResponse`).
- **Tool `run_async` Methods:** Log important steps, parameters received, and results being returned by your custom tools.
- **`MemoryService` and `ArtifactService` Custom Implementations:** Log interactions with your backend storage.
- **`Runner` Customizations (if any):** Log high-level lifecycle events.


> ## Best Practice: Use Specific Loggers and Levels
> 
> - Get specific loggers for your modules (e.g., `logging.getLogger(__name__)` or `logging.getLogger("my_app.my_module")`). This allows fine-grained control over log output from different parts of your application.
> - Use appropriate log levels: `DEBUG` for detailed diagnostic information, `INFO` for general operational messages, `WARNING` for potential issues, `ERROR` for failures, `CRITICAL` for severe errors.
> - Control ADK's internal logging verbosity: 
> 
> `logging.getLogger('google_adk').setLevel(logging.INFO)` (or `DEBUG`).
> {: .prompt-info }


> ## Logging Sensitive Data
> 
> Be extremely careful about what you log, especially at INFO or DEBUG levels. Avoid logging Personally Identifiable Information (PII), API keys, full prompts/responses if they might contain sensitive user data, or any other confidential information, particularly if logs are sent to a centralized logging system. Implement redaction or selective logging if necessary.
> {: .prompt-info }

## Debugging Techniques

Effective debugging is key to efficient agent development.

**1. The ADK Development UI's Trace View:**

- **This is your primary debugging tool.** As emphasized before, the Trace view (`adk web .` then toggle to Trace) provides a hierarchical, timed breakdown of:
    - The full `InvocationContext` at each stage.
    - The exact `LlmRequest` sent to the model (including system prompt, history, tool declarations).
    - The raw `LlmResponse` from the model (including text, function calls requested, usage metadata).
    - Tool invocations: which tool, what arguments were passed by the LLM.
    - Tool responses: the exact data returned by the tool.
    - State deltas applied at each step.
- By examining the trace, you can pinpoint:
    - Why an LLM didn't call a tool (e.g., poor tool description, missing in prompt).
    - Why an LLM called a tool with wrong arguments (e.g., misinterpreting parameter schema).
    - Errors within your tool's execution.
    - Unexpected state changes.
    - The flow of control in multi-agent systems.

**2. Python Debugger (`pdb` or IDE Debugger):**

- You can set breakpoints in your ADK code (agent definitions, tool functions, callbacks) just like any other Python application.
- **Attaching to `adk web`:** If you're using `adk web .`, the Uvicorn server runs your agent code. To debug this:
    - You might need to run Uvicorn directly with your ADK FastAPI app instance, rather than via `adk web`, to have more control for attaching a debugger.
    - Alternatively, launch `adk web` and then attach your IDE's debugger to the Python process running Uvicorn.
    - Or, simpler for specific points: add `import pdb; pdb.set_trace()` in your code where you want to break. When that code is hit during an interaction via the Dev UI, your terminal running `adk web` will drop into the `pdb` debugger.
    
    ```python
    # Example: Debugging a tool call
    def my_buggy_tool(param1: str, tool_context: ToolContext):
        logger.info("Tool entered")
        import pdb; pdb.set_trace() # Execution will pause here
        # ... rest of tool logic ...
        result = f"Processed: {param1.upper()}" # Potential error if param1 is not string
        logger.info("Tool finishing")
        return {"output": result}
    
    ```
    
**3. Print Statements (Judiciously):**

- While the Trace view is superior, targeted print statements or `logger.debug()` calls can still be useful for quick checks, especially in complex Python logic within tools or callbacks that might be hard to step through otherwise.

**4. Analyzing `Session` Objects:**

- If using a persistent `SessionService` like `DatabaseSessionService`, you can inspect the raw session data (events, state) directly in your database.
- For `InMemorySessionService`, you can programmatically access and print the session object after a run:
    
    ```python
    # ... after runner.run(...) ...
    # session_id = "the_session_id_you_used"
    # user_id = "the_user_id_you_used"
    # app_name = runner.app_name
    #
    # # Accessing internal structure for InMemorySessionService (for debugging only!)
    # if isinstance(runner.session_service, InMemorySessionService):
    #     session_obj = runner.session_service.sessions.get(app_name, {}).get(user_id, {}).get(session_id)
    #     if session_obj:
    #         print("\n--- Full Session Object Dump ---")
    #         print(session_obj.model_dump_json(indent=2))
    
    ```
    

**Common Pitfalls and How to Debug Them:**

- **Agent Not Using a Tool:**
    - **Trace Check:** Is the tool's `FunctionDeclaration` actually in the `LlmRequest`?
    - **Tool Description:** Is the tool's `description` (and parameter descriptions) clear and compelling for the LLM? Does it accurately state when the tool should be used?
    - **Agent Instruction:** Does your agent's main `instruction` encourage or permit tool use for the relevant task?
- **Agent Calling Tool with Wrong Arguments:**
    - **Trace Check:** What arguments did the LLM actually try to pass?
    - **Schema Clarity:** Are the type hints and Pydantic models for your tool parameters clear? Is the LLM misinterpreting a type (e.g., sending a string for an expected integer)?
    - **Parameter Descriptions:** Are parameter descriptions in the docstring precise?
- **Tool Errors:**
    - **Trace Check:** What was the `tool_response` (often contains an error message)?
    - **Local Tool Test:** Can you call your tool's Python function directly with the arguments the LLM tried to use to reproduce the error outside of ADK?
    - **Debugger:** Set breakpoints in your tool's `run_async` method.
- **Unexpected Agent Behavior/Response:**
    - **Trace Check:** Examine the full prompt (system instruction + history) sent to the LLM. Is it what you expected? Does it clearly guide the LLM to the desired output?
    - **Temperature:** If responses are too random or off-topic, try lowering the `temperature` in `GenerateContentConfig`.
    - **Instruction Refinement:** This is often an iterative process. Modify the agent's `instruction` and re-test.
- **Authentication Issues with Tools:**
    - **Trace Check:** Look for any error messages in `tool_response` related to auth.
    - **Credential Configuration:** Double-check that `auth_credential` is correctly configured for the tool/toolset.
    - **Scopes/Permissions:** Ensure the credentials used have the necessary permissions/scopes for the API operation the tool is trying to perform.
    - **OAuth Flow (Dev UI):** If using OAuth, ensure you're completing the consent flow correctly when prompted by the Dev UI.


> ## Best Practice: Isolate and Test Components
> 
> - **Test Tools Independently:** Before integrating a complex tool into an agent, test its Python function directly with various inputs.
> - **Test Agent Logic with Mocked Tools/LLMs:** For unit testing an agent's orchestration logic, you can mock the LLM responses or tool outputs to simulate different scenarios without actual external calls. (This is an advanced topic, but `before_model_callback` and `before_tool_callback` can help here).
> {: .prompt-info }

**What's Next?**

With robust telemetry, logging, and debugging strategies, you're well-equipped to build and maintain reliable ADK agents. Understanding what's happening "under the hood" is key to resolving issues and optimizing performance. Next "Security Best Practices for ADK Agents," we'll discuss important considerations for building secure agents, especially when they interact with external systems and user data.
