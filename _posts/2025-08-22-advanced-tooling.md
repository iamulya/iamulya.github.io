---
title: Chapter 8 - Advanced MCP Tooling and Framework Integrations 
date: "2025-08-22 11:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

In the preceding chapters, we've explored how to build custom tools with `FunctionTool` and integrate with standard REST APIs using OpenAPI and API Hub. ADK's flexibility extends further, allowing integration with specialized toolsets and even other agent development frameworks. This chapter delves into these advanced tooling capabilities, including the Model Context Protocol (MCP) based tools, Google Application Integration, and adaptors for Langchain and CrewAI tools.

## Integrating with Model Context Protocol (MCP) Based Tools

The **Model Context Protocol (MCP)** is a specification designed to enable interoperability between AI models/agents and external tools or services. Tools that adhere to the MCP standard can be discovered and invoked by MCP-compatible clients. ADK provides the `google.adk.tools.mcp_tool.MCPToolset` to connect to an MCP server and make its tools available to your ADK agents.

**How it Works:**

1. You provide connection parameters to an MCP server (either a local Stdio-based server or a remote SSE-based server).
2. `MCPToolset` establishes a session with the MCP server.
3. It lists the tools available on that server.
4. For each MCP tool, it creates an `MCPTool` wrapper (which is an ADK `BaseTool`).
5. Your ADK agent can then use these `MCPTool` instances like any other ADK tool. When called, the `MCPTool` uses the active MCP session to invoke the actual tool on the MCP server.

**Example: Using a Local Filesystem MCP Server**

The MCP ecosystem provides example servers. One common one is:

`@modelcontextprotocol/server-filesystem`: exposes tools to interact with a local filesystem (list files, read files, write files). This requires Node.js and `npx` to run.

```python
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset 
from mcp import StdioServerParameters 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

filesystem_mcp_params = StdioServerParameters(
    command='npx', args=["-y", "@modelcontextprotocol/server-filesystem", "."]
)
mcp_fs_toolset = None

try:
    mcp_fs_toolset = MCPToolset(connection_params=filesystem_mcp_params)
except ImportError:
    print("MCP Toolset requires Python 3.10+ and 'mcp' package.")
    print("Also ensure Node.js and npx are available for this example.")

mcp_agent = None
if mcp_fs_toolset:
    mcp_agent = Agent(
        name="filesystem_navigator", model=DEFAULT_LLM,
        instruction="Interact with local filesystem using tools (listFiles, readFile, writeFile).",
        tools=[mcp_fs_toolset]
    )

if __name__ == "__main__":
    runner = InMemoryRunner(agent=mcp_agent, app_name="ArtifactApp")
    session_id = "s_artifact_test"
    user_id="artifact_user"
    create_session(runner, session_id, user_id)

    async def main():
        if not mcp_agent:
            print("MCP Agent not initialized. Skipping example.")
            return

        test_file_content = "Hello from ADK to MCP!"
        test_filename = "mcp_test_file.txt"

        with open(test_filename, "w") as f: f.write(test_file_content)
        prompts = [ f"Contents of '{test_filename}'?"]
        for prompt_text in prompts:
            print(f"
YOU: {prompt_text}")
            user_message = Content(parts=[Part(text=prompt_text)], role="user")
            print("FILESYSTEM_NAVIGATOR: ", end="", flush=True)
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
                if event.content and event.content.parts and event.content.parts[0].text:
                    print(event.content.parts[0].text, end="")
            print()
        if os.path.exists(test_filename): os.remove(test_filename)
        if mcp_fs_toolset: # Ensure it was initialized before trying to close
            print("
Closing MCP toolset...")
            await mcp_fs_toolset.close()
            print("MCP toolset closed.")

    asyncio.run(main())
```

![*Diagram: Sequence of an `LlmAgent` using an `MCPTool` from an `MCPToolset`.*](/assets/img/2025-08-22-advanced-tooling/figure-1.png)



> ## Interoperability with MCP
> 
> MCPToolset allows ADK agents to tap into the growing ecosystem of MCP-compliant tools and servers. This promotes interoperability and allows you to leverage tools developed independently of ADK.
> {: .prompt-info }


> ## MCP Server Management and Lifecycle
> 
> - When using `StdioServerParameters` (local MCP servers like the `npx` example), the `MCPToolset` attempts to manage the lifecycle of the server process. It's crucial to call `await mcp_fs_toolset.close()` when your application shuts down to ensure these external processes are terminated properly. Failure to do so can leave orphaned processes.
> - The `MCPToolset` uses an `AsyncExitStack` internally to manage resources. Proper cleanup via `close()` is vital.
> - For `SseServerParams` (connecting to a remote, already running MCP server), `close()` will primarily close the SSE connection.
> {: .prompt-info }


> ## Best Practice: Specific Tool Filtering with MCPToolset
> 
> MCP servers can expose many tools. If your agent only needs a subset, use the `tool_filter` argument in the `MCPToolset` constructor. This can be a list of tool names or a ToolPredicate function to selectively expose tools to the LLM, reducing prompt clutter and potential misuse.
> 
> ```python
> from google.adk.tools.base_toolset import ToolPredicate
> from google.adk.tools import BaseTool
> from google.adk.agents import ReadonlyContext
>  
> def my_mcp_tool_filter(tool: BaseTool, context: ReadonlyContext | None = None) -> bool:
>     # Only allow tools related to reading, not writing
>     return "read" in tool.name.lower() or "list" in tool.name.lower()
> 
> # mcp_fs_toolset_filtered = MCPToolset(
> #     connection_params=filesystem_mcp_params,
> #     tool_filter=my_mcp_tool_filter
> # )
> # Or by name:
> # mcp_fs_toolset_filtered_by_name = MCPToolset(
> # connection_params=filesystem_mcp_params,
> # tool_filter=["listFiles", "readFile"] # Exact MCP tool names
> # ) 
> ```
> {: .prompt-info }


## `ApplicationIntegrationToolset`: Connecting to Enterprise Systems

Google Cloud Application Integration allows you to connect to various enterprise systems (SaaS applications, databases, custom services) through pre-built connectors and custom integrations. The `google.adk.tools.application_integration_tool.ApplicationIntegrationToolset` enables your ADK agents to trigger these integrations or interact with these connectors.

It typically works by:

1. Taking your Application Integration or Integration Connector configuration (project, location, integration/connection name, specific triggers/entities/actions).
2. Internally generating an OpenAPI specification that represents the actions your agent can take.
3. Using an `OpenAPIToolset` (or similar logic) to parse this generated spec into ADK tools.
4. These tools then make authenticated calls to the Application Integration execution APIs.

The following agent connects with an application integrion on Google Cloud to send out an email using an API trigger.

```python
from google.adk.agents import Agent
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

GCP_PROJECT_AI = os.getenv("GOOGLE_CLOUD_PROJECT") # Your GCP Project
GCP_LOCATION_AI = os.getenv("APP_INTEGRATION_LOCATION", "us-central1")
# Name of your deployed integration with an API trigger
INTEGRATION_NAME = os.getenv("MY_APP_INTEGRATION_NAME")
# Trigger ID from your integration's API trigger
TRIGGER_ID = os.getenv("MY_APP_INTEGRATION_TRIGGER_ID")
# Path to your service account JSON key file with permissions for App Integration
SERVICE_ACCOUNT_JSON_PATH_AI = os.getenv("APP_INTEGRATION_SA_KEY_PATH")

app_integration_agent = None
if all([GCP_PROJECT_AI, GCP_LOCATION_AI, INTEGRATION_NAME, TRIGGER_ID]):
    sa_json_content_ai = None
    if SERVICE_ACCOUNT_JSON_PATH_AI and os.path.exists(SERVICE_ACCOUNT_JSON_PATH_AI):
        with open(SERVICE_ACCOUNT_JSON_PATH_AI, 'r') as f:
            sa_json_content_ai = f.read()
    elif SERVICE_ACCOUNT_JSON_PATH_AI: # Path provided but not found
        print(f"Warning: Service account key file not found at {SERVICE_ACCOUNT_JSON_PATH_AI}")

    try:
        # Example: Toolset for a specific integration trigger
        app_int_toolset = ApplicationIntegrationToolset(
            project=GCP_PROJECT_AI,
            location=GCP_LOCATION_AI,
            integration=INTEGRATION_NAME,
            triggers=[TRIGGER_ID], # List of trigger IDs
            service_account_json=sa_json_content_ai # Can be None to use ADC
        )

        app_integration_agent = Agent(
            name="enterprise_gateway_agent",
            model=DEFAULT_LLM,
            instruction="You can trigger enterprise workflows via Application Integration.",
            tools=[app_int_toolset] # Add the toolset
        )
        print("ApplicationIntegrationToolset initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ApplicationIntegrationToolset: {e}")
        print("Ensure Application Integration is set up and SA has permissions.")

else:
    print("Skipping ApplicationIntegrationToolset example due to missing environment variables.")


if __name__ == "__main__":
    if app_integration_agent:
        runner = InMemoryRunner(agent=app_integration_agent, app_name="AppIntApp")
        session_id = "s_app_integration_test"
        user_id = "app_integration_user"
        create_session(runner, session_id, user_id)

        prompts = [
            "Send email"
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
    else:
        print("ApplicationIntegration agent not created.")
```


> ## Connecting Agents to Enterprise Application
> 
> IntegrationToolset is a powerful way to bridge ADK agents with existing enterprise applications and workflows managed by Google Cloud Application Integration. This enables agents to perform meaningful business actions.
> {: .prompt-info }


> ## Permissions and Configuration
> 
> Setting up ApplicationIntegrationToolset requires:
>  
> - Correct GCP project, location, and resource names.
> - The service account (or default credentials) used by ADK must have appropriate IAM roles to execute Application Integrations and/or access Integration Connectors (e.g., "Application Integration Invoker", "Connectors Admin/User").
> - The integrations themselves must be correctly configured with API triggers or the connectors must be properly set up.
> {: .prompt-info }

## `ToolboxToolset`: Utilizing the Generic MCP Toolbox

The `ToolboxToolset` is another MCP-based toolset, but it's designed to connect to a generic "Toolbox" server. A Toolbox server is an MCP server that can host a variety of tools, often discovered or loaded dynamically.

```python
from google.adk.agents import Agent
from google.adk.tools import ToolboxToolset # Key import
from google.adk.runners import InMemoryRunner
# from mcp import StdioServerParameters # If using a local toolbox via stdio
# from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams # If remote

# This example is conceptual as it requires a running Toolbox server.
# Let's assume a Toolbox server is running at <http://localhost:5001>
# and exposes a toolset named "utility_tools" containing a 'uuid_generator' tool.

# If the Toolbox server requires auth, you might need auth_token_getters:
# def get_my_toolbox_token(): return "some_bearer_token"
# auth_getters = {"my_auth_service_name_in_tool_spec": get_my_toolbox_token}

try:
    # To use a specific toolset from the toolbox server:
    # utility_toolbox = ToolboxToolset(
    #     server_url="<http://localhost:5001>", # URL of your Toolbox server
    #     toolset_name="utility_tools",
    #     # auth_token_getters=auth_getters # If auth is needed
    # )

    # Or to load specific tools by name:
    uuid_generator_from_toolbox = ToolboxToolset(
         server_url="<http://localhost:5001>", # Placeholder URL
         tool_names=["uuid_generator"] # Name of the tool on the Toolbox server
    )
    print("ToolboxToolset conceptually initialized (requires a running Toolbox server).")
    toolbox_connected_agent = Agent(
        name="toolbox_user_agent",
        model="gemini-1.5-flash-latest",
        instruction="You can use utility tools from the connected Toolbox.",
        tools=[uuid_generator_from_toolbox]
    )
except ImportError:
    print("ToolboxToolset requires 'toolbox-core' (install via 'google-adk[extensions]').")
    toolbox_connected_agent = None
except Exception as e:
    print(f"Could not initialize ToolboxToolset (is a Toolbox server running at the URL?): {e}")
    toolbox_connected_agent = None

if __name__ == "__main__":
    if toolbox_connected_agent:
        # runner = InMemoryRunner(agent=toolbox_connected_agent, app_name="ToolboxApp")
        # prompt = "Generate a new UUID for me."
        # ... (runner logic) ...
        print("Agent with ToolboxToolset created. Run requires a live Toolbox server at the specified URL.")
    else:
        print("Toolbox agent not created.")

```

The `ToolboxToolset` is useful when you have a central MCP-compliant server hosting various general-purpose or shared tools.

## Bridging with Other Frameworks

ADK is designed to be interoperable. It provides adapter tools to leverage tools built with other popular agent frameworks.

**`google.adk.tools.LangchainTool`** - This tool allows you to wrap existing Langchain tools (those inheriting from `langchain_core.tools.BaseTool` or simple callables compatible with Langchain's `Tool` class) for use within an ADK agent.

```python
from google.adk.agents import Agent
from google.adk.tools.langchain_tool import LangchainTool 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

langchain_integrated_agent = None
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    langchain_search_tool_instance = DuckDuckGoSearchRun()
    adk_wrapped_duckduckgo = LangchainTool(
        tool=langchain_search_tool_instance, name="internet_search_duckduckgo", 
        description="A tool to perform a web search using DuckDuckGo to find information on a given topic."
    )
    langchain_integrated_agent = Agent(
        name="langchain_search_user", model=DEFAULT_LLM,
        instruction="You can search the internet using DuckDuckGo.",
        tools=[adk_wrapped_duckduckgo]
    )
    print("LangchainTool wrapper initialized.")
except ImportError:
    print("Langchain or DuckDuckGoSearchRun not found.")

if __name__ == "__main__":
    if langchain_integrated_agent:
        runner = InMemoryRunner(agent=langchain_integrated_agent, app_name="LangchainApp")
        session_id = "s_langchain_test"
        user_id = "langchain_user"
        create_session(runner, session_id, user_id)
        prompt = "What is Langchain?"
        print(f"
YOU: {prompt}")
        user_message = Content(parts=[Part(text=prompt)], role="user")
        print("ASSISTANT: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(event.content.parts[0].text, end="")
        print()
```

The `LangchainTool` adapter handles the schema conversion and invocation, allowing the LLM to see and use the Langchain tool as if it were a native ADK tool.

**`google.adk.tools.CrewaiTool`** - Similarly, `CrewaiTool` allows you to integrate tools built for the CrewAI framework.

```python
from google.adk.agents import Agent
from google.adk.tools.crewai_tool import CrewaiTool 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import os

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

crewai_integrated_agent = None

try:
    from crewai_tools import SerperDevTool 
    if os.getenv("SERPER_API_KEY"):
        crewai_serper_tool_instance = SerperDevTool()
        adk_wrapped_serper = CrewaiTool(
            tool=crewai_serper_tool_instance, name="google_search_serper", 
            description=crewai_serper_tool_instance.description or "A tool to search Google using Serper API."
        )
        crewai_integrated_agent = Agent(
            name="crewai_search_user", model=DEFAULT_LLM,
            instruction="You can search Google using the SerperDevTool.",
            tools=[adk_wrapped_serper]
        )
        print("CrewaiTool wrapper initialized.")
    else: print("SERPER_API_KEY not set. Skipping CrewAI SerperDevTool example.")
except ImportError:
    print("CrewAI or SerperDevTool not found.")

if __name__ == "__main__":
    if crewai_integrated_agent:
        print("CrewAI Agent ready. Run with SERPER_API_KEY set to test.")
        runner = InMemoryRunner(agent=crewai_integrated_agent, app_name="CrewAIApp")
        session_id = "s_crewai_test"
        user_id = "crewai_user"
        create_session(runner, session_id, user_id)

        prompt = "What is CrewAI?"
        print(f"
YOU: {prompt}")
        user_message = Content(parts=[Part(text=prompt)], role="user")
        print("ASSISTANT: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(event.content.parts[0].text, end="")
        print()
```


> ## Best Practice: Leverage Existing Tool Investments
> 
> If you have existing tools built for Langchain or CrewAI, the LangchainTool and CrewaiTool adapters provide an easy migration path or way to use them within ADK without rewriting them. This promotes code reuse and allows you to benefit from the specific strengths of different frameworks.
> {: .prompt-info }


> ## Dependency Management for Adapters
> 
> Using LangchainTool or CrewaiTool means your ADK project will also depend on langchain or crewai (and their dependencies) respectively. Manage these using your pyproject.toml or requirements.txt. ADK's extensions optional dependency group (pip install "google-adk[extensions]") includes many of these. Also, ensure any API keys or environment variables required by the original Langchain/CrewAI tools are properly set.
> {: .prompt-info }

**What's Next?**

We've now covered a wide array of tooling options, from custom Python functions to sophisticated integrations with external services and other agent frameworks. The ability to equip agents with diverse tools is fundamental to their power. Next, we'll explore a specialized and critical capability: enabling agents with Code Execution, allowing them to write and run code to solve problems or perform complex data manipulations.
