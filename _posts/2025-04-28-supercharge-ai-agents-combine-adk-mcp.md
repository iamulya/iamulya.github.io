---
title: Supercharge Your AI Agents: Combining Google's ADK and MCP
date: 2025-04-28 17:00:00 +0100
categories: [Gen AI, Agent Development Kit]
tags: [Generative AI, Agent Development Kit, Model Context Protocol]
---

# Supercharge Your AI Agents: Combining Google's ADK and MCP

Building sophisticated AI agents that can interact with various tools and services can be complex. You need to manage the agent's core logic, its interaction with language models, and how it communicates with external capabilities. Today, we'll explore how Google's Agent Development Kit (ADK) and the Model Context Protocol (MCP) work together to simplify this process, using a fun example project: **YouBuddy**, an AI assistant for YouTube.

YouBuddy can fetch videos from channels or playlists, summarize them, and even combine multiple summaries into one. It's a perfect showcase for how ADK orchestrates an agent and MCP allows it to talk to specialized microservices. You can find the code on [Github](https://github.com/iamulya/youbuddy-adk-mcp).

## First, What's the Agent Development Kit (ADK)?

The **Google Agent Development Kit (ADK)** is a framework designed to streamline the development, debugging, and deployment of AI agents. Think of it as a toolkit that provides:

*   **Simplified Model Integration:** Easily connect to and use powerful language models like Gemini.
*   **Tool Management:** A structured way to define and integrate "tools" – external functions or services your agent can use to perform actions (e.g., search the web, call an API, query a database).
*   **Conversation Flow:** Helpers for managing the state and flow of conversations.
*   **Debugging & Testing:** Features to help you understand what your agent is doing and test its behavior.
*   **Web UI:** A built-in web interface to interact with your agent during development.

The goal of ADK is to let you focus on the agent's unique logic and capabilities rather than boilerplate code.

## And What's the Model Context Protocol (MCP)?

The **Model Context Protocol (MCP)** is a specification that defines a standard way for an AI agent (like one built with ADK) to communicate with its tools. When your tools are separate microservices, MCP provides the "language" they speak.

Key benefits of MCP:

*   **Decoupling:** Your agent doesn't need to know the internal implementation details of a tool. It just needs to know how to make an MCP request.
*   **Language Agnostic Tools:** You can build your tools in any programming language (Python, Node.js, Go, etc.) as long as they can expose an MCP-compliant server endpoint.
*   **Standardization:** Ensures consistent communication patterns, making it easier to build and maintain a suite of tools.

In practice, you'll often use libraries like `fastapi-mcp` (for Python/FastAPI tool servers) and ADK's built-in MCP client capabilities to handle the protocol details.

## Meet YouBuddy: An ADK + MCP Example

YouBuddy is an AI agent designed to help users find and understand YouTube content. It can:

1.  Fetch video URLs from a specific YouTube channel for a given date.
2.  Fetch all video URLs from a public YouTube playlist.
3.  Summarize an individual YouTube video.
4.  Combine multiple video summaries into a single, coherent overview.

To achieve this, YouBuddy (the ADK agent) relies on several backend microservices, each acting as an MCP tool:

*   `youtube-urls-mcp`: Fetches channel videos.
*   `playlist-videos-mcp`: Fetches playlist videos.
*   `video-summary-mcp`: Summarizes a single video using Gemini.
*   `final-summary-mcp`: Combines multiple text summaries using Gemini.

Let's see how these pieces fit together.

### The MCP Tools: Specialized Microservices

Each backend service is a FastAPI application that uses the `fastapi-mcp` library to expose its functionality as an MCP tool. For instance, the `video-summary-mcp` service has an endpoint to summarize a video:

**(Simplified snippet from `video-summary-mcp/main.py`)**
```python
# video-summary-mcp/main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel, HttpUrl
from google import genai # For Gemini
from fastapi_mcp import FastApiMCP
import os

app = FastAPI(title="Video Summarization Service")

# --- Pydantic Models ---
class SummaryRequest(BaseModel):
    video_url: HttpUrl

class SummaryResponse(BaseModel):
    summary: str

@app.post(
    '/summary',
    response_model=SummaryResponse,
    operation_id="get_youtube_video_summary", # Important for ADK tool discovery
    tags=["Summarization"]
)
async def generate_summary(request_data: SummaryRequest = Body()):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    # ... (logic to call Gemini with video_url to get summary) ...
    video_url_str = str(request_data.video_url)
    
    msg1_video1 = types.Part.from_uri(
        file_uri=video_url,
        mime_type="video/*",
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
        role="user",
        parts=[
            msg1_video1,
            types.Part.from_text(text="""identify the main topics and provide concise summary for each""")
        ]
        ),
    ]
    # Code to create generate config and generate content stream
    
    summary_text = "".join(chunk.text for chunk in response_chunks)

    return SummaryResponse(summary=summary_text)

# Mount MCP to expose tools
mcp = FastApiMCP(
    app,
    name="YouTube Video Summarization Service MCP",
    include_tags=["Summarization"],
    # ... other mcp configs ...
)
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```
Each of these services is Dockerized and can be deployed independently (e.g., on Cloud Run).

### The YouBuddy ADK Agent: The Orchestrator

Now, let's look at the `YouBuddy/src/agent.py` file, which defines our ADK agent.

**1. Configuration (`YouBuddy/.env`):**
The agent needs to know where its MCP tools are located. This is configured in an `.env` file:
```bash
# YouBuddy/.env
MCP_URL_GET_PLAYLIST_VIDEOS="<URL_FOR_GET_PLAYLIST_VIDEOS_TOOL_SERVICE>"
MCP_URL_SUMMARIZE_VIDEO="<URL_FOR_SUMMARIZE_VIDEO_TOOL_SERVICE>"
MCP_URL_COMBINE_SUMMARIES="<URL_FOR_COMBINE_SUMMARIES_TOOL_SERVICE>"
# ... other URLs and config ...
GOOGLE_API_KEY_SECRET_RESOURCE_NAME="projects/your-project/secrets/your-gemini-key/versions/latest"
```

**2. Loading MCP Tools:**
The agent uses `MCPToolset` from ADK to connect to these URLs and load the available tools. An `AsyncExitStack` is used to manage the lifecycle of these connections.

**(Snippet from `YouBuddy/src/agent.py`)**
```python
import os
import logging
from contextlib import AsyncExitStack
from typing import List, Tuple

from google.adk.agents import Agent
from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
# ... (Secret Manager import and fetch_secret function) ...

logger = logging.getLogger(__name__)

# Load MCP URLs from environment variables
MCP_URL_GET_PLAYLIST_VIDEOS = os.getenv("MCP_URL_GET_PLAYLIST_VIDEOS")
MCP_URL_SUMMARIZE_VIDEO = os.getenv("MCP_URL_SUMMARIZE_VIDEO")
# ... other URLs ...

async def load_mcp_tools(exit_stack: AsyncExitStack) -> List[BaseTool]:
    all_tools = []
    mcp_connections = {
        "get_playlist_videos": SseServerParams(url=MCP_URL_GET_PLAYLIST_VIDEOS),
        "summarize_video": SseServerParams(url=MCP_URL_SUMMARIZE_VIDEO),
        # ... other tool connections ...
    }

    logger.info("Connecting to MCP servers and loading tools...")
    for name, params in mcp_connections.items():
        if not params.url or params.url == "<URL_FOR_..._TOOL>": # Check if URL is placeholder
            logger.warning(f"Skipping tool '{name}' due to missing or placeholder URL.")
            continue
        try:
            toolset = MCPToolset(connection_params=params, exit_stack=exit_stack)
            await exit_stack.enter_async_context(toolset) # Manage toolset lifecycle
            tools = await toolset.load_tools()
            all_tools.extend(tools)
            logger.info(f"Loaded {len(tools)} tools from {name}")
        except Exception as e:
            logger.error(f"Failed to load tools from {name} ({params.url}): {e}")
    return all_tools
```

**3. Defining the Agent:**
The core agent is defined using ADK's `Agent` class. This includes specifying the language model, a description, and most importantly, the `instruction` prompt that guides its behavior and how it should use the loaded `tools`.

**(Snippet from `YouBuddy/src/agent.py`)**
```python
async def load_youbuddy_agent() -> Tuple[Agent, AsyncExitStack]:
    exit_stack = AsyncExitStack()
    try:
        # Fetch GOOGLE_API_KEY from Secret Manager (using fetch_secret function)
        # and set os.environ["GOOGLE_API_KEY"] = secret_value
        # ... (secret fetching logic as in your provided file) ...

        mcp_tools = await load_mcp_tools(exit_stack)

        youbuddy_agent = Agent(
            model="gemini-2.5-flash-preview-04-17", # Or your preferred model
            name="youbuddy_agent",
            description=(
                "An agent specializing in fetching and summarizing YouTube videos"
                " from channels or playlists using specialized tools."
            ),
            instruction=f"""You are YouBuddy, an expert YouTube assistant created by Amulya Bhatia. Your goal is to help users by fetching videos and creating summaries based on their requests.

    You have access to the following tools:
    {chr(10).join([f'- {tool.name}: {tool.description}' for tool in mcp_tools if tool.description])}

    Based on the user's request:
    1.  Determine if you need videos from a specific channel and date OR from a specific playlist. Use the appropriate tool (`get_channel_videos` or `get_playlist_videos`) to fetch the list of video URLs or IDs.
    2.  For each relevant video identified in step 1, use the `summarize_video` tool to get its summary.
    3.  If multiple summaries were generated, use the `combine_summaries` tool to create a final, consolidated summary.
    4.  Present the final summary to the user. If only one video was summarized, present that summary directly.
    5.  If a tool fails, inform the user you couldn't complete the request due to a tool issue.
    """,
            tools=mcp_tools, # Pass the loaded MCP tools here
        )

        return youbuddy_agent, exit_stack
    except Exception as e:
        await exit_stack.aclose()
        logger.exception("Failed to initialize YouBuddy agent.")
        raise

# ADK expects root_agent to be an awaitable that returns (Agent, AsyncExitStack)
root_agent = load_youbuddy_agent()
```
The `instruction` prompt is crucial. It tells the Gemini model how to behave and how to interpret the available tools (whose descriptions are automatically pulled from the MCP services). The model then uses "function calling" to decide which tool to invoke based on the user's query.

**4. Running the Agent:**
With the ADK, running the agent for local development is straightforward:
```bash
cd YouBuddy
adk web . # This looks for root_agent in src/agent.py by default
```
This command starts a local web server (usually on `http://localhost:8000`), where you can interact with YouBuddy.

## Interacting with YouBuddy

Once running, you can chat with YouBuddy:

*   "Summarize the videos in the playlist `https://www.youtube.com/playlist?list=PL6gx4Cwl9DGCkg2uj3PxUWhMDuTw3VKjM`?"
*   "Find videos from channel `UC_x5XG1OV2P6uZZ5FSM9Ttw` on 2024-03-10 and give me a combined summary."

YouBuddy will:
1.  Parse your request.
2.  Decide to use `get_playlist_videos` (or `get_youtube_videos_for_channel_date`).
3.  Call the respective MCP tool.
4.  Receive a list of video URLs.
5.  For each URL, call the `get_youtube_video_summary` MCP tool.
6.  Receive individual summaries.
7.  Call the `generate_final_summary` MCP tool with all individual summaries.
8.  Present the final, combined summary to you.

## Why ADK + MCP is Powerful

The YouBuddy example highlights several advantages of this architecture:

*   **Modularity:** The agent's brain (ADK agent with Gemini) is separate from its "hands and feet" (the MCP tools).
*   **Scalability:** Each MCP tool is a microservice that can be scaled independently.
*   **Reusability:** MCP tools can be shared across different agents or applications.
*   **Focus:** ADK lets you focus on the agent's high-level logic and prompting, while MCP handles the tool communication details.
*   **Development Speed:** The `adk web` interface and clear separation of concerns accelerate development and debugging.

## Get Started!

Building AI agents that can perform complex tasks by leveraging external tools is becoming increasingly important. Google's Agent Development Kit (ADK) and the Model Context Protocol (MCP) provide a powerful and flexible framework for creating such agents.

The YouBuddy project is just one example. Think about what specialized tools *your* agent might need and how you could build them as MCP-compliant microservices.

Happy agent building!
