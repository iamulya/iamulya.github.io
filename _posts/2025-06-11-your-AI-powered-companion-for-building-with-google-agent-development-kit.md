---
title: Your AI-Powered Companion for Building with Google's Agent Development Kit
date: 2025-06-11 14:00:00 +0100
categories: [Gen AI, Gemini, Agent Development Kit (ADK)]
tags: [Generative AI, Agent Development Kit (ADK), Agentic AI, Gen AI, Gemini]
---

Building powerful AI agents is an exciting frontier, and Google's Agent Development Kit (ADK) provides a robust framework for doing just that. But as with any powerful tool, there's a learning curve. I've often found myself juggling documentation, debugging tricky issues, and spending valuable time on project artifacts instead of core agent logic.

While looking for a project for [Google's ADK Hackathon](https://googlecloudmultiagents.devpost.com/), I asked myself - What if there was an expert AI-peer to help me navigate this process?

That's the question I set out to answer with the **ADK Expert Agent** — a sophisticated, multi-purpose AI assistant built *with* ADK, *for* ADK developers. It’s more than just a chatbot; it's a developer companion I designed to accelerate the entire workflow, from the first line of code to the final presentation.

### Who is this For?

ADK Expert Agent is designed to be valuable for anyone working with ADK, no matter their experience level:

*   **The ADK Newcomer:** Feeling overwhelmed? The agent can answer fundamental questions, explain core concepts, and provide code examples to help you get up and running quickly.
*   **The Seasoned Developer:** Need to debug a specific error or understand a nuance of the framework? The agent can analyze your code, provide guidance on GitHub issues, and even generate unit tests and `evalsets` to streamline your development cycle.
*   **The ADK Contributor:** Want to understand the context of a specific GitHub issue before contributing? The agent can fetch the issue details and provide a summary and expert guidance, helping you make more impactful contributions.

### What Can It Do? The Core Features

The ADK Expert Agent is packed with features designed to make you more productive.

*   📚 **Instant ADK Expertise:** Ask it anything about ADK v1.2.0, from "What is a `SequentialAgent`?" to "How do I use `after_tool_callback`?" and get detailed, context-aware answers.
*   🐛 **GitHub Issue Guidance:** Mention a GitHub issue number (e.g., "Can you help with ADK Github issue #123?"), and the agent will fetch its details from the `google/adk-python` repo, analyze the content, and provide ADK-specific guidance or potential solutions.
*   📄 **Automatic Document Generation:** Say goodbye to tedious report writing. Ask the agent to "Create a PDF document about ADK tools" or "Generate a PowerPoint presentation on agent observability," and it will produce a professional-looking document (PDF, HTML, or PPTX) using `marp-cli`.
*   🎨 **Architecture Visualization:** Need to explain your agent's design? Just describe it—"Create a sequence diagram for a user login flow" — and the agent will generate a clean PNG architecture diagram using Mermaid syntax and provide a shareable link.

### Under the Hood: How I Built It

The agent's power comes from a modular architecture built on **Python**, the **Google ADK framework**, and **Gemini 2.5 Pro**. The core design follows the **Orchestrator-Specialist pattern**:

1.  **The Root Orchestrator:** A central `adk_expert_orchestrator` agent acts as the brain. It analyzes user queries and intelligently delegates tasks to the appropriate specialist.
2.  **Specialist Agents:** I built a suite of dedicated agents for complex, multi-step tasks. For example, the `github_issue_processing_agent` is a `SequentialAgent` that first calls a tool to fetch issue data, then passes that data to another tool to generate guidance. This keeps each component focused and maintainable.
3.  **A Rich Toolbox:** The agents are empowered by a suite of custom ADK `Tools` that integrate with:
    *   **External CLIs:** `marp-cli` and `mermaid-cli` are executed to handle document and diagram generation.
    *   **Google Cloud:** I use **Cloud Storage** for hosting artifacts and **Secret Manager** for securely managing API keys.
    *   **GitHub API:** For fetching live issue data.
4.  **An Insightful UI:** The frontend is an **Angular** application with a developer-focused side panel that offers deep observability into the agent's **Events**, **Session State**, and generated **Artifacts**, making it a fantastic tool for debugging and learning.

Here’s the high-level architecture diagram:

![architecture diagram](assets/img/adk-expert.png)

### The Bumps in the Road: Challenges I Overcame

Building a complex agent is a journey of discovery. Here are some of the challenges I tackled:

*   **Ensuring Consistency:** Even with a low temperature, LLMs can be non-deterministic. I had to build resilient parsers and error-handling logic to manage cases where the model produced slightly malformed Mermaid or Markdown syntax.
*   **Chaining Agents:** Passing data cleanly between agents in a sequence was tricky. I learned that explicit output formatting is key and even created a dedicated `FormatOutputAgent` to ensure data was correctly passed from one step to the next.
*   **Dockerizing CLI Dependencies:** Integrating Node.js-based tools like `marp-cli` into a Python Docker image was complex, requiring careful management of system dependencies for the headless Chrome instance used by Puppeteer.
*   **Prompt Engineering an Orchestrator:** Crafting prompts that could reliably distinguish between a request for a PDF, a diagram, or a GitHub issue required significant iteration and a combination of keyword matching and regex.

### Get Your Hands Dirty: Try It Yourself!

I've open-sourced the entire project so you can run it, learn from it, and extend it.

**Prerequisites:**
*   Git
*   Python 3.12+
*   Node.js

**1. Clone the Repository**
```bash
git clone https://github.com/iamulya/adk-expert-agent.git
cd adk-expert-agent
```

**2. Configure the Backend**
Navigate to the `expert-agents/` directory.
*   Copy `.env.example` to `.env`.
*   Fill in your GCP `project_number`, Secret Manager `secret_id`s for your Gemini and GitHub keys, and your GCS bucket details.

**4. Run Locally (for Development)**

*   **Start the Backend:**
    ```bash
    # From the adk-expert-agent/ directory
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    adk api_server --host 0.0.0.0 --port 8000 --allow_origins "http://localhost:4200" .
    ```

*   **Start the Frontend:**
    ```bash
    # From the adk-expert-agent/webui/ directory
    npm install
    npm run serve --backend=http://localhost:8000
    ```

Now, open your browser to `http://localhost:4200` and start chatting with your own ADK Expert Agent!

I believe the ADK Expert Agent is a powerful example of how AI can augment the developer workflow. I encourage you to try it out, explore the code, and share your feedback. Happy building!