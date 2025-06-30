---
title: "Gemini CLI Series: Architecting Secure, Extensible AI Workflows"
date: 2025-06-30 10:00:00 +0100
categories: [Gen AI, AI Coding Agents, Gemini CLI]
tags: [Gen AI, Gemini, Gemini CLI, AI Coding Agents, Gemini CLI Series]
image:
  path: /assets/img/gemini-cli.png
  alt: "Gemini CLI"
---

In my [last post](https://iamulya.one/posts/gemini-cli-series-from-first-prompt-to-personalized-powerhouse), we mastered the basics of using the Gemini CLI for day-to-day tasks. Now, it's time to peel back the layers and look at the engine that drives its power: a sophisticated system of tools, security features, and extension points that transform it from a command-line assistant into a full-fledged automation platform.

This post will explore how you can leverage these advanced capabilities to build secure, powerful, and custom-tailored AI workflows directly in your terminal.

## **The Heart of the CLI: The Tooling Ecosystem**

Ever wonder what happens when you ask Gemini to "read `main.py`"? It's not magic — it's **Function Calling**. This is the core mechanism that allows the AI model to request the execution of local code to interact with your environment.

Here’s the lifecycle of a prompt that uses a tool:

1.  **You:** "Refactor the function in `utils.js` to be more efficient."
2.  **CLI to Model:** The CLI sends your prompt to the Gemini API, along with a list of available tools (like `read_file`, `replace`, etc.) and their schemas.
3.  **Model to CLI:** The model determines it needs to read the file first. It responds not with text, but with a structured request: `functionCall: { name: 'read_file', args: { path: '/path/to/utils.js' } }`.
4.  **CLI Executes:** The CLI executes the `read_file` tool locally with the provided arguments.
5.  **CLI to Model:** The content of `utils.js` is sent back to the model as the result of the tool's execution.
6.  **Model to CLI:** Now, armed with the file's content, the model generates the refactored code and may request another tool call, like `replace`, to apply the changes.
7.  **You:** See the final, refactored code suggestion.

This entire back-and-forth orchestration is what makes the Gemini CLI so powerful.

## A Tour of the Built-in Toolkit

Gemini CLI comes loaded with a suite of essential tools to interact with your environment.

*   **File System Mastery:** The CLI provides a rich set of tools for file manipulation.
    *   **Tools:** `read_file`, `write_file`, `list_directory`, `glob` (for finding files with patterns), and the surgical `replace` tool.
    *   **Details:** The `replace` tool is particularly powerful. For precision, it's designed to work best when you provide it with several lines of context around the code you want to change, ensuring it modifies the exact right spot.
    *   **Example Prompt:** *"Use `glob` to find all `*.css` files in the `styles` directory. Then, read `theme.css` and `replace` the hex code `#333` with `#444` everywhere it appears."*

*   **System Interaction:** Execute shell commands without leaving the CLI.
    *   **Tool:** `run_shell_command`.
    *   **Details:** This gives the model the ability to run build scripts, check git status, or use other command-line utilities. For safety, it will almost always ask for your explicit confirmation before running.
    *   **Example Prompt:** *"Run the test suite for this project using `npm test` and then summarize any failures reported in the output."*

*   **Web Intelligence:** Ground the model in real-time information from the internet.
    *   **Tools:** `google_web_search` and `web_fetch`.
    *   **Details:** These tools allow Gemini to answer questions about current events, look up documentation, or summarize articles from a URL.
    *   **Example Prompt:** *"Please `web_fetch` the content of the article at [URL] and then `google_web_search` for more information on the main topic it discusses."*

Let's create a practical example that combines these tools:

### Deep Research Workflow

Let's imagine you're a developer tasked with researching the "impact of WebAssembly on server-side applications" to prepare for a new project.

**Step 1: Broad Exploration with Web Search**

You start with a general query to understand the landscape.

```text
> Search for the current state of WebAssembly (Wasm) on the server-side, focusing on performance and security benefits. 
```
- **Tool Used:** `google_web_search`
- **Action:** The CLI will perform a web search and return a generated summary with cited sources, giving you an overview and links to key articles, blogs, and official documentation.

**Step 2: Diving Deeper into Specific Sources**

From the search results, you find a few promising links, including a technical blog post and a whitepaper. You want to consume their content without leaving the terminal.

```text
> That's a good start. Please fetch the full content from the following two articles: <https://wasm.io/blog/case-study-1> and <https://some-research-site.com/wasm-on-server.pdf>
```

- **Tool Used:** `web_fetch`
- **Action:** Gemini will ingest the content from these URLs. It can now answer detailed questions about them, summarize them, or compare their arguments.

**Step 3: Storing Key Insights with Memory**

As you're reading, you discover a critical fact you want the AI to remember throughout your entire research session.

```text
> Remember that the Wasm sandbox model provides better security isolation than traditional containers for multi-tenant environments. This is a key advantage.
```

- **Tool Used:** `save_memory`
- **Action:** This fact is now saved to your global `GEMINI.md` file and will be part of the context for all subsequent prompts in this and future sessions, ensuring the AI keeps this important point in mind.

**Step 4: Branching Your Research without Losing Context**

You suddenly want to explore a side topic (e.g., Kubernetes integration) but don't want to derail your main research thread on performance and security.

```text
> /chat save wasm-main-thread

> Now, search specifically for "Wasm and Kubernetes integration patterns".
```

- **Commands Used:** `/chat save`, `google_web_search`
- **Action:** You've saved your entire conversation history. Now you can dive deep into the Kubernetes topic. When you're done, you can instantly return to where you were.

**Step 5: Synthesizing and Saving Your Research**

After exploring the side topic, you're ready to produce your final report.

```text
> /chat resume wasm-main-thread

> Based on our entire conversation, including the key advantage I asked you to remember, write a detailed summary of my research into a file named 'wasm_research_summary.md'. Include a section for performance, security, and a list of the URLs we visited for a bibliography.
```

- **Commands Used:** `/chat resume`, `write_file`
- **Action:** The CLI restores your original conversation. Then, it uses its full contextual understanding of everything you've discussed to generate a comprehensive Markdown report and save it directly to your file system.

You could also put this entire workflow into a script, allowing you to run it with a single command in the future. This is the power of Gemini CLI: it turns complex, multi-step tasks into simple, repeatable commands.

## Fort Knox for Your AI: Sandboxing

Executing AI-generated code and shell commands carries inherent risks. Gemini CLI provides a robust sandboxing system to isolate these operations from your host machine.

You can enable it via the `--sandbox` (`-s`) flag or `"sandbox": true` in your `settings.json`.

1.  **macOS Seatbelt:** On macOS, the CLI can use the built-in `sandbox-exec` utility, which restricts file system access.
2.  **Container-based (Docker/Podman):** For maximum isolation, the CLI can run tools inside a container. You can even provide your own custom `.gemini/sandbox.Dockerfile` to install project-specific dependencies (e.g., Python, Go, etc.) right into the execution environment. This means you can safely ask Gemini to compile and run code in a language you don't even have installed on your host machine!

## Infinite Extensibility with MCP Servers

What if you want to teach Gemini CLI how to interact with your company's internal APIs, a database, or a proprietary system? The answer is the **Model Context Protocol (MCP)**.

An MCP server is a separate application that exposes a set of custom tools to the Gemini CLI. You simply tell the CLI how to launch your server in `settings.json`, and it handles the rest.

Here's an example configuration to connect the official `Github MCP Server`, which provides tools for interacting with GitHub:

```json
{
    "mcpServers": {  
      "github": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-github"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN"
        }
      }
    }
}
```

With this configured, you can now make natural language requests like:

```text
> Find the 3 most recent open issues in the `google-gemini/gemini-cli` repo and add a comment to each asking for a status update.
```

The CLI will discover the `get_issues` and `add_comment` tools from your MCP server and orchestrate the calls. You can check the status of all connected servers with the `/mcp` command.

### Automated triage assistant

Now, let's take this a step further and create a practical example that showcases how to build a custom workflow using the Gemini CLI and an MCP server.
Let's say you have a backlog of GitHub issues in your project that need triaging. Manually going through each issue, reading descriptions, and applying labels can be tedious and time-consuming.

We will create a automated triage assistant that monitors your GitHub issues and automatically labels them based on their content, using the Github MCP server we just configured above.

**Step 1: Information Gathering - Surveying the Backlog**

First, you ask the assistant to get a high-level view of the work that needs to be done.

```text
> Using our GitHub tools, find all open issues in the 'corp/webapp' repo that currently have no labels.
```

- **Tool Used:** `github_get_issues` (a custom tool from your MCP server)
- **Action:** The CLI connects to your MCP server, which in turn queries the GitHub API. It returns a concise list of untriaged issues, for example:
    - `#123: Login button unresponsive on Safari`
    - `#124: Add dark mode theme`

**Step 2: Deep Dive and Analysis - Triaging a Single Issue**

Now you pick one issue and ask the agent to perform a detailed analysis.

```text
> Let's analyze issue #123. Get its full description and the user's comments. Based on the content, suggest appropriate labels from our standard set: 'bug', 'feature-request', 'ui-ux', 'backend', 'needs-investigation'.
```

- **Tool Used:** `github_get_issue_details` (custom tool)
- **Action:**
    1. The agent calls the tool to fetch the full text of the issue, including the original post and all comments.
    2. The agent uses its core language model capabilities to **analyze the sentiment and technical details**. It reads phrases like "unresponsive," "Safari," and "doesn't work when I click," to understand the problem.
    3. It then provides a synthesized recommendation:
    `"This issue appears to be a 'bug' affecting the 'ui-ux'. The user specifically mentions Safari, so it might also need the 'browser-specific' label."`

**Step 3: Taking Direct Action - Applying Labels and Commenting**

You agree with the agent's analysis and want to update the issue on GitHub.

```
> Excellent analysis. Apply the 'bug' and 'ui-ux' labels to issue #123. Then, add a comment thanking the user for their report and letting them know the team is looking into it.
```

- **Tools Used:** `github_add_labels_to_issue`, `github_add_comment_to_issue` (custom tools)
- **Action:**
    1. The CLI will stop and ask for your confirmation for the first action: `Apply labels 'bug', 'ui-ux' to issue #123? (y/n)`
    2. After you approve, it will ask for confirmation for the second action: `Add comment '...' to issue #123? (y/n)`
    3. With your approval, the agent executes these calls, and the issue is instantly updated on GitHub—no need to switch to your browser.

**Step 4: The Automation Loop - Powering Through the Queue**

This is where you turn a one-off action into a true workflow.

```text
> Great. Now, let's create a triage queue. Go through the original list of untriaged issues we found. For each one, fetch its details, suggest a set of labels, and then wait for me to approve applying them. Let's start with the first 5.
```

- **Tools Used:** A repeating sequence of `github_get_issue_details` and `github_add_labels_to_issue`.
- **Action:** The Gemini CLI now enters an interactive loop. It will:
    1. Fetch details for issue `#123`.
    2. Present its analysis and suggested labels.
    3. Wait for your `y/n` confirmation to apply them.
    4. **Automatically move to the next issue (`#124`)** and repeat the process.

You have effectively transformed a tedious, multi-click process into a streamlined, conversational workflow in your terminal, allowing you to triage dozens of issues in a fraction of the time. This demonstrates how Gemini CLI can orchestrate complex, multi-step tasks that involve both understanding and action.

## Final Thoughts

You are now equipped with the knowledge to not only use the Gemini CLI, but to shape and extend it. The true power of this platform is unlocked when you move beyond built-in features and start creating custom, automated workflows that solve your unique challenges. By combining custom tools via MCP, secure execution via sandboxing, and persistent context via `GEMINI.md`, you can build a development environment that is truly intelligent.