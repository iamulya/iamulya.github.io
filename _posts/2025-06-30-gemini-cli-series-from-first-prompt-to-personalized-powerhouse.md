---
title: "Gemini CLI Series: From First Prompt to Personalized Powerhouse"
date: 2025-06-30 11:00:00 +0100
categories: [Gen AI, AI Coding Agents, Gemini CLI]
tags: [Gen AI, Gemini, Gemini CLI, AI Coding Agents, Gemini CLI Series]
image:
  path: /assets/img/gemini-cli.png
  alt: "Gemini CLI"
---

Welcome to the new frontier of developer productivity! In a world where AI is reshaping how we write, debug, and manage code, the terminal remains the developer's essential workspace. The **Gemini CLI** bridges these two worlds, bringing Google's powerful AI models directly to your command line.

This isn't just another chatbot. It's an interactive, extensible, and context-aware tool designed to understand your code, automate your workflows, and become an indispensable part of your development loop.

In this first post, I'll guide you through your initial journey with Gemini CLI, from installation to creating a personalized and powerful experience tailored to your exact needs.

## Getting Started: Installation and Authentication

Getting started is refreshingly simple. The recommended way to begin is with a single `npx` command, which runs the latest version without a permanent installation.

**Prerequisites:** Ensure you have [Node.js version 18](https://nodejs.org/en/download) or higher.

```bash
npx https://github.com/google-gemini/gemini-cli
```
Or, for a global install:
```bash
npm install -g @google/gemini-cli
gemini
```

On your first run, you'll be greeted with two setup steps:
1.  **Theme Selection:** Pick a color scheme that suits your terminal.
2.  **Authentication:** You'll be prompted to log in with your Google account. This is a one-time setup that caches your credentials securely for future sessions.

**How you log in matters.** Your choice of authentication method impacts API limits and how your data is used. Gemini CLI offers several options:

*   **Login with Google (Default):** Uses your personal Google account. Your prompts and code may be used to improve Google's products, including model training. This is the quickest way to get started.
*   **Gemini API Key:** Generate a key from [Google AI Studio](https://aistudio.google.com/apikey). If you have an account in the paid tier, your data is **not** used for model training. Set it in your environment: `export GEMINI_API_KEY="YOUR_API_KEY"`.
*   **Vertex AI / Workspace:** For enterprise users, your data is treated as confidential and is **not** used for training. This typically requires setting a `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and `GOOGLE_GENAI_USE_VERTEXAI` environment variable.

Refer to the official [Gemini CLI documentation](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/authentication.md) for detailed instructions on each method.

> Always refer to the official [Terms of Service and Privacy Notice](https://github.com/google-gemini/gemini-cli/blob/main/docs/tos-privacy.md) to understand the policy for your chosen method.
{: .prompt-info }

With that, you're in! Let's try your first prompt:

```text
> Give me a simple "Hello, World" example in Rust.

✦ Of course. Here is a simple "Hello, World!" program in Rust:


   1 fn main() {
   2     println!("Hello, World!");
   3 }


  To run this code:


   1. Save it to a file named main.rs.
   2. Compile it using the Rust compiler: rustc main.rs
   3. Run the resulting executable: ./main
```

## The Interactive Experience

The Gemini CLI operates in a Read-Eval-Print Loop (REPL), making it feel like a persistent conversation. Beyond simple Q&A, you have a powerful set of commands at your fingertips.

*   **Slash (`/`) Commands:** These are your meta-commands for controlling the CLI itself. Type `/help` to see them all, but some essentials include:
    *   `/mcp`: List configured mcp servers and tools.
    *   `/tools`: List all available tools, including custom ones from your MCP server.
    *   `/memory`: Manage your project's context with `GEMINI.md` files.
    *   `/chat save <tag>` & `/chat resume <tag>`: Save and restore conversation branches.
    *   `/stats`: See your token usage for the session.

*   **The `@` Command:** This is where the magic begins. You can inject the content of files or entire directories directly into your prompt. The CLI's internal `read_many_files` tool will intelligently handle this, even respecting your `.gitignore` file by default.

```text
# Ask a question about your project's contribution guidelines
> Summarize the main points of @CONTRIBUTING.md

# Get a high-level overview of your source code
> Describe the architecture of the code in the @src/ directory.
```

>  **Shell Passthrough (`!`):** Need to run a quick `git status` or `ls -la`? No need to exit. Just prefix your command with `!` to execute it directly in your system's shell.
{: .prompt-info }

## Mastering Context with `GEMINI.md`

This is the feature that elevates Gemini CLI from a tool to a true copilot. You can provide persistent, project-specific instructions to the AI by creating `GEMINI.md` files. The CLI cleverly loads these files in a hierarchical manner:

1.  **Global:** `~/.gemini/GEMINI.md` (for your personal defaults).
2.  **Project Root:** `/path/to/your/project/GEMINI.md` (for project-wide rules).
3.  **Sub-directory:** `/path/to/your/project/src/component/GEMINI.md` (for component-specific instructions).

An example `GEMINI.md` at your project root might look like this:

```markdown
# Typesript Project Guidelines

## General Instructions:
- When generating new TypeScript code, please follow the existing coding style.
- All new functions and classes must have JSDoc comments.
- Prefer functional programming paradigms where appropriate.

## Coding Style:
- Use 2 spaces for indentation.
- Interface names should be prefixed with `I`.
```

You can manage this "instructional memory" with the `/memory` command (`show`, `refresh`, `add`).

And as a safety net, the **Checkpointing** feature automatically saves a snapshot of your project before any file-modifying tools are run, letting you undo changes with ease using `/restore`.

## Personalizing with `settings.json`

For ultimate control, the CLI uses `settings.json` files, located in your project's `.gemini/` directory (for project-specific settings) or `~/.gemini/` (for global user settings).

Here's a taste of what you can configure:

```json
{
  "theme": "GitHub",
  "autoAccept": true,
  "fileFiltering": {
    "respectGitIgnore": true
  },
  "checkpointing": {
    "enabled": true
  }
}
```

This configuration sets your theme, automatically approves safe, read-only tool calls, and enables the `/restore` command. Check out the [full documentation](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md) for all available settings.

> **Agent mode (Currently in Preview)**: The Gemini Code Assist agent mode in VS Code is also powered by Gemini CLI. Simply add `"geminicodeassist.updateChannel": "Insiders"` to your VS Code User Settings, after you have set up Gemini Code Assist. After this simply click the Agent toggle in Gemini Code Assist chat and you are good to go.
> 
> ![Gemini Code Assist Agent Mode](/assets/img/agent-preview.png)
> 
> Please note the use of the insider channel is currently required to use the agent mode in Gemini Code Assist. This will be available in the stable channel in the future.
{: .prompt-info }

#### **Next Steps**

You've now journeyed from a first-time user to a power user, capable of customizing the CLI's look, feel, and AI context. But this is just the beginning.

In the next post, [Gemini CLI Series: Architecting Secure, Extensible AI Workflows](https://iamulya.one/posts/gemini-cli-series-architecting-secure-extensible-ai-workflows), we will dive under the hood to explore the powerful tooling ecosystem, security sandboxing, and how to extend the CLI with your own custom capabilities, while providing you detailed examples of creating AI workflows.

