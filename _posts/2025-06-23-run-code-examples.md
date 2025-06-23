---
title: "Appendix A - Running the Code Examples"
date: "2025-06-23 17:30:00 +0200"
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

This appendix is your practical guide to setting up a development environment and running the code examples from this book. All the code is available in a companion [GitHub repository](https://github.com/iamulya/openai-agentsdk-code), designed to be run with minimal setup, allowing you to focus on experimenting with the concepts we've discussed.

## The Recommended Path: GitHub Codespaces

The fastest and most reliable way to get started is by using GitHub Codespaces. This feature provisions a complete, containerized development environment in the cloud that runs directly in your browser. All dependencies and configurations are handled for you automatically.

1.  **Launch the Codespace**: Navigate to the book's companion repository and click the **"Open in GitHub Codespaces"** button. This will begin the process of building your personal cloud environment. It may take a few minutes the first time.

2.  **Configure API Keys**: Once the Codespace has loaded, a terminal will appear at the bottom. The environment will automatically install all necessary Python packages.

    *   In the file explorer on the left, you will see a file named `.env.example`. Right-click this file and rename it to **`.env`**.
    *   Open the new `.env` file and paste in your secret API keys.

3.  **Run the Examples**: Your environment is now fully configured. You can proceed to the "Executing the Examples" section below to start running the code.

## Local Development: Using VS Code Dev Containers

If you prefer to work on your local machine but want to ensure a consistent and correct environment, VS Code's Dev Containers feature is the perfect alternative. It uses Docker to create the same containerized environment as Codespaces, but on your own computer.

**Prerequisites:**

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running.
-   [Visual Studio Code](https://code.visualstudio.com/) is required.
-   The [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) must be installed.

**Steps:**

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/iamulya/openai-agentsdk-code.git
    cd openai-agentsdk-code/src/tinib00k
    ```
    
2.  **Configure API Keys:** Just as with Codespaces, rename `.env.example` to `.env` and add your keys.
3.  **Open in Dev Container:**
    -   Open the cloned repository folder in VS Code.
    -   VS Code will detect the `.devcontainer` folder and show a notification in the bottom-right corner asking to "Reopen in Container". Click it.
    -   (If you miss the notification, open the command palette with `Ctrl+Shift+P` or `Cmd+Shift+P`, type `Dev Containers`, and select **"Dev Containers: Reopen in Container"**.)
4.  VS Code will build the Docker image and connect to it. This can take several minutes on the first run. Once complete, you will have a terminal inside the fully configured container, ready to run the examples.

## Executing the Examples

The `.env` file is the standard way to manage secret keys in a project. The `utils.py` helper script, used in each chapter's example, will automatically load these keys.

*   `GOOGLE_API_KEY`: Required for the Gemini models used in most examples.
*   `OPENAI_API_KEY`: Required for the SDK's built-in tracing functionality.
*   `ANTHROPIC_API_KEY`: Optional, only needed for the multi-provider example.

If a required key is missing, the script will raise an error with instructions.

The project is structured as a Python package. This means you must run the example files as modules to ensure that all imports resolve correctly.


> Make sure you are running the examples from the following directory: `openai-agentsdk-code/src/tinib00k`
{: .prompt-info }

For example, to run the basic agent example from the second chapter, you would use the following command:

```bash
python -m chapter2.basic_agent
```

Similarly, to run the Financial Research Bot from the case study:

```bash
python -m chapter10.financial_bot.main
```

You will often be prompted to type input directly into the terminal to interact with the agent systems. You can stop any running script by pressing `Ctrl+C`.


> Using the `-m` flag tells the Python interpreter to run a module as a script. This is crucial for our project structure. It adds the project's root directory to Python's path, allowing the interpreter to correctly find and import modules from other chapters or from the `utils.py` helper (e.g., `from utils import load_and_check_keys`).
{: .prompt-info }

## Understanding the Project Structure

The companion repository is organized to be clean and intuitive.

![*High-level overview of the companion repository structure.*](/assets/img/2025-06-23-run-code-examples/figure-1.png)


*   **/src/tinib00k/chapter.../**: Each folder contains the Python scripts corresponding to a chapter in this book.
*   **/src/tinib00k/utils.py**: The helper script to load and validate your API keys.
*   **/.devcontainer/**: Contains the `devcontainer.json` and `Dockerfile` that define the development environment for both Codespaces and local Dev Containers.
*   **pyproject.toml**: The standard Python project definition file, listing all dependencies.
*   **.env.example**: A template you should copy to `.env` to store your secret API keys.

## Viewing Your Traces

The Agents SDK includes a powerful tracing feature that logs the detailed execution of your agentic workflows. By default, these traces are sent to the OpenAI platform.

To view a trace:

1.  Ensure your `OPENAI_API_KEY` is correctly set in your `.env` file.
2.  Run any example. Many of them are wrapped in a `trace()` context manager.
3.  Look for a line in the console output similar to this:
    ```text
    View trace: https://platform.openai.com/traces/trace?trace_id=trace_...
    ```4.  Copy this URL and paste it into your web browser. You will be taken to a detailed, interactive Gantt chart visualizing the entire run, including every LLM call, tool execution, and handoff, along with their timings and data payloads.


> Even if your agents exclusively use other model providers like Gemini or Claude, the tracing feature **requires a valid OpenAI API key** to upload and host the trace data. If you do not have one or do not wish to use the feature, you can disable it globally by setting the environment variable `OPENAI_AGENTS_DISABLE_TRACING=1`.
{: .prompt-info }
