---
title: Appendix A - Run Code Examples
date: "2025-08-22 20:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

This appendix provides detailed instructions on how to set up your environment and run the code examples accompanying this book.

### Choosing Your Setup Method

You can set up your development environment in one of three ways. Using a container-based approach (**Option 1 or 2**) is highly recommended for a quick, consistent, and pre-configured setup.

#### Option 1: Using GitHub Codespaces (Easiest)

This method runs a fully configured development environment in the cloud, accessible through your browser. It is the fastest way to get started, as all prerequisites (Python, `uv`, Docker, Node.js) are pre-installed inside the container.

**Prerequisites:**

*   A GitHub account.

**Steps:**

1.  Navigate to the main page of the [code repository on GitHub](https://github.com/iamulya/adk-book-code).
2.  Click the green **`< > Code`** button.
3.  Select the **Codespaces** tab.
4.  Click **"Create codespace on main"**. GitHub will prepare the environment, which may take a few minutes.
5.  Once ready, the environment opens in a browser-based VS Code editor with the virtual environment automatically activated.
6.  Proceed to the **[Environment Variable Configuration](#environment-variable-configuration)** section below.

#### Option 2: Using a Local Dev Container

This method uses VS Code and Docker on your local machine to create the same consistent environment as Codespaces.

**Prerequisites:**

*   Visual Studio Code
*   The [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.
*   Docker Desktop installed and running on your machine.

**Steps:**

1.  Clone the repository to your local machine: 

    ```bash
    git clone https://github.com/iamulya/adk-book-code.git
    cd adk-book-code
    ```

2.  Open the cloned `adk-book-code` folder in VS Code.
3.  A notification will appear in the bottom-right corner asking to "Reopen in Container". Click it. (If not, open the Command Palette with `Ctrl+Shift+P` or `Cmd+Shift+P` and run `Dev Containers: Reopen in Container`).
4.  VS Code will build and start the container. The virtual environment will be automatically activated in the terminal.
5.  Proceed to the **[Environment Variable Configuration](#environment-variable-configuration)** section below.


> ## For Option 1 and Option 2 users
> When the setup is done for the *first time*, it might not have the Python virtual environment activated in the terminal open by default. Simply open a new terminal - there you should have the virtual environment automatically activated, verifiable by the presence of `(adk-book-code)` in your prompt. 
> {: .prompt-info }

#### Option 3: Manual Local Setup

Follow these steps only if you prefer to configure your local machine manually without using containers.

**Prerequisites:**

*   **Python**: Version 3.12 or higher.
*   **`uv`**: Install the package manager: `pip install uv`.
*   **Git**: For cloning the repository.
*   **Docker**: Docker Desktop or Engine must be installed and running for Chapter 9 examples.
*   **Node.js & npx**: Required for the MCP Filesystem server example in Chapter 8.

**Steps:**

1.  Clone the repository: 

    ```bash
    git clone https://github.com/iamulya/adk-book-code.git
    cd adk-book-code
    ```

2.  Create and activate a virtual environment using `uv`:

    ```bash
    uv venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scriptsctivate  # On Windows
    ```

3.  Install dependencies using the lock file for a reproducible build, then install the project in editable mode:

    ```bash
    uv pip sync uv.lock
    uv pip install --no-deps -e .
    ```

4.  Proceed to the **[Environment Variable Configuration](#environment-variable-configuration)** section below.

### Environment Variable Configuration

This step is **required for all setup options**. You must provide API keys and other secrets to the application.

*   **For GitHub Codespaces Users (Recommended Method)**: Instead of creating a `.env` file, you should use **Codespaces Secrets**. Go to your repository's **Settings > Secrets and variables > Codespaces** and add your secrets there. The dev container is configured to automatically expose these as environment variables. Check out the `.env.example` file to find all of the environment variables used in the code.

*   **For Local Dev Container & Manual Setup Users**: You will use a `.env` file in the project root.

#### Steps to configure the `.env` file:

1.  From the project root (`adk-book-code/`), copy the example file:

    ```bash
    cp .env.example .env
    ```

2.  Open the new `.env` file and add your actual secret values.

**Required Keys and Credentials:**

It is not necessary to enter values for all keys mentioned in the `.env.example` file. However, you **must** provide an API key for the LLM you intend to use. By default, examples use Google Gemini.

*   For **Gemini models via Google AI Studio (Recommended for easy start)**: Set `GOOGLE_API_KEY`. You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

*   For **OpenAI models via LiteLLM**: Set `OPENAI_API_KEY`.
*   For **other LLMs via LiteLLM**: Consult the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for the correct environment variable for your chosen model and key.

Make sure to update the default LLM variables in `src/building_intelligent_agents/utils.py` accordingly, if not using Gemini models. You only need to provide keys for the examples you intend to run. Refer to the `.env.example` file for the environment variables used in the book.

### Running the Examples

Once your environment is set up and your secrets are configured, you can run the code.

**Crucial Step:** All examples must be run from the `src/building_intelligent_agents` directory.

1.  **Navigate to the correct directory**:
    If you are in the project root (`adk-book-code`), change to the source directory:

    ```bash
    cd src/building_intelligent_agents
    ```

    **All subsequent commands assume you are in this `src/building_intelligent_agents` directory.**

#### Method 1: Running from the Command Line (`python -m ...`)

This method is suitable for simple, interactive command-line agents.

**Steps:**

1.  Ensure your virtual environment is active (it should be automatic in Codespaces/Dev Containers).

2.  Execute the module using its Python path:

    *Example: Running the Simple Assistant from Chapter 1*

    ```bash
    python -m chapter1.simple_assistant
    ```

#### Method 2: Using the ADK Dev UI (`adk web .`)

This is the best method for examples involving OAuth (like Google Calendar), or for visualizing complex tool calls, artifacts, and agent traces.

**Steps:**

1.  **Configure `__init__.py` for the desired agent**: To tell the Dev UI which agent to load, you must edit the `__init__.py` file inside the relevant chapter's directory. For example, to run the `calendar_agent` from Chapter 7:

    *   Open `src/building_intelligent_agents/chapter7/__init__.py`.
    *   Uncomment the line that imports `calendar_agent` as `root_agent` and ensure all other `root_agent` imports in that file are commented out.

        ```python
        # from .apihub_agent import apihub_connected_agent as root_agent
        from .calendar_agent import calendar_agent as root_agent # Ensure this line is active
        # from .openapi_petstore_agent import petstore_agent as root_agent
        # from .spotify_agent import spotify_agent as root_agent
        ```

2.  **Start the ADK Web Server** (from `src/building_intelligent_agents`):

    ```bash
    adk web .
    ```

3.  **Open in Browser**: Open the URL provided (usually `http://localhost:8000`). If in Codespaces, VS Code will prompt you to forward the port so you can open it in your local browser.

4.  **Interact**: In the Dev UI, select the chapter folder (e.g., `chapter7`) from the agent selector on the top-left. You can then interact with the agent defined as `root_agent` for that chapter.

### Troubleshooting

*   **`ModuleNotFoundError`**: Ensure your virtual environment is active and that you are in the `src/building_intelligent_agents` directory when running commands. If using manual setup, confirm you ran `uv pip install --no-deps -e .`. You can activate the virtual environment manually by running `source .venv/bin/activate` on Linux/macOS or `.venv\Scriptsctivate` on Windows.
*   **API Errors**: These are almost always due to missing, incorrect, or permission-denied API keys. Double-check your `.env` file or Codespaces Secrets.
*   **Docker Errors**: For Chapter 9 examples or when using a Local Dev Container, ensure Docker Desktop is running.

By following these instructions, you should be able to successfully run and explore the diverse range of intelligent agents and ADK features presented in this book.

Happy building with Google ADK!
