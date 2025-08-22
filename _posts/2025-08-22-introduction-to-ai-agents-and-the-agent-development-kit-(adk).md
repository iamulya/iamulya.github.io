---
title: Chapter 1 - Introduction to AI Agents and the Agent Development Kit (ADK) 
date: "2025-08-22 08:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Welcome to the world of intelligent agents! In recent years, Artificial Intelligence (AI) has made remarkable strides, moving beyond simple pattern recognition and into the realm of sophisticated decision-making and task execution. The evolution of Large Language Models (LLMs) has been a breathtaking journey from early statistical approaches and rule-based systems to the sophisticated deep learning architectures of today, particularly the transformative impact of the transformer model. This progression, marked by exponential increases in model size, training data, and algorithmic ingenuity, has unlocked unprecedented capabilities in natural language understanding, generation, and reasoning. 

As a result, LLMs are rapidly reshaping our world: they're democratizing content creation, revolutionizing how we access and synthesize information, powering more intuitive and capable virtual assistants, automating complex tasks from coding to customer service, and even accelerating scientific discovery by helping to analyze data and formulate hypotheses, heralding a new era of human-AI collaboration.

Building upon this foundation, Agentic AI represents the next logical and transformative step in the application of LLMs. Instead of merely responding to prompts or generating content on demand, agentic systems leverage LLMs as their core reasoning engine to autonomously pursue goals, make decisions, and take actions within digital or even physical environments. This involves endowing LLMs with capabilities like planning, tool use (e.g., accessing APIs, browsing the web, running code), memory, and self-correction. Consequently, we're seeing the emergence of AI Agents that can independently manage complex projects, conduct in-depth research, automate sophisticated workflows, or act as truly proactive personal assistants, moving far beyond passive information processing to become active participants and collaborators in achieving user-defined objectives.

## AI Agents vs. Agentic AI

Before moving further, there can be some confusion around the terms AI Agents and Agentic AI. Let's breakdown the difference between them.

| Feature                 | AI Agent                                     | Agentic AI                                                    |
| :---------------------- | :------------------------------------------- | :------------------------------------------------------------ |
| **Definition**          | A broad term for any system that perceives its environment, reasons, and acts to achieve goals. | A more specific term describing AI agents with a high degree of autonomy, proactivity, and sophisticated reasoning, typically LLM-powered. |
| **Scope**    | General category of AI systems.              | A sophisticated *type* or *quality* of AI agent.              |
| **Core Engine/Reasoning** | Can be rule-based, statistical, simple algorithms, machine learning models, or advanced AI. | Typically relies heavily on Large Language Models (LLMs) for core reasoning, planning, and decision-making. |
| **Level of Autonomy**   | Varies widely from low (reactive) to high.   | Characterized by **high autonomy** and proactivity.           |
| **Proactivity**         | Can be reactive or proactive to varying degrees. | Designed to be **highly proactive** in pursuing goals.        |
| **Key Capabilities**    | Basic perception, action, decision-making based on its design. | Emphasizes **planning, tool use (APIs, web, code), memory, self-correction, and goal decomposition.** |
| **Task Complexity**     | Can handle simple to complex tasks, depending on its specific design. | Geared towards handling **complex, multi-step, open-ended tasks** that require sustained effort and adaptation. |
| **Decision-Making**     | Can be predefined or learned, but often within narrower constraints. | Makes more nuanced, context-aware decisions, capable of dynamic replanning. |
| **Goal Pursuit**        | Achieves predefined goals.                   | Can break down high-level user-defined goals into actionable sub-goals and pursue them independently. |
| **Examples**            | Thermostat, spam filter, simple chatbot, game NPCs, robotic vacuum. | LLM-powered systems like Auto-GPT, BabyAGI, AI-driven project managers, advanced research assistants. |
| **Relationship**        | "Agentic AI" is a type of "AI Agent."       | All Agentic AI systems are AI agents, but not all AI agents exhibit the advanced qualities of "Agentic AI." |

In short:

-   **AI Agent** is the umbrella term.
-   **Agentic AI** refers to a new wave of highly capable AI agents, distinguished by their sophisticated reasoning (often from LLMs), autonomy, and ability to use tools to achieve complex goals.

This book is your guide to building all kinds of intelligent **AI agents** using Google's **Agent Development Kit (ADK)**, an open-source, code-first Python toolkit designed to empower developers with flexibility and control.

## AI Agents: Evolution and Capabilities

At their core, AI agents are systems that can:

1. **Perceive/Observe:** Gather information about their current state or environment. This could be user input, data from sensors, API responses, or the content of a document.
2. **Reason/Plan:** Process the perceived information, consult their knowledge, and formulate a plan or decide on the next best action to achieve a predefined goal or respond to a user's request. This often involves interaction with Large LanguageModels (LLMs).
3. **Act:** Execute the chosen action. This might involve calling an API, generating text, modifying a file, or interacting with another system.

The evolution of AI agents has been rapid, fueled by advancements in LLMs. Early agents were often rule-based and limited in scope. Today's agents, powered by LLMs like GPT, Gemini or Claude, exhibit far more complex capabilities:

- **Natural Language Understanding and Generation:** Engaging in human-like conversations.
- **Tool Use:** Interacting with external APIs, databases, and other software tools to gather information or perform actions beyond their inherent capabilities.
- **Planning and Reasoning:** Breaking down complex tasks into smaller, manageable steps.
- **Memory:** Remembering past interactions and context to inform future decisions.
- **Adaptability:** Learning (in some contexts) and adjusting strategies based on new information or feedback.

Imagine a customer service agent that can understand a user's complex query, look up order details from a database (using a tool), check shipping status via another API (another tool), and then compose a helpful, contextual response. Or consider a research assistant that can browse the web, summarize articles, and compile a report on a given topic. These are the kinds of sophisticated systems ADK helps you build.


> The "Perceive, Reason/Plan, Act" cycle is fundamental to understanding how most AI agents operate, including those built with ADK. Visualizing your agent's tasks in terms of this loop can help in designing its logic and tool interactions.
> {: .prompt-info }

## Why ADK? Core Philosophy: Code-first, Modularity, Flexibility

While various frameworks exist for building AI agents, Google's ADK stands out with its core philosophy centered around:

- **Code-First Development:** ADK empowers developers to define agent logic, tools, and orchestration directly in Python. This provides ultimate flexibility, testability, and version control, treating agent development more like traditional software engineering. Instead of relying heavily on GUIs or declarative configurations, you have the full power of Python at your fingertips.
- **Modularity:** ADK encourages breaking down complex agent systems into smaller, specialized, and reusable components. Agents, tools, and services can be developed and tested independently and then composed into larger applications.
- **Flexibility:**
    - **Model-Agnostic:** While optimized for Google's Gemini models, ADK is designed to work with various LLMs (e.g., Anthropic's Claude, models accessible via LiteLLM).
    - **Deployment-Agnostic:** Agents built with ADK can be deployed in various environments, from simple VMs to scalable cloud solutions like Google Cloud Run or Vertex AI Agent Engine.
    - **Framework Compatibility:** ADK aims for compatibility, allowing integration with tools and concepts from other popular frameworks like Langchain and CrewAI.

ADK is not just about building a single chatbot; it's about engineering robust, maintainable, and scalable AI agent systems.


> ## Best Practice: Embrace Code-First for Complex Agents
> 
> While visual builders have their place, ADK's code-first approach shines for complex, production-grade agents. It allows for robust testing, version control (e.g., with Git), and easier integration into existing software development lifecycles. Treat your agent code like any other critical software component.
> {: .prompt-info }

## Key Features and Advantages of ADK

Building on its core philosophy, ADK offers several key features:

- **Rich Tool Ecosystem:**
    - Utilize pre-built tools for common tasks (e.g., Google Search, web page loading).
    - Easily create custom tools by wrapping Python functions.
    - Generate tools from OpenAPI specifications for interacting with REST APIs.
    - Integrate tools from Google's API Hub, MCP (Model Context Protocol) Toolbox, and Application Integration.
    - Leverage tools from other frameworks like Langchain and CrewAI.
- **Modular Multi-Agent Systems:**
    - Design scalable applications by composing multiple specialized agents into flexible hierarchies or sequences.
    - Facilitates clear separation of concerns, making systems easier to understand, debug, and extend.
- **Code-First Development Paradigm:**
    - Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
    - Promotes software engineering best practices in agent development.
- **Deployment Flexibility:**
    - Easily containerize and deploy agents anywhere, including serverless platforms like Google Cloud Run.
    - Scale seamlessly with enterprise-grade solutions like Vertex AI Agent Engine (conceptual focus in ADK, with integrations).
- **Comprehensive Session and State Management:**
    - Robust mechanisms for managing conversation history and agent state, both in-memory and with persistent backends (databases, Vertex AI).
- **Built-in Evaluation Framework:**
    - Tools and CLI commands to define evaluation datasets and assess agent performance on various metrics.
- **Developer-Friendly UI:**
    - An integrated web UI for local development, testing, debugging, and showcasing your agents.
    

> ## Avoid Over-Reliance on a Single "Mega-Agent"
> 
> While ADK allows for powerful single agents, its modularity encourages breaking down complex problems into smaller, specialized agents. Avoid the temptation to build one monolithic agent that tries to do everything; this often leads to systems that are hard to debug, maintain, and scale.
> {: .prompt-info }

## ADK vs. LangGraph

Let's do a quick comparison of ADK and LangGraph, which is one of the most popular framework for AI Agent development.

ADK seems to position itself as a comprehensive, "code-first" Python toolkit for the entire lifecycle of agent development (building, evaluating, deploying), with a strong emphasis on flexibility, control, and integration within the Google ecosystem, while remaining model-agnostic.

LangGraph, on the other hand, is an extension of LangChain specifically designed for building stateful, multi-actor applications with LLMs by representing them as graphs. It excels at creating cyclical and complex control flows.

### How ADK Differs Specifically from LangGraph:

1. **Scope:** ADK is a broader toolkit concerned with the entire agent lifecycle, including scaffolding, evaluation, deployment, and a dedicated dev UI. LangGraph is more focused on the *runtime logic and control flow* of how an agent or group of agents operate, particularly for complex, stateful interactions. LangGraph has extension projects like LangGraph Platform and LangGraph Studio which aim to enhance the developer and operations experience.
2. **Orchestration Paradigm:**
    - ADK often uses a more declarative, hierarchical approach for multi-agent systems (`sub_agents` and shell agents like `SequentialAgent`). The `LlmAgent`'s `AutoFlow` handles LLM-driven agent transfers.
    - LangGraph provides a more granular, explicit way to define any arbitrary control flow using nodes and edges, making it very powerful for custom loops and conditional logic beyond predefined patterns.
3. **Opinionation vs. Granularity:**
    - ADK provides higher-level abstractions like the `Agent` class and predefined shell agents, which can make common patterns quicker to implement. It offers a "code-first software development feel."
    - LangGraph offers finer-grained control over the execution flow, which is ideal for highly custom or experimental agent architectures.
4. **Ecosystem Focus:** While ADK aims to be model-agnostic and compatible, its tooling and deployment options show a strong leaning towards and optimization for the Google Cloud ecosystem (Gemini, Vertex AI). LangGraph is inherently tied to the LangChain ecosystem.

**Key Differences Summarized:**

| Feature | Agent Development Kit (ADK) | LangGraph |
| --- | --- | --- |
| **Primary Focus** | Full lifecycle (build, evaluate, deploy) of sophisticated, modular AI agents; "software dev feel." | Building stateful, multi-actor applications with complex, cyclical control flow. |
| **Core Abstraction** | Code-first Python classes (`Agent`, `LlmAgent`, shell agents), services, CLI, Dev UI. | Supervisor/Swarm/React Agent, Stateful graphs (nodes and edges). |
| **Orchestration** | Hierarchical multi-agent systems, shell agents (`Sequential`, `Parallel`, `Loop`), `AutoFlow` with agent transfer, callbacks. | Explicit graph-based control flow, conditional edges, cycles. |
| **State Management** | `SessionService`, `State` objects, modified via callbacks or `output_key`. | Central to the graph; state explicitly passed and updated between nodes; checkpointers. |
| **Tooling** | Rich ecosystem, OpenAPI, custom, LangChain/CrewAI wrappers, strong Google focus (Vertex, BigQuery). | Leverages LangChain's extensive tool ecosystem. Nodes can be tools. |
| **Evaluation** | Built-in `adk eval` CLI and evaluation framework. | Relies on LangChain's evaluation tools (LangSmith) or custom setups. |
| **Deployment** | `adk deploy` to Cloud Run, Vertex AI Agent Engine. `adk api_server` to run ADK anywhere. | Langgraph Platform and deployable where LangChain apps run (e.g., LangServe, custom). |
| **Ecosystem** | Strong Google Cloud integration; aims for compatibility with other frameworks. | Deeply integrated with LangChain. |
| **Integration Point** | ADK *can use* LangGraph. | LangGraph itself is a way to structure the "brain" or logic of an agent. |

Crucially, they are **not strictly mutually exclusive**.

The `LangGraphAgent` class within ADK suggests that ADK can leverage LangGraph as one way to define a component agent within its broader framework. You could build a complex reasoning loop using LangGraph and then wrap that LangGraph agent as an LangGraphAgent to be used within an ADK multi-agent system, benefiting from ADK's deployment and evaluation tools.

## Overview of the ADK Architecture

Understanding the main components of ADK is key to effectively using it.

![*Diagram: Main components in the ADK Architecture*](/assets/img/2025-08-22-introduction-to-ai-agents-and-the-agent-development-Kit-(adk)/figure-1.png)


- **Runner (`google.adk.runners.Runner`):** The engine that executes agents. It manages the interaction flow, session state, and communication between the user, agents, and various services.
- **Agent (`google.adk.agents.BaseAgent`, `LlmAgent`):** The core building block. An agent encapsulates logic, its connection to an LLM, and the tools it can use. `LlmAgent` is the primary class for building LLM-powered agents.
- **Tools/Toolsets (`google.adk.tools`):** Extend an agent's capabilities. These can be simple Python functions, integrations with external APIs (via OpenAPI, API Hub, etc.), or specialized functionalities like search or code execution. Toolsets group related tools.
- **LLM Models (`google.adk.models`):** Abstractions for interacting with different Large Language Models (e.g., Gemini, Claude). ADK provides a registry for managing these models.
- **Session Service (`google.adk.sessions.BaseSessionService`):** Manages the lifecycle of a conversation (session), including storing and retrieving conversation history (events) and agent state. Implementations exist for in-memory, database, and Vertex AI-managed sessions.
- **Artifact Service (`google.adk.artifacts.BaseArtifactService`):** Manages the storage and retrieval of files (artifacts) generated or used by agents, such as images, documents, or code outputs.
- **Memory Service (`google.adk.memory.BaseMemoryService`):** Provides agents with the ability to store and retrieve information over the long term, going beyond a single session's context.
- **ADK CLI & Dev UI:** Command-line tools and a web-based UI to aid in agent creation, local development, testing, and evaluation.


> ## Services are Pluggable
> 
> A key architectural strength of ADK is that services like SessionService, ArtifactService, and MemoryService are defined by interfaces (Base...Service). This means you can start with simple in-memory versions for development and later swap them out for persistent, cloud-based implementations (like DatabaseSessionService or GcsArtifactService) without changing your core agent logic.
> {: .prompt-info }

## A Glimpse of a Simple ADK Agent

Let's look at a very basic ADK agent to illustrate the code-first nature. Don't worry about understanding every detail yet; we'll cover these components thoroughly in later chapters.

```python
from google.adk.agents import Agent 
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM

load_environment_variables()

simple_assistant_agent = Agent(
    name="simple_assistant_agent",
    model=DEFAULT_LLM,
    instruction="You are a friendly and helpful assistant. Be concise.",
    description="A basic assistant to answer questions.",
)

if __name__ == "__main__":
    print("Initializing Simple Assistant...")
    runner = InMemoryRunner(agent=simple_assistant_agent, app_name="MySimpleApp")

    current_session_id = "my_first_session"
    current_user_id = "local_dev_user"

    create_session(runner, current_session_id, current_user_id)

    print("Simple Assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting Simple Assistant. Goodbye!")
            break
        if not user_input.strip():
            continue

        user_message = Content(parts=[Part(text=user_input)], role="user")
        print("Assistant: ", end="", flush=True)
        try:
            for event in runner.run(
                user_id=current_user_id,
                session_id=current_session_id,
                new_message=user_message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print()
        except Exception as e:
            print(f"
An error occurred: {e}")
```

> As mentioned in the Preface, check out Appendix A to get started on running the code examples in your local environment.
> {: .prompt-info }

Upon running the code in the CLI, you should see:
    
```bash
Initializing Simple Assistant...
Simple Assistant is ready. Type 'exit' to quit.
You:
```
    
Now you can type your questions or prompts. For example:
    
```bash
You: Hello, who are you?
Assistant: I am a friendly and helpful assistant.
You: What is the capital of France?
Assistant: The capital of France is Paris.
You: exit
Exiting Simple Assistant. Goodbye!
```


> All the code that you see in the `if __name__ == "__main__"` condition in **all of the examples in this book** is only there so that you can run the example through CLI simply using `python -m`, without using the `adk` commands. One of the main aims of this book is to show the main ADK components involved in running your agents and how the control flows. However, when you use ADK CLI commands like `adk web`, `adk run` or `adk api_server`, all you need to run is the agent definition from the example. The creation of runner, session, event handling etc. is handled by ADK.
> 
> ```python
> from dotenv import load_dotenv
> from google.adk.agents import Agent 
> from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
> 
> load_environment_variables()
> 
> simple_assistant_agent = Agent(
>     name="simple_assistant_agent",
>     model=DEFAULT_LLM,
>     instruction="You are a friendly and helpful assistant. Be concise.",
>     description="A basic assistant to answer questions.",
> )
> ```
> This shows how easy it is to create a functioning agent in ADK.
> 
> {: .prompt-info }

**What's Next?**

Next we'll dive into setting up your ADK development environment more formally, exploring the ADK Command Line Interface (CLI) for creating and managing projects, and getting comfortable with the ADK Development UI for a more interactive development experience. This will lay the groundwork for building more complex and powerful agents.
