---
title: Chapter 17 - Session Management and State Persistence 
date: "2025-08-22 16:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

For AI agents to engage in meaningful, multi-turn conversations and provide personalized experiences, they need to remember past interactions and maintain contextual information. ADK addresses this through its **Session Management** system, centered around the `Session` object and the `BaseSessionService` interface.

This chapter explores how ADK defines and manages conversational sessions, how agent state is handled and scoped, and the different `SessionService` implementations available for persisting this vital information, from simple in-memory storage to robust database solutions and cloud-managed services.

## The `BaseSessionService` Interface

The `google.adk.sessions.BaseSessionService` is an abstract base class that defines the contract for all session management services in ADK. It provides a standardized way for the `Runner` to interact with the underlying session storage, regardless of its actual implementation.

**Key Abstract Methods of `BaseSessionService`:**

- **`create_session(app_name, user_id, state=None, session_id=None) -> Session`**: Creates a new session for a given application and user. Allows for an optional client-provided `session_id` and initial `state`.
- **`get_session(app_name, user_id, session_id, config=None) -> Optional[Session]`**: Retrieves an existing session. The `config` (a `GetSessionConfig` object) can specify how much history to load (e.g., number of recent events, events after a certain timestamp).
- **`list_sessions(app_name, user_id) -> ListSessionsResponse`**: Lists all sessions for a given application and user (typically returns session metadata without full event history).
- **`delete_session(app_name, user_id, session_id)`**: Deletes a specified session.
- **`append_event(session: Session, event: Event) -> Event`**: (This method has a default implementation in `BaseSessionService` that updates the in-memory session object, but persistent services override it to also write to storage). Appends a non-partial `Event` to the session and updates the session's state based on `event.actions.state_delta`.

By programming against this interface, the ADK `Runner` and your agent logic remain decoupled from the specifics of session storage.

![*Diagram: `BaseSessionService` interface and its concrete implementations.*](/assets/img/2025-08-22-session-management-and-state-persistence/figure-1.png)


## The `Session` Object: Structure and Lifecycle

The `google.adk.sessions.Session` Pydantic model is the core data structure representing a single conversation or interaction sequence.

**Key Attributes of `Session`:**

- **`id: str`**: A unique identifier for this specific session.
- **`app_name: str`**: The name of the application this session belongs to (e.g., "MyChatbot", "CustomerSupportAgent"). Used for namespacing sessions.
- **`user_id: str`**: An identifier for the user engaging in the session.
- **`events: list[Event]`**: A chronological list of all `Event` objects that have occurred in this session (user inputs, agent responses, tool calls, etc.). This forms the conversation history.
- **`state: dict[str, Any]`**: A dictionary holding key-value pairs representing the current state associated with this session. This can include data set by agents, tools, or callbacks.
- **`last_update_time: float`**: A Unix timestamp indicating when the session was last modified. This is crucial for optimistic locking or detecting stale sessions if multiple processes might interact with the same session.

**Lifecycle of a Session (Managed by `Runner` via `SessionService`):**

1. **Creation:** When a user starts a new interaction and no `session_id` is provided (or a new one is desired), `session_service.create_session()` is called by the `Runner`.
2. **Retrieval:** For subsequent turns in an ongoing conversation, `session_service.get_session()` is called with the existing `session_id` to load its history and state.
3. **Interaction & Updates:**
    - The `Runner` processes a new user message.
    - The `InvocationContext` is populated with the loaded `Session` object.
    - Agents and tools run, potentially modifying `context.state` (which updates `event.actions.state_delta`) and generating new `Event`s.
    - For each non-partial `Event` generated, `session_service.append_event(session, event)` is called. This method:
        - Appends the `Event` to the `session.events` list (in memory).
        - Merges `event.actions.state_delta` into `session.state` (in memory).
        - **Crucially, for persistent services, it also writes the new event and updated state to the backend storage.**
        - Updates `session.last_update_time`.
4. **Termination/Deletion (Optional):** Sessions might naturally expire or be explicitly deleted via `session_service.delete_session()`.

![*Diagram: Session lifecycle during a `Runner.run_async()` call.*](/assets/img/2025-08-22-session-management-and-state-persistence/figure-2.png)


## Working with `State`: App-level, User-level, and Session-level

The `session.state` dictionary is a versatile key-value store. ADK supports a simple namespacing convention using prefixes to define the scope and persistence characteristics of state variables, especially when using persistent `SessionService` implementations like `DatabaseSessionService`.

- **Session-Scoped State (Default):**
    - Keys *without* a special prefix (e.g., `my_var`, `current_task_id`).
    - Stored directly within the specific `Session` object.
    - Persists only for the duration of that single session.
    - **Example:** `context.state["current_search_results"] = results`
- **User-Scoped State (Prefix: `user:`)**:
    - Keys prefixed with `"user:"` (e.g., `user:theme_preference`, `user:language_code`).
    - Intended to store information specific to a `user_id` that should persist across *multiple sessions* for that user within the same `app_name`.
    - Persistent `SessionService` implementations (like `DatabaseSessionService`) store this in a separate table or mechanism associated with the `(app_name, user_id)`.
    - When a session is loaded, the `Runner` (or the `SessionService` itself) merges this user-scoped state into the `session.state` object, making it accessible as `context.state["user:theme_preference"]`.
    - **Example:** `context.state["user:preferred_language"] = "fr"`
- **Application-Scoped State (Prefix: `app:`)**:
    - Keys prefixed with `"app:"` (e.g., `app:api_version_info`, `app:global_feature_flags`).
    - Intended for information global to the `app_name` that should persist across *all users and all sessions*.
    - Stored by persistent services in a way that's accessible to all sessions of that app.
    - Merged into `session.state` when a session is loaded.
    - **Example:** `context.state["app:system_announcement"] = "Maintenance tonight"`
- **Temporary State (Prefix: `temp:`)**:
    - Keys prefixed with `"temp:"` (e.g., `temp:current_tool_call_id`, `temp:is_first_turn`).
    - This state is **not persisted** by `DatabaseSessionService` or `VertexAiSessionService` even if it's in `event.actions.state_delta`.
    - It's useful for transient information needed only within a single `runner.run_async()` invocation (i.e., across multiple LLM calls or tool uses within one user turn) but not meant to be saved permanently.
    - `InMemorySessionService` will store it for the lifetime of the Python process.
    - **Example:** `tool_context.state["temp:last_api_retry_count"] = 1`

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import InMemoryRunner
from google.adk.sessions.state import State # For prefix constants
from google.genai.types import Content, Part
import asyncio

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()  # Load environment variables for ADK configuration

def manage_preferences(tool_context: ToolContext, theme: str, language: str = "") -> dict:
    """Sets or gets user and app preferences."""
    changes = {}
    if theme:
        tool_context.state[State.USER_PREFIX + "theme"] = theme # user:theme
        changes["user_theme_set"] = theme
    if language:
        tool_context.state[State.APP_PREFIX + "default_language"] = language # app:default_language
        changes["app_language_set"] = language

    # Example of session-specific state
    tool_context.state["last_preference_tool_call_id"] = tool_context.function_call_id
    # Example of temporary state
    tool_context.state[State.TEMP_PREFIX + "last_tool_name"] = "manage_preferences"

    return {
        "status": "Preferences updated.",
        "changes_made": changes,
        "current_user_theme": tool_context.state.get(State.USER_PREFIX + "theme"),
        "current_app_language": tool_context.state.get(State.APP_PREFIX + "default_language"),
        "session_specific_info": tool_context.state.get("last_preference_tool_call_id")
    }

preference_tool = FunctionTool(func=manage_preferences)

state_demo_agent = Agent(
    name="preference_manager",
    model=DEFAULT_LLM,
    instruction="Manage user and application preferences using the 'manage_preferences' tool. "
                "You can set a user's theme or the app's default language.",
    tools=[preference_tool]
)

if __name__ == "__main__":
    # Using InMemorySessionService for this demo.
    # Scoped state behavior is fully realized with persistent services like DatabaseSessionService.
    runner = InMemoryRunner(agent=state_demo_agent, app_name="PrefsDemo")

    user1_id = "user_alpha"
    user2_id = "user_beta"
    session1_user1_id = "s1_alpha"
    session2_user1_id = "s2_alpha" # Different session for same user
    session1_user2_id = "s1_beta"

    create_session(runner, user_id=user1_id, session_id=session1_user1_id)
    create_session(runner, user_id=user1_id, session_id=session2_user1_id)
    create_session(runner, user_id=user2_id, session_id=session1_user2_id)

    async def run_and_print_state(user_id: str, session_id: str, prompt: str, app_name="PrefsDemo"):
        print(f"
--- Running for User: {user_id}, Session: {session_id} ---")
        print(f"YOU: {prompt}")
        user_message = Content(parts=[Part(text=prompt)], role="user")
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.author == state_demo_agent.name and event.content and event.content.parts[0].text:
                 if not event.get_function_calls() and not event.get_function_responses():
                    print(f"AGENT: {event.content.parts[0].text.strip()}")

        # Inspect state after the run
        # With InMemorySessionService, app and user scopes are emulated within its single dict.
        # DatabaseSessionService would store them in separate tables.
        s = await runner.session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        print(f"  Session State for {session_id}: { {k:v for k,v in s.state.items() if not (k.startswith(State.APP_PREFIX) or k.startswith(State.USER_PREFIX))} }")
        print(f"  User-Scoped State for {user_id} (via session merge): { {k:v for k,v in s.state.items() if k.startswith(State.USER_PREFIX)} }")
        print(f"  App-Scoped State for {app_name} (via session merge): { {k:v for k,v in s.state.items() if k.startswith(State.APP_PREFIX)} }")
        print(f"  Temp state (would not persist in DB): { {k:v for k,v in s.state.items() if k.startswith(State.TEMP_PREFIX)} }")

    async def main():

        # User Alpha, Session 1: Set theme and app language
        await run_and_print_state(user1_id, session1_user1_id, "Set my theme to 'dark' and app language to 'English'.")

        # User Alpha, Session 2: Check theme (should persist for user) and app language (should persist for app)
        await run_and_print_state(user1_id, session2_user1_id, "What's my theme and the app language?")

        # User Beta, Session 1: Set their theme, check app language (should be what Alpha set)
        await run_and_print_state(user2_id, session1_user2_id, "Set my theme to 'light'. What's the app language?")

        # User Alpha, Session 1 (again): Check theme (should still be dark)
        await run_and_print_state(user1_id, session1_user1_id, "Just checking my theme again.")

    asyncio.run(main())
```


> ## Best Practice: Scoped State for Personalization and Configuration
> 
> - Use **user-scoped state** (`user:key`) for user preferences, past summaries relevant to that user, or any data that should follow the user across different conversations.
> - Use **app-scoped state** (`app:key`) for global configurations, system-wide announcements, or data shared among all users of the application.
> - Use **session-scoped state** (no prefix) for context relevant only to the current ongoing conversation.
> {: .prompt-info }


> ## State Merging Order and Overwrites
> 
> When a session is loaded, ADK (or the SessionService implementation) merges these states into the session.state object. Typically, session-specific values can override user-scoped values, and user-scoped can override app-scoped values if keys conflict (though using distinct keys is better). Be aware of this potential if you use identical keys across scopes. DatabaseSessionService manages these scopes in distinct tables, merging them on load.
> {: .prompt-info }

## Session Service Implementations

ADK offers several `BaseSessionService` implementations:

**1. `InMemorySessionService`:**

- Stores all session data (sessions, events, state for all scopes) in Python dictionaries in memory.
- **Pros:** Fastest, no external dependencies, perfect for local development, testing, and examples. Used by default in `InMemoryRunner`.
- **Cons:** Data is lost when the Python process ends. Not suitable for production or any scenario requiring persistence.
- Emulates app/user scopes within its single internal dictionary structure.

**2. `DatabaseSessionService` (`google.adk.sessions.database_session_service`):**

- Persists session data to a SQL database using [SQLAlchemy](https://www.sqlalchemy.org/) as the ORM.
- **Pros:**
    - Robust, persistent storage.
    - Supports various SQL backends (SQLite, PostgreSQL, MySQL, etc.).
    - Properly separates app, user, and session state into different database tables for true scoping.
- **Cons:**
    - Requires a database setup and SQLAlchemy installation (`pip install sqlalchemy psycopg2-binary` for PostgreSQL, `pip install sqlalchemy pymysql` for MySQL, etc.).
    - Slightly more overhead than in-memory.
- **Initialization:**
The service defines tables for `sessions`, `events`, `app_states`, and `user_states`.
    
    ```python
    from google.adk.sessions import DatabaseSessionService
    
    # For SQLite (creates a file `my_app_sessions.db` in the current directory)
    db_url_sqlite = "sqlite:///./my_app_sessions.db"
    db_session_service_sqlite = DatabaseSessionService(db_url=db_url_sqlite)
    
    # For PostgreSQL (example, replace with your actual connection string)
    # db_url_postgres = "postgresql+psycopg2://user:password@host:port/database"
    # db_session_service_postgres = DatabaseSessionService(db_url=db_url_postgres)
    
    # For MySQL (example)
    # db_url_mysql = "mysql+pymysql://user:password@host:port/database"
    # db_session_service_mysql = DatabaseSessionService(db_url=db_url_mysql)
    
    ```
    

> ## Best Practice: DatabaseSessionService for Production with Relational DBs
> 
> If you need persistent sessions and are using a relational database, DatabaseSessionService is a solid choice. SQLite is great for single-process local persistence, while PostgreSQL or MySQL are suitable for production deployments.
> {: .prompt-info }

**3. `VertexAiSessionService` (`google.adk.sessions.vertex_ai_session_service`):**

- Leverages Google Cloud Vertex AI for managed session storage. This is often used when deploying agents to Vertex AI Agent Engine or similar Google Cloud managed environments.
- **Pros:**
    - Fully managed, scalable, and integrated with the Google Cloud ecosystem.
    - No need to manage your own database infrastructure.
- **Cons:**
    - Ties your session storage to Google Cloud.
    - Requires GCP project setup, Vertex AI API enabled, and appropriate authentication/permissions.
- **Initialization:**
It interacts with the Vertex AI "Reasoning Engines" API endpoints for session operations. The `app_name` you provide to the `Runner` when using this service usually corresponds to a deployed Reasoning Engine ID or its full resource name.
    
    ```python
    # from google.adk.sessions import VertexAiSessionService
    # import os
    
    # project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    # location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # if project_id:
    #     vertex_session_service = VertexAiSessionService(project=project_id, location=location)
    # else:
    # print("GOOGLE_CLOUD_PROJECT not set, cannot init VertexAiSessionService")
    
    ```
    

> ## Choose SessionService Based on Deployment Needs
> 
> - **Local Dev/Test:** `InMemorySessionService`.
> - **Self-Hosted with DB:** `DatabaseSessionService`.
> - **Google Cloud Managed Deployment:** `VertexAiSessionService`.
> {: .prompt-info }

**What's Next?**

Mastering session and state management is fundamental for creating conversational and personalized agents. We've seen how ADK provides a flexible system with different persistence backends. Next we'll explore how agents can work with files, saving and loading data that might be too large or unsuitable for session state, using ADK's Artifact Service.
