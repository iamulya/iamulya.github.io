---
title: "Chapter 5 - Equipping Agents with Tools: The FunctionTool"
date: "2025-08-22 10:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Previously we learned how to build `LlmAgent` instances, provide them with instructions, and configure their interaction with Large Language Models. While LLMs are incredibly powerful for understanding and generating text, their knowledge is typically frozen at the time of their training, and they cannot directly interact with external systems or perform real-time computations beyond their inherent capabilities. This is where **Tools** come into play.

Tools are the primary mechanism in ADK for extending an agent's abilities, allowing it to fetch live data, interact with APIs, perform calculations, or execute any custom Python logic. This chapter focuses on the most straightforward way to create custom tools: using the `google.adk.tools.FunctionTool`.

## The `BaseTool` Abstraction

Before diving into `FunctionTool`, let's briefly revisit the foundation: `google.adk.tools.BaseTool`. All tools in ADK, whether custom-made or pre-built, inherit from this abstract base class. It defines the essential contract for a tool:

- **`name: str`**: A unique name for the tool. This is the name the LLM will use when it decides to invoke the tool.
- **`description: str`**: A natural language description of what the tool does, its parameters, and what it returns. This is *critically important* as the LLM relies heavily on this description to understand when and how to use the tool.
- **`_get_declaration() -> Optional[types.FunctionDeclaration]`**: This method is responsible for returning the tool's interface as a `FunctionDeclaration`. This declaration, similar to an OpenAPI schema snippet, tells the LLM about the tool's parameters, their types, and whether they are required.
- **`run_async(args: dict[str, Any], tool_context: ToolContext) -> Any`**: This asynchronous method contains the actual logic of the tool. It receives the arguments provided by the LLM (parsed into a dictionary) and a `ToolContext` object, which provides access to session state, artifacts, etc. It should return the result of the tool's execution.

![*Diagram: Core components of the `BaseTool` interface.*](/assets/img/2025-08-22-equipping-agents-with-tools/figure-1.png)


While you can create tools by directly subclassing `BaseTool` and implementing these methods (which is useful for complex tools or those integrating with external SDKs), ADK provides a much simpler way for most custom Python logic: the `FunctionTool`.

## Creating Custom Python Tools with `FunctionTool`

The `google.adk.tools.FunctionTool` is a powerful and convenient wrapper that allows you to turn almost any Python callable (a regular function, a method, or an object with a `__call__` method) into a fully functional ADK tool with minimal effort.

**How it Works:**

1. You provide your Python callable to the `FunctionTool` constructor.
2. ADK uses Python's `inspect` module to:
    - Infer the tool's `name` from the function's name.
    - Infer the tool's `description` from the function's docstring.
    - Infer the tool's parameters (name, type, optionality, and description) from the function's signature (type hints and default values in the function definition) and the parameter descriptions in the docstring.
3. It automatically generates the `FunctionDeclaration` required by the LLM.

**Example: A Simple Calculator Tool**

Let's create a tool that can perform basic arithmetic operations.

```python
from google.adk.tools import FunctionTool
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part 

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM

load_environment_variables()

# 1. Define the Python function
def simple_calculator(
    operand1: float,
    operand2: float,
    operation: str
) -> float | str: # Using | for Union type hint (Python 3.10+)
    """
    Performs a basic arithmetic operation on two numbers.

    Args:
        operand1: The first number.
        operand2: The second number.
        operation: The operation to perform. Must be one of 'add', 'subtract', 'multiply', 'divide'.

    Returns:
        The result of the calculation, or an error message string if the operation is invalid or division by zero occurs.
    """
    if operation == 'add':
        return operand1 + operand2
    elif operation == 'subtract':
        return operand1 - operand2
    elif operation == 'multiply':
        return operand1 * operand2
    elif operation == 'divide':
        if operand2 == 0:
            return "Error: Cannot divide by zero."
        return operand1 / operand2
    else:
        return f"Error: Invalid operation '{operation}'. Valid operations are 'add', 'subtract', 'multiply', 'divide'."
    
calculator_tool = FunctionTool(func=simple_calculator)

calculator_agent = Agent(
    name="math_wiz", model=DEFAULT_LLM,
    instruction="You are a helpful assistant that can perform basic calculations...",
    tools=[calculator_tool]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=calculator_agent, app_name="CalculatorApp")
    prompts = ["What is 5 plus 3?", "Calculate 10 divided by 2?"]

    user_id="calc_user"
    session_id="s_calc"
    
    create_session(runner, session_id, user_id)
    
    for prompt_text in prompts:
        print(f"
YOU: {prompt_text}")
        user_message = Content(parts=[Part(text=prompt_text)], role="user")
        print("MATH_WIZ: ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts and event.content.parts[0].text:
                 print(event.content.parts[0].text, end="")
        print()
```

When you run `calculator.py`:

- ADK inspects `simple_calculator`.
- The `name` becomes `"simple_calculator"`.
- The `description` is taken from its docstring.
- The parameters `operand1` (float, required), `operand2` (float, required), and `operation` (str, required) are inferred.
- When the `calculator_agent` receives a prompt like "What is 5 plus 3?", the Gemini LLM will see the `simple_calculator` tool and its description. It will decide to call it with `operand1=5.0`, `operand2=3.0`, and `operation='add'`.
- ADK will execute `simple_calculator(operand1=5.0, operand2=3.0, operation='add')`.
- The result (`8.0`) will be sent back to the LLM, which will then formulate the final natural language response.


> ## Best Practice: Clear Docstrings for FunctionTools
> 
> The quality of your Python function's docstring directly impacts how well the LLM can use your tool.
> 
> - **Overall Description:** The main docstring of the function becomes the tool's `description`. Make it clear what the tool does and when it should be used.
> - **Argument Descriptions:** Describe each argument clearly in the `Args:` section of your docstring (Google Style Python Docstrings are recommended). ADK attempts to parse these to provide richer parameter descriptions to the LLM.
> - **Return Description:** Describe what the function returns in the `Returns:` section.
> {: .prompt-info }

## Designing Tool Inputs and Outputs: Type Hinting and Pydantic

ADK leverages Python's type hints to define the schema for your tool's parameters. This is crucial for the LLM to understand what kind of data to provide for each argument.

- **Basic Python Types:** `str`, `int`, `float`, `bool` are directly supported.
- **Optional Parameters:** Use `Optional[type]` from the `typing` module or the `| None` syntax (Python 3.10+), and provide a default value in the function signature if the parameter is truly optional from the LLM's perspective.
    
    ```python
    from typing import Optional
    
    def search(query: str, num_results: Optional[int] = 5) -> list[str]:
        """Searches for a query.
        Args:
            query: The search query.
            num_results: The number of results to return. Defaults to 5.
        """
        # ... search logic ...
        return [f"Result for '{query}' #{i}" for i in range(num_results or 5)]
    
    ```
    
- **Lists and Dictionaries:** You can use `list[type]` or `dict[str, type]`.
    
    ```python
    def process_items(items: list[str], config: dict[str, bool]) -> str:
        """Processes a list of items with a given configuration."""
        # ... logic ...
        return "Processed"
    
    ```
    
- **Enums (Literals):** For parameters that can only take a specific set of string values, use `typing.Literal`.
    
    ```python
    from typing import Literal
    
    def set_status(status: Literal["pending", "active", "completed"]) -> str:
        """Sets the status of an item."""
        return f"Status set to {status}"
    
    ```
    
    In the `simple_calculator` example, `operation: str` could be improved to `operation: Literal['add', 'subtract', 'multiply', 'divide']` for stricter input from the LLM.
    
- **Complex Objects (Pydantic Models):** For tools that require structured object inputs, you can define a Pydantic `BaseModel` and use it as a type hint for a parameter. ADK will automatically convert this into the appropriate JSON schema for the LLM.

Following is an agent for User Profile Management that uses a Pydantic model for user profiles. 
    
```python
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from pydantic import BaseModel, Field

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM

load_environment_variables()

# Here we use Pydantic to define a model for user profiles. Pydantic sometimes has types like EmailStr that are not known to the LLM.
# If you want to use types like EmailStr, you need to ensure the LLM can understand them by creating a Custom Tool class that manually defines its schema for the LLM.
class UserProfile(BaseModel):
    username: str = Field(description="The username of the user.")
    email: str = Field(description="The email address of the user.") 
    age: int = Field(description="The age of the user.")

def update_user_profile(profile: dict) -> dict:
    """Updates a user's profile information. Args: profile: A UserProfile object..."""
    try:
        # Validate and convert the incoming dictionary to UserProfile instance
        user_profile_instance = UserProfile.model_validate(profile)
    except Exception as e: # Catch PydanticValidationError or other potential errors
        print(f"Error validating profile data: {profile}. Error: {e}")
        return {"status": "error", "message": f"Invalid profile data provided. {e}"}
    
    print(f"Updating profile for {user_profile_instance.username} with email {user_profile_instance.email}")

    # Here you would typically update the profile in a database or some storage.

    return {"status": "success", "updated_username": user_profile_instance.username}

user_profile_updater_tool = FunctionTool(func=update_user_profile)

# Sometimes you have to be specific about the instruction for the tool, especially if the LLM needs to understand how to use it.
# This instruction provider is NOT currently used, but if used, provides clear and concise instructions to the LLM on how to use the tool.
def user_profile_tool_instruction(context: ReadonlyContext) -> str:
    """Generates the instruction for the user profile tool."""
    print(f"User profile schema: {UserProfile.model_json_schema()}")
    return f"""You are a user profile manager. Your task is to call the tool named '{user_profile_updater_tool.name}' "
        The tool expects {UserProfile.model_json_schema()}. 
    """

profile_agent = Agent(
    name="profile_manager",
    model=DEFAULT_LLM,
    instruction="Manage user profiles using the provided tool.",#user_profile_tool_instruction
    tools=[user_profile_updater_tool]
)

if __name__ == "__main__":
    from google.adk.runners import InMemoryRunner
    runner = InMemoryRunner(agent=profile_agent, app_name="ProfileManagerApp")

    user_id="user"
    session_id="profile_session"

    create_session(runner, session_id, user_id)
    from google.genai.types import Content, Part
    
    print("
Profile Manager is ready. Type 'exit' to quit.")
    print("Example prompts:")
    print(" - Update my profile: username is 'testuser', email is 'test@example.com', age is 36")

    # This will test if the agent understands that it requires the age parameter
    print(" - Set user 'janedoe' profile with email 'jane.doe@email.net'. (This should fail since age is not provided!)")
    print("-" * 30)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting Profile Manager. Goodbye!")
            break
        if not user_input.strip():
            continue

        user_message = Content(parts=[Part(text=user_input)], role="user")
        print("Agent: ", end="", flush=True)
        try:
            for event in runner.run(
                user_id=user_id, session_id=session_id, new_message=user_message,
            ):
                if event.content and event.content.parts:
                    # Print tool calls for debugging/visibility
                    if event.get_function_calls():
                        for fc in event.get_function_calls():
                            print(f"
[AGENT INTENDS TO CALL TOOL] Name: {fc.name}, Args: {fc.args}")
                    # Print tool responses for debugging/visibility
                    if event.get_function_responses():
                        for fr in event.get_function_responses():
                            print(f"
[TOOL RESPONSE RECEIVED BY AGENT] Name: {fr.name}, Response: {fr.response}")
                    # Print text from the agent
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print() # Newline after agent's full response
        except Exception as e:
            print(f"
An error occurred: {e}")
```

When the LLM decides to use `update_user_profile`, it will attempt to construct a JSON object matching the `UserProfile` schema.

Pydantic sometimes has types like `EmailStr` that are not known to the LLM. If you want to use types like `EmailStr`, you need to ensure the LLM can understand them by creating a Custom Tool class that manually defines its schema for the LLM.

```python
# Custom Tool class that manually defines its schema for the LLM
class UpdateUserProfileTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="update_user_profile",
            description=(
                "Updates a user's profile information. "
                "Requires username and a valid email. Age and interests are optional."
            )
        )

    @override
    def _get_declaration(self) -> FunctionDeclaration:
        # Manually define the schema that the LLM will see
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=Schema(
                type=GeminiType.OBJECT,
                properties={
                    "profile": Schema( # The LLM will see an argument named 'profile'
                        type=GeminiType.OBJECT,
                        description="The user's profile information.",
                        properties={
                            "username": Schema(type=GeminiType.STRING, description="The username of the user."),
                            "email": Schema(type=GeminiType.STRING, format="email", description="The email address of the user. Must be a valid email format."),
                            "age": Schema(type=GeminiType.INTEGER, description="The age of the user (optional).", nullable=True)
                        },
                        required=["username", "email"] # Fields required within the 'profile' object
                    )
                },
                required=["profile"] # The 'profile' object itself is required by the tool
            )
        )

    @override
    async def run_async(self, *, args: Dict[str, Any], tool_context: Optional[Any] = None) -> Any:
        # 'args' will come from the LLM based on the schema defined in _get_declaration()
        # It will likely be: {'profile': {'username': '...', 'email': '...', ...}}
        profile_data = args.get("profile", {})
        if not isinstance(profile_data, dict):
            return {"status": "error", "message": "Invalid 'profile' argument format. Expected an object."}
        return _update_user_profile_logic(profile_data)

# Instantiate your custom tool
user_profile_updater_tool = UpdateUserProfileTool()
```
    

> ## Pydantic for Robust Tool Inputs
> 
> Using Pydantic models for complex tool inputs is highly recommended. It provides:
>  
> - Clear schema definition for the LLM.
> - Automatic data validation when the LLM provides arguments.
> - Easy serialization/deserialization.
> 
> This makes your tools more robust and easier for the LLM to use correctly.
> {: .prompt-info }

**Return Types:**
The return type of your Python function will also be hinted to the LLM if possible (though LLMs primarily focus on input schemas for tool selection). Simple types (`str`, `int`, `float`, `bool`, `list`, `dict`) are generally fine. Complex return objects are usually serialized to JSON (or a string representation) by ADK before being sent back to the LLM as the tool's output. The LLM then typically processes this string output.

## Automatic Function Declaration Generation

As mentioned, `FunctionTool` automatically generates the `FunctionDeclaration` (the schema that the LLM sees) based on your Python function's signature and docstring.

Let's look at the `simple_calculator` tool again. ADK would generate a `FunctionDeclaration` somewhat equivalent to this (simplified for illustration):

```json
{
  "name": "simple_calculator",
  "description": "Performs a basic arithmetic operation on two numbers.",
  "parameters": {
    "type": "OBJECT",
    "properties": {
      "operand1": {
        "type": "NUMBER",
        "description": "The first number."
      },
      "operand2": {
        "type": "NUMBER",
        "description": "The second number."
      },
      "operation": {
        "type": "STRING",
        "description": "The operation to perform. Must be one of 'add', 'subtract', 'multiply', 'divide'."
      }
    },
    "required": ["operand1", "operand2", "operation"]
  }
  // "response" schema might also be included for some models/variants
}

```

This JSON-like structure is what the LLM uses to understand:

- The tool's name (`simple_calculator`).
- What it does (from the `description`).
- What parameters it expects (`operand1`, `operand2`, `operation`).
- The type of each parameter (e.g., `NUMBER`, `STRING`).
- Any descriptions for those parameters (parsed from the docstring's `Args:` section).
- Which parameters are `required`.


> ## LLM Schema Interpretation
> 
> While ADK does its best to generate an accurate schema, LLMs can sometimes misinterpret complex schemas or have subtle preferences for how parameters are described. If a tool isn't being called correctly:
>  
> 1. **Simplify:** Try simplifying your function signature or Pydantic model.
> 2. **Clarify Descriptions:** Make your function and parameter docstrings extremely clear and explicit.
> 3. **Inspect the Trace:** Use the ADK Dev UI's Trace view to see the exact `FunctionDeclaration` being sent to the LLM and how the LLM attempts to fill in the arguments. This is invaluable for debugging tool usage.
> {: .prompt-info }


## Tool Context (`ToolContext`): Accessing Session, State, and Artifacts

When your tool's function is executed by ADK, it can optionally receive a `ToolContext` object as its first parameter if your function is type-hinted to accept it. This object provides access to the broader execution environment.

```python
from google.adk.tools import FunctionTool, ToolContext 
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from building_intelligent_agents.utils import DEFAULT_LLM, load_environment_variables,create_session

load_environment_variables()

def get_and_increment_counter(tool_context: ToolContext) -> str:
    """Retrieves a counter from session state, increments it... Args: tool_context: ..."""
    session_counter = tool_context.state.get("session_counter", 0); session_counter += 1
    tool_context.state["session_counter"] = session_counter
    return f"Counter is now: {session_counter}. Invocation: {tool_context.invocation_id}, FuncCall: {tool_context.function_call_id}"

counter_tool = FunctionTool(func=get_and_increment_counter)

stateful_agent = Agent(
    name="stateful_counter_agent", model=DEFAULT_LLM,
    instruction="You have a tool to get and increment a counter. Use it when asked.",
    tools=[counter_tool]
)

if __name__ == "__main__":
    runner = InMemoryRunner(agent=stateful_agent, app_name="StatefulApp")
    
    session_id = "stateful_session_1"
    user_id = "state_user"
    create_session(runner, session_id, user_id)
    
    for i in range(3):
        user_message = Content(parts=[Part(text="Increment counter.")], role="user"); print(f"
YOU: Increment counter. (Turn {i+1})")
        print("AGENT: ", end="", flush=True)
        for event in runner.run(user_id="state_user", session_id=session_id, new_message=user_message):
            if event.content and event.content.parts and event.content.parts[0].text: print(event.content.parts[0].text, end="")
        print()
        current_session = runner.session_service._get_session_impl(app_name="StatefulApp", user_id=user_id, session_id=session_id)
        print(f"(Session state 'session_counter': {current_session.state.get('session_counter')})")
```

The `ToolContext` object provides:

- `state: State`: Access to the session's state (application-scoped, user-scoped, and session-scoped). Modifications made here are captured as `state_delta` in the resulting `Event`.
- `save_artifact(filename, artifact)` and `load_artifact(filename)`: Methods to interact with the `ArtifactService`.
- `search_memory(query)`: Method to interact with the `MemoryService`.
- `invocation_id: str`: The ID of the overall user turn.
- `function_call_id: str`: The ID of the specific function call part generated by the LLM that triggered this tool.
- `user_content: Optional[Content]`: The initial user message that started the current invocation.
- `actions: EventActions`: Allows the tool to signal actions like `skip_summarization` or `transfer_to_agent` (though direct transfer from tools is less common than from agent logic).


> ## ToolContext for Stateful and Contextual Tools
> 
> ToolContext is essential for building tools that:
> 
> - Need to remember information across their own invocations within the same session (using `tool_context.state`).
> - Need to read or write files related to the session (using `tool_context.save/load_artifact`).
> - Need to consult long-term memory (using `tool_context.search_memory`).
> {: .prompt-info }


## Tool Callbacks: `before_tool_callback`, `after_tool_callback` 

Similar to agent and model callbacks, `LlmAgent` also supports callbacks specifically for tool invocations:

- **`before_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext) -> Optional[dict]`**:
Called before a tool's `run_async` method is executed.
    - It receives the `tool` instance, the `args` the LLM provided, and the `tool_context`.
    - If it returns a dictionary, that dictionary is used as the tool's result, and the actual `tool.run_async()` is skipped. Useful for mocking, caching tool results, or input argument validation/transformation.
- **`after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]`**:
Called after `tool.run_async()` completes and returns its `tool_response`.
    - It can inspect or modify the `tool_response`. If it returns a dictionary, that dictionary replaces the original tool response. Useful for post-processing, sanitizing, or logging tool outputs.

These callbacks provide fine-grained interception points for managing tool execution, which we'll explore more in advanced scenarios.

```python
# Conceptual example for an LlmAgent using tool callbacks
# from google.adk.agents import Agent
# from google.adk.tools import BaseTool, ToolContext
# import logging

# def my_before_tool_cb(tool: BaseTool, args: dict, tool_context: ToolContext) -> Optional[dict]:
#     logging.info(f"Before calling tool '{tool.name}' with args: {args}")
#     if tool.name == "sensitive_tool" and not context.state.get("user:is_admin"):
#         return {"error": "Unauthorized to use sensitive_tool"}
#     return None # Proceed with actual tool call

# def my_after_tool_cb(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
#     logging.info(f"After tool '{tool.name}' responded: {tool_response}")
#     if "api_key" in tool_response: # Example: sanitize sensitive data
#         tool_response["api_key"] = "[REDACTED]"
#     return tool_response

# agent_with_tool_callbacks = Agent(
#     # ... other params ...
#     tools=[some_tool, sensitive_tool],
#     before_tool_callback=my_before_tool_cb,
#     after_tool_callback=my_after_tool_cb
# )
```

**What's Next?**

You now have the power to create custom Python tools and integrate them into your `LlmAgent`s using `FunctionTool`. This dramatically expands what your agents can achieve. Next we'll explore the rich ecosystem of pre-built tools provided by ADK for common tasks like web searching, memory access, and user interaction, further accelerating your agent development.
