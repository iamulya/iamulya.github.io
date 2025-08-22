---
title: Chapter 20 - Evaluating Agent Performance 
date: "2025-08-22 17:30:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Building intelligent agents is an iterative process. A crucial part of this process is **evaluation**: systematically assessing how well your agent performs its intended tasks, uses tools correctly, and provides accurate and helpful responses. Without robust evaluation, it's difficult to measure progress, identify weaknesses, or ensure your agent meets quality standards before deployment.

ADK provides a built-in evaluation framework, including CLI commands, core classes like `AgentEvaluator`, and data structures like `EvalSet` and `EvalCase`, to help you rigorously test your agents. This chapter explores these components and how to set up and run evaluations.

## The Importance of Agent Evaluation

Why is evaluating agents so important?

- **Quality Assurance:** To ensure the agent behaves as expected and provides correct, relevant, and helpful responses.
- **Regression Testing:** As you modify your agent (change instructions, update tools, add new capabilities), evaluations help catch regressions where previously working functionality breaks.
- **Performance Benchmarking:** To compare different agent versions, model configurations, or prompting strategies.
- **Identifying Weaknesses:** To pinpoint areas where the agent struggles, misunderstands queries, or misuses tools.
- **Iterative Improvement:** Evaluation results provide concrete data to guide further development and refinement.
- **Building Trust:** Demonstrating that an agent has been thoroughly evaluated builds confidence in its reliability.

Manual testing by chatting with your agent is a good starting point, but for comprehensive assessment, automated evaluation against predefined test cases is essential.

## The ADK Evaluation Ecosystem: `adk eval` and `AgentEvaluator`

ADK offers two primary ways to run evaluations:

1. **The `adk eval` Command Line Interface:**
    - This is the most common way to run evaluations.
    - **Syntax:** `adk eval <agent_module_path> <eval_set_file_or_dir> [options]`
        - `<agent_module_path>`: Path to your agent's Python module (directory or specific file containing `root_agent`).
        - `<eval_set_file_or_dir>`: Path to a JSON file defining an `EvalSet` or a directory containing multiple such files (often ending in `.evalset.json` or the older `.test.json`).
        - `[options]`: Can include `-num_runs` (how many times to repeat each eval case), `-agent_name` (if evaluating a specific sub-agent), `-initial_session_file` (for older eval formats).
    - It automates the process of running your agent against each test case in the evaluation set and comparing the actual outputs against expected outcomes or metrics.
2. **The `google.adk.evaluation.AgentEvaluator` Class:**
    - This class provides the programmatic interface for agent evaluation, which the `adk eval` CLI uses internally.
    - You can use it directly in your Python scripts for more custom evaluation workflows or integration into larger testing frameworks.
    - Key static methods:
        - `evaluate_eval_set(agent_module, eval_set, criteria, num_runs, agent_name)`: Evaluates a single `EvalSet` object.
        - `evaluate(agent_module, eval_dataset_file_path_or_dir, ...)`: Similar to the CLI, handles loading from files/directories.
        - `migrate_eval_data_to_new_schema(...)`: A utility to convert older eval data formats to the current `EvalSet` Pydantic schema.


> ## `adk eval` for Standardized Testing
> 
> For most evaluation needs, the adk eval CLI is the recommended approach. It provides a consistent and reproducible way to test your agents. Reserve direct use of AgentEvaluator for scenarios requiring deeper programmatic control over the evaluation loop.
> {: .prompt-info }

## Defining Evaluation Data: `EvalSet` and `EvalCase`

The core of ADK evaluation is well-defined test data. This data is structured using Pydantic models:

- **`google.adk.evaluation.eval_case.Invocation`**: Represents a single user turn and the expected (or actual) agent interaction.
    - `user_content: types.Content`: The input from the user.
    - `final_response: Optional[types.Content]`: The *expected* final textual response from the agent for this turn.
    - `intermediate_data: Optional[IntermediateData]`: Contains expected intermediate steps.
        - `tool_uses: list[types.FunctionCall]`: The *expected* sequence of tool calls (name and arguments) the agent should make.
        - `intermediate_responses: list[Tuple[str, list[types.Part]]]`: Expected textual outputs from sub-agents before the final response (for multi-agent systems).
    - `invocation_id: str` (populated during generation), `creation_timestamp: float`.
- **`google.adk.evaluation.eval_case.EvalCase`**: Represents a single test case, which can be a multi-turn conversation.
    - `eval_id: str`: A unique identifier for this test case (e.g., "test_find_available_pets").
    - `conversation: list[Invocation]`: A list of `Invocation` objects representing the expected sequence of user inputs and agent interactions for this test case.
    - `session_input: Optional[SessionInput]`: Defines initial session state (`app_name`, `user_id`, `state: dict`) needed before this test case runs.
- **`google.adk.evaluation.eval_set.EvalSet`**: A collection of `EvalCase` objects.
    - `eval_set_id: str`: A unique identifier for this set of test cases.
    - `name: Optional[str]`: A descriptive name for the eval set.
    - `description: Optional[str]`: A description of what this eval set tests.
    - `eval_cases: list[EvalCase]`: The list of test cases.
    - `creation_timestamp: float`.

This data is typically stored in a JSON file (e.g., `my_agent_tests.evalset.json`).

**Example `EvalSet` JSON structure (`my_petstore_evals.evalset.json`):**

```json
{
  "eval_set_id": "petstore_core_functionality_v1",
  "name": "Petstore Core Functionality Tests",
  "description": "Tests basic pet finding and adding operations for the Petstore agent.",
  "eval_cases": [
    {
      "eval_id": "find_available_pets",
      "session_input": { // Optional: Initial state for this specific case
        "app_name": "PetStoreAppEval",
        "user_id": "eval_user_find",
        "state": { "user:location_preference": "US" }
      },
      "conversation": [
        { // First turn
          "user_content": {
            "role": "user",
            "parts": [{ "text": "Are there any pets available for adoption?" }]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "find_pets_by_status",
                "args": { "status": "available" }
              }
            ]
          },
          "final_response": {
            "role": "model",
            "parts": [{ "text": "Yes, I found several available pets: Buddy (dog), Whiskers (cat)." }]
          }
        }
      ]
    },
    {
      "eval_id": "add_new_dog",
      "conversation": [
        { // First turn
          "user_content": {
            "role": "user",
            "parts": [{ "text": "Please add a new dog named 'Rex' with ID 789 who is available." }]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "add_pet",
                "args": { "name": "Rex", "id": 789, "status": "available" }
              }
            ]
          },
          "final_response": {
            "role": "model",
            "parts": [{ "text": "Okay, I've added Rex (ID 789) to the store as an available pet." }]
          }
        }
      ]
    }
  ],
  "creation_timestamp": 1678886400.0
}

```


> ## Best Practice: Craft Comprehensive EvalCases
> 
> Good EvalCases are the cornerstone of effective evaluation. For each case:
> 
> - Define realistic `user_content`.
> - Specify the `expected_tool_use` precisely, including tool names and arguments, if tool interaction is a key part of the test.
> - Provide a clear `final_response` that represents the ideal agent output.
> - Use `session_input` to set up any prerequisite state needed for the test.
> - Give descriptive `eval_id`s.
> {: .prompt-info }

## Understanding Evaluation Metrics and Criteria

ADK's evaluation framework, via `AgentEvaluator`, supports different types of metrics to assess various aspects of agent performance. These are typically defined in a `test_config.json` file within the same directory as your older `.test.json` eval files, or you pass them programmatically if using `AgentEvaluator` directly. For `EvalSet` files, criteria are often implicitly understood by the evaluators.

**Key Evaluators and Associated Metrics:**

1. **`google.adk.evaluation.TrajectoryEvaluator`**:
    - **Focus:** Assesses the correctness of the agent's tool usage and reasoning path (the "trajectory").
    - **Metric (`TOOL_TRAJECTORY_SCORE_KEY` -> "tool\_trajectory\_avg\_score"):** Compares the actual sequence of tool calls (`name` and `args`) made by the agent against the `expected_tool_use` defined in the `Invocation` part of your `EvalCase`.
        - Scores 1.0 for an exact match in a given invocation, 0.0 otherwise.
        - The `overall_score` is the average across all invocations.
    - **When to Use:** Crucial for agents that rely heavily on tools to ensure they are calling the right tools with the correct parameters in the right order.
2. **`google.adk.evaluation.ResponseEvaluator`**:
    - **Focus:** Assesses the quality of the agent's final textual response.
    - **Metrics:**
        - **`RESPONSE_MATCH_SCORE_KEY` ("response\_match\_score"):** Uses ROUGE-1 (or similar n-gram overlap metrics) to compare the agent's generated `final_response` text against the reference text in the `EvalCase`.
            - Score range: [0, 1]. Higher is better (more overlap).
            - Useful for factual recall or when a specific phrasing is expected.
        - **`RESPONSE_EVALUATION_SCORE_KEY` ("response\_evaluation\_score"):** Uses an LLM to perform a pointwise evaluation of the agent's response quality (e.g., coherence, helpfulness, safety) against the user query and potentially the reference answer.
            - Score range: Typically [0, 5] or similar, depending on the specific LLM-based metric used (e.g., `MetricPromptTemplateExamples.Pointwise.COHERENCE`). Higher is better.
            - Useful for assessing more nuanced aspects of response quality.
    - **When to Use:** To ensure the agent's final answers are accurate, relevant, coherent, and meet desired quality standards.

**Defining Criteria (Historically with `test_config.json` for older `.test.json` files):**
You would create a `test_config.json` file in the same directory as your evaluation data:

```json
// test_config.json (for older .test.json format)
{
  "criteria": {
    "tool_trajectory_avg_score": 0.9, // Expect 90% accuracy in tool trajectories
    "response_match_score": 0.75,     // Expect ROUGE-1 score of at least 0.75
    "response_evaluation_score": 4.0  // Expect LLM-based coherence score of at least 4.0
  }
}

```

When using the new `EvalSet` format and `AgentEvaluator.evaluate_eval_set`, you pass the criteria dictionary programmatically.


> ## Choose Metrics Relevant to Your Agent's Task
> 
> - If your agent's primary job is accurate tool use (e.g., an API orchestrator), `tool_trajectory_avg_score` is paramount.
> - If factual accuracy in the response is key, `response_match_score` against a good reference is important.
> - For overall conversational quality, helpfulness, and coherence, `response_evaluation_score` (LLM-based) provides a more holistic measure.
> 
> Often, a combination of these metrics is used.
> {: .prompt-info }

## Automated Evaluation with `EvaluationGenerator`

To get actual agent outputs to compare against your `EvalSet`, ADK uses the `EvaluationGenerator`.

- `EvaluationGenerator.generate_responses(eval_set, agent_module_path, repeat_num, agent_name)`:
    1. Takes an `EvalSet` object and the path to your agent's module.
    2. For each `EvalCase` in the `EvalSet`:
        - It initializes a new `Runner` (typically `InMemoryRunner`) with your agent.
        - If `eval_case.session_input` is defined, it creates a session with that initial state.
        - It then simulates the conversation defined in `eval_case.conversation` by feeding each `user_content` from the `Invocation`s to the agent via the runner.
        - It collects the actual `Invocation` data produced by your agent (final response, tool calls made).
    3. It can repeat this process `repeat_num` times for each `EvalCase` to account for LLM variability.
    4. Returns a list of `EvalCaseResponses` objects, where each object contains the original `EvalCase` and a list of lists of actual `Invocation`s (one list of actuals per `repeat_num`).

This generated data is then consumed by `AgentEvaluator` and its underlying metric evaluators (`TrajectoryEvaluator`, `ResponseEvaluator`).

![*Diagram: The ADK Evaluation Workflow.*](/assets/img/2025-08-22-evaluating-agent-performance/figure-1.png)


## Visualizing Evaluation Results in the Dev UI

The ADK Development UI (`adk web`) typically includes an "Eval" tab. This tab allows you to:

- Discover and load `EvalSet` files from your project directory.
- Trigger evaluations (similar to `adk eval` but within the UI).
- View the results of evaluations:
    - Overall scores for each metric.
    - A detailed breakdown for each `EvalCase`.
    - A side-by-side comparison of expected vs. actual tool calls and responses.
    - Links to the full trace of the agent's execution for that specific eval case run, allowing you to debug failures directly.


> ## Best Practice: Use Dev UI for Debugging Eval Failures
> 
> When an evaluation fails, the Dev UI's Eval tab is invaluable. It not only shows you what failed (e.g., wrong tool called, response didn't match) but often links directly to the Trace view for that specific failing run. This allows you to immediately inspect the LLM prompts, tool arguments, and agent reasoning that led to the failure.
> {: .prompt-info }

**Example: Running an Evaluation via CLI**

Let's create:

1. A simple agent with a multiplication tool.
```python
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# can't use `from ...utils import load_environment_variables` here because of top-level import error.
def load_environment_variables():
    """
    Load environment variables from a .env file located in the parent directory of the current script.
    """
    # Ensure the .env file is loaded from the parent directory
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the .env file in the parent directory
    dotenv_path = os.path.join(current_script_dir, ".env")

    # Load the .env file if it exists
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")
        
load_environment_variables()

# --- 1. Define a Simple Tool ---
def multiply_numbers(a: int, b: int) -> int:
  """
  Multiplies two integer numbers a and b.
  Args:
    a: The first integer.
    b: The second integer.
  Returns:
    The product of a and b.
  """
  print(f"Tool 'multiply_numbers' called with a={a}, b={b}")
  return a * b

multiply_tool = FunctionTool(func=multiply_numbers)

# --- 2. Define a Simple Agent ---
# This agent will use the 'multiply_tool'.
root_agent = Agent( 
    name="calculator_agent",
    model="gemini-2.0-flash", # Use a model that supports tool use
    instruction=(
        "You are a helpful calculator assistant. "
        "When asked to multiply numbers, use the 'multiply_numbers' tool. "
        "After using the tool, you MUST respond in the exact format: "
        "'Okay, [number1] times [number2] is [result].' "
        "Replace [number1], [number2], and [result] with the actual numbers."
    ),
    tools=[multiply_tool]
)

print(f"Agent 'calculator_agent' defined.")
```
2. An evaluation set that tests the calculator agent's ability to perform simple multiplication using its tool.
```json
{
    "eval_set_id": "calculator_basic_multiplication_eval_set_v1",
    "name": "Calculator Basic Multiplication Tests",
    "description": "Tests the calculator agent's ability to perform simple multiplication using its tool.",
    "eval_cases": [
      {
        "eval_id": "multiply_7_by_6_case",
        "conversation": [
          {
            "invocation_id": "eval_inv_001",
            "user_content": {
              "parts": [
                {
                  "text": "Hi calculator, what is 7 times 6?"
                }
              ],
              "role": "user"
            },
            "final_response": {
              "parts": [
                {
                  "text": "Okay, 7 times 6 is 42."
                }
              ],
              "role": "model"
            },
            "intermediate_data": {
              "tool_uses": [
                {
                  "name": "multiply_numbers",
                  "args": {
                    "a": 7,
                    "b": 6
                  }
                }
              ]
            },
            "creation_timestamp": 1678886400.0
          }
        ],
        "session_input": {
          "app_name": "calculator_app_eval",
          "user_id": "eval_user_123",
          "state": {
            "previous_calculation": null
          }
        },
        "creation_timestamp": 1678886400.0
      }
    ],
    "creation_timestamp": 1678886400.0
  }
```

From the agent directory, you would run:

```bash
# Ensure GOOGLE_API_KEY or GCP auth is set for the agent's model
adk eval . ./eval_sets/tests.evalset.json 

```

The output will be similar to the following:

```jsx
Using evaluation creiteria: {'tool_trajectory_avg_score': 1.0, 'response_match_score': 0.8}
Loaded environment variables from: /Users/amulya.bhatia/adk-book-code/chapter19/eval_agent/.env
Agent 'calculator_agent' defined.
Running Eval: eval_sets/tests.evalset.json:multiply_7_by_6_case
Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
Tool 'multiply_numbers' called with a=7, b=6
Computing metrics with a total of 1 Vertex Gen AI Evaluation Service API requests.
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.19it/s]
All 1 metric requests are successfully computed.
Evaluation Took:0.8597189590000198 seconds
<IPython.core.display.HTML object>
Result: ✅ Passed

*********************************************************************
Eval Run Summary
eval_sets/tests.evalset.json:
  Tests passed: 1
  Tests failed: 0
```


> ## `AgentEvaluator` and Vertex AI SDK
> 
> `AgentEvaluator` (and specifically its `ResponseEvaluator` component) in `google-adk` leverages the `vertexai.preview.evaluation.EvalTask` from the `google-cloud-aiplatform` (Vertex AI) library. This evaluation task, even for metrics that might seem computable locally (like `ROUGE`), often performs an initialization step that expects a valid Google Cloud Project to be configured for Vertex AI.
> 
> This is why when running `adk eval` through command-line, you would need to set the GOOGLE_CLOUD_PROJECT environment variable to a valid Google Cloud Project ID, even if you are not using Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI` is set to `false`).
> {: .prompt-info }

**What's Next?**

Thorough evaluation is key to building reliable and effective AI agents. With ADK's evaluation framework, you have the tools to systematically test and improve your creations. Next we'll take our well-tested agents and explore how to package and deploy them to various environments, including Google Cloud Run.
