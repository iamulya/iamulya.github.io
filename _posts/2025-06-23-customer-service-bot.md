---
title: "Chapter 11 - Case Study: The Airline Customer Service Bot"
date: 2025-06-23 17:00:00 +0200
categories: [Gen AI, Agentic SDKs, OpenAI Agents SDK]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, OpenAI Agents SDK, Tinib00k]
image:
  path: /assets/img/tinibook-openai-agents-sdk-final.jpg
  alt: "Tinib00k: OpenAI Agents SDK"
---

> This article is part of my book [Tinib00k: OpenAI Agents SDK](https://gumroad.iamulya.one#WuWH2AdBm4qtHQojmRxBog==), which explores the OpenAI Agents SDK, its primitives, and patterns for building agentic AI systems. All of the chapters can be found [here](https://iamulya.one/tags/Tinib00k/) and the code is available on [Github](https://github.com/iamulya/openai-agentsdk-code). The book is available for free online and as a PDF eBook. If you find this content valuable, please consider supporting my work by purchasing the eBook (or download it for free!) or sharing it with others. For any issues around the book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

We have now covered the theory, primitives, and patterns of the Agents SDK. To see how these concepts coalesce into a practical, real-world application, this chapter presents an in-depth case study: a multi-agent system designed to handle customer service inquiries for an airline.

This is a classic use case for agentic workflows. A single, monolithic agent would struggle to be an expert in every possible customer need—from baggage policies to flight changes. A multi-agent system, however, allows us to build a team of specialists and a smart router to direct users to the right one. Our airline bot will be able to answer frequently asked questions and handle specific tasks like changing a passenger's seat, demonstrating a robust and scalable architecture.

## Architectural Overview

Our system is designed around a central "triage" agent that acts as an intelligent router. It doesn't answer questions itself; its sole purpose is to understand the user's intent and delegate the task to the appropriate specialist using **Handoffs**.

The components are:

1.  **`TriageAgent`**: The entry point for all user requests. It analyzes the user's query and hands off to either the `FAQAgent` or the `SeatBookingAgent`.
2.  **`FAQAgent`**: A specialist agent equipped with a tool to look up answers to common questions about baggage, Wi-Fi, and aircraft information.
3.  **`SeatBookingAgent`**: A specialist agent with a tool to modify a user's seat assignment for their flight.
4.  **`AirlineAgentContext`**: A shared Python object that holds the state for a single user's session, such as their confirmation number and flight details. This context is passed between agents, allowing them to collaborate seamlessly.

This architecture creates a clear separation of concerns, making the system easier to build, test, and extend.

![*The architecture of the Airline Customer Service Bot.*](/assets/img/2025-06-23-customer-service-bot/figure-1.png)


## Managing State: The `AirlineAgentContext`

Before we define the agents, we must define the shared state they will operate on. A simple Pydantic `BaseModel` is perfect for this. This context object will be passed through the entire run, allowing a tool in one agent to populate a field that another agent will later rely on.

```python
from pydantic import BaseModel

class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None
```

This simple class will act as the "single source of truth" for a user's session.

## Component Deep Dive: The Agents and Their Tools

Let's examine the implementation of each agent and its specialized tools.

### The FAQ Agent

This agent's job is to answer general questions. It has one tool, `faq_lookup_tool`, which simulates a call to a knowledge base. Notice how the agent's instructions explicitly tell it *not* to use its own knowledge and to hand off back to the triage agent if it can't find an answer.

```python
#
# The FAQ specialist agent and its tool.
#
from agents import Agent, function_tool, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# The Tool
@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    if "bag" in question or "baggage" in question:
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif "seats" in question or "plane" in question:
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif "wifi" in question:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."

# The Agent
faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
    model=DEFAULT_LLM,
)
```

### The Seat Booking Agent

This agent handles a specific, stateful task: changing a seat. Its tool, `update_seat`, demonstrates a key pattern: reading from and writing to the shared `AirlineAgentContext`.

```python
#
# The Seat Booking specialist agent and its tool.
#
from agents import Agent, function_tool, RunContextWrapper

# The Tool
@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"

# The Agent
seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    tools=[update_seat],
    model=DEFAULT_LLM,
)
```


>  Context as a Side Channel
> 
> The `update_seat` tool reads `context.context.flight_number`. Where does this value come from? The user never provides it. As we'll see next, it's injected into the context *at the moment of handoff* by the `TriageAgent`. This "side channel" communication via a shared context object is a clean and powerful way for agents to collaborate without cluttering the main conversation history.
{: .prompt-info }

### The Triage Agent

This is the orchestrator. Its only job is to route. Its `handoffs` list is where the system's logic is defined. Note the use of the `handoff()` helper for the `SeatBookingAgent`. We use it to attach an `on_handoff` callback, which is the perfect place for side effects that should occur during a transfer.

```python
#
# The Triage agent, the central router of our system.
#
import random
from agents import handoff

# The on_handoff callback
async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """A side effect that runs when handing off to the seat booking agent."""
    # This simulates a database lookup or API call to get flight details.
    flight_number = f"FLT-{random.randint(100, 999)}"
    print(f"[Handoff Hook]: Assigning flight number {flight_number} to context.")
    context.context.flight_number = flight_number

# The Agent
triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
    model=DEFAULT_LLM,
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)
```

## Orchestration and the Run Loop

With all the components defined, the final piece is the main application loop that simulates a user conversation. This code manages the conversation history (`input_items`) and the `current_agent`, updating them after each turn based on the `RunResult`.

```python
#
# The main application run loop.
#
import uuid
from agents import Runner, trace, ItemHelpers, TResponseInputItem

async def main():
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    # Normally, each input from the user would be an API request to your app, and you can wrap the request in a trace()
    # Here, we'll just use a random UUID for the conversation ID
    conversation_id = uuid.uuid4().hex[:16]

    while True:
        user_input = input(">: ")
        with trace("Customer service", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()
            current_agent = result.last_agent
```

This loop demonstrates how to maintain a continuous conversation. The `result.to_input_list()` and `result.last_agent` properties provide all the state needed to correctly resume the conversation on the next user turn.

### Example Walkthrough

Here is a sample console session showing the system in action.

```text
Welcome to Airline Customer Service. Type 'quit' to exit.
How can I help you today?

 > What is the baggage allowance?
----------------------------------------
Triage Agent: Skipping item: HandoffCallItem
Handed off from Triage Agent to FAQ Agent
FAQ Agent: Calling a tool
FAQ Agent: Tool call output: You are allowed to bring one bag on the plane. It must be under 50 pounds and 22 inches x 14 inches x 9 inches.
FAQ Agent: You are allowed to bring one bag on the plane. It must be under 50 pounds and 22 inches x 14 inches x 9 inches.
----------------------------------------

 > Thanks! Now I'd like to change my seat.
----------------------------------------
[Handoff Hook]: Assigning flight number FLT-417 to context.
FAQ Agent: Skipping item: HandoffCallItem
Handed off from FAQ Agent to Triage Agent
Triage Agent: Skipping item: HandoffCallItem
Handed off from Triage Agent to Seat Booking Agent
Seat Booking Agent: Okay, I can help you with that. First, can I get your confirmation number? And what seat number would you like to change to?
----------------------------------------

 > My confirmation is ABC123.
----------------------------------------
Seat Booking Agent: Okay, and what seat number would you like to change to?
----------------------------------------

 > I'd like seat 14A.
----------------------------------------
Seat Booking Agent: Calling a tool
Seat Booking Agent: Tool call output: Updated seat to 14A for confirmation number ABC123
Seat Booking Agent: Okay, I have updated your seat to 14A. Is there anything else I can help you with?
----------------------------------------

 > quit
Seat Booking Agent: Okay, have a great day!
```

This interaction flawlessly demonstrates the power of the multi-agent architecture. The `TriageAgent` correctly routed to the `FAQAgent`. When the topic changed, the `FAQAgent` correctly handed back to the `TriageAgent`, which then correctly routed to the `SeatBookingAgent`. The `on_handoff` hook injected the necessary flight number into the context, and the `SeatBookingAgent` successfully used its tool to complete the stateful task.

## Chapter Summary

This case study synthesized everything we've learned into a single, cohesive application that mirrors real-world requirements. We saw how to:

-   **Decompose a Problem:** Break a broad "customer service" task into specialized roles for different agents.
-   **Orchestrate with Handoffs:** Use a `TriageAgent` to intelligently route user requests, forming the backbone of the application logic.
-   **Manage State:** Employ a shared `AirlineAgentContext` object to pass crucial information between agents under the hood.
-   **Execute Side Effects:** Use the `on_handoff` callback to simulate interactions with external systems at the precise moment of delegation.
-   **Build Conversational Loops:** Maintain state across multiple user turns by updating the agent and conversation history from the `RunResult`.

The patterns here—triage and delegate, shared context, and callback hooks—are not limited to customer service. They form a powerful and reusable blueprint for building a vast range of sophisticated, multi-agent AI systems.

This brings us to the end of our journey in this book. The future of AI agents is bright, and with tools like OpenAI Agents SDK, developers are at the forefront of shaping that future. I encourage you to experiment, build, share, and continue learning as this technology unfolds.

*Happy Agent Building!*
