---
title: "Chapter 7 - Integrating with External APIs: OpenAPI & API Hub"
date: "2025-08-22 11:00:00 +0200"
categories: [Gen AI, Agentic SDKs, Agent Development Kit]
tags: [Generative AI, Agentic AI, Gen AI, Agentic SDKs, Agent Development Kit, Building Intelligent Agents with Google ADK]
image:
  path: /assets/img/adk-book-cover.jpg
  alt: "Building Intelligent Agents with Google ADK"
---

> This article is part of my web book series. All of the chapters can be found [here](https://iamulya.one/tags/building-intelligent-agents-with-google-adk/) and the code is available on [Github](https://github.com/iamulya/adk-book-code). For any issues around this book, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

So far, we've equipped our agents with custom Python functions and pre-built ADK tools. However, a vast amount of the world's information and functionality resides in external REST APIs. To truly empower our agents, we need a way for them to understand and interact with these APIs. This chapter focuses on the key ADK mechanisms for achieving this: securely leveraging **OpenAPI specifications** directly.

## Understanding OpenAPI (Swagger) Specs

The **OpenAPI Specification** (formerly known as Swagger Specification) is a standard, language-agnostic interface description for RESTful APIs. It allows both humans and computers to discover and understand the capabilities of a service without requiring access to source code, additional documentation, or network traffic inspection.

An OpenAPI document (usually a JSON or YAML file) describes:

- **API Endpoints (Paths):** The URLs available (e.g., `/users`, `/products/{productId}`).
- **HTTP Methods:** The operations allowed on each path (e.g., `GET`, `POST`, `PUT`, `DELETE`).
- **Parameters:** Expected inputs for each operation (path parameters, query parameters, header parameters, request body).
- **Request Bodies:** The structure of data sent in `POST` or `PUT` requests.
- **Responses:** The structure of data returned for different HTTP status codes.
- **Authentication Schemes:** How the API is secured (e.g., API keys, OAuth2).
- **Data Models (Schemas):** Definitions of complex data types used in requests and responses.

**Example Snippet of an OpenAPI (JSON) Document:**

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Simple Pet Store API",
    "version": "1.0.0",
    "description": "A simple API to manage pets"
  },
  "servers": [
    {
      "url": "<https://api.example.com/v1>"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "responses": {
          "200": {
            "description": "A list of pets.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/Pet"
                  }
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPet",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/PetInput"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Pet created"
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "getPetById",
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Information about the pet",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "PetInput": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "tag": { "type": "string" }
        },
        "required": ["name"]
      },
      "Pet": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "tag": { "type": "string" }
        }
      }
    }
  }
}

```

ADK can parse these specifications and automatically generate tools for each defined API operation.

## The `OpenAPIToolset`: Generating Tools from Specs

The `google.adk.tools.openapi_tool.OpenAPIToolset` is a powerful `BaseToolset` implementation that takes an OpenAPI specification (as a dictionary or a string) and creates a `RestApiTool` for each operation defined in it.

- **`google.adk.tools.openapi_tool.RestApiTool`**: A specialized `BaseTool` that knows how to:
    - Construct HTTP requests based on the operation's definition and arguments provided by the LLM.
    - Handle parameters in the path, query, header, and request body.
    - Make the actual HTTP call (using the `requests` library internally).
    - Process the API's response.
    - Manage authentication.

**Creating an `OpenAPIToolset`:**

Let's create an agent that lets your use natural language to query the [Petstore API](https://petstore.swagger.io/)

```python
from google.adk.agents import Agent
from google.adk.tools.openapi_tool import OpenAPIToolset # Key import
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# Assume the OpenAPI spec JSON from section 7.1 is saved in "petstore_openapi.json"
# Or, for this example, let's embed it as a string:
petstore_spec_str = """
{
  "openapi": "3.0.0",
  "info": { "title": "Simple Pet Store API", "version": "1.0.0" },
  "servers": [ { "url": "https://petstore.swagger.io/v2" } ],   
  "paths": {
    "/pet/findByStatus": {
      "get": {
        "summary": "Finds Pets by status",
        "operationId": "findPetsByStatus",
        "parameters": [
          {
            "name": "status", "in": "query", "required": true,
            "description": "Status values that need to be considered for filter",
            "schema": { "type": "string", "enum": ["available", "pending", "sold"] }
          }
        ],
        "responses": {
          "200": { "description": "successful operation", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/Pet" } } } } }
        }
      }
    },
    "/pet": {
      "post": {
        "summary": "Add a new pet to the store",
        "operationId": "addPet",
        "requestBody": {
          "required": true,
          "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Pet" } } }
        },
        "responses": { "200": { "description": "Successfully added pet" }}
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "properties": {
          "id": { "type": "integer", "format": "int64" },
          "name": { "type": "string", "example": "doggie" },
          "status": { "type": "string", "description": "pet status in the store", "enum": ["available", "pending", "sold"] }
        },
        "required": ["name"]
      }
    }
  }
}
"""

# 1. Initialize the OpenAPIToolset
# We pass the spec string and specify its type.
petstore_toolset = OpenAPIToolset(
    spec_str=petstore_spec_str,
    spec_str_type="json" # or "yaml" if it were YAML
)

# If you had the spec as a Python dictionary:
# petstore_spec_dict = json.loads(petstore_spec_str)
# petstore_toolset = OpenAPIToolset(spec_dict=petstore_spec_dict)


# 2. Create an agent and provide the toolset
# The agent will automatically get all tools generated from the spec.
petstore_agent = Agent(
    name="petstore_manager",
    model=DEFAULT_LLM,
    instruction="You are an assistant for managing a pet store. Use the available tools to find or add pets.",
    tools=[petstore_toolset] # Pass the toolset instance
)

# --- Example of running this agent ---
if __name__ == "__main__":
    runner = InMemoryRunner(agent=petstore_agent, app_name="PetStoreApp")

    user_id="pet_user" 
    session_id="s_petstore"
    create_session(runner, session_id, user_id)

    prompts = [
        "Find all available pets.",
        "Can you add a new pet named 'Buddy' with status 'available' and id 12345?",
    ]

    async def main(): # Using async for runner.run
        for prompt_text in prompts:
            print(f"
YOU: {prompt_text}")
            user_message = Content(parts=[Part(text=prompt_text)], role="user")
            print("PETSTORE_MANAGER: ", end="", flush=True)
            # In a real scenario, the Petstore API would be called.
            # Since petstore.swagger.io is a live mock, these calls will actually work!
            for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print()

    import asyncio
    asyncio.run(main())
```

**How it Works:**

1. `OpenAPIToolset` parses the `petstore_spec_str`.
2. It identifies operations like `findPetsByStatus` (GET `/pet/findByStatus`) and `addPet` (POST `/pet`).
3. For each operation, it creates a `RestApiTool` instance:
    - `RestApiTool(name="find_pets_by_status", ...)`
    - `RestApiTool(name="add_pet", ...)`
    (Note: `operationId` is converted to `snake_case` for the tool name).
4. When the `petstore_agent` needs to find available pets, the LLM sees the `find_pets_by_status` tool, its description (from the spec's `summary`), and its `status` parameter.
5. The LLM generates a function call like `find_pets_by_status(status="available")`.
6. The `RestApiTool` for `find_pets_by_status` receives these arguments, constructs an HTTP GET request to `https://petstore.swagger.io/v2/pet/findByStatus?status=available`, executes it, and returns the JSON response to the LLM.
7. The LLM then formulates a natural language answer based on the API response.

![*Diagram: Sequence of an `LlmAgent` using a tool from an `OpenAPIToolset`.*](/assets/img/2025-08-22-integrating-with-external-apis-openapi-&-api-hub/figure-1.png)



> ## Best Practice: Well-Defined operationId and summary/description
> 
> The operationId in your OpenAPI spec is typically used to generate the tool name (converted to snake_case). Make it descriptive. The summary and description fields for paths and operations are crucial for the LLM to understand what each tool does and when to use it. Invest time in writing clear and concise OpenAPI documentation.
> {: .prompt-info }


> ## Reusability and Standardization
> 
> OpenAPI is a widely adopted standard. If an API has an OpenAPI spec, you can integrate it into ADK with minimal effort using OpenAPIToolset. This promotes reusability and standardization in how your agents interact with diverse APIs.
> {: .prompt-info }

## Handling Authentication with OpenAPI Tools

APIs are rarely public. `OpenAPIToolset` and `RestApiTool` provide mechanisms to handle common authentication schemes defined in OpenAPI specs.

- **Global Authentication:** You can provide a default `auth_scheme` and `auth_credential` when initializing the `OpenAPIToolset`. This will apply to all tools generated by that toolset unless overridden at the operation level in the spec.
- **Operation-Specific Authentication:** If an operation in the OpenAPI spec defines its own `security` requirements, the `RestApiTool` generated for that operation will attempt to use that specific scheme.
- **`ToolAuthHandler`**: Internally, `RestApiTool` uses a `ToolAuthHandler` to manage the authentication lifecycle. This handler can:
    - Check if existing credentials (e.g., an access token stored in `ToolContext.state`) are available and valid.
    - If not, and if the scheme requires user interaction (like OAuth2 Authorization Code flow), it can signal back to the ADK framework to request credentials from the user (via `tool_context.request_credential()`).
    - Use `CredentialExchangers` (e.g., `OAuth2CredentialExchanger`) to trade initial credentials for an actual access token.

**Example: `OpenAPIToolset` with API Key Authentication**

Let's imagine our Petstore API required an API key.

Part of `petstore_openapi_auth.json` (conceptual):

```json
{
  // ... other spec parts ...
  "components": {
    "securitySchemes": {
      "ApiKeyAuth": { // Name of the security scheme
        "type": "apiKey",
        "in": "header",   // Or "query" or "cookie"
        "name": "X-API-KEY" // The name of the header/query param
      }
    }
    // ... schemas ...
  },
  "security": [ // Global security requirement
    {
      "ApiKeyAuth": [] // Requires ApiKeyAuth for all operations
    }
  ]
}

```

Let's write a Spotify Agent (using a simplified Spec) which will be able to get Spotify catalog information about artists, albums, tracks or playlists that match a keyword string. Pay special attention to how the authorization is handled.

```python
import asyncio

from google.adk.agents import Agent
from google.adk.tools.openapi_tool import OpenAPIToolset
from google.adk.auth import AuthCredential, AuthCredentialTypes
from fastapi.openapi.models import APIKey, APIKeyIn
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM
load_environment_variables()

# --- Spotify OpenAPI Spec (Simplified) ---
SPOTIFY_API_SPEC_STR = """
openapi: 3.0.0
info:
  title: Spotify Web API (Simplified for ADK Demo)
  version: v1
  description: Subset of Spotify Web API for searching tracks.
servers:
  - url: https://api.spotify.com/v1
components:
  securitySchemes:
    SpotifyApiKeyAuth:
      type: apiKey
      in: header
      name: Authorization # The name of the header is 'Authorization'
security:
  - SpotifyApiKeyAuth: [] # Applies this scheme globally
paths:
  /search:
    get:
      summary: Search for an Item
      operationId: searchForItem # This will become the tool name (search_for_item)
      description: |
        Get Spotify catalog information about artists, albums, tracks or playlists
        that match a keyword string.
      parameters:
        - name: q
          in: query
          description: "Search query keywords and optional field filters and operators. Example: 'track:The Sign artist:Ace of Base'"
          required: true
          schema:
            type: string
        - name: type
          in: query
          description: "A comma-separated list of item types to search across. Valid types are: album, artist, playlist, track, show, episode, audiobook."
          required: true
          schema:
            type: string
        - name: market
          in: query
          description: "An ISO 3166-1 alpha-2 country code. If a market is given, only content playable in that market will be returned."
          required: false
          schema:
            type: string
        - name: limit
          in: query
          description: "Maximum number of results to return. Default: 20. Minimum: 1. Maximum: 50."
          required: false
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 50
        - name: offset
          in: query
          description: "The index of the first result to return. Default: 0 (the first result). Maximum offset: 100,000. Use with limit to get the next page of search results."
          required: false
          schema:
            type: integer
            default: 0
      responses:
        '200':
          description: Search results.
          content:
            application/json:
              schema:
                type: object # A more complete spec would detail the response structure
                properties:
                  tracks:
                    type: object
                    properties:
                      items:
                        type: array
                        items:
                          type: object
                          properties:
                            name:
                              type: string
                            artists:
                              type: array
                              items:
                                type: object
                                properties:
                                  name:
                                    type: string
                            album:
                              type: object
                              properties:
                                name:
                                  type: string
                            external_urls:
                              type: object
                              properties:
                                spotify:
                                  type: string
                  # ... other types like artists, albums, etc.
        '400':
          description: Bad request.
        '401':
          description: Bad or expired token. This can happen if the user revoked a token or the access token has expired. You should re-authenticate the user.
        '403':
          description: Bad OAuth request (wrong consumer key, bad nonce, expired timestamp...). Unfortunately, re-authenticating the user won't help here.
        '429':
          description: The app has exceeded its rate limits.
"""

# --- Authentication Setup ---
# IMPORTANT: You need a Spotify Access Token for this to work.
# 1. Go to https://developer.spotify.com/dashboard/ and create an app.
# 2. Get your Client ID and Client Secret.
# 3. Use the Client Credentials Flow to obtain an Access Token.
#    A simple way to do this is with curl:
#    curl -X "POST" -H "Authorization: Basic <BASE64_ENCODED_CLIENT_ID:CLIENT_SECRET>" -d grant_type=client_credentials https://accounts.spotify.com/api/token
#    (Replace <BASE64_ENCODED_CLIENT_ID:CLIENT_SECRET> with your actual base64 encoded "client_id:client_secret" string)
# 4. The response will contain an "access_token". Prepend "Bearer " to it.
#
# For this example, we'll treat the "Bearer <ACCESS_TOKEN>" string as our "API Key".
SPOTIFY_BEARER_TOKEN = "Bearer YOUR_SPOTIFY_ACCESS_TOKEN"  # <<< REPLACE THIS!

spotify_api_key_auth_scheme = APIKey(
    type="apiKey",
    name="Authorization",
    **{"in": APIKeyIn.header} # We use **{"in": ...} to correctly pass the 'in' parameter, which is a reserved keyword in Python.
)

# The credential type must also match (`API_KEY`).
spotify_api_key_credential = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key=SPOTIFY_BEARER_TOKEN
)

# --- Toolset Setup ---
spotify_toolset = OpenAPIToolset(
    spec_str=SPOTIFY_API_SPEC_STR,
    spec_str_type="yaml",
    auth_scheme=spotify_api_key_auth_scheme,
    auth_credential=spotify_api_key_credential
)

# --- Agent Definition ---
spotify_agent = Agent(
    name="SpotifySearchAgent",
    model=DEFAULT_LLM, 
    instruction=(
        "You are a Spotify music search assistant. "
        "Use the 'searchForItem' tool to find tracks, artists, or albums on Spotify. "
        "When searching, you must specify the 'type' parameter (e.g., 'track', 'artist', 'album'). "
        "If the user asks to search for a song, use type 'track'. "
        "After getting results, list the names of up to 3 found items. "
        "If searching for tracks, also include the artist and album name if available."
    ),
    tools=[
        spotify_toolset
    ]
)

# --- Example of running this agent ---
if __name__ == "__main__":
    if "YOUR_SPOTIFY_ACCESS_TOKEN" in SPOTIFY_BEARER_TOKEN:
        print("="*80)
        print("!!! IMPORTANT WARNING !!!")
        print("It appears 'YOUR_SPOTIFY_ACCESS_TOKEN' might still be the placeholder")
        print("in the SPOTIFY_BEARER_TOKEN variable in this script.")
        print("The agent will likely fail to authenticate with the Spotify API.")
        print("Please update it with your actual Spotify Bearer token and try again.")
        print("See comments in the script for details on obtaining a token.")
        print("="*80)
        
        exit(1)

    runner = InMemoryRunner(agent=spotify_agent, app_name="SpotifySearchApp")

    user_id = "spotify_user"
    session_id = "s_spotify"
    create_session(runner, session_id, user_id)

    # Selected prompts for testing
    prompts = [
        "Find the song 'Stairway to Heaven'.",
        "Search for artists named 'Queen'.",
        "Can you find albums by 'Daft Punk'?",
        "What tracks are there by an artist named 'Lorde'?",
        "Search for the track 'Bohemian Rhapsody' by Queen.",
        "search for item q='Never Gonna Give You Up' type='track' limit=1"
    ]

    async def main_loop():
        print("-" * 70)

        for prompt_text in prompts:
            print(f"
YOU: {prompt_text}")
            user_message = Content(parts=[Part(text=prompt_text)], role="user")
            print(f"{spotify_agent.name.upper()}: ", end="", flush=True)

            # Use run_async as we are in an async main function
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=user_message
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print("
" + "-" * 70) # Separator for readability
            # Optional: Add a small delay if you're making many API calls in a loop
            await asyncio.sleep(1) # Helps avoid hitting rate limits if prompts are processed very fast

    asyncio.run(main_loop())
```


> ## Declarative Auth in OpenAPI
> 
> Defining security schemes directly in your OpenAPI spec is the best practice. RestApiTool will automatically pick these up. You then only need to provide the corresponding AuthCredential (e.g., the actual API key, OAuth client secrets) to the tool or toolset.
> {: .prompt-info }

## The `APIHubToolset`: Connecting to Google's API Hub

Google API Hub is a service for discovering, managing, and governing APIs within an organization. If your organization uses API Hub, the `google.adk.tools.apihub_tool.APIHubToolset` provides a convenient way to generate ADK tools directly from API specifications registered in API Hub.

It essentially wraps the `OpenAPIToolset` by first fetching the OpenAPI spec from API Hub and then using `OpenAPIToolset` to parse it.

Following example uses the [Swagger Petstore - OpenAPI 3.0 spec](https://petstore3.swagger.io/api/v3/openapi.json) as one of the APIs on Google's API Hub. This spec presents a few challenges to the `OpenAPIToolset` which we solve using a speciality function `is_valid_adk_tool` and a wrapper class `PatchedAPIHubToolset`.

**Challenges:**

- Certain OpenAPI specs use a relative URL as the Server URL. This leads to incorrect URL formations which leads to errors.

  *Solution*: Wrap the `OpenAPIToolset` with a custom `PatchedAPIHubToolset` which allows to set an override base url. This base url can be passed as a parameter in the `PatchedAPIHubToolset` constructor.

- ADK's `OpenApiSpecParser` currently creates a tool parameter with an empty name if the requestBody is *not* a JSON object. This will cause a`400 INVALID_ARGUMENT` error from the LLM Endpoints.

  *Solution*: Write a special function `is_valid_adk_tool` that will act as a filter to exclude tools that the ADK parser cannot handle correctly. This function is then used as `tool_filter` when initializing the `PatchedAPIHubToolset`.

```python
from google.adk.agents import Agent
from google.adk.tools.apihub_tool import APIHubToolset
from google.adk.runners import InMemoryRunner
from google.adk.tools.openapi_tool import RestApiTool
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.genai.types import Content, Part
import os
import sys
from building_intelligent_agents.utils import load_environment_variables, create_session, DEFAULT_LLM

load_environment_variables()

# This requires:
# 1. `google-cloud-secret-manager` to be installed if using Secret Manager for API Hub client auth.
# 2. Your environment to be authenticated to GCP with permissions to access API Hub
#    and potentially Secret Manager if API Hub client uses it.
#    (e.g., `gcloud auth application-default login`)
# 3. The API to be registered in API Hub.

# Replace with your actual API Hub resource name
# Format: projects/{project}/locations/{location}/apis/{api}
# Or optionally: projects/{project}/locations/{location}/apis/{api}/versions/{version}
# Or even: projects/{project}/locations/{location}/apis/{api}/versions/{version}/specs/{spec}
APIHUB_RESOURCE_NAME = os.getenv("MY_APIHUB_API_RESOURCE_NAME") # e.g., "projects/my-gcp-project/locations/us-central1/apis/my-customer-api"

class PatchedAPIHubToolset(APIHubToolset):
    """
    A patched version of APIHubToolset that manually sets the server URL
    to work around a bug where the base_url or server.url is a relative url.
    """
    def __init__(self, *args, **kwargs):
        # Allow passing a custom base_url
        self.override_base_url = kwargs.pop("override_base_url", None)
        super().__init__(*args, **kwargs)

    async def get_tools(self, readonly_context=None) -> list[RestApiTool]:
        # Get the tools from the parent class
        tools = await super().get_tools(readonly_context)

        if self.override_base_url:
            print(f"🔧 Applying patch: Overriding base URL for all tools with '{self.override_base_url}'")
            for tool in tools:
                # Manually set the base_url on the tool's endpoint
                tool.endpoint.base_url = self.override_base_url
        return tools

def is_valid_adk_tool(tool: RestApiTool, ctx=None) -> bool:
    """
    A filter to exclude tools that the ADK parser cannot handle correctly.

    The ADK's OpenApiSpecParser currently creates a tool parameter with an
    empty name if the requestBody is not a JSON object. This causes a
    `400 INVALID_ARGUMENT` error from the Gemini API.

    This filter identifies and excludes such tools.
    """
    operation = tool._operation_parser._operation
    if not operation.requestBody or not operation.requestBody.content:
        # No request body, so it's a valid tool (e.g., GET with params in URL)
        return True

    # Check the first content type (usually 'application/json')
    for media_type in operation.requestBody.content.values():
        if media_type.schema_ and media_type.schema_.type != 'object':
            # This is the problematic case: a body that isn't a named object.
            print(f"Filtering out tool '{tool.name}' due to non-object requestBody (type: {media_type.schema_.type}).")
            return False

    # If all checks pass, the tool is considered valid.
    return True

apihub_connected_agent = None

if not APIHUB_RESOURCE_NAME:
    print("Error: MY_APIHUB_API_RESOURCE_NAME environment variable must be set.", file=sys.stderr)
else:
    try:
        petstore_auth_scheme, petstore_auth_credential = token_to_scheme_credential(
            token_type="apikey",
            location="header",
            name="api_key", # This must match the name in the OpenAPI spec's security scheme
            credential_value="special-key"
        )

        # Apply the filter and the auth config when creating the toolset
        # Use the new PatchedAPIHubToolset class
        apihub_toolset = PatchedAPIHubToolset(
            apihub_resource_name=APIHUB_RESOURCE_NAME,
            tool_filter=is_valid_adk_tool,
            auth_scheme=petstore_auth_scheme,
            auth_credential=petstore_auth_credential,
            # Provide the correct server URL here
            override_base_url="https://petstore3.swagger.io/api/v3"
        )

        apihub_connected_agent = Agent(
            name="apihub_connector",
            model=DEFAULT_LLM,
            instruction="You can interact with our company's custom API, registered in API Hub. Use the available tools.",
            tools=[apihub_toolset]
        )
        print("APIHubToolset and Agent initialized successfully.")

    except Exception as e:
        print(f"Could not initialize APIHubToolset. Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if not apihub_connected_agent:
        print("Agent could not be created. Exiting.", file=sys.stderr)
        sys.exit(1)

    runner = InMemoryRunner(agent=apihub_connected_agent, app_name="APIHubApp")
    user_id = "apihub_user"
    session_id = "s_apihub"
    create_session(runner, session_id, user_id)

    prompt_text = "Can you add a new pet named 'Buddy' with status 'available' and id 12345?"
    print(f"
YOU: {prompt_text}")

    async def main():
        user_message = Content(parts=[Part(text=prompt_text)], role="user")
        print("PETSTORE_MANAGER: ", end="", flush=True)

        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
        print()

    import asyncio
    asyncio.run(main())
```

The `APIHubToolset` handles:

1. Authenticating with the API Hub service (using application default credentials, an explicit access token, or a service account JSON).
2. Fetching the specified API resource.
3. Resolving which API Spec to use (e.g., latest version, specific spec).
4. Downloading the OpenAPI spec content.
5. Instantiating an `OpenAPIToolset` with the fetched spec.


> ## Centralized API Management with API Hub
> 
> If your organization uses API Hub, APIHubToolset is the preferred way to integrate those APIs into ADK. It ensures your agents are always using the centrally managed and governed API definitions.
> {: .prompt-info }

## Using Pre-packaged Google API Toolsets

ADK provides pre-packaged toolsets for common Google APIs (like BigQuery, Calendar, Gmail, Docs, Sheets, YouTube) under `google.adk.tools.google_api_tool.google_api_toolsets`. These are essentially `GoogleApiToolset` instances pre-configured for specific Google APIs.

`google.adk.tools.google_api_tool.GoogleApiToolset` is a specialized `BaseToolset` that:

1. Uses `GoogleApiToOpenApiConverter` to fetch the Google API Discovery document for a given API (e.g., `calendar`, `v3`).
2. Converts this Discovery document into an OpenAPI specification.
3. Initializes an `OpenAPIToolset` with this generated spec.
4. Wraps the resulting `RestApiTool`s into `GoogleApiTool` instances, which are pre-configured to use Google's OAuth2 (OpenID Connect) for authentication.


> ## OAuth2/OpenID Connect User Interaction Flow
> 
> For OAuth2 Authorization Code Grant or OpenID Connect, the process involves redirecting the user to an authorization server. ADK's ToolAuthHandler facilitates this by:
>  
> 1. The tool determines it needs OAuth.
> 2. It calls `tool_context.request_credential(auth_config)`, where `auth_config` includes the necessary OAuth parameters (client ID, scopes, redirect URI placeholder).
> 3. The ADK framework (e.g., Dev UI or your custom frontend) intercepts this, presents the auth URL from `auth_config.exchanged_auth_credential.oauth2.auth_uri` to the user.
> 4. User authenticates and is redirected back with an authorization code.
> 5. This code is sent back to the ADK agent (e.g., as a user message or a special event).
> 6. The `ToolAuthHandler` then uses this code to exchange it for an access token with the `OAuth2CredentialExchanger`.
> 
> This flow requires coordination between ADK, the agent, and the user interface. The Dev UI has some built-in support for handling these requests. We will use this when we write the Calendar agent next.
> {: .prompt-info }

You typically need to provide `client_id` and `client_secret` for your OAuth 2.0 application that has been authorized for the required Google API scopes.

Following is a Calendar agent which uses `CalendarToolset` from Google API Toolsets and filters in only the tools related to Events. You can use it to answer queries like "What are the next 3 events on my primary calendar?"


> This example should be run using the `adk web .` command, since OAuth Flow is triggered to authorize the reading of the calendar data.
> {: .prompt-info }

```python
from google.adk.agents import Agent
from google.adk.tools.google_api_tool.google_api_toolsets import CalendarToolset # Pre-packaged
from google.adk.runners import InMemoryRunner
import os

from building_intelligent_agents.utils import DEFAULT_LLM

# For Google API tools, you'll need OAuth 2.0 Client ID and Secret
# Get these from Google Cloud Console -> APIs & Services -> Credentials
# Ensure your OAuth consent screen is configured and you've added necessary scopes
# (e.g., <https://www.googleapis.com/auth/calendar.events.readonly> for listing events)
GOOGLE_CLIENT_ID = os.getenv("CALENDAR_OAUTH_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CALENDAR_OAUTH_CLIENT_SECRET")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("Error: CALENDAR_OAUTH_CLIENT_ID and CALENDAR_OAUTH_CLIENT_SECRET env vars must be set.")
    exit()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    calendar_tools = CalendarToolset(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        # Example filter: only expose tools to list events and get a specific event
        tool_filter=["calendar_events_list", "calendar_events_get"]
    )

    calendar_agent = Agent(
        name="calendar_assistant",
        model=DEFAULT_LLM,
        instruction="You are a helpful Google Calendar assistant. "
                    "When the user refers to 'my calendar' or 'my primary calendar', "
                    "you should use the special calendarId 'primary'. "
                    "Use the tools to manage calendar events.",
        tools=[calendar_tools]
    )

if __name__ == "__main__":
    if not calendar_agent:
        print("Skipping Google Calendar agent example due to missing OAuth credentials.")
    else:
        runner = InMemoryRunner(agent=calendar_agent, app_name="CalendarApp")
        # This will likely trigger an OAuth flow the first time in the Dev UI.
        # The Dev UI has mechanisms to help guide you through this.
        # For command-line, handling the full OAuth redirect flow is complex
        # and often requires a web server component for the redirect URI.
        # The Dev UI simplifies this for local development.
        prompt = "What are the next 3 events on my primary calendar?"
        print(f"
YOU: {prompt}")
        # ... (runner and event processing logic) ...
        # This interaction is best tested by running `adk web .` in the parent directory 
        # Make sure you have correctly set up the CALENDAR_OAUTH_CLIENT_ID and CALENDAR_OAUTH_CLIENT_SECRET variables, 
        # preferably through a .env file, otherwise the agent won't initialize properly.

        print("ASSISTANT: (This example is best run with `adk web .` to handle the OAuth flow)")
        print("  The Dev UI will guide you through authorizing access to your Google Calendar.")
```


> ## Use adk web for Google API Tools Requiring OAuth
> 
> Tools for Google APIs (Calendar, Gmail, Docs, etc.) usually require OAuth 2.0. The ADK Development UI (adk web) has built-in support to facilitate the OAuth consent and authorization code flow during local development, making it much easier to test these tools. Running them purely from a command-line script that doesn't handle web redirects for OAuth is challenging.
> {: .prompt-info }

When you send your query for the first time, it will kickoff the OAuth Flow. Upon successful authorization, the LLM will be in a position to provide the correct answer.

![Calendar Agent](/assets/img/adk-calendar.png)

**What's Next?**

You've now learned how to make your ADK agents vastly more powerful by integrating them with external REST APIs using OpenAPI specifications, API Hub, and pre-packaged Google API toolsets. This opens up a world of possibilities. Next we'll explore another set of powerful integrations, focusing on the MCP (Model Context Protocol) Toolbox, Application Integration, and bridging ADK with other popular agent frameworks like Langchain and CrewAI.
