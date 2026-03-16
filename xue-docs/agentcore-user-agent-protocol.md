# Protocol Between AgentCore Agents and Users/Clients

## Overview

AgentCore does not enforce a single rigid protocol — it provides a flexible HTTP-based invocation API with support for open standards. This document explains the protocols involved when you build an agent with AgentCore, host it, and expose it to end users.

## 1. Primary Protocol: HTTP REST API (`/invocations`)

When you deploy an agent to AgentCore, it is exposed via a managed REST endpoint:

```
POST https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{agent_arn}/invocations?qualifier=DEFAULT
```

### Request (JSON)

```json
{
  "input": {
    "prompt": "Book me a flight from NYC to London on March 20",
    "conversation_id": "abc-123",
    "user_id": "user-42"
  }
}
```

### Response (JSON)

```json
{
  "output": {
    "message": "I found several flights from NYC to London on March 20...",
    "timestamp": "2026-03-16T02:30:00Z",
    "metadata": { "tools_used": ["flight_search", "booking_api"] }
  }
}
```

### Key Headers

| Header | Description |
|---|---|
| `Authorization` | Bearer token (OAuth2/JWT via Cognito) or SigV4 for IAM |
| `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` | Maintains conversation state across requests |
| `Content-Type` | `application/json` |

> **Note:** This is a custom AWS-specific REST API, not an off-the-shelf standard like OpenAI's Chat Completions API. The payload schema (`input.prompt`, `output.message`) is defined by your agent code inside the `@app.entrypoint` decorator — you have control over it.

## 2. Agent Entry Point Pattern

Agents are wrapped in `BedrockAgentCoreApp` which handles the runtime abstraction:

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload, context):
    session_id = getattr(context, "session_id", "default-session")
    prompt = payload.get("prompt", "")
    # Agent execution code...
    yield response_text  # Streaming support

app.run()
```

### Context Object Properties

| Property | Description |
|---|---|
| `session_id` | Unique conversation/session identifier |
| `request_headers` | HTTP headers dict |
| `custom_attributes` | Custom runtime attributes |

## 3. Optional Standard: A2A Protocol (Agent-to-Agent)

AgentCore also supports the [A2A protocol](https://a2a-protocol.org/dev/specification/), an open standard for agent interoperability. This is useful when:

- Other **agents** (not just human users) need to interact with your agent
- You want cross-framework interoperability (e.g., a Google ADK agent calling your Strands-based agent)

### A2A Client Example

```python
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

# Resolve agent card
resolver = A2ACardResolver(httpx_client=httpx_client, base_url=runtime_url)
agent_card = await resolver.get_agent_card()

# Create A2A client and send message
client = factory.create(agent_card)
msg = Message(
    kind="message",
    role=Role.user,
    parts=[Part(TextPart(kind="text", text="Book a flight to London"))],
    message_id=uuid4().hex,
)
async for event in client.send_message(msg):
    response_text = format_agent_response(event)
```

### Repository Examples

- `01-tutorials/01-AgentCore-runtime/05-hosting-a2a/` — Hosting A2A agents with SigV4 auth
- `02-use-cases/A2A-multi-agent-incident-response/` — Multi-agent orchestration via A2A

## 4. MCP (Model Context Protocol) for Tools

For an agent's **tools** (e.g., flight booking, hotel booking, Google Calendar), AgentCore uses **MCP** as the standardized tool protocol. The AgentCore **Gateway** can automatically convert REST APIs and Lambda functions into MCP-compatible tools.

### Gateway MCP Client Example

```python
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

client = MCPClient(
    lambda: streamablehttp_client(
        url=gateway_url,
        headers={"Authorization": f"Bearer {access_token}"}
    ),
    prefix="gateway",
    tool_filters={"allowed": [re.compile(tool_filter_pattern)]}
)
```

## 5. Authentication Mechanisms

### OAuth2 / Bearer Token (Gateway Integration)

Uses Cognito OAuth2 client credentials flow:

```python
token_url = f"https://{domain}.auth.{region}.amazoncognito.com/oauth2/token"
response = requests.post(token_url, data={
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
    "scope": scope,
})
access_token = response.json()["access_token"]
```

### AWS SigV4 (IAM Authentication for A2A)

```python
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
SigV4Auth(credentials, "bedrock-agentcore", region).add_auth(aws_request)
```

### JWT Token Extraction in Agent Runtime

```python
def _get_bearer_token(context) -> Optional[str]:
    auth = (getattr(context, "request_headers", None) or {}).get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None
```

## 6. Typical Webapp Architecture

For a travel agent exposed as a webapp, the architecture looks like:

```
Users (Browser) ──HTTP──▶ Your Webapp (Streamlit/React/FastAPI)
                              │
                              │ POST /runtimes/{arn}/invocations
                              │ + Bearer Token + Session-Id header
                              ▼
                         AgentCore Runtime (your agent)
                              │
                              │ MCP (via Gateway)
                              ▼
                    Flight API, Hotel API, Google Calendar API
```

- **User ↔ Webapp**: Whatever protocol you choose (standard HTTP, WebSocket, etc.)
- **Webapp ↔ Agent**: AWS HTTP REST API with JSON payloads, authenticated via OAuth2 (Cognito) or IAM SigV4
- **Agent ↔ Tools**: MCP via AgentCore Gateway

### Blueprint Examples

| Blueprint | Directory | Description |
|---|---|---|
| Travel Concierge | `05-blueprints/travel-concierge-agent/` | Multi-agent orchestration with Gateway |
| Customer Service | `05-blueprints/end-to-end-customer-service-agent/` | Streamlit + FastAPI + AgentCore |
| Customer Support | `05-blueprints/customer-support-agent-with-agentcore/` | Memory integration |
| Shopping Concierge | `05-blueprints/shopping-concierge-agent/` | Shopping/cart tools |

## 7. Streaming & Async Responses

Agents support streaming by yielding events from the entrypoint:

```python
@app.entrypoint
async def invoke(payload, context):
    stream = agent.stream_async(payload.get("prompt"))
    async for event in stream:
        if "data" in event and isinstance(event["data"], str):
            yield event["data"]
```

## 8. Key Takeaway

| Layer | Protocol | Standard? |
|---|---|---|
| User ↔ Webapp | Your choice (HTTP, WebSocket) | Your choice |
| Webapp ↔ Agent | AWS HTTP REST API (`/invocations`) | AWS-specific |
| Agent ↔ Agent | A2A Protocol | Open standard |
| Agent ↔ Tools | MCP via Gateway | Open standard |
| Authentication | OAuth2 (Cognito) or SigV4 | Industry standards |

The user-facing protocol is **HTTP REST with JSON payloads** — the exact schema is flexible and defined by your `@app.entrypoint` handler. It is **not** OpenAI-compatible or any other third-party standard out of the box, but it **does** support the open A2A standard for agent-to-agent communication. For tool integration, it uses MCP as the standard protocol via the Gateway.
