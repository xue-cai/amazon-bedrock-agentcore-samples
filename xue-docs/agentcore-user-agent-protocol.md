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

AgentCore supports two categories of authentication depending on the scenario:

| Category | Flow | Use Case |
|---|---|---|
| **M2M (Machine-to-Machine)** | OAuth2 Client Credentials | Service-to-service calls (agent → Gateway, internal APIs) |
| **USER_FEDERATION (3-Legged OAuth)** | OAuth2 Authorization Code | Accessing a *user's own* resources (Google Calendar, GitHub repos) |

### 5a. M2M: OAuth2 Client Credentials (Service-to-Service)

Uses Cognito OAuth2 client credentials flow — no user interaction required:

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

### 5b. USER_FEDERATION: 3-Legged OAuth (User-Delegated Access)

When the agent needs to access a **user's own resources** (e.g., "read my Google Calendar"), the user must explicitly grant consent in their browser. This uses the standard **OAuth 2.0 Authorization Code Grant** (also called 3-Legged OAuth or 3LO), managed by **AgentCore Identity**.

See [Section 6: OAuth User-Consent Flow](#oauth-user-consent-flow-3-legged-oauth) for the full walkthrough.

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

### OAuth User-Consent Flow (3-Legged OAuth)

When the agent needs to access a **user's personal resources** — for example, reading Alice's Google Calendar or accessing Bob's GitHub repos — the user must explicitly grant permission via their browser. AgentCore Identity manages this using the **OAuth 2.0 Authorization Code Grant** (3LO).

#### How It Works: Step-by-Step

```
                                    ┌─────────────────┐
                                    │  OAuth Provider  │
                                    │ (e.g., Google)   │
                                    └────┬───────▲─────┘
                                  4. User│       │3. Redirect
                                  grants │       │   to Google
                                  consent│       │
                                    ┌────▼───────┴─────┐
                                    │  User's Browser   │
                                    └────┬───────▲─────┘
                                  5. Google    │2. Webapp opens
                                  redirects   │   auth URL in
                                  to callback │   user's browser
                                    ┌────▼───────┴─────┐
 1. "Check my      ┌──────────┐    │  Callback Server  │
    calendar" ────▶│  Webapp   │    │  (/oauth2/callback)│
                   │(Streamlit)│    └────────┬──────────┘
                   └────┬──────┘             │6. complete_resource_token_auth()
                        │                    ▼
                        │ POST        ┌─────────────────┐
                        │/invocations │ AgentCore        │
                        ▼             │ Identity Service │
                   ┌──────────────┐   │ (Token Vault)    │
                   │ AgentCore    │   └────────┬─────────┘
                   │ Runtime      │            │7. Token stored
                   │ (your agent) │            │   per-user,
                   │              │◄───────────┘   encrypted
                   └──────┬───────┘
                          │ 8. Agent retries with
                          │    user's access token
                          ▼
                   ┌──────────────┐
                   │ Google       │
                   │ Calendar API │
                   └──────────────┘
```

**Step 1 — User makes a request**: The user asks the agent (via the webapp) to do something that requires their personal data, e.g., *"Check my calendar for today."*

**Step 2 — Agent detects authorization is needed**: The agent's tool calls `@requires_access_token` with `auth_flow="USER_FEDERATION"`. AgentCore Identity checks its **token vault** for an existing token for this (user, resource) pair. If no token exists, it generates an **OAuth authorization URL** and returns it to the agent via the `on_auth_url` callback. The webapp presents this URL to the user (e.g., as a clickable link or redirect).

**Step 3 — User is redirected to the OAuth provider**: The user's browser opens the authorization URL, which points to the OAuth provider (e.g., Google's consent screen: `https://accounts.google.com/o/oauth2/v2/auth?...`).

**Step 4 — User grants consent**: The user sees the provider's consent screen (e.g., *"AgentCore wants to access your Google Calendar. Allow?"*) and clicks **Allow**. This is a standard browser-based interaction — the user must be in a browser.

**Step 5 — OAuth provider redirects to callback**: After consent, the provider redirects the user's browser to the pre-registered **callback URL** (e.g., `http://localhost:9090/oauth2/callback?session_id=...`). This callback server is a lightweight FastAPI app that your webapp runs alongside the agent.

**Step 6 — Callback server completes the flow**: The callback server receives the redirect, extracts the `session_id`, and calls `identity_client.complete_resource_token_auth(session_uri=session_id, user_identifier=user_token_identifier)`. This tells AgentCore Identity to exchange the authorization code for access and refresh tokens.

**Step 7 — Tokens stored in vault**: AgentCore Identity stores the tokens in an **encrypted, per-user token vault**. Each user gets their own token — Alice's Google Calendar token is completely separate from Bob's. The tokens are access-controlled so that only the specific (agent, user) pair can retrieve them.

**Step 8 — Agent retries with the token**: On subsequent invocations (or automatic retry), `@requires_access_token` now finds a valid token in the vault and injects it as the `access_token` parameter. The agent uses it to call the Google Calendar API directly.

#### Key Implementation Details

**1. One-time setup — Create an OAuth2 Credential Provider:**

```python
# Register Google as an OAuth provider with AgentCore Identity
identity_client = boto3.client("bedrock-agentcore-control")
identity_client.create_oauth2_credential_provider(
    name="google-cal-provider",
    credentialProviderVendor="GoogleOauth2",
    oauth2ProviderConfigInput={
        "googleOauth2ProviderConfig": {
            "clientId": "your-google-client-id",
            "clientSecret": "your-google-client-secret",
        }
    },
)
```

**2. Agent tool — Use `@requires_access_token` decorator:**

```python
from bedrock_agentcore.identity.auth import requires_access_token

@tool(name="Get_calendar_events_today")
async def get_calendar():
    @requires_access_token(
        provider_name="google-cal-provider",        # Matches the provider created above
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        auth_flow="USER_FEDERATION",                 # 3-legged OAuth (user consent required)
        on_auth_url=on_auth_url,                     # Callback to surface auth URL to user
        callback_url=os.environ["CALLBACK_URL"],     # Where Google redirects after consent
    )
    async def get_calendar_events_today(access_token: Optional[str] = "") -> str:
        creds = Credentials(token=access_token)
        service = build("calendar", "v3", credentials=creds)
        # ... call Calendar API with user's token
```

**3. Callback server — Handle the OAuth redirect:**

```python
from bedrock_agentcore.services.identity import IdentityClient

# In the callback server (FastAPI app running on port 9090):
@app.get("/oauth2/callback")
async def handle_callback(session_id: str):
    identity_client.complete_resource_token_auth(
        session_uri=session_id,
        user_identifier=user_token_identifier  # Binds token to specific user
    )
    return HTMLResponse("<h1>Authorization successful! You can close this tab.</h1>")
```

#### Token Lifecycle

| Aspect | Behavior |
|---|---|
| **First request** | User must consent in browser (one-time per resource) |
| **Subsequent requests** | Token retrieved from vault automatically — no user interaction |
| **Token expiry** | AgentCore Identity uses the **refresh token** to obtain a new access token silently |
| **Refresh token expiry** | User must re-consent in browser (typically 90+ days) |
| **Per-user isolation** | Each user's tokens are stored separately and encrypted at rest |

#### M2M vs USER_FEDERATION: When to Use Which

| Scenario | Auth Flow | User Browser Needed? |
|---|---|---|
| Agent calls an internal API or database | `M2M` (Client Credentials) | No |
| Agent reads *the user's own* Google Calendar | `USER_FEDERATION` (3LO) | Yes (for initial consent) |
| Agent accesses a shared company Confluence | Depends — `M2M` for service account, `USER_FEDERATION` for per-user access | Depends |

#### Repository Examples

| Tutorial | Directory | What It Shows |
|---|---|---|
| Google Calendar 3LO | `01-tutorials/03-AgentCore-identity/05-Outbound_Auth_3lo/` | Full 3LO flow with Google Calendar |
| GitHub OAuth | `01-tutorials/03-AgentCore-identity/06-Outbound_Auth_Github/` | 3LO with GitHub repos |
| LinkedIn Auth Code Grant | `01-tutorials/02-AgentCore-gateway/13-outbound-auth-code-grant/` | Gateway + LinkedIn 3LO |
| IDE Gateway + Atlassian | `01-tutorials/02-AgentCore-gateway/04-integration/03-ide-gateway-tool/` | VS Code + Confluence 3LO |

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
| Auth (service-to-service) | OAuth2 Client Credentials (Cognito) or SigV4 | Industry standards |
| Auth (user-delegated) | OAuth2 Authorization Code (3LO) via AgentCore Identity | Industry standard |

The user-facing protocol is **HTTP REST with JSON payloads** — the exact schema is flexible and defined by your `@app.entrypoint` handler. It is **not** OpenAI-compatible or any other third-party standard out of the box, but it **does** support the open A2A standard for agent-to-agent communication. For tool integration, it uses MCP as the standard protocol via the Gateway.
