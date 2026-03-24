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

## 5. Authentication

When your agent calls external APIs or tools, it needs authentication tokens. AgentCore Identity provides a unified decorator — `@requires_access_token` — that works for both service-to-service and user-delegated scenarios. The underlying OAuth flow differs, but the developer experience is intentionally similar. (For inbound authentication — how clients authenticate *to* your agent — see [Section 5.8](#58-inbound-authentication-clients--your-agent).)

| | **M2M (Service-to-Service)** | **USER_FEDERATION (User-Delegated)** |
|---|---|---|
| **When to use** | Agent calls an API as *itself* (service account) | Agent calls an API *on behalf of a specific user* |
| **Example** | Agent → Gateway → internal REST API | Agent → Google Calendar API *for Alice* |
| **OAuth grant type** | Client Credentials | Authorization Code (3-Legged OAuth / 3LO) |
| **User browser needed?** | Never | Yes — for initial consent (one-time per resource) |
| **Callback server needed?** | No | Yes — developer-deployed |

### 5.1 Agent Code Comparison

Both flows use the same decorator — `@requires_access_token` from `bedrock_agentcore.identity.auth`. The key differences are `auth_flow`, `scopes`, and callback parameters.

**M2M (Service-to-Service):**

```python
from bedrock_agentcore.identity.auth import requires_access_token

@requires_access_token(
    provider_name="my-cognito-provider",   # Cognito-backed credential provider
    scopes=[],                              # M2M: empty scopes
    auth_flow="M2M",                        # Client Credentials flow
)
def get_gateway_token(access_token: str) -> str:
    return access_token

# Usage: call the function, token is injected automatically
token = get_gateway_token()
headers = {"Authorization": f"Bearer {token}"}
# Pass headers to Gateway MCP client, REST API, etc.
```

**USER_FEDERATION (User-Delegated):**

```python
from bedrock_agentcore.identity.auth import requires_access_token
from strands import tool

@tool(name="Get_calendar_events")
async def get_calendar():
    @requires_access_token(
        provider_name="google-cal-provider",                # Google/GitHub/etc.
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],  # OAuth scopes
        auth_flow="USER_FEDERATION",                        # 3-Legged OAuth
        on_auth_url=on_auth_url,                            # Callback for consent URL
        callback_url=os.environ["CALLBACK_URL"],            # Redirect URI after consent
    )
    async def get_events(access_token: Optional[str] = "") -> str:
        creds = Credentials(token=access_token)
        service = build("calendar", "v3", credentials=creds)
        # ... call API with user's token
    return await get_events()
```

**Parameter differences at a glance:**

| Parameter | M2M | USER_FEDERATION |
|---|---|---|
| `auth_flow` | `"M2M"` | `"USER_FEDERATION"` |
| `scopes` | `[]` (empty) | OAuth provider scopes (e.g., `calendar.readonly`) |
| `on_auth_url` | Not needed | **Required** — receives auth URL when user consent needed |
| `callback_url` | Not needed | **Required** — where browser redirects after consent |
| Function pattern | Top-level function | Typically nested inside a `@tool` |
| `access_token` injection | Always injected (no user interaction) | Injected after first consent; empty on very first call |

### 5.2 One-Time Setup Comparison

Both flows require creating an **OAuth2 Credential Provider** and a **Workload Identity**. USER_FEDERATION additionally requires a **callback server** and registering its URL.

**M2M Setup:**

```
Step 1: Create a Cognito User Pool + M2M app client
        └─ AllowedOAuthFlows: ["client_credentials"]
        └─ GenerateSecret: True

Step 2: Register as a Credential Provider with AgentCore Identity
        └─ create_oauth2_credential_provider(name="my-cognito-provider", ...)

Step 3: Create Workload Identity for your agent
        └─ create_workload_identity(name="my-agent-workload")
```

```python
# Step 1: Cognito M2M client (via AWS SDK or console)
cognito.create_user_pool_client(
    UserPoolId=pool_id,
    ClientName="agent-m2m-client",
    GenerateSecret=True,
    AllowedOAuthFlows=["client_credentials"],
    AllowedOAuthScopes=[scope_names],
    AllowedOAuthFlowsUserPoolClient=True,
)

# Step 2: Register with AgentCore Identity
control = boto3.client("bedrock-agentcore-control")
control.create_oauth2_credential_provider(
    name="my-cognito-provider",
    credentialProviderVendor="CustomOauth2",
    oauth2ProviderConfigInput={
        "customOauth2ProviderConfig": {
            "clientId": cognito_client_id,
            "clientSecret": cognito_client_secret,
            "oauthDiscovery": {
                "authorizationServerMetadata": {
                    "issuer": issuer_url,
                    "authorizationEndpoint": auth_url,
                    "tokenEndpoint": token_url,
                    "responseTypes": ["code", "token"],
                }
            },
        }
    },
)

# Step 3: Workload Identity
control.create_workload_identity(name="my-agent-workload")
```

**USER_FEDERATION Setup:**

```
Step 1: Register your app with the OAuth provider (Google Console, GitHub Settings, etc.)
        └─ Obtain a client ID and client secret

Step 2: Register as a Credential Provider with AgentCore Identity
        └─ create_oauth2_credential_provider(name="google-cal-provider", ...)

Step 3: Create Workload Identity + register callback URL
        └─ create_workload_identity(name="my-agent-workload")
        └─ update_workload_identity(allowedResourceOauth2ReturnUrls=[callback_url])

Step 4: Deploy a Callback Server (see Section 5.5)
```

```python
# Step 1: Obtain client_id and client_secret from Google/GitHub developer console

# Step 2: Register with AgentCore Identity
control = boto3.client("bedrock-agentcore-control")
control.create_oauth2_credential_provider(
    name="google-cal-provider",
    credentialProviderVendor="GoogleOauth2",   # Or "CustomOauth2" for generic providers
    oauth2ProviderConfigInput={
        "googleOauth2ProviderConfig": {
            "clientId": google_client_id,
            "clientSecret": google_client_secret,
        }
    },
)

# Step 3: Workload Identity + register callback URL
control.create_workload_identity(name="my-agent-workload")
control.update_workload_identity(
    name="my-agent-workload",
    allowedResourceOauth2ReturnUrls=[
        "http://localhost:9090/oauth2/callback",       # Local dev
        # "https://your-domain.com/oauth2/callback",   # Production
    ],
)

# Step 4: Deploy callback server (see Section 5.5)
```

### 5.3 Runtime Auth Flow Comparison

#### M2M Flow (Automatic — No User Interaction)

```
Agent tool call
      │
      ▼
@requires_access_token (auth_flow="M2M")
      │
      ▼
AgentCore Identity: Use client credentials
to get access token from Cognito token endpoint
      │
      ▼
Token injected into function → Agent calls API
```

Every call follows the same path. No user interaction is ever needed. The decorator handles the client credentials exchange with the Cognito token endpoint automatically.

#### USER_FEDERATION Flow (First Call Requires User Consent)

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

**Step-by-step (USER_FEDERATION only — M2M skips all of this):**

1. **User makes request**: *"Check my calendar for today."*
2. **Agent detects no token**: `@requires_access_token` checks the token vault for this (agent identity, user identity, credential provider, scope) tuple — where the agent identity and user identity are both encoded in the WAT. No token → generates an OAuth authorization URL → calls `on_auth_url` callback so the webapp can show it to the user.
3. **User is redirected to OAuth provider**: Browser opens Google/GitHub consent screen.
4. **User grants consent**: Clicks "Allow" on the provider's permission prompt.
5. **OAuth provider redirects to callback**: Browser redirects to your callback server with a `session_id`.
6. **Callback server completes the flow**: Calls `complete_resource_token_auth()` — tells AgentCore Identity to exchange the authorization code for access + refresh tokens.
7. **Tokens stored in vault**: Per-user, encrypted. Each **(agent identity, user identity, credential provider, scope)** tuple gets its own token pair — the agent identity comes from the WAT.
8. **Agent retries with token**: `@requires_access_token` now finds the token in the vault and injects it as the `access_token` parameter.

#### USER_FEDERATION via Gateway (Auth Code Grant through MCP Elicitation)

When the agent uses tools through **AgentCore Gateway** (instead of calling the external API directly), the Gateway itself manages the outbound OAuth flow. The agent no longer uses `@requires_access_token` — instead, the Gateway handles credential acquisition, storage, and injection transparently.

```
                                         ┌─────────────────┐
                                         │  OAuth Provider  │
                                         │ (e.g., LinkedIn) │
                                         └────┬───────▲─────┘
                                       5. User│       │4. Redirect
                                       grants │       │   to provider
                                       consent│       │
                                         ┌────▼───────┴─────┐
                                         │  User's Browser   │
                                         └────┬───────▲─────┘
                                       6. Provider    │3. MCP Client opens
                                       redirects     │   elicitation URL
                                       to callback   │   in user's browser
                                         ┌────▼───────┴─────┐
  1. "Get my        ┌──────────┐         │  Callback Server  │
     LinkedIn ────▶│  Webapp   │         │  (/oauth2/callback)│
     profile"      │(Streamlit)│         └────────┬──────────┘
                   └────┬──────┘                  │7. complete_resource_token_auth()
                        │                         ▼
                        │ POST           ┌─────────────────────┐
                        │/invocations    │ AgentCore Identity   │
                        ▼                │ Service (Token Vault)│
                   ┌──────────────┐      └────────┬────────────┘
                   │ AgentCore    │               │8. Tokens stored;
                   │ Runtime      │               │   vault key:
                   │ (your agent) │               │   (cred_provider,
                   └──────┬───────┘               │    user, scope)
                          │                       │
                          │ 2. MCP tools/call      │
                          │    (Bearer token from  │
                          │     inbound auth)      │
                          ▼                       │
                   ┌──────────────────┐           │
                   │ AgentCore        │◄──────────┘
                   │ Gateway          │  9. Gateway uses IAM role
                   │ (MCP Server)     │     + user identity from JWT
                   └──────┬───────────┘     to retrieve token from vault
                          │
                          │ 10. Gateway injects token
                          │     into outbound API call
                          ▼
                   ┌──────────────┐
                   │ LinkedIn     │
                   │ API          │
                   └──────────────┘
```

**Step-by-step (USER_FEDERATION via Gateway):**

1. **User makes request**: *"Get my LinkedIn profile."*
2. **Agent sends MCP `tools/call`**: The agent calls the Gateway MCP endpoint with an inbound JWT as the `Authorization: Bearer` header. This JWT is validated by the Gateway's configured authorizer (e.g., Cognito CUSTOM_JWT). The Gateway extracts the caller's identity from the JWT claims. The agent does **not** use `@requires_access_token` — the Gateway handles outbound auth entirely.
3. **Gateway detects no token**: The Gateway checks the AgentCore Identity token vault for this (credential provider, user identity, scope) combination. No token found → Gateway returns an **MCP Elicitation Response** containing an OAuth authorization URL. The MCP client (or webapp) opens this URL in the user's browser.
4. **User is redirected to OAuth provider**: Browser navigates to the provider's (e.g., LinkedIn) consent screen.
5. **User grants consent**: Clicks "Allow" on the provider's permission prompt.
6. **OAuth provider redirects to callback**: Browser redirects to the developer's callback server with a `session_id`.
7. **Callback server completes the flow**: Calls `complete_resource_token_auth()` — tells AgentCore Identity to exchange the authorization code for access + refresh tokens.
8. **Tokens stored in vault**: Per-user, encrypted. The vault key is **(credential provider, user identity, scope)** — not per-agent or per-gateway. **AgentCore Identity** stores and manages both the access token and the refresh token. See [Token Sharing Across Agents](#token-sharing-across-agents) below.
9. **Agent retries the MCP `tools/call`**: The Gateway uses its own **IAM role** to call AgentCore Identity APIs, presenting the user identity extracted from the inbound JWT. Identity returns the stored access token. See [How Does the Gateway Identify the User?](#how-does-the-gateway-identify-the-user) below.
10. **Gateway injects token into outbound API call**: The Gateway translates the MCP request into an HTTP API call to LinkedIn, injecting the user's access token as a `Bearer` header. The agent never sees or handles the external access token.

**Key differences from the direct (non-Gateway) flow:**

| Aspect | Direct (agent calls API) | Via Gateway |
|---|---|---|
| **Who acquires tokens?** | Agent code via `@requires_access_token` decorator triggers the OAuth flow | **Gateway** triggers the OAuth flow; returns MCP Elicitation Response to signal consent is needed |
| **Who stores tokens?** | **AgentCore Identity** Token Vault — keyed by (agent identity, user identity, credential provider, scope). Each agent gets its own token pair per user. | **AgentCore Identity** Token Vault — keyed by (credential provider, user identity, scope). No agent identity in the key — tokens are shared across agents using the same Gateway target (see below) |
| **Who refreshes tokens?** | **AgentCore Identity** via the decorator (automatic, silent) | **AgentCore Identity** on behalf of the Gateway (automatic, silent) |
| **How is the user identified?** | Runtime exchanges inbound JWT → **WAT** (encodes agent + user identity); decorator reads WAT from SDK context | Gateway validates inbound JWT via its authorizer, extracts user identity from JWT claims; uses its **IAM role** to call Identity APIs |
| **Does the agent see the external token?** | Yes — injected as `access_token` parameter | **No** — Gateway injects it into the outbound call; agent only sees MCP responses |
| **Callback server needed?** | Yes — developer-deployed | Yes — developer-deployed (same pattern) |
| **Consent mechanism** | `on_auth_url` callback in decorator | MCP Elicitation Response (URL mode, per MCP 2025-11-25 spec) |
| **Credential provider config** | In `@requires_access_token` decorator parameters | In Gateway Target configuration (`credentialProviderConfigurations`) |
| **Where is provider registered?** | AgentCore Identity (credential provider) + Workload Identity | AgentCore Identity (credential provider) + Gateway Target |

> **Why use Gateway for USER_FEDERATION?** The Gateway approach decouples your agent code from OAuth complexity entirely. The agent only speaks MCP — it never handles tokens, refresh logic, or OAuth URLs. The Gateway and Identity service collaborate to handle the full token lifecycle, while the MCP Elicitation protocol (introduced in MCP spec 2025-11-25) provides a standardized way to signal that user consent is needed.

#### Token Sharing Across Agents

In the **direct** (non-Gateway) flow, tokens in the Identity vault are keyed by **(agent identity, user identity, credential provider, scope)**. Each agent has its own WAT, so Agent A and Agent B calling the same Google Calendar API for the same user get *separate* token pairs — the user must consent once per agent.

In the **Gateway** flow, the vault key is **(credential provider, user identity, scope)**. The credential provider is configured on the Gateway Target, not on the individual agent. This means:

- **Yes — different agents that call the same Gateway target share the same access/refresh tokens for a given user.** If Agent A triggers user consent for LinkedIn via the Gateway, Agent B calling the same Gateway target for the same user will find the token already in the vault — no second consent needed.
- The sharing boundary is the **Gateway Target** + **credential provider configuration**. If you create two Gateway targets pointing to the same LinkedIn API but with different credential providers, they would NOT share tokens.

```
Agent A ──MCP──▶ Gateway Target "LinkedIn" ──┐
                  (credential provider: X)     ├──▶ Identity Vault key:
Agent B ──MCP──▶ Gateway Target "LinkedIn" ──┘     (provider X, user Alice, scope "profile")
                  (same credential provider: X)
                                               → Same token pair. One consent.
```

This is intentional: the Gateway acts as a shared infrastructure layer. Users consent to the *Gateway target's* access (a specific provider + scope combination), not to each individual agent that routes through it.

#### What If You Need Per-Agent User Consent?

**Scenario**: Alice wants Agent A to access her Google Calendar, but does **not** want Agent B to have that access — even though both agents route through the same Gateway.

The Gateway token-sharing model does **not** support this at the token vault level. Once Alice consents via the Gateway's LinkedIn target, any agent that can reach that Gateway target inherits her consent. So where does the access control live?

**The control point is the inbound authorization layer, not the token vault.**

The Gateway's `CUSTOM_JWT` authorizer + Cognito scopes determine which agents can call which Gateway targets *before* the token vault is ever consulted:

```
                                     ┌──────────────────────────────┐
Agent A (Cognito client A)           │  Gateway                     │
  JWT scopes: ["calendar-target"]    │                              │
  ─────────────────────────────────▶ │  1. Validate JWT ✅          │
                                     │  2. Check scope matches      │
                                     │     target ✅                │
                                     │  3. Retrieve/inject token    │
                                     │                              │
Agent B (Cognito client B)           │                              │
  JWT scopes: [] (no calendar scope) │                              │
  ─────────────────────────────────▶ │  1. Validate JWT ✅          │
                                     │  2. Check scope matches      │
                                     │     target ❌ → 403 Forbidden│
                                     └──────────────────────────────┘
```

The [Gateway tutorial](../01-tutorials/02-AgentCore-gateway/13-outbound-auth-code-grant/01-outbound-auth-code-grant-linkedin.ipynb) shows this pattern: each Gateway target gets a Cognito **resource server scope** (e.g., `agentcore-gateway-id/LinkedInAuthCode`). An agent's Cognito app client must be granted that scope to obtain a JWT the Gateway will accept for that target. The Gateway operator (developer/admin) controls which app clients get which scopes.

**But this is operator-level control, not end-user control.** The user doesn't get to choose "Agent A yes, Agent B no" at consent time — the operator decides which agents can reach which targets. If the operator grants both Agent A and Agent B the scope for the calendar target, both get access once Alice consents.

**When to use which model:**

| Requirement | Recommended approach |
|---|---|
| **Operator controls which agents access which APIs** (most common) | Gateway flow — use Cognito scopes to gate which agents can reach which targets. Token sharing is a feature (one user consent covers all authorized agents). |
| **Users must consent per-agent** (e.g., "I trust Agent A but not Agent B with my calendar") | **Direct flow** (`@requires_access_token` + WAT). Vault key includes agent identity → separate tokens per agent → separate user consent per agent. |
| **Mixed**: some APIs are shared infrastructure, others need per-agent consent | Use Gateway for shared APIs (e.g., company directory lookup). Use direct flow for sensitive per-user APIs (e.g., personal calendar, email). |

**Mitigation strategies if you must use Gateway but want isolation:**

1. **Separate Gateway targets with separate credential providers**: Create `Calendar-for-AgentA` and `Calendar-for-AgentB` targets, each with its own credential provider pointing to Google Calendar. Different credential providers → different vault keys → separate tokens → separate user consent. But this duplicates configuration.
2. **Separate Gateways entirely**: If Agent A and Agent B serve different trust domains, deploy separate Gateways. Each Gateway has its own IAM role and targets.
3. **Cognito scope-based exclusion**: Don't grant Agent B's Cognito app client the scope for the calendar target. Agent B physically cannot call that target — the Gateway rejects the JWT at the inbound auth layer.

> **Rule of thumb**: If the user's relationship with each agent matters for authorization (e.g., personal assistants with varying trust levels), prefer the direct flow. If the Gateway is shared infrastructure controlled by a single operator (e.g., a company's API gateway), the Gateway token-sharing model is appropriate.

#### How Does the Gateway Identify the User?

The Gateway does **not** use the Workload Access Token (WAT) mechanism that the Runtime + SDK use. The two paths differ:

| | Runtime + SDK (direct flow) | Gateway flow |
|---|---|---|
| **Inbound auth** | Runtime validates inbound JWT (signature, issuer, audience) | Gateway validates inbound JWT via its configured authorizer (`CUSTOM_JWT`) |
| **User identity extraction** | Runtime exchanges JWT → **WAT** via AgentCore Identity token exchange API. WAT encodes both agent identity and user identity. | Gateway extracts user identity directly from the **validated JWT claims** (e.g., `sub`, `client_id`). No WAT is created. |
| **How tokens are retrieved** | `@requires_access_token` reads WAT from SDK runtime context → presents WAT to Identity API → Identity resolves (agent + user) and returns the resource token | Gateway uses its **IAM role** to call Identity APIs, passing the user identity from the JWT. Identity resolves (credential provider + user) and returns the resource token. |
| **Authentication to Identity service** | Agent's workload identity (embedded in WAT) | Gateway's **IAM role** (the `roleArn` assigned when creating the Gateway) |

In the Gateway flow, the chain of trust is:

```
Inbound JWT ──validated by──▶ Gateway Authorizer ──extracts──▶ User Identity
                                                                    │
Gateway IAM Role ──authenticates──▶ AgentCore Identity API ◄───────┘
                                         │                  (user identity
                                         ▼                   as parameter)
                                   Token Vault lookup:
                                   (credential_provider, user, scope)
```

The Gateway's IAM role is the credential that Identity trusts. The user identity from the validated JWT is the lookup key. Together, they replace the WAT's role of proving "this agent is authorized to act on behalf of this user." The difference is that the Gateway is a first-party AWS service with its own IAM role, so it authenticates to Identity via IAM rather than via a WAT.

### 5.4 Credentials, Tokens & Lifecycle

This table compares every credential and token involved — where it comes from, where it's stored, who caches it, and who refreshes it.

| Credential / Token | M2M | USER_FEDERATION (3LO) |
|---|---|---|
| **OAuth Client ID** | Cognito User Pool client ID | Google/GitHub/etc. app client ID |
| **OAuth Client Secret** | Cognito client secret | Google/GitHub/etc. app client secret |
| ↳ *Created by* | Developer (Cognito console or API) | Developer (provider's developer console) |
| ↳ *Stored in* | AgentCore Identity (encrypted, via credential provider config) | AgentCore Identity (encrypted, via credential provider config) |
| ↳ *Lifetime* | Permanent (until rotated manually) | Permanent (until rotated manually) |
| | | |
| **Access Token** | Cognito JWT — proves the agent's service identity | Google/GitHub API token — proves user's delegated permission |
| ↳ *Created by* | Cognito token endpoint (client credentials grant) | OAuth provider token endpoint (auth code exchange, or refresh) |
| ↳ *Stored / cached in* | AgentCore Identity (in-process SDK cache) | AgentCore Identity Token Vault (encrypted, per-user) |
| ↳ *Lifetime* | ~1 hour (Cognito default) | ~1 hour (varies by provider) |
| ↳ *Refreshed by* | **AgentCore Identity** — re-fetches using client credentials | **AgentCore Identity** — uses stored refresh token silently |
| ↳ *User action needed to refresh?* | No — fully automatic | No — fully automatic (as long as refresh token is valid) |
| | | |
| **Refresh Token** | ❌ Not applicable (client credentials don't issue refresh tokens) | Long-lived token for re-obtaining access tokens without user consent |
| ↳ *Stored in* | — | AgentCore Identity Token Vault (encrypted, per-user) |
| ↳ *Lifetime* | — | ~90 days (varies by provider; requires `offline_access` scope) |
| ↳ *When it expires* | — | User must re-consent in browser |
| | | |
| **Authorization Code** | ❌ Not applicable | Temporary code from OAuth provider after user grants consent |
| ↳ *Stored in* | — | Never stored — used once, immediately exchanged for tokens |
| ↳ *Lifetime* | — | ~10 minutes (single use) |
| | | |
| **User Identity Token** | ❌ Not applicable (no user involved) | User's Cognito JWT — identifies *which user* the token belongs to |
| ↳ *Stored in* | — | Client-side (browser/notebook) |
| ↳ *Lifetime* | — | ~1 hour (Cognito) |

### 5.5 Callback Server (USER_FEDERATION Only)

**You must write and deploy the callback server yourself** — AgentCore Identity does not provide a built-in callback endpoint. It's a lightweight HTTP server (~20–30 lines of route logic) that:

1. Listens for the OAuth provider's redirect: `GET /oauth2/callback?session_id=...`
2. Calls `identity_client.complete_resource_token_auth()` to finalize the token exchange
3. Shows the user a success page

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from bedrock_agentcore.services.identity import IdentityClient

app = FastAPI()
identity_client = IdentityClient(region="us-east-1")

@app.get("/oauth2/callback")
async def handle_callback(session_id: str):
    # user_token_identifier: the user's Cognito JWT or unique ID,
    # stored by your callback server when the OAuth flow began
    identity_client.complete_resource_token_auth(
        session_uri=session_id,
        user_identifier=user_token_identifier
    )
    return HTMLResponse("<h1>Authorization successful! You can close this tab.</h1>")
```

**Sample implementations in this repo:**

| Deployment Model | Sample | When to Use |
|---|---|---|
| **Local FastAPI** | `01-tutorials/03-AgentCore-identity/05-Outbound_Auth_3lo/oauth2_callback_server.py` | Development, Streamlit, SageMaker |
| **AWS Lambda** | `01-tutorials/02-AgentCore-gateway/04-integration/03-ide-gateway-tool/lambda/callback_lambda.py` | Production, behind API Gateway |

> **Why isn't the callback built into AgentCore?** The callback URL must be reachable by the user's browser *and* registered with the OAuth provider — it's part of *your* webapp's domain, not the AgentCore control plane. Different apps have different domains, ports, and deployment models.

### 5.6 Token Refresh: Who Does What?

**You never write refresh logic.** The `@requires_access_token` decorator handles all token lifecycle management for both flows.

```
@requires_access_token called
        │
        ▼
Check token vault for (user, resource, scope)
        │
   ┌────┴──────────────────────┐
   │              │              │
No token      Token valid     Token expired
   │              │              │
   ▼              ▼              ▼
(M2M)          Inject         (M2M) Re-fetch with
Fetch with     immediately    client credentials
client creds                  (3LO) Use refresh token
   │                           to get new access token
   ▼                           silently
(3LO)                              │
Trigger OAuth                      ▼
flow → user                   Inject refreshed token
must consent
```

| Scenario | M2M | USER_FEDERATION (3LO) |
|---|---|---|
| **No token yet** | Auto-fetches with client credentials — instant, no user action | Triggers OAuth flow — user must consent in browser (one-time) |
| **Token valid** | Injected immediately | Injected immediately |
| **Access token expired** | Re-fetches with client credentials — instant, automatic | Uses stored refresh token silently — no user action |
| **Refresh token expired** | N/A (M2M doesn't use refresh tokens) | User must re-consent in browser (typically every 90+ days) |

> **Tip:** To get refresh tokens from the OAuth provider, request the `offline_access` scope (or the provider's equivalent). Without it, you'll only get short-lived access tokens and users will need to re-authorize on every expiry.

### 5.7 Summary of Responsibilities

| Responsibility | M2M | USER_FEDERATION |
|---|---|---|
| Writing agent tool with `@requires_access_token` | **Developer** | **Developer** |
| Registering the OAuth provider (credential provider) | **Developer** (one-time) | **Developer** (one-time) |
| Creating a Workload Identity | **Developer** (one-time) | **Developer** (one-time) |
| Writing and deploying the callback server | Not needed | **Developer** |
| Registering callback URL in workload identity | Not needed | **Developer** (one-time) |
| Storing tokens securely | **AgentCore Identity** | **AgentCore Identity** (per-user vault) |
| Retrieving tokens on subsequent requests | **AgentCore Identity** (via decorator) | **AgentCore Identity** (via decorator) |
| Refreshing expired access tokens | **AgentCore Identity** (re-fetches with client creds) | **AgentCore Identity** (uses refresh token silently) |
| Re-prompting user when refresh token expires | N/A | **AgentCore Identity** (via `on_auth_url` callback) |

### 5.8 Inbound Authentication (Clients → Your Agent)

Separately from outbound tool auth, your agent's `/invocations` endpoint can be authenticated via:

**AWS SigV4 (IAM Authentication):**

```python
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
SigV4Auth(credentials, "bedrock-agentcore", region).add_auth(aws_request)
```

**JWT Token Extraction in Agent Runtime:**

```python
def _get_bearer_token(context) -> Optional[str]:
    auth = (getattr(context, "request_headers", None) or {}).get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None
```

### Repository Examples

| Example | Directory | Auth Flow |
|---|---|---|
| Customer Support (VPC) | `02-use-cases/customer-support-assistant-vpc/` | M2M for Gateway |
| Device Management | `02-use-cases/device-management-agent/` | M2M for Gateway |
| A2A Incident Response | `02-use-cases/A2A-multi-agent-incident-response/` | M2M for agent-to-agent |
| AWS Operations Agent | `02-use-cases/AWS-operations-agent/` | M2M for Gateway |
| Google Calendar 3LO | `01-tutorials/03-AgentCore-identity/05-Outbound_Auth_3lo/` | USER_FEDERATION |
| GitHub OAuth | `01-tutorials/03-AgentCore-identity/06-Outbound_Auth_Github/` | USER_FEDERATION |
| LinkedIn Auth Code | `01-tutorials/02-AgentCore-gateway/13-outbound-auth-code-grant/` | USER_FEDERATION via Gateway |
| IDE + Confluence | `01-tutorials/02-AgentCore-gateway/04-integration/03-ide-gateway-tool/` | USER_FEDERATION |

### Blueprint Examples

| Blueprint | Directory | Description |
|---|---|---|
| Travel Concierge | `05-blueprints/travel-concierge-agent/` | Multi-agent orchestration with Gateway |
| Customer Service | `05-blueprints/end-to-end-customer-service-agent/` | Streamlit + FastAPI + AgentCore |
| Customer Support | `05-blueprints/customer-support-agent-with-agentcore/` | Memory integration |
| Shopping Concierge | `05-blueprints/shopping-concierge-agent/` | Shopping/cart tools |

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
- **Agent ↔ Tools**: MCP via AgentCore Gateway, authenticated via M2M or USER_FEDERATION (see [Section 5](#5-authentication))

## 7. End-to-End Auth Flow: From User Login to Tool Execution

This section traces the complete authentication flow through actual code, answering a common question: *"The webapp authenticates the user, gets a JWT, extracts the user_id… but how does `@requires_access_token` know which user's token to fetch if it doesn't take a `user_id` parameter?"*

### 7.1 The Two Channels for User Identity

User identity flows through the system via **two separate channels**:

| Channel | What it carries | How it's passed | Used by |
|---------|----------------|-----------------|---------|
| **Explicit** | `actor_id` (e.g., `cognito:username`) | In the HTTP request payload, threaded through function parameters | Memory (per-user conversation history), application logic |
| **Implicit** | User's JWT → **Workload Access Token** | In the `Authorization` header → token exchange by Runtime → stored in SDK runtime context | `@requires_access_token` decorator (token vault lookups) |

The `@requires_access_token` decorator uses the **implicit** channel. It never needs a `user_id` parameter because the AgentCore Runtime has already exchanged the user's inbound JWT for a **Workload Access Token (WAT)** that encodes both the agent's identity and the user's identity. The SDK stores this WAT in an internal runtime context, and the decorator reads it automatically.

### 7.2 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: USER AUTHENTICATION (Webapp ↔ Cognito)                           │
│                                                                             │
│  ┌──────────┐     1. Redirect to Cognito      ┌──────────────────┐         │
│  │  User's   │────── login page (PKCE) ──────▶│  Amazon Cognito   │         │
│  │  Browser  │                                 │  (OAuth2/OIDC)    │         │
│  │           │◀──── 2. Authorization code ─────│                   │         │
│  └─────┬─────┘                                 └────────┬──────────┘         │
│        │                                                │                    │
│        │  3. Code + code_verifier                       │                    │
│        ▼                                                │                    │
│  ┌──────────────┐   4. Exchange code ──────────────────▶│                    │
│  │  Webapp       │      for tokens                       │                    │
│  │  (Streamlit)  │◀──── 5. id_token + access_token ─────┘                    │
│  │               │                                                           │
│  │  auth.py:     │   6. Decode id_token (no sig verify)                      │
│  │  get_user_    │      → user_claims = {                                    │
│  │  claims()     │          "cognito:username": "alice",                      │
│  │               │          "email": "alice@acme.com", ...                    │
│  │               │        }                                                  │
│  └───────┬───────┘                                                           │
│          │                                                                   │
└──────────┼───────────────────────────────────────────────────────────────────┘
           │
           │  PHASE 2: AGENT INVOCATION (Webapp → AgentCore Runtime)
           │
           │  chat.py builds the HTTP request:
           │
           │  POST /runtimes/{agent_arn}/invocations
           │  Headers:
           │    Authorization: Bearer <cognito_access_token>  ← IMPLICIT channel
           │    X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: <uuid>
           │  Body:
           │    {"prompt": "Show my calendar",
           │     "actor_id": "alice"}                         ← EXPLICIT channel
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: RUNTIME PROCESSES THE REQUEST                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────┐                           │
│  │  AgentCore Runtime                            │                           │
│  │                                               │                           │
│  │  7. Validates Bearer JWT (signature, expiry,  │                           │
│  │     issuer, audience, scopes)                 │                           │
│  │                                               │                           │
│  │  8. Token Exchange: JWT → Workload Access     │                           │
│  │     Token (WAT) via AgentCore Identity API    │──────┐                    │
│  │     WAT encodes: agent identity + user        │      │                    │
│  │     identity (from the validated JWT)         │      │                    │
│  │                                               │      ▼                    │
│  │  9. Stores WAT in SDK runtime context         │  ┌────────────────┐      │
│  │     (Python contextvars / thread-local)       │  │ AgentCore      │      │
│  │                                               │  │ Identity       │      │
│  │  10. Calls @app.entrypoint:                   │  │ Service        │      │
│  │      invoke(payload, context)                 │  │                │      │
│  │        → payload["actor_id"] = "alice"        │  │ Validates JWT  │      │
│  │        → context.session_id = <from header>   │  │ Issues WAT     │      │
│  └──────────────────┬───────────────────────────┘  └────────────────┘      │
│                     │                                                       │
└─────────────────────┼───────────────────────────────────────────────────────┘
                      │
                      │  PHASE 4: AGENT EXECUTION
                      │
                      │  main.py → agent_task.py:
                      │
                      │  11. get_gateway_access_token()
                      │      → @requires_access_token(auth_flow="M2M")
                      │      → reads WAT from implicit context
                      │      → AgentCore Identity returns service token
                      │
                      │  12. MemoryHook(actor_id="alice", session_id=...)
                      │      → uses EXPLICIT actor_id for per-user memory
                      │
                      │  13. Agent processes user message with LLM
                      │      → LLM decides to call a tool
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: TOOL CALLS @requires_access_token                                 │
│                                                                             │
│  google.py — get_google_access_token():                                     │
│                                                                             │
│  @requires_access_token(                                                    │
│      provider_name="acme-google-calendar",                                  │
│      scopes=["...calendar"],                                                │
│      auth_flow="USER_FEDERATION",  ← 3-Legged OAuth                        │
│      on_auth_url=on_auth_url,                                               │
│  )                                                                          │
│  def get_google_access_token(access_token: str):                            │
│      return access_token                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │  Inside the decorator (SDK internals):               │                    │
│  │                                                      │                    │
│  │  14. Read WAT from runtime context                   │                    │
│  │      (WAT contains agent identity + user identity)   │                    │
│  │                                                      │                    │
│  │  15. Call AgentCore Identity:                         │                    │
│  │      "Give me a Google Calendar token for             │                    │
│  │       (this agent, this user, these scopes)"          │                    │
│  │                                                      │                    │
│  │  16a. Token found in vault?                          │                    │
│  │       → Return it → injected as access_token param   │                    │
│  │                                                      │                    │
│  │  16b. No token? (first time for this user+resource)  │                    │
│  │       → AgentCore Identity returns an auth URL        │                    │
│  │       → Decorator calls on_auth_url(url)              │                    │
│  │       → Webapp shows URL to user                      │                    │
│  │       → User consents in browser                      │                    │
│  │       → Callback server calls                         │                    │
│  │         complete_resource_token_auth()                 │                    │
│  │       → Token stored in vault per (agent, user,       │                    │
│  │         provider, scopes)                              │                    │
│  │       → On retry, token is found and injected          │                    │
│  └──────────────────────────────────────────────────────┘                    │
│                                                                             │
│  17. Tool uses the injected access_token to call                            │
│      Google Calendar API → returns Alice's events                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Why `@requires_access_token` Doesn't Need a `user_id` Parameter

Looking at the decorator signature:

```python
@requires_access_token(
    provider_name: str,           # Which OAuth provider (e.g., "google-cal-provider")
    scopes: List[str],            # OAuth scopes needed
    auth_flow: "M2M" | "USER_FEDERATION",
    on_auth_url: Callable,        # Callback when user consent is needed
    into: str = "access_token",   # Parameter name to inject token into
    force_authentication: bool,
    callback_url: str,            # OAuth redirect URI
)
```

There is no `user_id` parameter. The user's identity is already embedded in the **Workload Access Token (WAT)** that the AgentCore Runtime created during Phase 3 (step 8). The SDK stores this WAT in an internal runtime context that the decorator reads automatically. This is by design:

- **Separation of concerns**: Authentication (proving identity) is handled at the Runtime level during request processing. Tool code doesn't need to thread user IDs through every function call.
- **Security**: The user identity comes from a cryptographically validated JWT → WAT chain, not from an easily-spoofable function parameter.
- **Simplicity**: Tool developers only need to declare *what resource* they need (`provider_name`, `scopes`) and *what type of access* (`auth_flow`). The decorator handles the rest.

### 7.4 The `actor_id` vs. WAT Distinction

| | `actor_id` (Explicit) | Workload Access Token (Implicit) |
|---|---|---|
| **Source** | Extracted from JWT claims in webapp code | Created by Runtime via token exchange |
| **Passed via** | HTTP payload → function parameters | SDK runtime context (automatic) |
| **Contains** | A string identifier (e.g., `"alice"`) | Cryptographic proof of agent + user identity |
| **Used by** | Memory (per-user history), app logic | `@requires_access_token` (token vault lookups) |
| **Security model** | Application-level (trusts the webapp) | Zero-trust (validated by AgentCore Identity) |

Both carry the user's identity, but through different mechanisms for different purposes. The `actor_id` is a convenience for application-level features like memory namespacing. The WAT is a security primitive that enables the decorator to request per-user tokens without the developer explicitly passing user identity.

### 7.5 Code References

The complete flow can be traced through the [customer-support-assistant](../02-use-cases/customer-support-assistant/) sample:

| Step | File | Key Code |
|------|------|----------|
| User login (PKCE) | `app_modules/auth.py` | `AuthManager.handle_oauth_callback()` — exchanges auth code for JWT tokens |
| Extract user claims | `app_modules/auth.py` | `get_user_claims()` — decodes `id_token` to get `cognito:username` |
| Build invocation request | `app_modules/chat.py` | `invoke_endpoint()` — sends Bearer token + `actor_id` in payload |
| Runtime entrypoint | `main.py` | `invoke(payload, context)` — extracts `actor_id` and `session_id` |
| M2M token for Gateway | `agent_config/access_token.py` | `@requires_access_token(auth_flow="M2M")` — gets service token |
| Memory with user scope | `agent_config/agent_task.py` | `MemoryHook(actor_id=actor_id)` — per-user conversation history |
| User-delegated Google token | `agent_config/tools/google.py` | `@requires_access_token(auth_flow="USER_FEDERATION")` — gets Alice's Google token |

### 7.6 FAQ: Who Sends the `session_id`? When Is It Created vs. Reused?

**The webapp creates and manages the session ID — not the browser directly and not AgentCore Runtime.**

In the customer-support-assistant sample, the Streamlit webapp generates a fresh UUID on first page load and stores it in Streamlit's session state:

```python
# app_modules/chat.py — _init_session_state()
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())   # ← New UUID per browser tab
```

Every subsequent message in the same browser tab reuses this value, which is sent as a header:

```python
# app_modules/chat.py — invoke_endpoint()
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,   # ← Same UUID throughout
}
```

| Event | What happens to `session_id` |
|-------|------------------------------|
| **User opens a new browser tab** | Webapp generates a fresh `uuid.uuid4()` → new session |
| **User sends another message in the same tab** | Webapp reuses `st.session_state["session_id"]` → same container |
| **Container times out** (15-min idle / 8-hr max) | Next request with the same `session_id` creates a new container — all in-memory state is lost, but AgentCore Memory persists |
| **User refreshes the page** (Streamlit) | `st.session_state` resets → new UUID → new session |

**Key point**: The `session_id` is opaque to AgentCore — the platform does no user-binding validation. Whoever sends the same session ID header is routed to the same container. It is the **webapp's responsibility** to generate unique, per-user session IDs (which the sample does via `uuid.uuid4()`).

For a deeper treatment of session lifecycle, see [session-lifecycle.md](./session-lifecycle.md).

### 7.7 FAQ: What If `actor_id` Is Missing or Wrong?

> **⚠️ SECURITY NOTE**: The `actor_id` is an application-level string with **no platform validation**. AgentCore does not verify that `actor_id` matches the authenticated user's JWT. If your webapp passes the wrong `actor_id`, the agent will read and write **another user's memory** — with no error or warning. Always extract `actor_id` from the validated JWT claims; never accept it from client/browser input.

#### If `actor_id` is missing from the request body

The agent entrypoint crashes with a `KeyError`:

```python
# main.py — invoke()
actor_id = payload["actor_id"]   # ← KeyError if key is absent
```

There is no graceful fallback — this is a hard crash. A production webapp should always extract `actor_id` from the validated JWT claims before calling `/invocations`.

#### If `actor_id` is wrong (e.g., `"bob"` instead of `"alice"`)

**Yes — the agent will retrieve and write to the wrong user's memory.** There is no platform-level validation that `actor_id` matches the authenticated user's JWT.

Memory is namespaced by `actor_id` via direct string interpolation:

```python
# agent_config/memory_hook_provider.py
namespace=f"support/user/{self.actor_id}/preferences"   # ← Whatever string was passed
namespace=f"support/user/{self.actor_id}/facts"
```

And conversation history is scoped by the `(memory_id, actor_id, session_id)` tuple:

```python
# agent_config/memory_hook_provider.py — on_agent_initialized()
recent_turns = self.memory_client.get_last_k_turns(
    memory_id=self.memory_id,
    actor_id=self.actor_id,      # ← Any string — no validation
    session_id=self.session_id,
    k=5,
)
```

If the webapp has a bug and sends `"actor_id": "bob"` with Alice's valid JWT:
1. ✅ Alice's JWT is validated (she's authorized to call the agent)
2. ❌ The agent loads **Bob's** conversation history, preferences, and facts
3. ❌ Alice's conversation is saved **under Bob's** namespace
4. → **Cross-user memory contamination and disclosure**

#### Why the current sample is safe *in practice*

The webapp extracts `actor_id` directly from the validated JWT — never from user input:

```python
# app_modules/chat.py
payload = json.dumps({
    "prompt": prompt,
    "actor_id": user_claims.get("cognito:username")   # ← From decoded id_token
})
```

A bug would have to be introduced in this webapp code path to send the wrong value.

#### Why `@requires_access_token` is NOT affected

The `@requires_access_token` decorator uses the **implicit** identity channel (Workload Access Token), not the explicit `actor_id`. Even if `actor_id` is wrong, the WAT still carries the real user's cryptographically validated identity. So tool access (e.g., Google Calendar) always resolves to the correct user — only **Memory** is affected by a wrong `actor_id`.

| Concern | `actor_id` (Explicit) | WAT (Implicit) |
|---------|----------------------|----------------|
| Can a webapp bug cause cross-user access? | ⚠️ **Yes** — Memory reads/writes use the string as-is | ✅ **No** — cryptographically bound to the validated JWT |
| Platform validates it? | ❌ No | ✅ Yes (JWT → token exchange → WAT) |
| Mitigation | Webapp must extract from JWT; never accept from client input | Built-in — no developer action needed |

## 8. Streaming & Async Responses

Agents support streaming by yielding events from the entrypoint:

```python
@app.entrypoint
async def invoke(payload, context):
    stream = agent.stream_async(payload.get("prompt"))
    async for event in stream:
        if "data" in event and isinstance(event["data"], str):
            yield event["data"]
```

## 9. Key Takeaway

| Layer | Protocol | Standard? |
|---|---|---|
| User ↔ Webapp | Your choice (HTTP, WebSocket) | Your choice |
| Webapp ↔ Agent | AWS HTTP REST API (`/invocations`) | AWS-specific |
| Agent ↔ Agent | A2A Protocol | Open standard |
| Agent ↔ Tools | MCP via Gateway | Open standard |
| Auth (service-to-service) | OAuth2 Client Credentials (Cognito) or SigV4 | Industry standards |
| Auth (user-delegated) | OAuth2 Authorization Code (3LO) via AgentCore Identity | Industry standard |

The user-facing protocol is **HTTP REST with JSON payloads** — the exact schema is flexible and defined by your `@app.entrypoint` handler. It is **not** OpenAI-compatible or any other third-party standard out of the box, but it **does** support the open A2A standard for agent-to-agent communication. For tool integration, it uses MCP as the standard protocol via the Gateway.
