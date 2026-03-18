# Session Lifecycle in Amazon Bedrock AgentCore

This document analyzes how sessions work in AgentCore — their lifecycle, scoping, user binding, and behavior across agent versions and deployments. It references the samples in this repository and AWS documentation.

---

## Table of Contents

1. [Two Types of Sessions](#1-two-types-of-sessions)
2. [Runtime Session Lifecycle](#2-runtime-session-lifecycle)
3. [How Sessions Are Scoped](#3-how-sessions-are-scoped)
4. [Does a Session Belong to One User?](#4-does-a-session-belong-to-one-user)
5. [Sessions Across Agent Versions and Deployments](#5-sessions-across-agent-versions-and-deployments)
6. [Session and AgentCore Memory](#6-session-and-agentcore-memory)
7. [Code Examples from This Repository](#7-code-examples-from-this-repository)
8. [Summary](#8-summary)

---

## 1. Two Types of Sessions

AgentCore has two distinct session concepts that are often conflated:

| Session Type | What It Is | How It's Created | Default Lifetime |
|---|---|---|---|
| **Runtime Session** | An isolated microVM/container hosting your agent code | Automatically by AgentCore when a request arrives with a `session_id` via the `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header | 15-min idle / 8-hr max |
| **Tool Session** | An isolated sandbox for Code Interpreter or Browser Tool | Explicitly via `start_code_interpreter_session()` or `start_browser_session()` | Configurable via `sessionTimeoutSeconds` (default ~15 min) |

The **Runtime Session** is the primary concept and the focus of this document.

### Tool Sessions (Code Interpreter)

Tool sessions are separate from and managed independently of runtime sessions. An agent can create and destroy multiple tool sessions within a single runtime session:

```python
# From: xue-docs/GUIDE.md — Code Interpreter section
session_response = client.start_code_interpreter_session(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    name="mySession",
    sessionTimeoutSeconds=900,  # 15-minute timeout
)
session_id = session_response["sessionId"]  # e.g., "01K00Z3F8WZ9KBBW4QGRJCVBHH"
```

Key points about tool sessions:
- **`sessionId`** is a sandbox identifier, not an auth token ([GUIDE.md, line 591](./GUIDE.md))
- **`clearContext=False`** (default) preserves Python variables across calls within the same session
- Multiple agents can each have their own independent tool session
- Tool sessions auto-terminate after inactivity timeout

---

## 2. Runtime Session Lifecycle

From the tutorial at [`01-tutorials/01-AgentCore-runtime/03-advanced-concepts/02-understanding-runtime-context/`](../01-tutorials/01-AgentCore-runtime/03-advanced-concepts/02-understanding-runtime-context/README.md):

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐
│ Creation  │ ──→ │  Active  │ ──→ │   Idle   │ ──→ │ Termination │
└──────────┘     └──────────┘     └──────────┘     └─────────────┘
```

### Stage 1: Creation

A new session (microVM) is created on the **first invocation** with a unique `runtimeSessionId`. The session ID is sent by the caller via the HTTP header:

```
X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: <session_id>
```

### Stage 2: Active

The container processes requests. All in-memory state persists across invocations with the same session ID:
- Conversation history
- Application state (variables, objects)
- File system modifications
- Environment variables

### Stage 3: Idle

The session waits for the next invocation while preserving all context.

### Stage 4: Termination

The container is destroyed and its memory is sanitized. Termination occurs due to:
- **Inactivity timeout**: 15 minutes (configurable via `idleRuntimeSessionTimeout`)
- **Maximum lifetime**: 8 hours
- **Health check failures**
- **Redeployment** of a new agent version

### Isolation Guarantees

Each session runs in its own microVM with:
- **Dedicated resources**: Isolated CPU, memory, and filesystem
- **Security boundaries**: Complete separation between user sessions
- **Deterministic cleanup**: On termination, the microVM is destroyed and memory is sanitized — zero data contamination between sessions

---

## 3. How Sessions Are Scoped

A session is scoped along three dimensions:

### 3.1 Session ID (primary key)

The caller sends a session ID via the HTTP header. AgentCore routes all requests with the **same session ID to the same container**, providing session affinity.

The agent code receives the session ID via the `context` object:

```python
# From: xue-docs/agentcore-user-agent-protocol.md
@app.entrypoint
async def invoke(payload, context):
    session_id = getattr(context, "session_id", "default-session")
    prompt = payload.get("prompt", "")
    yield response_text
```

Context object properties:

| Property | Description |
|---|---|
| `session_id` | Unique conversation/session identifier |
| `request_headers` | HTTP headers dict |
| `custom_attributes` | Custom runtime attributes |

### 3.2 Agent Deployment

Sessions are scoped to a specific agent deployment endpoint:

```
POST https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{agent_arn}/invocations?qualifier=DEFAULT
```

A session ID used with Agent A is completely unrelated to the same session ID used with Agent B — they route to different containers.

### 3.3 Authentication / Identity (implicit, not enforced)

The session itself is **not inherently scoped to a user** at the platform level. The `session_id` is an opaque string — see [Section 4](#4-does-a-session-belong-to-one-user) for details.

---

## 4. Does a Session Belong to One User?

### At the Platform Level: No Enforcement

The `session_id` is an opaque identifier with no built-in user binding. Whoever sends the same session ID header gets routed to the same container. AgentCore does not validate that a session ID "belongs" to a particular caller.

### In Practice: Yes, by Convention

Every sample in this repository associates sessions with a single user. The typical pattern is:

1. **Extract user identity** from the JWT/OAuth token
2. **Generate a session ID** that is inherently per-user
3. **Pass both** `user_id`/`actor_id` and `session_id` into the agent state

The SRE Agent generates session IDs with a timestamp prefix:

```python
# From: 02-use-cases/SRE-agent/sre_agent/supervisor.py
def _get_session_from_env(mode: str) -> str:
    session_id = os.getenv("SESSION_ID")
    if session_id:
        return session_id
    else:
        return f"{mode}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
```

Memory APIs always use the tuple `(memory_id, actor_id, session_id)`, so memory is inherently scoped to both user AND session:

```python
# From: 02-use-cases/site-reliability-agent-workshop/lab_helpers/short_term_memory.py
self.memory_client.get_last_k_turns(
    memory_id=self.memory_id,
    actor_id=actor_id,      # user scoping
    session_id=session_id,  # session scoping
    k=self.max_context_turns,
)
```

### Security Implication

If two users share a session ID, they would share the same container state — this is a security concern the application developer must prevent by ensuring unique session IDs per user.

---

## 5. Sessions Across Agent Versions and Deployments

### Ephemeral Container State Does NOT Survive Redeployment

> **Sourcing note**: The claim below is inferred from the general session–microVM model, not directly stated by AWS. See the [Evidence Assessment](./session-design-comparison.md#appendix-evidence-assessment-for-tight-session-compute-coupling) in the comparison doc for a confidence breakdown.

When you deploy a new agent version, existing containers are terminated. Since a runtime session is backed by a container, the session dies with it. Key evidence:

- **Container destruction on redeployment**: Existing microVMs are torn down when a new version is deployed
- **Session affinity is container-bound**: The routing of `session_id → container` only works while that container is alive
- **No session migration API**: There is no API to transfer or migrate a session from one version to another

### Persistent State CAN Survive via AgentCore Memory

AgentCore Memory is a separate service that persists independently of runtime sessions. If you write state to Memory during a session, that state is retrievable by future sessions — even after agent version upgrades:

```python
# User-scoped memory (persists across ALL sessions AND versions):
"/sre/users/{actorId}/preferences"

# Session-scoped memory (persists externally, but keyed to a specific session):
"/sre/infrastructure/{actorId}/{sessionId}"

# Cross-session search (finds data from ALL past sessions):
memories = client.retrieve_memories(
    memory_id=memory_id,
    namespace="/sre/infrastructure/user123",  # No {sessionId} → matches all sessions
    query="database connection issues",
    top_k=5
)
```

### Session ID Resets

The SRE Agent resets the session ID after major operations (e.g., saving an investigation report), starting a fresh conversational context while preserving long-term memory:

```python
# From: 02-use-cases/SRE-agent/sre_agent/multi_agent_langgraph.py
if save_report_response:
    current_session_id = f"interactive-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print("✨ New conversation session started.")
```

---

## 6. Session and AgentCore Memory

AgentCore Memory has a two-tier architecture that interacts with sessions in specific ways.

### 6.1 Short-Term Memory (Events)

Events are conversation turns stored via `create_event()`, scoped by `(memory_id, actor_id, session_id)`:

```python
# From: 02-use-cases/site-reliability-agent-workshop/lab_helpers/short_term_memory.py
self.memory_client.create_event(
    memory_id=self.memory_id,
    actor_id=actor_id,
    session_id=session_id,  # Scopes this event to the current session
    messages=[(message_text, message_role)]
)
```

Short-term events are **session-scoped** — retrieving conversation history with `get_last_k_turns()` returns only turns from the specified session.

### 6.2 Long-Term Memory (Extracted Memories)

Long-term memories are automatically extracted from events via configurable triggers:

```python
# From: xue-docs/GUIDE.md — Memory section
"triggerConditions": [
    {"messageBasedTrigger": {"messageCount": 5}},      # After 5 messages
    {"tokenBasedTrigger": {"tokenCount": 1000}},        # After 1000 tokens
    {"timeBasedTrigger": {"idleSessionTimeout": 900}}   # After 15 min idle
]
```

Long-term memories are stored with namespace paths that can include or exclude `{sessionId}`.

### 6.3 Memory Namespace Scoping

Namespaces are hierarchical paths that control memory isolation:

| Namespace Pattern | Scope | Survives Session End? | Survives Redeployment? |
|---|---|---|---|
| `/sre/users/{actorId}/preferences` | User-scoped (all sessions) | ✅ Yes | ✅ Yes |
| `/sre/infrastructure/{actorId}/{sessionId}` | Session-scoped | ✅ Yes (persisted externally) | ✅ Yes |
| Cross-session search (omit `{sessionId}`) | All sessions for a user | N/A | ✅ Yes |

The SRE Agent's memory hooks demonstrate this dual pattern — **cross-session search** for investigation planning but **session-scoped saving** for investigation results:

```python
# From: 02-use-cases/SRE-agent/sre_agent/memory/hooks.py

# Cross-session search (session_id=None → searches all sessions):
all_knowledge = self.memory_client.retrieve_memories(
    memory_type="infrastructure",
    actor_id=user_id,
    query=query,
    session_id=None,  # Cross-session
)

# Session-scoped save:
success = _save_investigation_summary(
    self.memory_client, actor_id, incident_id, summary, session_id  # This session only
)
```

---

## 7. Code Examples from This Repository

### 7.1 Sending a Session ID (Client Side)

```python
# Pattern used across multiple samples in this repo
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json",
    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
}
response = requests.post(f"{runtime_url}/invocations", headers=headers, json=payload)
```

Files using this pattern include:
- `02-use-cases/customer-support-assistant/app_modules/chat.py`
- `02-use-cases/AWS-operations-agent/chatbot-client/src/client.py`
- `02-use-cases/A2A-multi-agent-incident-response/scripts/shared_utils.py`
- `01-tutorials/01-AgentCore-runtime/06-bi-directional-streaming/websocket_helpers.py`

### 7.2 Receiving a Session ID (Agent Side)

```python
# From: 01-tutorials/01-AgentCore-runtime/03-advanced-concepts/02-understanding-runtime-context/
@app.entrypoint
def strands_agent_bedrock_handling_context(payload, context):
    session_id = context.session_id
    print("Runtime Session ID:", context.session_id)
    # Session ID available for memory operations, logging, etc.
```

### 7.3 Session ID in Agent State (Multi-Agent)

```python
# From: 02-use-cases/SRE-agent/sre_agent/agent_state.py
class AgentState(TypedDict):
    # ... other fields ...
    user_id: Optional[str]       # For user preference tracking
    actor_id: Optional[str]      # Actor ID for memory storage and retrieval
    session_id: Optional[str]    # Session ID for conversation grouping
    memory_context: Optional[Dict[str, Any]]
```

---

## 8. Summary

> **Sourcing note**: The characterization "Session = MicroVM" is an inference from AWS tutorial content that states "each session runs in its own microVM" — the tutorials describe sessions as running in microVMs, not as being identical to them. See the [Evidence Assessment](./session-design-comparison.md#appendix-evidence-assessment-for-tight-session-compute-coupling) for details.

```
┌───────────────────────────────────────────────────────┐
│              EPHEMERAL (in-container)                  │
│                                                       │
│  Runtime Session = MicroVM                            │
│  ├── Scoped to: agent_arn + session_id                │
│  ├── Lifetime: 15-min idle / 8-hr max                 │
│  ├── Destroyed on: timeout, redeployment, failure     │
│  ├── User binding: NOT enforced (app responsibility)  │
│  └── Cross-version: NO                                │
│                                                       │
│  Contains: Python state, local variables, files,      │
│  tool sessions (Code Interpreter, Browser)            │
└───────────────────────┬───────────────────────────────┘
                        │ Agent code writes to
                        ▼
┌───────────────────────────────────────────────────────┐
│              PERSISTENT (AgentCore Memory)             │
│                                                       │
│  Scoped by: (memory_id, actor_id, session_id)         │
│  ├── Session-scoped: "/path/{actorId}/{sessionId}"    │
│  ├── User-scoped:    "/path/{actorId}/preferences"    │
│  ├── Cross-session:  retrieve with prefix (no sid)    │
│  ├── Survives: redeployments, version upgrades        │
│  └── Extraction: auto via triggers (msg/token/idle)   │
└───────────────────────────────────────────────────────┘
```

| Question | Answer |
|----------|--------|
| What is a session? | An isolated microVM hosting your agent, identified by `session_id` |
| How is it scoped? | By `(agent_arn, session_id)` — NOT inherently by user |
| Does it belong to one user? | Platform doesn't enforce it; all samples bind it to one user via `actor_id` |
| Does it survive redeployment? | **No** — ephemeral state dies. Only AgentCore Memory persists across versions |
| How long does it live? | 15-min idle timeout (configurable), 8-hour max lifetime |
| Can multiple invocations share a session? | **Yes** — same session ID → same container → state preserved |

### References

- [Understanding Runtime Context Tutorial](../01-tutorials/01-AgentCore-runtime/03-advanced-concepts/02-understanding-runtime-context/README.md)
- [AgentCore Technical Guide — Code Interpreter Sessions](./GUIDE.md) (lines 540–650)
- [AgentCore Technical Guide — Memory Namespaces](./GUIDE.md) (lines 1304–1333)
- [AgentCore User-Agent Protocol](./agentcore-user-agent-protocol.md)
- [AWS Docs: Use isolated sessions for agents](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html)
- [AWS Docs: Configure lifecycle settings](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html)
