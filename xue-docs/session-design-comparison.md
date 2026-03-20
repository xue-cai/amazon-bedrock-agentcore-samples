# Session Design Comparison: AgentCore, Hosted Agents, and OpenClaw

This document compares three session models:

- **Model A — Current AgentCore** (as analyzed in [session-lifecycle.md](./session-lifecycle.md), based on today's runtime behavior and samples)
- **Model B — Hosted Agents Proposal** (the "Sessions in Hosted Agents" design spec)
- **Model C — OpenClaw** (open-source AI coding agent, per the [Session Management & Compaction Deep Dive](https://docs.openclaw.ai/reference/session-management-compaction#session-management-%26-compaction-deep-dive))

---

## Table of Contents

1. [The Fundamental Difference](#1-the-fundamental-difference)
2. [Side-by-Side Comparison](#2-side-by-side-comparison)
3. [Deep Dives on Key Differences](#3-deep-dives-on-key-differences)
   - [3.1 What Is a Session?](#31-what-is-a-session)
   - [3.2 Durability and State Persistence](#32-durability-and-state-persistence)
   - [3.3 Lifecycle Across Restarts](#33-lifecycle-across-restarts)
   - [3.4 Versioning and Version Pinning](#34-versioning-and-version-pinning)
   - [3.5 Isolation Model](#35-isolation-model)
   - [3.6 Session–Memory Relationship](#36-session-memory-relationship)
   - [3.7 Context Window Management and Compaction](#37-context-window-management-and-compaction)
4. [What Changes for Developers](#4-what-changes-for-developers)
5. [Migration Implications](#5-migration-implications)
6. [Open Questions](#6-open-questions)
7. [Appendix: Evidence Assessment for Tight Session-Compute Coupling](#appendix-evidence-assessment-for-tight-session-compute-coupling)

---

## 1. The Fundamental Difference

The three models disagree on what a session **is**:

| | Current AgentCore (Model A) | Hosted Agents Proposal (Model B) | OpenClaw (Model C) |
|---|---|---|---|
| **Session =** | A microVM / container | A logical, durable execution context | A routing namespace backed by an append-only transcript file |
| **Compute =** | The session itself | An ephemeral implementation detail | A single Gateway process (not per-session compute) |
| **Analogy** | A session is a pet (you care for it, when it dies it's gone) | A session is a workflow (it outlives any single executor) | A session is a journal (append-only log that can be compacted but never lost) |

```
MODEL A (Current)                    MODEL B (Proposed)                   MODEL C (OpenClaw)

  Session ≡ MicroVM                    Session (logical)                    Session (file-backed)
  ┌──────────────┐                     ┌──────────────┐                    ┌──────────────────────┐
  │ Python state  │                     │ Durable store │ ← persists        │ sessions.json (meta)  │ ← persists
  │ Local files   │                     │ Isolation key │                    │ <sessionId>.jsonl     │ ← append-only
  │ Env vars      │ ← all ephemeral    │ Version pin   │                    │  (transcript tree)    │   transcript
  │ Memory (RAM)  │                     └──────┬───────┘                    └──────────┬───────────┘
  └──────────────┘                            │ attached to                            │ owned by
        │ dies with container                  ▼                                        ▼
        ▼                               Sandbox (ephemeral)                      Gateway process
     [gone]                             ┌──────────────┐                    ┌──────────────────────┐
                                        │ RAM, process  │ ← ephemeral       │ Reads transcript     │
                                        │ state, env    │                    │ Rebuilds context     │
                                        └──────────────┘                    │ Compacts when needed │
                                              │ dies                        └──────────────────────┘
                                              ▼                                        │
                                        New Sandbox spun up                   Transcript survives
                                        Durable store reattached              restarts, resets,
                                        Execution resumes                     and compaction
```

---

## 2. Side-by-Side Comparison

| Dimension | Current AgentCore (Model A) | Hosted Agents Proposal (Model B) | OpenClaw (Model C) |
|---|---|---|---|
| **Session identity** | `session_id` (opaque string via HTTP header) | `session_id` (logical, long-lived identifier) | `sessionKey` (routing bucket) + `sessionId` (current transcript file) |
| **Session creation** | Implicit on first request with a session ID | Implicit on first request **or** explicit via Create Session API | Implicit on first message; explicit via `/new` or `/reset` commands |
| **Session–compute coupling** | **Tight** — session IS the container | **Loose** — session outlives any container | **None** — single Gateway process serves all sessions |
| **In-memory state** | Preserved while container lives | **Never guaranteed** across restarts | No per-session in-memory state; context rebuilt from transcript each turn |
| **Durable storage** | None built-in; requires external AgentCore Memory | **Built-in** session-scoped artifact storage | **Built-in** append-only JSONL transcripts + `sessions.json` metadata store |
| **Idle behavior** | Container stays alive (15-min timeout) | Sandbox terminated; session persists; resumes on next request | Session persists on disk; idle expiry creates a **new** `sessionId` (old transcript retained) |
| **Max lifetime** | 8 hours (hard limit) | TTL-based (inactivity), no stated hard max | No hard max; configurable idle reset, daily reset, and disk-budget pruning |
| **Redeployment** | Session dies with container | Session survives; sandbox replaced | Session transcripts survive Gateway restarts (files on disk) |
| **Version pinning** | No concept — session destroyed on version change | Existing sessions pinned to original version by default | No version-pinning concept; sessions are agent-scoped, not version-scoped |
| **Version migration** | Not possible | Opt-in via endpoint configuration | Not applicable (single-process, no per-session versioning) |
| **Isolation mechanism** | MicroVM boundary (hardware-level) | Explicit isolation key (logical) | `sessionKey` routing (logical); `dmScope` for per-user DM isolation |
| **User binding** | Not enforced at platform level | Not enforced — uses isolation key | Via `sessionKey` patterns (e.g., `agent:<id>:dm:<userId>`) |
| **Tool sessions** | Separate concept (Code Interpreter, Browser) | Not mentioned (potentially subsumed) | Tool calls stored in transcript alongside messages; no separate tool sessions |
| **External memory** | AgentCore Memory service (separate) | Not discussed (session storage may replace some use cases) | Workspace-based memory files (`memory/YYYY-MM-DD.md`, `MEMORY.md`) written by agent |
| **Conceptual hierarchy** | `agent_arn → session_id → container` | `Agent → Endpoint → Session → Sandbox → MicroVM` | `Agent → sessionKey → sessionId → transcript (.jsonl)` |
| **Analogous system** | Standard container session affinity | Temporal workflows / Azure Durable Functions | Git log / append-only event sourcing |
| **Context window management** | Not managed (full history in RAM) | Not discussed | **Compaction**: summarizes older turns into a compaction entry; keeps recent messages intact |
| **Maintenance / cleanup** | Container terminated on timeout | TTL-based session expiry | Configurable store maintenance: `pruneAfter`, `maxEntries`, `maxDiskBytes`, archive rotation |

---

## 3. Deep Dives on Key Differences

### 3.1 What Is a Session?

**Model A**: A session is a running microVM. It exists only while the container exists. The `session_id` is a routing key that provides container affinity.

```
# Model A: session_id → container affinity
X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: abc-123
→ Routes to the SAME microVM every time
→ If microVM dies, session is gone
```

**Model B**: A session is an abstract entity that owns durable state and is _attached to_ (not _equivalent to_) compute. The session persists even when no compute is running.

```
# Model B: session_id → logical session → attached sandbox
session_id: abc-123
→ Session exists independently of any sandbox
→ Sandbox may be terminated and re-created
→ Session's durable artifacts survive
```

**Model C**: A session is a **routed conversation bucket** backed by an append-only transcript file on disk. There are two levels of identity: a `sessionKey` (the routing bucket, e.g., per-user or per-channel) and a `sessionId` (the current transcript file for that bucket). The Gateway process reads the transcript to rebuild model context on every turn — there is no per-session compute or in-memory state.

```
# Model C: sessionKey → sessionId → transcript file
sessionKey: agent:my-agent:main
  └─ sessionId: 2026-03-15-abc123
       └─ ~/.openclaw/agents/<agentId>/sessions/2026-03-15-abc123.jsonl
            ├── Session header (type: "session")
            ├── User message
            ├── Assistant message + tool calls
            ├── ... (append-only tree)
            └── Compaction summary (when older turns are summarized)
→ Transcript file is permanent (survives Gateway restarts)
→ /new or /reset creates a NEW sessionId under the same sessionKey
→ Old transcript file is retained on disk
```

**Impact**: Model C takes a fundamentally different approach from both A and B — it decouples session persistence from compute entirely by treating sessions as files, not processes. There is no sandbox, container, or microVM per session. The Gateway reads from disk, sends context to the model API, and appends the response. This makes sessions inherently durable (as durable as the filesystem), but context must fit within the model's context window. Model C addresses this via compaction — summarizing older turns to stay within bounds.

### 3.2 Durability and State Persistence

This is the most consequential difference.

| State Type | Model A | Model B | Model C |
|---|---|---|---|
| Python variables (RAM) | ✅ Preserved while container lives | ❌ Never guaranteed | ❌ No per-session process |
| Local files | ✅ Preserved while container lives | ✅ Persisted if in session storage | ✅ Transcript files always persisted |
| Conversation history (in-memory) | ✅ Preserved while container lives | ❌ Must be explicitly checkpointed | ✅ Rebuilt from transcript each turn |
| Environment variables | ✅ Preserved while container lives | ❌ Never guaranteed | N/A (single Gateway process) |
| Durable artifacts | ❌ No built-in mechanism | ✅ First-class concept | ✅ Transcript is the artifact (append-only JSONL) |
| External memory | ✅ Via AgentCore Memory service | Not discussed | ✅ Workspace files (`MEMORY.md`, `memory/YYYY-MM-DD.md`) |
| Context window management | ❌ Not managed | Not discussed | ✅ Compaction (summarize old turns, keep recent) |

**Key tension**: Model A gives developers in-memory convenience (state "just works" within a session) but fragility (everything vanishes on timeout). Model B gives durability guarantees but requires explicit checkpointing — nothing survives automatically. Model C takes a third path: **the transcript is always durable**, but in-memory state never exists per-session; the model's context is reconstructed from the transcript every turn, and compaction keeps it within bounds.

**Example — Model A (current samples)**:

```python
# In Model A, conversation history accumulates in-memory automatically.
# The Strands agent framework keeps messages in the agent's tool_use loop.
# This "just works" across invocations within the 15-min window.
@app.entrypoint
async def invoke(payload, context):
    # Agent's in-memory state includes all prior turns
    response = agent(payload["prompt"])
    yield response
```

**Example — Model B (proposed pattern)**:

```python
# In Model B, the developer must checkpoint state to session storage.
# In-memory state is not preserved across sandbox restarts.
@app.entrypoint
async def invoke(payload, context):
    # Load prior state from session storage
    history = load_from_session_storage(context.session_id, "history.json")
    history.append(payload["prompt"])
    response = agent(payload["prompt"], history=history)
    # Persist updated state
    save_to_session_storage(context.session_id, "history.json", history)
    yield response
```

**Example — Model C (OpenClaw pattern)**:

```
# In Model C, the transcript IS the history. Every turn is appended to a JSONL file.
# The Gateway reads the transcript and rebuilds context for the model automatically.
# No developer checkpointing needed — persistence is built into the transport layer.
#
# Transcript file: ~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl
#
# {"type":"session","id":"...","timestamp":"..."}
# {"type":"message","role":"user","content":"Hello","id":"1","parentId":"root"}
# {"type":"message","role":"assistant","content":"Hi!","id":"2","parentId":"1"}
# {"type":"message","role":"user","content":"What's 2+2?","id":"3","parentId":"2"}
# {"type":"message","role":"assistant","content":"4","id":"4","parentId":"3"}
#
# When context grows too large, compaction summarizes older entries:
# {"type":"compaction","summary":"User greeted, asked math question...","firstKeptEntryId":"3"}
```

### 3.3 Lifecycle Across Restarts

**Model A lifecycle**:
```
Create → Active → Idle (container alive) → Timeout → Terminated (gone)
```

**Model B lifecycle**:
```
Create → Active → Idle (sandbox terminated) → Resume (new sandbox) → ... → Expire/Delete
```

**Model C lifecycle**:
```
First message → sessionKey created → sessionId assigned → Active (turns appended to transcript)
  → Idle → Next message: if idle < threshold → continue same sessionId
                          if idle > threshold → new sessionId created (old transcript retained)
  → /new or /reset → new sessionId (old transcript retained)
  → Daily reset boundary crossed → new sessionId on next message
  → Compaction (when context nears window limit) → summary entry appended, old turns compressed
  → Disk maintenance (pruneAfter, maxDiskBytes) → old transcripts archived/removed
```

The critical difference is that Model B has a **resume** step that Model A lacks. In Model A, idle → timeout → destroyed is one-way. In Model B, idle → sandbox terminated, but session remains, and a new request triggers a new sandbox with durable state reattached.

Model C takes yet another approach: there is no "resume" because there is nothing to resume — sessions are files, not processes. An idle timeout in Model C doesn't destroy or suspend anything; it simply means the next message starts a **new transcript** under the same routing key. The old transcript remains on disk subject to configured retention policies (disk-budget maintenance may eventually archive or remove old transcripts).

**Implication for the samples in this repo**: The SRE Agent currently resets `session_id` after saving an investigation report ([multi_agent_langgraph.py](../02-use-cases/SRE-agent/sre_agent/multi_agent_langgraph.py)). Under Model B, this reset would still create a new logical session, but the old session would remain alive (in idle/expired state) rather than being immediately destroyed.

### 3.4 Versioning and Version Pinning

**Model A**: No versioning concept at the session level. When a new agent version is deployed, existing containers are terminated. Sessions die with their containers. There is no version affinity.

**Model B**: Sessions are version-aware:
- New sessions are routed to the endpoint's current version
- Existing sessions are **pinned to their original version by default**
- Version migration is opt-in via endpoint configuration

This is a significant addition. It means:

| Scenario | Model A | Model B | Model C |
|---|---|---|---|
| Deploy v2 while v1 session active | v1 session destroyed | v1 session continues on v1 code | Transcript preserved; v2 agent reads same transcript immediately |
| User returns after v2 deploy | Gets v2, loses context | Gets v1 (pinned), keeps session | Gets v2, keeps full transcript history |
| Want to migrate session to v2 | Not possible | Opt-in via endpoint policy | Automatic — transcripts are agent-version-agnostic |

**Analogy**: Model B's version pinning mirrors Temporal's workflow versioning — running executions continue on their original workflow definition until explicitly migrated. Model C sidesteps the version-pinning problem entirely because sessions are data (files), not running processes — upgrading the agent is like upgrading the reader, not the document.

### 3.5 Isolation Model

**Model A**: Isolation is via **microVM boundaries** — hardware-level separation. Each session gets its own microVM with dedicated CPU, memory, and filesystem. This is strong isolation, but tightly coupled to compute.

**Model B**: Isolation is via an **explicit isolation key** assigned at session creation. The spec states sessions are "isolated via an explicit isolation key" but doesn't specify the enforcement mechanism (microVM, container, namespace, etc.).

**Model C**: Isolation is via **`sessionKey` routing** — a purely logical mechanism. The `sessionKey` pattern determines which transcript a message is appended to. For multi-user scenarios, OpenClaw provides `dmScope` configuration to ensure DMs from different users route to separate session keys. However, all sessions share the same Gateway process and filesystem — isolation is at the data-routing level, not the compute level.

**Comparison**:

| Aspect | Model A | Model B | Model C |
|---|---|---|---|
| Mechanism | MicroVM (hardware) | Isolation key (TBD) | `sessionKey` routing (logical) |
| Strength | Strong (process/memory isolation) | Depends on implementation | Weak (data routing only, shared process) |
| Multi-tenant safety | High | TBD | Low (single-user/single-org design) |
| Filesystem isolation | ✅ Dedicated per session | TBD | ❌ Shared agent workspace |

**Question**: Does Model B's isolation key map to a microVM, or is it a logical/namespace-based isolation? If logical, it may be less secure than Model A's hardware isolation. If it still maps to microVMs under the hood, the difference is mainly in terminology. Model C explicitly does not target multi-tenant isolation — it is designed for single-user or trusted-team scenarios.

### 3.6 Session–Memory Relationship

**Model A**: AgentCore Memory is a **separate, external service** with its own APIs, namespaces, and scoping:

```
Session (ephemeral container)
    │ writes to ↓
AgentCore Memory Service (persistent)
    ├── Short-term: events scoped by (memory_id, actor_id, session_id)
    └── Long-term:  extracted memories in namespace paths
```

**Model B**: The proposal introduces **session-scoped durable storage** as a built-in concept:

```
Session (logical, durable)
    ├── Durable artifacts (files, checkpoints) ← NEW built-in
    └── Sandbox (ephemeral execution)
```

**Model C**: OpenClaw has a **two-tier memory model** that is tightly integrated with the session lifecycle:

```
Session (transcript-backed)
    ├── Transcript (.jsonl)          ← conversation history (auto-persisted)
    ├── Compaction summaries         ← compressed older context (in transcript)
    └── Workspace memory files       ← durable knowledge (agent-written)
         ├── MEMORY.md               ← persistent notes
         └── memory/YYYY-MM-DD.md    ← dated memory entries
```

In Model C, the **pre-compaction memory flush** is a key innovation: before auto-compaction summarizes and compresses older turns, the Gateway triggers a silent agentic turn (an automated agent invocation not visible to the user) where the model writes important facts to workspace memory files. This ensures critical context survives compaction even if the compaction summary is lossy. The memory files are regular files in the agent workspace — not a separate service.

**Cross-model comparison**:

| Aspect | Model A | Model B | Model C |
|---|---|---|---|
| Memory service | External (AgentCore Memory) | Built-in session storage | Workspace files (agent-written) |
| Conversation persistence | In-memory only | Explicit checkpointing | Append-only transcript (automatic) |
| Cross-session memory | Via Memory service namespaces | Not discussed | Via workspace files (shared across sessions) |
| Pre-compaction safety | N/A (no compaction) | N/A | Pre-compaction memory flush (auto-writes durable notes) |

**Open question**: How does Model B's session storage relate to AgentCore Memory? Possibilities:
1. **Replaces it** for session-scoped data (Memory only needed for cross-session/user-scoped persistence)
2. **Coexists** as a lower-level primitive (raw files vs. semantic memory)
3. **Subsumes it** (Memory becomes an implementation of session storage)

Model C's approach suggests a fourth possibility: **memory is just files**, and the session system manages when and how to write them, making the distinction between "session storage" and "memory" a matter of convention rather than infrastructure.

### 3.7 Context Window Management and Compaction

This is a dimension where Model C introduces a concept absent from both Models A and B: **compaction** — the active management of conversation history to fit within a model's context window.

| Aspect | Model A | Model B | Model C |
|---|---|---|---|
| Context window awareness | None — all history in RAM | Not discussed | Core feature — tracks `contextTokens` vs `contextWindow` |
| What happens when context is full | Agent fails or truncates ad-hoc | Not discussed | Auto-compaction: summarize older turns, keep recent messages |
| Compaction trigger | N/A | N/A | Overflow recovery **or** threshold maintenance (`contextTokens > contextWindow - reserveTokens`) |
| Manual compaction | N/A | N/A | `/compact` command with optional custom instructions |
| Pre-compaction safety | N/A | N/A | Memory flush: silent agentic turn writes durable notes before compaction |
| Compaction result | N/A | N/A | `compaction` entry in transcript with `summary` + `firstKeptEntryId` |
| Pruning (separate from compaction) | N/A | N/A | In-memory trimming of tool results only (not persisted) |

**Why this matters**: Models A and B both implicitly assume that conversation history either fits in memory/storage or is managed externally. Model C confronts the context window limit head-on as a core session management concern. For long-running agent sessions, this is arguably the most important practical problem — and Model C's compaction provides a principled, configurable solution.

**Compaction configuration (Model C)**:

```json5
{
  compaction: {
    enabled: true,
    reserveTokens: 16384,   // headroom for prompts + next output
    keepRecentTokens: 20000, // recent messages to keep in full
  }
}
```

**Lesson for Models A and B**: Any production session model will likely need a compaction-like mechanism. Model A's approach (state dies with the container) sidesteps the problem but at a high cost. Model B's session storage could store conversation history but doesn't describe how to manage its growth. Model C's approach — append-only transcripts with periodic compaction — is a pattern worth considering for both.

---

## 4. What Changes for Developers

### What Gets Easier

| Capability | Model A (today) | Model B (proposed) | Model C (OpenClaw) |
|---|---|---|---|
| Survive idle timeout | ❌ State lost after 15 min | ✅ Session resumes with durable artifacts | ✅ Transcript always on disk; idle creates new sessionId but old transcript retained |
| Long-running workflows | Limited to 8-hr max | TTL-based, potentially longer | No hard max; compaction keeps context bounded |
| Version upgrades | Breaks all sessions | Existing sessions unaffected (pinned) | Transcripts unaffected; new agent version reads existing transcripts |
| Checkpointing | DIY via AgentCore Memory | Built-in session storage | Automatic (transcript) + workspace memory files |
| Create session explicitly | Not supported | Supported via API | Via `/new`, `/reset` commands or idle/daily reset |
| Context window management | Not managed | Not discussed | Built-in compaction with configurable thresholds |

### What Gets Harder

| Capability | Model A (today) | Model B (proposed) | Model C (OpenClaw) |
|---|---|---|---|
| In-memory state | "Just works" within timeout | Must explicitly checkpoint everything | No per-session in-memory state (context rebuilt from transcript) |
| Simple stateless agents | No ceremony needed | Same (no change) | Same (no change) |
| Framework integration | Frameworks manage state in-memory | Frameworks must integrate with session storage | Frameworks must work with transcript-based context (Gateway handles this) |
| Tool sessions | Clear separation | Unclear how they fit | Tool calls inline in transcript (no separate concept) |
| Multi-tenant isolation | Strong (microVM) | TBD | Weak (shared Gateway process, logical routing only) |

### Code Pattern Changes

**Current pattern (Model A)** — frameworks like Strands handle state in-memory:
```python
# Works today: in-memory state persists across invocations
agent = Agent(model=model, tools=tools)

@app.entrypoint
async def invoke(payload, context):
    response = agent(payload["prompt"])  # Agent remembers prior turns
    yield response
```

**Proposed pattern (Model B)** — explicit checkpoint/restore:
```python
agent = Agent(model=model, tools=tools)

@app.entrypoint
async def invoke(payload, context):
    # Restore from session storage
    state = context.session_storage.load("agent_state.json")
    if state:
        agent.restore(state)

    response = agent(payload["prompt"])

    # Checkpoint to session storage
    context.session_storage.save("agent_state.json", agent.snapshot())
    yield response
```

---

## 5. Migration Implications

If AgentCore moves from Model A to Model B, existing samples would need updates:

| Sample / Pattern | Impact | Migration Effort |
|---|---|---|
| Simple chat agents (Strands) | Must add checkpoint/restore logic | Medium |
| SRE Agent multi-agent state | Already uses AgentCore Memory; may need session storage for in-flight state | Low–Medium |
| Short-term memory hooks | May be simplified (session storage replaces some Memory API usage) | Medium |
| Session ID generation | Unchanged (same concept) | None |
| Tool sessions (Code Interpreter) | Unclear — needs clarification | Unknown |
| Client-side HTTP headers | Unchanged (`X-Amzn-Bedrock-AgentCore-Runtime-Session-Id`) | None |

### Backward Compatibility Concern

Model B's explicit "in-memory state is never guaranteed" is a **breaking behavioral change** from Model A, where in-memory state reliably persists within the idle timeout. Agents built for Model A that rely on in-memory continuity would break silently under Model B if the sandbox is reclaimed during an idle period.

---

## 6. Open Questions

| # | Question | Context |
|---|---|---|
| 1 | **How does session storage relate to AgentCore Memory?** | Model B introduces built-in session storage. Does it coexist with, replace, or subsume the Memory service? Model C's workspace files suggest "memory is just files" as a viable approach. |
| 2 | **What happens to Tool Sessions?** | Model B doesn't mention Code Interpreter or Browser Tool sessions. Are they subsumed into the session concept? Model C stores tool calls inline in transcripts. |
| 3 | **Is in-memory state ever preserved?** | Model B says "not guaranteed." Is there a warm-resume path where the sandbox isn't terminated, or is checkpoint/restore always required? Model C avoids this question entirely (no per-session process). |
| 4 | **What is the isolation key?** | Model B mentions an explicit isolation key. Is this the same as `session_id`? Does it map to microVM isolation or logical namespacing? |
| 5 | **What is the TTL for session expiration?** | Model B says TTL-based on inactivity but doesn't specify defaults. Is 15-min still the default? Is there still an 8-hr hard max? |
| 6 | **How do frameworks (Strands, LangGraph) integrate?** | Do agent frameworks need to add built-in checkpoint/restore support for session storage, or is this transparent? |
| 7 | **Is there a migration period?** | Will both models coexist? Can developers opt into Model B while Model A remains the default? |
| 8 | **What is the session storage API?** | Model B mentions durable artifacts and files/checkpoints but doesn't specify the developer API. What does read/write look like? |
| 9 | **Can a session be moved between agents?** | Model B says sessions are associated with agents, not versions. Can a session be reassigned to a different agent entirely? |
| 10 | **How does version pinning interact with AgentCore Memory?** | If a session is pinned to v1 but Memory schemas change in v2, is there a compatibility guarantee? |
| 11 | **Should Models A/B adopt compaction?** | Model C's compaction mechanism addresses a real problem (context window limits) that Models A and B don't discuss. Should a compaction-like mechanism be part of any production session model? |
| 12 | **Is the append-only transcript pattern viable at cloud scale?** | Model C's JSONL transcripts work for single-user scenarios. Could this pattern scale to a multi-tenant cloud platform (Model A/B), potentially backed by a database instead of files? |
| 13 | **Should pre-compaction memory flush be a standard pattern?** | Model C's innovation of triggering a silent memory-write turn before compaction prevents context loss. Is this a pattern that Model B's session storage should support natively? |

---

## Summary Table

| Dimension | Model A (Current) | Model B (Proposed) | Model C (OpenClaw) | Verdict |
|---|---|---|---|---|
| **Abstraction** | Session = container | Session = logical entity | Session = routed transcript file | Model B & C are cleaner; C is simplest |
| **Durability** | None built-in | Built-in session storage | Append-only transcript (inherently durable) | Model C is most robust (durability by default) |
| **In-memory convenience** | State persists in container | Must checkpoint everything | No per-session memory (context rebuilt each turn) | Model A simplest for short-lived agents |
| **Version management** | No versioning | Version pinning + migration | No versioning needed (transcripts are version-agnostic) | Model B most sophisticated; Model C sidesteps the problem |
| **Isolation** | MicroVM (strong) | Isolation key (TBD) | `sessionKey` routing (weak) | Model A strongest for multi-tenant |
| **Developer effort** | Low (stateless or short-lived) | Higher (must checkpoint) | Low (transcript persistence is automatic) | Model A & C are low-effort; Model B requires more work |
| **Long-running workflows** | Limited (8-hr max) | TTL-based (potentially unlimited) | No hard max (compaction manages context growth) | Model B & C win |
| **Framework compatibility** | Works with current frameworks as-is | Requires framework updates | Gateway handles transcript management (frameworks unaware) | Model A has momentum; Model C is transparent |
| **External memory** | AgentCore Memory (well-defined) | Unclear relationship | Workspace files (simple, file-based) | Needs clarification for Model B |
| **Context window management** | Not managed | Not discussed | Built-in compaction (configurable) | Model C uniquely addresses this |
| **Maintenance / cleanup** | Container termination | TTL-based expiry | Configurable disk budget, pruning, archival | Model C most operationally mature |

**Bottom line**: Model B is a more principled architecture than Model A — sessions as durable workflows is the right long-term abstraction. However, it introduces a breaking change in developer expectations (no more implicit in-memory persistence) and leaves several integration questions open (Memory service, Tool sessions, framework support).

Model C (OpenClaw) offers a compelling third perspective: **sessions as append-only files**. This approach achieves durability by default (no checkpointing needed), handles context window limits via compaction, and integrates memory as workspace files rather than a separate service. Its current implementation lacks compute isolation (single Gateway process, no per-session sandboxing), which limits its applicability for multi-tenant cloud platforms — though the underlying architectural pattern (append-only transcripts with compaction) could be adapted with database backing for cloud-scale deployments. Its compaction mechanism and pre-compaction memory flush are patterns that Models A and B should consider adopting.

---

## Appendix: Evidence Assessment for Tight Session-Compute Coupling

This document characterizes the current AgentCore session model as having "tight session–compute coupling" (session IS the container). This appendix distinguishes between what AWS sources directly state versus what was inferred, and flags where the claim could be challenged.

### Directly Stated by AWS (Primary Sources)

The following facts come from the AWS-authored tutorial notebook at [`01-tutorials/.../02-understanding-runtime-context/understanding_runtime_context.ipynb`](../01-tutorials/01-AgentCore-runtime/03-advanced-concepts/02-understanding-runtime-context/understanding_runtime_context.ipynb):

| Claim | Source (notebook cell) | Exact Quote |
|---|---|---|
| Each session runs in a dedicated microVM | Cell 1 (Prerequisites) | "Each session runs in its own microVM with isolated CPU, memory, and filesystem" |
| In-memory state persists across invocations | Cell 1 (Context Persistence) | "Within a session, AgentCore Runtime maintains: Conversation History, Application State, File System, Environment Variables" |
| Session termination destroys the microVM and sanitizes memory | Cell 1 (Isolation and Security) | "After session completion, the microVM is terminated and memory is sanitized" |
| Sessions are ephemeral — not for permanent storage | Cell 1 (Best Practices) | "Ephemeral Nature: Don't rely on sessions for permanent data storage (use AgentCore Memory for persistence)" |
| Session lifecycle has no "resume" step | Cell 1 (Session Lifecycle) | "1. Creation → 2. Active → 3. Idle → 4. Termination: Session ends due to: Inactivity (15 min), Maximum lifetime (8 hrs), Health check failures" |
| Context is lost on new session ID | Cell 18 (Session Isolation results) | Demonstrated: new session ID → completely isolated environment, no access to prior state |

### Inferred (Not Directly Stated)

The following claims are **inferences** built on the above facts. They are reasonable characterizations of observed behavior but are not explicitly stated by AWS:

| Inference | Reasoning | Confidence |
|---|---|---|
| **"Session = container"** (identity equivalence) | The tutorial says sessions *run in* microVMs and are *terminated with* microVMs. The lifecycle has no separation between session termination and compute termination. But the tutorial never uses the phrase "a session IS a container." | **Medium-High** |
| **Sessions die on redeployment** | The lifecycle doc at [`session-lifecycle.md` §5](./session-lifecycle.md#5-sessions-across-agent-versions-and-deployments) lists redeployment as a termination cause. This is inferred from: (a) containers are terminated on new version deploy, (b) session is backed by a container, therefore (c) session dies. No AWS doc was found that explicitly states this. | **Medium** |
| **No session migration API** | Inferred from absence — no such API appears in the SDK, docs, or samples. Absence of evidence is not evidence of absence. | **Medium** |
| **Session routing = container affinity** | The tutorial shows that same session ID → same execution context with preserved state, which is consistent with container affinity. But the mechanism ("routing") is inferred, not described. | **Medium-High** |

### Where the Claim Could Be Challenged

1. **"Runs in" ≠ "Is"**: The tutorial says "each session runs in its own microVM" — which could be read as sessions being *backed by* microVMs rather than *being identical to* microVMs. If AWS were to introduce a durable layer beneath the microVM (as Model B proposes), the current tutorial language would not actually contradict it.

2. **Describing behavior, not contract**: The tutorial may be describing the current runtime's *observable behavior* rather than the intended *architectural contract*. It's possible sessions are already abstractly separate from compute in the platform's internal model, but the current implementation happens not to persist state across sandbox restarts.

3. **Redeployment behavior is underdocumented**: The claim that sessions die on redeployment is inferred from the general container termination model. AWS could implement rolling deployments or session draining without contradicting any currently published documentation.

### The Inference Chain

For transparency, the "tight coupling" characterization follows this logic:

```
Premise 1: Each session runs in its own microVM               (stated)
Premise 2: In-memory state persists across invocations         (stated & demonstrated)
Premise 3: Session termination = microVM termination + cleanup (stated)
Premise 4: No resume-after-termination step in lifecycle       (observed absence)
Premise 5: "Don't rely on sessions for permanent storage"      (stated)
─────────────────────────────────────────────────────────────────────
Conclusion: Session lifecycle is bound to compute lifecycle    (inferred)
```

### Confidence Summary

| Aspect | Confidence | Evidence Type |
|---|---|---|
| Sessions run in dedicated microVMs | **High** | Directly stated |
| In-memory state persists within a session | **High** | Stated and demonstrated |
| Session termination destroys all state | **High** | Directly stated ("memory sanitized") |
| No built-in durable session storage | **High** | Stated ("don't rely on sessions for permanent storage") |
| No resume-after-termination | **Medium-High** | Inferred from lifecycle description (no resume step) |
| Sessions die on redeployment | **Medium** | Inferred (not directly stated anywhere) |
| "Session = container" as architectural identity | **Medium** | Inferred from behavioral descriptions |

**Recommendation**: When discussing the tight-coupling model with AWS stakeholders, lead with the high-confidence directly-stated facts (sessions run in microVMs, state is ephemeral, no built-in durability) and present the equivalence claim ("session IS the container") as a reasonable interpretation of current behavior rather than a documented architectural guarantee.
