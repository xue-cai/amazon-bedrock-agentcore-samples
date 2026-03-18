# Session Design Comparison: Current AgentCore vs Hosted Agents Proposal

This document compares two session models:

- **Model A — Current AgentCore** (as analyzed in [session-lifecycle.md](./session-lifecycle.md), based on today's runtime behavior and samples)
- **Model B — Hosted Agents Proposal** (the "Sessions in Hosted Agents" design spec)

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
4. [What Changes for Developers](#4-what-changes-for-developers)
5. [Migration Implications](#5-migration-implications)
6. [Open Questions](#6-open-questions)
7. [Appendix: Evidence Assessment for Tight Session-Compute Coupling](#appendix-evidence-assessment-for-tight-session-compute-coupling)

---

## 1. The Fundamental Difference

The two models disagree on what a session **is**:

| | Current AgentCore (Model A) | Hosted Agents Proposal (Model B) |
|---|---|---|
| **Session =** | A microVM / container | A logical, durable execution context |
| **Compute =** | The session itself | An ephemeral implementation detail |
| **Analogy** | A session is a pet (you care for it, when it dies it's gone) | A session is a workflow (it outlives any single executor) |

```
MODEL A (Current)                    MODEL B (Proposed)

  Session ≡ MicroVM                    Session (logical)
  ┌──────────────┐                     ┌──────────────┐
  │ Python state  │                     │ Durable store │ ← persists
  │ Local files   │                     │ Isolation key │
  │ Env vars      │ ← all ephemeral    │ Version pin   │
  │ Memory (RAM)  │                     └──────┬───────┘
  └──────────────┘                            │ attached to
        │ dies with container                  ▼
        ▼                               Sandbox (ephemeral)
     [gone]                             ┌──────────────┐
                                        │ RAM, process  │ ← ephemeral
                                        │ state, env    │
                                        └──────────────┘
                                              │ dies
                                              ▼
                                        New Sandbox spun up
                                        Durable store reattached
                                        Execution resumes
```

---

## 2. Side-by-Side Comparison

| Dimension | Current AgentCore (Model A) | Hosted Agents Proposal (Model B) |
|---|---|---|
| **Session identity** | `session_id` (opaque string via HTTP header) | `session_id` (logical, long-lived identifier) |
| **Session creation** | Implicit on first request with a session ID | Implicit on first request **or** explicit via Create Session API |
| **Session–compute coupling** | **Tight** — session IS the container | **Loose** — session outlives any container |
| **In-memory state** | Preserved while container lives | **Never guaranteed** across restarts |
| **Durable storage** | None built-in; requires external AgentCore Memory | **Built-in** session-scoped artifact storage |
| **Idle behavior** | Container stays alive (15-min timeout) | Sandbox terminated; session persists; resumes on next request |
| **Max lifetime** | 8 hours (hard limit) | TTL-based (inactivity), no stated hard max |
| **Redeployment** | Session dies with container | Session survives; sandbox replaced |
| **Version pinning** | No concept — session destroyed on version change | Existing sessions pinned to original version by default |
| **Version migration** | Not possible | Opt-in via endpoint configuration |
| **Isolation mechanism** | MicroVM boundary (hardware-level) | Explicit isolation key (logical) |
| **User binding** | Not enforced at platform level | Not enforced — uses isolation key |
| **Tool sessions** | Separate concept (Code Interpreter, Browser) | Not mentioned (potentially subsumed) |
| **External memory** | AgentCore Memory service (separate) | Not discussed (session storage may replace some use cases) |
| **Conceptual hierarchy** | `agent_arn → session_id → container` | `Agent → Endpoint → Session → Sandbox → MicroVM` |
| **Analogous system** | Standard container session affinity | Temporal workflows / Azure Durable Functions |

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

**Impact**: Model B fundamentally changes the developer mental model. In Model A, "my session timed out" means "my state is lost." In Model B, "my sandbox was reclaimed" means "my session will resume with durable state when I send the next request."

### 3.2 Durability and State Persistence

This is the most consequential difference.

| State Type | Model A | Model B |
|---|---|---|
| Python variables (RAM) | ✅ Preserved while container lives | ❌ Never guaranteed |
| Local files | ✅ Preserved while container lives | ✅ Persisted if in session storage |
| Conversation history (in-memory) | ✅ Preserved while container lives | ❌ Must be explicitly checkpointed |
| Environment variables | ✅ Preserved while container lives | ❌ Never guaranteed |
| Durable artifacts | ❌ No built-in mechanism | ✅ First-class concept |
| External memory | ✅ Via AgentCore Memory service | Not discussed |

**Key tension**: Model A gives developers in-memory convenience (state "just works" within a session) but fragility (everything vanishes on timeout). Model B gives durability guarantees but requires explicit checkpointing — nothing survives automatically.

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

### 3.3 Lifecycle Across Restarts

**Model A lifecycle**:
```
Create → Active → Idle (container alive) → Timeout → Terminated (gone)
```

**Model B lifecycle**:
```
Create → Active → Idle (sandbox terminated) → Resume (new sandbox) → ... → Expire/Delete
```

The critical difference is that Model B has a **resume** step that Model A lacks. In Model A, idle → timeout → destroyed is one-way. In Model B, idle → sandbox terminated, but session remains, and a new request triggers a new sandbox with durable state reattached.

**Implication for the samples in this repo**: The SRE Agent currently resets `session_id` after saving an investigation report ([multi_agent_langgraph.py](../02-use-cases/SRE-agent/sre_agent/multi_agent_langgraph.py)). Under Model B, this reset would still create a new logical session, but the old session would remain alive (in idle/expired state) rather than being immediately destroyed.

### 3.4 Versioning and Version Pinning

**Model A**: No versioning concept at the session level. When a new agent version is deployed, existing containers are terminated. Sessions die with their containers. There is no version affinity.

**Model B**: Sessions are version-aware:
- New sessions are routed to the endpoint's current version
- Existing sessions are **pinned to their original version by default**
- Version migration is opt-in via endpoint configuration

This is a significant addition. It means:

| Scenario | Model A | Model B |
|---|---|---|
| Deploy v2 while v1 session active | v1 session destroyed | v1 session continues on v1 code |
| User returns after v2 deploy | Gets v2, loses context | Gets v1 (pinned), keeps session |
| Want to migrate session to v2 | Not possible | Opt-in via endpoint policy |

**Analogy**: Model B's version pinning mirrors Temporal's workflow versioning — running executions continue on their original workflow definition until explicitly migrated.

### 3.5 Isolation Model

**Model A**: Isolation is via **microVM boundaries** — hardware-level separation. Each session gets its own microVM with dedicated CPU, memory, and filesystem. This is strong isolation, but tightly coupled to compute.

**Model B**: Isolation is via an **explicit isolation key** assigned at session creation. The spec states sessions are "isolated via an explicit isolation key" but doesn't specify the enforcement mechanism (microVM, container, namespace, etc.).

**Question**: Does Model B's isolation key map to a microVM, or is it a logical/namespace-based isolation? If logical, it may be less secure than Model A's hardware isolation. If it still maps to microVMs under the hood, the difference is mainly in terminology.

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

**Open question**: How does Model B's session storage relate to AgentCore Memory? Possibilities:
1. **Replaces it** for session-scoped data (Memory only needed for cross-session/user-scoped persistence)
2. **Coexists** as a lower-level primitive (raw files vs. semantic memory)
3. **Subsumes it** (Memory becomes an implementation of session storage)

---

## 4. What Changes for Developers

### What Gets Easier

| Capability | Model A (today) | Model B (proposed) |
|---|---|---|
| Survive idle timeout | ❌ State lost after 15 min | ✅ Session resumes with durable artifacts |
| Long-running workflows | Limited to 8-hr max | TTL-based, potentially longer |
| Version upgrades | Breaks all sessions | Existing sessions unaffected (pinned) |
| Checkpointing | DIY via AgentCore Memory | Built-in session storage |
| Create session explicitly | Not supported | Supported via API |

### What Gets Harder

| Capability | Model A (today) | Model B (proposed) |
|---|---|---|
| In-memory state | "Just works" within timeout | Must explicitly checkpoint everything |
| Simple stateless agents | No ceremony needed | Same (no change) |
| Framework integration | Frameworks manage state in-memory | Frameworks must integrate with session storage |
| Tool sessions | Clear separation | Unclear how they fit |

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
| 1 | **How does session storage relate to AgentCore Memory?** | Model B introduces built-in session storage. Does it coexist with, replace, or subsume the Memory service? |
| 2 | **What happens to Tool Sessions?** | Model B doesn't mention Code Interpreter or Browser Tool sessions. Are they subsumed into the session concept? |
| 3 | **Is in-memory state ever preserved?** | Model B says "not guaranteed." Is there a warm-resume path where the sandbox isn't terminated, or is checkpoint/restore always required? |
| 4 | **What is the isolation key?** | Model B mentions an explicit isolation key. Is this the same as `session_id`? Does it map to microVM isolation or logical namespacing? |
| 5 | **What is the TTL for session expiration?** | Model B says TTL-based on inactivity but doesn't specify defaults. Is 15-min still the default? Is there still an 8-hr hard max? |
| 6 | **How do frameworks (Strands, LangGraph) integrate?** | Do agent frameworks need to add built-in checkpoint/restore support for session storage, or is this transparent? |
| 7 | **Is there a migration period?** | Will both models coexist? Can developers opt into Model B while Model A remains the default? |
| 8 | **What is the session storage API?** | Model B mentions durable artifacts and files/checkpoints but doesn't specify the developer API. What does read/write look like? |
| 9 | **Can a session be moved between agents?** | Model B says sessions are associated with agents, not versions. Can a session be reassigned to a different agent entirely? |
| 10 | **How does version pinning interact with AgentCore Memory?** | If a session is pinned to v1 but Memory schemas change in v2, is there a compatibility guarantee? |

---

## Summary Table

| Dimension | Model A (Current) | Model B (Proposed) | Verdict |
|---|---|---|---|
| **Abstraction** | Session = container | Session = logical entity | Model B is cleaner |
| **Durability** | None built-in | Built-in session storage | Model B is more robust |
| **In-memory convenience** | State persists in container | Must checkpoint everything | Model A is simpler for short-lived agents |
| **Version management** | No versioning | Version pinning + migration | Model B is production-ready |
| **Isolation** | MicroVM (strong) | Isolation key (TBD) | Depends on implementation |
| **Developer effort** | Low (stateless or short-lived) | Higher (must checkpoint) | Trade-off: effort vs durability |
| **Long-running workflows** | Limited (8-hr max) | TTL-based (potentially unlimited) | Model B wins |
| **Framework compatibility** | Works with current frameworks as-is | Requires framework updates | Model A has momentum |
| **External memory** | AgentCore Memory (well-defined) | Unclear relationship | Needs clarification |

**Bottom line**: Model B is a more principled architecture — sessions as durable workflows is the right long-term abstraction. However, it introduces a breaking change in developer expectations (no more implicit in-memory persistence) and leaves several integration questions open (Memory service, Tool sessions, framework support).

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
