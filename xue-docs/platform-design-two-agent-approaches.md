# Platform Design: Two Approaches to Defining and Building Agents

This document analyzes how a cloud platform (AWS, Azure, GCP) should design its agent runtime to support two fundamentally different approaches to building AI agents. It references the architecture and patterns found in this repository's [GUIDE.md](./GUIDE.md), [session-lifecycle.md](./session-lifecycle.md), and [session-design-comparison.md](./session-design-comparison.md).

---

## Table of Contents

1. [The Two Approaches](#1-the-two-approaches)
2. [Naming the Two Types](#2-naming-the-two-types)
3. [Should Both Run on the Same Platform?](#3-should-both-run-on-the-same-platform)
4. [How to Define an Agent — Is IaC Involved?](#4-how-to-define-an-agent--is-iac-involved)
5. [How to Deploy and Update an Agent](#5-how-to-deploy-and-update-an-agent)
6. [Agent Identity and Permission Management](#6-agent-identity-and-permission-management)
7. [Where Does the Agent Run? VPC/VNet Integration](#7-where-does-the-agent-run-vpcvnet-integration)
8. [How Does the Agent Call Tools? Customer Resources?](#8-how-does-the-agent-call-tools-customer-resources)
9. [Where Is Memory Stored? Customer-Controlled Storage?](#9-where-is-memory-stored-customer-controlled-storage)
10. [Overall Architecture Recommendation](#10-overall-architecture-recommendation)
11. [Summary of Design Principles](#11-summary-of-design-principles)

---

## 1. The Two Approaches

There are two fundamentally different ways developers define and build agents:

### Approach 1 — Declarative / Managed

The developer provides:
- The LLM to use
- A system prompt
- A list of tools the agent can call
- A set of memory strategies

The **platform** provides a pre-built agent loop that orchestrates everything — model invocation, tool calling, memory retrieval, and response generation. The developer never writes agent loop code.

### Approach 2 — Code-Based / Bring Your Own Agent (BYOA)

The developer writes the agent code, possibly using a framework SDK (Strands, LangGraph, CrewAI, etc.). The code handles:
- The agent loop (or a pre-defined workflow)
- Tool invocation logic
- Memory management

The developer packages the code with dependencies (typically as a Docker container) for deployment.

### Where AgentCore Fits Today

Looking at this repository, **AgentCore today is squarely Approach 2**. The developer packages their agent code into a Docker container, and AgentCore provides the hosting infrastructure (microVMs, networking, identity, memory, tools). There is no "declarative agent definition" where you specify an LLM + prompt + tools and the platform runs an agent loop for you.

However, the original Bedrock Agents (pre-AgentCore) **was** Approach 1 — you define an agent via API/console with a model, instructions, action groups (tools), and knowledge bases, and AWS runs the orchestration loop.

---

## 2. Naming the Two Types

There are many valid ways to name these two approaches. Each name emphasizes a different aspect of the distinction:

| Approach 1 | Approach 2 | What the Name Emphasizes |
|---|---|---|
| **Declarative Agent** | **Code-Based Agent** | How the agent is *defined* — configuration vs. code |
| **Managed Agent** | **Custom Agent** | Who owns the orchestration — platform vs. developer |
| **Config-Driven Agent** | **Framework-Driven Agent** | What drives the agent loop — platform config vs. SDK framework |
| **Hosted Agent** | **BYOA (Bring Your Own Agent)** | What the platform receives — a spec vs. a container |
| **No-Code Agent** | **Pro-Code Agent** | Developer skill level and control expectations |
| **Platform-Orchestrated Agent** | **Self-Orchestrated Agent** | Who runs the agent loop |
| **Schema-Defined Agent** | **Container-Defined Agent** | The artifact format — JSON/YAML schema vs. Docker image |
| **Turnkey Agent** | **Bespoke Agent** | Effort and customization level |

### Analysis of Each Naming Pair

**Declarative vs. Code-Based** — The most technically precise. "Declarative" correctly implies the developer says *what* they want, not *how* it works. "Code-Based" is clear but broad (Approach 1 agents also involve code, just not agent-loop code).

**Managed vs. Custom** — Used by AWS in other contexts (e.g., managed vs. custom runtimes in Lambda). "Managed" implies the platform handles lifecycle, scaling, and orchestration. "Custom" implies developer-controlled. Risk: "managed" can be confused with "managed infrastructure" (which AgentCore provides for both approaches).

**Config-Driven vs. Framework-Driven** — Highlights what controls the agent loop. "Config-driven" means the platform interprets a configuration to produce behavior. "Framework-driven" means a developer-chosen SDK (LangGraph, Strands, CrewAI) determines behavior. Clear, but "framework-driven" is slightly misleading — an agent could be code-based without using a framework.

**Hosted vs. BYOA** — Emphasizes what the developer hands to the platform: a specification (Hosted) or a runnable artifact (BYOA). AWS uses "Hosted Agent" in the [Hosted Agents Proposal](./session-design-comparison.md). "BYOA" is descriptive but jargon-heavy.

**No-Code vs. Pro-Code** — Borrowed from the low-code/no-code movement. Intuitive for non-technical stakeholders but reductive — Approach 1 developers may still write tool implementations (Lambda functions, API handlers). Not fully accurate.

**Platform-Orchestrated vs. Self-Orchestrated** — The most precise about the key technical difference: who runs the agent loop? But verbose for everyday use.

**Schema-Defined vs. Container-Defined** — Focuses on the deployment artifact. Schema-defined agents are a JSON/YAML definition that the platform compiles; container-defined agents are Docker images the developer builds. Technically precise but narrow.

**Turnkey vs. Bespoke** — Good for product marketing. "Turnkey" implies ready-to-use. "Bespoke" implies custom-tailored. Not technical enough for engineering docs.

### Recommended Naming

For **engineering and documentation**: **Declarative** vs. **Code-Based** — most precise, least ambiguous.

For **product and marketing**: **Managed** vs. **Custom** — familiar AWS pattern, accessible to non-engineers.

For **architecture discussions**: **Platform-Orchestrated** vs. **Self-Orchestrated** — most precise about the core technical distinction (who owns the agent loop).

---

## 3. Should Both Run on the Same Platform?

**Yes.** A unified runtime platform is strongly preferable over two separate systems.

### Why Unify?

1. **Shared infrastructure services.** Identity, memory, networking, observability, and tool access are the same regardless of how the agent loop is implemented. Building these twice would be wasteful and create divergent behavior.

2. **Graduation path.** Developers often start with Approach 1 (declarative) for prototyping, then graduate to Approach 2 (code-based) when they need custom logic. If both run on the same platform, this transition is seamless — same identity, same memory, same tools.

3. **Operational consistency.** Platform operators (SRE, security, compliance) want one set of controls — one IAM model, one VPC integration, one audit log format. Two platforms means two of everything.

4. **The key insight: Approach 1 compiles to Approach 2.** A declarative agent definition (LLM + prompt + tools + memory) can be compiled by the platform into a container that implements the agent loop. Under the hood, both approaches are containers running in microVMs. The only difference is who authored the container — the platform or the developer.

### The Compilation Model

```
┌─────────────────────┐    ┌─────────────────────────────┐
│ Approach 1           │    │ Approach 2                   │
│ Declarative Config   │    │ Container Image              │
│ (LLM + prompt +      │    │ (Any framework,              │
│  tools + memory)     │    │  any language)               │
└──────────┬──────────┘    └──────────────┬──────────────┘
           │  platform compiles            │  developer
           │  to container                 │  builds
           └──────────┬───────────────────┘
                      ▼
            ┌──────────────────┐
            │  AgentRuntime    │  ← Single unified resource
            │  (one IaC type)  │    type for both approaches
            └────────┬─────────┘
                     ▼
          Unified Runtime Platform
          (microVMs, identity, memory,
           networking, tools, observability)
```

---

## 4. How to Define an Agent — Is IaC Involved?

### For Declarative Agents

```yaml
# agent-definition.yaml
type: managed
model: anthropic.claude-sonnet-4-20250514-v1:0
system_prompt: "You are a helpful SRE assistant..."
tools:
  - type: mcp
    gateway_target: "arn:aws:bedrock-agentcore:...:gateway/sre-tools"
  - type: code_interpreter
  - type: custom_function
    handler: "arn:aws:lambda:...:function/get-metrics"
memory:
  strategies:
    - type: semantic
      namespace: "/sre/{actorId}/{sessionId}"
    - type: user_preference
      namespace: "/sre/{actorId}/preferences"
orchestration: react  # or plan-and-execute, chain-of-thought
```

Under the hood, the platform **generates** a container image from this definition. The generated code uses the platform's built-in agent loop with the specified configuration.

### For Code-Based Agents

```yaml
# agent-runtime.yaml
type: custom
artifact:
  container_uri: "123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:v1.2"
  # OR: source_code: ./agent/  (platform builds the image)
```

### IaC

Both approaches should be fully IaC-able. AgentCore already supports CloudFormation (`AWS::BedrockAgentCore::AgentRuntime`), Terraform (`aws_bedrockagentcore_agent_runtime`), and CDK ([`04-infrastructure-as-code/`](../04-infrastructure-as-code/)). The resource model should be the **same** — both approaches create an `AgentRuntime` resource. The only difference is what `AgentRuntimeArtifact` contains:
- Approach 1: A declarative spec that the platform compiles into a container
- Approach 2: A container URI pointing to the developer's image

**Design principle:** Keep one unified resource type. Don't create `ManagedAgent` and `CustomAgent` as separate resource types — that fragments identity, networking, and memory systems.

---

## 5. How to Deploy and Update an Agent

The [session-design-comparison.md](./session-design-comparison.md) reveals a critical tension here: today (Model A), **session = microVM**, so deploying a new version destroys all existing sessions with no migration path.

| Concern | Declarative Agent | Code-Based Agent |
|---|---|---|
| **Update mechanism** | Config change → platform rebuilds and deploys | Developer pushes new image → redeploy |
| **Version strategy** | Automatic (platform manages versions) | Developer-controlled (image tags, qualifiers) |
| **Session continuity** | Platform should migrate sessions (→ [Model B](./session-design-comparison.md): sessions as durable logical entities) | Developer chooses: drain or kill |
| **Rollback** | Platform reverts config version | Developer reverts image tag |

### Why Model B Matters Here

The [Model B proposal](./session-design-comparison.md#1-the-fundamental-difference) — sessions as durable logical entities separate from compute — is essential for declarative agents. When the platform promises "just configure and it works," you can't tell users "sorry, all your conversations were lost because we updated the prompt."

For code-based agents, session durability should be **opt-in** (explicit checkpointing), with ephemeral sessions as the default for backward compatibility.

---

## 6. Agent Identity and Permission Management

From [GUIDE.md](./GUIDE.md), AgentCore already has a solid identity model:
- **Inbound:** IAM SigV4 or Cognito JWT
- **Outbound:** OAuth2 M2M (client credentials) or USER_FEDERATION (3-legged OAuth) via `@requires_access_token`
- **Agent's own identity:** IAM role assumed by `bedrock-agentcore.amazonaws.com`

### Design Principles

1. **One identity model for both approaches.** Whether the agent was defined declaratively or as code, it gets the same IAM execution role. No divergent permission stories.

2. **Least-privilege by default for declarative agents.** Since the platform knows exactly what tools a declarative agent uses, it can **auto-generate** a least-privilege IAM policy. For code-based agents, the developer specifies permissions.

3. **User-delegation identity must be consistent.** The `@requires_access_token` pattern from AgentCore works for both approaches using the same Identity service with the same token vault. For declarative agents, the platform wires this up automatically based on tool definitions. For code-based agents, the developer uses the SDK decorator.

4. **End-user binding is the developer's responsibility** (as [session-lifecycle.md](./session-lifecycle.md#4-does-a-session-belong-to-one-user) correctly notes). The platform provides primitives (`session_id`, `actor_id`, token vault) but doesn't enforce user-session binding.

---

## 7. Where Does the Agent Run? VPC/VNet Integration

From the docs, AgentCore runs agents in Firecracker microVMs with two network modes: **PUBLIC** and **PRIVATE**.

```
┌─────────────────────────────────────────────────────┐
│  PLATFORM CONTROL PLANE (multi-tenant)              │
│  ├── Agent Registry (definitions, versions)         │
│  ├── Identity Service (OAuth, token vault)          │
│  ├── Memory Service (events, vector store)          │
│  ├── Gateway Service (MCP proxy, tool routing)      │
│  └── Orchestration Engine (for declarative only)    │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  DATA PLANE (per-customer isolation)                │
│  ┌─────────────────────────────────────────────┐    │
│  │  MicroVM (Firecracker)                      │    │
│  │  ├── Agent Container (declarative or custom)│    │
│  │  ├── SDK sidecar (identity, memory, tools)  │    │
│  │  └── Session state (ephemeral or durable)   │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Network Modes:                                      │
│  ├── PUBLIC: API Gateway → MicroVM                   │
│  └── PRIVATE: PrivateLink → Customer VPC → MicroVM   │
│       ├── ENI in customer VPC (for DB/service access)│
│       └── VPC endpoints for platform services        │
└──────────────────────────────────────────────────────┘
```

### Key Decisions

- **Same compute model for both approaches.** The microVM doesn't care if the container was auto-generated (declarative) or developer-built (code-based). Identical isolation, scaling, and networking.

- **PRIVATE mode with VPC attachment** is essential for enterprise. The agent needs an ENI in the customer's VPC to reach their RDS, DynamoDB, and internal services.

- **Platform services (Memory, Identity, Gateway) accessed via VPC endpoints** — so even PRIVATE agents can reach them without internet exposure.

---

## 8. How Does the Agent Call Tools? Customer Resources?

This is where the two approaches diverge most.

### For Declarative Agents

- Tools are **declared** in the agent definition
- The platform manages the tool-calling protocol (function calling → execution → result injection)
- Tool types should include:
  - **Lambda functions** — developer's code, in their account
  - **MCP servers** — via [AgentCore Gateway](./GUIDE.md) (MCP proxy)
  - **Platform tools** — Code Interpreter, Browser (managed by the platform)
  - **API endpoints** — customer's HTTP services, with OAuth via Identity service
  - **Knowledge bases** — RAG over customer data

### For Code-Based Agents

- The developer's code calls tools directly (using their framework's tool abstraction)
- The platform provides **optional** tool infrastructure:
  - `boto3` client for Code Interpreter/Browser
  - `MCPClient` for Gateway-proxied tools
  - `@requires_access_token` for OAuth-protected external APIs

### For Customer Resources (DBs, Internal Services)

- **Network access:** Via VPC attachment (PRIVATE mode) — the agent's ENI is in the customer VPC, so it can reach RDS, Elasticache, and internal microservices
- **Authentication:** The agent's IAM role is granted cross-account access via resource policies, or OAuth tokens are fetched via the Identity service
- **No platform intermediary for data-plane calls.** The platform should NOT proxy database queries — the agent talks directly to the customer's DB over the VPC network. The platform's role is to provide the network path (VPC attachment) and credential (IAM role / OAuth token), not to be a data-plane proxy.

### Design Principle

The AgentCore Gateway (MCP proxy) is the right pattern for **tool discovery and invocation routing**, but for high-throughput customer resources (databases, streaming services), direct VPC connectivity is non-negotiable.

---

## 9. Where Is Memory Stored? Customer-Controlled Storage?

From [GUIDE.md](./GUIDE.md) and [session-lifecycle.md](./session-lifecycle.md#6-session-and-agentcore-memory), AgentCore Memory is a managed service with two tiers:
- **Short-term (Events):** Conversation turns, scoped by `(memory_id, actor_id, session_id)`
- **Long-term (Memories):** Auto-extracted via LLM pipeline, stored with embeddings for semantic search

### Memory Placement Options

| Memory Tier | Default (Managed) | Customer-Controlled Option |
|---|---|---|
| **Short-term events** | Platform-managed (e.g. DynamoDB) | Customer's DynamoDB/Aurora in their VPC |
| **Long-term memories** | Platform-managed (e.g. OpenSearch Serverless) | Customer's OpenSearch/pgvector in their VPC |
| **Session state (Model B)** | Platform-managed durable storage | Customer's S3/EFS in their VPC |

### Design Decisions

1. **Default to managed for simplicity** — especially for declarative agents. Most users don't want to manage memory infrastructure.

2. **Offer "bring your own store" for compliance-sensitive customers.** Financial services and healthcare customers may require that all data (including conversation history and extracted memories) stay within their VPC under their encryption keys:
   ```yaml
   memory:
     backend: customer_managed
     short_term:
       type: dynamodb
       table_arn: "arn:aws:dynamodb:...:table/agent-events"
     long_term:
       type: opensearch
       endpoint: "vpc-my-domain.us-west-2.es.amazonaws.com"
   ```

3. **The extraction pipeline is the harder problem.** When using managed memory, the platform runs the LLM extraction pipeline (Events → LLM → Embeddings → Vector Store). When using customer-managed storage, the customer needs to either:
   - Let the platform's extraction pipeline write to their store (via IAM cross-account access)
   - Run their own extraction pipeline (the "self-managed" strategy with SNS → Lambda, as documented in [GUIDE.md](./GUIDE.md))

4. **Encryption with customer-managed keys.** All memory should support encryption with customer-managed KMS keys (CMK), regardless of whether the storage is managed or customer-controlled.

---

## 10. Overall Architecture Recommendation

```
┌──────────────────────────────────────────────────────────────┐
│                    AGENT DEFINITION LAYER                     │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │ Declarative          │    │ Code-Based                   │  │
│  │ (LLM + prompt +      │    │ (Any framework,              │  │
│  │  tools + memory)     │    │  any language)               │  │
│  └──────────┬──────────┘    └──────────────┬──────────────┘  │
│             │  platform compiles            │  developer      │
│             │  to container                 │  builds         │
│             └──────────┬───────────────────┘                 │
│                        ▼                                      │
│              ┌──────────────────┐                             │
│              │  AgentRuntime    │  ← Single unified resource  │
│              │  (one IaC type)  │    for both approaches      │
│              └────────┬─────────┘                             │
└───────────────────────┼──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                 UNIFIED RUNTIME PLATFORM                      │
│                                                              │
│  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌───────────────┐  │
│  │ Identity   │ │ Memory   │ │ Gateway │ │ Observability │  │
│  │ (IAM,OAuth)│ │ (events, │ │ (MCP    │ │ (traces,      │  │
│  │            │ │ vectors) │ │  proxy) │ │  metrics)     │  │
│  └────────────┘ └──────────┘ └─────────┘ └───────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Compute: MicroVMs (Firecracker)                        │  │
│  │ Network: PUBLIC | PRIVATE (VPC PrivateLink)            │  │
│  │ Sessions: Ephemeral (Model A) | Durable (Model B)     │  │
│  │ Scaling: Auto (per-session, per-agent)                 │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. Summary of Design Principles

1. **One runtime, two surfaces.** Declarative agents compile down to containers. Don't build two runtimes.

2. **One identity model.** Both approaches get IAM roles, OAuth token vault, SigV4 auth. No divergence.

3. **Memory is a service, not embedded in compute.** Memory is external to the microVM ([session-lifecycle.md § 6](./session-lifecycle.md#6-session-and-agentcore-memory)). Add customer-managed storage as an option.

4. **Sessions should evolve toward Model B** (durable logical entities from [session-design-comparison.md](./session-design-comparison.md)) — essential for declarative agents and any production agent that can't afford state loss on redeployment.

5. **VPC attachment for customer resources.** Don't proxy database calls through the platform. Give the agent an ENI in the customer's VPC and let it talk directly.

6. **Gateway (MCP) for tool discovery, direct access for data-plane.** The MCP proxy is great for tool catalog management but shouldn't be in the hot path for high-throughput data access.

7. **IaC-first.** Everything should be definable in CloudFormation/Terraform/CDK. The `AgentRuntime` resource type is the single point of truth, regardless of approach.

| Question | Design Answer |
|----------|---------------|
| How to define an agent? | One `AgentRuntime` resource type; artifact is either a declarative spec or a container URI |
| Is IaC involved? | Yes — CloudFormation, Terraform, CDK all support the same resource |
| How to deploy/update? | Config change (declarative) or image push (code-based); both trigger redeployment |
| Identity & permissions? | Unified IAM role + OAuth token vault; auto-least-privilege for declarative |
| Where does it run? | Firecracker microVMs; PUBLIC or PRIVATE (VPC PrivateLink) network modes |
| VPC integration? | ENI in customer VPC for PRIVATE mode; VPC endpoints for platform services |
| Tool calling? | Platform-managed (declarative) or developer-controlled (code-based); Gateway for MCP tools |
| Customer resource access? | Direct VPC connectivity + IAM/OAuth credentials; no platform proxy for data-plane |
| Memory storage? | Managed by default; "bring your own store" option for compliance |
| Customer-controlled memory? | Customer DynamoDB/OpenSearch in their VPC, encrypted with customer KMS keys |

### References

- [AgentCore Technical Guide](./GUIDE.md)
- [AgentCore User-Agent Protocol](./agentcore-user-agent-protocol.md)
- [Session Lifecycle Analysis](./session-lifecycle.md)
- [Session Design Comparison: Model A vs Model B](./session-design-comparison.md)
- [Infrastructure as Code Samples](../04-infrastructure-as-code/)
