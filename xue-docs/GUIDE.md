# Building AI Agents with Amazon Bedrock AgentCore: A Deep-Dive Technical Guide

This guide explains how to build, test, deploy, and invoke AI agents using **Amazon Bedrock AgentCore**, based on the code and patterns in this repository. It covers multiple agent frameworks, agent capabilities (tools, memory, multi-agent orchestration), deployment infrastructure, and the underlying technology.

---

## Table of Contents

1. [What Is Amazon Bedrock AgentCore?](#1-what-is-amazon-bedrock-agentcore)
2. [Writing Agent Code: Multiple Frameworks](#2-writing-agent-code-multiple-frameworks)
3. [Agent Capabilities: From Simple Chat to Powerful Tooling](#3-agent-capabilities-from-simple-chat-to-powerful-tooling)
4. [Agent Memory: Short-Term, Long-Term, and Memory Types](#4-agent-memory-short-term-long-term-and-memory-types)
5. [Gateway: Turning APIs into Agent Tools via MCP](#5-gateway-turning-apis-into-agent-tools-via-mcp)
6. [Identity and Authentication](#6-identity-and-authentication)
7. [Observability and Evaluation](#7-observability-and-evaluation)
8. [Policy: Fine-Grained Access Control with Cedar](#8-policy-fine-grained-access-control-with-cedar)
9. [Testing Your Agent](#9-testing-your-agent)
10. [Deploying to AWS](#10-deploying-to-aws)
11. [Invoking Your Deployed Agent](#11-invoking-your-deployed-agent)
12. [Multi-Agent Orchestration and A2A](#12-multi-agent-orchestration-and-a2a)
13. [Production Blueprints](#13-production-blueprints)
14. [Architecture Summary](#14-architecture-summary)

---

## 1. What Is Amazon Bedrock AgentCore?

Amazon Bedrock AgentCore is a **managed infrastructure layer** for deploying, operating, and scaling AI agents. It is:

- **Framework-agnostic**: Bring your own agent framework (Strands, LangGraph, CrewAI, LlamaIndex, OpenAI Agents, etc.)
- **Model-agnostic**: Use any LLM (Claude, GPT, Gemini, Llama, Amazon Nova, etc.)
- **Infrastructure-focused**: Handles hosting, networking, identity, memory, observability, tools, and policy — so you focus on agent logic

### How It Works Under the Hood

> **[Hypothesis]** Based on the code patterns in this repo, AgentCore appears to work as follows:

AgentCore provides a **serverless container runtime** (similar to AWS Lambda or ECS Fargate, but optimized for agents). When you deploy an agent:

1. Your agent code is packaged into a **Docker container image** and pushed to **Amazon ECR**.
2. AgentCore provisions a **managed compute environment** that runs your container, exposing it via an internal HTTP endpoint (`/invocations`).
3. The runtime handles **session management**, **auto-scaling**, **networking** (public or VPC-private), and **IAM-based authentication**.
4. You invoke your agent via the AWS SDK (`boto3`), REST API, or WebSocket for streaming.

The key abstraction is `BedrockAgentCoreApp` — a lightweight ASGI/HTTP wrapper that standardizes how agent code is hosted:

```python
# From the root README.md (Quick Start)
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()
agent = Agent()

@app.entrypoint
def invoke(payload):
    user_message = payload.get("prompt", "Hello!")
    result = agent(user_message)
    return {"result": result.message}

if __name__ == "__main__":
    app.run()
```

`BedrockAgentCoreApp` wraps your agent logic in an HTTP server that listens on port 8080 at the `/invocations` endpoint. This is the **contract** between your code and the AgentCore runtime — regardless of which framework you use.

> **[Hypothesis]** The runtime likely uses a health-check endpoint and scales instances based on request volume, similar to SageMaker hosting endpoints. The `@app.entrypoint` decorator registers your function as the handler for POST requests to `/invocations`.

---

## 2. Writing Agent Code: Multiple Frameworks

AgentCore supports **10+ agent frameworks**. Each uses `BedrockAgentCoreApp` as the hosting wrapper. Below are code examples for each, with links to their implementations in this repo.

### 2.1 Strands Agents (Primary Framework)

**Strands Agents** is AWS's own open-source agent framework ([strandsagents.com](https://strandsagents.com)). It provides a simple, Pythonic API for building agents with tool use, memory, and multi-agent patterns.

**How Strands works under the hood**: The `Agent` class wraps a model (defaulting to Claude on Bedrock). When you call `agent("some prompt")`, it sends the prompt to the LLM, checks if the response contains tool-use requests, executes the tools, feeds results back to the LLM, and loops until the LLM produces a final text response. This is the classic **ReAct (Reasoning + Acting) loop**.

```python
# 03-integrations/agentic-frameworks/strands-agents/strands_agent_file_system.py
from strands import Agent
from strands_tools import file_read, file_write, editor
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent(tools=[file_read, file_write, editor])
app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    user_message = payload.get("prompt", "No prompt found")
    result = agent(user_message)
    return {"result": result.message}

app.run()
```

**Key concepts:**
- **Tools** are Python functions decorated with `@tool` or imported from `strands_tools`
- **System prompt** customizes agent personality and behavior
- **Hooks** provide lifecycle callbacks (agent initialized, message added, tool called) — critical for memory integration
- **`context`** parameter provides session metadata (session ID, request headers)

📁 **Code**: [`03-integrations/agentic-frameworks/strands-agents/`](./03-integrations/agentic-frameworks/strands-agents/)

### 2.2 LangGraph

**LangGraph** uses a **graph-based execution model**. You define nodes (functions) and edges (transitions), and the agent executes as a state machine. This is powerful for complex workflows where you need explicit control over the agent's decision-making flow.

**How LangGraph works under the hood**: Instead of a simple ReAct loop, LangGraph compiles your nodes and edges into a `StateGraph`. Each node receives the current state, performs an action (e.g., call the LLM, invoke a tool), and returns updated state. Conditional edges determine which node runs next. This gives you explicit control over branching, looping, and error handling.

```python
# 03-integrations/agentic-frameworks/langgraph/langgraph_agent_web_search.py
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize model with Bedrock
llm = init_chat_model(
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_provider="bedrock_converse"
)

# Bind tools to the model
tools = [DuckDuckGoSearchRun()]
llm_with_tools = llm.bind_tools(tools)

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    messages = {"messages": [{"role": "user", "content": payload.get("prompt")}]}
    output = graph.invoke(messages)
    return {"result": output['messages'][-1].content}

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/langgraph/`](./03-integrations/agentic-frameworks/langgraph/)

### 2.3 OpenAI Agents SDK

The **OpenAI Agents SDK** provides agent primitives (Agent, Runner, tools). Despite being an OpenAI framework, it can be deployed on AgentCore to run any model.

```python
# 03-integrations/agentic-frameworks/openai-agents/openai_agents_hello_world.py
from agents import Agent, Runner, WebSearchTool
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent(name="Assistant", tools=[WebSearchTool()])

app = BedrockAgentCoreApp()

@app.entrypoint
async def agent_invocation(payload, context):
    query = payload.get("prompt", "How can I help you today?")
    result = await Runner.run(agent, query)
    return {"result": result.final_output}

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/openai-agents/`](./03-integrations/agentic-frameworks/openai-agents/)

### 2.4 CrewAI

**CrewAI** is designed for **multi-agent collaboration**. You define Agents with specific roles and a Crew that coordinates them.

> **[Hypothesis]** Under the hood, CrewAI orchestrates multiple agent instances in a conversation loop where agents can delegate tasks to each other. Each agent has its own system prompt (role, goal, backstory) and tool set.

📁 **Code**: [`01-tutorials/01-AgentCore-runtime/01-hosting-agent/04-crewai-with-bedrock-model/`](./01-tutorials/01-AgentCore-runtime/01-hosting-agent/04-crewai-with-bedrock-model/)

### 2.5 AutoGen

**AutoGen** (by Microsoft) uses a **conversation-based multi-agent** model. Agents communicate through message passing.

```python
# 03-integrations/agentic-frameworks/autogen/autogen_agent_hello_world.py
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
)

app = BedrockAgentCoreApp()

@app.entrypoint
async def main(payload):
    result = await Console(agent.run_stream(task=payload.get("prompt")))
    return {"result": result.messages[-1].content}

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/autogen/`](./03-integrations/agentic-frameworks/autogen/)

### 2.6 LlamaIndex

**LlamaIndex** excels at **RAG (Retrieval-Augmented Generation)** and document-centric agents.

```python
# 03-integrations/agentic-frameworks/llamaindex/llama_agent_hello_world.py
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=finance_tools + [multiply, add],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant.",
)

app = BedrockAgentCoreApp()

@app.entrypoint
async def main(payload):
    response = await agent.run(payload.get("prompt"))
    return response.response.content

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/llamaindex/`](./03-integrations/agentic-frameworks/llamaindex/)

### 2.7 PydanticAI

**PydanticAI** leverages Pydantic for type-safe agent definitions and structured outputs.

```python
# 03-integrations/agentic-frameworks/pydanticai-agents/pydantic_bedrock_claude.py
from pydantic_ai.agent import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

model = BedrockConverseModel('global.anthropic.claude-haiku-4-5-20251001-v1:0')
agent = Agent(model=model, system_prompt="You're a helpful assistant.")

@agent.tool
def get_current_date(ctx):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

app = BedrockAgentCoreApp()

@app.entrypoint
def pydantic_bedrock_claude_main(payload):
    result = agent.run_sync(payload.get("prompt"))
    return result.output

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/pydanticai-agents/`](./03-integrations/agentic-frameworks/pydanticai-agents/)

### 2.8 Claude SDK

The **Claude SDK** (Anthropic's agent framework) has native support for tool use and code interpretation.

```python
# 03-integrations/agentic-frameworks/claude-agent/claude-sdk/agent.py
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

app = BedrockAgentCoreApp()

@app.entrypoint
async def run_main(payload):
    async for message in main(payload["prompt"], payload["mode"]):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
        yield message  # Streaming response

app.run()
```

📁 **Code**: [`03-integrations/agentic-frameworks/claude-agent/`](./03-integrations/agentic-frameworks/claude-agent/)

### 2.9 Mastra (TypeScript)

**Mastra** is a TypeScript-based agent framework, showing AgentCore isn't limited to Python:

```typescript
// 03-integrations/agentic-frameworks/typescript_mastra/src/index.ts
app.post('/invocations', async (req: Request, res: Response) => {
  const sessionId = req.headers['x-amzn-bedrock-agentcore-runtime-session-id'];
  const { prompt } = req.body;

  const agent = mastra.getAgent('utilityAgent');
  const stream = await agent.stream(prompt, { maxSteps: 5 });

  for await (const chunk of stream.textStream) {
    res.write(chunk);
  }
  res.end();
});
```

📁 **Code**: [`03-integrations/agentic-frameworks/typescript_mastra/`](./03-integrations/agentic-frameworks/typescript_mastra/)

### 2.10 Google ADK / Java ADK

**Google ADK** supports multi-agent patterns with built-in A2A (Agent-to-Agent) protocol support:

```python
# 03-integrations/agentic-frameworks/adk/adk_agent_google_search.py
from google.adk.agents import Agent
from google.adk.tools import google_search

agent = Agent(
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="I can answer questions by searching the internet."
)
```

📁 **Code**: [`03-integrations/agentic-frameworks/adk/`](./03-integrations/agentic-frameworks/adk/), [`03-integrations/agentic-frameworks/java_adk/`](./03-integrations/agentic-frameworks/java_adk/)

### Framework Comparison

| Framework | Language | Execution Model | Best For |
|-----------|----------|----------------|----------|
| **Strands** | Python | ReAct loop | General-purpose, AWS-native |
| **LangGraph** | Python | State graph | Complex workflows with explicit control |
| **OpenAI Agents** | Python | Runner-based | Simple agents, web search |
| **CrewAI** | Python | Role-based crews | Multi-agent collaboration |
| **AutoGen** | Python | Conversation-based | Multi-agent conversations |
| **LlamaIndex** | Python | RAG-focused | Document-heavy agents |
| **PydanticAI** | Python | Type-safe | Structured outputs |
| **Claude SDK** | Python | Streaming | Native Anthropic integration |
| **Mastra** | TypeScript | Stream-based | TypeScript/Node.js teams |
| **Google ADK** | Python/Java | Agent graph | Multi-agent with A2A protocol |

---

## 3. Agent Capabilities: From Simple Chat to Powerful Tooling

### 3.1 Simple Chat (No Tools)

The simplest agent is just an LLM wrapper:

```python
from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent(system_prompt="You are a helpful assistant.")
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    result = agent(payload.get("prompt", "Hello!"))
    return {"result": result.message}

app.run()
```

This agent can only generate text — no external interactions.

### 3.2 Agent with Custom Tools

Tools are the **bridge between the LLM's reasoning and the real world**. When the LLM decides it needs to perform an action (e.g., check the weather), it emits a structured tool-call request. The framework intercepts this, executes the corresponding Python function, and feeds the result back.

**How tool calling works under the hood**: Modern LLMs support "function calling" — a special output format where instead of generating text, the model outputs a JSON object specifying a function name and arguments. The agent framework parses this, runs the function, then sends the result back to the LLM as a "tool result" message. The LLM then generates the final response incorporating the tool's output.

```python
# Custom tool definition with Strands
from strands import Agent, tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    weather_data = {"new york": "72°F, Partly Cloudy", "london": "58°F, Rainy"}
    return weather_data.get(location.lower(), "70°F, Clear")

@tool
def search_database(query: str) -> str:
    """Search the product database."""
    # Connect to DynamoDB, RDS, etc.
    return f"Found 3 results for '{query}'"

agent = Agent(
    tools=[get_weather, search_database],
    system_prompt="You help users with weather and product queries."
)
```

### 3.3 Code Interpreter Tool

AgentCore provides a **managed Code Interpreter** — a sandboxed Python execution environment that agents can use to write and run code, read/write files, and run shell commands.

#### 3.3.1 SDK & Protocol

The Code Interpreter is accessed via the **AWS REST API** using the `boto3` client for the `bedrock-agentcore` service. There is no MCP or gRPC involved at the transport layer — it's standard AWS API calls with SigV4-signed HTTP requests.

**Two levels of abstraction are available:**

| Level | Package | Class/Function | Description |
|-------|---------|---------------|-------------|
| **High-level SDK** | `bedrock-agentcore` | `CodeInterpreter`, `code_session()` | Convenience wrapper with context manager |
| **Low-level boto3** | `boto3` | `client("bedrock-agentcore")` | Direct AWS API calls |

**High-level SDK usage** (from [`01-tutorials/05-AgentCore-tools/01-Agent-Core-code-interpreter/`](./01-tutorials/05-AgentCore-tools/01-Agent-Core-code-interpreter/)):
```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session

# Context manager pattern — auto-starts and auto-stops session
with code_session("us-west-2") as code_client:
    result = code_client.invoke("executeCode", {"code": "print(2+2)", "language": "python"})
```

**Low-level boto3 usage** (from [`03-integrations/agentic-frameworks/claude-agent/claude-with-code-interpreter/code_int_mcp/client.py`](./03-integrations/agentic-frameworks/claude-agent/claude-with-code-interpreter/code_int_mcp/client.py)):
```python
import boto3

client = boto3.client("bedrock-agentcore", region_name="us-west-2")

# Step 1: Start a session (creates the sandbox)
session_response = client.start_code_interpreter_session(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    name="mySession",
    sessionTimeoutSeconds=900,  # 15-minute timeout
)
session_id = session_response["sessionId"]  # e.g., "01K00Z3F8WZ9KBBW4QGRJCVBHH"

# Step 2: Invoke an operation (returns a streaming response)
response = client.invoke_code_interpreter(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    sessionId=session_id,
    name="executeCode",  # Operation name
    arguments={"code": "import pandas as pd; print(pd.__version__)", "language": "python"},
)

# Step 3: Read the streaming response
for event in response["stream"]:
    result = event["result"]
    # result contains: output text, isError flag, structuredContent (stdout, stderr, exitCode)

# Step 4: Stop the session (destroys the sandbox)
client.stop_code_interpreter_session(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    sessionId=session_id,
)
```

#### 3.3.2 Authentication

Authentication is **standard AWS IAM** — the same credential chain used by all AWS services:

1. **No API keys or tokens** to manage for tool access. Your `boto3` client uses the ambient AWS credentials (environment variables, IAM role, `~/.aws/credentials`, or STS tokens).
2. The `session_id` is **not an authentication token** — it's a sandbox identifier that isolates your execution environment.
3. Required IAM permissions: `bedrock-agentcore:StartCodeInterpreterSession`, `bedrock-agentcore:InvokeCodeInterpreter`, `bedrock-agentcore:StopCodeInterpreterSession`.

#### 3.3.3 Session & Sandbox Architecture

```
┌─────────────────┐     SigV4-signed HTTPS     ┌──────────────────────────┐
│  Your Agent Code │ ──────────────────────────→ │ AgentCore API Endpoint   │
│  (boto3 client)  │                             │ bedrock-agentcore service │
└─────────────────┘                             └──────────┬───────────────┘
                                                           │
                                                           ▼
                                                ┌──────────────────────┐
                                                │   Sandbox (Session)  │
                                                │  ┌────────────────┐  │
                                                │  │ Python Runtime │  │
                                                │  │ (pandas, numpy,│  │
                                                │  │  matplotlib...) │  │
                                                │  └────────────────┘  │
                                                │  ┌────────────────┐  │
                                                │  │  File System   │  │
                                                │  │  (read/write)  │  │
                                                │  └────────────────┘  │
                                                │  ┌────────────────┐  │
                                                │  │ Shell (bash)   │  │
                                                │  └────────────────┘  │
                                                └──────────────────────┘
```

> **[Hypothesis]** Each session likely runs in an isolated container/microVM (similar to AWS Lambda's Firecracker). The sandbox has a pre-installed Python environment with common data science libraries. The `clearContext=False` parameter allows variables/state to persist across multiple `invoke_code_interpreter` calls within the same session, suggesting the Python interpreter process stays alive between calls.

**Key concepts:**
- **`codeInterpreterIdentifier`**: Always `"aws.codeinterpreter.v1"` — identifies the managed service version.
- **`sessionId`**: Returned by `start_code_interpreter_session()`. Identifies your isolated sandbox. Multiple agents can each have their own session.
- **`sessionTimeoutSeconds`**: The sandbox auto-terminates after this period of inactivity (default ~15 min).
- **`clearContext`**: When `False` (default), Python variables persist across calls within the same session. When `True`, each call starts fresh.

#### 3.3.4 Available Operations

Five operations are available via `invoke_code_interpreter`:

| Operation | Arguments | What It Does |
|-----------|-----------|-------------|
| `executeCode` | `{"code": str, "language": "python", "clearContext": bool}` | Execute Python code, return stdout/stderr |
| `executeCommand` | `{"command": str}` | Run a shell command (e.g., `ls`, `pip install`) |
| `writeFiles` | `{"content": [{"path": str, "text": str}]}` | Write files to the sandbox filesystem |
| `readFiles` | `{"paths": [str]}` | Read file contents from the sandbox |
| `listFiles` | `{"path": str}` | List directory contents |

**Response structure** (from streaming events):
```json
{
  "sessionId": "01K00Z3F8WZ9KBBW4QGRJCVBHH",
  "id": "...",
  "isError": false,
  "content": [{"type": "text", "text": "4\n"}],
  "structuredContent": {
    "stdout": "4\n",
    "stderr": "",
    "exitCode": 0,
    "executionTime": 0.71
  }
}
```

#### 3.3.5 Wrapping Code Interpreter as an MCP Server

The repo includes an example of wrapping the Code Interpreter boto3 client as an **MCP server** for use with the Claude SDK. This shows how the REST API can be bridged to any protocol:

```python
# 03-integrations/agentic-frameworks/claude-agent/claude-with-code-interpreter/code_int_mcp/server.py
from .client import CodeInterpreterClient
from claude_agent_sdk import tool, create_sdk_mcp_server

client = CodeInterpreterClient()  # boto3 wrapper

@tool("execute_code", "Execute code using Code Interpreter.",
      {"code": str, "language": str, "code_int_session_id": str})
async def execute_code(args):
    result = client.execute_code(args["code"], args.get("language", "python"))
    return {"content": [{"type": "text", "text": result.model_dump_json()}]}

# ... similar wrappers for execute_command, write_files, read_files

code_int_mcp_server = create_sdk_mcp_server(
    name="codeinterpretertools", version="1.0.0",
    tools=[execute_code, execute_command, write_files, read_files],
)
```

This pattern lets you use the Code Interpreter with **any framework that supports MCP** — even though the underlying transport is AWS REST API.

📁 **Code**: [`01-tutorials/05-AgentCore-tools/01-Agent-Core-code-interpreter/`](./01-tutorials/05-AgentCore-tools/01-Agent-Core-code-interpreter/), [`03-integrations/agentic-frameworks/claude-agent/claude-with-code-interpreter/`](./03-integrations/agentic-frameworks/claude-agent/claude-with-code-interpreter/)

---

### 3.4 Browser Tool

The **Browser Tool** gives agents the ability to navigate websites, fill forms, click elements, and extract information — using a **real managed Chromium browser** running in the AWS cloud.

#### 3.4.1 SDK & Protocol

The Browser Tool uses a **two-protocol architecture**:

| Phase | Protocol | Authentication | Purpose |
|-------|----------|---------------|---------|
| **Control plane** | AWS REST API (boto3) | IAM SigV4 | Create/delete browsers, start/stop sessions |
| **Data plane** | **WebSocket (WSS)** with **Chrome DevTools Protocol (CDP)** | SigV4-signed headers | Actually control the browser (navigate, click, extract) |

This is fundamentally different from the Code Interpreter. The Code Interpreter uses REST for everything; the Browser Tool uses REST to set up the session, then **upgrades to a WebSocket connection** that speaks CDP — the same protocol Chrome DevTools uses internally.

**SDK packages:**

| Package | Purpose |
|---------|---------|
| `bedrock-agentcore` | `BrowserClient` class for session lifecycle + WebSocket URL generation |
| `boto3` | Low-level `bedrock-agentcore` and `bedrock-agentcore-control` clients |
| `playwright` | CDP client for browser automation (Microsoft's Playwright library) |
| Optional: `nova-act` | Amazon's LLM-powered browser agent (uses CDP under the hood) |
| Optional: `browser-use` | Open-source LLM browser agent (uses CDP under the hood) |
| Optional: `strands-agents-tools` | Strands framework browser wrapper |

#### 3.4.2 Session Lifecycle (Detailed)

The browser session has **three distinct phases**: Create → Connect → Destroy.

**Phase 1: Create browser + start session (REST API)**

```python
# Low-level boto3 pattern
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/09-browser-with-domain-filtering/verify_domain_filtering.py

import boto3

client = boto3.client("bedrock-agentcore")

# Start a session on an existing browser resource
response = client.start_browser_session(browserIdentifier=BROWSER_ID)
session_id = response["sessionId"]
```

Or using the higher-level SDK:

```python
# High-level SDK pattern
# From: 02-use-cases/market-trends-agent/tools/browser_tool.py

from bedrock_agentcore.tools.browser_client import browser_session

with browser_session("us-east-1") as client:
    ws_url, headers = client.generate_ws_headers()
    # ... use ws_url and headers with Playwright
```

For advanced use cases (recording, custom network config), use the control plane:

```python
# Control plane: create browser with recording
# From: 02-use-cases/enterprise-web-intelligence-agent/strands/browser_tools.py

from bedrock_agentcore._utils.endpoints import get_control_plane_endpoint

control_plane_url = get_control_plane_endpoint(region)
control_client = boto3.client("bedrock-agentcore-control",
    region_name=region, endpoint_url=control_plane_url)

response = control_client.create_browser(
    name="my_browser",
    executionRoleArn=role_arn,
    networkConfiguration={"networkMode": "PUBLIC"},
    recording={
        "enabled": True,
        "s3Location": {"bucket": "my-bucket", "prefix": "recordings/"}
    }
)
browser_id = response["browserId"]
```

**Phase 2: Connect via WebSocket + CDP**

This is the key part. The agent code connects to a **SigV4-signed WebSocket URL** that tunnels Chrome DevTools Protocol:

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/helpers/browser_helper.py

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from urllib.parse import urlparse

# The WebSocket URL format
ws_url = f"wss://bedrock-agentcore.{REGION}.amazonaws.com/browser-streams/{browser_id}/sessions/{session_id}/automation"

# SigV4 signing: same technique as signing any AWS API request
def get_signed_headers(ws_url):
    credentials = boto3.Session().get_credentials()
    https_url = ws_url.replace("wss://", "https://")
    parsed = urlparse(https_url)

    request = AWSRequest(method="GET", url=https_url, headers={"host": parsed.netloc})
    SigV4Auth(credentials, "bedrock-agentcore", REGION).add_auth(request)
    return {k: v for k, v in request.headers.items()}

headers = get_signed_headers(ws_url)

# Now connect Playwright to the remote browser via CDP
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.connect_over_cdp(ws_url, headers=headers)
    page = browser.contexts[0].pages[0]

    # From here, it's standard Playwright API
    await page.goto("https://example.com")
    content = await page.inner_text("body")
    await page.click("button#submit")
    await page.fill("input#search", "query")
```

**Phase 3: Stop session (REST API)**

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/09-browser-with-domain-filtering/verify_domain_filtering.py

client.stop_browser_session(browserIdentifier=BROWSER_ID, sessionId=session_id)
```

#### 3.4.3 Authentication Deep Dive

The Browser Tool's auth is more nuanced than the Code Interpreter because of the WebSocket upgrade:

```
┌──────────────┐   (1) SigV4 REST   ┌──────────────────────┐
│ Agent Code   │ ──────────────────→ │ AgentCore Control    │
│              │  start_browser_     │ Plane (REST API)     │
│              │  session()          │                      │
│              │ ←────────────────── │ Returns: sessionId   │
│              │                     └──────────────────────┘
│              │
│              │   (2) SigV4 WSS     ┌──────────────────────┐
│ (Playwright) │ ──────────────────→ │ AgentCore Browser    │
│              │  connect_over_cdp   │ Stream (WebSocket)   │
│              │  w/ signed headers  │                      │
│              │ ←──────────────────→│ Chromium (via CDP)   │
│              │  bidirectional CDP  │                      │
└──────────────┘                     └──────────────────────┘
```

1. **Control plane calls** (start/stop session) use standard boto3 SigV4 signing — automatic.
2. **WebSocket connection** requires **manual SigV4 signing** of the WebSocket URL. You must:
   - Convert `wss://` to `https://` for signing purposes
   - Create an `AWSRequest` with the URL
   - Sign it with `SigV4Auth(credentials, "bedrock-agentcore", region)`
   - Extract the signed headers (`Authorization`, `X-Amz-Date`, `X-Amz-Security-Token`)
   - Pass these headers to `playwright.chromium.connect_over_cdp(ws_url, headers=headers)`
3. The `bedrock-agentcore` SDK's `BrowserClient.generate_ws_headers()` encapsulates this — it returns the signed `(ws_url, headers)` tuple.

> **Why WebSocket and not REST?** CDP (Chrome DevTools Protocol) is inherently bidirectional — the browser sends events (page loaded, network request completed, console message) back to the client. REST wouldn't work for this. The WebSocket connection is a tunnel for the full CDP protocol, giving your Playwright code direct control over the remote Chromium instance as if it were running locally.

#### 3.4.4 Integration Patterns

The repo shows **four ways** to use the Browser Tool, from lowest to highest abstraction:

**Pattern A: Raw Playwright + CDP** (most control)

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/09-browser-with-domain-filtering/verify_domain_filtering.py

async with async_playwright() as p:
    browser = await p.chromium.connect_over_cdp(ws_url, headers=signed_headers)
    page = browser.contexts[0].pages[0]
    await page.goto("https://example.com")
    content = await page.inner_text("body")
```

**Pattern B: Nova Act** (LLM-driven browser agent by Amazon)

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/01-browser-with-NovaAct/

from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct

with browser_session(region) as client:
    ws_url, headers = client.generate_ws_headers()
    with NovaAct(
        cdp_endpoint_url=ws_url,
        cdp_headers=headers,
        nova_act_api_key=key,
        starting_page="https://example.com"
    ) as nova:
        result = nova.act("Find the price of the first product")  # Natural language!
```

**Pattern C: Browser-Use** (open-source LLM browser agent)

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/02-browser-with-browserUse/

from bedrock_agentcore.tools.browser_client import BrowserClient
from browser_use import Agent, Browser, BrowserProfile

client = BrowserClient(region)
client.start()
ws_url, headers = client.generate_ws_headers()

profile = BrowserProfile(headers=headers, timeout=1500000)
browser_session = Browser(cdp_url=ws_url, browser_profile=profile, keep_alive=True)

agent = Agent(task="Search for AI news", llm=ChatAnthropicBedrock(...), browser_session=browser_session)
await agent.run()
```

**Pattern D: Strands browser tool** (highest abstraction)

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/04-browser-with-Strands/

from strands_tools.browser import AgentCoreBrowser
from strands import Agent

browser = AgentCoreBrowser(region="us-west-2")
agent = Agent(
    tools=[browser.browser],  # Browser exposed as a Strands tool
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
result = agent("Visit example.com and tell me what you see")
```

#### 3.4.5 Advanced Features

**Live View (DCV streaming)**

AgentCore can stream the browser's visual output in real-time using **AWS DCV (NICE DCV)** protocol:

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/05-browser-live-view/

live_view_url = browser_client.generate_live_view_url(expires=300)
# Returns a presigned URL that opens a DCV viewer in a web browser
# Uses DCVjs JavaScript SDK on the frontend
```

This enables a **human-in-the-loop** pattern: a human can watch the agent browse and take over control.

**Domain Filtering (Network Firewall)**

Domain filtering is implemented at the **network level** using AWS Network Firewall, not in the browser itself:

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/09-browser-with-domain-filtering/

# The browser runs inside a VPC with a Network Firewall
# CloudFormation defines allowed/denied domains:
#   AllowedDomains: .example.com, .github.com, .wikipedia.org
#   DeniedDomains: .facebook.com, .twitter.com
# All unlisted domains are blocked by default (default-deny)

# The agent code doesn't need to do anything special —
# the firewall transparently blocks/allows requests:
await page.goto("https://example.com")     # ✅ Allowed
await page.goto("https://facebook.com")    # ❌ Timeout/blocked by firewall
await page.goto("https://randomsite.com")  # ❌ Blocked (default deny)
```

**Web Bot Auth Signing**

For enterprise use cases where the browser needs to authenticate with internal services:

```python
# From: 01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/06-Web-Bot-Auth-Signing/

# Browser created with signing enabled — all outgoing HTTP requests
# get cryptographic signatures automatically:
response = control_client.create_browser(
    name="signed_browser",
    browserSigning={"enabled": True}  # Automatic request signing
)
# Signed headers: Signature-Input, Signature-Agent, Signature
```

#### 3.4.6 Communication Architecture Summary

```
┌───────────────────────────────────────────────────────────────┐
│                     YOUR AGENT CODE                            │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ boto3 client  │  │ BrowserClient│  │ Playwright/Nova/   │  │
│  │ (control)     │  │ (SDK wrapper)│  │ BrowserUse/Strands │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘  │
│         │                  │                     │             │
└─────────┼──────────────────┼─────────────────────┼─────────────┘
          │                  │                     │
          │ (REST/SigV4)     │ (REST/SigV4)        │ (WSS/SigV4 + CDP)
          ▼                  ▼                     ▼
┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐
│ Control Plane   │  │ Session Mgmt │  │ Browser Stream         │
│ create_browser  │  │ start/stop   │  │ (WebSocket endpoint)   │
│ delete_browser  │  │ session      │  │                        │
└─────────────────┘  └──────────────┘  │  ┌──────────────────┐  │
                                       │  │  Chromium (CDP)   │  │
                                       │  │  ┌─────────────┐  │  │
                                       │  │  │ Page/Tab    │  │  │
                                       │  │  │ Navigation  │  │  │
                                       │  │  │ DOM access  │  │  │
                                       │  │  │ Network     │  │  │
                                       │  │  │ Screenshots │  │  │
                                       │  │  └─────────────┘  │  │
                                       │  └──────────────────┘  │
                                       └────────────────────────┘
```

📁 **Code**: [`01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/`](./01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/), [`02-use-cases/market-trends-agent/tools/browser_tool.py`](./02-use-cases/market-trends-agent/tools/browser_tool.py), [`02-use-cases/enterprise-web-intelligence-agent/strands/browser_tools.py`](./02-use-cases/enterprise-web-intelligence-agent/strands/browser_tools.py)

---

## 4. Agent Memory: Short-Term, Long-Term, and Memory Types

AgentCore Memory provides **managed memory infrastructure** so agents can remember context across turns and sessions.

### 4.1 Architecture

> **[Hypothesis]** AgentCore Memory likely uses a combination of:
> - **DynamoDB** or similar for event/conversation storage (fast writes, ordered reads)
> - **Vector database** (possibly Amazon OpenSearch or a custom embedding store) for semantic retrieval
> - A **background extraction pipeline** that processes conversation events, extracts facts/preferences/episodes using an LLM, and stores them as embeddings

### 4.2 Memory Types

AgentCore supports three **memory strategies**, configured when you create a memory resource:

#### Semantic Memory ("What is true?")
Stores **factual information** extracted from conversations as vector embeddings for similarity search.

```python
# Creating memory with semantic strategy
# 02-use-cases/market-trends-agent/deploy.py
strategies = [
    {
        "SEMANTIC": {
            "name": "MarketTrendsSemantic",
            "description": "Stores financial facts, market analysis, investment insights",
            "namespaces": ["market-trends/broker/{actorId}/semantic"],
        }
    },
]

memory = memory_client.create_memory_and_wait(
    name="market-trends-memory",
    description="Market Trends Agent memory",
    strategies=strategies,
    event_expiry_days=90,
)
```

#### User Preference Memory ("What does the user like?")
Captures **user-specific preferences** and settings — automatically extracted from conversations.

```python
# User preference strategy
{
    "USER_PREFERENCE": {
        "name": "BrokerPreferences",
        "description": "Captures broker preferences, risk tolerance, investment styles",
        "namespaces": ["market-trends/broker/{actorId}/preferences"],
    }
}
```

#### Episodic Memory ("What happened?")
Stores **complete interaction episodes** with full temporal context (situation → intent → assessment → outcome).

The key difference: episodic memory captures the **narrative arc** of an interaction, not just individual facts.

📁 **Code**: [`01-tutorials/04-AgentCore-memory/02-long-term-memory/01-single-agent/using-langgraph-agent-hooks/episodic-memory/`](./01-tutorials/04-AgentCore-memory/02-long-term-memory/01-single-agent/using-langgraph-agent-hooks/episodic-memory/)

### 4.3 Short-Term Memory (Within a Session)

Short-term memory maintains conversation context within a single session:

```python
# 02-use-cases/AWS-operations-agent/agentcore-runtime/src/utils/memory_manager.py
def store_conversation_turn(self, user_message: str, assistant_response: str, actor_id: str):
    messages = [
        (user_message, "USER"),
        (assistant_response, "ASSISTANT")
    ]
    self.memory_client.create_event(
        memory_id=self.memory_id,
        actor_id=actor_id,
        session_id=self.session_id,
        messages=messages
    )

def get_conversation_context(self, actor_id: str, max_results: int = 10):
    return self.memory_client.list_events(
        memory_id=self.memory_id,
        actor_id=actor_id,
        session_id=self.session_id,
        max_results=max_results
    )
```

### 4.4 Long-Term Memory (Across Sessions)

Long-term memory persists across sessions using **memory hooks** — lifecycle callbacks that automatically save and retrieve memories:

```python
# 02-use-cases/customer-support-assistant/agent_config/memory_hook_provider.py
class MemoryHook(HookProvider):
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load conversation history when agent starts."""
        recent_turns = self.memory_client.get_last_k_turns(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            k=5,
        )
        event.agent.messages = context_messages

    def on_message_added(self, event: MessageAddedEvent):
        """Retrieve relevant memories and save new ones."""
        # Semantic retrieval
        memories = self.memory_client.retrieve_memories(
            memory_id=self.memory_id,
            namespace=f"support/user/{self.actor_id}/preferences",
            query=user_query,
            top_k=3
        )
        # Save the conversation
        self.memory_client.save_conversation(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=[(message_content, message_role)]
        )
```

**Using hooks with Strands Agent:**

```python
agent = Agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    hooks=[memory_hooks],  # Hooks automatically handle memory I/O
)
```

📁 **Code**: [`01-tutorials/04-AgentCore-memory/`](./01-tutorials/04-AgentCore-memory/)

### 4.5 Memory Comparison

| Aspect | Short-Term | Long-Term (Semantic) | Long-Term (User Pref) | Long-Term (Episodic) |
|--------|-----------|---------------------|----------------------|---------------------|
| **Scope** | Single session | Cross-session | Cross-session | Cross-session |
| **Storage** | Event log | Vector embeddings | Structured preferences | Narrative episodes |
| **Retrieval** | Sequential reads | Similarity search | Direct lookup | Episode-based search |
| **Use case** | Conversation context | "What facts are relevant?" | "What does user prefer?" | "What happened before?" |
| **Consolidation** | N/A | Merges similar facts | Updates existing prefs | Merges related episodes |

---

## 5. Gateway: Turning APIs into Agent Tools via MCP

### 5.1 What Is the Gateway?

AgentCore Gateway **automatically converts existing APIs** (REST, Lambda functions, services) into **MCP (Model Context Protocol) compatible tools** that any agent can use.

### 5.2 What Is MCP?

**MCP (Model Context Protocol)** is an open standard (originated by Anthropic) for connecting LLMs to external tools and data sources. It uses **JSON-RPC 2.0** as the transport protocol.

> **How MCP works under the hood**: An MCP server exposes a set of "tools" — each described with a name, description, and JSON Schema for its input parameters. The MCP client (the agent's framework) first calls `tools/list` to discover available tools, then passes these tool descriptions to the LLM. When the LLM decides to use a tool, the framework calls `tools/call` with the tool name and arguments.

### 5.3 Gateway as MCP Proxy

The Gateway acts as an **MCP proxy** that sits between your agent and your backend services:

```
Agent Framework → MCP Client → AgentCore Gateway (MCP Server) → Your APIs/Lambdas
```

This means you **don't need to write MCP servers** for existing APIs. The Gateway auto-generates MCP tool definitions from OpenAPI specs or Lambda function schemas.

### 5.4 Using the Gateway (MCP Client Code)

```python
# 01-tutorials/02-AgentCore-gateway/12-agents-as-tools-using-mcp/lab_helpers/lab_02/mcp_client.py
class MCPClient:
    def __init__(self, gateway_url: str, access_token: str):
        self.gateway_url = gateway_url
        self.access_token = access_token

    def _mcp_request(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC 2.0 request to Gateway."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params or {}
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.gateway_url, json=payload, headers=headers)
        return response.json()

    def initialize(self, client_name: str = "mcp-client"):
        """Initialize MCP session."""
        return self._mcp_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "1.0.0"}
        })

    def list_tools(self):
        """Discover available tools."""
        return self._mcp_request("tools/list")

    def call_tool(self, name: str, arguments: dict):
        """Invoke a tool."""
        return self._mcp_request("tools/call", {"name": name, "arguments": arguments})
```

### 5.5 Hosting Your Own MCP Server on AgentCore

You can also deploy custom MCP servers on AgentCore Runtime:

```python
# 01-tutorials/02-AgentCore-gateway/12-agents-as-tools-using-mcp/lab_helpers/lab_03/runtime_mcp_agent_code.py
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("SRE Remediation Agent", host="0.0.0.0", stateless_http=True)

@mcp.tool()
def remediation_tool(action: str, target: str) -> str:
    """Execute remediation action on a target system."""
    return f"Executing {action} on {target}"

# Agents on other runtimes can now discover and call this tool via MCP
```

📁 **Code**: [`01-tutorials/02-AgentCore-gateway/`](./01-tutorials/02-AgentCore-gateway/)

---

## 6. Identity and Authentication

AgentCore Identity handles **both inbound authentication** (who is calling the agent?) and **outbound authentication** (how does the agent authenticate with external services?).

### 6.1 Inbound Auth: Who's Calling?

AgentCore uses **IAM-based authentication** by default. Callers need:
- `BedrockAgentCoreFullAccess` managed policy
- SigV4-signed requests via the AWS SDK

For human-facing apps, you can integrate **Amazon Cognito** for OAuth2/OIDC authentication.

### 6.2 Outbound Auth: Agent Accessing External Services

When your agent needs to call external APIs (Google Calendar, Slack, GitHub) on behalf of a user, AgentCore provides **3-legged OAuth**:

```python
# 01-tutorials/03-AgentCore-identity/05-Outbound_Auth_3lo/strands_claude_google_3lo.py
from bedrock_agentcore.identity.auth import requires_access_token

@tool(name="Get_calendar_events_today")
async def get_calendar():
    @requires_access_token(
        provider_name="google-cal-provider",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"]
    )
    async def _inner(access_token):
        credentials = Credentials(token=access_token)
        service = build("calendar", "v3", credentials=credentials)
        return service.events().list(
            calendarId="primary",
            timeMin=start_of_day,
            timeMax=end_of_day
        ).execute()

    return await _inner()
```

> **[Hypothesis]** The `@requires_access_token` decorator likely triggers an OAuth flow where: (1) AgentCore stores the provider configuration (client ID, secret, auth URL), (2) on first use, the user is redirected to authenticate with the external service, (3) tokens are stored securely and refreshed automatically.

📁 **Code**: [`01-tutorials/03-AgentCore-identity/`](./01-tutorials/03-AgentCore-identity/)

---

## 7. Observability and Evaluation

### 7.1 Observability

AgentCore uses **OpenTelemetry** for distributed tracing, with automatic integration with CloudWatch:

```hcl
# 04-infrastructure-as-code/terraform/end-to-end-weather-agent/observability.tf
resource "aws_cloudwatch_log_group" "agent_runtime_logs" {
  name              = "/aws/vendedlogs/bedrock-agentcore/${agent_runtime_id}"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_delivery_source" "logs" {
  name         = "${agent_id}-logs-source"
  log_type     = "APPLICATION_LOGS"
  resource_arn = aws_bedrockagentcore_agent_runtime.weather_agent.agent_runtime_arn
}
```

Environment variables for OpenTelemetry integration:

```bash
OTEL_PYTHON_DISTRO=aws_distro
OTEL_PYTHON_CONFIGURATOR=aws_configurator
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
AGENT_OBSERVABILITY_ENABLED=true
```

Third-party integrations include: **Dynatrace**, **Langfuse**, **OpenLit**, **Braintrust**.

📁 **Code**: [`01-tutorials/06-AgentCore-observability/`](./01-tutorials/06-AgentCore-observability/)

### 7.2 Evaluation

AgentCore provides **13 built-in evaluators** across 4 categories:

| Category | Evaluators | What They Measure |
|----------|-----------|-------------------|
| **Response Quality** | Correctness, Helpfulness, Fluency | Is the answer right, useful, well-written? |
| **Task Completion** | Goal success | Did the agent achieve the user's intent? |
| **Tool Level** | Tool selection, Parameter accuracy | Did the agent pick the right tool with correct inputs? |
| **Safety** | Harmful content detection | Is the output safe and appropriate? |

Evaluations can run:
- **On-demand**: Synchronous evaluation of individual traces during development
- **Online**: Automatic sampling and evaluation in production with CloudWatch dashboards

You can also build **custom evaluators** and **simulation-based tests** that generate synthetic interactions.

📁 **Code**: [`01-tutorials/07-AgentCore-evaluations/`](./01-tutorials/07-AgentCore-evaluations/)

---

## 8. Policy: Fine-Grained Access Control with Cedar

AgentCore Policy uses **Cedar** (an AWS-developed open-source policy language) to enforce fine-grained access control on agent actions.

### How It Works

```
Agent wants to call tool → Gateway intercepts → Policy Engine evaluates Cedar policy → ALLOW/DENY
```

Example Cedar policy that limits insurance coverage amounts:

```cedar
permit(
  principal,
  action == AgentCore::Action::"ApplicationToolTarget___create_application",
  resource == AgentCore::Gateway::"<gateway-arn>"
) when {
  context.input.coverage_amount <= 1000000
};
```

This means: "Allow any principal to create an insurance application via the Gateway, but only if the coverage amount is ≤ $1,000,000."

📁 **Code**: [`01-tutorials/08-AgentCore-policy/`](./01-tutorials/08-AgentCore-policy/)

---

## 9. Testing Your Agent

### 9.1 Local Testing

Every agent can be tested locally before deployment:

```bash
# Start your agent
python my_agent.py

# Test it (in another terminal)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'
```

This works because `BedrockAgentCoreApp.run()` starts a local HTTP server on port 8080.

### 9.2 Unit Tests

```python
# 04-infrastructure-as-code/terraform/basic-runtime/test_basic_agent.py
import boto3
import json

def test_agent(client, agent_arn, prompt):
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        qualifier="DEFAULT",
        payload=json.dumps({"prompt": prompt})
    )
    response_body = response["response"].read().decode("utf-8")
    result = json.loads(response_body)
    assert "result" in result or "response" in result
    return result
```

### 9.3 Evaluation-Based Testing

Use AgentCore Evaluations for systematic testing:

1. **Ground-truth evaluation**: Compare agent outputs against known-correct answers
2. **Simulation-based testing**: Generate synthetic user interactions and evaluate agent behavior
3. **Custom evaluators**: Define your own quality metrics

📁 **Code**: [`01-tutorials/07-AgentCore-evaluations/02-running-evaluations/`](./01-tutorials/07-AgentCore-evaluations/02-running-evaluations/)

### 9.4 End-to-End Testing

The E2E tutorial walks through testing a complete agent lifecycle:

📁 **Code**: [`01-tutorials/09-AgentCore-E2E/`](./01-tutorials/09-AgentCore-E2E/)

---

## 10. Deploying to AWS

### 10.1 CLI Deployment (Quickest)

```bash
pip install bedrock-agentcore strands-agents bedrock-agentcore-starter-toolkit

# Configure your agent
agentcore configure -e my_agent.py

# Deploy (auto-creates ECR repo, builds Docker image, creates runtime)
agentcore launch

# Test
agentcore invoke '{"prompt": "tell me a joke"}'
```

> **[Hypothesis]** Under the hood, `agentcore launch` likely: (1) Builds a Docker image from your agent code using a base image, (2) Pushes the image to ECR, (3) Creates an `AgentRuntime` resource via the Bedrock AgentCore API, (4) Configures IAM roles, networking, and environment variables. The runtime ID and ARN are stored locally for subsequent invocations.

### 10.2 CloudFormation

```yaml
# 04-infrastructure-as-code/cloudformation/basic-runtime/template.yaml
AWSTemplateFormatVersion: "2010-09-09"
Parameters:
  AgentName:
    Type: String
    Default: "BasicAgent"
  NetworkMode:
    Type: String
    Default: "PUBLIC"
    AllowedValues: ["PUBLIC", "PRIVATE"]
Resources:
  ECRRepository:
    Type: AWS::ECR::Repository
    # ...
  AgentRuntime:
    Type: AWS::BedrockAgentCore::AgentRuntime
    Properties:
      AgentRuntimeName: !Ref AgentName
      RoleArn: !GetAtt ExecutionRole.Arn
      NetworkConfiguration:
        NetworkMode: !Ref NetworkMode
      AgentRuntimeArtifact:
        ContainerConfiguration:
          ContainerUri: !Sub "${ECRRepository.RepositoryUri}:latest"
```

📁 **Code**: [`04-infrastructure-as-code/cloudformation/`](./04-infrastructure-as-code/cloudformation/)

### 10.3 Terraform

```hcl
# 04-infrastructure-as-code/terraform/basic-runtime/main.tf
resource "aws_bedrockagentcore_agent_runtime" "basic_agent" {
  agent_runtime_name = var.agent_name
  role_arn           = aws_iam_role.agent_execution.arn

  agent_runtime_artifact {
    container_configuration {
      container_uri = "${aws_ecr_repository.agent_ecr.repository_url}:${var.image_tag}"
    }
  }

  network_configuration {
    network_mode = var.network_mode  # "PUBLIC" or "PRIVATE"
  }

  environment_variables = {
    AWS_REGION = var.aws_region
  }
}
```

📁 **Code**: [`04-infrastructure-as-code/terraform/`](./04-infrastructure-as-code/terraform/)

### 10.4 AWS CDK (Python)

```python
# 04-infrastructure-as-code/cdk/python/basic-runtime/basic_runtime_stack.py
from aws_cdk import Stack
import aws_cdk.aws_ecr as ecr
import aws_cdk.aws_iam as iam

class BasicRuntimeStack(Stack):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ECR repository for agent container image
        repo = ecr.Repository(self, "AgentRepo",
            repository_name="basic-agent"
        )

        # IAM execution role
        role = iam.Role(self, "AgentRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess")
            ]
        )

        # AgentRuntime resource (via custom resource or L1 construct)
        # ...
```

📁 **Code**: [`04-infrastructure-as-code/cdk/`](./04-infrastructure-as-code/cdk/)

### 10.5 Docker Patterns

**Python agent:**
```dockerfile
# Typical pattern from various Dockerfiles in the repo
FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENV HOST="0.0.0.0"
ENV PORT=8080
EXPOSE 8080
CMD ["python", "agent.py"]
```

**TypeScript MCP server (multi-stage):**
```dockerfile
# 01-tutorials/01-AgentCore-runtime/04-hosting-ts-MCP-server/Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install && npm install typescript -g
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 8000
CMD ["node", "dist/index.js"]
```

---

## 11. Invoking Your Deployed Agent

### 11.1 AWS SDK (boto3)

```python
# Standard invocation pattern
import boto3
import json
import os

region = os.environ.get("AWS_REGION", "us-east-1")
client = boto3.client("bedrock-agentcore", region_name=region)

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:<region>:<account-id>:runtime/<agent-id>",
    qualifier="DEFAULT",
    payload=json.dumps({"prompt": "What's the weather in NYC?"})
)

# Read response
result = json.loads(response["response"].read().decode("utf-8"))
print(result)
```

### 11.2 Streaming Invocation

```python
# 02-use-cases/cost-optimization-agent/test_agentcore_runtime.py
response = client.invoke_agent_runtime(
    agentRuntimeArn=runtime_arn,
    qualifier="DEFAULT",
    payload=json.dumps({"prompt": query})
)

# Stream response line by line
for line in response["response"].iter_lines(chunk_size=1):
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            data = line_str[6:]  # Remove SSE "data: " prefix
            print(data, end="", flush=True)
```

### 11.3 CLI Invocation

```bash
agentcore invoke '{"prompt": "tell me a joke"}'
```

### 11.4 WebSocket (Bi-directional Streaming)

For real-time, bi-directional communication (e.g., voice agents):

📁 **Code**: [`01-tutorials/01-AgentCore-runtime/06-bi-directional-streaming/`](./01-tutorials/01-AgentCore-runtime/06-bi-directional-streaming/)

---

## 12. Multi-Agent Orchestration and A2A

### 12.1 Supervisor Pattern (Same Runtime)

Multiple agents in a single runtime, with one **supervisor** routing to **sub-agents** exposed as tools:

```python
# Pattern from 05-blueprints/shopping-concierge-agent/
from strands import Agent, tool

# Sub-agent tools
@tool
def ask_shopping_agent(query: str) -> str:
    """Route to shopping specialist."""
    response = shopping_agent(query)
    return response.message

@tool
def ask_cart_agent(query: str) -> str:
    """Route to cart management specialist."""
    response = cart_agent(query)
    return response.message

# Supervisor agent
supervisor = Agent(
    name="supervisor",
    tools=[ask_shopping_agent, ask_cart_agent],
    system_prompt="""You coordinate between specialists:
    - shopping_agent: Product search and recommendations
    - cart_agent: Cart operations and checkout
    Route each request to the appropriate specialist."""
)
```

### 12.2 A2A Protocol (Cross-Runtime)

**Agent-to-Agent (A2A)** is a protocol for agents running on **separate runtimes** to communicate. It uses:
- **Agent Cards** (`.well-known/agent-card.json`) for service discovery
- **JSON-RPC 2.0** for communication
- **OAuth 2.0 M2M tokens** for authentication

```python
# 02-use-cases/A2A-multi-agent-incident-response/host_adk_agent/agent.py
from a2a.client import ClientConfig, ClientFactory
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Remote agents discovered via agent cards
monitor_agent = RemoteA2aAgent(
    name="monitor_agent",
    description="Handles monitoring tasks",
    agent_card=monitor_agent_card_url,
    a2a_client_factory=create_client_factory(provider_name="monitor-provider"),
)

websearch_agent = RemoteA2aAgent(
    name="websearch_agent",
    description="Web search for finding solutions",
    agent_card=websearch_agent_card_url,
    a2a_client_factory=create_client_factory(provider_name="search-provider"),
)

# Root orchestrator delegates to remote agents
root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="Coordinate incident response across monitoring and search agents.",
    sub_agents=[monitor_agent, websearch_agent],
)
```

**How A2A works under the hood**:
1. Each agent runtime exposes an **agent card** at `/.well-known/agent-card.json` describing its capabilities
2. The host agent **discovers** remote agents via their card URLs
3. Authentication uses **M2M OAuth tokens** (machine-to-machine) via AgentCore Identity
4. Communication uses **JSON-RPC 2.0** over HTTP
5. Each sub-agent runs on its own runtime, potentially with different frameworks and models

📁 **Code**: [`02-use-cases/A2A-multi-agent-incident-response/`](./02-use-cases/A2A-multi-agent-incident-response/)

---

## 13. Production Blueprints

The `05-blueprints/` directory contains **complete, deployment-ready applications**:

| Blueprint | Description | Key Features |
|-----------|-------------|-------------|
| **Shopping Concierge** | E-commerce assistant | Supervisor + shopping/cart sub-agents, Visa payment, React UI |
| **Travel Concierge** | Travel booking agent | Itinerary management, hotel/flight search, payment processing |
| **Customer Support** | Multi-component support | Knowledge base, ticket routing, session management |
| **End-to-End Customer Service** | Full CX platform | Backend + React/Amplify frontend, AppSync GraphQL |

### Architecture Pattern (Shopping Concierge)

```
┌────────────────────────────────────────────┐
│            React Frontend (Amplify)         │
│  Chat UI │ Cart Panel │ Wishlist │ Profile  │
└──────────────────┬─────────────────────────┘
                   │ WebSocket/HTTP
                   ▼
┌──────────────────────────────────────────────┐
│          AgentCore Runtime                    │
│  ┌────────────────────────────────────────┐  │
│  │         Supervisor Agent               │  │
│  │  (Routes to sub-agents as tools)       │  │
│  │  Model: Claude Sonnet 4.5             │  │
│  │  Memory: AgentCore Memory             │  │
│  └──┬──────────────────────────────┬─────┘  │
│     │                              │         │
│  ┌──▼─────────────┐  ┌───────────▼───────┐ │
│  │ Shopping Agent  │  │   Cart Agent      │ │
│  │ (Product search)│  │ (Cart + Payment)  │ │
│  └──┬─────────────┘  └───────┬───────────┘ │
└─────┼─────────────────────────┼─────────────┘
      │ MCP                     │ MCP
      ▼                         ▼
┌──────────────┐  ┌──────────────────────────┐
│ Shopping MCP │  │ Cart MCP   │ Visa Server │
│   Server     │  │  Server    │ (Payments)  │
└──────────────┘  └──────────────────────────┘
```

📁 **Code**: [`05-blueprints/`](./05-blueprints/)

---

## 14. Architecture Summary

### The Complete AgentCore Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR AGENT CODE                           │
│  (Strands / LangGraph / CrewAI / OpenAI / Any Framework)    │
├─────────────────────────────────────────────────────────────┤
│                BedrockAgentCoreApp                           │
│  (HTTP wrapper: @app.entrypoint → /invocations)             │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Runtime  │ Gateway  │ Memory   │ Identity │ Observability   │
│          │          │          │          │                 │
│ Container│ API→MCP  │ Short &  │ IAM +    │ OpenTelemetry   │
│ hosting, │ proxy,   │ Long     │ OAuth2 + │ + CloudWatch    │
│ scaling, │ tool     │ term,    │ Cognito  │ + 3rd party     │
│ sessions │ discovery│ semantic │          │                 │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                     AWS Infrastructure                       │
│  ECR │ IAM │ VPC │ DynamoDB │ CloudWatch │ Bedrock (LLMs)  │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Framework-agnostic**: Your agent code stays the same regardless of where it runs. The `BedrockAgentCoreApp` is the only AgentCore-specific code in your agent.

2. **Container-based**: Agents are Docker containers pushed to ECR. This gives you full control over dependencies and runtime environments.

3. **MCP-first tooling**: Tools are accessed via the Model Context Protocol, whether through the Gateway (auto-converted APIs) or custom MCP servers.

4. **Managed memory**: Memory is a separate service, not embedded in your agent. This enables memory sharing across agents and sessions.

5. **Policy-enforced actions**: Cedar policies govern what agents can do, providing governance and compliance guardrails.

### Getting Started Path

1. **Start here**: [Quick Start in README.md](./README.md#quick-start---amazon-bedrock-agentcore-runtime) — deploy a simple Strands agent
2. **Learn fundamentals**: [`01-tutorials/`](./01-tutorials/) — hands-on notebooks for each component
3. **See real examples**: [`02-use-cases/`](./02-use-cases/) — 24 production use cases
4. **Try frameworks**: [`03-integrations/`](./03-integrations/) — 10+ framework integrations
5. **Deploy with IaC**: [`04-infrastructure-as-code/`](./04-infrastructure-as-code/) — CloudFormation, Terraform, CDK
6. **Build full apps**: [`05-blueprints/`](./05-blueprints/) — complete reference architectures
