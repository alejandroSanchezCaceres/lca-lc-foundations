# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **LangChain Academy learning repository** teaching progressive agent development across three modules: foundational concepts, advanced patterns, and production-ready features. Each module contains Jupyter notebooks for learning and standalone Python files for production deployment.

## Development Environment

### Setup Commands

```bash
# Environment setup
uv sync
uv run python env_utils.py  # Verify environment

# Run notebooks
uv run jupyter lab
```

### Environment Verification

**Always run `env_utils.py` before starting work** to verify:
- Python version (must be >=3.12, <3.14)
- Virtual environment activation
- Package installations with correct versions
- API key configuration in `.env`
- Environment variable conflicts between system and `.env`

### Testing Agents

```bash
# Test with LangGraph dev server (from notebook directory)
cd notebooks/module-X
langgraph dev

# Test from root with specific config
langgraph dev --config notebooks/module-1/langgraph.json
```

## Code Architecture

### Module Structure

```
notebooks/
├── module-1/    # Foundation: models, tools, memory, multimodal
│   ├── *.ipynb  # Learning notebooks
│   ├── 1.5_personal_chef.py  # Production agent
│   └── langgraph.json        # Deployment config
├── module-2/    # Advanced: MCP, state, multi-agent systems
│   ├── resources/
│   │   ├── 2.1_mcp_server.py  # FastMCP server example
│   │   └── Chinook.db         # SQLite database
│   └── *.ipynb
└── module-3/    # Production: middleware, HITL, dynamic agents
    ├── 3.5_email_agent.py     # Production agent
    ├── langgraph.json
    └── agent-chat-ui/         # Next.js chat interface
```

### Agent Creation Pattern

All agents follow this standard pattern:

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()  # Always load environment first

agent = create_agent(
    model="gpt-5-nano",              # String or ChatModel instance
    tools=[...],                     # List of @tool decorated functions
    system_prompt="instructions",    # System prompt string
    checkpointer=InMemorySaver(),    # For conversation memory
    state_schema=CustomState,        # Optional: custom state class
    context_schema=ContextClass,     # Optional: runtime context
    middleware=[...],                # Optional: list of middleware
)

# Standard invocation
response = agent.invoke(
    {"messages": [HumanMessage(content="...")]},
    config={"configurable": {"thread_id": "unique-id"}},
    context=ContextClass(...),  # If using context_schema
)
```

### Tool Definition Pattern

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def tool_name(arg: str, runtime: ToolRuntime) -> str | Command:
    """Docstring becomes the tool description sent to the model."""
    # Access state: runtime.state["key"]
    # Access context: runtime.context.attribute
    # Access tool call ID: runtime.tool_call_id

    # Return string for simple tools
    # Return Command for state updates:
    return Command(
        update={"state_key": value},
        resume=...,  # For HITL
    )
```

## Key Patterns by Module

### Module 1: Foundation

**Core Pattern**: Simple agent with tools and memory
- Tool decoration with `@tool`
- InMemorySaver for conversation history
- Thread IDs for session management
- Tavily integration for web search

### Module 2: Advanced

**Custom State**:
```python
from langgraph.types import AgentState

class CustomState(AgentState):
    custom_field: str
```

**State-Writing Tools**:
```python
@tool
def update_tool(value: str, runtime: ToolRuntime) -> Command:
    return Command(update={
        "custom_field": value,
        "messages": [ToolMessage("Updated", tool_call_id=runtime.tool_call_id)]
    })
```

**Runtime Context**:
```python
from dataclasses import dataclass

@dataclass
class MyContext:
    setting: str = "default"

@tool
def context_tool(runtime: ToolRuntime) -> str:
    return runtime.context.setting

# Usage
agent.invoke({...}, context=MyContext(setting="value"))
```

**Multi-Agent Systems**:
- Create specialized subagents with focused tools
- Coordinator agent delegates to subagents via tools
- Each subagent has own model, tools, and system prompt

**MCP Integration**:
```python
from langchain_mcp_adapters import MultiServerMCPClient

# See notebooks/module-2/resources/2.1_mcp_server.py for FastMCP server
client = MultiServerMCPClient()
await client.add_server({"command": "uvx", "args": ["mcp-server-fetch"]})
tools = client.list_tools()
```

### Module 3: Production

**Middleware Pattern**:
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest

@wrap_model_call
def middleware_name(request: ModelRequest, handler):
    # Modify request (tools, prompt, model, etc.)
    modified = request.override(tools=[...])
    return handler(modified)

agent = create_agent(..., middleware=[middleware_name])
```

**Dynamic Tools**:
```python
@wrap_model_call
def dynamic_tools(request: ModelRequest, handler):
    if request.state.get("condition"):
        tools = [tool_a, tool_b]
    else:
        tools = [tool_c]
    return handler(request.override(tools=tools))
```

**Dynamic Prompts**:
```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    return f"Context: {request.context.info}\n\n{base_prompt}"
```

**Human-in-the-Loop**:
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "safe_tool": False,
        "dangerous_tool": True,  # Interrupt before execution
    }
)

# After interrupt, resume with:
response = agent.invoke(
    Command(resume={
        "decisions": [{"type": "approve"}]  # or "reject" or "edit"
    }),
    config=config
)
```

**Message Management**:
```python
from langchain.agents.middleware import SummarizationMiddleware

middleware = SummarizationMiddleware(
    model="gpt-4o-mini",
    trigger=("tokens", 100),  # Summarize at 100 tokens
    keep=("messages", 1)      # Keep 1 most recent
)
```

## LangGraph Deployment

### Configuration Format

All production agents have a `langgraph.json`:

```json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./script.py:agent"
    },
    "env": "../../.env"
}
```

Points to the agent variable in the Python file for deployment to LangGraph Cloud/Studio.

## Agent Chat UI (Module 3 Bonus)

**Location**: `notebooks/module-3/agent-chat-ui/`

**Tech Stack**: Next.js, React, shadcn/ui, `@langchain/langgraph-sdk`

**Setup**:
```bash
cd notebooks/module-3/agent-chat-ui
pnpm install
pnpm dev  # Runs on localhost:3000
```

**Environment Variables** (`.env`):
```bash
# Local development
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=agent

# Production (with API passthrough)
LANGGRAPH_API_URL=https://my-agent.us.langgraph.app
NEXT_PUBLIC_API_URL=https://my-site.com/api
LANGSMITH_API_KEY=lsv2_...
```

**Features**:
- Streaming responses with `useStream` hook
- Artifact rendering in side panel (via `thread.meta.artifact`)
- Message visibility control with `langsmith:nostream` tag
- Hide messages with `do-not-render-` ID prefix
- Production auth via API passthrough or custom auth

## Important Notes

### Environment Variables

- **Always call `load_dotenv()` first** in notebooks and scripts
- Required keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`
- Optional: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `LANGSMITH_API_KEY`
- LangSmith tracing: Set `LANGSMITH_TRACING=true` (requires valid API key)

### Python Version

**Critical**: Python >=3.12, <3.14 is required. The `env_utils.py` script will warn if wrong version detected.

### Package Manager

This project uses **`uv`** for all operations:
- `uv sync` - Install dependencies
- `uv run python script.py` - Run scripts
- `uv run jupyter lab` - Launch notebooks

`uv` ensures correct Python version and virtual environment automatically.

### Working with Notebooks

- Notebooks are teaching materials with step-by-step explanations
- Corresponding `.py` files are production-ready, deployable code
- Don't modify notebooks unless updating course content
- For agent development, work with `.py` files directly

### Testing Before Commits

When modifying agents:
1. Verify environment: `uv run python env_utils.py`
2. Test agent locally: `langgraph dev --config path/to/langgraph.json`
3. Check notebooks still run: Open in Jupyter and execute cells
4. Verify API keys are not hardcoded

### Common Patterns

**Response Inspection**:
```python
# Final message content
response['messages'][-1].content

# Tool calls
response['messages'][1].tool_calls

# Check for interrupts (HITL)
if '__interrupt__' in response:
    # Handle interrupt
```

**Streaming**:
```python
for chunk in agent.stream({"messages": [...]}, config):
    print(chunk)
```

**Async Invocation**:
```python
result = await agent.ainvoke({"messages": [...]}, config)
```

## Resources

- **MCP Server Example**: `notebooks/module-2/resources/2.1_mcp_server.py`
- **SQL Database**: `notebooks/module-2/resources/Chinook.db` (SQLite)
- **RAG Document**: `notebooks/module-2/resources/acmecorp-employee-handbook.pdf`
- **Bonus Notebooks**: `bonus_rag.ipynb`, `bonus_sql.ipynb` in module-2

## Progressive Learning Path

1. **Module 1**: Build personal chef agent (tools + memory)
2. **Module 2**: Build wedding planner (multi-agent + MCP + state)
3. **Module 3**: Build email agent (HITL + dynamic tools + middleware)
4. **Bonus**: Deploy with Agent Chat UI

Each module builds on previous concepts. Study notebooks for learning, reference `.py` files for production patterns.
