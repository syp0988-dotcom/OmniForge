# Architecture

## Overview

The system follows a layered, plugin-based architecture built around a unified **Tool Framework** that abstracts all execution capabilities behind a common interface.

```
User
  │
  ▼
ConversationManager    ← context resolution, session state, rewrite
  │
  ▼
Router                 ← rule-based intent classification
  │
  ▼
Planner                ← LLM-first, rule-fallback plan generation
  │
  ▼
Tool Executor          ← central dispatch for all tool tasks
  │
  ▼
Tool Registry          ← plugin registration, validation, logging
  │
  ├── FileSystemTool   ← safe file & directory operations
  ├── SearchTool       ← web search (DuckDuckGo / Tavily)
  ├── PythonTool       ← sandboxed code execution
  ├── GitTool          ← git repository operations
  ├── BrowserTool      ← web automation (interface)
  ├── DatabaseTool     ← structured data queries (interface)
  └── MCPTool          ← Model Context Protocol (interface)
```

## Layers

### 1. API Layer
FastAPI REST endpoints for chat (sync + streaming), file upload, knowledge base, sessions, models, memory, and tool introspection.

### 2. Graph Layer (LangGraph)
Orchestrates agent nodes as a `StateGraph`. Key nodes:
  - `conversation_manager` — context resolution, session state
  - `router` — intent classification
  - `planner` — task planning
  - `search` / `python` / `tool_executor` — tool execution
  - `answer` — final LLM response generation
  - `memory` — conversation history and long-term memory

### 3. Agent Layer
- **ConversationManager**: Resolves anaphora, options, slots; determines continue mode vs new task.
- **QueryRouterAgent**: Regex-based intent classification (search, coding, identity, etc.).
- **PlannerAgent**: LLM-driven plan generation with rule-based fallback. Outputs explicit `{tool, action, goal, input}` tasks.
- **ProjectStructurePlanner**: Generates project directory trees from user requirements.
- **AnswerAgent**: Synthesises final answer from all context sources.
- **MemoryAgent**: Maintains conversation history and long-term memory.

### 4. Tool Framework Layer (NEW)
The Tool Framework is a plugin-based execution layer:

```
BaseTool (ABC)
  ├── execute(**kwargs) → ToolResult       ← required
  ├── validate(**kwargs) → (bool, str)     ← optional safety checks
  ├── capabilities() → list[str]           ← semantic capability names
  └── metadata() → dict                    ← introspection data

ToolRegistry
  ├── register(tool)                       ← plugin registration
  ├── unregister(name)                     ← plugin removal
  ├── execute_task(tool, action, **kwargs) → ToolResult
  └── execute_batch(tasks) → [ToolResult]

Executor
  ├── uses ToolRegistry for dispatch
  ├── manages Task lifecycle (PENDING → RUNNING → COMPLETED/FAILED)
  └── emits structured events via EventBus
```

### 5. Infrastructure Layer
- **SQLiteStore**: Persistence for sessions, messages, knowledge base, model configs.
- **LLMService**: OpenAI-compatible client with retry and fallback.
- **SearchService**: Business logic and result normalisation for search.
- **LongTermMemory**: Cross-session fact extraction and recall.
- **KnowledgeStore**: Document ingestion, TF-IDF indexing, chunk search.

## Tool Framework Design

### Unified Return Format
Every tool invocation returns a `ToolResult`:

```json
{
  "success": true,
  "tool": "filesystem",
  "action": "read_file",
  "result": {"content": "...", "size": 123},
  "message": "File read successfully",
  "duration": 0.023,
  "error": null
}
```

### Plugin Registration
New tools are added without modifying any core code:

```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    def execute(self, **kwargs) -> ToolResult:
        ...

# Register it
registry.register(MyTool())

# Or via Executor
executor.registry.register(MyTool())
```

### Safety & Validation
- All tools implement `validate()` for pre-execution parameter checks.
- FileSystemTool restricts all operations to the configured workspace.
- Path traversal, absolute paths outside workspace, and dangerous system directories are rejected.
- Agent source code directories are read-only; write/delete operations are blocked.

### Logging
Every tool call through the Registry is logged with:
- Timestamp and tool name
- Action and parameters (truncated for readability)
- Duration in seconds
- Success/failure status
- Error details on failure

## Workflow Flow

```
conversation_manager
  ├── continue mode → knowledge → answer → memory → END
  └── new task → router → knowledge (if needed) → planner
                                                    │
                              ┌─────────────────────┤
                              ▼                     ▼
                        tool_executor          search / python
                              │                     │
                              └─────────┬───────────┘
                                        ▼
                                     answer
                                        │
                                        ▼
                                     memory → END
```

## Adding a New Tool

1. Create a new file in `agentflow/tools/` that subclasses `BaseTool`.
2. Implement `execute()` (required) and optionally `validate()`, `capabilities()`, `metadata()`.
3. Register it in `agentflow/graph/workflow.py` `_build_executor()`.
4. Add the capability in `agentflow/agents/planner/capability.py`.
5. Update the Planner prompt in `agentflow/agents/planner/prompt.py`.
6. Add routing in `_route_after_planner()` in `workflow.py`.

No changes to PlannerAgent, Executor, or other agents are needed.
