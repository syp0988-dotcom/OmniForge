# Workflow

## Processing Flow

The updated workflow uses the **Tool Framework** as the central execution layer:

```
User Input
    │
    ▼
ConversationManager
  - Resolve anaphora / options / slots
  - Rewrite question with context
  - Determine: continue mode or new task
    │
    ├── Continue Mode → Knowledge → Answer → Memory → END
    │
    └── New Task → Router
                        │
                        ├── identity/search → Planner
                        └── other → Knowledge → Planner
                                              │
                                              ▼
                                          Planner
  - LLM-driven plan generation (primary)
  - Rule-based fallback when LLM unavailable
  - Outputs: Plan with explicit {tool, action, goal, input} tasks
    │
    ▼
Tool Executor / Search / Python
  - Dispatches tasks through ToolRegistry
  - Each task routed to correct BaseTool
  - Results collected as ToolResult list
    │
    ▼
AnswerAgent
  - ContextBuilder assembles all sources
  - LLM generates final user-facing answer
    │
    ▼
MemoryAgent
  - Stores conversation history
  - Extracts long-term memories
  - Finalizes session state
    │
    ▼
END
```

## Tool Execution Detail

The `tool_executor` LangGraph node processes all tasks from the Plan:

```
Plan.tasks = [
    {"tool": "filesystem", "action": "mkdir",  "path": "app"},
    {"tool": "filesystem", "action": "write_file", "path": "main.py", "content": "..."},
    {"tool": "search",     "action": "web.search", "query": "FastAPI docs"},
]
    │
    ▼
Executor.execute_batch(tasks)
    │
    ├── FileSystemTool.execute(action="mkdir", path="app")
    ├── FileSystemTool.execute(action="write_file", path="main.py", content="...")
    └── SearchTool.execute(action="web.search", query="FastAPI docs")
    │
    ▼
[ToolResult, ToolResult, ToolResult]
```

## Execution Modes

1. **Normal Mode**: Full flow through Router → Planner → Tools → Answer → Memory
2. **Continue Mode**: Bypasses Router and Planner when session state indicates continuation
   - Triggered by: `SessionState.is_waiting`, option selections, slot filling, "continue" signals
   - Flow: ConversationManager → Knowledge → Answer → Memory
3. **Direct Answer**: When Planner determines no tools needed
   - Bypasses all tool nodes, goes straight to AnswerAgent

## Task Lifecycle

Each `Task` goes through these status transitions:

```
PENDING → READY → RUNNING → COMPLETED
                       └── → FAILED
```

The Executor manages this lifecycle for every task, emitting events at each transition via EventBus.
