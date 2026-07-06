# Development

## Setup

- Install Python 3.12+
- Install uv
- Run `uv sync`
- Copy `.env.example` to `.env` and fill in API credentials

## Testing

Run all tests with:

```bash
pytest -q
```

Run specific test files:

```bash
pytest tests/test_tool_framework.py -v        # Tool Framework tests
pytest tests/test_conversation_runtime.py -v   # Conversation Runtime tests
pytest tests/test_workflow.py -v              # Workflow integration tests
```

## Adding a New Tool

The Tool Framework follows a **plugin model**. To add a new tool:

1. **Create the tool class** in `agentflow/tools/`:

```python
from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    def execute(self, **kwargs) -> ToolResult:
        # Implement your logic here
        return ToolResult.ok(self.name, "my_action", result={...})

    def capabilities(self) -> list[str]:
        return ["my_tool.action"]

    def validate(self, **kwargs) -> tuple[bool, str]:
        return True, ""
```

2. **Register the tool** in `agentflow/graph/workflow.py` `_build_executor()`:

```python
from agentflow.tools.my_tool import MyTool
executor.registry.register(MyTool())
```

3. **Add capability** in `agentflow/agents/planner/capability.py`:

```python
("my_tool.action", "my_tool", "执行我的自定义操作"),
```

4. **Update Planner prompt** in `agentflow/agents/planner/prompt.py` if the LLM should know about it.

5. **Add routing** in `_route_after_planner()` in `agentflow/graph/workflow.py`.

That's it. No changes to PlannerAgent, Executor, AnswerAgent, or other core modules.

## Plugin Registration API

```python
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.filesystem_tool import FileSystemTool

registry = ToolRegistry()
registry.register(FileSystemTool())

# Execute a task
result = registry.execute_task("filesystem", action="mkdir", path="app/models")

# Or via task dict
result = registry.execute_task_dict({
    "tool": "filesystem",
    "action": "write_file",
    "path": "main.py",
    "content": "print('hello')",
})

# List registered tools
registry.list_tools()
```

## Architecture Principles

1. **Single Responsibility**: Tools execute capabilities; Executor dispatches; Planner plans.
2. **Plugin Model**: Tools register themselves; no core code changes needed.
3. **Unified Interface**: Every tool returns `ToolResult`.
4. **Safety First**: Validation layer blocks dangerous operations before execution.
5. **Backward Compatible**: All existing agents continue to work unchanged.
