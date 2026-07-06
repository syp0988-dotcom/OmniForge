"""ProjectStructurePlanner — generates complete project directory trees.

This agent analyses user requirements and produces a structured project
structure (directory tree + file templates).  The output is a list of
FileSystemTool-compatible task dicts that the Executor can process.

Usage flow::

    1. User says "Create a FastAPI e-commerce project"
    2. ProjectStructurePlanner analyses the request and generates a tree
    3. Returns a list of task dicts::
        [
            {"tool": "filesystem", "action": "mkdir", "path": "project/app/api"},
            {"tool": "filesystem", "action": "mkdir", "path": "project/app/models"},
            {"tool": "filesystem", "action": "write_file", "path": "project/main.py",
             "content": "..."},
            ...
        ]
    4. Executor processes all tasks in order
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("project_structure_planner")

# Common project templates for quick matching when LLM is unavailable.
_PROJECT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "fastapi": [
        {"tool": "filesystem", "action": "mkdir", "goal": "创建项目根目录",
         "input": {"path": "app"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建 API 路由目录",
         "input": {"path": "app/api"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建数据模型目录",
         "input": {"path": "app/models"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建业务逻辑目录",
         "input": {"path": "app/services"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建配置目录",
         "input": {"path": "app/config"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建核心工具目录",
         "input": {"path": "app/core"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建测试目录",
         "input": {"path": "tests"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建文档目录",
         "input": {"path": "docs"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建主入口文件",
         "input": {"path": "app/main.py",
                   "content": "from fastapi import FastAPI\n\napp = FastAPI(title=\"My API\")\n\n@app.get(\"/\")\nasync def root():\n    return {\"message\": \"Hello World\"}\n"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建配置文件",
         "input": {"path": "app/config/settings.py",
                   "content": "from pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    app_name: str = \"My API\"\n    debug: bool = True\n\nsettings = Settings()\n"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建 requirements.txt",
         "input": {"path": "requirements.txt",
                   "content": "fastapi>=0.111.0\nuvicorn[standard]>=0.30.0\npydantic>=2.7.0\n"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建 README",
         "input": {"path": "README.md", "content": "# My API\n\nFastAPI 项目\n"}},
    ],
    "flask": [
        {"tool": "filesystem", "action": "mkdir", "goal": "创建项目根目录",
         "input": {"path": "app"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建路由目录",
         "input": {"path": "app/routes"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建模型目录",
         "input": {"path": "app/models"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建模板目录",
         "input": {"path": "app/templates"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建静态文件目录",
         "input": {"path": "app/static"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建主入口文件",
         "input": {"path": "app/main.py",
                   "content": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建 requirements.txt",
         "input": {"path": "requirements.txt",
                   "content": "flask>=3.0.0\n"}},
    ],
    "python": [
        {"tool": "filesystem", "action": "mkdir", "goal": "创建源码目录",
         "input": {"path": "src"}},
        {"tool": "filesystem", "action": "mkdir", "goal": "创建测试目录",
         "input": {"path": "tests"}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建包初始化文件",
         "input": {"path": "src/__init__.py", "content": ""}},
        {"tool": "filesystem", "action": "write_file", "goal": "创建主模块",
         "input": {"path": "src/main.py", "content": "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"}},
    ],
}

_LLM_SYSTEM_PROMPT = """你是一个项目结构规划器。根据用户的需求，生成一个完整的项目目录树和必要的文件内容。

输出格式为 JSON，包含一个 tasks 数组，每个 task 描述一个文件系统操作：

```json
{
    "project_name": "项目名称",
    "tasks": [
        {
            "tool": "filesystem",
            "action": "mkdir",
            "goal": "创建目录说明",
            "input": {"path": "目录路径"}
        },
        {
            "tool": "filesystem",
            "action": "write_file",
            "goal": "创建文件说明",
            "input": {"path": "文件路径", "content": "文件内容"}
        }
    ]
}
```

要求：
1. 目录和文件路径使用相对路径（不要以 / 开头）
2. 文件内容要完整、可用，包含必要的导入和配置
3. 按顺序排列：先创建目录，再创建文件
4. 包含必要的基础文件：README.md、requirements.txt（如适用）
5. 包含完整的项目结构，从根目录开始"""


class ProjectStructurePlanner(AgentProtocol):
    """Generate project directory structures from user requirements.

    Uses LLM to generate the structure, with template-based fallback for
    common project types (FastAPI, Flask, Python package).
    """

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate project structure tasks and store in state.

        Sets:
            state["project_structure"] → list of task dicts
        """
        question = str(state.get("question", ""))
        category = str(state.get("category", ""))

        # Try LLM first
        tasks = self._llm_generate(question)

        # Fallback to template matching
        if tasks is None:
            tasks = self._template_match(question)

        if tasks is None:
            tasks = []

        state["project_structure"] = tasks
        logger.info(
            "ProjectStructurePlanner: generated %d tasks (category=%s)",
            len(tasks), category,
        )
        return state

    def _llm_generate(self, question: str) -> list[dict[str, Any]] | None:
        """Use LLM to generate project structure from requirements."""
        if not question:
            return None
        try:
            messages = [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"请为以下需求生成项目结构：\n{question}"},
            ]
            raw = self._llm.complete(messages=messages)
            if not raw or not raw.strip():
                return None

            parsed = self._parse_json(raw)
            if parsed is None:
                return None

            tasks = parsed.get("tasks", [])
            if not isinstance(tasks, list):
                return None
            return tasks
        except Exception as exc:
            logger.warning("LLM project structure generation failed: %s", exc)
            return None

    @staticmethod
    def _template_match(question: str) -> list[dict[str, Any]] | None:
        """Match common project types by keyword.

        Returns template tasks or None if no match.
        """
        q = question.lower()
        if any(k in q for k in ("fastapi", "fast api")):
            return _PROJECT_TEMPLATES["fastapi"]
        if "flask" in q:
            return _PROJECT_TEMPLATES["flask"]
        if any(k in q for k in ("python package", "python 包", "python 库")):
            return _PROJECT_TEMPLATES["python"]
        return None

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from LLM output."""
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        for marker in ("```json", "```JSON", "```"):
            start = text.find(marker)
            if start == -1:
                continue
            content = text[start + len(marker):]
            end = content.rfind("```")
            if end != -1:
                content = content[:end]
            content = content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue
        return None
