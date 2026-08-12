"""ProjectTemplate — standard project structure templates for Task Queue planning.

Each template defines the initial task queue for a common project type.
The Planner uses the template to seed the queue, then dynamically adds
and adjusts tasks based on workspace state.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentflow.graph.task import Task, TaskStatus


# ── Template definitions ─────────────────────────────────────────────

TEMPLATES: dict[str, dict[str, Any]] = {
    "python_web": {
        "name": "Python Web Application",
        "keywords": [
            "web", "网站", "应用", "后台", "管理系统",
            "backend", "api", "server", "服务",
            
        ],
        "initial_tasks": [
            {
                "task_id": "create_project_dir",
                "title": "创建项目目录",
                "priority": 100,
                "tool": "filesystem",
                "goal": "创建项目根目录",
            },
            {
                "task_id": "create_gitignore",
                "title": "创建 .gitignore",
                "priority": 95,
                "tool": "filesystem",
                "goal": "创建 .gitignore",
            },
            {
                "task_id": "create_requirements",
                "title": "创建 requirements.txt",
                "priority": 95,
                "tool": "filesystem",
                "goal": "创建 requirements.txt",
            },
            {
                "task_id": "create_readme",
                "title": "创建 README.md",
                "priority": 90,
                "tool": "filesystem",
                "goal": "创建 README.md",
            },
            {
                "task_id": "create_app",
                "title": "创建后端应用入口",
                "priority": 80,
                "tool": "filesystem",
                "goal": "创建 app.py",
            },
            {
                "task_id": "create_config",
                "title": "创建应用配置",
                "priority": 75,
                "tool": "filesystem",
                "goal": "创建 config.py",
            },
            {
                "task_id": "create_models",
                "title": "创建数据库模型",
                "priority": 70,
                "tool": "filesystem",
                "goal": "创建 models.py",
            },
            {
                "task_id": "create_schema",
                "title": "创建数据库 Schema",
                "priority": 65,
                "tool": "filesystem",
                "goal": "创建 schema.sql",
            },
            {
                "task_id": "create_routes",
                "title": "创建 API 路由",
                "priority": 60,
                "tool": "filesystem",
                "goal": "创建 routes.py",
            },
            {
                "task_id": "create_frontend_html",
                "title": "创建前端 HTML",
                "priority": 50,
                "tool": "filesystem",
                "goal": "创建 index.html",
            },
            {
                "task_id": "create_frontend_css",
                "title": "创建前端样式",
                "priority": 45,
                "tool": "filesystem",
                "goal": "创建 CSS",
            },
            {
                "task_id": "create_frontend_js",
                "title": "创建前端逻辑",
                "priority": 40,
                "tool": "filesystem",
                "goal": "创建 JavaScript",
            },
            {
                "task_id": "create_dockerfile",
                "title": "创建 Dockerfile",
                "priority": 30,
                "tool": "filesystem",
                "goal": "创建 Dockerfile",
            },
            {
                "task_id": "create_docker_compose",
                "title": "创建 docker-compose.yml",
                "priority": 25,
                "tool": "filesystem",
                "goal": "创建 docker-compose.yml",
            },
            {
                "task_id": "create_tests",
                "title": "创建测试文件",
                "priority": 20,
                "tool": "filesystem",
                "goal": "创建测试",
            },
        ],
        "required_ids": [
            "create_project_dir", "create_app", "create_models",
            "create_routes", "create_frontend_html",
        ],
    },
    "python_cli": {
        "name": "Python CLI Tool",
        "keywords": [
            "cli", "命令行", "工具", "script", "脚本",
        ],
        "initial_tasks": [
            {
                "task_id": "create_project_dir",
                "title": "创建项目目录",
                "priority": 100,
                "tool": "filesystem",
                "goal": "创建项目根目录",
            },
            {
                "task_id": "create_main",
                "title": "创建主入口",
                "priority": 90,
                "tool": "filesystem",
                "goal": "创建 main.py",
            },
            {
                "task_id": "create_utils",
                "title": "创建工具模块",
                "priority": 80,
                "tool": "filesystem",
                "goal": "创建 utils.py",
            },
            {
                "task_id": "create_tests",
                "title": "创建测试文件",
                "priority": 70,
                "tool": "filesystem",
                "goal": "创建测试",
            },
            {
                "task_id": "create_readme",
                "title": "创建 README.md",
                "priority": 60,
                "tool": "filesystem",
                "goal": "创建 README.md",
            },
            {
                "task_id": "create_gitignore",
                "title": "创建 .gitignore",
                "priority": 55,
                "tool": "filesystem",
                "goal": "创建 .gitignore",
            },
        ],
        "required_ids": ["create_main", "create_utils"],
    },
    "frontend": {
        "name": "Frontend Application",
        "keywords": [
            "frontend", "前端", "ui", "界面", "页面",
        ],
        "initial_tasks": [
            {
                "task_id": "create_project_dir",
                "title": "创建项目目录",
                "priority": 100,
                "tool": "filesystem",
                "goal": "创建项目根目录",
            },
            {
                "task_id": "create_html",
                "title": "创建 HTML 入口",
                "priority": 90,
                "tool": "filesystem",
                "goal": "创建 index.html",
            },
            {
                "task_id": "create_css",
                "title": "创建样式文件",
                "priority": 80,
                "tool": "filesystem",
                "goal": "创建 CSS",
            },
            {
                "task_id": "create_js",
                "title": "创建脚本文件",
                "priority": 70,
                "tool": "filesystem",
                "goal": "创建 JavaScript",
            },
            {
                "task_id": "create_readme",
                "title": "创建 README.md",
                "priority": 60,
                "tool": "filesystem",
                "goal": "创建 README.md",
            },
        ],
        "required_ids": ["create_project_dir", "create_html"],
    },
}


# ── Public API ───────────────────────────────────────────────────────


def match_template(goal: str, goal_type: str = "") -> dict[str, Any] | None:
    """Find the best matching template for a goal.

    Checks keywords in the goal against each template's keyword list.
    Requires at least 2 keyword matches to avoid false positives
    (e.g. "图形界面" matching "frontend" template via a single keyword "界面").

    Returns None if no template meets the threshold.
    """
    if goal_type != "project":
        return None

    goal_lower = goal.lower()
    best_match = None
    best_count = 0

    for tpl_id, tpl in TEMPLATES.items():
        count = sum(1 for kw in tpl["keywords"] if kw in goal_lower)
        if count > best_count:
            best_count = count
            best_match = tpl_id

    # Require at least 2 keyword matches to avoid false positives
    if best_match and best_count >= 2:
        return {"id": best_match, **TEMPLATES[best_match]}
    return None


def get_initial_tasks(
    template: dict[str, Any],
    goal: str,
    existing_files: set[str],
) -> list[Task]:
    """Create the initial Task queue from a template.

    Args:
        template: Matched template dict.
        goal: The user's goal string.
        existing_files: Already-existing file paths (relative to project root).

    Returns:
        List of Task objects.  Tasks whose target files already exist
        in the workspace are marked DONE.
    """
    project_name = extract_project_name(goal)
    tasks: list[Task] = []

    for cfg in template.get("initial_tasks", []):
        task_id = cfg["task_id"]
        title = cfg.get("title", task_id)
        priority = cfg.get("priority", 50)
        tool = cfg.get("tool", "filesystem")
        tgoal = cfg.get("goal", title)

        target_path = _infer_path(task_id, project_name)

        # Check if this task's output file already exists
        if target_path and target_path in existing_files:
            status = TaskStatus.DONE
        else:
            status = TaskStatus.TODO

        # Embed path into task input so the executor has enough info
        # to dispatch without LLM intervention for template tasks.
        task_input: dict[str, Any] = {}
        if target_path:
            if task_id == "create_project_dir":
                task_input["action"] = "mkdir"
                task_input["path"] = target_path
            else:
                task_input["action"] = "write_file"
                task_input["path"] = target_path
                # Add default content for known file types so stub files
                # are not created empty.
                default_content = _get_default_content(target_path, project_name)
                if default_content:
                    task_input["content"] = default_content

        task = Task(
            task_id=task_id,
            title=title,
            priority=priority,
            tool=tool,
            goal=tgoal,
            status=status,
            input=task_input,
        )
        tasks.append(task)

    return tasks


def is_goal_completed(
    template: dict[str, Any],
    task_queue: list[Task],
) -> bool:
    """Check whether all ``required_ids`` tasks are DONE."""
    required = template.get("required_ids", [])
    if not required:
        return False

    task_map = {t.task_id: t for t in task_queue}
    for rid in required:
        t = task_map.get(rid)
        if t is None:
            return False
        if t.status not in (TaskStatus.DONE, TaskStatus.COMPLETED):
            return False
    return True


def get_existing_files(workspace_path: str | Path) -> set[str]:
    """Scan a directory and return relative paths of all files."""
    root = Path(workspace_path)
    if not root.exists() or not root.is_dir():
        return set()

    files: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_file():
            rel = entry.relative_to(root).as_posix()
            files.add(rel)
    return files


def extract_project_name(goal: str) -> str:
    """Extract a project directory name from the goal string.

    Strips common conversational prefixes ("\u5e2e\u6211\u5199\u4e2a", "\u5f00\u53d1\u4e00\u4e2a", ...)
    before capturing the name, so requests like "\u5e2e\u6211\u5199\u4e2a\u56fe\u4e66\u7ba1\u7406\u7cfb\u7edf"
    yield a clean directory name ("\u56fe\u4e66\u7ba1\u7406") instead of the full sentence.
    """
    goal_clean = goal.strip()

    prefixes = [
        "\u5e2e\u6211\u5199\u4e2a", "\u5e2e\u6211\u5199\u4e00\u4e2a",
        "\u5e2e\u6211\u521b\u5efa\u4e00\u4e2a", "\u5e2e\u6211\u521b\u5efa\u4e2a",
        "\u5e2e\u6211\u505a\u4e00\u4e2a", "\u5e2e\u6211\u505a\u4e2a",
        "\u7ed9\u6211\u5199\u4e2a", "\u7ed9\u6211\u505a\u4e00\u4e2a", "\u7ed9\u6211\u505a\u4e2a",
        "\u8bf7\u5e2e\u6211", "\u5e2e\u6211", "\u7ed9\u6211",
        "\u5f00\u53d1\u4e00\u4e2a", "\u521b\u5efa\u4e00\u4e2a", "\u521b\u5efa\u4e2a",
        "\u505a\u4e00\u4e2a", "\u505a\u4e2a", "\u642d\u5efa", "\u6784\u5efa",
        "\u5b9e\u73b0", "\u751f\u6210", "\u5f00\u53d1", "\u521b\u5efa",
        "\u5199\u4e00\u4e2a", "\u5199\u4e2a",
    ]
    for prefix in prefixes:
        if goal_clean.startswith(prefix):
            goal_clean = goal_clean[len(prefix):].lstrip()
            break

    m = re.match(r"^(.*?)(?:\u7cfb\u7edf|\u9879\u76ee|\u5e94\u7528|\u7f51\u7ad9|\u5e73\u53f0)", goal_clean)
    raw = m.group(1).strip() if m else goal_clean
    raw = raw.replace("\u4e00\u4e2a", "").replace("\u7684", "").strip()
    if raw:
        return raw.replace(" ", "_").replace("-", "_").lower()[:48]
    name = goal_clean[:20].replace(" ", "_").lower()
    return name


# ── Internal helpers ─────────────────────────────────────────────────


def _infer_path(task_id: str, project_name: str) -> str | None:
    """Infer the likely target file path for a task_id.

    Used to check if a task's output already exists in the workspace.
    """
    path_map: dict[str, str] = {
        "create_project_dir": project_name,
        "create_gitignore": f"{project_name}/.gitignore",
        "create_requirements": f"{project_name}/requirements.txt",
        "create_readme": f"{project_name}/README.md",
        "create_app": f"{project_name}/app.py",
        "create_config": f"{project_name}/config.py",
        "create_models": f"{project_name}/models.py",
        "create_schema": f"{project_name}/schema.sql",
        "create_routes": f"{project_name}/routes.py",
        "create_frontend_html": f"{project_name}/index.html",
        "create_frontend_css": f"{project_name}/static/css/style.css",
        "create_frontend_js": f"{project_name}/static/js/app.js",
        "create_dockerfile": f"{project_name}/Dockerfile",
        "create_docker_compose": f"{project_name}/docker-compose.yml",
        "create_tests": f"{project_name}/tests/test_app.py",
        "create_main": f"{project_name}/main.py",
        "create_utils": f"{project_name}/utils.py",
        "create_html": f"{project_name}/index.html",
        "create_css": f"{project_name}/css/style.css",
        "create_js": f"{project_name}/js/app.js",
    }
    return path_map.get(task_id)


# ── Default content for template stub files ──────────────────────────

_DEFAULT_CONTENT: dict[str, str] = {
    "README.md": "# {project_name}\n\n项目描述\n",
    ".gitignore": "__pycache__/\n*.pyc\n.env\nvenv/\n.venv/\n",
    "requirements.txt": "# 依赖\n",
    "index.html": "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>{project_name}</title>\n</head>\n<body>\n    <h1>{project_name}</h1>\n</body>\n</html>\n",
    "app.py": "# {project_name}\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n",
    "style.css": "/* {project_name} */\n",
    "app.js": "// {project_name}\n",
    "config.py": "# {project_name} 配置\n",
    "models.py": "# {project_name} 数据模型\n",
    "main.py": "# {project_name}\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n",
    "utils.py": "# {project_name} 工具函数\n",
    "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD [\"python\", \"app.py\"]\n",
}


def _get_default_content(path: str, project_name: str) -> str | None:
    """Return default content for a template stub file path.

    Matches the filename portion of the path against known defaults.
    Returns ``None`` when no default is available (file stays empty).
    """
    import os
    fname = os.path.basename(path)
    template = _DEFAULT_CONTENT.get(fname)
    if template is None:
        return None
    return template.replace("{project_name}", project_name or "project")
