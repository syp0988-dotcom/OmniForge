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
    Returns None if no template matches.
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

    if best_match:
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
    """Extract a project directory name from the goal string."""
    goal_clean = goal.strip()
    m = re.search(
        r"(?:创建|开发|做|实现|搭建|构建)\s*(.*?)(?:系统|项目|应用|网站|平台)",
        goal_clean,
    )
    if m:
        raw = m.group(1).strip().replace("一个", "").replace("的", "").strip()
        if raw:
            return raw.replace(" ", "_").replace("-", "_").lower()
    # Fallback: use first few chars
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
