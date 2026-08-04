"""Tests for the code-generation quality improvements:

- project-design-first generation (shared brief for all files)
- project brief injection into every codegen prompt
- per-file role inference
- syntax validation with one corrective retry
"""

from agentflow.agents.planner.agent import (
    PlannerAgent,
    _infer_file_role,
    _is_fillable_text_file,
    _looks_substantial,
    _validate_generated_code,
)
from agentflow.agents.planner.prompt import (
    build_codegen_prompt,
    build_project_design_prompt,
)
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, node_name="", max_tokens=None, model=None):
        self.calls.append({
            "node_name": node_name,
            "model": model,
            "messages": messages,
        })
        if self.responses:
            return self.responses.pop(0)
        return ""


def _task(path: str, code_prompt: str = "") -> Task:
    inp = {"action": "write_file", "path": path}
    if code_prompt:
        inp["code_prompt"] = code_prompt
    return Task(
        task_id=f"t_{path}",
        title=f"创建 {path}",
        priority=80,
        tool="filesystem",
        goal=f"创建 {path}",
        input=inp,
    )


def _planner(fake: _FakeLLM) -> PlannerAgent:
    p = PlannerAgent.__new__(PlannerAgent)
    p._llm = fake
    p.registry = None
    return p


# -- prompt injection ---------------------------------------------------------


def test_codegen_prompt_includes_project_brief():
    msgs = build_codegen_prompt("创建 books 表", "SQL", "项目背景：图书管理系统")
    user = msgs[1]["content"]
    assert "项目背景" in user
    assert "图书管理系统" in user
    assert "创建 books 表" in user


def test_design_prompt_lists_files():
    msgs = build_project_design_prompt("帮我写个图书管理系统", ["app.py", "models.py"])
    assert "app.py" in msgs[1]["content"]
    assert "帮我写个图书管理系统" in msgs[1]["content"]


# -- role inference -----------------------------------------------------------


def test_infer_file_role():
    assert "入口" in _infer_file_role("project/app.py")
    assert "路由" in _infer_file_role("project/routes.py")
    assert "模型" in _infer_file_role("project/models.py")
    assert "schema" in _infer_file_role("project/schema.sql").lower()
    assert "测试" in _infer_file_role("project/tests/test_app.py")
    assert "页面" in _infer_file_role("project/index.html")


def test_looks_substantial_rejects_template_stubs():
    # Comment-only stub
    assert not _looks_substantial("# 学生成绩管理系统", "models.py")
    # def main(): pass stub without imports
    stub = "# 学生成绩管理系统\n\ndef main():\n    pass\n"
    assert not _looks_substantial(stub, "app.py")
    # Long config text is substantial for non-code files
    assert _looks_substantial("# 依赖\nflask\n" * 40, "requirements.txt")


def test_looks_substantial_accepts_real_code():
    real = (
        "import flask\nfrom flask import Flask\n\n"
        "app = Flask(__name__)\n\n@app.route('/')\ndef index():\n    return 'ok'\n"
    )
    assert _looks_substantial(real, "app.py")
    assert _looks_substantial("x" * 600, "app.py")


def test_fillable_text_files():
    assert _is_fillable_text_file("project/requirements.txt")
    assert _is_fillable_text_file("pyproject.toml")
    assert not _is_fillable_text_file("README.md")
    assert not _is_fillable_text_file("app.py")


def test_fill_code_content_generates_requirements():
    fake = _FakeLLM(["flask\nflask-sqlalchemy\n"])
    p = _planner(fake)
    plan = Plan(
        goal="图书管理系统",
        category="project",
        tasks=[_task("requirements.txt")],
    )
    # Stub content must be regenerated, not kept.
    plan.tasks[0].input["content"] = "# 依赖"
    errors = p._fill_code_content(plan, "项目背景：图书管理系统")
    assert errors == []
    assert plan.tasks[0].input["content"] == "flask\nflask-sqlalchemy"


# -- syntax validation --------------------------------------------------------


def test_validate_generated_code():
    ok, _ = _validate_generated_code("app.py", "def main():\n    pass\n")
    assert ok
    ok, detail = _validate_generated_code("app.py", "def main(:\n    pass\n")
    assert not ok
    assert "line" in detail
    # Non-python files are not validated.
    ok, _ = _validate_generated_code("app.js", "this is not (valid js")
    assert ok


# -- design-first -------------------------------------------------------------


def test_generate_project_design_parses_llm_json():
    design_json = (
        '{"tech_stack": "Flask + SQLite", '
        '"data_model": [{"table": "books", "fields": ["id:int", "title:str"]}], '
        '"api_endpoints": ["GET /api/books - 图书列表"], '
        '"modules": [{"file": "models.py", "responsibility": "数据模型"}], '
        '"key_requirements": ["支持图书增删改查"]}'
    )
    fake = _FakeLLM([design_json])
    p = _planner(fake)
    plan = Plan(
        goal="帮我写个图书管理系统",
        category="project",
        tasks=[_task("models.py", "创建数据模型")],
    )
    brief = p._generate_project_design("帮我写个图书管理系统", plan)
    assert "Flask" in brief
    assert "books" in brief
    assert "GET /api/books" in brief
    assert "支持图书增删改查" in brief


def test_build_project_brief_deterministic():
    p = _planner(_FakeLLM([]))
    plan = Plan(
        goal="图书管理系统",
        category="project",
        tasks=[_task("models.py", "x"), _task("routes.py", "y")],
    )
    brief = p._build_project_brief(plan, "图书管理系统")
    assert "图书管理系统" in brief
    assert "models.py" in brief
    assert "routes.py" in brief


# -- fill with brief + retry --------------------------------------------------


def test_fill_code_content_passes_brief_and_removes_code_prompt():
    fake = _FakeLLM(["def main():\n    pass\n"])
    p = _planner(fake)
    plan = Plan(
        goal="图书管理系统",
        category="project",
        tasks=[_task("app.py", "创建入口")],
    )
    errors = p._fill_code_content(plan, "项目背景：图书管理系统")
    assert errors == []
    assert plan.tasks[0].input["content"] == "def main():\n    pass"
    assert "code_prompt" not in plan.tasks[0].input
    # The brief must have been sent to the LLM.
    user = fake.calls[0]["messages"][1]["content"]
    assert "项目背景" in user
    assert "图书管理系统" in user
    assert fake.calls[0]["model"] is None


def test_fill_code_content_retries_on_syntax_error():
    fake = _FakeLLM(["def broken(:\n    pass\n", "def main():\n    pass\n"])
    p = _planner(fake)
    plan = Plan(
        goal="图书管理系统",
        category="project",
        tasks=[_task("app.py", "创建入口")],
    )
    errors = p._fill_code_content(plan, "brief")
    assert errors == []
    assert len(fake.calls) == 2
    retry_user = fake.calls[1]["messages"][1]["content"]
    assert "语法错误" in retry_user
    content = plan.tasks[0].input["content"]
    ok, _ = _validate_generated_code("app.py", content)
    assert ok


def test_fill_code_content_reports_persistent_invalid():
    fake = _FakeLLM(["def broken(:\n", "def still_broken(:\n"])
    p = _planner(fake)
    plan = Plan(
        goal="图书管理系统",
        category="project",
        tasks=[_task("app.py", "创建入口")],
    )
    errors = p._fill_code_content(plan, "brief")
    assert len(errors) == 1
    assert "app.py" in errors[0]
