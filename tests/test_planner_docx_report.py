from __future__ import annotations

from agentflow.agents.planner.agent import PlannerAgent


def test_docx_report_goal_uses_docx_create_before_direct_answer():
    planner = PlannerAgent()

    state = {
        "question": "整理成docx报告",
        "goal_analysis": {
            "goal": "整理成docx报告",
            "goal_type": "question",
            "knowledge_source": "general",
        },
        "history": [
            {"role": "user", "content": "介绍 OmniForge 亮点"},
            {"role": "assistant", "content": "OmniForge 的亮点包括多 Agent 工作流、知识库检索和工具执行。"},
        ],
    }

    result = planner.run(state)

    queue = result["task_queue"]
    assert len(queue) == 1
    task = queue[0]
    assert task["tool"] == "docx"
    assert task["input"]["action"] == "create"
    assert task["input"]["path"] == "OmniForge报告.docx"
    assert "多 Agent 工作流" in task["input"]["content"]
    assert result["plan"].direct_answer is False


def test_no_snake_game_special_case():
    """Snake-game requests must go through the generic planner, not a
    hard-coded shortcut (benchmark-era special case was removed)."""
    import agentflow.agents.planner.special_goals as sg

    assert not hasattr(sg, "is_snake_game_goal")
    assert not hasattr(sg, "build_snake_game_files_plan")
    assert not hasattr(sg, "python_snake_content")
