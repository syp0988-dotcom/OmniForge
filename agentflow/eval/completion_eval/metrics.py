"""Completion 评估指标：完成率、平均任务数、平均轮次等。"""

from __future__ import annotations


def completion_rate(results: list[dict]) -> float:
    """goal_completed=True 的样本占比。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("goal_completed")) / len(results)


def avg_tasks_done(results: list[dict]) -> float:
    """平均完成的 Task 数。"""
    if not results:
        return 0.0
    return sum(r.get("tasks_done", 0) for r in results) / len(results)


def avg_turns_taken(results: list[dict]) -> float:
    """平均达到完成所需的轮次。"""
    if not results:
        return 0.0
    return sum(r.get("turns_taken", 0) for r in results) / len(results)


def min_tasks_done_rate(results: list[dict], samples: list[dict]) -> float:
    """min_expected_tasks_done 达标率。"""
    if not results:
        return 0.0
    count = 0
    for r, s in zip(results, samples):
        if r.get("tasks_done", 0) >= s.get("min_expected_tasks_done", 0):
            count += 1
    return count / len(results)


def completion_match_rate(results: list[dict], samples: list[dict]) -> float:
    """expected_completed 匹配率。"""
    if not results:
        return 0.0
    match = 0
    for r, s in zip(results, samples):
        if r.get("goal_completed") == s.get("expected_completed"):
            match += 1
    return match / len(results)


def compute_all(results: list[dict], samples: list[dict]) -> dict:
    """计算全部任务完成率指标。"""
    return {
        "completion_rate": completion_rate(results),
        "avg_tasks_done": avg_tasks_done(results),
        "avg_turns_taken": avg_turns_taken(results),
        "min_tasks_done_rate": min_tasks_done_rate(results, samples),
        "completion_match_rate": completion_match_rate(results, samples),
    }


def aggregate(per_sample_metrics: list[dict[str, float]]) -> dict[str, float]:
    """汇总指标均值。"""
    if not per_sample_metrics:
        return {}
    keys = per_sample_metrics[0].keys()
    return {k: sum(m[k] for m in per_sample_metrics) / len(per_sample_metrics) for k in keys}
