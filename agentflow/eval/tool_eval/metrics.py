"""Tool 评估指标：成功率、错误分布等。"""

from __future__ import annotations

from typing import Any



def success_rate(results: list[dict]) -> float:
    """整体成功率：success=True 的占比。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("success", False)) / len(results)


def action_success_rate(results: list[dict]) -> dict[str, float]:
    """每个 action 的成功率：{"tool.action": rate}。"""
    groups: dict[str, list[bool]] = {}
    for r in results:
        key = f"{r.get('tool', '')}.{r.get('action', '')}"
        groups.setdefault(key, []).append(r.get("success", False))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def tool_success_rate(results: list[dict]) -> dict[str, float]:
    """每个 tool 的成功率：{"tool": rate}。"""
    groups: dict[str, list[bool]] = {}
    for r in results:
        key = r.get("tool", "")
        groups.setdefault(key, []).append(r.get("success", False))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def error_distribution(results: list[dict]) -> dict[str, int]:
    """错误类型分布：{error_prefix: count}。"""
    dist: dict[str, int] = {}
    for r in results:
        if not r.get("success", False) and r.get("error"):
            prefix = r["error"].split(":")[0].strip()
            dist[prefix] = dist.get(prefix, 0) + 1
    return dist


def check_pass_rate(samples: list[dict], results: list[dict]) -> float:
    """expected_result_checks 通过率。

    对每条样本的 result 做字段/值断言检查，
    支持 ``"message_contains"`` 等特殊检查键。
    """
    if not samples:
        return 0.0
    passed = 0
    total = 0
    for sample, result in zip(samples, results):
        checks = sample.get("expected_result_checks", {})
        if not checks:
            continue
        total += 1
        actual = result.get("result", {})
        if _check_match(actual, result, checks):
            passed += 1
    return passed / total if total else 1.0


def _check_match(actual: Any, full_result: dict, checks: dict) -> bool:
    """逐条检查 expected_result_checks。"""
    for check_key, expected_val in checks.items():
        if check_key == "message_contains":
            message = full_result.get("message", "")
            if expected_val not in message:
                return False
        elif check_key.startswith("result."):
            field = check_key[len("result."):]
            if isinstance(actual, dict):
                if actual.get(field) != expected_val:
                    return False
            else:
                return False
        else:
            if isinstance(actual, dict):
                if actual.get(check_key) != expected_val:
                    return False
            else:
                return False
    return True


def compute_all(samples: list[dict], results: list[dict]) -> dict[str, Any]:
    """计算全部工具评估指标。"""
    return {
        "success_rate": success_rate(results),
        "action_rates": action_success_rate(results),
        "tool_rates": tool_success_rate(results),
        "error_distribution": error_distribution(results),
        "check_pass_rate": check_pass_rate(samples, results),
    }


def aggregate(per_sample_metrics: list[dict[str, float]]) -> dict[str, float]:
    """汇总多次运行的指标均值。"""
    if not per_sample_metrics:
        return {}
    keys = per_sample_metrics[0].keys()
    result: dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in per_sample_metrics if key in m]
        result[key] = sum(vals) / len(vals) if vals else 0.0
    return result
