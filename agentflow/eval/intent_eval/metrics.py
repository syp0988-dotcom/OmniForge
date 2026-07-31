"""Intent evaluation metrics: classification accuracy, embedding hit rate, etc."""

from __future__ import annotations



def goal_type_accuracy(actual_types: list[str], expected_types: list[str]) -> float:
    """Exact match accuracy for goal_type classification."""
    if not actual_types:
        return 0.0
    correct = sum(1 for a, e in zip(actual_types, expected_types) if a == e)
    return correct / len(actual_types)


def embedding_hit_rate(embedding_flags: list[bool]) -> float:
    """Fraction of queries matched by embedding (not LLM fallback)."""
    if not embedding_flags:
        return 0.0
    return sum(1 for f in embedding_flags if f) / len(embedding_flags)


def embedding_accuracy(
    actual_types: list[str],
    expected_types: list[str],
    embedding_flags: list[bool],
) -> float:
    """Accuracy for embedding-matched samples only."""
    subset = [(a, e) for a, e, f in zip(actual_types, expected_types, embedding_flags) if f]
    if not subset:
        return 1.0
    return sum(1 for a, e in subset if a == e) / len(subset)


def llm_accuracy(
    actual_types: list[str],
    expected_types: list[str],
    embedding_flags: list[bool],
) -> float:
    """Accuracy for LLM-fallback samples only."""
    subset = [(a, e) for a, e, f in zip(actual_types, expected_types, embedding_flags) if not f]
    if not subset:
        return 1.0
    return sum(1 for a, e in subset if a == e) / len(subset)


def confidence_mean(confidences: list[float]) -> float:
    """Average confidence score."""
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def per_label_accuracy(
    actual_types: list[str],
    expected_types: list[str],
) -> dict[str, float]:
    """Per-label accuracy: {label: accuracy}."""
    groups: dict[str, list[bool]] = {}
    for a, e in zip(actual_types, expected_types):
        groups.setdefault(e, []).append(a == e)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in groups.items()}


def confusion_summary(
    actual_types: list[str],
    expected_types: list[str],
) -> dict[str, dict[str, int]]:
    """Confusion matrix as {expected: {actual: count}}."""
    matrix: dict[str, dict[str, int]] = {}
    for a, e in zip(actual_types, expected_types):
        matrix.setdefault(e, {})
        matrix[e][a] = matrix[e].get(a, 0) + 1
    return matrix


def compute_all(
    actual_types: list[str],
    expected_types: list[str],
    embedding_flags: list[bool],
    confidences: list[float],
) -> dict:
    """Compute all intent evaluation metrics."""
    return {
        "goal_type_accuracy": goal_type_accuracy(actual_types, expected_types),
        "embedding_hit_rate": embedding_hit_rate(embedding_flags),
        "embedding_accuracy": embedding_accuracy(actual_types, expected_types, embedding_flags),
        "llm_accuracy": llm_accuracy(actual_types, expected_types, embedding_flags),
        "confidence_mean": confidence_mean(confidences),
        "per_label_accuracy": per_label_accuracy(actual_types, expected_types),
        "confusion": confusion_summary(actual_types, expected_types),
    }
