"""Intent Eval Runner — evaluate GoalAnalyzer intent classification accuracy."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agentflow.eval.common import MockLLMService, save_results
from agentflow.eval.intent_eval.dataset import IntentEvalDataset
from agentflow.eval.intent_eval.metrics import compute_all

logger = logging.getLogger("intent_eval")


class IntentEvalRunner:
    """Evaluate GoalAnalyzer intent classification.

    Paths tested:
    1. Embedding match — real IntentIndex / QwenEmbedder (majority of queries)
    2. LLM fallback — inject MockLLMService with pre-configured responses
    3. Degraded fallback — no LLM, minimal fallback (other / confidence=0.1)
    """

    def __init__(self, dataset: IntentEvalDataset) -> None:
        self.dataset = dataset
        self.per_sample: list[dict[str, Any]] = []
        self.summary: dict[str, float] = {}

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Run all samples through GoalAnalyzer and compute metrics."""
        from agentflow.agents.goal_analyzer.agent import GoalAnalyzer

        actual_types: list[str] = []
        expected_types: list[str] = []
        embedding_flags: list[bool] = []
        confidences: list[float] = []

        for i, sample in enumerate(self.dataset, 1):
            sid = sample.get("id", f"intent_{i}")
            question = sample["question"]
            expected = sample["expected_goal_type"]

            if verbose:
                print(f"[{i}/{len(self.dataset)}] {question[:64]} ...", end=" ")

            # Run goal analysis
            mock_resp = sample.get("mock_llm_response")
            degraded = sample.get("_degraded", False)

            state = {"question": question}
            if degraded:
                state["_degraded"] = {"goal_analyzer", "planner", "answer"}

            if mock_resp is not None:
                result = self._run_with_mock_llm(state, mock_resp)
            else:
                result = GoalAnalyzer().run(state)

            goal = result.get("goal_analysis", {})
            actual = goal.get("goal_type", "other")
            confidence = goal.get("confidence", 0.0)
            is_embedding = bool(goal.get("_embedding_match", False))

            actual_types.append(actual)
            expected_types.append(expected)
            embedding_flags.append(is_embedding)
            confidences.append(confidence)

            match_ok = "OK" if actual == expected else f"MISMATCH ({actual} != {expected})"
            emb_flag = "[E]" if is_embedding else "[L]"
            if verbose:
                print(f"{match_ok} conf={confidence:.2f} {emb_flag}")

            self.per_sample.append({
                "sample_id": sid,
                "question": question,
                "expected_goal_type": expected,
                "actual_goal_type": actual,
                "confidence": confidence,
                "embedding_match": is_embedding,
                "match": actual == expected,
            })

        metrics = compute_all(actual_types, expected_types, embedding_flags, confidences)
        self.summary = metrics

        return {
            "summary": self.summary,
            "per_sample": self.per_sample,
            "config": {
                "total_samples": len(self.dataset),
            },
        }

    def _run_with_mock_llm(self, state: dict, mock_resp: dict) -> dict:
        """Inject MockLLMService and run GoalAnalyzer."""
        import agentflow.services.llm_service as llm_mod
        from agentflow.agents.goal_analyzer.agent import GoalAnalyzer

        original_llm = llm_mod._llm_service

        resp_type = mock_resp.get("type", "json")
        if resp_type == "degraded":
            mock_llm = MockLLMService(degraded=True)
        elif resp_type == "json":
            content = mock_resp.get("content", {})
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            mock_llm = MockLLMService(responses={"default": str(content)})
        else:
            mock_llm = MockLLMService(responses={"default": str(mock_resp.get("content", ""))})

        try:
            llm_mod._llm_service = mock_llm
            analyzer = GoalAnalyzer()
            analyzer._llm = mock_llm
            return analyzer.run(state)
        finally:
            llm_mod._llm_service = original_llm

    def save_results(self, path: str | Path) -> None:
        save_results(path, self.summary, self.per_sample, {
            "total_samples": len(self.dataset),
        })
