"""Run all evaluation suites and print results (ASCII-safe).

Covers: Tool Eval, Planner Eval, Completion Eval, Intent Eval, RAG Eval.
"""

from __future__ import annotations

import traceback
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent

SEP = "=" * 60
SUB = "-" * 40


def run_tool_eval():
    """Tool Eval -- mock mode."""
    print(SEP)
    print("SUITE 1/5: Tool Eval (mock mode)")
    print(SEP)

    from agentflow.eval.tool_eval.dataset import ToolEvalDataset
    from agentflow.eval.tool_eval.mock_runner import MockToolEvalRunner

    ds = ToolEvalDataset.load(str(_EVAL_DIR / "tool_eval" / "data" / "mock_tool_dataset.jsonl"))
    print(f"Loaded {len(ds)} samples")
    print(f"Stats: {ds.stats()}")

    runner = MockToolEvalRunner(ds)
    result = runner.run(verbose=True)

    summary = result["summary"]
    print("\n--- Tool Eval Results ---")
    print(f"Overall success_rate: {summary['success_rate']:.2%}")
    print("Action rates:")
    for action, rate in sorted(summary.get("action_rates", {}).items()):
        print(f"  {action}: {rate:.2%}")
    print(f"Error distribution: {summary.get('error_distribution', {})}")

    mismatches = [s for s in runner.per_sample if not s.get("expected_match", True)]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches[:15]:
            print(f"  [{m['sample_id']}] {m['tool']}.{m['action']} "
                  f"expected={m['expected_success']} actual={m['actual_success']}")
    else:
        print("No mismatches.")

    runner.save_results(str(_EVAL_DIR / "tool_eval" / "data" / "mock_tool_results.json"))
    print("Results saved.")
    return summary


def run_planner_eval():
    """Planner Eval."""
    print(f"\n{SEP}")
    print("SUITE 2/5: Planner Eval")
    print(SEP)

    from agentflow.eval.planner_eval.dataset import PlannerEvalDataset
    from agentflow.eval.planner_eval.runner import PlannerEvalRunner

    ds = PlannerEvalDataset.load(str(_EVAL_DIR / "planner_eval" / "data" / "planner_dataset.jsonl"))
    print(f"Loaded {len(ds)} samples")
    print(f"Stats: {ds.stats()}")

    runner = PlannerEvalRunner(ds)
    result = runner.run(verbose=True)

    summary = result["summary"]
    print("\n--- Planner Eval Results ---")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value:.4f}")

    runner.save_results(str(_EVAL_DIR / "planner_eval" / "data" / "planner_results.json"))
    print("Results saved.")
    return summary


def run_completion_eval():
    """Completion Eval."""
    print(f"\n{SEP}")
    print("SUITE 3/5: Completion Eval")
    print(SEP)

    from agentflow.eval.completion_eval.dataset import CompletionEvalDataset
    from agentflow.eval.completion_eval.runner import CompletionEvalRunner

    ds = CompletionEvalDataset.load(
        str(_EVAL_DIR / "completion_eval" / "data" / "completion_dataset.jsonl")
    )
    print(f"Loaded {len(ds)} samples")
    print(f"Stats: {ds.stats()}")

    runner = CompletionEvalRunner(ds)
    result = runner.run(max_turns=5, verbose=True)

    summary = result["summary"]
    print("\n--- Completion Eval Results ---")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value:.4f}")

    print("\nPer-sample:")
    for s in runner.per_sample:
        status = "OK" if s.get("expected_match", False) else "MISMATCH"
        print(f"  [{status}] {s['sample_id']}: turns={s.get('turns_taken',0)} "
              f"tasks_done={s.get('tasks_done',0)}/{s.get('tasks_total',0)} "
              f"completed={s.get('goal_completed', False)} "
              f"expected_completed={s.get('expected_completed', False)}")

    runner.save_results(
        str(_EVAL_DIR / "completion_eval" / "data" / "completion_results.json")
    )
    print("Results saved.")
    return summary


def run_intent_eval():
    """Intent Eval -- GoalAnalyzer intent classification accuracy."""
    print(f"\n{SEP}")
    print("SUITE 4/5: Intent Eval")
    print(SEP)

    from agentflow.eval.intent_eval.dataset import IntentEvalDataset
    from agentflow.eval.intent_eval.runner import IntentEvalRunner

    ds = IntentEvalDataset.load(str(_EVAL_DIR / "intent_eval" / "data" / "intent_dataset.jsonl"))
    print(f"Loaded {len(ds)} samples")
    print(f"Stats: {ds.stats()}")

    runner = IntentEvalRunner(ds)
    result = runner.run(verbose=True)

    summary = result["summary"]
    print("\n--- Intent Eval Results ---")
    print(f"goal_type_accuracy:  {summary.get('goal_type_accuracy', 0):.2%}")
    print(f"embedding_hit_rate:  {summary.get('embedding_hit_rate', 0):.2%}")
    print(f"embedding_accuracy:  {summary.get('embedding_accuracy', 0):.2%}")
    print(f"llm_accuracy:        {summary.get('llm_accuracy', 0):.2%}")
    print(f"confidence_mean:     {summary.get('confidence_mean', 0):.4f}")
    print("\nPer-label accuracy:")
    for label, acc in sorted(summary.get("per_label_accuracy", {}).items()):
        print(f"  {label}: {acc:.2%}")
    print("\nConfusion matrix:")
    confusion = summary.get("confusion", {})
    for expected_label, actuals in sorted(confusion.items()):
        for actual_label, count in sorted(actuals.items()):
            if expected_label != actual_label:
                print(f"  {expected_label} -> {actual_label}: {count}")

    mismatches = [s for s in runner.per_sample if not s.get("match", False)]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  [{m['sample_id']}] \"{m['question'][:50]}\" "
                  f"expected={m['expected_goal_type']} actual={m['actual_goal_type']}")
    else:
        print("All samples matched!")

    runner.save_results(str(_EVAL_DIR / "intent_eval" / "data" / "intent_results.json"))
    print("Results saved.")
    return summary


def run_rag_eval():
    """RAG Eval -- requires indexed KnowledgeStore."""
    print(f"\n{SEP}")
    print("SUITE 5/5: RAG Eval")
    print(SEP)

    from agentflow.knowledge.eval.dataset import EvalDataset
    from agentflow.knowledge.eval.runner import EvalRunner
    from agentflow.knowledge.store import KnowledgeStore

    ds_path = _EVAL_DIR.parent / "knowledge" / "eval" / "data" / "eval_data.jsonl"
    ds = EvalDataset.load(str(ds_path))
    print(f"Loaded {len(ds)} samples")
    print(f"Stats: {ds.stats()}")

    store = KnowledgeStore()
    # ``KnowledgeStore`` exposes the vector count via ``index_size`` (the
    # previous ``store.index.ntotal`` raised AttributeError, so the RAG suite
    # always reported FAILED instead of running).
    if store.index_size == 0:
        print("WARNING: KnowledgeStore index is empty. Run indexing first.")
        print("Skipping RAG eval.")
        return None

    runner = EvalRunner(store, ds)
    try:
        result = runner.run(verbose=True)
    finally:
        # Release the Qdrant local file lock before the process exits.
        if store.qdrant_index is not None:
            store.qdrant_index.close()

    summary = result["summary"]
    print("\n--- RAG Eval Results ---")
    for key in ["recall@1", "recall@3", "recall@5", "recall@10",
                 "precision@1", "precision@3", "precision@5", "precision@10",
                 "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
                 "hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
        if key in summary:
            print(f"  {key}: {summary[key]:.4f}")

    runner.save_results(str(_EVAL_DIR.parent / "knowledge" / "eval" / "data" / "eval_results.json"))
    print("Results saved.")
    return summary


def main():
    all_summaries = {}

    suites = [
        ("tool_eval", run_tool_eval),
        ("planner_eval", run_planner_eval),
        ("completion_eval", run_completion_eval),
        ("intent_eval", run_intent_eval),
        ("rag_eval", run_rag_eval),
    ]

    for name, runner_fn in suites:
        try:
            all_summaries[name] = runner_fn()
        except Exception:
            print(f"FAILED: {name}")
            traceback.print_exc()

    # Final summary
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(SEP)
    for suite_name, summary in all_summaries.items():
        print(f"{suite_name}:")
        if isinstance(summary, dict):
            for k, v in sorted(summary.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    print(f"  {k}: {len(v)} entries")
                else:
                    print(f"  {k}: {v}")
        elif summary is None:
            print("  (skipped)")
        print()


if __name__ == "__main__":
    main()
