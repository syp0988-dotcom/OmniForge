"""Debug script to trace the workflow for Chinese input."""
import asyncio
import json
import sys

sys.path.insert(0, 'g:/multi_agent')

# Set UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agentflow.graph.workflow import build_workflow
from agentflow.conversation.session_state import SessionState


async def test():
    workflow = build_workflow()

    initial_state = {
        "question": "你好",
        "workflow": [],
        "history": [],
    }

    print("=" * 60)
    print("Starting workflow for: 你好")
    print("=" * 60)

    node_count = 0
    final_state = None

    try:
        async for event in workflow.astream(initial_state):
            for node_name, state_update in event.items():
                node_count += 1
                print(f"\n--- Node {node_count}: {node_name} ---")

                if node_name == "conversation_manager":
                    print(f"  question: {state_update.get('question', '')[:80]}")
                    print(f"  _continue_mode: {state_update.get('_continue_mode')}")

                elif node_name == "goal_analyzer":
                    ga = state_update.get("goal_analysis", {})
                    if isinstance(ga, dict):
                        print(f"  goal_type: {ga.get('goal_type')}")
                        print(f"  goal: {ga.get('goal', '')[:80]}")
                        print(f"  confidence: {ga.get('confidence')}")
                    else:
                        print(f"  goal_analysis: {ga}")

                elif node_name == "planner":
                    plan = state_update.get("plan", {})
                    if isinstance(plan, dict):
                        print(f"  goal_completed: {plan.get('goal_completed')}")
                        print(f"  direct_answer: {plan.get('direct_answer')}")
                        print(f"  tasks: {len(plan.get('tasks', []))}")
                    else:
                        print(f"  goal_completed: {getattr(plan, 'goal_completed', 'N/A')}")
                        print(f"  direct_answer: {getattr(plan, 'direct_answer', 'N/A')}")
                        print(f"  tasks: {len(getattr(plan, 'tasks', []))}")

                    task_queue = state_update.get("task_queue", [])
                    print(f"  task_queue in update: {len(task_queue)} items")

                elif node_name == "tool_executor":
                    tr = state_update.get("tool_results", [])
                    print(f"  tool_results: {len(tr)} items")
                    if tr:
                        for r in tr:
                            print(f"    success: {r.get('success')}, error: {r.get('error', '')[:60]}")

                elif node_name == "reflector":
                    result = state_update.get("_reflection_result", "N/A")
                    print(f"  _reflection_result: {result}")
                    msg = state_update.get("_reflection_message", "")
                    print(f"  _reflection_message: {msg[:80]}")

                elif node_name == "answer":
                    answer = state_update.get("answer", "")
                    print(f"  answer length: {len(answer)}")
                    print(f"  answer (first 200): {answer[:200]}")
                    final_state = dict(state_update)

                elif node_name == "memory":
                    if final_state is None:
                        final_state = dict(state_update)
                    memory = state_update.get("memory", {})
                    if isinstance(memory, dict):
                        hist = memory.get("history", [])
                        print(f"  history length: {len(hist)}")

                elif node_name == "knowledge":
                    kc = state_update.get("knowledge_context", "")
                    print(f"  knowledge_context: {kc[:60] if kc else '(empty)'}")

                else:
                    # Print what this node output
                    keys = list(state_update.keys())[:5]
                    print(f"  keys: {keys}")

    except Exception as e:
        print(f"\n!!! WORKFLOW ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("FINAL STATE:", "captured" if final_state else "NOT captured")
    if final_state:
        answer = final_state.get("answer", "")
        print(f"Answer length: {len(answer)}")
        print(f"Answer: {answer[:200]}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
