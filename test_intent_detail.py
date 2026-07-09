"""Detailed test: show all intent scores for each query."""
import sys
sys.path.insert(0, ".")

import numpy as np
from agentflow.agents.goal_analyzer.intent_index import (
    IntentIndex, INTENT_DESCRIPTIONS, CONFIDENCE_RATIO, MIN_SCORE_FLOOR,
    _cosine_similarity,
)

idx = IntentIndex()
idx._ensure_ready()
if not idx._ready:
    print("IntentIndex not ready!")
    sys.exit(1)

tests = [
    ("帮我写一个登录页面", "coding"),
    ("什么是REST API", "question"),
    ("创建一个图书管理系统", "project"),
    ("这段代码有什么bug", "coding"),
    ("搜索最新AI新闻", "search"),
    ("你好", "chat"),
    ("git提交代码", "tool"),
    ("帮我写一篇文章", "question"),
    ("翻译成英文", "question"),
    ("优化这个函数的性能", "coding"),
    ("今天天气怎么样", "search"),
    ("初始化一个React项目", "project"),
]

intent_labels = ["coding", "project", "question", "search", "tool", "chat"]

print(f"Ratio threshold: {CONFIDENCE_RATIO}x, Score floor: {MIN_SCORE_FLOOR}")
print()
print(f"{'Query':28s} | {'Best':8s}/{'(2nd)':8s} | Ratio  | coding  project questn search  tool    chat    | Status")
print("-" * 110)

correct = 0
hit = 0
total = 0
for query, expected in tests:
    query_vec = idx._embed_query(query)
    scores = {}
    for label, anchor in idx._anchors.items():
        dot = float(np.dot(query_vec, anchor))
        norm_q = float(np.linalg.norm(query_vec))
        norm_a = float(np.linalg.norm(anchor))
        scores[label] = dot / (norm_q * norm_a) if norm_q > 0 and norm_a > 0 else 0.0

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]
    ratio = best_score / second_score if second_score > 0 else 999

    # Determine if match() would accept
    would_match = best_score >= MIN_SCORE_FLOOR and ratio >= CONFIDENCE_RATIO
    if would_match:
        hit += 1
        status = "OK" if best_label == expected else f"MIS({best_label}!={expected})"
        if best_label == expected:
            correct += 1
    else:
        status = "FALLBACK" if best_label == expected else f"FALL({best_label}!={expected})"
        if best_label == expected:
            correct += 1

    total += 1

    score_str = " ".join(f"{scores.get(l, 0):.3f}" for l in intent_labels)
    print(f"{query:28s} | {best_label:8s} {best_score:.3f} / {second_label:8s} {second_score:.3f} | {ratio:5.1f}x | {score_str} | {status}")

print("-" * 110)
print(f"Embedding hit rate: {hit}/{total} ({hit/total*100:.0f}%)")
print(f"Accuracy (of hits): {correct}/{hit} ({correct/hit*100:.0f}%)" if hit > 0 else "Accuracy: N/A (no hits)")
print(f"Fallback rate: {total - hit}/{total} ({(total-hit)/total*100:.0f}%)")

# Score distribution
all_best = []
for query, _ in tests:
    query_vec = idx._embed_query(query)
    scores = {label: float(np.dot(query_vec, anchor)) / (float(np.linalg.norm(query_vec)) * float(np.linalg.norm(anchor)))
              for label, anchor in idx._anchors.items()}
    all_best.append(max(scores.values()))

print(f"\nBest score range: {min(all_best):.3f} - {max(all_best):.3f}")
print(f"Best score mean: {np.mean(all_best):.3f}")
