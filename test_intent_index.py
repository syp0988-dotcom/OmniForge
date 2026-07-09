"""Integration test for ratio-based intent matching."""
import sys
sys.path.insert(0, ".")

import numpy as np
from agentflow.agents.goal_analyzer.intent_index import (
    IntentIndex, CONFIDENCE_RATIO, MIN_SCORE_FLOOR,
    _cosine_similarity,
)

idx = IntentIndex()
idx._ensure_ready()

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

print(f"{'Query':30s} | {'Expected':10s} | {'Matched':10s} | {'Best':7s} | {'2nd':7s} | {'Ratio':6s} | Status")
print("-" * 100)

ok = 0
wrong = 0
fallback = 0
for query, expected in tests:
    result = idx.match(query)
    if result is None:
        # Show what would have matched for diagnostics
        qv = idx._embed_query(query)
        scores = []
        for label, anchor in idx._anchors.items():
            s = float(_cosine_similarity(qv, anchor))
            scores.append((label, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_l, best_s = scores[0]
        second_s = scores[1][1]
        ratio = best_s / second_s if second_s > 0 else 999
        print(f"{query:30s} | {expected:10s} | {'FALLBACK':10s} | {best_s:.3f}  | {second_s:.3f}  | {ratio:4.1f}x  | (best={best_l})")
        fallback += 1
    else:
        label, goal_type, conf = result
        # Also get ratio for display
        qv = idx._embed_query(query)
        scores = []
        for lab, anchor in idx._anchors.items():
            s = float(_cosine_similarity(qv, anchor))
            scores.append((lab, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        ratio = scores[0][1] / scores[1][1] if scores[1][1] > 0 else 999

        status = "OK" if label == expected else f"WRONG({label})"
        if label == expected:
            ok += 1
        else:
            wrong += 1
        print(f"{query:30s} | {expected:10s} | {goal_type:10s} | {scores[0][1]:.3f}  | {scores[1][1]:.3f}  | {ratio:4.1f}x  | {status}")

print("-" * 100)
print(f"OK={ok}  Wrong={wrong}  Fallback={fallback}  Total={len(tests)}")
print(f"Embedding hit rate: {ok+wrong}/{len(tests)} ({(ok+wrong)/len(tests)*100:.0f}%)")
if ok + wrong > 0:
    print(f"Accuracy (of hits): {ok}/{ok+wrong} ({ok/(ok+wrong)*100:.0f}%)")
print(f"Threshold: ratio > {CONFIDENCE_RATIO}x, floor > {MIN_SCORE_FLOOR}")
