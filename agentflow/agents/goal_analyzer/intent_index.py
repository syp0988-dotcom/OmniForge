"""Offline-first embedding-based intent matching.

The goal analyzer uses this as a cheap fast path before falling back to the
LLM.  It deliberately defaults to the local TF-IDF embedder so tests and normal
startup never try to download sentence-transformers models.  Set
``AGENTFLOW_INTENT_EMBEDDER=semantic`` to opt into the semantic embedder.
"""

from __future__ import annotations

import os

import numpy as np

from agentflow.utils.logging import build_logger

logger = build_logger("intent_index")

INTENT_LABEL_TO_GOAL_TYPE: dict[str, str] = {
    "coding": "coding",
    "project": "project",
    "question": "question",
    "search": "search",
    "tool": "tool_use",
    "chat": "other",
}

INTENT_DESCRIPTIONS: dict[str, str] = {
    "coding": (
        "编写代码 实现功能 写函数 写类 修复 bug 调试错误 排查问题 重构代码 "
        "优化性能 改善代码结构 登录页面 接口 后端 前端"
    ),
    "project": (
        "创建完整项目 搭建系统 构建多文件应用 生成项目模板 初始化脚手架 "
        "创建 React Vue FastAPI Flask 应用 图书管理系统 管理后台"
    ),
    "question": (
        "知识问答 询问概念 理解原理 请求解释 分析方案 翻译文本 学习技术 "
        "什么是 为什么 怎么理解 帮我写文章"
    ),
    "search": (
        "搜索实时信息 查询最新新闻 动态 天气 价格 互联网数据 当前热点 "
        "今天 明天 现在 最新 AI 新闻"
    ),
    "tool": (
        "使用工具 执行命令 操作文件 git 操作 数据库操作 shell 命令 运行脚本 "
        "自动化流程 提交代码 查看状态"
    ),
    "chat": (
        "闲聊 打招呼 问候 自我介绍 测试对话连接 没有明确技术目标 你好 谢谢 "
        "好的 简单确认"
    ),
}

CONFIDENCE_RATIO = 1.5
MIN_SCORE_FLOOR = 0.20


class IntentIndex:
    """Lazily initialized six-class intent matcher."""

    def __init__(self) -> None:
        self._embedder: object | None = None
        self._anchors: dict[str, np.ndarray] = {}
        self._ready = False

    def match(self, question: str) -> tuple[str, str, float] | None:
        """Return ``(label, goal_type, confidence)`` for confident matches."""
        self._ensure_ready()
        if not self._ready or not question.strip():
            return None

        query_vec = self._embed_query(question)
        scores = [
            (label, float(_cosine_similarity(query_vec, anchor_vec)))
            for label, anchor_vec in self._anchors.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores:
            return None

        best_label, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        if best_score < MIN_SCORE_FLOOR:
            logger.info(
                "Score below floor (%.3f < %.2f) for '%s'; fallback to LLM",
                best_score,
                MIN_SCORE_FLOOR,
                question[:60],
            )
            return None

        ratio = best_score / second_score if second_score > 0 else 999.0
        if ratio < CONFIDENCE_RATIO:
            logger.info(
                "Low ratio (%.2fx < %.1fx, best=%.3f, second=%.3f) for '%s'; fallback to LLM",
                ratio,
                CONFIDENCE_RATIO,
                best_score,
                second_score,
                question[:60],
            )
            return None

        goal_type = INTENT_LABEL_TO_GOAL_TYPE.get(best_label, "other")
        logger.info(
            "Intent matched: label=%s goal_type=%s ratio=%.1fx score=%.3f query='%s'",
            best_label,
            goal_type,
            ratio,
            best_score,
            question[:60],
        )
        return best_label, goal_type, best_score

    def _ensure_ready(self) -> None:
        """Build the tiny anchor index on first use."""
        if self._ready:
            return
        try:
            from agentflow.knowledge.embedder import SemanticEmbedder, TfidfEmbedder

            if os.getenv("AGENTFLOW_INTENT_EMBEDDER", "tfidf").lower() == "semantic":
                embedder = SemanticEmbedder()
            else:
                embedder = TfidfEmbedder().fit(list(INTENT_DESCRIPTIONS.values()))

            vectors = embedder.embed(list(INTENT_DESCRIPTIONS.values()))
            self._anchors = {
                label: vec
                for (label, _), vec in zip(INTENT_DESCRIPTIONS.items(), vectors)
            }
            self._embedder = embedder
            self._ready = True
            logger.info("IntentIndex ready: %d intents, dim=%d", len(self._anchors), vectors[0].shape[0])
        except Exception as exc:
            logger.warning("IntentIndex init failed: %s; embedding match disabled", exc)
            self._ready = False

    def _embed_query(self, question: str) -> np.ndarray:
        embedder: object = self._embedder
        return embedder.embed([question])[0]  # type: ignore[union-attr]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two one-dimensional vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
