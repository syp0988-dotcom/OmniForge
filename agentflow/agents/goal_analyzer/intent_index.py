"""Embedding-based intent matching (fast path before the LLM).

The goal analyzer uses this as a cheap fast path before falling back to the
LLM. Anchors are embedded once with the configured Qwen embedder (results are
cached by the embedding cache); each query is compared against the anchors
with cosine similarity.  When the embedder is unavailable (e.g. no API key),
``available`` is False and callers should fall back to the LLM explicitly
rather than silently degrading.
"""

from __future__ import annotations


import numpy as np

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger

logger = build_logger("intent_index")

INTENT_LABEL_TO_GOAL_TYPE: dict[str, str] = {
    "coding": "coding",
    "project": "project",
    "question": "question",
    "search": "search",
    "tool": "tool_use",
    "chat": "other",
    "analysis": "analysis",
    "document": "document",
    "translation": "translation",
    "editing": "editing",
}

INTENT_ANCHORS: dict[str, list[str]] = {
    "coding": [
        "编写代码", "实现功能", "写一个函数", "写一个类", "修复 bug",
        "调试错误", "排查问题", "重构代码", "优化代码性能", "改善代码结构",
        "帮我写一个 Python 冒泡排序函数", "这段代码有个空指针异常，帮我修复",
        "帮我重构这个 UserService 类", "写一个登录页面的前后端接口",
        "怎么优化这个循环的性能", "用 Python 写一个爬虫抓取网页",
        "为什么我的代码跑不起来", "写一个 Vue 组件", "实现一个接口",
        "write a Python function", "fix this bug", "refactor this class",
        "implement a binary search tree in Java", "optimize this loop",
    ],
    "project": [
        "创建完整项目", "搭建系统", "构建多文件应用", "生成项目模板",
        "初始化脚手架", "创建一个 FastAPI 图书管理系统项目", "帮我搭建一个管理后台系统",
        "初始化一个 React 前端脚手架项目", "创建一个 Python 爬虫项目，包含多个模块",
        "帮我搭建一个博客系统，包含前后端", "开发一个电商网站",
        "create a full-stack blog project", "build a React admin dashboard",
        "scaffold a FastAPI project",
    ],
    "question": [
        "知识问答", "询问概念", "理解原理", "请求解释", "学习技术",
        "什么是 Kubernetes", "解释一下什么是微服务架构", "为什么 TCP 要三次握手",
        "Python 和 Java 有什么区别", "介绍一下 Transformer 模型",
        "怎么理解依赖注入", "帮我写一篇关于 AI 的文章",
        "what is Kubernetes", "explain microservices",
        "what is the difference between Python and Java",
    ],
    "search": [
        "搜索实时信息", "查询最新新闻", "互联网数据", "当前热点",
        "今天上海天气怎么样", "帮我查一下最新的 AI 新闻", "今天有什么热点新闻",
        "查询一下今天的股价", "帮我搜索一下房价", "比特币现在多少钱",
        "what is the weather in Shanghai today", "latest AI news",
        "bitcoin price now",
    ],
    "tool": [
        "使用工具", "执行命令", "操作文件", "shell 命令", "运行脚本",
        "自动化流程", "git status 看一下", "帮我提交代码",
        "运行一下这个脚本看看结果", "用 shell 批量重命名文件",
        "连接数据库查询数据", "查看一下当前目录的文件列表",
        "run this script", "git status", "execute this command",
        "list files in this directory",
    ],
    "chat": [
        "闲聊", "打招呼", "问候", "自我介绍", "测试对话连接",
        "你好", "在吗", "谢谢", "你是谁", "介绍一下你自己",
        "好的", "随便聊聊", "再见",
        "hello", "who are you", "thanks", "goodbye",
    ],
    "analysis": [
        "分析问题", "评估方案", "对比优缺点", "分析数据", "解读结果",
        "原因分析", "影响分析", "趋势分析", "可行性分析",
        "分析一下这两个方案各自的优缺点", "这份数据报告帮我分析一下趋势",
        "为什么用户流失率上升", "评估一下这个方案的可行性",
    ],
    "document": [
        "整理文档", "生成报告", "撰写文档", "制作表格", "整理笔记",
        "会议纪要", "项目文档", "论文", "简历", "文档模板",
        "整理一下上周的会议纪要", "帮我写一份项目周报", "生成一份 Word 报告",
    ],
    "translation": [
        "翻译成英文", "翻译成中文", "中译英", "英译中", "翻译这段文字",
        "帮我把这段文字翻译成日语", "把标题翻译成英文", "翻译一下",
    ],
    "editing": [
        "润色", "改写", "修改措辞", "调整语气", "语法检查", "校正文案",
        "编辑文章", "优化文案", "帮我润色一下这段自我介绍",
        "帮我把这几段话改得更正式一些", "检查一下这段文字的语法错误",
    ],
}


class IntentIndex:
    """Lazily initialized multi-anchor intent matcher.

    Each intent label has multiple anchor phrases (keywords plus representative
    real queries).  A query is scored per label by its best cosine similarity
    across that label's anchors, which is far more robust than comparing a
    single keyword-soup vector.
    """

    def __init__(
        self,
        anchors: dict[str, list[str]] | None = None,
        embedder: object | None = None,
    ) -> None:
        self._anchor_defs = anchors or INTENT_ANCHORS
        self._embedder = embedder
        self._anchor_vectors: dict[str, list[np.ndarray]] = {}
        self._ready = False

    def match(self, question: str) -> tuple[str, str, float] | None:
        """Return ``(label, goal_type, confidence)`` for confident matches."""
        self._ensure_ready()
        if not self._ready or not question.strip():
            return None

        query_vec = self._embed_query(question)
        label_scores = _score_labels(query_vec, self._anchor_vectors)
        if not label_scores:
            return None
        ranked = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)

        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < settings.intent_min_score_floor:
            logger.info(
                "Score below floor (%.3f < %.2f) for '%s'; fallback to LLM",
                best_score,
                settings.intent_min_score_floor,
                question[:60],
            )
            return None

        ratio = best_score / second_score if second_score > 0 else 999.0
        if ratio < settings.intent_confidence_ratio:
            logger.info(
                "Low ratio (%.2fx < %.1fx, best=%.3f, second=%.3f) for '%s'; fallback to LLM",
                ratio,
                settings.intent_confidence_ratio,
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

    @property
    def available(self) -> bool:
        """True when the embedding fast path is usable."""
        return self._ready

    def _ensure_ready(self) -> None:
        """Embed all anchor phrases and group vectors per label."""
        if self._ready:
            return
        try:
            from agentflow.knowledge.embedder import QwenEmbedder

            embedder = self._embedder or QwenEmbedder()
            texts: list[str] = []
            labels: list[str] = []
            for label, phrases in self._anchor_defs.items():
                for phrase in phrases:
                    texts.append(phrase)
                    labels.append(label)
            if not texts:
                return
            vectors = embedder.embed(texts, batch_size=20)
            self._anchor_vectors = {}
            for label, vec in zip(labels, vectors):
                self._anchor_vectors.setdefault(label, []).append(vec)
            self._embedder = embedder
            self._ready = True
            logger.info(
                "IntentIndex ready: %d intents, %d anchors, dim=%d",
                len(self._anchor_vectors), len(texts), vectors[0].shape[0],
            )
        except Exception as exc:
            logger.warning("IntentIndex init failed: %s; embedding match disabled", exc)
            self._ready = False

    def _embed_query(self, question: str) -> np.ndarray:
        embedder: object = self._embedder
        return embedder.embed([question])[0]  # type: ignore[union-attr]


def _score_labels(
    query_vec: np.ndarray,
    anchor_vectors: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """Per-label score = best (max) cosine across the label's anchors."""
    scores: dict[str, float] = {}
    for label, vecs in anchor_vectors.items():
        if not vecs:
            continue
        scores[label] = max(
            float(_cosine_similarity(query_vec, vec)) for vec in vecs
        )
    return scores


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
