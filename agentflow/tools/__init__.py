"""Tool abstractions for AgentFlow.

All tools implement ``BaseTool`` with a uniform ``execute(**kwargs)``
interface that the Executor calls directly — no adapters needed.
"""

from agentflow.tools.base import BaseTool

__all__ = [
    "BaseTool",
]
