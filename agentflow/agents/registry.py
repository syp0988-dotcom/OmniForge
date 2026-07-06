"""Agent registry — central inventory of all available agents in the system.

Each agent is registered with structured metadata (name, key, description,
category, capabilities, status). This enables the API /agents endpoint to
serve structured agent info to the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class AgentInfo:
    """Immutable metadata for a single agent."""

    key: str
    name: str
    description: str
    category: str
    status: str  # "active" | "inactive"
    capabilities: list[str]
    module_path: str


_registry: dict[str, AgentInfo] = {}


def register(info: AgentInfo) -> None:
    """Register an agent. Idempotent — later calls overwrite earlier ones."""
    _registry[info.key] = info


def get_all() -> list[dict[str, Any]]:
    """Return all registered agents as plain dicts, active first."""
    agents = [asdict(info) for info in _registry.values()]
    agents.sort(key=lambda a: (0 if a["status"] == "active" else 1, a["name"]))
    return agents


def get(key: str) -> AgentInfo | None:
    """Get a single agent by key."""
    return _registry.get(key)


# ---- Populate registry ------------------------------------------------ #

register(AgentInfo(
    key="router",
    name="Query Router",
    description="Classifies user queries by intent using regex pattern matching",
    category="routing",
    status="active",
    capabilities=["intent classification", "pattern matching"],
    module_path="agentflow.agents.router.agent",
))

register(AgentInfo(
    key="planner",
    name="Workflow Planner",
    description="Defines the execution workflow steps based on query category",
    category="planning",
    status="active",
    capabilities=["workflow generation", "step sequencing"],
    module_path="agentflow.agents.planner.agent",
))

register(AgentInfo(
    key="knowledge",
    name="Knowledge Retriever",
    description="Retrieves relevant document chunks from the local knowledge base using TF-IDF",
    category="retrieval",
    status="active",
    capabilities=["document search", "TF-IDF retrieval", "semantic ranking"],
    module_path="agentflow.agents.knowledge.agent",
))

register(AgentInfo(
    key="search",
    name="Web Search Agent",
    description="Performs real-time web search via DuckDuckGo and returns structured results",
    category="search",
    status="active",
    capabilities=["web search", "result extraction"],
    module_path="agentflow.agents.search.agent",
))

register(AgentInfo(
    key="answer",
    name="Answer Generator",
    description="Synthesizes final answers using LLM with context from knowledge, search, and memory",
    category="generation",
    status="active",
    capabilities=["LLM completion", "context-aware answering", "multi-source synthesis"],
    module_path="agentflow.agents.answer.agent",
))

register(AgentInfo(
    key="memory",
    name="Conversation Memory",
    description="Maintains conversation history across turns for context-aware responses",
    category="memory",
    status="active",
    capabilities=["conversation tracking", "history accumulation", "context windowing"],
    module_path="agentflow.agents.memory.agent",
))

register(AgentInfo(
    key="python",
    name="Python Executor",
    description="Executes Python code in a sandboxed subprocess with timeout and output limits",
    category="execution",
    status="active",
    capabilities=["code execution", "subprocess sandbox", "output capture"],
    module_path="agentflow.agents.python.agent",
))

register(AgentInfo(
    key="report",
    name="Report Generator",
    description="Alternative answer generator that produces concise reports from workflow results",
    category="generation",
    status="inactive",
    capabilities=["report generation", "LLM summarization"],
    module_path="agentflow.agents.report.agent",
))

register(AgentInfo(
    key="project_structure",
    name="Project Structure Planner",
    description="Generates complete project directory trees from user requirements",
    category="planning",
    status="active",
    capabilities=["project scaffolding", "directory tree generation", "template matching"],
    module_path="agentflow.agents.project_structure_planner.agent",
))

register(AgentInfo(
    key="tool_executor",
    name="Tool Executor",
    description="Central dispatch for all tool tasks — routes Planner tasks to registered tools via ToolRegistry",
    category="execution",
    status="active",
    capabilities=["task dispatch", "tool lifecycle management", "batch execution"],
    module_path="agentflow.graph.executor",
))
