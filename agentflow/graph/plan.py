"""Plan — workflow plan produced by the PlannerAgent.

A Plan is a structured, executable blueprint for a workflow invocation.
Unlike a simple node list, a Plan contains concrete Task objects with:

  - goal (what to accomplish)
  - tool (which tool to invoke)
  - input (arguments for the tool)

The Executor consumes ``plan.tasks`` directly, routing each Task to the
correct Tool and managing its lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentflow.graph.task import Task


@dataclass
class Plan:
    """A structured workflow plan — a sequence of concrete Tasks.

    Attributes:
        goal: The overall goal this plan achieves (typically the user question).
        category: Query category from RouterAgent (e.g. "search", "identity").
        tasks: Ordered list of Task objects to execute.
        direct_answer: If True, answer directly without invoking any tool.
            Planner sets this when the question requires no tool capability.
        priority: Execution hint ("normal", "high", "low").
        reasoning: Human-readable explanation of why this plan was chosen.
        metadata: Extensible metadata for future use.
    """

    goal: str
    category: str
    tasks: list[Task] = field(default_factory=list)
    direct_answer: bool = False
    priority: str = "normal"
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Derived helpers -------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of tasks in this plan."""
        return len(self.tasks)

    @property
    def required_tools(self) -> list[str]:
        """Unique list of tool names required by this plan."""
        seen: set[str] = set()
        tools: list[str] = []
        for t in self.tasks:
            if t.tool and t.tool not in seen:
                seen.add(t.tool)
                tools.append(t.tool)
        return tools

    @property
    def active_agents(self) -> list[str]:
        """Unique list of agent names involved in this plan."""
        seen: set[str] = set()
        agents: list[str] = []
        for t in self.tasks:
            if t.agent and t.agent not in seen:
                seen.add(t.agent)
                agents.append(t.agent)
        return agents

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "category": self.category,
            "tasks": [t.to_dict() for t in self.tasks],
            "direct_answer": self.direct_answer,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "step_count": self.step_count,
            "required_tools": self.required_tools,
            "active_agents": self.active_agents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Restore a Plan from a dict produced by ``to_dict``."""
        tasks_raw = data.get("tasks", [])
        tasks = [Task.from_dict(t) if isinstance(t, dict) else t for t in tasks_raw]
        return cls(
            goal=data.get("goal", ""),
            category=data.get("category", ""),
            tasks=tasks,
            direct_answer=data.get("direct_answer", False),
            priority=data.get("priority", "normal"),
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {}),
        )
