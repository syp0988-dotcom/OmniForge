"""Agent Protocol — interface contract for all agents in AgentFlow.

All agents implicitly conform to ``AgentProtocol`` through structural typing.
Explicitly inheriting from ``AgentProtocol`` enables ``isinstance`` checks
and makes the contract visible in the class hierarchy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """Interface contract for all agents in the system.

    Every agent must implement ``run(state)`` which receives the current
    workflow state dict and returns the updated state dict.
    """

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the workflow state and return the updated state.

        Args:
            state: The current workflow state dict. Must not be mutated
                   in place — return a new or updated dict instead.

        Returns:
            Updated workflow state dict.
        """
        ...
