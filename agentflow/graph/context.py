"""WorkflowContext — core data object for the Agent Framework.

WorkflowContext is a dict subclass that flows through the entire workflow.
It provides:
  - Full dict compatibility (LangGraph passes it as state)
  - Type-safe property access for well-known fields
  - Structured Task management (``add_task``, ``tasks``)
  - Structured Event management (``add_event``, ``events``)
  - ``to_dict()`` for safe serialization to API responses

Principles:
  - Agents receive and mutate the context (they do NOT replace it)
  - All workflow state lives in one object — no ad-hoc globals
  - Backward compatible: ``context["key"]`` and ``context.get("key")`` work
"""

from __future__ import annotations

from typing import Any

from agentflow.conversation.session_state import SessionState
from agentflow.graph.task import Task


class WorkflowContext(dict):
    """Core data object — dict subclass for full LangGraph compatibility.

    Usage::

        ctx = WorkflowContext({"question": "hello"})
        ctx.category = "identity"          # property setter
        ctx["answer"] = "Hi there"         # dict setitem (also works)
        ctx.add_task(Task(goal="greet"))
        for event in ctx.events: ...        # Runtime event log

    Versioning:
      The ``version`` field (default ``"1.0"``) tracks the context schema
      version.  Bump it when making backward-incompatible changes so that
      external consumers (e.g. frontend, logging pipeline) can adapt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Ensure default version is set
        if "version" not in self:
            self["version"] = "1.0"

    # -- Version ----------------------------------------------------------------

    @property
    def version(self) -> str:
        """Context schema version.  Default ``"1.0"``."""
        return str(self.get("version", "1.0"))

    @version.setter
    def version(self, value: str) -> None:
        self["version"] = value

    # -- Serialization ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce a plain dict safe for API responses and logging.

        Task / Plan / Event objects are serialised via ``.to_dict()``.
        """
        result: dict[str, Any] = {}
        for k, v in self.items():
            if k == "tasks":
                result[k] = [t.to_dict() if isinstance(t, Task) else t for t in v]
            elif k == "events":
                result[k] = [_e_to_dict(e) for e in v]
            elif k == "session_state" and isinstance(v, SessionState):
                result[k] = v.to_dict()
            elif isinstance(v, Task):
                result[k] = v.to_dict()
            elif hasattr(v, "to_dict") and callable(v.to_dict):
                # Duck-type serialization for Plan, Event, etc.
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    # -- Properties: well-known fields ------------------------------------------

    @property
    def question(self) -> str:
        """The user's original question / message."""
        return str(self.get("question", ""))

    @question.setter
    def question(self, value: str) -> None:
        self["question"] = value

    @property
    def category(self) -> str:
        """Query category assigned by RouterAgent."""
        return str(self.get("category", "reasoning"))

    @category.setter
    def category(self, value: str) -> None:
        self["category"] = value

    @property
    def answer(self) -> str:
        """Final answer produced by AnswerAgent."""
        return str(self.get("answer", ""))

    @answer.setter
    def answer(self, value: str) -> None:
        self["answer"] = value

    @property
    def plan(self) -> dict[str, Any]:
        """Workflow plan produced by PlannerAgent."""
        return self.get("plan", {})

    @plan.setter
    def plan(self, value: dict[str, Any]) -> None:
        self["plan"] = value

    @property
    def workflow(self) -> list[str]:
        """List of node names the workflow will traverse."""
        return self.get("workflow", [])

    @workflow.setter
    def workflow(self, value: list[str]) -> None:
        self["workflow"] = value

    @property
    def history(self) -> list[dict[str, str]]:
        """Conversation history (list of {role, content})."""
        return self.get("history", [])

    @history.setter
    def history(self, value: list[dict[str, str]]) -> None:
        self["history"] = value

    @property
    def memory(self) -> dict[str, Any]:
        """Memory object maintained by MemoryAgent."""
        return self.get("memory", {})

    @memory.setter
    def memory(self, value: dict[str, Any]) -> None:
        self["memory"] = value

    @property
    def search_results(self) -> list[dict[str, Any]]:
        """Search results from SearchAgent."""
        return self.get("search_results", [])

    @search_results.setter
    def search_results(self, value: list[dict[str, Any]]) -> None:
        self["search_results"] = value

    @property
    def knowledge_context(self) -> str:
        """Knowledge base context assembled by KnowledgeAgent."""
        return str(self.get("knowledge_context", ""))

    @knowledge_context.setter
    def knowledge_context(self, value: str) -> None:
        self["knowledge_context"] = value

    @property
    def python_result(self) -> dict[str, Any]:
        """Result from Python code execution."""
        return self.get("python_result", {})

    @python_result.setter
    def python_result(self, value: dict[str, Any]) -> None:
        self["python_result"] = value

    @property
    def router(self) -> dict[str, Any]:
        """Router metadata (e.g. {"category": "search"})."""
        return self.get("router", {})

    @router.setter
    def router(self, value: dict[str, Any]) -> None:
        self["router"] = value

    # -- Conversation Context (Phase 7) -----------------------------------------

    @property
    def conversation_context(self) -> Any:
        """Structured conversation context for the current turn.

        Contains turn type (NEW_TASK / FOLLOW_UP / OPTION_SELECTION / etc.),
        rewritten question, entities, summary, and more.
        """
        return self.get("conversation_context")

    @conversation_context.setter
    def conversation_context(self, value: Any) -> None:
        self["conversation_context"] = value

    # -- Session State (Conversation Runtime) -----------------------------------

    @property
    def session_state(self) -> SessionState:
        """Runtime session state — what the system is currently doing.

        Returns a ``SessionState`` object (never *None*).  The session state
        persists across turns and enables continuation planning.
        """
        raw = self.get("session_state")
        if isinstance(raw, SessionState):
            return raw
        if isinstance(raw, dict):
            return SessionState.from_dict(raw)
        return SessionState()

    @session_state.setter
    def session_state(self, value: SessionState | dict[str, Any]) -> None:
        if isinstance(value, SessionState):
            self["session_state"] = value
        else:
            self["session_state"] = SessionState.from_dict(value)

    # -- Task management --------------------------------------------------------

    @property
    def tasks(self) -> list[Task]:
        """All tasks created during this workflow invocation."""
        return self.setdefault("tasks", [])

    def add_task(self, task: Task) -> None:
        """Append a task and register it in the context.

        The task is also stored under ``tasks/`` for dict-based access.
        """
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Task | None:
        """Look up a task by its ID."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    # -- Event management -------------------------------------------------------

    @property
    def events(self) -> list[Any]:
        """All runtime events emitted during this workflow invocation.

        Each event is an ``Event`` dataclass instance (or a plain dict).
        Use ``EventBus`` to emit typed events.
        """
        return self.setdefault("events", [])

    def add_event(self, event: Any) -> None:
        """Register an event (``Event`` dataclass or object with ``to_dict``)."""
        self.events.append(event)

    def clear_events(self) -> None:
        """Remove all events (useful in tests / replay scenarios)."""
        self["events"] = []


# -- Internal helpers ---------------------------------------------------------


def _e_to_dict(e: Any) -> dict[str, Any]:
    """Convert an event-like object to a dict."""
    if isinstance(e, dict):
        return e
    if hasattr(e, "to_dict"):
        return e.to_dict()
    return {"type": str(type(e).__name__), "value": str(e)}
