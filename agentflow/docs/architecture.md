# Architecture

AgentFlow follows a modular, layered architecture:

- API layer exposes FastAPI endpoints.
- Graph layer composes agent nodes through LangGraph.
- Agent layer encapsulates planner, search, knowledge, Python, report, and memory behaviors.
- Infrastructure layer manages persistence, configuration, logging, and deployment.
