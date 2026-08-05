# OmniForge

OmniForge is a modular AI agent workspace built with Python, FastAPI, and LangGraph. It orchestrates specialized agents — planning, search, knowledge retrieval, code execution, reflection, and memory — through a dynamic task-queue workflow. OmniForge is the successor to AgentFlow, redesigned with a developer-first AI workspace experience.

## System Overview

OmniForge receives a user query, understands intent via a **hybrid embedding + LLM goal analyzer**, plans executable tasks, dispatches them to the right agent (knowledge, search, code, or tools), evaluates results, and synthesizes a final answer — all tracked through conversational memory.

![Main Chat Interface](screenshots/chat-interface.png)

## Agent Workflow

```mermaid
flowchart TD
    CM[Conversation Manager] --> GA[Goal Analyzer]
    GA -- conversational --> AG[Answer Generator]
    GA -- task-oriented --> PL[Planner]
    KR[Knowledge Retriever] --> PL
    PL --> QW[Query Rewriter] --> SE[Web Search]
    PL --> PE[Python Executor]
    PL --> TE[Tool Executor]
    SE --> RF[Reflection Evaluator]
    PE --> RF
    TE --> RF
    RF -- replan --> PL
    RF -- retry --> TE
    RF -- done --> AG
    AG --> MM[Conversation Memory] --> END
```

![Agent Workflow Execution](screenshots/workflow-execution.png)

## Agent Matrix

| Agent | Role | Decision Mode |
|---|---|---|
| **Goal Analyzer** | Hybrid intent classification (embedding + LLM fallback), 6 goal types | Embedding-first, LLM fallback |
| **Planner** | Dynamic task queue generation (3–5 tasks/cycle), blueprint & template support | Blueprint → Template → Function-call → JSON |
| **Knowledge Retriever** | Hybrid RAG: vector (Qdrant) + lexical (SQLite FTS5) with RRF fusion | Parallel retrieval + score fusion |
| **Web Search** | Real-time search via DuckDuckGo / Tavily | Pluggable provider |
| **Python Executor** | Sandboxed subprocess code execution | Configurable safety |
| **Tool Executor** | Central dispatch for filesystem, git, database, browser, DOCX, MCP tools | Plugin registry auto-discovery |
| **Reflection Evaluator** | Task result evaluation and routing decision | Rule-first, LLM fallback |
| **Answer Generator** | Synthesizes final answer from all agent outputs | Context-aware prompt adaptation |
| **Conversation Manager** | Session state, slot filling, anaphora resolution, rewrite | Rule-based pipeline |
| **Conversation Memory** | Cross-turn history and entity tracking | Lightweight state tracking |

## Knowledge RAG Pipeline

The retrieval-augmented generation system supports PDF, DOCX, TXT, MD, HTML, XLSX, PPTX, CSV, EPUB, and source code files with structure-aware chunking.

```
Document → Parser → Chunker → Qwen Embedder (v2/v3)
                                   ├── Qdrant (COSINE vector index)
                                   └── SQLite FTS5 (lexical index)
                                         ↓
                              Hybrid Retriever (RRF fusion)
                                   ↓
                              Top-K Results + Scores
```

![Knowledge Base Management](screenshots/knowledge-base.png)

- **Parser**: Multi-format document reader (pypdf, python-docx, openpyxl, python-pptx, BeautifulSoup, ebooklib)
- **Chunker**: Structure-aware strategies — paragraph, markdown (heading-preserving), code (function boundaries), table, slide
- **Embedder**: Qwen text-embedding-v2/v3 via DashScope (1024/1536-d, dimension auto-detected)
- **Retrieval**: Reciprocal Rank Fusion (RRF) combining vector (α=0.7) and lexical (β=0.3) scores; jieba CJK segmentation
- **Evaluation**: Built-in eval framework with recall@k, precision@k, ndcg@k, hit@k, MRR metrics

## Tool System

All tools follow a plugin architecture — extend `BaseTool`, register in the auto-discovery path, and the system picks up actions, schemas, and routing automatically.

| Tool | Capabilities |
|---|---|
| **FileSystem** | mkdir, write_file, read_file, edit_file, delete_file, list_files |
| **Search** | web.search via DuckDuckGo / Tavily |
| **Python** | Sandboxed subprocess execution |
| **Git** | status, diff, add, commit, branch, log |
| **Database** | SQL query execution |
| **Browser** | Browser automation |
| **DOCX** | Create formatted Word reports |
| **MCP** | MCP protocol integration |
| **Composio** | 500+ app integrations via Composio API |

## Project Structure

```
agentflow/
  agents/          Agent implementations (9 agents + base protocol)
  api/             FastAPI route handlers (split by domain: chat, knowledge,
                   sessions, files/workspace, models, memory, executions, system)
  app/             Application entry point (FastAPI app, CORS, startup tasks)
  blueprints/      YAML-based project scaffolding (Jinja2 templates)
  config/          Pydantic settings (.env), prompt templates, termination policy
  conversation/    Session state, context rewrite, conversation manager,
                   layered-memory compression
  database/        SQLite persistence (sessions, chats, documents, FTS5)
  eval/            Offline eval suites + runtime feedback collection (eval loop)
  knowledge/       RAG pipeline (parser, chunker, embedder, index, retriever, eval)
  models/          Pydantic models (chat, model_config)
  services/        LLM service, search, memory, file proposer
  tools/           Plugin tool implementations (10 tools)
  graph/           LangGraph workflow (nodes, edges, executor, context)
  utils/           Logging, decorators
frontend/          Vue 3 + TypeScript + Vite SPA (TailwindCSS, markdown-it)
tests/             Pytest suite (13+ test files)
```

## API Endpoints

| Category | Endpoints |
|---|---|
| **Chat** | `POST /chat`, `POST /chat/stream` (SSE) |
| **Knowledge** | `POST /upload`, `GET/DELETE /knowledge/documents`, `POST /knowledge/search` |
| **Sessions** | `POST /sessions/create`, `GET/PUT/DELETE /sessions/{id}`, `GET /history` |
| **Files** | `POST /files/create`, `POST /files/read`, `GET /files` |
| **Models** | `GET/POST/PUT/DELETE /models`, `POST /models/{id}/activate` |
| **Agents** | `GET /agents` |
| **Tools** | `GET /tools`, `GET /tools/capabilities`, `GET /tools/executor` |
| **Memory** | `GET/DELETE /memory`, `GET /memory/search` |
| **Workspace** | `GET/POST /workspace`, `POST /workspace/create-folder`, `GET /workspace/browse` |

## Quickstart

```bash
# 1. Clone and configure
git clone <repo-url> && cd multi_agent
cp .env.example .env   # fill in DEEPSEEK_API_KEY and DASHSCOPE_API_KEY

# 2. Install and run
uv sync
uv run uvicorn agentflow.app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open frontend
cd frontend && npm install && npm run dev
```

## Docker Deployment

```bash
docker compose up --build -d
```

The root [`docker-compose.yml`](docker-compose.yml) runs two services:

- **app** — the API (root [`Dockerfile`](Dockerfile), non-root user,
  healthcheck). Persistent volumes keep the SQLite database, local Qdrant
  index, embedding cache, uploaded/knowledge documents and logs across
  container recreation.
- **web** — nginx serving the built Vue SPA and reverse-proxying the API
  (`/chat`, `/chat/stream` with SSE, `/knowledge`, ...). HTTPS on port 443
  requires certificates in `deploy/nginx/certs/`
  (see [certs/README.md](deploy/nginx/certs/README.md)); for a no-certificate
  test build use `docker compose build --build-arg NGINX_CONF=nginx-http.conf web`.

`agentflow/docker/` is the older single-service compose layout kept for local
testing. The `deploy/k8s/` manifests are the Kubernetes reference:
`configmap.yaml` (non-secret settings), `pvc.yaml` + volume mounts (persistent
RAG/database data), `ingress.yaml` (TLS termination), and the `agentflow-web`
nginx deployment for the frontend.

### Production environment checks

- Set `APP_ENV=production` (or `ENFORCE_REQUIRED_ENV=true`): startup then fails
  fast when `DEEPSEEK_API_KEY` / `EMBEDDING_API_KEY` are missing.
- Logs are JSON-formatted with `trace_id` and rotate daily
  (`LOG_ROTATION_WHEN`, `LOG_ROTATION_BACKUP_COUNT`).
- If Qdrant or the embedding API is unavailable at runtime, knowledge
  retrieval automatically degrades to lexical (FTS5) search.

## Development

```bash
uv sync --dev                               # install runtime + dev dependencies (pytest, ruff)
uv run python -m pytest -q                  # full suite
uv run python -m pytest tests/test_workflow.py -q  # single file
uv run ruff check agentflow tests           # lint
```

## Runtime Feedback Loop

Every chat turn is a potential evaluation example. The chat endpoints append
structured records (outcome, goal type, errors, question/answer) to
`data/feedback/feedback.jsonl` — failures always, successful project/coding
completions as well. Export and review them with:

```bash
python scripts/export_feedback.py --out data/feedback/failure_cases.json
```

Feed reviewed cases back into the offline eval datasets
(`agentflow/eval/*`) so tuning decisions are backed by real usage, not just
hand-written fixtures.

## Key Configuration

The system degrades gracefully when optional API keys are missing (a startup
log lists exactly which capabilities are affected):

| Env var | Enables | When missing |
|---|---|---|
| `DEEPSEEK_API_KEY` | Core LLM | Everything degrades |
| `EMBEDDING_API_KEY` | RAG vector search, intent fast path | Lexical-only retrieval + LLM intent |
| `TAVILY_API_KEY` | Structured web search | DuckDuckGo fallback |
| `COMPOSIO_API_KEY` | 500+ app integrations | Tool reports "not configured" |

Agent loop guards are tunable via `MAX_PLANNER_CYCLES`, `MAX_REPLAN_COUNT`,
`MAX_STUCK_ROUNDS`, `REFLECTOR_PLANNER_CYCLE_CAP`; layered memory via
`ENABLE_HISTORY_COMPRESSION`, `HISTORY_TOKEN_BUDGET`,
`HISTORY_MIN_KEEP_MESSAGES`. See [.env.example](.env.example) for all options.

## Screenshots

Replace the placeholder images in `screenshots/` with your own captures.

| Screenshot | File |
|---|---|
| Main Chat Interface | [screenshots/chat-interface.png](screenshots/chat-interface.png) |
| Agent Workflow Execution | [screenshots/workflow-execution.png](screenshots/workflow-execution.png) |
| Knowledge Base Management | [screenshots/knowledge-base.png](screenshots/knowledge-base.png) |
| Session & History Management | [screenshots/session-management.png](screenshots/session-management.png) |
| Model Configuration | [screenshots/model-settings.png](screenshots/model-settings.png) |
