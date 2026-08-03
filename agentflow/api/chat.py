"""API router: chat endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.config.settings import settings
from agentflow.graph.workflow import build_workflow
from agentflow.models.chat import ChatRequest, ChatResponse
from agentflow.services.file_proposer import propose_files
from agentflow.utils.trace_context import set_trace_id
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
import asyncio
import json

from agentflow.api.routes import (
    get_store,
)

router = APIRouter()
logger = build_logger("api.chat")

@router.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests — delegates to the streaming generator internally.

    This endpoint is non-blocking (async) and streams the workflow execution
    in the background, returning only the final result after completion.
    Previously this was a blocking synchronous call — now it uses the same
    async infrastructure as ``/chat/stream`` for consistency.
    """
    from agentflow.conversation.session_state import SessionState
    from agentflow.graph.context import WorkflowContext

    # -- Session setup --
    trace_id = set_trace_id()
    logger.debug("[%s] Chat request: %s", trace_id, request.message[:80])

    session_id = request.session_id
    if session_id is None:
        sess = get_store().create_session()
        session_id = sess["id"]
    else:
        existing = get_store().get_session(session_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Session not found")

    history_dicts = [
        {"role": m.role, "content": m.content} for m in request.history
    ]

    # Load session_state from DB
    saved_state_str = get_store().get_session_state(session_id)
    session_state_dict = json.loads(saved_state_str) if saved_state_str else None

    workflow = build_workflow()

    initial_state: dict = {
        "question": request.message,
        "workflow": [],
        "task_queue": [],
        "history": history_dicts,
        "source_mode": request.source_mode,
        "trace_id": trace_id,
    }
    if history_dicts:
        initial_state["memory"] = {"history": list(history_dicts)}
    if session_state_dict:
        initial_state["session_state"] = SessionState.from_dict(session_state_dict)

    # Use astream to capture all node outputs (same approach as test_debug.py).
    # Each node in our graph returns the full state dict, so we accumulate
    # the latest output from each node as the graph progresses.
    try:
        final_state: dict | None = None
        stream = workflow.astream(initial_state)
        deadline = asyncio.get_running_loop().time() + settings.max_request_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            try:
                event = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            for node_name, state_update in event.items():
                # Each node returns the full state dict — capture the latest.
                final_state = dict(state_update)
                logger.debug("Node '%s' emitted keys: %s", node_name, list(state_update.keys())[:10])

        # After all nodes complete, final_state should hold the state after "memory" (the end node).
        answer = (final_state or {}).get("answer", "")
        error = (final_state or {}).get("error", "")

        logger.info("Chat final_state keys: %s", list((final_state or {}).keys())[:20])
        if not answer:
            logger.warning("Chat: answer is empty. error=%s keys=%s",
                           error, list((final_state or {}).keys())[:20])

        # Save execution record for observability
        _maybe_save_execution(final_state, session_id, request.message, answer)
        # Collect runtime feedback for the eval-driven iteration loop
        from agentflow.eval.feedback import record_chat_feedback
        record_chat_feedback(final_state, session_id, request.message, answer)

        # Persist messages
        get_store().add_message("user", request.message, session_id=session_id)
        if answer:
            get_store().add_message("assistant", answer, session_id=session_id)

        # Persist session_state
        if final_state:
            ctx = WorkflowContext(final_state)
            result_dict = ctx.to_dict()
            new_state = result_dict.get("session_state")
            if new_state and isinstance(new_state, dict):
                get_store().update_session_state(session_id, json.dumps(new_state, ensure_ascii=False))

        # Auto-title
        sess = get_store().get_session(session_id)
        if sess and sess["title"] == "新对话":
            title = request.message[:50]
            if len(request.message) > 50:
                title += "…"
            get_store().update_session_title(session_id, title)

        debug_data = {
            "goal": (final_state or {}).get("goal_analysis", {}),
            "category": (final_state or {}).get("category"),
            "workflow": (final_state or {}).get("workflow"),
            "search_results": (final_state or {}).get("search_results", []),
        }
        return ChatResponse(
            reply=answer,
            metadata={"status": "ok", "session_id": session_id},
            debug=debug_data,
            proposed_files=propose_files(answer),
        )
    except TimeoutError:
        logger.error("Chat request timed out after %ds", settings.max_request_seconds)
        raise HTTPException(status_code=504, detail="Request timed out") from None
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, raw_request: Request):
    """SSE streaming endpoint — yields events as workflow nodes complete.

    Now supports client-disconnect detection: when the frontend AbortController
    fires, the backend detects ``request.is_disconnected()`` between workflow
    nodes and stops early, skipping persistence and sending a ``cancelled``
    SSE event.

    Events::

        event: thinking\\n data: {{"phase": "分析问题", "category": "search"}}
        event: searching\\n data: {{"phase": "搜索网络信息"}}
        event: generating\\n data: {{"phase": "生成回答"}}
        event: cancelled\\n data: {{"reason": "用户中断了对话"}}
        event: done\\n data: {{"answer": "..."}}
    """
    from agentflow.conversation.session_state import SessionState
    from agentflow.graph.context import WorkflowContext

    async def _event_generator():
        # -- Helper: check client disconnect --
        async def _is_disconnected() -> bool:
            try:
                return await raw_request.is_disconnected()
            except Exception:
                return False

        # -- Session setup --
        stream_trace_id = set_trace_id()
        logger.debug("[%s] Stream request: %s", stream_trace_id, body.message[:80])

        session_id = body.session_id
        if session_id is None:
            sess = get_store().create_session()
            session_id = sess["id"]
        else:
            existing = get_store().get_session(session_id)
            if existing is None:
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found'})}\n\n"
                return

        history_dicts = [
            {"role": m.role, "content": m.content} for m in body.history
        ]

        # Load session_state from DB
        saved_state_str = get_store().get_session_state(session_id)
        session_state_dict = json.loads(saved_state_str) if saved_state_str else None

        workflow = build_workflow()

        initial_state: dict = {
            "question": body.message,
            "workflow": [],
            "task_queue": [],
            "history": history_dicts,
            "_stream_answer": True,
            "source_mode": body.source_mode,
            "trace_id": stream_trace_id,
        }
        if history_dicts:
            initial_state["memory"] = {"history": list(history_dicts)}
        if session_state_dict:
            initial_state["session_state"] = SessionState.from_dict(session_state_dict)

        # -- Emit immediate start event (before any LLM call) --
        yield _sse_event("start", {"phase": "正在处理请求..."})

        # -- Stream workflow execution --
        final_state: dict | None = None
        answer_text: str | None = None  # captured from answer node for chunked delivery
        streamed_answer = ""
        did_stream_answer = False
        cancelled: bool = False
        try:
            stream = workflow.astream(initial_state)
            deadline = asyncio.get_running_loop().time() + settings.max_request_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                # Check for client disconnect between nodes
                if await _is_disconnected():
                    cancelled = True
                    logger.info("Client disconnected during workflow execution")
                    break

                for node_name, state_update in event.items():
                    if node_name == "goal_analyzer":
                        goal = state_update.get("goal_analysis", {})
                        goal_type = goal.get("goal_type", "") if isinstance(goal, dict) else ""
                        yield _sse_event("thinking", {"phase": "分析用户目标", "goal_type": goal_type})
                    elif node_name == "planner":
                        yield _sse_event("planning", {"phase": "制定执行计划"})
                        yield _emit_task_update(state_update)
                    elif node_name == "knowledge":
                        yield _sse_event("searching", {"phase": "检索知识库"})
                    elif node_name == "query_rewriter":
                        yield _sse_event("searching", {"phase": "优化搜索查询"})
                        yield _emit_task_update(state_update)
                    elif node_name == "search":
                        yield _sse_event("searching", {"phase": "搜索网络信息"})
                        yield _emit_task_update(state_update)
                    elif node_name == "python":
                        yield _sse_event("executing", {"phase": "执行代码"})
                        yield _emit_task_update(state_update)
                    elif node_name == "tool_executor":
                        yield _emit_task_update(state_update)
                    elif node_name == "reflector":
                        yield _sse_event("thinking", {"phase": "检查执行结果"})
                        yield _emit_task_update(state_update)
                    elif node_name == "answer":
                        answer_text = state_update.get("answer", "")
                        yield _sse_event("generating", {"phase": "生成回答"})
                        yield _emit_task_update(state_update)
                        stream_messages = state_update.get("_answer_stream_messages")
                        if stream_messages and isinstance(stream_messages, list):
                            from agentflow.services.llm_service import get_llm_service

                            did_stream_answer = True
                            for token in get_llm_service().complete_stream(
                                messages=stream_messages,
                                max_tokens=settings.answer_max_tokens,
                            ):
                                if await _is_disconnected():
                                    cancelled = True
                                    logger.info("Client disconnected during LLM streaming")
                                    break
                                if token:
                                    streamed_answer += token
                                    yield _sse_event("text", {"text": token})
                            answer_text = streamed_answer.strip()
                            if cancelled:
                                break
                    elif node_name == "memory":
                        final_state = dict(state_update)
                if cancelled:
                    break
        except TimeoutError:
            logger.error("Stream timed out after %ds", settings.max_request_seconds)
            yield _sse_event("error", {"error": "请求超时，请重试"})
            return
        except Exception as exc:
            logger.exception("Streaming workflow failed")
            yield _sse_event("error", {"error": str(exc)})
            return

        # If cancelled, notify frontend and skip persistence
        if cancelled:
            yield _sse_event("cancelled", {"reason": "用户中断了对话"})
            return

        # -- Deliver answer text in chunks for true streaming feel --
        answer = answer_text or (final_state.get("answer", "") if final_state else "")
        if did_stream_answer:
            answer = answer.strip()
            if final_state is not None:
                final_state["answer"] = answer

        # Stream answer text incrementally
        if answer and not did_stream_answer:
            chunk_size = 15  # characters per chunk
            for i in range(0, len(answer), chunk_size):
                # Re-check disconnect during chunk delivery
                if await _is_disconnected():
                    logger.info("Client disconnected during chunk delivery")
                    yield _sse_event("cancelled", {"reason": "用户中断了对话"})
                    return
                chunk = answer[i:i + chunk_size]
                yield _sse_event("text", {"text": chunk})
                await asyncio.sleep(0.02)  # small delay for streaming effect

        # -- Save execution record for observability --
        _maybe_save_execution(final_state, session_id, body.message, answer)
        from agentflow.eval.feedback import record_chat_feedback
        record_chat_feedback(final_state, session_id, body.message, answer)

        # -- Persist messages --
        get_store().add_message("user", body.message, session_id=session_id)
        if answer:
            get_store().add_message("assistant", answer, session_id=session_id)

        # -- Persist session_state --
        if final_state:
            ctx = WorkflowContext(final_state)
            result_dict = ctx.to_dict()
            new_state = result_dict.get("session_state")
            if new_state and isinstance(new_state, dict):
                get_store().update_session_state(session_id, json.dumps(new_state, ensure_ascii=False))

        # -- Auto-title --
        sess = get_store().get_session(session_id)
        if sess and sess["title"] == "新对话":
            title = body.message[:50]
            if len(body.message) > 50:
                title += "…"
            get_store().update_session_title(session_id, title)

        done_data: dict[str, object] = {"answer": answer, "session_id": session_id}
        if final_state and final_state.get("_degraded"):
            done_data["degraded"] = True
            done_data["degraded_reason"] = str(final_state.get("_llm_error", "unknown"))
        # Final task queue sync — ensure frontend shows correct final state
        if final_state:
            final_tasks = _emit_task_update(final_state)
            if final_tasks:
                yield final_tasks
        yield _sse_event("done", done_data)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _maybe_save_execution(
    final_state: dict | None,
    session_id: int,
    question: str,
    answer: str,
) -> None:
    """Save execution record + final checkpoint when a final_state is available."""
    if not final_state:
        return
    try:
        goal_analysis = final_state.get("goal_analysis", {}) or {}
        goal_type = goal_analysis.get("goal_type", "") if isinstance(goal_analysis, dict) else ""
        trace = final_state.get("_trace", []) or []
        errors = final_state.get("_errors", []) or []
        degraded = bool(final_state.get("_degraded", False))
        trace_id = str(final_state.get("trace_id", "") or "")

        # Calculate total duration from trace
        duration_ms = 0.0
        if trace:
            durations = [t.get("duration_ms", 0) or 0 for t in trace]
            duration_ms = sum(durations)

        execution_id = get_store().save_execution(
            session_id=session_id,
            trace_id=trace_id,
            question=question,
            answer=answer,
            goal_type=goal_type,
            trace_json=json.dumps(trace, ensure_ascii=False),
            errors_json=json.dumps(errors, ensure_ascii=False),
            degraded=degraded,
            duration_ms=round(duration_ms, 2),
        )

        # Save final task queue checkpoint
        task_queue = final_state.get("task_queue", []) or []
        if task_queue and execution_id:
            get_store().save_checkpoint(
                execution_id=execution_id,
                node_name="__final__",
                task_queue_json=json.dumps(task_queue, ensure_ascii=False),
            )
    except Exception:
        logger.exception("Failed to save execution record (non-fatal)")


def _sse_event(event: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit_task_update(state_update: dict) -> str:
    """Extract task queue from workflow state and return an SSE ``task_update`` event.

    Returns empty string when the task_queue is empty.
    """
    queue = state_update.get("task_queue", []) or []
    if not queue:
        return ""
    tasks = [
        {
            "id": t.get("task_id", ""),
            "title": t.get("title", ""),
            "tool": t.get("tool", ""),
            "status": t.get("status", "todo"),
        }
        for t in queue
    ]
    return _sse_event("task_update", {"tasks": tasks})
