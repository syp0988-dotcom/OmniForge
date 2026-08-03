"""API router: knowledge endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.config.settings import settings
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import io
import shutil

from agentflow.api.routes import (
    KNOWLEDGE_FILES_DIR,
    UPLOAD_DIR,
    _read_upload_limited,
    get_knowledge_store,
)
from agentflow.api.files_workspace import (
    _read_docx_preview,
    _read_pdf_preview,
    _read_pptx_preview,
    _read_xlsx_preview,
)

router = APIRouter()
logger = build_logger("api.knowledge")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document, parse it, and index it into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    allowed_types = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
        ".html", ".htm", ".xlsx", ".xls", ".pptx", ".csv", ".epub",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".zip",
    }
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Allowed: PDF, DOCX, TXT, MD, HTML, XLSX, PPTX, CSV, EPUB, 代码文件",
        )

    # Handle ZIP archives: extract and ingest each file
    if ext == ".zip":
        return await _handle_zip_upload(file)

    # Save uploaded file to permanent knowledge storage
    safe_filename = Path(file.filename).name
    temp_path = UPLOAD_DIR / safe_filename
    try:
        content = await _read_upload_limited(file)
        temp_path.write_bytes(content)
        logger.info("Saved uploaded file: %s (%d bytes)", safe_filename, len(content))

        # Content-based dedup: identical bytes → reuse the existing document.
        existing_id = get_knowledge_store().find_duplicate(temp_path)
        if existing_id is not None:
            logger.info(
                "Upload dedup: '%s' already indexed as document #%d",
                safe_filename, existing_id,
            )
            return JSONResponse(
                content={
                    "status": "duplicate",
                    "document_id": existing_id,
                    "filename": safe_filename,
                    "size": len(content),
                }
            )

        # Ingest into knowledge base
        doc_id = get_knowledge_store().add_document(temp_path, safe_filename)

        # Move from temp to permanent storage (keyed by doc_id)
        perm_path = KNOWLEDGE_FILES_DIR / f"{doc_id}_{safe_filename}"
        shutil.move(str(temp_path), str(perm_path))
        # Update document metadata with permanent path
        get_knowledge_store().db.update_document_metadata(
            doc_id, {"permanent_path": str(perm_path)}
        )

        return JSONResponse(
            content={
                "status": "ok",
                "document_id": doc_id,
                "filename": safe_filename,
                "size": len(content),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload ingestion failed for %s", safe_filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Clean up temp file if it still exists
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


async def _handle_zip_upload(file: UploadFile) -> JSONResponse:
    """Extract a ZIP archive and ingest each contained document."""
    import zipfile
    from agentflow.knowledge.parser import _read_raw_from_bytes

    content = await _read_upload_limited(file)
    with zipfile.ZipFile(io.BytesIO(content)) as probe:
        entries = probe.infolist()
        if len(entries) > settings.max_zip_entries:
            raise HTTPException(
                status_code=413,
                detail=f"ZIP contains {len(entries)} entries "
                       f"(limit {settings.max_zip_entries})",
            )
        if sum(info.file_size for info in entries) > settings.max_zip_uncompressed_bytes:
            raise HTTPException(
                status_code=413,
                detail="ZIP uncompressed size exceeds the configured limit",
            )
    total = 0
    success = 0
    failed: list[str] = []
    allowed_exts = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
        ".html", ".htm", ".xlsx", ".xls", ".pptx", ".csv", ".epub",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp",
    }

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            fname = Path(info.filename)
            if info.filename.startswith("__MACOSX/") or info.filename.startswith("."):
                continue
            if fname.suffix.lower() not in allowed_exts:
                continue
            total += 1
            try:
                raw = zf.read(info.filename)
                file_type = fname.suffix.lstrip(".").lower()
                text = _read_raw_from_bytes(raw, file_type)
                if not text.strip():
                    continue
                # Save to temp file and ingest
                temp_path = UPLOAD_DIR / fname.name
                temp_path.write_bytes(raw)
                try:
                    existing_id = get_knowledge_store().find_duplicate(temp_path)
                    if existing_id is not None:
                        logger.info(
                            "ZIP dedup: '%s' already indexed as document #%d",
                            fname.name, existing_id,
                        )
                        continue
                    doc_id = get_knowledge_store().add_document(temp_path, fname.name)
                    # Move to permanent storage
                    perm_path = KNOWLEDGE_FILES_DIR / f"{doc_id}_{fname.name}"
                    shutil.move(str(temp_path), str(perm_path))
                    get_knowledge_store().db.update_document_metadata(
                        doc_id, {"permanent_path": str(perm_path)}
                    )
                    success += 1
                finally:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("ZIP entry failed: %s (%s)", info.filename, exc)
                failed.append(info.filename)

    return JSONResponse(content={
        "status": "ok",
        "total": total,
        "success": success,
        "failed": failed,
        "filename": file.filename,
        "size": len(content),
    })


@router.get("/knowledge/documents")
def list_documents() -> list[dict[str, object]]:
    """List all indexed documents."""
    return get_knowledge_store().list_documents()


@router.delete("/knowledge/documents/{doc_id}")
def delete_document(doc_id: int) -> JSONResponse:
    """Delete a document and its chunks/embeddings from the knowledge base."""
    # Clean up permanent file if it exists
    import json as _json
    docs = get_knowledge_store().db.get_all_documents()
    for d in docs:
        if d.get("id") == doc_id:
            meta = _json.loads(d.get("doc_metadata", "{}") or "{}")
            perm_path = meta.get("permanent_path", "")
            if perm_path:
                p = Path(perm_path)
                if p.exists():
                    p.unlink(missing_ok=True)
            break
    get_knowledge_store().delete_document(doc_id)
    return JSONResponse(content={"status": "deleted", "document_id": doc_id})


@router.post("/knowledge/search")
def search_knowledge(
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict[str, object]]:
    """Search the knowledge base for relevant chunks."""
    doc_ids = [document_id] if document_id is not None else None
    return get_knowledge_store().search(query, top_k=top_k, document_ids=doc_ids)


@router.post("/knowledge/rebuild")
def rebuild_knowledge() -> JSONResponse:
    """Rebuild the whole knowledge index with the current embedding model."""
    try:
        result = get_knowledge_store().rebuild()
        return JSONResponse(content={"status": "ok", **result})
    except Exception as exc:
        logger.exception("Knowledge rebuild failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/knowledge/documents/{doc_id}/read")
def read_knowledge_document(doc_id: int) -> JSONResponse:
    """Read an uploaded knowledge base document for in-app preview."""
    docs = get_knowledge_store().db.get_all_documents()
    target_doc = None
    for d in docs:
        if d.get("id") == doc_id:
            target_doc = d
            break
    if target_doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    import json as _json
    meta = _json.loads(target_doc.get("doc_metadata", "{}") or "{}")
    perm_path = meta.get("permanent_path", "")
    if not perm_path:
        raise HTTPException(status_code=404, detail="Original file not available for this document")

    target = Path(perm_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Original file no longer exists on disk")

    max_bytes = 1024 * 1024  # 1 MB
    data = target.read_bytes()
    truncated = len(data) > max_bytes

    ext = target.suffix.lower()
    if ext == ".docx":
        text = _read_docx_preview(target)
        truncated = False
    elif ext in (".xlsx", ".xls"):
        text = _read_xlsx_preview(target)
        truncated = False
    elif ext == ".pptx":
        text = _read_pptx_preview(target)
        truncated = False
    elif ext == ".pdf":
        text = _read_pdf_preview(target)
        truncated = False
    else:
        text = data[:max_bytes].decode("utf-8", errors="replace")

    return JSONResponse(
        content={
            "filename": target.name,
            "path": str(target),
            "content": text,
            "truncated": truncated and ext not in (".docx", ".xlsx", ".xls", ".pptx", ".pdf"),
            "size": len(data),
        }
    )
