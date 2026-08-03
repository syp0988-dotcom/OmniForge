"""API router: files_workspace endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel

from agentflow.api.routes import (
    OUTPUT_DIR,
    _is_relative_to,
    _resolve_workspace_child,
    _set_workspace_root,
)

router = APIRouter()
logger = build_logger("api.files_workspace")

class CreateFileRequest(BaseModel):
    filename: str
    content: str
    workspace_path: str | None = None


class ReadFileRequest(BaseModel):
    path: str
    workspace_path: str | None = None


def _safe_relative_file_path(raw_path: str) -> Path:
    """Return a safe relative file path, preserving subdirectories."""
    rel = Path(raw_path.replace("\\", "/"))
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise HTTPException(status_code=400, detail="Invalid file path")
    return rel


def _resolve_generated_file(path: str, workspace_path: str | None = None) -> Path:
    """Resolve a generated file path under outputs/ or the active workspace."""
    base = _resolve_workspace_child(workspace_path) if workspace_path else OUTPUT_DIR.resolve()
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (base / _safe_relative_file_path(path)).resolve()
    if not _is_relative_to(target, base):
        raise HTTPException(status_code=400, detail="File path is outside the workspace")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return target


@router.post("/files/create")
def create_file(req: CreateFileRequest) -> JSONResponse:
    """Write a proposed file to the outputs/ or workspace directory."""
    rel_path = _safe_relative_file_path(req.filename)

    if req.workspace_path:
        base = _resolve_workspace_child(req.workspace_path)
        if not base.exists() or not base.is_dir():
            raise HTTPException(status_code=400, detail="Invalid workspace path")
        target = (base / rel_path).resolve()
    else:
        base = OUTPUT_DIR.resolve()
        target = (base / rel_path).resolve()

    if not _is_relative_to(target, base):
        raise HTTPException(status_code=400, detail="File path is outside the workspace")

    if target.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    return JSONResponse(
        content={
            "status": "created",
            "filename": rel_path.as_posix(),
            "path": str(target),
        }
    )


@router.post("/files/read")
def read_generated_file(req: ReadFileRequest) -> JSONResponse:
    """Read a generated file for in-app preview."""
    target = _resolve_generated_file(req.path, req.workspace_path)
    max_bytes = 1024 * 1024
    data = target.read_bytes()
    truncated = len(data) > max_bytes
    if target.suffix.lower() == ".docx":
        text = _read_docx_preview(target)
        truncated = False
    else:
        text = data[:max_bytes].decode("utf-8", errors="replace")
    return JSONResponse(
        content={
            "filename": target.name,
            "path": str(target),
            "content": text,
            "truncated": truncated,
            "size": len(data),
        }
    )


def _read_docx_preview(path: Path) -> str:
    """Extract readable text from a .docx file for preview."""
    try:
        from docx import Document
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="python-docx is not installed") from exc

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read docx file: {exc}") from exc

    parts: list[str] = []
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if paragraphs:
        parts.extend(paragraphs)

    for table_index, table in enumerate(doc.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"\n[表格 {table_index}]")
            parts.extend(rows)

    return "\n\n".join(parts) if parts else "（该 Word 文档没有可预览的文本内容）"


def _read_xlsx_preview(path: Path) -> str:
    """Extract readable text from a .xlsx/.xls file for preview."""
    try:
        import openpyxl
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="openpyxl is not installed") from exc

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read xlsx file: {exc}") from exc

    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: list[str] = [f"[工作表: {sheet_name}]"]
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                rows.append(" | ".join(cells))
                row_count += 1
                if row_count >= 200:
                    rows.append("...（表格行数过多，仅显示前 200 行）")
                    break
        if len(rows) > 1:
            parts.append("\n".join(rows))
    return "\n\n".join(parts) if parts else "（该 Excel 文件没有可预览的内容）"


def _read_pptx_preview(path: Path) -> str:
    """Extract readable text from a .pptx file for preview."""
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="python-pptx is not installed") from exc

    try:
        prs = Presentation(str(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read pptx file: {exc}") from exc

    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    t = paragraph.text.strip()
                    if t:
                        texts.append(t)
        if texts:
            parts.append(f"[幻灯片 {i}]\n" + "\n".join(texts))
    return "\n\n".join(parts) if parts else "（该 PPT 没有可预览的文本内容）"


def _read_pdf_preview(path: Path) -> str:
    """Extract readable text from a .pdf file for preview."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            import pdfplumber
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail="PyMuPDF or pdfplumber is not installed"
            ) from exc

        try:
            with pdfplumber.open(path) as pdf:
                parts: list[str] = []
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        parts.append(f"[第 {i} 页]\n{text}")
                return "\n\n".join(parts) if parts else "（该 PDF 没有可预览的文本内容）"
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read pdf file: {exc}") from exc

    try:
        doc = fitz.open(str(path))
        parts: list[str] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                parts.append(f"[第 {i} 页]\n{text.strip()}")
        doc.close()
        return "\n\n".join(parts) if parts else "（该 PDF 没有可预览的文本内容）"
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read pdf file: {exc}") from exc


@router.get("/files")
def list_output_files(workspace_path: str | None = None) -> list[dict[str, object]]:
    """List files in the outputs/ directory or a workspace path."""
    if workspace_path:
        base = _resolve_workspace_child(workspace_path)
        if not base.exists() or not base.is_dir():
            raise HTTPException(status_code=400, detail="Invalid workspace path")
    else:
        base = OUTPUT_DIR

    if not base.exists():
        return []
    files: list[dict[str, object]] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base)
            files.append(
                {
                    "filename": str(rel),
                    "size": p.stat().st_size,
                    "created_at": datetime.fromtimestamp(p.stat().st_ctime).isoformat(),
                    "path": str(p),
                }
            )
    return files


@router.get("/workspace")
def get_workspace() -> JSONResponse:
    """Check if a path is a valid workspace directory."""
    return JSONResponse(content={"status": "ok", "message": "Provide a path via POST /workspace/set"})


class SetWorkspaceRequest(BaseModel):
    path: str


@router.post("/workspace/set")
def set_workspace(req: SetWorkspaceRequest) -> JSONResponse:
    """Validate and set workspace folder path."""
    p = Path(req.path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {req.path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    # Test write permission
    test_file = p / ".omni_forge_write_test"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"No write permission: {exc}")
    _set_workspace_root(p)
    return JSONResponse(content={"status": "ok", "path": str(p)})


class CreateFolderRequest(BaseModel):
    parent_path: str
    folder_name: str


@router.post("/workspace/create-folder")
def create_workspace_folder(req: CreateFolderRequest) -> JSONResponse:
    """Create a new folder under the given parent path."""
    parent = _resolve_workspace_child(req.parent_path)
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(status_code=404, detail=f"Parent path does not exist: {req.parent_path}")
    safe_name = Path(req.folder_name).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    target = parent / safe_name
    if target.exists():
        raise HTTPException(status_code=409, detail=f"Folder already exists: {safe_name}")
    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {exc}")
    return JSONResponse(content={"status": "created", "path": str(target)})


@router.get("/workspace/browse")
def browse_directory(path: str = ".") -> JSONResponse:
    """List directories and files at the given path for folder browsing."""
    base = _resolve_workspace_child(path)
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")
    entries: list[dict[str, object]] = []
    for p in sorted(base.iterdir()):
        entries.append({
            "name": p.name,
            "is_dir": p.is_dir(),
            "path": str(p),
        })
    return JSONResponse(content={"current_path": str(base), "entries": entries})
