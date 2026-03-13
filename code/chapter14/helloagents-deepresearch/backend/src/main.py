"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from agent import DeepResearchAgent
from services.session_store import SqliteSessionStore

SESSION_STATE_START = "<!-- DEEP_RESEARCH_SESSION_START -->"
SESSION_STATE_END = "<!-- DEEP_RESEARCH_SESSION_END -->"

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


class HistoryItem(BaseModel):
    """Summary of a past research record."""

    note_id: str = Field(..., description="Unique note identifier")
    title: str = Field(..., description="Research topic/title")
    created_at: str = Field(..., description="ISO format timestamp from filename")
    file_path: str = Field(..., description="Absolute path to the note file")


class HistoryResponse(BaseModel):
    """List of past research records."""

    items: list[HistoryItem] = Field(default_factory=list)


class ResearchDetailResponse(BaseModel):
    """Full detail of a specific research record."""

    note_id: str = Field(..., description="Unique note identifier")
    title: str = Field(..., description="Research topic/title")
    content: str = Field(..., description="Full markdown content of the report")
    report_markdown: str = Field(..., description="Rendered report markdown")
    research_topic: str = Field(..., description="Original research topic")
    search_api: str = Field(default="", description="Search backend used for the session")
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recorded SSE event stream for reconstructing the session UI",
    )
    tasks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured task snapshots captured with the report",
    )
    created_at: str = Field(..., description="ISO format timestamp")
    file_path: str = Field(..., description="Path to the note file")


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


def _get_notes_dir() -> Path:
    """Get the notes workspace directory."""
    config = Configuration.from_env()
    notes_dir = Path(config.notes_workspace)
    if not notes_dir.is_absolute():
        notes_dir = Path.cwd() / notes_dir
    return notes_dir


def _get_session_store() -> SqliteSessionStore:
    """Get the SQLite-backed session history store."""
    config = Configuration.from_env()
    return SqliteSessionStore(config.history_db_path)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata, text[match.end() :]


def _extract_session_payload(text: str) -> tuple[dict[str, Any] | None, str]:
    pattern = re.compile(
        rf"{re.escape(SESSION_STATE_START)}\s*(.*?)\s*{re.escape(SESSION_STATE_END)}\s*",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None, text

    payload: dict[str, Any] | None = None
    try:
        candidate = json.loads(match.group(1))
        if isinstance(candidate, dict):
            payload = candidate
    except json.JSONDecodeError:
        payload = None

    body = f"{text[:match.start()]}{text[match.end():]}".strip()
    return payload, body


def _parse_note_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a note file and extract metadata and content."""
    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract note_id from filename (without .md extension)
        note_id = file_path.stem

        frontmatter, body = _parse_frontmatter(content)
        session_payload, body = _extract_session_payload(body)

        # Try to extract title from structured metadata, frontmatter, or first heading
        title = (
            str(
                (session_payload or {}).get("research_topic")
                or frontmatter.get("research_topic")
                or frontmatter.get("title")
                or note_id
            )
            .strip()
        )
        note_type = str(frontmatter.get("note_type") or "").strip()

        if title == note_id:
            heading_match = re.search(r"^#\s*(.+)$", body, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip()

        # Get file modification time for created_at
        mtime = file_path.stat().st_mtime
        created_at = datetime.fromtimestamp(mtime).isoformat()

        return {
            "note_id": note_id,
            "title": title,
            "content": content,
            "body": body.strip(),
            "session_payload": session_payload or {},
            "note_type": note_type,
            "created_at": created_at,
            "file_path": str(file_path),
        }
    except Exception as exc:
        logger.warning("Failed to parse note file {}: {}", file_path, exc)
        return None


def create_app() -> FastAPI:
    app = FastAPI(title="HelloAgents Deep Researcher")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider={} model={} base_url={} search_api={} "
            "max_loops={} fetch_full_page={} tool_calling={} strip_thinking={} api_key={}",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/research/history")
    def get_research_history() -> HistoryResponse:
        """List all past research records from the notes workspace."""
        store = _get_session_store()
        session_items = store.list_sessions()
        if session_items:
            return HistoryResponse(
                items=[
                    HistoryItem(
                        note_id=str(item.get("note_id") or item.get("session_id") or ""),
                        title=str(item.get("research_topic") or ""),
                        created_at=str(item.get("created_at") or ""),
                        file_path=str(item.get("file_path") or ""),
                    )
                    for item in session_items
                    if str(item.get("note_id") or item.get("session_id") or "").strip()
                ]
            )

        notes_dir = _get_notes_dir()

        if not notes_dir.exists():
            return HistoryResponse(items=[])

        history_items: list[HistoryItem] = []

        # Find all .md files that are conclusion-type reports
        for file_path in notes_dir.glob("*.md"):
            parsed = _parse_note_file(file_path)
            if not parsed:
                continue

            # Check if this is a conclusion-type report by looking at content
            content = parsed["content"]
            note_type = str(parsed.get("note_type") or "").strip()
            session_payload = parsed.get("session_payload") or {}
            if note_type == "task_state":
                continue
            is_conclusion = (
                note_type == "conclusion" or
                bool(session_payload) and bool(session_payload.get("report_markdown")) or
                "deep_research" in content and
                "conclusion" in content or
                "研究报告" in content or
                "报告" in content
            )

            if not is_conclusion:
                continue

            history_items.append(HistoryItem(
                note_id=parsed["note_id"],
                title=parsed["title"],
                created_at=parsed["created_at"],
                file_path=parsed["file_path"],
            ))

        # Sort by created_at descending (newest first)
        history_items.sort(key=lambda x: x.created_at, reverse=True)

        return HistoryResponse(items=history_items)

    @app.get("/research/history/{note_id}")
    def get_research_detail(note_id: str) -> ResearchDetailResponse:
        """Get full details of a specific research record."""
        store = _get_session_store()
        session_payload = store.get_session(note_id)
        if session_payload:
            return ResearchDetailResponse(
                note_id=str(session_payload.get("note_id") or note_id),
                title=str(
                    session_payload.get("research_topic")
                    or session_payload.get("title")
                    or note_id
                ),
                content=str(
                    session_payload.get("content")
                    or session_payload.get("report_markdown")
                    or ""
                ),
                report_markdown=str(session_payload.get("report_markdown") or ""),
                research_topic=str(session_payload.get("research_topic") or ""),
                search_api=str(session_payload.get("search_api") or ""),
                events=[
                    item
                    for item in (session_payload.get("events") or [])
                    if isinstance(item, dict)
                ],
                tasks=[
                    item
                    for item in (session_payload.get("tasks") or [])
                    if isinstance(item, dict)
                ],
                created_at=str(session_payload.get("created_at") or ""),
                file_path=str(session_payload.get("file_path") or ""),
            )

        notes_dir = _get_notes_dir()
        file_path = notes_dir / f"{note_id}.md"

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Research record '{note_id}' not found")

        parsed = _parse_note_file(file_path)
        if not parsed:
            raise HTTPException(status_code=500, detail="Failed to parse research record")

        return ResearchDetailResponse(
            note_id=parsed["note_id"],
            title=parsed["title"],
            content=parsed["body"],
            report_markdown=str(
                (parsed.get("session_payload") or {}).get("report_markdown")
                or parsed["body"]
            ),
            research_topic=str(
                (parsed.get("session_payload") or {}).get("research_topic")
                or parsed["title"]
            ),
            search_api=str((parsed.get("session_payload") or {}).get("search_api") or ""),
            events=[
                item
                for item in ((parsed.get("session_payload") or {}).get("events") or [])
                if isinstance(item, dict)
            ],
            tasks=[
                item
                for item in ((parsed.get("session_payload") or {}).get("tasks") or [])
                if isinstance(item, dict)
            ],
            created_at=parsed["created_at"],
            file_path=parsed["file_path"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
