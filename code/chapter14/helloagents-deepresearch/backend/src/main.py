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


def _parse_note_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a note file and extract metadata and content."""
    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract note_id from filename (without .md extension)
        note_id = file_path.stem

        # Try to extract title from frontmatter or first heading
        title = note_id
        content_match = content

        # Look for YAML frontmatter title
        frontmatter_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            title_match = re.search(r'^title:\s*(.+)$', frontmatter, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
                content_match = content[frontmatter_match.end():]

        # If no frontmatter title, look for first heading
        if title == note_id:
            heading_match = re.search(r'^#\s*(.+)$', content_match, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip()

        # Get file modification time for created_at
        mtime = file_path.stat().st_mtime
        created_at = datetime.fromtimestamp(mtime).isoformat()

        return {
            "note_id": note_id,
            "title": title,
            "content": content,
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
            is_conclusion = (
                "deep_research" in content and
                "conclusion" in content or
                "研究报告" in content or
                "报告" in content
            )

            # For now, include all note files (can be filtered later if needed)
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
            content=parsed["content"],
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
