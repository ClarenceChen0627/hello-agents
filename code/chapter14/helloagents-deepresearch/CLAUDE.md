# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HelloAgents DeepResearch is a full-stack web research assistant that generates structured reports by autonomously planning and executing multiple search tasks. Built with FastAPI (backend) and Vue 3 (frontend).

## Architecture

```
helloagents-deepresearch/
├── backend/
│   ├── src/
│   │   ├── main.py          # FastAPI entrypoint, /research and /research/stream endpoints
│   │   ├── agent.py         # DeepResearchAgent orchestrator
│   │   ├── config.py        # Configuration via pydantic, loads from env
│   │   ├── models.py        # Dataclasses: TodoItem, SummaryState
│   │   ├── prompts.py       # System prompts for planner/summarizer/reporter agents
│   │   ├── utils.py         # Text processing helpers
│   │   └── services/
│   │       ├── planner.py   # Plans TODO tasks from research topic
│   │       ├── search.py    # Dispatches search via HelloAgents SearchTool
│   │       ├── summarizer.py# Summarizes search results per task
│   │       ├── reporter.py  # Generates final markdown report
│   │       └── tool_events.py# Tracks tool call events for streaming
│   └── pyproject.toml
└── frontend/
    ├── src/
    │   ├── App.vue          # Main UI component
    │   ├── services/api.ts  # SSE client for /research/stream
    │   └── main.ts
    └── package.json
```

## Core Workflow

1. **Planning**: `PlanningService` uses a `ToolAwareSimpleAgent` to break the research topic into 3-5 TODO tasks
2. **Execution**: Each task runs in parallel:
   - `dispatch_search()` calls `SearchTool` (supports DuckDuckGo, Tavily, Perplexity, SearXNG)
   - `SummarizationService` summarizes search results
   - Notes are persisted via `NoteTool` if enabled
3. **Reporting**: `ReportingService` aggregates all task summaries into a structured markdown report

## Key Technologies

- **hello-agents**: Core SDK providing `HelloAgentsLLM`, `ToolAwareSimpleAgent`, `ToolRegistry`, and built-in tools (`SearchTool`, `NoteTool`)
- **Backend**: FastAPI + uvicorn, pydantic for config/validation, loguru for logging
- **Frontend**: Vue 3 + Vite + TypeScript, axios for HTTP, SSE for streaming
- **LLM Providers**: Ollama (default), LMStudio, or any OpenAI-compatible API

## Common Commands

### Backend

```bash
cd backend

# Install dependencies
uv sync

# Run development server
uv run python src/main.py

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build
```

## Configuration

Configuration is managed via environment variables (loaded from `.env`):

```bash
# LLM (required)
LLM_PROVIDER=ollama          # ollama, lmstudio, or custom
LLM_API_KEY=your-key
LLM_MODEL_ID=qwen2.5:7b      # Model to use
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Search (optional, defaults to duckduckgo)
SEARCH_API=duckduckgo        # duckduckgo, tavily, perplexity, searxng

# Features
ENABLE_NOTES=true            # Persist task notes to disk
NOTES_WORKSPACE=./notes      # Notes directory
FETCH_FULL_PAGE=true         # Scrape full page content
MAX_WEB_RESEARCH_LOOPS=3     # Max search iterations
STRIP_THINKING_TOKENS=true   # Remove <think> tokens from output
```

## API Endpoints

- `GET /healthz` - Health check
- `POST /research` - Run research, return full report
- `POST /research/stream` - Server-sent events for real-time progress
- `GET /research/history` - List all past research records
- `GET /research/history/{note_id}` - Get full details of a specific research record

## Stream Events

The streaming endpoint emits these event types:
- `status` - General status updates
- `todo_list` - Initial task list
- `task_status` - Task state changes (pending/in_progress/completed/failed)
- `sources` - Search results received
- `task_summary_chunk` - Streaming summary text
- `report_note` - Final report persisted
- `final_report` - Complete report markdown
- `done` - Stream finished

## Notes

- The project uses `uv` (Python's fast package manager) instead of pip for dependency management
- Local LLM via Ollama is the default; ensure the model is pulled before running
- Notes are stored as markdown files in `./notes/` when `ENABLE_NOTES=true`
- The `hello-agents` SDK handles tool calling, history management, and streaming
