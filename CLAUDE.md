# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Maintainer:** GP

## What This Project Does

A full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Users ask natural language questions; the system performs semantic search over course documents and uses Claude to synthesize answers with source citations.

## Setup & Running

**Requirements:** Python 3.13+, `uv` package manager, Anthropic API key.

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# Start the server
cd backend
uv run uvicorn app:app --reload --port 8000
```

Access: `http://localhost:8000` (UI) or `http://localhost:8000/docs` (API docs).

There is a test suite in `backend/tests/`. Tests run automatically on every server startup (printed to console). `main.py` at the repo root is an unused stub.

## Architecture

**Stack:** FastAPI backend + vanilla JS frontend + ChromaDB vector store + Anthropic Claude.

**Data flow:**
1. Query arrives at `POST /api/query` in `app.py`
2. `RAGSystem` (`rag_system.py`) orchestrates the response:
   - Retrieves session history from `SessionManager`
   - Passes the query to `AIGenerator` with a `CourseSearchTool` available
   - Claude invokes the tool autonomously; `VectorStore` performs semantic search
   - Claude generates a response using retrieved chunks
3. Response + sources returned to the frontend

**Key components:**
- `rag_system.py` — top-level orchestrator; entry point for understanding query handling
- `models.py` — Pydantic domain types (`Course`, `Lesson`, `CourseChunk`); `course.title` is the primary key used as ChromaDB document IDs throughout
- `vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunked text with embeddings)
- `document_processor.py` — parses course `.txt` files and splits them into overlapping sentence chunks; expects a fixed header format (see Conventions below)
- `ai_generator.py` — Anthropic SDK client; uses a two-pass pattern: first API call may return `tool_use`, then `_handle_tool_execution` runs the tool and makes a second API call with the result
- `search_tools.py` — defines `CourseSearchTool` with the Anthropic tool schema; supports optional `course_name` and `lesson_number` filters; `ToolManager` registers and dispatches tools
- `session_manager.py` — in-memory conversation history, capped at `MAX_HISTORY` turns (default 2)
- `config.py` — single source of truth for all tunable constants (model, chunk size, embedding model, etc.)

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. ChromaDB persists to `./chroma_db/` (relative to `backend/`) on first run and reloads on subsequent starts.

**Document loading:** On startup, `app.py` loads documents from `../docs` (relative to `backend/`). `add_course_folder` deduplicates by course title — courses already present in ChromaDB are skipped.

## Testing

Tests live in `backend/tests/` and use `pytest` with `unittest.mock` — no real API calls or ChromaDB I/O.

| File | Covers |
|------|--------|
| `tests/test_ai_generator.py` | `AIGenerator` init, `generate_response` (direct + tool-use path), `_handle_tool_execution` message assembly |
| `tests/test_search_tools.py` | `CourseSearchTool` schema, execute (results / empty / error / filters), source tracking; `ToolManager` register, dispatch, sources, reset |
| `tests/test_rag_system.py` | `RAGSystem` init, `add_course_document`, `add_course_folder` (new / skip / clear), `query` (session / no session), `get_course_analytics` |

**Running tests manually:**
```bash
cd backend
uv run pytest tests/ -v
```

**Automatic run on startup:** `app.py`'s `startup_event` executes pytest as a subprocess before loading documents. Results are printed to the server console. A test failure prints a warning but does **not** prevent the server from starting.

**Adding new tests:** Drop a `test_*.py` file in `backend/tests/`. `conftest.py` adds `backend/` to `sys.path` so all backend modules are importable without package prefixes.

## Important Conventions

- **Course document format** expected by `DocumentProcessor`:
  - Line 1: `Course Title: <title>`
  - Line 2: `Course Link: <url>`
  - Line 3: `Course Instructor: <name>`
  - Then lesson sections starting with `Lesson N: <title>`, each optionally followed by `Lesson Link: <url>` on the very next line
- **ChromaDB metadata limitation:** Lesson data is serialized as a JSON string in `lessons_json` because ChromaDB does not support nested objects in metadata.
- The `ANTHROPIC_MODEL` in `config.py` is currently `claude-sonnet-4-20250514`. When updating, use the latest stable Sonnet model ID.
- Sessions are in-memory only — they are lost on server restart.
- CORS is open (`allow_origins=["*"]`); restrict this before any production deployment.
- The system prompt in `ai_generator.py` limits Claude to **one search per query** — this is enforced by instruction, not code.
