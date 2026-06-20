# CLAUDE.md

**Maintainer:** GP

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

There is no test suite currently.

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
- `vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunked text with embeddings)
- `document_processor.py` — parses course `.txt` files and splits them into overlapping sentence chunks; expects a fixed header format (Course Title / Course Link / Course Instructor, then `Lesson N:` markers)
- `ai_generator.py` — Anthropic SDK client; uses tool-use (function calling) to let Claude call the search tool
- `search_tools.py` — defines `CourseSearchTool` with the Anthropic tool schema; supports optional `course_name` and `lesson_number` filters
- `session_manager.py` — in-memory conversation history, capped at `MAX_HISTORY` turns (default 2)
- `config.py` — single source of truth for all tunable constants (model, chunk size, embedding model, etc.)

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. ChromaDB persists to `./chroma_db/` on first run and reloads on subsequent starts.

## Important Conventions

- The course document format expected by `DocumentProcessor` is strict: line 1 = `Course Title:`, line 2 = `Course Link:`, line 3 = `Course Instructor:`, then lesson sections starting with `Lesson N:`.
- The `ANTHROPIC_MODEL` in `config.py` is currently `claude-sonnet-4-20250514`. When updating, use the latest stable Sonnet model ID.
- Sessions are in-memory only — they are lost on server restart.
- CORS is open (`allow_origins=["*"]`); restrict this before any production deployment.
