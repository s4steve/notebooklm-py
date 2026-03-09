"""MCP (Model Context Protocol) server for NotebookLM.

Exposes NotebookLM operations as MCP tools over SSE transport,
allowing any MCP-compatible client to use NotebookLM from any machine
on the network.

Usage:
    notebooklm-mcp --host 127.0.0.1 --port 8765
    python -m notebooklm.mcp_server

Client configuration (Claude Code / Claude Desktop):
    claude mcp add notebooklm --transport sse http://server-ip:8765/sse

Or in settings.json:
    {
      "mcpServers": {
        "notebooklm": {
          "url": "http://server-ip:8765/sse"
        }
      }
    }
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import click

from .rpc import ReportFormat
from .types import ArtifactType

logger = logging.getLogger(__name__)

_DEFAULT_DOWNLOAD_DIR = Path.home() / "notebooklm-downloads"


def _get_download_dir() -> Path:
    """Get the download directory from env or default."""
    download_dir = Path(os.environ.get("NOTEBOOKLM_DOWNLOAD_DIR", str(_DEFAULT_DOWNLOAD_DIR)))
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def _sanitize_text_input(value: str | None, max_len: int = 2000) -> str | None:
    """Sanitize a free-text input to prevent prompt injection.

    - Returns None if input is None
    - Strips leading/trailing whitespace
    - Strips ASCII control characters (except newline, carriage return, tab)
    - Truncates to max_len characters
    """
    if value is None:
        return None
    value = value.strip()
    value = "".join(c for c in value if c >= " " or c in "\n\r\t")
    return value[:max_len]


def _artifact_to_dict(artifact: Any) -> dict:
    """Convert an Artifact object to a serializable dict."""
    return {
        "id": artifact.id,
        "title": artifact.title,
        "kind": str(artifact.kind) if artifact.kind else None,
        "status": str(artifact.status) if artifact.status else None,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
        "url": artifact.url if hasattr(artifact, "url") else None,
    }


def _source_to_dict(source: Any) -> dict:
    """Convert a Source object to a serializable dict."""
    return {
        "id": source.id,
        "title": source.title,
        "kind": str(source.kind) if source.kind else None,
        "url": source.url,
        "status": str(source.status) if source.status else None,
        "created_at": source.created_at.isoformat() if source.created_at else None,
    }


def _notebook_to_dict(notebook: Any) -> dict:
    """Convert a Notebook object to a serializable dict."""
    return {
        "id": notebook.id,
        "title": notebook.title,
        "created_at": notebook.created_at.isoformat() if notebook.created_at else None,
        "is_owner": notebook.is_owner,
    }


def _note_to_dict(note: Any) -> dict:
    """Convert a Note object to a serializable dict."""
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "notebook_id": note.notebook_id,
    }


def create_mcp_server():
    """Create and configure the MCP server with all NotebookLM tools.

    Returns:
        Configured mcp.server.Server instance.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as e:
        raise ImportError(
            "MCP package not found. Install with: pip install 'notebooklm-py[mcp]'"
        ) from e

    server = Server("notebooklm")

    # Shared client instance — initialized at startup
    _client_holder: list[Any] = []

    def get_client():
        if not _client_holder:
            raise RuntimeError("NotebookLM client not initialized")
        return _client_holder[0]

    # =========================================================================
    # Tool Definitions
    # =========================================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            # --- Notebook tools ---
            Tool(
                name="list_notebooks",
                description="List all NotebookLM notebooks.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="create_notebook",
                description="Create a new NotebookLM notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title for the new notebook"},
                    },
                    "required": ["title"],
                },
            ),
            Tool(
                name="delete_notebook",
                description="Delete a NotebookLM notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="rename_notebook",
                description="Rename a NotebookLM notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "title": {"type": "string", "description": "New title"},
                    },
                    "required": ["notebook_id", "title"],
                },
            ),
            Tool(
                name="describe_notebook",
                description="Get an AI-generated summary and suggested topics for a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            # --- Source tools ---
            Tool(
                name="list_sources",
                description="List all sources in a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="add_url_source",
                description="Add a URL (web page or YouTube video) as a source to a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "url": {"type": "string", "description": "URL to add"},
                    },
                    "required": ["notebook_id", "url"],
                },
            ),
            Tool(
                name="add_text_source",
                description="Add pasted text as a source to a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "title": {"type": "string", "description": "Title for the source"},
                        "content": {"type": "string", "description": "Text content"},
                    },
                    "required": ["notebook_id", "title", "content"],
                },
            ),
            Tool(
                name="delete_source",
                description="Delete a source from a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_id": {"type": "string", "description": "The source ID"},
                    },
                    "required": ["notebook_id", "source_id"],
                },
            ),
            Tool(
                name="get_source_guide",
                description="Get an AI-generated summary and keywords for a specific source.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_id": {"type": "string", "description": "The source ID"},
                    },
                    "required": ["notebook_id", "source_id"],
                },
            ),
            # --- Chat tools ---
            Tool(
                name="ask",
                description=(
                    "Ask the notebook a question. Returns the answer and a conversation_id "
                    "for follow-up questions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "question": {"type": "string", "description": "The question to ask"},
                        "conversation_id": {
                            "type": "string",
                            "description": "Existing conversation ID for follow-ups (optional)",
                        },
                    },
                    "required": ["notebook_id", "question"],
                },
            ),
            Tool(
                name="get_chat_history",
                description="Get Q&A history for the most recent conversation in a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "limit": {
                            "type": "integer",
                            "description": "Max number of Q&A pairs (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            # --- Artifact tools ---
            Tool(
                name="list_artifacts",
                description="List all AI-generated artifacts in a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_audio",
                description=(
                    "Generate an Audio Overview (podcast) for a notebook. "
                    "Waits for completion and returns the saved file path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional, uses all if omitted)",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Custom instructions for the podcast hosts (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_video",
                description=(
                    "Generate a Video Overview for a notebook. "
                    "Waits for completion and returns the saved file path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional, uses all if omitted)",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Custom instructions (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_report",
                description=(
                    "Generate a report (Briefing Doc, Study Guide, Blog Post, or Custom). "
                    "Waits for completion and returns the content."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "report_format": {
                            "type": "string",
                            "enum": ["briefing_doc", "study_guide", "blog_post", "custom"],
                            "description": "Report format (default: briefing_doc)",
                            "default": "briefing_doc",
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional)",
                        },
                        "custom_prompt": {
                            "type": "string",
                            "description": "Custom prompt for 'custom' format (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_quiz",
                description=(
                    "Generate a Quiz for a notebook. "
                    "Waits for completion and returns the quiz content."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional)",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Custom instructions (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_flashcards",
                description=(
                    "Generate Flashcards for a notebook. "
                    "Waits for completion and returns the flashcard content."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_infographic",
                description=(
                    "Generate an Infographic for a notebook. "
                    "Waits for completion and returns the saved file path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_slide_deck",
                description=(
                    "Generate a Slide Deck for a notebook. "
                    "Waits for completion and returns the saved file path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source IDs to include (optional)",
                        },
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="generate_mind_map",
                description=(
                    "Generate a Mind Map for a notebook. "
                    "Waits for completion and returns the artifact info."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="delete_artifact",
                description="Delete an artifact from a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "artifact_id": {"type": "string", "description": "The artifact ID"},
                    },
                    "required": ["notebook_id", "artifact_id"],
                },
            ),
            # --- Note tools ---
            Tool(
                name="list_notes",
                description="List all user-created notes in a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                    },
                    "required": ["notebook_id"],
                },
            ),
            Tool(
                name="create_note",
                description="Create a new note in a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "title": {"type": "string", "description": "Note title"},
                        "content": {"type": "string", "description": "Note content"},
                    },
                    "required": ["notebook_id", "title", "content"],
                },
            ),
            Tool(
                name="update_note",
                description="Update an existing note's title and content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "note_id": {"type": "string", "description": "The note ID"},
                        "title": {"type": "string", "description": "New title"},
                        "content": {"type": "string", "description": "New content"},
                    },
                    "required": ["notebook_id", "note_id", "title", "content"],
                },
            ),
            Tool(
                name="delete_note",
                description="Delete a note from a notebook.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "notebook_id": {"type": "string", "description": "The notebook ID"},
                        "note_id": {"type": "string", "description": "The note ID"},
                    },
                    "required": ["notebook_id", "note_id"],
                },
            ),
        ]

    # =========================================================================
    # Helper coroutines
    # =========================================================================

    async def _wait_and_download_media(
        client: Any, notebook_id: str, status: Any, kind: str, timeout: float = 600.0
    ) -> dict:
        """Wait for a media artifact to complete, download it, and return file info."""
        artifact = await client.artifacts.wait_for_completion(
            notebook_id, status.task_id, timeout=timeout
        )
        download_dir = _get_download_dir()

        ext_map = {
            "audio": ".mp3",
            "video": ".mp4",
            "infographic": ".png",
            "slides": ".pptx",
        }
        ext = ext_map.get(kind, "")
        artifact_id = artifact.id if hasattr(artifact, "id") else status.task_id
        filename = f"{kind}_{artifact_id}{ext}"
        output_path = download_dir / filename

        download_methods = {
            "audio": client.artifacts.download_audio,
            "video": client.artifacts.download_video,
            "infographic": client.artifacts.download_infographic,
            "slides": client.artifacts.download_slide_deck,
        }
        file_path = None
        download_func = download_methods.get(kind)
        if download_func:
            try:
                await download_func(notebook_id, str(output_path))
                file_path = str(output_path)
            except Exception as e:
                logger.warning("Download failed for %s: %s", kind, e)

        result = (
            _artifact_to_dict(artifact) if hasattr(artifact, "id") else {"task_id": status.task_id}
        )
        result["file_path"] = file_path
        return result

    async def _wait_and_get_text_artifact(
        client: Any, notebook_id: str, status: Any, timeout: float = 600.0
    ) -> dict:
        """Wait for a text artifact to complete and return its content."""
        artifact = await client.artifacts.wait_for_completion(
            notebook_id, status.task_id, timeout=timeout
        )
        result = (
            _artifact_to_dict(artifact) if hasattr(artifact, "id") else {"task_id": status.task_id}
        )

        if hasattr(artifact, "content") and artifact.content:
            result["content"] = artifact.content
        elif hasattr(artifact, "id"):
            try:
                artifact_id = artifact.id
                download_dir = _get_download_dir()
                if artifact.kind == ArtifactType.QUIZ:
                    out_path = download_dir / f"quiz_{artifact_id}.md"
                    await client.artifacts.download_quiz(
                        notebook_id,
                        str(out_path),
                        artifact_id=artifact_id,
                        output_format="markdown",
                    )
                    result["file_path"] = str(out_path)
                elif artifact.kind == ArtifactType.FLASHCARDS:
                    out_path = download_dir / f"flashcards_{artifact_id}.md"
                    await client.artifacts.download_flashcards(
                        notebook_id,
                        str(out_path),
                        artifact_id=artifact_id,
                        output_format="markdown",
                    )
                    result["file_path"] = str(out_path)
                elif artifact.kind == ArtifactType.REPORT:
                    out_path = download_dir / f"report_{artifact_id}.md"
                    await client.artifacts.download_report(notebook_id, str(out_path))
                    result["file_path"] = str(out_path)
            except Exception as e:
                logger.warning("Failed to get artifact content: %s", e)

        return result

    # =========================================================================
    # Individual tool handlers
    # =========================================================================

    async def _handle_list_notebooks(client: Any, args: dict) -> str:
        notebooks = await client.notebooks.list()
        return json.dumps([_notebook_to_dict(nb) for nb in notebooks], indent=2, ensure_ascii=False)

    async def _handle_create_notebook(client: Any, args: dict) -> str:
        title = _sanitize_text_input(args["title"])
        nb = await client.notebooks.create(title)
        return json.dumps(_notebook_to_dict(nb), indent=2, ensure_ascii=False)

    async def _handle_delete_notebook(client: Any, args: dict) -> str:
        await client.notebooks.delete(args["notebook_id"])
        return f"Deleted notebook {args['notebook_id']}"

    async def _handle_rename_notebook(client: Any, args: dict) -> str:
        title = _sanitize_text_input(args["title"])
        nb = await client.notebooks.rename(args["notebook_id"], title)
        return json.dumps(_notebook_to_dict(nb), indent=2, ensure_ascii=False)

    async def _handle_describe_notebook(client: Any, args: dict) -> str:
        desc = await client.notebooks.get_description(args["notebook_id"])
        result = {
            "summary": desc.summary,
            "suggested_topics": [
                {"question": t.question, "prompt": t.prompt} for t in desc.suggested_topics
            ],
        }
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _handle_list_sources(client: Any, args: dict) -> str:
        sources = await client.sources.list(args["notebook_id"])
        return json.dumps([_source_to_dict(s) for s in sources], indent=2, ensure_ascii=False)

    async def _handle_add_url_source(client: Any, args: dict) -> str:
        source = await client.sources.add_url(args["notebook_id"], args["url"], wait=True)
        return json.dumps(_source_to_dict(source), indent=2, ensure_ascii=False)

    async def _handle_add_text_source(client: Any, args: dict) -> str:
        title = _sanitize_text_input(args["title"])
        content = _sanitize_text_input(args["content"], max_len=50000)
        source = await client.sources.add_text(args["notebook_id"], title, content, wait=True)
        return json.dumps(_source_to_dict(source), indent=2, ensure_ascii=False)

    async def _handle_delete_source(client: Any, args: dict) -> str:
        await client.sources.delete(args["notebook_id"], args["source_id"])
        return f"Deleted source {args['source_id']}"

    async def _handle_get_source_guide(client: Any, args: dict) -> str:
        guide = await client.sources.get_guide(args["notebook_id"], args["source_id"])
        return json.dumps(guide, indent=2, ensure_ascii=False)

    async def _handle_ask(client: Any, args: dict) -> str:
        question = _sanitize_text_input(args["question"])
        result = await client.chat.ask(
            args["notebook_id"],
            question,
            conversation_id=args.get("conversation_id"),
        )
        return json.dumps(
            {
                "answer": result.answer,
                "conversation_id": result.conversation_id,
                "turn_number": result.turn_number,
            },
            indent=2,
            ensure_ascii=False,
        )

    async def _handle_get_chat_history(client: Any, args: dict) -> str:
        limit = args.get("limit", 20)
        history = await client.chat.get_history(args["notebook_id"], limit=limit)
        pairs = [{"question": q, "answer": a} for q, a in history]
        return json.dumps(pairs, indent=2, ensure_ascii=False)

    async def _handle_list_artifacts(client: Any, args: dict) -> str:
        artifacts = await client.artifacts.list(args["notebook_id"])
        return json.dumps([_artifact_to_dict(a) for a in artifacts], indent=2, ensure_ascii=False)

    async def _handle_generate_audio(client: Any, args: dict) -> str:
        instructions = _sanitize_text_input(args.get("instructions"))
        status = await client.artifacts.generate_audio(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
            instructions=instructions,
        )
        artifact = await _wait_and_download_media(client, args["notebook_id"], status, "audio")
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    async def _handle_generate_video(client: Any, args: dict) -> str:
        instructions = _sanitize_text_input(args.get("instructions"))
        status = await client.artifacts.generate_video(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
            instructions=instructions,
        )
        artifact = await _wait_and_download_media(client, args["notebook_id"], status, "video")
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    async def _handle_generate_report(client: Any, args: dict) -> str:
        fmt_map = {
            "briefing_doc": ReportFormat.BRIEFING_DOC,
            "study_guide": ReportFormat.STUDY_GUIDE,
            "blog_post": ReportFormat.BLOG_POST,
            "custom": ReportFormat.CUSTOM,
        }
        report_format = fmt_map.get(
            args.get("report_format", "briefing_doc"), ReportFormat.BRIEFING_DOC
        )
        custom_prompt = _sanitize_text_input(args.get("custom_prompt"))
        status = await client.artifacts.generate_report(
            args["notebook_id"],
            report_format=report_format,
            source_ids=args.get("source_ids"),
            custom_prompt=custom_prompt,
        )
        content = await _wait_and_get_text_artifact(client, args["notebook_id"], status)
        return json.dumps(content, indent=2, ensure_ascii=False)

    async def _handle_generate_quiz(client: Any, args: dict) -> str:
        instructions = _sanitize_text_input(args.get("instructions"))
        status = await client.artifacts.generate_quiz(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
            instructions=instructions,
        )
        content = await _wait_and_get_text_artifact(client, args["notebook_id"], status)
        return json.dumps(content, indent=2, ensure_ascii=False)

    async def _handle_generate_flashcards(client: Any, args: dict) -> str:
        status = await client.artifacts.generate_flashcards(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
        )
        content = await _wait_and_get_text_artifact(client, args["notebook_id"], status)
        return json.dumps(content, indent=2, ensure_ascii=False)

    async def _handle_generate_infographic(client: Any, args: dict) -> str:
        status = await client.artifacts.generate_infographic(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
        )
        artifact = await _wait_and_download_media(
            client, args["notebook_id"], status, "infographic"
        )
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    async def _handle_generate_slide_deck(client: Any, args: dict) -> str:
        status = await client.artifacts.generate_slide_deck(
            args["notebook_id"],
            source_ids=args.get("source_ids"),
        )
        artifact = await _wait_and_download_media(client, args["notebook_id"], status, "slides")
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    async def _handle_generate_mind_map(client: Any, args: dict) -> str:
        status = await client.artifacts.generate_mind_map(args["notebook_id"])
        await client.artifacts.wait_for_completion(args["notebook_id"], status.task_id)
        artifact = await client.artifacts.get(args["notebook_id"], status.task_id)
        return json.dumps(
            _artifact_to_dict(artifact) if artifact else {"task_id": status.task_id},
            indent=2,
            ensure_ascii=False,
        )

    async def _handle_delete_artifact(client: Any, args: dict) -> str:
        await client.artifacts.delete(args["notebook_id"], args["artifact_id"])
        return f"Deleted artifact {args['artifact_id']}"

    async def _handle_list_notes(client: Any, args: dict) -> str:
        notes = await client.notes.list(args["notebook_id"])
        return json.dumps([_note_to_dict(n) for n in notes], indent=2, ensure_ascii=False)

    async def _handle_create_note(client: Any, args: dict) -> str:
        title = _sanitize_text_input(args["title"])
        content = _sanitize_text_input(args["content"], max_len=50000)
        note = await client.notes.create(args["notebook_id"], title=title, content=content)
        return json.dumps(_note_to_dict(note), indent=2, ensure_ascii=False)

    async def _handle_update_note(client: Any, args: dict) -> str:
        title = _sanitize_text_input(args["title"])
        content = _sanitize_text_input(args["content"], max_len=50000)
        await client.notes.update(args["notebook_id"], args["note_id"], content, title)
        return f"Updated note {args['note_id']}"

    async def _handle_delete_note(client: Any, args: dict) -> str:
        await client.notes.delete(args["notebook_id"], args["note_id"])
        return f"Deleted note {args['note_id']}"

    # =========================================================================
    # Dispatch dictionary
    # =========================================================================

    _TOOL_DISPATCH = {
        "list_notebooks": _handle_list_notebooks,
        "create_notebook": _handle_create_notebook,
        "delete_notebook": _handle_delete_notebook,
        "rename_notebook": _handle_rename_notebook,
        "describe_notebook": _handle_describe_notebook,
        "list_sources": _handle_list_sources,
        "add_url_source": _handle_add_url_source,
        "add_text_source": _handle_add_text_source,
        "delete_source": _handle_delete_source,
        "get_source_guide": _handle_get_source_guide,
        "ask": _handle_ask,
        "get_chat_history": _handle_get_chat_history,
        "list_artifacts": _handle_list_artifacts,
        "generate_audio": _handle_generate_audio,
        "generate_video": _handle_generate_video,
        "generate_report": _handle_generate_report,
        "generate_quiz": _handle_generate_quiz,
        "generate_flashcards": _handle_generate_flashcards,
        "generate_infographic": _handle_generate_infographic,
        "generate_slide_deck": _handle_generate_slide_deck,
        "generate_mind_map": _handle_generate_mind_map,
        "delete_artifact": _handle_delete_artifact,
        "list_notes": _handle_list_notes,
        "create_note": _handle_create_note,
        "update_note": _handle_update_note,
        "delete_note": _handle_delete_note,
    }

    # =========================================================================
    # Tool Handler (call_tool)
    # =========================================================================

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        client = get_client()
        try:
            handler = _TOOL_DISPATCH.get(name)
            if handler is None:
                raise ValueError(f"Unknown tool: {name}")
            result = await handler(client, arguments)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return [TextContent(type="text", text=f"Error: {e}")]

    return server, _client_holder


async def run_server(host: str, port: int, api_key: str | None = None) -> None:
    """Start the MCP SSE server."""
    try:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import Response
        from starlette.routing import Mount, Route
    except ImportError as e:
        raise ImportError(
            "SSE server dependencies not found. Install with: pip install 'notebooklm-py[mcp]'"
        ) from e

    from .client import NotebookLMClient

    if not api_key:
        logger.warning(
            "No API key configured. Anyone with network access can use this server. "
            "Set --api-key or NOTEBOOKLM_MCP_API_KEY to restrict access."
        )

    logger.info("Starting NotebookLM MCP server on %s:%d", host, port)
    logger.info("Loading NotebookLM client from storage...")

    server, client_holder = create_mcp_server()

    # Initialize the client
    client = await NotebookLMClient.from_storage()
    await client.__aenter__()
    client_holder.append(client)

    logger.info("NotebookLM client initialized successfully")
    logger.info("Download directory: %s", _get_download_dir())

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options(), stateless=True
            )

    # Build optional API key middleware
    middleware: list[Middleware] = []
    if api_key:
        _key = api_key  # capture for closure

        class _ApiKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                auth = request.headers.get("Authorization", "")
                if not (auth.startswith("Bearer ") and auth[7:] == _key):
                    return Response("Unauthorized", status_code=401)
                return await call_next(request)

        middleware = [Middleware(_ApiKeyMiddleware)]

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        middleware=middleware,
    )

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)

    try:
        await server_instance.serve()
    finally:
        await client.__aexit__(None, None, None)
        logger.info("NotebookLM MCP server stopped")


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--port", default=8765, show_default=True, help="Port to listen on")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--api-key",
    envvar="NOTEBOOKLM_MCP_API_KEY",
    default=None,
    help="Require Bearer token auth on all requests (env: NOTEBOOKLM_MCP_API_KEY)",
)
def main(host: str, port: int, verbose: bool, api_key: str | None) -> None:
    """Start the NotebookLM MCP server.

    Exposes all NotebookLM operations as MCP tools over SSE transport.
    Authentication is loaded from ~/.notebooklm/storage_state.json.

    \b
    Add to Claude Code:
        claude mcp add notebooklm --transport sse http://localhost:8765/sse

    \b
    From another machine:
        claude mcp add notebooklm --transport sse http://server-ip:8765/sse
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_server(host, port, api_key=api_key))


if __name__ == "__main__":
    main()
