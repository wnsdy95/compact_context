"""
AILang-IR MCP Server — Model Context Protocol integration.

Exposes AILang-IR as an MCP tool server so AI agents (Claude Code, etc.)
can compress, store, search, and export conversation context in real-time.

Protocol: JSON-RPC 2.0 over stdio (MCP standard).
Dependencies: stdlib only (no mcp SDK required).

Usage:
    # Add to Claude Code MCP config (~/.claude/claude_code_config.json):
    {
      "mcpServers": {
        "ailang-ir": {
          "command": "python",
          "args": ["-m", "ailang_ir.mcp_server"],
          "env": {}
        }
      }
    }

    # Or run directly:
    python -m ailang_ir.mcp_server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ailang_ir.pipeline import Pipeline
from ailang_ir.models.domain import SpeakerRole
from ailang_ir.llm.format_spec import get_format_spec, get_format_spec_full
from ailang_ir.llm.codec import LLMCodec

# Global pipeline instance (persists across calls within a session)
_pipeline: Pipeline | None = None
_store_path: str | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
        # Try loading existing memory
        if _store_path:
            p = Path(_store_path)
            if p.exists():
                from ailang_ir.memory.store import MemoryStore
                _pipeline.memory, ct = MemoryStore.load(p)
                if ct is not None:
                    _pipeline.concept_table = ct
    return _pipeline


def _save_if_configured():
    if _store_path and _pipeline:
        _pipeline.memory.save(Path(_store_path), _pipeline.concept_table)


SPEAKER_MAP = {
    "user": SpeakerRole.USER,
    "agent": SpeakerRole.AGENT,
    "assistant": SpeakerRole.AGENT,
    "system": SpeakerRole.SYSTEM,
}


# ===================================================================
# Tool definitions
# ===================================================================

TOOLS = [
    {
        "name": "compress_text",
        "description": "Compress natural language text into AILang-IR semantic code. Returns the compact code and stores in memory for later export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to compress"},
                "speaker": {"type": "string", "enum": ["user", "agent", "system"], "default": "user",
                            "description": "Who said this text"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "compress_conversation",
        "description": "Compress a multi-turn conversation into AILang-IR. Each turn is a [speaker, text] pair. Stores all turns in memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "List of [speaker, text] pairs. Speaker: user/agent/system",
                },
            },
            "required": ["turns"],
        },
    },
    {
        "name": "export_context",
        "description": "Export stored memories as compressed context, ready for system prompt injection. Use source_snippets=true for detail preservation (numbers, conditions).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 50,
                      "description": "Number of recent memories to export"},
                "source_snippets": {"type": "boolean", "default": True,
                                    "description": "Include source snippets for detail preservation"},
                "include_spec": {"type": "boolean", "default": False,
                                 "description": "Prepend FORMAT_SPEC for LLM understanding"},
            },
        },
    },
    {
        "name": "search_memory",
        "description": "Search stored memories by entity keyword. Uses fuzzy matching.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Entity to search for"},
                "threshold": {"type": "number", "default": 0.5,
                              "description": "Similarity threshold (0-1)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get memory store statistics: total memories, active count, dedup info.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_format_spec",
        "description": "Get the AILang-IR format specification for system prompt injection. Use this when you need to explain the compressed format to another LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "full": {"type": "boolean", "default": False,
                         "description": "Return full reference spec instead of compact version"},
            },
        },
    },
    {
        "name": "clear_memory",
        "description": "Clear all stored memories. Use with caution.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ===================================================================
# Tool implementations
# ===================================================================

def handle_compress_text(args: dict) -> str:
    pipe = _get_pipeline()
    text = args["text"]
    speaker = SPEAKER_MAP.get(args.get("speaker", "user"), SpeakerRole.USER)

    results = pipe.process_multi(text, speaker)
    codec = LLMCodec()

    codes = []
    for r in results:
        code = codec.encode(r.frame, act_labels=True, source_snippet=True)
        codes.append(code)

    _save_if_configured()

    return "\n".join(codes) + f"\n\n({len(results)} frames, {pipe.memory_size} total memories)"


def handle_compress_conversation(args: dict) -> str:
    pipe = _get_pipeline()
    turns = [(t[0], t[1]) for t in args["turns"]]

    results = pipe.process_conversation(turns)
    codec = LLMCodec()

    codes = []
    for r in results:
        code = codec.encode(r.frame, act_labels=True, source_snippet=True)
        codes.append(code)

    _save_if_configured()

    dedup_rate = (1 - pipe.memory_size / len(turns)) * 100 if turns else 0
    return "\n".join(codes) + f"\n\n({len(turns)} turns → {pipe.memory_size} unique, {dedup_rate:.0f}% dedup)"


def handle_export_context(args: dict) -> str:
    pipe = _get_pipeline()
    n = args.get("n", 50)
    snippets = args.get("source_snippets", True)
    include_spec = args.get("include_spec", False)

    if pipe.memory_size == 0:
        return "No memories stored. Use compress_text or compress_conversation first."

    parts = []
    if include_spec:
        parts.append(get_format_spec())
        parts.append("\nCompressed context:")

    exported = pipe.export_context(n=n, source_snippets=snippets)
    parts.append(exported)

    actual = min(n, pipe.memory_size)
    parts.append(f"\n({actual} of {pipe.memory_size} memories exported)")

    return "\n".join(parts)


def handle_search_memory(args: dict) -> str:
    pipe = _get_pipeline()
    query = args["query"]
    threshold = args.get("threshold", 0.5)

    results = pipe.memory.query_by_entity_fuzzy(query, threshold=threshold)

    if not results:
        return f"No results for '{query}' (threshold={threshold})"

    codec = LLMCodec()
    lines = []
    for mem, sim in results[:10]:
        code = codec.encode(mem.frame, act_labels=True, source_snippet=True)
        lines.append(f"  [{sim:.2f}] {code}")

    return f"Found {len(results)} results for '{query}':\n" + "\n".join(lines)


def handle_get_stats(args: dict) -> str:
    pipe = _get_pipeline()
    stats = pipe.stats()
    return json.dumps({
        "total_memories": stats["total_memories"],
        "active_memories": stats["active_memories"],
        "store_path": _store_path or "(in-memory only)",
    }, indent=2)


def handle_get_format_spec(args: dict) -> str:
    if args.get("full", False):
        return get_format_spec_full()
    return get_format_spec()


def handle_clear_memory(args: dict) -> str:
    global _pipeline
    _pipeline = Pipeline()
    return "Memory cleared."


TOOL_HANDLERS = {
    "compress_text": handle_compress_text,
    "compress_conversation": handle_compress_conversation,
    "export_context": handle_export_context,
    "search_memory": handle_search_memory,
    "get_stats": handle_get_stats,
    "get_format_spec": handle_get_format_spec,
    "clear_memory": handle_clear_memory,
}


# ===================================================================
# MCP JSON-RPC protocol
# ===================================================================

def handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        global _store_path
        # Check for store path in initialization options
        init_options = params.get("initializationOptions", {})
        _store_path = init_options.get("store_path")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "ailang-ir",
                    "version": "0.1.0",
                },
            },
        }

    elif method == "notifications/initialized":
        # No response needed for notifications
        return None  # type: ignore

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS,
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            }

        try:
            result_text = handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": result_text},
                    ],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": f"Error: {e}"},
                    ],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    else:
        # Unknown method — return error for requests, ignore notifications
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        return None  # type: ignore


def main():
    """Run the MCP server over stdio."""
    # MCP uses newline-delimited JSON over stdio
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
