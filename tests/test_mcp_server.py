"""Tests for MCP server JSON-RPC protocol and tool handlers."""

from __future__ import annotations

import json
import pytest

from ailang_ir.mcp_server import handle_request, TOOLS, TOOL_HANDLERS, _get_pipeline

# Reset global state before each test
@pytest.fixture(autouse=True)
def reset_mcp_state():
    import ailang_ir.mcp_server as srv
    srv._pipeline = None
    srv._store_path = None
    yield
    srv._pipeline = None
    srv._store_path = None


class TestMCPProtocol:
    """Test JSON-RPC 2.0 protocol handling."""

    def test_initialize(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "ailang-ir"

    def test_initialize_with_store_path(self):
        import ailang_ir.mcp_server as srv
        handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "initializationOptions": {"store_path": "/tmp/test.json"}
        }})
        assert srv._store_path == "/tmp/test.json"

    def test_tools_list(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = resp["result"]["tools"]
        assert len(tools) == 7
        names = {t["name"] for t in tools}
        assert names == {"compress_text", "compress_conversation", "export_context",
                         "search_memory", "get_stats", "get_format_spec", "clear_memory"}

    def test_ping(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}})
        assert resp["result"] == {}

    def test_notification_returns_none(self):
        resp = handle_request({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        assert resp is None

    def test_unknown_method_with_id(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 99, "method": "nonexistent", "params": {}})
        assert resp["error"]["code"] == -32601

    def test_unknown_method_without_id(self):
        resp = handle_request({"jsonrpc": "2.0", "method": "nonexistent/notify", "params": {}})
        assert resp is None

    def test_unknown_tool(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
            "name": "nonexistent_tool", "arguments": {}
        }})
        assert resp["error"]["code"] == -32601
        assert "nonexistent_tool" in resp["error"]["message"]


class TestToolHandlers:
    """Test individual tool implementations."""

    def test_compress_text(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "We should use Redis for caching.", "speaker": "user"}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "1 frames" in content
        assert "1 total memories" in content

    def test_compress_text_default_speaker(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "The tests are passing now."}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "frames" in content

    def test_compress_conversation(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_conversation",
            "arguments": {"turns": [
                ["user", "Fix the login bug."],
                ["agent", "I found the issue in auth.py."],
            ]}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "2 turns" in content
        assert "unique" in content

    def test_export_context_empty(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "export_context", "arguments": {}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "No memories" in content

    def test_export_context_with_data(self):
        # First compress something
        handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "Deploy to production server."}
        }})
        # Then export
        resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "export_context", "arguments": {"n": 5, "source_snippets": True}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "1 of 1 memories exported" in content

    def test_export_context_with_spec(self):
        handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "Add unit tests."}
        }})
        resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "export_context", "arguments": {"include_spec": True}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "FORMAT_SPEC" in content or "HEADER" in content

    def test_search_memory_no_results(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "search_memory", "arguments": {"query": "nonexistent_xyz"}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "No results" in content

    def test_search_memory_with_results(self):
        handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "The database needs optimization."}
        }})
        resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "search_memory", "arguments": {"query": "database"}
        }})
        content = resp["result"]["content"][0]["text"]
        assert "Found" in content
        assert "database" in content.lower()

    def test_get_stats(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "get_stats", "arguments": {}
        }})
        stats = json.loads(resp["result"]["content"][0]["text"])
        assert stats["total_memories"] == 0
        assert stats["store_path"] == "(in-memory only)"

    def test_get_format_spec_compact(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "get_format_spec", "arguments": {}
        }})
        content = resp["result"]["content"][0]["text"]
        assert len(content) > 100

    def test_get_format_spec_full(self):
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "get_format_spec", "arguments": {"full": True}
        }})
        content = resp["result"]["content"][0]["text"]
        assert len(content) > 100

    def test_clear_memory(self):
        # Add data first
        handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_text",
            "arguments": {"text": "Something to remember."}
        }})
        # Clear
        resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "clear_memory", "arguments": {}
        }})
        assert "cleared" in resp["result"]["content"][0]["text"].lower()
        # Verify empty
        resp = handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "get_stats", "arguments": {}
        }})
        stats = json.loads(resp["result"]["content"][0]["text"])
        assert stats["total_memories"] == 0

    def test_tool_error_handling(self):
        """Tool errors should return isError=True, not crash."""
        resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "compress_conversation",
            "arguments": {"turns": "not_a_list"}
        }})
        assert resp["result"]["isError"] is True
        assert "Error" in resp["result"]["content"][0]["text"]


class TestToolDefinitions:
    """Verify tool schema completeness."""

    def test_all_tools_have_handlers(self):
        tool_names = {t["name"] for t in TOOLS}
        handler_names = set(TOOL_HANDLERS.keys())
        assert tool_names == handler_names

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_all_tools_have_description(self):
        for tool in TOOLS:
            assert "description" in tool
            assert len(tool["description"]) > 10
