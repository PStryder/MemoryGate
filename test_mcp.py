#!/usr/bin/env python3
"""Test MemoryGate MCP protocol."""
import json
import os

import pytest
import requests

BASE_URL = os.getenv("MEMORYGATE_MCP_BASE_URL")

if not BASE_URL:
    pytest.skip(
        "Set MEMORYGATE_MCP_BASE_URL to run MCP integration tests",
        allow_module_level=True,
    )


def mcp_call(method, params=None):
    """Call MCP tool via JSON-RPC."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": method, "arguments": params or {}},
    }

    resp = requests.post(f"{BASE_URL}/mcp", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def test_mcp_protocol():
    result = mcp_call(
        "memory_store",
        {
            "observation": "Test memory via MCP protocol",
            "domain": "mcp_test",
            "confidence": 0.9,
        },
    )
    assert result["jsonrpc"] == "2.0"

    result = mcp_call("memory_recall", {"domain": "mcp_test", "limit": 10})
    assert result["jsonrpc"] == "2.0"

    result = mcp_call("memory_stats", {})
    assert result["jsonrpc"] == "2.0"
