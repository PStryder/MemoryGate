#!/usr/bin/env python3
"""Test MemoryGate v2 MCP protocol (integration)."""
import os

import pytest
import requests

MCP_URL = os.getenv("MEMORYGATE_MCP_BASE_URL")

if not MCP_URL:
    pytest.skip(
        "Set MEMORYGATE_MCP_BASE_URL to run MCP integration tests",
        allow_module_level=True,
    )


def test_mcp_v2_protocol():
    resp = requests.get(f"{MCP_URL}/", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    assert "service" in info

    resp = requests.get(f"{MCP_URL}/health", timeout=10)
    resp.raise_for_status()
    health = resp.json()
    assert "status" in health
