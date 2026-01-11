import os
from dataclasses import dataclass

import httpx
import pytest

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")
os.environ.setdefault("EMBEDDING_PROVIDER", "none")

from core.models import Concept, Observation
from app.main import app
from core.mcp import mcp_stream_app


@dataclass
class ToolSeed:
    observation_id: int
    concept_name: str
    concept_peer: str
    alias: str


def _seed_context(server_db) -> ToolSeed:
    db = server_db.SessionLocal()
    try:
        observation = Observation(
            observation="Tool routing seed observation",
            confidence=0.9,
            domain="tool_seed",
        )
        concept_name = "ToolSeedConceptA"
        concept_peer = "ToolSeedConceptB"
        concept_a = Concept(
            name=concept_name,
            name_key=concept_name.lower(),
            type="project",
            description="Tool seed concept A",
        )
        concept_b = Concept(
            name=concept_peer,
            name_key=concept_peer.lower(),
            type="project",
            description="Tool seed concept B",
        )
        db.add_all([observation, concept_a, concept_b])
        db.commit()
        db.refresh(observation)
        return ToolSeed(
            observation_id=observation.id,
            concept_name=concept_name,
            concept_peer=concept_peer,
            alias="ToolSeedAlias",
        )
    finally:
        db.close()


_MISSING = object()


def _field_override(field_name: str, seed: ToolSeed):
    overrides = {
        "conversation_id": "tool-session-1",
        "title": "Tool Session",
        "ai_name": "ToolAI",
        "ai_platform": "ToolPlatform",
        "query": "tool seed",
        "category": "tool_test",
        "pattern_name": "tool_pattern",
        "pattern_text": "Tool pattern text",
        "observation": "Tool observation",
        "domain": "tool_seed",
        "name": seed.concept_name,
        "concept_name": seed.concept_name,
        "concept_type": "project",
        "description": "Tool concept description",
        "from_concept": seed.concept_name,
        "to_concept": seed.concept_peer,
        "rel_type": "related_to",
        "alias": seed.alias,
        "doc_type": "article",
        "url": "https://example.com",
        "content_summary": "Tool doc summary",
        "key_concepts": [seed.concept_name],
        "publication_date": "2020-01-01",
        "metadata": {"source": "tool"},
        "evidence_observation_ids": [seed.observation_id],
        "memory_ids": [f"observation:{seed.observation_id}"],
        "reason": "tool routing test",
        "confidence": 0.9,
        "min_confidence": 0.0,
        "min_weight": 0.0,
        "limit": 1,
        "include_cold": False,
    }
    return overrides.get(field_name, _MISSING)


def _value_for_schema(field_name: str, schema: dict, seed: ToolSeed):
    override = _field_override(field_name, seed)
    if override is not _MISSING:
        return override

    if "default" in schema:
        return schema["default"]

    if "enum" in schema:
        return schema["enum"][0]

    schema_type = schema.get("type")
    if schema_type == "string":
        return "test"
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 0.5
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        item_schema = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        if min_items > 0:
            return [
                _value_for_schema(field_name, item_schema, seed)
                for _ in range(min_items)
            ]
        return []
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return {
            req: _value_for_schema(req, properties.get(req, {}), seed)
            for req in required
        }
    return None


def _tool_overrides(tool_name: str, seed: ToolSeed) -> dict:
    overrides = {
        "archive_memory": {
            "dry_run": True,
            "reason": "tool routing test",
            "memory_ids": [f"observation:{seed.observation_id}"],
        },
        "rehydrate_memory": {
            "dry_run": True,
            "reason": "tool routing test",
            "memory_ids": [f"observation:{seed.observation_id}"],
        },
        "memory_update_pattern": {
            "evidence_observation_ids": [seed.observation_id],
        },
    }
    return overrides.get(tool_name, {})


def _build_args(tool: dict, seed: ToolSeed) -> dict:
    schema = tool.get("inputSchema") or {}
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    args = {}
    for field in required:
        args[field] = _value_for_schema(field, properties.get(field, {}), seed)
    args.update(_tool_overrides(tool.get("name", ""), seed))
    return args


async def _jsonrpc_call(client: httpx.AsyncClient, url: str, method: str, params: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
    response = await client.post(
        url,
        json=payload,
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    assert "error" not in data
    return data.get("result", {})


@pytest.mark.anyio
async def test_tool_routes_inventory_and_calls(server_db):
    seed = _seed_context(server_db)
    base_urls = [
        "http://testserver/MemoryGate/",
        "http://testserver/MemoryGate/link_test/",
    ]

    async with mcp_stream_app.lifespan(mcp_stream_app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for base_url in base_urls:
                list_result = await _jsonrpc_call(client, base_url, "tools/list", {})
                tools = list_result.get("tools", [])
                assert tools
                for tool in tools:
                    args = _build_args(tool, seed)
                    call_result = await _jsonrpc_call(
                        client,
                        base_url,
                        "tools/call",
                        {"name": tool.get("name"), "arguments": args},
                    )
                    assert call_result.get("isError") is not True
