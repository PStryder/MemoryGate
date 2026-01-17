"""
MCP server wiring and tool registration.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from fastmcp import FastMCP

import core.config as config
from core.context import AuthContext, RequestContext
from core.services import memory_service
from core.mcp.auth_middleware import get_current_context, MCPAuthMiddleware

READ_ONLY_TOOL_ANNOTATIONS = {"readOnlyHint": True}
DESTRUCTIVE_TOOL_ANNOTATIONS = {"destructiveHint": True}

mcp = FastMCP("MemoryGate")

_REGISTERED_TOOLS: list[tuple[Callable[..., dict], tuple[Any, ...], dict[str, Any]]] = []
_TOOL_REGISTRY_LOCK = threading.Lock()
_LAST_TOOL_COUNT: Optional[int] = None
_TOOL_INVENTORY_EMPTY_LOGGED = False


def mcp_tool(*args, **kwargs):
    """Register a tool with FastMCP and keep a local registry for rebinding."""
    def decorator(fn: Callable[..., dict]):
        _REGISTERED_TOOLS.append((fn, args, kwargs))
        mcp.tool(*args, **kwargs)(fn)
        return fn
    return decorator


def _record_tool_inventory_count(tool_count: int) -> None:
    global _LAST_TOOL_COUNT, _TOOL_INVENTORY_EMPTY_LOGGED
    with _TOOL_REGISTRY_LOCK:
        if tool_count == 0 and (_LAST_TOOL_COUNT is None or _LAST_TOOL_COUNT > 0):
            config.logger.warning(
                "tool_inventory_empty",
                extra={"tool_count": tool_count},
            )
            _TOOL_INVENTORY_EMPTY_LOGGED = True
        elif tool_count > 0 and _LAST_TOOL_COUNT == 0:
            config.logger.info(
                "tool_inventory_restored",
                extra={"tool_count": tool_count},
            )
            _TOOL_INVENTORY_EMPTY_LOGGED = False
        _LAST_TOOL_COUNT = tool_count


def _rebind_tool_registry(reason: str) -> None:
    with _TOOL_REGISTRY_LOCK:
        for fn, args, kwargs in _REGISTERED_TOOLS:
            mcp.tool(*args, **kwargs)(fn)
        tool_count = len(mcp._tool_manager._tools)
        config.logger.warning(
            "tool_registry_rebind",
            extra={"reason": reason, "tool_count": tool_count},
        )


async def tool_inventory_status(refresh_if_empty: bool = False, reason: str = "") -> dict:
    """Return tool inventory details and optionally rebind when empty."""
    tools = await mcp.get_tools()
    tool_names = sorted(tools.keys())
    tool_count = len(tool_names)

    refreshed = False
    if refresh_if_empty and tool_count == 0:
        refreshed = True
        _rebind_tool_registry(reason or "inventory_empty")
        tools = await mcp.get_tools()
        tool_names = sorted(tools.keys())
        tool_count = len(tool_names)

    _record_tool_inventory_count(tool_count)

    return {
        "tool_count": tool_count,
        "tools": tool_names,
        "refreshed": refreshed,
        "retry_after_seconds": config.TOOL_INVENTORY_RETRY_SECONDS if tool_count == 0 else None,
    }


async def _mcp_tool_inventory_check() -> dict:
    return await tool_inventory_status(refresh_if_empty=True, reason="mcp_request")


@mcp.resource(
    "memorygate://tool-inventory",
    name="memorygate_tool_inventory",
    mime_type="application/json",
)
async def tool_inventory_resource() -> dict:
    """Expose tool inventory as a resource for discovery fallbacks."""
    return await tool_inventory_status(refresh_if_empty=True, reason="resource_read")


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_search(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        domain=domain,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def search_cold_memory(
    query: str,
    top_k: int = 10,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    type_filter: Optional[str] = None,
    source: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags: Optional[list[str]] = None,
    include_evidence: bool = True,
    bump_score: bool = False,
) -> dict:
    return memory_service.search_cold_memory(
        query=query,
        top_k=top_k,
        min_score=min_score,
        max_score=max_score,
        type_filter=type_filter,
        source=source,
        date_from=date_from,
        date_to=date_to,
        tags=tags,
        include_evidence=include_evidence,
        bump_score=bump_score,
        context=get_current_context(),
    )


@mcp_tool(annotations=DESTRUCTIVE_TOOL_ANNOTATIONS)
def archive_memory(
    memory_ids: Optional[list[str]] = None,
    summary_ids: Optional[list[int]] = None,
    cluster_ids: Optional[list[str]] = None,
    threshold: Optional[dict] = None,
    mode: str = "archive_and_tombstone",
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = True,
    limit: int = config.ARCHIVE_LIMIT_DEFAULT,
) -> dict:
    return memory_service.archive_memory(
        memory_ids=memory_ids,
        summary_ids=summary_ids,
        cluster_ids=cluster_ids,
        threshold=threshold,
        mode=mode,
        reason=reason,
        actor=actor,
        dry_run=dry_run,
        limit=limit,
        context=get_current_context(),
    )


@mcp_tool(annotations=DESTRUCTIVE_TOOL_ANNOTATIONS)
def rehydrate_memory(
    memory_ids: Optional[list[str]] = None,
    summary_ids: Optional[list[int]] = None,
    cluster_ids: Optional[list[str]] = None,
    threshold: Optional[dict] = None,
    query: Optional[str] = None,
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = False,
    limit: int = 50,
    bump_score: bool = True,
) -> dict:
    return memory_service.rehydrate_memory(
        memory_ids=memory_ids,
        summary_ids=summary_ids,
        cluster_ids=cluster_ids,
        threshold=threshold,
        query=query,
        reason=reason,
        actor=actor,
        dry_run=dry_run,
        limit=limit,
        bump_score=bump_score,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def list_archive_candidates(
    below_score: float = config.SUMMARY_TRIGGER_SCORE,
    limit: int = config.ARCHIVE_LIMIT_DEFAULT,
) -> dict:
    return memory_service.list_archive_candidates(
        below_score=below_score,
        limit=limit,
        context=get_current_context(),
    )


@mcp_tool()
def memory_store(
    observation: str,
    confidence: float = 0.8,
    domain: Optional[str] = None,
    evidence: Optional[list[str]] = None,
    ai_name: str = "Unknown",
    ai_platform: str = "Unknown",
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None,
) -> dict:
    return memory_service.memory_store(
        observation=observation,
        confidence=confidence,
        domain=domain,
        evidence=evidence,
        ai_name=ai_name,
        ai_platform=ai_platform,
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_recall(
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    ai_name: Optional[str] = None,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_recall(
        domain=domain,
        min_confidence=min_confidence,
        limit=limit,
        ai_name=ai_name,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_stats() -> dict:
    return memory_service.memory_stats(context=get_current_context())


@mcp_tool()
def memory_init_session(
    conversation_id: str,
    title: str,
    ai_name: str,
    ai_platform: str,
    source_url: Optional[str] = None,
) -> dict:
    return memory_service.memory_init_session(
        conversation_id=conversation_id,
        title=title,
        ai_name=ai_name,
        ai_platform=ai_platform,
        source_url=source_url,
        context=get_current_context(),
    )


@mcp_tool()
def memory_store_document(
    title: str,
    doc_type: str,
    url: str,
    content_summary: str,
    key_concepts: Optional[list[str]] = None,
    publication_date: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    return memory_service.memory_store_document(
        title=title,
        doc_type=doc_type,
        url=url,
        content_summary=content_summary,
        key_concepts=key_concepts,
        publication_date=publication_date,
        metadata=metadata,
        context=get_current_context(),
    )


@mcp_tool()
def memory_store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[dict] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
) -> dict:
    return memory_service.memory_store_concept(
        name=name,
        concept_type=concept_type,
        description=description,
        domain=domain,
        status=status,
        metadata=metadata,
        ai_name=ai_name,
        ai_platform=ai_platform,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_get_concept(
    name: str,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_get_concept(
        name=name,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool()
def memory_add_concept_alias(
    concept_name: str,
    alias: str,
) -> dict:
    return memory_service.memory_add_concept_alias(
        concept_name=concept_name,
        alias=alias,
        context=get_current_context(),
    )


@mcp_tool()
def memory_add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: Optional[str] = None,
) -> dict:
    return memory_service.memory_add_concept_relationship(
        from_concept=from_concept,
        to_concept=to_concept,
        rel_type=rel_type,
        weight=weight,
        description=description,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_related_concepts(
        concept_name=concept_name,
        rel_type=rel_type,
        min_weight=min_weight,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool()
def memory_update_pattern(
    category: str,
    pattern_name: str,
    pattern_text: str,
    confidence: float = 0.8,
    evidence_observation_ids: Optional[list[int]] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict:
    return memory_service.memory_update_pattern(
        category=category,
        pattern_name=pattern_name,
        pattern_text=pattern_text,
        confidence=confidence,
        evidence_observation_ids=evidence_observation_ids,
        ai_name=ai_name,
        ai_platform=ai_platform,
        conversation_id=conversation_id,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_get_pattern(
    category: str,
    pattern_name: str,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_get_pattern(
        category=category,
        pattern_name=pattern_name,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_patterns(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 20,
    include_cold: bool = False,
) -> dict:
    return memory_service.memory_patterns(
        category=category,
        min_confidence=min_confidence,
        limit=limit,
        include_cold=include_cold,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_user_guide(
    format: str = "markdown",
    verbosity: str = "short",
) -> dict:
    return memory_service.memory_user_guide(
        format=format,
        verbosity=verbosity,
        context=get_current_context(),
    )


@mcp_tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def memory_bootstrap(
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
) -> dict:
    return memory_service.memory_bootstrap(
        ai_name=ai_name,
        ai_platform=ai_platform,
        context=get_current_context(),
    )


mcp_sse_app = MCPAuthMiddleware(mcp.http_app(
    path="/",
    transport="sse",
    stateless_http=True,
    json_response=True,
))

mcp_stream_app = MCPAuthMiddleware(mcp.http_app(
    path="/",
    transport="streamable-http",
    stateless_http=True,
    json_response=True,
))


class MCPRouteNormalizerASGI:
    """Pure ASGI middleware - no response buffering, SSE-safe."""
    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path")
            if path == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
            elif path == "/MemoryGate":
                scope = dict(scope)
                scope["path"] = "/MemoryGate/"
        await self.wrapped_app(scope, receive, send)


class MemoryGateAliasASGI:
    """Handle /MemoryGate/link_<id> routing while preserving SSE root_path."""
    def __init__(self, mcp_entry_app):
        self.mcp_entry_app = mcp_entry_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.mcp_entry_app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        full_path = scope.get("path", "") or "/"
        relative_path = full_path
        if root_path and full_path.startswith(root_path):
            relative_path = full_path[len(root_path):] or "/"

        if relative_path.startswith("/link_"):
            parts = relative_path.split("/", 2)
            link_segment = parts[1]
            remaining_path = "/" + parts[2] if len(parts) > 2 else "/"
            new_scope = dict(scope)
            new_scope["root_path"] = f"{root_path.rstrip('/')}/{link_segment}"
            new_scope["path"] = remaining_path
            new_scope["raw_path"] = remaining_path.encode()
            await self.mcp_entry_app(new_scope, receive, send)
            return

        new_scope = dict(scope)
        new_scope["path"] = relative_path
        new_scope["raw_path"] = relative_path.encode()
        await self.mcp_entry_app(new_scope, receive, send)
