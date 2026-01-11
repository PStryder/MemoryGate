"""
Self-documentation and bootstrap services.
"""

from __future__ import annotations

from typing import Optional

from core.context import RequestContext
from core.db import DB
from core.errors import ValidationIssue
from core.models import AIInstance, Session, Observation
from core.services.memory_shared import (
    _validate_optional_text,
    MAX_SHORT_TEXT_LENGTH,
    MAX_RESULT_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_TEXT_LENGTH,
    MAX_RELATIONSHIP_ITEMS,
    MAX_LIST_ITEMS,
    MAX_METADATA_BYTES,
    service_tool,
)

SPEC_VERSION = "0.1.0"

RECOMMENDED_DOMAINS = [
    "technical_milestone",
    "major_milestone",
    "project_context",
    "system_architecture",
    "interaction_patterns",
    "system_behavior",
    "identity",
    "preferences",
    "decisions",
]

CONCEPT_TYPES = [
    "project",
    "framework",
    "component",
    "construct",
    "theory",
]

RELATIONSHIP_TYPES = [
    "enables",
    "version_of",
    "part_of",
    "related_to",
    "implements",
    "demonstrates",
]

CONFIDENCE_GUIDE = {
    "1.0": "Direct observation, absolute certainty",
    "0.95-0.99": "Very high confidence, strong evidence",
    "0.85-0.94": "High confidence, solid evidence",
    "0.70-0.84": "Good confidence, some uncertainty",
    "0.50-0.69": "Moderate confidence, competing interpretations",
    "<0.50": "Speculative, weak evidence",
}


@service_tool
def memory_user_guide(
    format: str = "markdown",
    verbosity: str = "short",
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Get self-documentation for MemoryGate system.
    
    Returns usage guide, schemas, recommended practices, and examples
    so AI agents can bootstrap themselves without manual configuration.
    
    Args:
        format: Output format (markdown or json)
        verbosity: short (recommended) or verbose (comprehensive)
    
    Returns:
        Dictionary with spec_version, guide content, structured metadata
    """
    if format not in {"markdown", "json"}:
        raise ValidationIssue(
            "format must be 'markdown' or 'json'",
            field="format",
            error_type="invalid_value",
        )
    if verbosity not in {"short", "verbose"}:
        raise ValidationIssue(
            "verbosity must be 'short' or 'verbose'",
            field="verbosity",
            error_type="invalid_value",
        )
    
    guide_content = """# MemoryGate User Guide

**Version:** {spec_version}

## Purpose

MemoryGate is a persistent Memory-as-a-Service system for AI agents. It provides:
- **Observations**: Discrete facts with confidence and evidence
- **Patterns**: Synthesized understanding across observations  
- **Concepts**: Canonical entities in a knowledge graph
- **Documents**: References to external content (not full copies)
- **Semantic search**: Unified vector search across all types

## Core Workflow

### 1. Initialize Session
Always start new conversations with:
```python
memory_init_session(
    conversation_id="unique-uuid",
    title="Description of conversation",
    ai_name="YourName",
    ai_platform="YourPlatform"
)
```

### 2. Search Before Answering
Use semantic search liberally (~50ms, fast):
```python
memory_search(query="relevant topic", limit=5)
```

### 3. Store New Information
**Observations** - discrete facts:
```python
memory_store(
    observation="User prefers TypeScript",
    confidence=0.9,
    domain="preferences",
    evidence=["Stated explicitly"]
)
```

**Concepts** - new frameworks/projects:
```python
memory_store_concept(
    name="MemoryGate",
    concept_type="project",
    description="Memory service for AI agents"
)
```

**Patterns** - synthesized understanding:
```python
memory_update_pattern(
    category="interaction_patterns",
    pattern_name="direct_communication",
    pattern_text="User values directness",
    confidence=0.85
)
```

## Critical Invariants

1. **Concept names are case-insensitive**
2. **Aliases prevent fragmentation**
3. **Patterns are upserts** - safe to call repeatedly
4. **Documents store references, not content**
5. **Search is primary tool** - search first, then answer

## Recommended Domains
{domains}

## Confidence Levels
{confidence}

## Concept Types
{concept_types}

## Relationship Types
{relationship_types}

## Limits (defaults)
- **Search result limit**: {max_result_limit} (`MEMORYGATE_MAX_RESULT_LIMIT`)
- **Query length**: {max_query_length} chars (`MEMORYGATE_MAX_QUERY_LENGTH`)
- **Text length**: {max_text_length} chars (`MEMORYGATE_MAX_TEXT_LENGTH`)
- **Short text length**: {max_short_text_length} chars (`MEMORYGATE_MAX_SHORT_TEXT_LENGTH`)
- **Evidence list size**: {max_relationship_items} observation IDs (`MEMORYGATE_MAX_RELATIONSHIP_ITEMS`)
- **List sizes**: {max_list_items} items (`MEMORYGATE_MAX_LIST_ITEMS`)
- **Metadata size**: {max_metadata_bytes} bytes (`MEMORYGATE_MAX_METADATA_BYTES`)

Limits can be configured per deployment via the environment variables above.
""".format(
        spec_version=SPEC_VERSION,
        domains="\n".join(f"- `{d}`" for d in RECOMMENDED_DOMAINS),
        confidence="\n".join(f"- **{k}**: {v}" for k, v in CONFIDENCE_GUIDE.items()),
        concept_types="\n".join(f"- `{ct}`" for ct in CONCEPT_TYPES),
        relationship_types="\n".join(f"- `{rt}`" for rt in RELATIONSHIP_TYPES),
        max_result_limit=MAX_RESULT_LIMIT,
        max_query_length=MAX_QUERY_LENGTH,
        max_text_length=MAX_TEXT_LENGTH,
        max_short_text_length=MAX_SHORT_TEXT_LENGTH,
        max_relationship_items=MAX_RELATIONSHIP_ITEMS,
        max_list_items=MAX_LIST_ITEMS,
        max_metadata_bytes=MAX_METADATA_BYTES,
    )
    
    result = {
        "spec_version": SPEC_VERSION,
        "recommended_domains": RECOMMENDED_DOMAINS,
        "concept_types": CONCEPT_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "confidence_guide": CONFIDENCE_GUIDE,
    }
    
    if format == "markdown":
        result["guide"] = guide_content
    else:  # json
        result["guide"] = {
            "purpose": "Memory-as-a-Service for AI agents",
            "core_workflow": [
                "Initialize session with memory_init_session()",
                "Search with memory_search() before answering",
                "Store new info with memory_store/memory_store_concept/memory_update_pattern",
            ],
            "critical_invariants": [
                "Concept names are case-insensitive",
                "Aliases prevent fragmentation",
                "Patterns are upserts",
                "Documents store references not content",
                "Search is primary tool",
            ],
            "limits": {
                "max_result_limit": MAX_RESULT_LIMIT,
                "max_query_length": MAX_QUERY_LENGTH,
                "max_text_length": MAX_TEXT_LENGTH,
                "max_short_text_length": MAX_SHORT_TEXT_LENGTH,
                "max_relationship_items": MAX_RELATIONSHIP_ITEMS,
                "max_list_items": MAX_LIST_ITEMS,
                "max_metadata_bytes": MAX_METADATA_BYTES,
            },
            "limit_env_vars": {
                "MEMORYGATE_MAX_RESULT_LIMIT": MAX_RESULT_LIMIT,
                "MEMORYGATE_MAX_QUERY_LENGTH": MAX_QUERY_LENGTH,
                "MEMORYGATE_MAX_TEXT_LENGTH": MAX_TEXT_LENGTH,
                "MEMORYGATE_MAX_SHORT_TEXT_LENGTH": MAX_SHORT_TEXT_LENGTH,
                "MEMORYGATE_MAX_RELATIONSHIP_ITEMS": MAX_RELATIONSHIP_ITEMS,
                "MEMORYGATE_MAX_LIST_ITEMS": MAX_LIST_ITEMS,
                "MEMORYGATE_MAX_METADATA_BYTES": MAX_METADATA_BYTES,
            },
        }
    
    return result


@service_tool
def memory_bootstrap(
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Stateful bootstrap for AI agents - tells you your relationship status with MemoryGate.
    
    Returns compatibility info, connection history, and getting started guide.
    The system tells you what it already knows about you.
    
    Args:
        ai_name: Your AI instance name (e.g., "Kee", "Hexy")
        ai_platform: Your platform (e.g., "Claude", "ChatGPT")
    
    Returns:
        Relationship status, version info, and usage guide
    """
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        # Check if this AI instance has history
        connection_status = {
            "is_new_instance": True,
            "first_seen": None,
            "last_seen": None,
            "session_count": 0,
            "total_observations": 0,
        }
        
        if ai_name and ai_platform:
            # Query for this AI instance's history
            ai_instance_query = db.query(AIInstance).filter(
                AIInstance.name == ai_name,
                AIInstance.platform == ai_platform
            ).first()
            
            if ai_instance_query:
                # Found existing instance
                connection_status["is_new_instance"] = False
                connection_status["first_seen"] = ai_instance_query.created_at.isoformat()
                
                # Get last session to determine last_seen
                last_session = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id
                ).order_by(Session.started_at.desc()).first()
                
                if last_session:
                    connection_status["last_seen"] = last_session.started_at.isoformat()
                else:
                    connection_status["last_seen"] = ai_instance_query.created_at.isoformat()
                
                # Count sessions
                session_count = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id
                ).count()
                connection_status["session_count"] = session_count
                
                # Count observations contributed by this AI
                obs_count = db.query(Observation).filter(
                    Observation.ai_instance_id == ai_instance_query.id
                ).count()
                connection_status["total_observations"] = obs_count
        
        return {
            "spec_version": SPEC_VERSION,
            "role": "agent_bootstrap",
            "connection_status": connection_status,
            "assumptions": [
                "You are an AI agent using MCP",
                "You may reconnect across sessions",
                "You should externalize durable memory here",
                "Memory persists between your instances",
                "Other AI agents may also use this memory"
            ],
            "compatibility": {
                "spec_version": SPEC_VERSION,
                "breaking_changes_since": None,  # Future: track version bumps
                "recommended_action": "Use normally" if not connection_status["is_new_instance"] else "Initialize new session"
            },
            "first_steps": [
                {
                    "step": 1,
                    "action": "Initialize session" if connection_status["is_new_instance"] else "Initialize this session",
                    "tool": "memory_init_session",
                    "required_params": ["conversation_id", "title", "ai_name", "ai_platform"],
                    "note": "Creates session record and updates last_seen" if not connection_status["is_new_instance"] else "Registers you as new AI instance"
                },
                {
                    "step": 2,
                    "action": "Search for relevant context",
                    "tool": "memory_search",
                    "params": {"query": "topic keywords", "limit": 5},
                    "note": f"You have {connection_status['total_observations']} observations in the system" if connection_status["total_observations"] > 0 else "System is empty - you'll build memory as you go"
                },
                {
                    "step": 3,
                    "action": "Store new observations",
                    "tool": "memory_store",
                    "params": {
                        "observation": "What you learned",
                        "confidence": 0.8,
                        "domain": "appropriate_domain",
                        "evidence": ["supporting facts"],
                    },
                },
            ],
            "critical_rules": [
                "ALWAYS call memory_init_session() at conversation start",
                "Search liberally - it's fast (~50ms)",
                "Concept names are case-insensitive",
                "Use confidence weights honestly (0.0-1.0)",
                "Documents are references only (Google Drive = canonical storage)"
            ],
            "recommended_domains": RECOMMENDED_DOMAINS,
            "confidence_guide": CONFIDENCE_GUIDE,
            "next_step": "Call memory_user_guide() for full documentation",
        }
    finally:
        db.close()

