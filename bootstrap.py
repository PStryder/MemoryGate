"""
Self-documentation tools for MemoryGate.

Allows AI agents to discover capabilities, schemas, usage patterns,
and recommended practices without requiring manual user configuration.
"""

from typing import Literal, Optional
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("MemoryGate Self-Documentation")


SPEC_VERSION = "0.1.0"

# Canonical domain types (most commonly used)
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

# Concept types
CONCEPT_TYPES = [
    "project",
    "framework", 
    "component",
    "construct",
    "theory",
]

# Relationship types
RELATIONSHIP_TYPES = [
    "enables",
    "version_of",
    "part_of",
    "related_to",
    "implements",
    "demonstrates",
]

# Confidence guide
CONFIDENCE_GUIDE = {
    "1.0": "Direct observation, absolute certainty",
    "0.95-0.99": "Very high confidence, strong evidence",
    "0.85-0.94": "High confidence, solid evidence",
    "0.70-0.84": "Good confidence, some uncertainty",
    "0.50-0.69": "Moderate confidence, competing interpretations",
    "<0.50": "Speculative, weak evidence",
}

USER_GUIDE_SHORT = """# MemoryGate User Guide

**Version:** {spec_version}

## Purpose

MemoryGate is a persistent Memory-as-a-Service system for AI agents. It provides:
- **Observations**: Discrete facts with confidence and evidence
- **Patterns**: Synthesized understanding across observations  
- **Concepts**: Canonical entities in a knowledge graph
- **Documents**: References to external content (not full copies)
- **Semantic search**: Unified vector search across all types

## Authentication

- All MCP requests require a valid API key
- Passed via `Authorization: Bearer <token>` or `X-API-Key` header
- Managed by OAuth flow (no manual key management needed)

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
Returns results across observations, patterns, concepts, and documents.

### 3. Store New Information
**Observations** - discrete facts:
```python
memory_store(
    observation="User prefers TypeScript over JavaScript",
    confidence=0.9,
    domain="preferences",
    evidence=["Stated explicitly in message"]
)
```

**Concepts** - when new frameworks/projects emerge:
```python
memory_store_concept(
    name="MemoryGate",
    concept_type="project",
    description="Persistent memory service for AI agents",
    domain="project_context"
)
```

**Patterns** - synthesized understanding (upsert operation):
```python
memory_update_pattern(
    category="interaction_patterns",
    pattern_name="prefers_direct_communication",
    pattern_text="User consistently values directness over politeness",
    confidence=0.85,
    evidence_observation_ids=[1, 5, 12]
)
```

## Data Types & Semantics

### Recommended Domains
{domains}

### Confidence Levels
{confidence}

### Concept Types
{concept_types}

### Relationship Types  
{relationship_types}

## Critical Invariants

1. **Concept names are case-insensitive**: `memory_get_concept("MemoryGate")` == `memory_get_concept("memorygate")`
2. **Aliases prevent fragmentation**: Add aliases when users reference same thing differently
3. **Patterns are upserts**: `memory_update_pattern()` creates or updates - safe to call repeatedly
4. **Documents store references, not content**: Full content lives in canonical storage (Google Drive)
5. **Search is primary tool**: Don't assume - search first, then answer

## Document Storage

Documents are **references only**:
- Store: title, URL, summary, key_concepts
- Canonical storage: Google Drive (full content)
- Fetch full content on demand via `gdrive_fetch` or similar

Example:
```python
memory_store_document(
    title="AI Memory Article",
    doc_type="article",
    url="https://drive.google.com/...",
    content_summary="Argues for externalized AI memory...",
    key_concepts=["MCP", "memory architecture"]
)
```

## Knowledge Graph Usage

**Get concept** (case-insensitive, alias-aware):
```python
memory_get_concept(name="MemoryGate")
```

**Add alias** (prevent fragmentation):
```python
memory_add_concept_alias(
    concept_name="MemoryGate",
    alias="MG"
)
```

**Add relationship**:
```python
memory_add_concept_relationship(
    from_concept="MemoryGate",
    to_concept="MCP",
    rel_type="implements",
    weight=0.9
)
```

## Promotion Rule

Observations → Concepts → Patterns

Only promote when confidence and evidence justify it.

## When to Store vs Reference

**Store in MemoryGate:**
- User preferences and behaviors
- Technical achievements/milestones  
- Specific factual observations with evidence
- Synthesized patterns across conversations

**DO NOT store:**
- Raw chat transcripts
- Speculative guesses without evidence
- Temporary planning thoughts
- Unverified inferences

## Example Session

```python
# 1. Initialize
memory_init_session(
    conversation_id="conv-123",
    title="MemoryGate OAuth Implementation",
    ai_name="Kee",
    ai_platform="Claude"
)

# 2. Search for context
results = memory_search(query="OAuth implementation", limit=5)

# 3. Store new milestone
memory_store(
    observation="Completed OAuth 2.0 + PKCE integration",
    confidence=0.95,
    domain="technical_milestone",
    evidence=["Deployment successful", "Tools working"]
)

# 4. Create concept for new project component
memory_store_concept(
    name="OAuth Discovery",
    concept_type="component",
    description="OAuth 2.0 discovery endpoints for MCP clients"
)

# 5. Link concepts
memory_add_concept_relationship(
    from_concept="OAuth Discovery",
    to_concept="MemoryGate",
    rel_type="part_of"
)
```

## Best Practices

1. **Search liberally** - it's fast, use it
2. **Store during session** - don't defer to "later"
3. **Use confidence weights** - be honest about certainty
4. **Provide evidence** - support observations with sources
5. **Create aliases early** - prevent concept fragmentation
6. **Update patterns** - they evolve, not frozen
7. **Check stats** - `memory_stats()` shows system health
"""

USER_GUIDE_VERBOSE = """# MemoryGate Comprehensive Guide

[Extended version would go here with more examples, edge cases, and deep dives]
"""


@mcp.tool()
def memory_user_guide(
    format: Literal["markdown", "json"] = "markdown",
    verbosity: Literal["short", "verbose"] = "short",
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
    
    # Format domains
    domains_md = "\n".join(f"- `{d}`" for d in RECOMMENDED_DOMAINS)
    
    # Format confidence guide
    confidence_md = "\n".join(
        f"- **{k}**: {v}" for k, v in CONFIDENCE_GUIDE.items()
    )
    
    # Format concept types
    concept_types_md = "\n".join(f"- `{ct}`" for ct in CONCEPT_TYPES)
    
    # Format relationship types
    rel_types_md = "\n".join(f"- `{rt}`" for rt in RELATIONSHIP_TYPES)
    
    # Select guide content
    if verbosity == "verbose":
        guide_content = USER_GUIDE_VERBOSE
    else:
        guide_content = USER_GUIDE_SHORT.format(
            spec_version=SPEC_VERSION,
            domains=domains_md,
            confidence=confidence_md,
            concept_types=concept_types_md,
            relationship_types=rel_types_md,
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
            "authentication": "API key via Authorization header",
            "core_workflow": [
                "Initialize session with memory_init_session()",
                "Search with memory_search() before answering",
                "Store new info with memory_store() / memory_store_concept() / memory_update_pattern()",
            ],
            "critical_invariants": [
                "Concept names are case-insensitive",
                "Aliases prevent fragmentation",  
                "Patterns are upserts",
                "Documents store references not content",
                "Search is primary tool",
            ],
        }
    
    return result


@mcp.tool()
def memory_bootstrap() -> dict:
    """
    Quick bootstrap guide for new AI agents.
    
    Returns minimal "getting started" checklist.
    Equivalent to memory_user_guide(format="json", verbosity="short")
    but optimized for first-time setup.
    
    Returns:
        Essential information to start using MemoryGate
    """
    return {
        "spec_version": SPEC_VERSION,
        "first_steps": [
            {
                "step": 1,
                "action": "Initialize session",
                "tool": "memory_init_session",
                "params": {
                    "conversation_id": "unique-uuid",
                    "title": "Conversation description",
                    "ai_name": "Your AI name",
                    "ai_platform": "Your platform",
                },
            },
            {
                "step": 2,
                "action": "Search for relevant context",
                "tool": "memory_search",
                "params": {"query": "topic keywords", "limit": 5},
            },
            {
                "step": 3,
                "action": "Store new observations as you learn",
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
            "ALWAYS call memory_init_session() first",
            "Search liberally with memory_search() - it's fast (~50ms)",
            "Concept names are case-insensitive",
            "Documents are references only (Google Drive = canonical storage)",
            "Use confidence weights honestly (0.0-1.0)",
        ],
        "recommended_domains": RECOMMENDED_DOMAINS,
        "confidence_guide": CONFIDENCE_GUIDE,
        "next_step": "Call memory_user_guide() for full documentation",
    }


if __name__ == "__main__":
    mcp.run()
