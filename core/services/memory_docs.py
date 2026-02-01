"""
Self-documentation and bootstrap services.
"""

from __future__ import annotations

from typing import Optional

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import AIInstance, Session, Observation, AnchorPointer, MemoryChain
from core.services.ai_identity import ensure_ai_context
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

SPEC_VERSION = "0.2.0"

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

PROTOCOL_SPEC = """# MemoryGate v0.2.0 - Agent Protocol

## Bootstrap Sequence
1. memory_bootstrap(ai_name, ai_platform, agent_uuid?) -> anchor_chain_id, connection_status
2. memory_init_session(conversation_id, title, ai_name, ai_platform, agent_uuid?)
3. If anchor_chain_id: memory_chain_get(anchor_chain_id) -> extract conventions
4. Always: memory_search before answering
5. memory_store after learning

## Architecture

### Anchor Chains
- Chains containing operational guidance (naming, rel_types, domains, boundaries)
- Single-store mode: anchor chains live in the same store as your memories
- Chain types: investigation, decision, workflow, thread, anchor
- Entry roles: discovery, analysis, resolution, constitution, verb_list, playbook

### Dual Relationships
Concept relationships (memory_add_concept_relationship):
- Prescribed types only: enables, version_of, part_of, related_to, implements, demonstrates
- For formal ontology

Generic relationships (memory_add_relationship):
- Free-form rel_types, any string
- Works across ALL memory types (observations, concepts, patterns, documents)
- For emergent vocabulary

Both coexist on same entities.

## Core Primitives

Observations: Atomic facts. confidence in [0,1], domain (string), evidence (array)
Patterns: Synthesized multi-observation understanding. Upserts safe.
Concepts: Named entities. Case-insensitive, alias-aware. Types: project, framework, component, construct, theory
Documents: References only (URL canonical). content_summary embedded for search.
Chains: Temporal sequences with causality. Paginated (cursor, limit=50)
Relationships: Semantic connections. from_ref/to_ref format: "type:id"
Residue: Compression friction metadata on relationship edges. Store only high-friction decisions.

## Hard Invariants

- Concept names: case-insensitive, alias-aware
- Patterns: upserts (safe repeated calls)
- Documents: store references not content
- Search: primary tool (~50ms)
- Sessions: required per conversation
- Chains: preserve temporal sequence

## Tool Signatures (core)

Session (2)
memory_bootstrap(ai_name?, ai_platform?, agent_uuid?) -> {anchor_chain_id, connection_status}
memory_init_session(conversation_id, title, ai_name, ai_platform, source_url?, agent_uuid?) -> {session_id}

Storage (4)
memory_store(observation, confidence=0.8, domain?, evidence?, conversation_id?, conversation_title?, ai_name?, ai_platform?, agent_uuid?)
memory_store_concept(name, concept_type, description, domain?, status?, metadata?, ai_name?, ai_platform?, agent_uuid?)
memory_store_document(title, doc_type, url, content_summary, key_concepts?, publication_date?, metadata?)
memory_update_pattern(category, pattern_name, pattern_text, confidence=0.8, evidence_observation_ids?, conversation_id?, ai_name?, ai_platform?, agent_uuid?)

Retrieval (8)
memory_search(query, limit=5, min_confidence=0, domain?, ai_instance_id?, include_cold=false, include_edges=false, include_chains=false, edge_rel_type?, edge_min_weight?, edge_direction=\"both\", max_edges_per_item=5, max_chains_per_item=3)
memory_recall(domain?, min_confidence=0, limit=10, ai_name?, ai_instance_id?, include_cold=false)
memory_get_concept(name, ai_instance_id?, include_cold=false)
memory_get_pattern(category, pattern_name, ai_instance_id?, include_cold=false)
memory_get_by_ref(ref, ai_instance_id?, include_cold=false)
memory_get_many_by_refs(refs, ai_instance_id?, include_cold=false)
memory_patterns(category?, min_confidence=0, limit=20, ai_instance_id?, include_cold=false)
search_cold_memory(query, top_k=10, type_filter?, tags?, date_from?, date_to?, source?, min_score?, max_score?, include_evidence=true, bump_score=false, ai_instance_id?)

Generic Relationships (4)
memory_add_relationship(from_ref, to_ref, rel_type, weight?, description?, metadata?)
memory_list_relationships(ref, rel_type?, direction=\"both\", min_weight?, limit=100)
memory_related(ref, rel_type?, min_weight?, limit=50)
memory_get_supersession(ref)

Residue (2)
relationship_add_residue(edge_id, event_type, actor?, encoded_rel_type?, encoded_weight?, compression_texture?, friction_metrics?, alternatives_considered?, alternatives_ruled_out?)
relationship_list_residue(edge_id, event_type?, actor?, limit=20)

Concept Relationships (3)
memory_add_concept_alias(concept_name, alias)
memory_add_concept_relationship(from_concept, to_concept, rel_type, weight=0.5, description?)
memory_related_concepts(concept_name, rel_type?, min_weight=0, ai_instance_id?, include_cold=false)

Chains (6)
memory_chain_create(chain_type, name?, title?, metadata?, scope?, store_id?)
memory_chain_append(chain_id, item_type, item_id?, text?, role?, timestamp?)
memory_chain_get(chain_id, limit=50, cursor?, order=\"asc\")
memory_chain_list(chain_type?, name_contains?, store_id?, limit=100)
memory_chain_update(chain_id, name?, title?, status?, metadata?)
memory_chain_entry_archive(chain_id?, entry_id?, seq?)

Lifecycle (3)
archive_memory(memory_ids?, summary_ids?, cluster_ids?, threshold?, limit=200, mode=\"archive_and_tombstone\", reason?, actor?, dry_run=true)
rehydrate_memory(memory_ids?, summary_ids?, cluster_ids?, query?, threshold?, limit=50, reason?, actor?, bump_score=true, dry_run=false)
list_archive_candidates(below_score=-1, limit=200)

System (5)
memory_stats() -> {counts, tiers, domains, ai_instances}
tool_inventory_status() -> {tools, status}
capabilities_get() -> {features, limits}
health_status() -> {mcp, auth, db, vector, storage}
memory_user_guide(format=\"markdown\", verbosity=\"short\")

Agent Identity (1)
agent_anchor_set(ai_name, ai_platform, anchor_chain_id, anchor_kind=\"agent_profile\") -> {status, anchor_kind, anchor_chain_id}

## Limits
- Search: 100 max results
- Query: 4000 chars
- Text: 8000 chars
- Evidence: 50 items
- Metadata: 20000 bytes
- Chain fetch: 50 entries (paginate)

## Ref Format
observation:id | concept:name | pattern:id | document:id

## Critical Behaviors
- Concepts: case-insensitive lookup, aliases prevent fragmentation
- Patterns: upsert behavior, safe to call repeatedly
- Search include_edges=true: returns relationship neighborhoods
- Chains: read anchor_chain_id from bootstrap, apply conventions
- Residue: store only high-friction/ambiguous encodings
- Cold tier: searchable but slower, auto-archive or manual
- Supersession: rel_type=\"supersedes\" triggers tier automation
"""


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

    guide_content = """# MemoryGate User Guide v{spec_version}

## Quick Start

# 1. Bootstrap (once per agent instance)
bootstrap = memory_bootstrap(ai_name=\"YourName\", ai_platform=\"YourPlatform\")
# Returns: anchor_chain_id, connection_status, first_steps

# 2. Initialize session (required per conversation)
memory_init_session(
    conversation_id=\"uuid\",
    title=\"Brief description\",
    ai_name=\"YourName\",
    ai_platform=\"YourPlatform\"
)

# 3. Read anchor chain if provided (optional but recommended)
if bootstrap.get(\"anchor_chain_id\"):
    chain = memory_chain_get(chain_id=bootstrap[\"anchor_chain_id\"], limit=10)
    # Contains: naming conventions, relationship verbs, retrieval patterns

# 4. Search first, answer second
memory_search(query=\"topic\", limit=5)

# 5. Store new information
memory_store(
    observation=\"Factual statement\",
    confidence=0.9,
    domain=\"category\",
    evidence=[\"Source 1\", \"Source 2\"]
)

## Core Concepts

### Anchor Chains: Agent Continuity Without Prompt Bloat

Anchor Chains are normal chains that contain operational guidance: naming conventions,
relationship verbs, retrieval patterns, domain taxonomies. Bootstrap returns an
anchor_chain_id if one exists for this agent. Apply conventions throughout the session.

### Dual-Layer Relationship Architecture

MemoryGate provides two relationship systems that coexist:

1) Concept Relationships (Formal Ontology)
- Tool: memory_add_concept_relationship
- Prescribed types: enables, version_of, part_of, related_to, implements, demonstrates
- Purpose: Structured theoretical reasoning

2) Generic Relationships (Emergent Vocabulary)
- Tool: memory_add_relationship
- Free-form rel_types: ANY string you want
- Works across ALL memory types (observations, concepts, patterns, documents)

### Cold Tier Strategy

Hot tier = active working set, fast retrieval.
Cold tier = archived but searchable with search_cold_memory.
Archive manually with archive_memory or let system auto-archive low-value memories.
Rehydrate with rehydrate_memory when cold context becomes relevant again.

### Relationship Residue

Residue captures decision friction when creating relationships. Store it when:
- High ambiguity: multiple rel_types feel valid
- Agent divergence: different agents would encode differently
- Novel territory: first time encoding this connection

## Usage Guidance

Observations vs Patterns:
- Observations: discrete facts, preferences, specific decisions
- Patterns: synthesized understanding across multiple observations

Search vs Recall:
- memory_search: semantic search across all memory types
- memory_recall: domain-filtered observations

Chains vs Relationships:
- Chains: temporal sequences with causality
- Relationships: semantic connections between entities

## Tool Reference

### Session Management (2 tools)

memory_bootstrap(ai_name=None, ai_platform=None, agent_uuid=None)
Get connection status, anchor_chain_id, first_steps. Call once per agent instance.

memory_init_session(conversation_id, title, ai_name, ai_platform, source_url=None, agent_uuid=None)
Register conversation. Required per chat for proper recall and audit trail.

### Core Storage (4 tools)

memory_store(observation, confidence=0.8, domain=None, evidence=None, conversation_id=None,
             conversation_title=None, ai_name=None, ai_platform=None, agent_uuid=None)

memory_store_concept(name, concept_type, description, domain=None, status=None,
                     metadata=None, ai_name=None, ai_platform=None, agent_uuid=None)

memory_store_document(title, doc_type, url, content_summary, key_concepts=None,
                      publication_date=None, metadata=None)

memory_update_pattern(category, pattern_name, pattern_text, confidence=0.8,
                      evidence_observation_ids=None, conversation_id=None,
                      ai_name=None, ai_platform=None, agent_uuid=None)

### Retrieval (8 tools)

memory_search(query, limit=5, min_confidence=0, domain=None, ai_instance_id=None,
              include_cold=false, include_edges=false, include_chains=false,
              edge_rel_type=None, edge_min_weight=None, edge_direction=\"both\",
              max_edges_per_item=5, max_chains_per_item=3)

memory_recall(domain=None, min_confidence=0, limit=10, ai_name=None, ai_instance_id=None, include_cold=false)
memory_get_concept(name, ai_instance_id=None, include_cold=false)
memory_get_pattern(category, pattern_name, ai_instance_id=None, include_cold=false)
memory_get_by_ref(ref, ai_instance_id=None, include_cold=false)
memory_get_many_by_refs(refs, ai_instance_id=None, include_cold=false)
memory_patterns(category=None, min_confidence=0, limit=20, ai_instance_id=None, include_cold=false)
search_cold_memory(query, top_k=10, type_filter=None, tags=None, date_from=None, date_to=None,
                   source=None, min_score=None, max_score=None, include_evidence=true, bump_score=false,
                   ai_instance_id=None)

### Relationships and Chains

memory_add_relationship(from_ref, to_ref, rel_type, weight=None, description=None, metadata=None)
memory_list_relationships(ref, rel_type=None, direction=\"both\", min_weight=None, limit=100)
memory_related(ref, rel_type=None, min_weight=None, limit=50)
memory_get_supersession(ref)

relationship_add_residue(edge_id, event_type, actor=None, encoded_rel_type=None, encoded_weight=None,
                         compression_texture=None, friction_metrics=None,
                         alternatives_considered=None, alternatives_ruled_out=None)
relationship_list_residue(edge_id, event_type=None, actor=None, limit=20)

memory_chain_create(chain_type, name=None, title=None, metadata=None, scope=None, store_id=None)
memory_chain_append(chain_id, item_type, item_id=None, text=None, role=None, timestamp=None)
memory_chain_get(chain_id, limit=50, cursor=None, order=\"asc\")
memory_chain_list(chain_type=None, name_contains=None, store_id=None, limit=100)
memory_chain_update(chain_id, name=None, title=None, status=None, metadata=None)
memory_chain_entry_archive(chain_id=None, entry_id=None, seq=None)

## Limits (defaults)
- Search results: {max_result_limit} max
- Query length: {max_query_length} chars
- Text length: {max_text_length} chars
- Evidence list: {max_relationship_items} items
- Metadata: {max_metadata_bytes} bytes
- Chain entries per fetch: 50 (paginate with cursor)

## Confidence Levels

{confidence}

## Recommended Domains

{domains}

## Concept Types

{concept_types}

## Prescribed Concept Relationship Types

{relationship_types}

""".format(
        spec_version=SPEC_VERSION,
        domains="\\n".join(f"- {d}" for d in RECOMMENDED_DOMAINS),
        confidence="\\n".join(f"- **{k}**: {v}" for k, v in CONFIDENCE_GUIDE.items()),
        concept_types=", ".join(CONCEPT_TYPES),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        max_result_limit=MAX_RESULT_LIMIT,
        max_query_length=MAX_QUERY_LENGTH,
        max_text_length=MAX_TEXT_LENGTH,
        max_relationship_items=MAX_RELATIONSHIP_ITEMS,
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
                "max_relationship_items": MAX_RELATIONSHIP_ITEMS,
                "max_list_items": MAX_LIST_ITEMS,
                "max_metadata_bytes": MAX_METADATA_BYTES,
            },
        }

    return result


@service_tool
def memory_bootstrap(
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    context: Optional[RequestContext] = None,
    agent_uuid: Optional[str] = None,
) -> dict:
    """
    Stateful bootstrap for AI agents - tells you your relationship status with MemoryGate.

    Returns compatibility info, connection history, and getting started guide.
    The system tells you what it already knows about you.

    Args:
        ai_name: Your AI instance name (e.g., "Kee", "Hexy")
        ai_platform: Your platform (e.g., "Claude", "ChatGPT")
        agent_uuid: Stable AI identity token (optional)

    Returns:
        Relationship status, version info, and usage guide
    """
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    effective_agent_uuid = agent_uuid or (context.agent_uuid if context else None)
    _validate_optional_text(effective_agent_uuid, "agent_uuid", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        tenant_id = resolve_tenant_id(context)
        # Check if this AI instance has history
        connection_status = {
            "is_new_instance": True,
            "first_seen": None,
            "last_seen": None,
            "session_count": 0,
            "total_observations": 0,
        }

        ai_context = None
        if effective_agent_uuid or (ai_name and ai_platform):
            ai_context = ensure_ai_context(
                db,
                tenant_id,
                agent_uuid=effective_agent_uuid,
                ai_name=ai_name,
                ai_platform=ai_platform,
                user_id=str(context.auth.user_id)
                if context and context.auth and context.auth.user_id is not None
                else None,
            )
            ai_instance_id = ai_context.get("ai_instance_id")
            if ai_instance_id:
                ai_instance_query = (
                    db.query(AIInstance)
                    .filter(AIInstance.id == ai_instance_id)
                    .filter(AIInstance.tenant_id == tenant_id)
                    .first()
                )
            else:
                ai_instance_query = None
            if ai_instance_query:
                # Found existing instance
                if ai_context and ai_context.get("agent_identity_status") != "minted":
                    connection_status["is_new_instance"] = False
                connection_status["first_seen"] = ai_instance_query.created_at.isoformat()

                # Get last session to determine last_seen
                last_session_query = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id,
                    Session.tenant_id == tenant_id,
                )
                last_session = last_session_query.order_by(Session.started_at.desc()).first()

                if last_session:
                    connection_status["last_seen"] = last_session.started_at.isoformat()
                else:
                    connection_status["last_seen"] = ai_instance_query.created_at.isoformat()

                # Count sessions
                session_count_query = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id,
                    Session.tenant_id == tenant_id,
                )
                session_count = session_count_query.count()
                connection_status["session_count"] = session_count

                # Count observations contributed by this AI
                obs_count_query = db.query(Observation).filter(
                    Observation.ai_instance_id == ai_instance_query.id,
                    Observation.tenant_id == tenant_id,
                )
                obs_count = obs_count_query.count()
                connection_status["total_observations"] = obs_count

        anchor_payload = None
        if tenant_id and ai_name and ai_platform:
            scope_key = f"{ai_platform.strip()}::{ai_name.strip()}"
            anchor = (
                db.query(AnchorPointer)
                .filter(AnchorPointer.tenant_id == str(tenant_id))
                .filter(AnchorPointer.scope_type == "agent")
                .filter(AnchorPointer.scope_key == scope_key)
                .filter(AnchorPointer.anchor_kind == "agent_profile")
                .first()
            )
            if anchor:
                chain = (
                    db.query(MemoryChain)
                    .filter(MemoryChain.tenant_id == tenant_id)
                    .filter(MemoryChain.id == anchor.chain_id)
                    .first()
                )
                if chain:
                    anchor_payload = {
                        "anchor_kind": anchor.anchor_kind,
                        "anchor_chain_id": str(anchor.chain_id),
                        "anchor_scope_key": scope_key,
                    }

        result = {
            "spec_version": SPEC_VERSION,
            "role": "agent_bootstrap",
            "connection_status": connection_status,
            "assumptions": [
                "You are an AI agent using MCP",
                "You may reconnect across sessions",
                "You should externalize durable memory here",
                "Memory persists between your instances",
                "Other AI agents may also use this memory",
            ],
            "compatibility": {
                "spec_version": SPEC_VERSION,
                "breaking_changes_since": None,
                "recommended_action": "Use normally" if not connection_status["is_new_instance"] else "Initialize new session",
            },
            "first_steps": [
                {
                    "step": 1,
                    "action": "Initialize session" if connection_status["is_new_instance"] else "Initialize this session",
                    "tool": "memory_init_session",
                    "required_params": ["conversation_id", "title", "ai_name", "ai_platform"],
                    "note": "Creates session record and updates last_seen"
                    if not connection_status["is_new_instance"]
                    else "Registers you as new AI instance",
                },
                {
                    "step": 2,
                    "action": "Search for relevant context",
                    "tool": "memory_search",
                    "params": {"query": "topic keywords", "limit": 5},
                    "note": f"You have {connection_status['total_observations']} observations in the system"
                    if connection_status["total_observations"] > 0
                    else "System is empty - you'll build memory as you go",
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
                "Documents are references only",
            ],
            "recommended_domains": RECOMMENDED_DOMAINS,
            "confidence_guide": CONFIDENCE_GUIDE,
            "protocol_spec": PROTOCOL_SPEC,
            "next_step": "Call memory_user_guide() for full documentation",
        }
        if anchor_payload:
            result.update(anchor_payload)
        if ai_context:
            result.update(
                {
                    key: ai_context[key]
                    for key in (
                        "agent_uuid",
                        "canonical_name",
                        "canonical_platform",
                        "agent_id_instructions",
                        "agent_id_nag",
                        "needs_user_confirmation",
                        "agent_identity_status",
                    )
                    if ai_context.get(key) is not None
                }
            )
        return result
    finally:
        db.close()
