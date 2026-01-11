# MCP Tools

MemoryGate exposes 20 MCP tools via FastMCP. All tools return JSON payloads with
status fields and relevant data.

## Session and Self-Documentation

- memory_init_session: Register a conversation session.
  - Required: conversation_id, title, ai_name, ai_platform
  - Optional: source_url
- memory_bootstrap: Return connection status and first-step guidance.
  - Optional: ai_name, ai_platform
- memory_user_guide: Return the self-documentation guide.
  - Params: format (markdown|json), verbosity (short|verbose)

## Storage

- memory_store: Store an observation.
  - Required: observation
  - Optional: confidence, domain, evidence, ai_name, ai_platform, conversation_id, conversation_title
- memory_store_document: Store a document reference and summary.
  - Required: title, doc_type, url, content_summary
  - Optional: key_concepts, publication_date, metadata
- memory_store_concept: Store a concept node.
  - Required: name, concept_type, description
  - Optional: domain, status, metadata, ai_name, ai_platform
- memory_update_pattern: Create or update a pattern (upsert).
  - Required: category, pattern_name, pattern_text
  - Optional: confidence, evidence_observation_ids, ai_name, ai_platform, conversation_id

## Retrieval

- memory_search: Semantic search across observations, patterns, concepts, documents.
  - Required: query
  - Optional: limit, min_confidence, domain, include_cold
- memory_recall: Filtered recall of observations.
  - Optional: domain, min_confidence, limit, ai_name, include_cold
- memory_get_concept: Resolve concept by name or alias.
  - Required: name
  - Optional: include_cold
- memory_get_pattern: Fetch a specific pattern.
  - Required: category, pattern_name
  - Optional: include_cold
- memory_patterns: List patterns by category/confidence.
  - Optional: category, min_confidence, limit, include_cold
- memory_stats: Return counts and system stats.

## Knowledge Graph

- memory_add_concept_alias: Add an alias to an existing concept.
  - Required: concept_name, alias
- memory_add_concept_relationship: Create or update a relationship.
  - Required: from_concept, to_concept, rel_type
  - Optional: weight, description
- memory_related_concepts: List related concepts.
  - Required: concept_name
  - Optional: rel_type, min_weight, include_cold

## Cold Storage and Retention

- search_cold_memory: Explicitly search cold-tier records.
  - Required: query
  - Optional: top_k, min_score, max_score, type_filter, source, date_from, date_to,
    tags, include_evidence, bump_score
- archive_memory: Archive hot records to cold tier or summarize.
  - Optional: memory_ids, summary_ids, cluster_ids, threshold, mode, reason, actor,
    dry_run, limit
- rehydrate_memory: Move cold records back to hot tier.
  - Optional: memory_ids, summary_ids, cluster_ids, threshold, query, reason,
    actor, dry_run, limit, bump_score
- list_archive_candidates: List hot records below a score threshold.
  - Optional: below_score, limit

## Tool Inventory Resource

The MCP server also publishes a discovery resource:

- Resource URI: memorygate://tool-inventory
- Purpose: list registered tools and retry hints when the inventory is empty
