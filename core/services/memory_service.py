"""
Facade for MemoryGate core services.

Keeps the public API stable while implementations live in submodules.
"""

from core.services.memory_shared import (
    init_http_client,
    cleanup_http_client,
    embed_text,
    embed_text_sync,
    embedding_circuit_breaker,
)
from core.services.memory_embeddings import (
    _run_embedding_backfill,
    _embedding_backfill_loop,
)
from core.services.memory_search import (
    memory_search,
    search_cold_memory,
    memory_recall,
    memory_stats,
)
from core.services.memory_archive import (
    archive_memory,
    rehydrate_memory,
    list_archive_candidates,
    list_archived_memories,
    purge_memory_to_archive,
    restore_archived_memory,
)
from core.services.memory_storage import (
    memory_store,
    memory_init_session,
    memory_store_document,
)
from core.services.memory_concepts import (
    memory_store_concept,
    memory_get_concept,
    memory_add_concept_alias,
    memory_add_concept_relationship,
    memory_related_concepts,
)
from core.services.memory_patterns import (
    memory_update_pattern,
    memory_get_pattern,
    memory_patterns,
)
from core.services.memory_docs import (
    SPEC_VERSION,
    RECOMMENDED_DOMAINS,
    CONCEPT_TYPES,
    RELATIONSHIP_TYPES,
    CONFIDENCE_GUIDE,
    memory_user_guide,
    memory_bootstrap,
)
from core.services.memory_chains import (
    memory_create_chain,
    memory_get_chain,
    memory_add_to_chain,
    memory_remove_from_chain,
    memory_list_chains_for_memory,
    memory_list_chains_for_observation,
    memory_chain_create,
    memory_chain_append,
    memory_chain_get,
    memory_chain_list,
    memory_chain_update,
    memory_chain_entry_archive,
    ChainNotFoundError,
    ChainConflictError,
)
from core.services.memory_relationships import (
    memory_add_relationship,
    memory_list_relationships,
    memory_related,
    memory_get_supersession,
    relationship_add_residue,
    relationship_list_residue,
    MemoryRelationshipNotFoundError,
    parse_ref,
    format_ref,
)
from core.services.memory_references import (
    memory_get_by_ref,
    memory_get_many_by_refs,
)
from core.services.memory_stores import (
    stores_list_accessible,
    stores_get_active,
    stores_set_active,
    PERSONAL_STORE_ID,
)
from core.services.memory_anchors import (
    agent_anchor_set,
    agent_anchor_get,
    ANCHOR_SCOPE_TYPE_AGENT,
    ANCHOR_KIND_AGENT_PROFILE,
)
from core.services.mcp_diagnostics import (
    tool_inventory_status,
    capabilities_get,
    health_status,
)
from core.services.memory_quota import (
    quota_status,
    recompute_usage_for_tenant,
)
