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
