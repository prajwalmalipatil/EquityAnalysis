# ADR-001: Repository Split

## Context
As the Macro Intelligence Service evolved, the central `EventRepositoryInterface` was tasked with handling both ingestion (saving events) and querying (retrieving active events, deduplication checks). This created a tight coupling between the data collection pipeline and the publishing/UI pipeline, risking accidental mutations during read operations and making future migrations (e.g., JSONL to SQLite or PostgreSQL) extremely complex.

## Decision
We are explicitly splitting `EventRepositoryInterface` into two segregated interfaces:
1. `EventWriteRepository`: Strictly responsible for insertions, deduplication checks during collection, and state updates.
2. `EventReadRepository`: Strictly responsible for retrieving, filtering, and projecting historical events for the UI and Analytics builders.

## Alternatives Considered
- **Single Interface with Read-Only Methods**: Rejected because it relies on developer discipline rather than compiler/type-checker enforcement.
- **ORM / Database Layering**: Rejected for now, as JSONL is sufficient for the current scale, but the split prepares us for this seamlessly.

## Consequences
- **Positive**: Complete isolation of the ingestion and publishing concerns. Contract testing is much simpler.
- **Negative**: Slight duplication in dependency injection configuration.
