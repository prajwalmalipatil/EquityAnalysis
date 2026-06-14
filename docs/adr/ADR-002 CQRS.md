# ADR-002: CQRS Pattern

## Context
Our read operations (filtering by date, searching, generating analytics) and write operations (collecting, deduplicating, updating AI enrichment) have fundamentally different performance and scaling requirements. The system was originally using a single CRUD-like repository approach. 

## Decision
We are adopting a Command Query Responsibility Segregation (CQRS) architectural pattern. 
- **Commands**: Executed exclusively by the Data Pipeline (Runtime). Commands enforce strict validation, duplicate checking, and state mutation.
- **Queries**: Executed exclusively by the Publish Pipeline (Build Time). Queries operate on read-optimized views (`EventReadRepository` -> `MacroQueryService`) and never mutate the underlying JSONL store.

## Alternatives Considered
- **Standard CRUD**: Rejected because the complex nested structure of `MacroEvent` makes complex querying and projection very slow. We need the freedom to build specialized Read Models without polluting the Write Domain.

## Consequences
- **Positive**: Queries can be optimized independently. We can safely introduce caching or in-memory projections for the Publish Pipeline without risking ingestion integrity.
- **Negative**: Increased class count (`WriteRepository`, `ReadRepository`, `QueryService`).
