# Runtime Pipeline

The Runtime Pipeline is the **Write** side of our CQRS architecture. It is responsible for the ingestion, validation, persistence, and enrichment of raw data.

## Responsibilities

1. **Collection**: Automated scraping or polling of official sources (e.g., RBI website) via robust, circuit-breaker-protected clients (`RBICollector`).
2. **Validation**: Enforcement of core domain models and validation rules (e.g., duplicate detection, schema validation) before any data touches the disk.
3. **Write Persistence**: Utilizing the `EventWriteRepository` to store the authoritative, raw representation of an event.
4. **Enrichment**: Passing the raw event to the AI Agent (`MacroIntelligenceAgent`) to generate `derived_data` (summaries, themes). This data is strictly separated from the `official_data` within the schema to guarantee data integrity.

## Resiliency and Idempotency
- **Fail-Fast**: Network issues during collection do not impact the existing repository state or downstream pipelines.
- **Idempotency**: Retrying a collection job multiple times will cleanly handle duplicates using hashing on URLs/IDs, ensuring safe re-runs.

## Data Model Guarantees
- The `official_data` is immutable.
- The `derived_data` can be safely regenerated if the AI model or prompt is updated.
- The `metadata` strictly tracks execution metrics like lifecycle status and processing latencies.
