# ADR-0008: Research Provenance

## Context
As research events accumulated in the event store, it became impossible to determine *which version* of the codebase generated a specific event months ago. If a bug was discovered in a specific pipeline version, identifying the corrupted historical events was unfeasible. Furthermore, non-deterministic `datetime.now()` and UUIDs prevented verifiable testing.

## Decision
Mandate absolute Research Provenance. Replace all temporal entropy with deterministic SHA-256 hashes derived from OHLCV candle dates. Additionally, expand the `ResearchEvent` model to inherently embed the `pipeline_version`, the `commit_sha`, and a `dataset_checksum`.

## Consequences
**Benefits**:
- 100% Backtest Reproducibility. Processing the same CSV twice yields identical byte-for-byte hashes.
- Auditable lineage; every signal can be traced back to the specific Git commit and dataset that spawned it.

**Trade-offs**:
- ID generation requires more computation (SHA-256 hashing) than simple UUID generation.

## Alternatives Considered
- **UUIDv5 (Namespace generation)**: Considered instead of SHA-256. Rejected because it still obscures the specific input string format, whereas a straightforward SHA-256 hash over a concatenated seed string is universally verifiable across languages.
