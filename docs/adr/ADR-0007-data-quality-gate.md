# ADR-0007: Data Quality Gate

## Context
Raw financial data (OHLCV) downloaded from public APIs frequently contained missing candles, duplicate dates, 0 volume rows, or anomalous negative prices. When these artifacts reached the VSA or ETE engines, they caused silent calculation errors, compromising the integrity of the research.

## Decision
Introduce an aggressive `DataQualityService` as the absolute first stage of the DAG. It evaluates incoming CSVs and explicitly categorizes them into outcome states: `PASS`, `WARN`, `QUARANTINE`, or `REJECT`. Invalid files are completely removed from the analytical path.

## Consequences
**Benefits**:
- Prevents "garbage in, garbage out" scenarios.
- Allows downstream engines (VSA, ETE) to safely assume the mathematical validity of their inputs, removing the need for defensive boundary checks scattered throughout the codebase.

**Trade-offs**:
- Strict quarantining means some ticker symbols may temporarily disappear from the dashboard until their upstream data source is repaired.

## Alternatives Considered
- **Inline Validation**: Having VSA/ETE validate data before processing. Rejected because it duplicates validation logic across multiple domains and pollutes business logic with infrastructural concerns.
