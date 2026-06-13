# ADR-0001: Event Sourcing for Research Data

## Context
The platform originally stored analytical outputs by overwriting previous state files (e.g., `data.json`). This resulted in an inability to audit how an analytical conclusion was reached over time, eliminated historical replayability, and made the state fragile to pipeline failures.

## Decision
Adopt an Event-Sourced architecture. All research engines (VSA, ETE, Macro) must emit immutable events to append-only JSON-Lines (`.jsonl`) ledgers. The current state is strictly derived by replaying the event log from genesis.

## Consequences
**Benefits**:
- Absolute auditability of all research outcomes.
- Native time-travel support (Replay Engine).
- Decouples research generation from UI presentation.

**Trade-offs**:
- Increased storage footprint over time.
- Requires snapshotting to maintain reconstruction performance as event logs grow.

## Alternatives Considered
- **SQL Database (PostgreSQL)**: Rejected because it introduces external infrastructure dependencies, whereas flat `.jsonl` files can be stored natively in the Git repository and processed in GitHub Actions seamlessly.
