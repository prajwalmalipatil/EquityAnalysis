# ADR-0005: Replay Engine

## Context
Analysts needed the ability to observe the exact state of the dashboard at a specific historical date (e.g., "What did the EigenTransitionEngine output on Nov 15, 2023?") without polluting the live environment or writing custom ad-hoc query scripts.

## Decision
Introduce an out-of-band `ReplayEngine` (`main_replay.py`) that generates a shadow directory of the `events.jsonl` files by filtering out any events with a timestamp strictly greater than the `target_date`. It then instructs the `ViewBuilderService` to reconstruct the dashboard state from this truncated shadow repository.

## Consequences
**Benefits**:
- Zero changes required to the core research engines.
- True "Time Machine" capability; guarantees perfectly accurate historical dashboard states.

**Trade-offs**:
- Replaying extremely old dates requires parsing massive JSONL files only to discard the majority of the rows, incurring a slight I/O overhead.

## Alternatives Considered
- **In-Memory Filtering within the Engine**: Modifying `reconstruct_state()` to accept a `target_date` argument. Rejected because it violated the Single Responsibility Principle and risked introducing temporal bugs into the live production pipeline.
