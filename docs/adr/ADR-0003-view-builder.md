# ADR-0003: View Builder

## Context
As the number of analytical engines increased, the UI code (`app.js`) became tightly coupled to the internal data structures of the VSA and ETE engines. This created fragility where a change in an engine's output format would break the frontend. Furthermore, GitHub Pages caches static files aggressively, leading to stale dashboards.

## Decision
Introduce a `ViewBuilderService` that acts as the dedicated Transformation Layer. It sits between the Research Repository and the static UI.

## Consequences
**Benefits**:
- Decouples the UI from core research engines.
- Enables content-hashed filenames (e.g. `summary.[hash].json`) to defeat aggressive CDN caching.
- Enforces Data Contract validation before the UI is updated.
- Facilitates the Atomic Swap (`os.rename()`), preventing live users from experiencing a broken half-state.

**Trade-offs**:
- Adds an additional processing step and slightly increases total pipeline duration.

## Alternatives Considered
- **Client-Side Processing**: Sending raw `.jsonl` event logs to the browser and having JS rebuild the state. Rejected due to immense payload sizes and slow page load times.
