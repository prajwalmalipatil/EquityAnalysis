# ADR-003: Runtime vs Publish Pipeline

## Context
Previously, the `JSONPublisher` was part of the main MacroPipeline loop. If the publisher failed (e.g., due to a schema change), the entire ingestion run would fail, or vice versa. The system conflated the act of *acquiring intelligence* with the act of *presenting intelligence*.

## Decision
We are explicitly splitting the platform into two independent pipelines:
1. **Pipeline A (Data Pipeline / Runtime)**: Collect -> Validate -> Persist -> Enrich. This runs on a scheduled cron job and is entirely focused on data acquisition.
2. **Pipeline B (Publish Pipeline / Build Time)**: Query -> Build -> Validate -> Publish. This runs strictly after a successful ingestion run, focused entirely on generating static web artifacts.

## Alternatives Considered
- **Monolithic Pipeline**: Rejected because it violates the Single Responsibility Principle at the orchestration level.
- **Dynamic API Server**: Rejected. We deliberately chose static site generation for the UI to eliminate server maintenance and hosting costs, meaning publishing must remain a discrete build step.

## Consequences
- **Positive**: We can retry publishing without re-triggering expensive LLM enrichment. We can run ingestion silently without triggering UI updates until a batch is fully validated.
- **Negative**: Requires slightly more complex orchestration to ensure Pipeline B only runs when Pipeline A succeeds.
