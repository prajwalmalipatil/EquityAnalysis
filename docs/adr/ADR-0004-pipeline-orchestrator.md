# ADR-0004: Pipeline Orchestrator

## Context
Historically, the GitHub Actions YAML file (`equity_pipeline.yml`) executed individual Python scripts sequentially. As the pipeline grew complex (Data Quality -> VSA -> ETE -> Macro -> Publish), managing retries, dependency graphs, and telemetry within bash scripts became unmanageable and brittle.

## Decision
Implement a central `PipelineOrchestrator` (`main_orchestrator.py`) acting as an explicit Directed Acyclic Graph (DAG) executor natively in Python. The orchestrator now controls module imports, sequencing, and telemetry generation.

## Consequences
**Benefits**:
- "No module calls another module." Strict separation of concerns.
- Granular telemetry and observability (duration, pass rates).
- Centralized error handling and fast-failing.

**Trade-offs**:
- Concentrates execution control into a single monolithic script.

## Alternatives Considered
- **Apache Airflow / Prefect**: Rejected due to the massive operational overhead of running a dedicated orchestrator server. The Python DAG script perfectly fits the GitHub Actions serverless constraint.
