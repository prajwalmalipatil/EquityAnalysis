# ADR-0006: Platform Registry

## Context
Initially, the pipeline orchestrator had hardcoded knowledge of every analytical module (VSA, ETE, Macro). As new modules were planned (Risk, Portfolio Construction), updating the central orchestrator and manually tracking dependencies became an unsustainable bottleneck.

## Decision
Create an authoritative `PlatformRegistry` using a strict Pydantic `ResearchModule` schema. Modules self-register upon import, declaring their version, inputs, outputs, dependencies, capabilities, and maturity levels. The Orchestrator validates this registry before DAG execution.

## Consequences
**Benefits**:
- True Inversion of Control.
- The pipeline fails safely and predictably if a dependency is missing.
- Exposes a dynamic, self-documenting `research_registry.json` payload for the UI.

**Trade-offs**:
- Requires discipline; developers must ensure module registration code is executed prior to validation (e.g., via orchestrator imports).

## Alternatives Considered
- **YAML Configuration Files**: Using a static `modules.yml` file to declare capabilities. Rejected because it risks falling out of sync with the actual Python codebase. Self-registration in code ensures the registry perfectly reflects the deployed runtime.
