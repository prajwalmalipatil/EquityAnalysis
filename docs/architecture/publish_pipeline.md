# Publish Pipeline

The Publish Pipeline is the **Read** side of our CQRS architecture. It isolates the generation of static artifacts from the complex logic of data ingestion.

## Workflow

1. **Read Repository**: `EventReadRepository` loads the persisted events from storage. It contains purely read-only methods.
2. **Query Service**: `MacroQueryService` handles the business logic of filtering, sorting, and projecting domain entities into Read Models (e.g., `DashboardViewModel`).
3. **Builders**: `DashboardBuilder` consumes the read models to generate the final, structured payloads required by the frontend application.
4. **Validation**: `ReleaseValidator` acts as a hard release gate. If the generated artifact is missing required keys, schema versions, or is bloated (e.g., >5MB), the publish is blocked.
5. **Publishers**: Pure I/O classes like `JSONPublisher` that write the validated payloads to disk (`data.json`, `manifest.json`) for static hosting.

## Key Principles
- **No Domain Logic**: The publishing layer does not modify or enrich domain data.
- **Contract Enforcement**: The output of the Publish Pipeline is guaranteed to match the exact schema expected by the static Frontend Workspace.
- **Static Hosting Optimized**: Outputs are flattened, statically generated JSON files perfectly optimized for GitHub Pages or AWS S3.
