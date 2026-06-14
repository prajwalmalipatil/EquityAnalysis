# ADR-004: Dashboard ViewModel

## Context
The frontend UI (`app.js`) was historically reading the raw `MacroEvent` JSON serialization directly. This meant the UI had to traverse `official_data`, `derived_data.impact`, and `metadata` to find what it needed. Any change to the backend domain model broke the UI.

## Decision
We are introducing a formal `MacroReadModel` and `DashboardViewModel`. The Publish Pipeline maps the complex backend Domain Object (`MacroEvent`) into a flattened, UI-optimized `DashboardViewModel` dictionary. The UI only ever consumes this flattened contract.

## Alternatives Considered
- **Direct Domain Exposure**: Rejected. Exposing Domain objects directly to presentation layers is a well-known anti-pattern that leads to brittle frontends.
- **GraphQL**: Overkill for our static deployment model.

## Consequences
- **Positive**: Total decoupling of the UI from the Domain. We can completely rewrite the backend data structures without touching a single line of JavaScript, as long as the `DashboardMapper` fulfills the ViewModel contract.
- **Negative**: Adds a mapping layer that must be maintained whenever new fields are requested by the UI.
