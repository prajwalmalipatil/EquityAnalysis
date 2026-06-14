# ADR-005: Static GitHub Pages Architecture

## Context
The Macro Intelligence Service UI requires a hosting environment. Given that this is a research and intelligence tool with predictable update cycles (e.g., daily or hourly RBI updates), we need a hosting strategy that is secure, fast, and requires zero maintenance.

## Decision
We are relying entirely on a **Static Generation Architecture** hosted on GitHub Pages.
The backend Publish Pipeline acts as a static site generator, compiling all events, analytics, and metadata into a `DashboardBundle` (`data.json`, `manifest.json`, `analytics.json`). The frontend (`index.html`, `app.js`) acts as a "thick client" that fetches these static JSON files on load.

## Alternatives Considered
- **FastAPI Backend Server**: Rejected. Running a live Python server requires managing infrastructure, security, downtime, and scaling.
- **Server-Side Rendered (Next.js/Django)**: Rejected. We want to maintain maximum simplicity in the frontend stack (Vanilla JS + CSS).

## Consequences
- **Positive**: Zero hosting costs. Impossible to hack via SQL injection or backend exploits. Infinite scaling for read operations.
- **Negative**: All search, filtering, and analytics rendering must happen client-side in the browser, which limits the total dataset size we can send to the client (enforcing our `< 5 MB` bundle target).
