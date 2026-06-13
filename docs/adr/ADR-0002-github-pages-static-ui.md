# ADR-0002: GitHub Pages Static UI

## Context
The UI required a secure, zero-maintenance hosting solution that could handle potentially infinite global traffic without backend scaling concerns. Since research calculations run periodically rather than interactively, dynamic backend infrastructure is unnecessary.

## Decision
Host the presentation layer strictly as a static Single Page Application (SPA) on GitHub Pages, served directly from the `dashboard/` directory.

## Consequences
**Benefits**:
- Free, highly available, CDN-backed hosting.
- Zero server maintenance, zero database latency.
- Extreme security (no attack surface on the backend).

**Trade-offs**:
- Requires the entire analytical state to be pre-calculated during the CI/CD pipeline (GitHub Actions).
- No real-time interactivity; users must wait for the next cron job for data updates.

## Alternatives Considered
- **FastAPI / React Backend**: Rejected due to hosting costs, database management overhead, and unnecessary complexity for a daily/weekly research cadence.
