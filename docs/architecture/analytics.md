# Analytics Architecture

The Analytics system mirrors the robust CQRS Publish Pipeline. All metrics and aggregations are computed backend-first, generating static artifacts for the frontend to render.

## Backend-First Pipeline

```
AnalyticsProvider
        ↓
AnalyticsReadModel
        ↓
AnalyticsBuilder
        ↓
analytics.json
        ↓
AnalyticsWorkspace
```

## Metric Categories

To avoid mixing operational health with business intelligence, metrics are strictly partitioned into three categories within the `AnalyticsProvider`:

### 1. Business Metrics
- Events published per day/week/month.
- Distribution of events by category (e.g., Press Releases vs Monetary Policy).
- High-priority circulars and upcoming effective dates.

### 2. AI Metrics
- Confidence distribution and scoring.
- Theme extraction frequency (e.g., most common policy themes).
- AI processing success rates and failures.
- LLM response latency.

### 3. Operational Metrics
- Collector success rate and failure occurrences.
- Duplicate detection rates.
- Publish pipeline duration.
- Attachment download/parsing success rates.
- Feed endpoint availability.

## Frontend Consumption
The UI fetches `analytics.json` and populates the `workspaceState` or charting libraries (e.g., Chart.js). The UI does not compute moving averages or data groupings—it solely manages presentation.
