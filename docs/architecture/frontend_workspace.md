# Frontend Workspace

The Frontend Workspace acts purely as a presentation layer. It contains no business logic or database coupling.

## State-Driven Architecture
The UI is driven entirely by a single source of truth:
```javascript
const workspaceState = {
    selectedEventId: null,
    selectedCategory: '',
    searchQuery: '',
    sortOrder: 'desc'
};
```
All UI controls (search bars, dropdowns, keyboard events) mutate this state object. The `renderMacroTimeline()` and `renderMacroWorkspace()` functions read exclusively from this state to generate the DOM.

## Component Independence
The workspace avoids monolithic render functions. It decomposes the interface into isolated functions, allowing independent testing and evolution:
- `renderHeaderWidget()`
- `renderOverviewWidget()`
- `renderAIInsightsWidget()`
- `renderAttachmentsWidget()`
- `renderMetadataWidget()`
- `renderRelatedEventsWidget()`

## Master-Detail Layout
- **Left Pane (Master)**: A timeline of events fetched statically from the published `data.json`.
- **Right Pane (Detail)**: A dynamic workspace that populates the isolated widgets when an event is selected.

## Production UX
- **Keyboard Navigation**: Native `ArrowUp`, `ArrowDown`, and `Enter` support.
- **Auto-Scrolling**: `scrollIntoView` keeps the selected card in focus.
- **Accessibility**: ARIA roles (`role="list"`, `role="listitem"`, `aria-selected`) to support enterprise compliance.
