---
trigger: always_on
---

## 1. Architecture-Level Problem Solving

### 1.1 Solve the Root, Not the Symptom
Before implementing, determine if the issue stems from poor data structure design or weak component relationships. Ask: Can this problem be solved by restructuring data? Are components doing things that belong together? Is there a missing abstraction layer?

### 1.2 Separation of Concerns
Three-layer model: **UI/Presentation** → **Domain/Business Logic** → **Infrastructure**

Each layer has specific responsibilities; details from lower layers must never leak upward. Business logic never imports database, HTTP, or file I/O modules. All external dependencies are injected via constructor.

### 1.3 Scalability & Performance
Always evaluate algorithmic complexity. Use HashMap for frequent lookups (O(1)), Arrays for sequential access, LinkedList for frequent insertions. Estimate data volume and ensure worst-case Big O < 1 second execution.

### 1.4 Decoupling
Design components to be independently testable and replaceable. Use abstraction (interfaces), not concrete implementations. New features should be **added** as new code, not **modifying** existing code (Open/Closed Principle).

---

## 2. SOLID Principles

### 2.1 Single Responsibility Principle (SRP)
A class should have only one reason to change. If you describe a class using "and," split it into multiple classes.

### 2.2 Open/Closed Principle (OCP)
Classes should be open for extension but closed for modification. Use inheritance, composition, and plugins to extend behavior.

### 2.3 Liskov Substitution Principle (LSP)
Derived classes must be substitutable for base classes without breaking behavior.

### 2.4 Interface Segregation Principle (ISP)
Many specific interfaces are better than one general-purpose interface. Don't force clients to depend on methods they don't use.

### 2.5 Dependency Inversion Principle (DIP)
Both high-level and low-level modules should depend on abstractions, not each other. Pass dependencies via constructor (Dependency Injection).

---

## 3. Best Practices

### 3.1 KISS (Keep It Simple, Stupid)
Favor simple, understandable solutions. If explaining your code takes more than 2 sentences, it's too complex.

### 3.2 DRY (Don't Repeat Yourself)
Apply the **Rule of Three**: 1st occurrence write normally, 2nd occurrence copy it, 3rd occurrence refactor into a shared function. Prevents premature abstraction.

### 3.3 YAGNI (You Ain't Gonna Need It)
Build only what is required now. Don't add "flexibility" or "what-if" features that don't exist yet.

### 3.4 Scout Rule
Always leave the codebase in a better state than you found it. Small improvements compound—fix typos, rename variables, extract duplicated methods.

---

## 4. Class & Method Design

### 4.1 The 30/300 Rule
- **Methods:** Rarely exceed 30 lines of code
- **Classes:** Rarely exceed 300 lines of code

Smaller units are easier to test, reuse, and reason about.

### 4.2 Composition Over Inheritance
Build complex behavior by combining small, focused objects instead of deep inheritance hierarchies. Inheritance creates rigid hierarchies; composition is flexible.

### 4.3 One Level of Abstraction
All logic within a method should operate at the same level of abstraction. Don't mix high-level business logic with low-level implementation details.

---

## 5. Error Handling

### 5.1 Fail Fast
Validate inputs at function entry. Don't let bad data travel through your system.

### 5.2 Null Safety
Never return `None` if you can return an empty collection or use Null Object pattern. Always return a valid collection.

### 5.3 Boundary Testing
Test edge cases: empty collections, None/Null, zero values, negative numbers, maximum values, duplicates, concurrent access.

### 5.4 Idempotency
Repeating an operation should have the same effect as performing it once. Critical for APIs, message processing, and payments.

---

## 6. Reducing Cognitive Complexity

### 6.1 Guard Clauses
Return early to handle edge cases, avoiding deep nesting. Keep the "happy path" obvious.

### 6.2 Indentation Limit
Never exceed 3 levels of indentation. Extract into helper methods if needed.

### 6.3 Intention-Revealing Names
Names must describe **intent**, not **implementation**. Boolean names should start with `is_`, `has_`, `should_`.

### 6.4 Avoid Magic Numbers
Replace all raw numbers with named constants using SCREAMING_SNAKE_CASE.

### 6.5 Pure Functions
Functions that take inputs and return outputs without side effects are easier to test, reason about, and parallelize.

---

## 7. Design Patterns

**Use patterns only when they reduce complexity, not to show off.**

### 7.1 Creational Patterns
- **Factory:** Decouple object creation from client code
- **Builder:** Complex initialization with many parameters

### 7.2 Structural Patterns
- **Decorator:** Dynamically add functionality without modifying structure
- **Adapter:** Make incompatible interfaces work together

### 7.3 Behavioral Patterns
- **Strategy:** Switch algorithms at runtime
- **Observer:** Multiple objects react to state changes without coupling
- **Repository:** Abstract data access logic
- **Dependency Injection:** Decouple component dependencies

---

## 8. Code Quality Standards

### 8.1 Naming Conventions
| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `UserRepository` |
| Functions | snake_case | `calculate_total()` |
| Variables | snake_case | `user_profile` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_RETRIES` |
| Booleans | `is_*`, `has_*`, `should_*` | `is_active` |

### 8.2 Type Hints
Use type hints to improve IDE support and catch bugs early: `def fetch_users(status: str = "active") -> List[User]`

### 8.3 Testing
- Every public method should have at least one unit test
- Test edge cases and error paths
- Target 80%+ code coverage

### 8.4 Security
- Validate all user input at entry points
- Never hardcode secrets—use environment variables or secret managers
- Encrypt sensitive data at rest and in transit
- Keep dependencies updated with security patches

### 8.5 Documentation
- Code should be self-documenting
- Comments explain **why**, not **what**
- Provide docstrings with examples and expected behavior

---

## 9. Multi-Tenant Architecture (Project-Specific)

### 9.1 Tenant Identification
Every request must carry a `tenant_id`. Extract from URL path, headers, or payload. All configuration and credentials are tenant-specific.

### 9.2 Custom Libraries
- **http-client:** For ALL external HTTP calls (built-in circuit breaker, retries, logging)
- **observability-lib:** For ALL logging (structured JSON, distributed tracing)

### 9.3 Service Discovery
All inter-service communication routes through **gateway-council**. Never hardcode service URLs.

### 9.4 Centralized Configuration
Fetch ALL tenant-specific config from **workspace-council** per-request. Never cache credentials.

### 9.5 Multi-Tenant Security
- Per-tenant webhook secrets (not global)
- Signature verification uses tenant-specific secrets
- API tokens fetched per-request (never cached)
- Errors isolated per tenant

### 9.6 FastAPI Best Practices
- **Dependency Injection**: Always use `Depends()` for service injection. Never instantiate services globally or in route handlers.
- **Router Organization**: One router per domain entity (`routes/jira_routes.py`, `routes/sprint_routes.py`).

### 10.1 Client vs. Service Split
- **Clients (`src/clients/`)**: Pure HTTP infrastructure. Handle auth, rate limits, and raw JSON. No business logic.
- **Services (`src/services/`)**: Business domain logic. manipulate data types, handle errors, and orchestrate workflows.
- **Pattern**: `JiraClientFactory` creates specific clients (`IssueClient`, `SprintClient`) to keep files small.

### 10.2 Resilience Patterns
- **Retry Decorators**: Annotate external calls with `@with_retry`. Never write `while` loops for retries.
- **Fallback Decorators**: Annotate non-critical reads with `@with_fallback`. Return safe defaults (e.g., `[]`, `False`) instead of crashing.

### 10.3 Centralized Constants
- Move ALL static values to `src/constants/`.
- Types: `jira_constants.py`, `kafka_topics.py`, `defaults.py`.
- **Rule**: If a string/number appears twice, it's a constant.

### 10.4 Webhook Strategy
- Dispatcher: `WebhookService` validates signature and identifies event type.
- Strategy: Route to specialized handlers in `src/services/handlers/` (e.g., `SprintWebhookHandler`, `IssueWebhookHandler`).
- **Benefit**: Changing webhook logic doesn't touch the main service.

### 10.5 Dependency Management Strategy
- **Centralized Providers**: Define factory functions in `src/dependencies.py` (e.g., `get_jira_service()`) that handle service instantiation.
- **FastAPI Injection**: Use `Depends(get_service)` in routes. Never instantiate services manually.
- **State Isolation**: Services should be stateless or initialized per-request to avoid cross-tenant contamination.


---

## 11. VSA Filter Implementation Pattern & Data Integrity (Project-Specific)

### 11.1 Filter Lifecycle Checklist
Every new post-VSA filter follows this standard 5-layer sequence:

| Layer | Files to Modify/Create | Action |
|-------|----------------------|--------|
| Constants | `src/constants/vsa_constants.py` | Add `{FILTER}_DIR_NAME`, `WEEKLY_{FILTER}_DIR_NAME`, `MONTHLY_{FILTER}_DIR_NAME` + configurable thresholds |
| Models | `src/models/vsa_models.py` | Add `{Filter}Classification` dataclass with `symbol`, `label`, `sentiment` + filter-specific metrics |
| Services | `src/services/vsa/{filter}_service.py` (x3) | Daily: scan `Results/`, copy to filter dir. Weekly/Monthly: consolidate completed periods then classify |
| Pipeline | `processor_service.py`, `pattern_router_service.py`, `consensus_engine_service.py` | Register dirs, wire execution, merge signals supplementally |
| Reporting | `data_aggregator.py`, `json_publisher.py`, `main_notifier.py`, `main_processor.py` | Aggregate, publish JSON, email section, distribution logging |
| Dashboard | `index.html`, `app.js` | Stat card + expandable timeframe tables (reuses existing `.eigen-row` / `.eigen-table-container` CSS) |

### 11.2 Candle Consolidation Pattern
- **Weekly candles**: Group by `df["Date"].dt.strftime("%G-W%V")`, **exclude current in-progress week** (`%G-W%V < current_period`).
- **Monthly candles**: Group by `df["Date"].dt.to_period("M")`, **exclude current in-progress month** (`%Y-%m < current_period`).
- **Aggregation**: `Open=first`, `High=max`, `Low=min`, `Close=last`, `Volume=sum`.
- **Derived metrics**: Always recalculate `Spread = High - Low` and `Close_Position = (Close - Low) / Spread`.
- **DRY Principle**: `DataAggregator` must reuse `Weekly{Filter}Service._consolidate_to_weekly` and `Monthly{Filter}Service._consolidate_to_monthly` instead of duplicating groupby code.

### 11.3 Aggregator & Reporting Precision Invariant
- **No Premature Rounding in Condition Checks**: When verifying if a stock in a filter folder meets reporting thresholds, always compare raw unrounded variables against thresholds (`body >= THRESHOLD * spread`). Only format with `round(..., 4)` after condition validation.

### 11.4 Centralized Constants Invariant
- All CLI utilities, logging functions, and routers must reference folder paths via `const.{DIR_NAME}` from `vsa_constants.py`. Never hardcode directory strings (e.g., `"AgeAgain"`, `"Trending"`).

### 11.5 Supplemental Consensus Blending
- Non-Eigen filters (such as Volume Trap) are merged into `ConsensusEngineService` as supplemental signals (`_populate_supplemental_signals()`).
- Signals only fill slots where the symbol/timeframe is `"None"`, preventing double-counting while expanding universe coverage.

### 11.6 Detailed Email Reporting Pattern
- Filter sections in `main_notifier.py` must include both a summary badge header (total counts, Bullish/Bearish breakdown) and detailed tabular views for each populated timeframe (Daily, Weekly, Monthly).
- Tables must show key filter metrics (`Symbol`, `Sentiment` badge, `Vol Δ%`, `Spread Δ%`, `Body Ratio`) in clean responsive HTML with an ASCII fallback in plain-text reports.

---

## Final Checklist

Before submitting:

- [ ] Methods ≤ 30 lines, classes ≤ 300 lines
- [ ] SOLID principles applied
- [ ] Single responsibility per class
- [ ] Dependency injection used (FastAPI `Depends`)
- [ ] Defensive programming with clear errors
- [ ] Type hints on all functions
- [ ] Structured logging with context
- [ ] Using http-client (not raw httpx)
- [ ] Using observability-lib (not print/raw logging)
- [ ] All config from workspace-council
- [ ] tenant_id in all operations
- [ ] Routes through gateway-council
- [ ] Per-tenant credentials/secrets
- [ ] Constants centralized in `src/constants/`
- [ ] External calls wrapped in `@with_retry`