# Testing

> [!NOTE]
> Integration and full pipeline tests require neuroimaging tools via `niwrap`.
> Pass `--runner <local|docker|singularity>` to select how they run.

## Running tests

```bash
# Unit tests only (fast, no runner needed)
pytest -m unit

# Quick integration tests
pytest -m "integration and not slow" --runner docker

# All integration tests (including slow)
pytest -m integration --runner docker

# Full pipeline
pytest -m full_pipeline --runner docker

# Everything
pytest --runner docker
```

## Test tiers

Directory-based markers (`unit`, `integration`, `full_pipeline`) are auto-applied
by `conftest.py`. The `slow` marker is applied manually on individual long-running
integration tests.

| Marker           | Source         | Typical duration |
| ---------------- | -------------- | ---------------- |
| `unit`           | auto (dir)     | < 1 s per test   |
| `integration`    | auto (dir)     | 1–5 min          |
| `slow`           | manual         | 5–30 min         |
| `full_pipeline`  | auto (dir)     | 30+ min          |

## Directory structure

```
tests/
├── conftest.py        # Shared fixtures (auto-markers, niwrap runner, test subject)
├── unit/              # Pure logic: BIDS parsing, file helpers
├── integration/       # Single-tool runs with real data (@slow on long ones)
├── full_pipeline/     # End-to-end workflow tests
└── data/              # Test datasets (not in version control)
```

## Coverage

```bash
pytest --cov=src --cov-report=term-missing
```
