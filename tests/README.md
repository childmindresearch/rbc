# Testing

> [!NOTE]
> Integration and full pipeline tests require neuroimaging tools via `niwrap`.
> Pass `--runner <local|docker|singularity>` to select how they run.

## Running tests

```bash
# Unit tests only (fast, no runner needed)
pytest -m unit

# Unit + integration (standard CI run)
pytest -m "not full_pipeline" --runner docker

# Full pipeline (slow, manual trigger on CI)
pytest -m full_pipeline --runner docker

# Everything
pytest --runner docker
```

## Test tiers

Markers are auto-applied based on directory — no need to decorate tests manually.

| Directory        | Marker           | Typical duration |
| ---------------- | ---------------- | ---------------- |
| `unit/`          | `unit`           | < 1 s per test   |
| `integration/`   | `integration`    | 1–5 min          |
| `full_pipeline/` | `full_pipeline`  | 30+ min          |

## Directory structure

```
tests/
├── conftest.py        # Shared fixtures (auto-markers, niwrap runner, test subject)
├── unit/              # Pure logic: BIDS parsing, file helpers
├── integration/       # Single-tool runs with real data
├── full_pipeline/     # End-to-end workflow tests
└── data/              # Test datasets (not in version control)
```

## Coverage

```bash
pytest --cov=src --cov-report=term-missing
```
