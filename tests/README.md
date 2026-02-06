# Testing Guide

Quick guide for running tests in this project.

> [!NOTE]
> Integration and full pipeline tests likely requires specific
> neuroimaging tools that can be called with `niwrap`. To perform
> these tests, also pass the `--runner` flag to `pytest`.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run fast tests (recommended during development)
pytest -m unit

# Run all tests (except slow ones)
pytest -m "not slow and not full_pipeline" --runner <local|docker|singularity>

# Run everything
pytest --runner <local|docker|singularity>
```

## Test Categories

| Marker          | Speed        |
| --------------- | ------------ |
| `unit`          | <1s per test |
| `integration`   | 1-5 min      |
| `slow`          | >5 min       |
| `full_pipeline` | 30+ min      |

## Common Commands

### Development Workflow

```bash
# Fast feedback loop (unit tests only)
pytest -m unit

# Test specific file
pytest tests/unit/test_bids_parsing.py

# Test specific function
pytest tests/unit/test_bids_parsing.py::test_parse_subject_id
```

<details>
<summary><h2>Useful Options</h2></summary>

```bash
# Stop on first failure
pytest -x

# Show which tests are slowest
pytest --durations=10

# Run tests matching a pattern
pytest -k "motion_correction"
```

</details>

## Test Data

Test data is stored in `tests/data/`.

## Coverage Requirements

- **Overall:** >85%
- **Unit tests:** >90%
- **Integration tests:** >80%

```bash
# Check coverage
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html
```

## CI/CD

Tests run automatically on GitHub Actions:

- **On every push:** Unit tests + fast integration tests
- **Manual trigger:** Full pipeline tests (slow)

To run the same tests as CI locally:

```bash
pytest -m "not slow and not full_pipeline" \
  --cov=src \
  --cov-report=xml \
  --junitxml=pytest.xml
```

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Fast tests (<1s)
├── integration/             # Medium tests (1-5 min)
├── full_pipeline/           # Slow tests (30+ min)
└── data/                    # Test datasets
```

## Best Practices

1. **Run unit tests frequently** - They're fast (<1s)
2. **Run integration tests before committing** - Catch issues early
3. **Use appropriate markers** - Auto-applied based on file location
4. **Write descriptive test names** - `test_motion_correction_preserves_dimensions`
5. **One assertion per test** - When practical
6. **Use fixtures** - Don't repeat setup code
7. **Keep tests independent** - No shared state between tests
