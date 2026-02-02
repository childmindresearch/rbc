# Testing Guide

Quick guide for running tests in this project.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run fast tests (recommended during development)
pytest -m unit

# Run all tests (except slow ones)
pytest -m "not slow and not full_pipeline"

# Run everything
pytest
```

## Test Categories

| Marker          | Speed        | When to Run                        |
| --------------- | ------------ | ---------------------------------- |
| `unit`          | <1s per test | Always - during active development |
| `integration`   | 1-5 min      | Before committing                  |
| `slow`          | >5 min       | Before merging to main             |
| `full_pipeline` | 30+ min      | Before releases only               |

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

### Before Committing

```bash
# Run unit + integration (skip slow tests)
pytest -m "not slow and not full_pipeline"

# With coverage report
pytest -m "not slow and not full_pipeline" --cov=src --cov-report=term-missing
```

## Useful Options

```bash
# Stop on first failure
pytest -x

# Show which tests are slowest
pytest --durations=10

# Run tests matching a pattern
pytest -k "motion_correction"
```

## Test Data

Test data is stored in `tests/data/`.

## Writing Tests

### Unit Test (Fast)

```python
import pytest

@pytest.mark.unit
def test_parse_bids_filename():
    """Test BIDS filename parsing."""
    filename = "sub-01_task-rest_bold.nii.gz"
    result = parse_bids_filename(filename)

    assert result["sub"] == "01"
    assert result["task"] == "rest"
```

### Integration Test (Uses Real Data)

```python
import pytest

@pytest.mark.integration
def test_motion_correction(sample_bold, temp_dir):
    """Test motion correction with real data."""
    output = temp_dir / "corrected.nii.gz"

    motion_correct(sample_bold, output)

    assert output.exists()
    img = nib.load(output)
    assert img.shape[3] == 50  # Same number of volumes
```

## Available Fixtures

```python
# Mock data (fast, for unit tests)
def test_with_mock_data(mock_bold_image):
    # Uses synthetic 10x10x10x5 volume
    pass

# Real data (for integration tests)
def test_with_real_data(sample_bold):
    # Uses actual 50-volume BOLD scan
    pass

# Temporary directory
def test_with_temp_dir(temp_dir):
    # Cleaned up automatically after test
    output = temp_dir / "result.nii.gz"
    pass
```

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
