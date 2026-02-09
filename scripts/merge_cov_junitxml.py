# /// script
# dependencies = ["junitparser"]
# requires-python = ">=3.12"
# ///
"""Merge pytest-cov's junitxml file for codecov commenting.

Reads all generated junitxml files and generates:
  - pytest.xml (merged xml file)

Run with:
    uv run scripts/merge_cov_junitxml.py
"""

from pathlib import Path

from junitparser import JUnitXml


def main() -> None:
    """Merge generated junitxml files."""
    merged = JUnitXml()
    for f in ["pytest-quick.xml", "pytest-slow.xml", "pytest-full.xml"]:
        if Path(f).exists():
            merged += JUnitXml.fromfile(f)
    merged.write("pytest.xml")


if __name__ == "__main__":
    main()
