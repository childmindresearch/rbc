# /// script
# dependencies = []
# requires-python = ">=3.12"
# ///
"""Download a sub-01 subset of OpenNeuro ds000114 for longitudinal tests.

ds000114 ("Test-Retest Reliability") is the canonical multi-session BIDS
demo dataset. We pull sub-01 only (both ses-test and ses-retest) plus the
top-level sidecars, because BIDS inheritance places RepetitionTime etc.
in the dataset-level task JSON, not per-subject.

OpenNeuro snapshot: 1.0.2 (tagged 2022-08-24, verified to contain
``TaskName`` in ``task-fingerfootlips_bold.json`` so bids-validator is
happy). The public S3 mirror only exposes the latest state at top-level
(``s3://openneuro.org/ds000114/...``), so we can't pin via a snapshot
path. Instead we verify the sha256 of ``task-fingerfootlips_bold.json``
after download and refuse to proceed if the content has drifted.

Runs against the public HTTP S3 endpoint with no credentials and no
third-party dependencies, so it works on any environment where
``uv run`` does.

Usage::

    uv run scripts/download_ds000114.py [TARGET_DIR]

Default TARGET_DIR is ``tests/data/ds000114`` (under the repo root).
Idempotent: exits early if the sentinel T1w file already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SNAPSHOT_TAG = "1.0.2"
S3_BASE = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds000114"
SUBJECT = "sub-01"
SIDECARS = (
    "dataset_description.json",
    "participants.tsv",
    "task-fingerfootlips_bold.json",
)
TASK_JSON_SHA256 = "9fd44f65a772e05282c20bdfa2a9775e02f9a7f562c5c96bbf4fd30632540355"
S3_XMLNS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

_DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "tests" / "data" / DATASET
_SENTINEL_REL = Path(SUBJECT) / "ses-test" / "anat" / f"{SUBJECT}_ses-test_T1w.nii.gz"

_logger = logging.getLogger("download_ds000114")


def _list_keys(prefix: str) -> list[str]:
    """Enumerate all S3 object keys beneath ``prefix`` (paginated)."""
    keys: list[str] = []
    continuation: str | None = None
    while True:
        query = f"list-type=2&prefix={prefix}"
        if continuation is not None:
            query += f"&continuation-token={urllib.parse.quote(continuation)}"
        with urllib.request.urlopen(f"{S3_BASE}/?{query}") as resp:  # noqa: S310
            root = ET.parse(resp).getroot()  # noqa: S314 (trusted S3 response)
        for contents in root.findall(f"{S3_XMLNS}Contents"):
            key = contents.findtext(f"{S3_XMLNS}Key")
            if key:
                keys.append(key)
        if root.findtext(f"{S3_XMLNS}IsTruncated") != "true":
            return keys
        continuation = root.findtext(f"{S3_XMLNS}NextContinuationToken")
        if not continuation:
            return keys


def _download(key: str, dest: Path) -> None:
    """Fetch ``key`` from the public openneuro S3 mirror to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE}/{key}"
    _logger.info("GET %s", key)
    with (
        urllib.request.urlopen(url) as resp,  # noqa: S310
        dest.open("wb") as out,
    ):
        while chunk := resp.read(1 << 20):
            out.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(target_dir: Path) -> None:
    """Download the ds000114 sub-01 subset into ``target_dir`` if missing."""
    sentinel = target_dir / _SENTINEL_REL
    if sentinel.exists():
        _logger.info("%s: already present at %s", DATASET, target_dir)
        return

    _logger.info(
        "%s: downloading %s (both sessions) + sidecars to %s (snapshot %s)",
        DATASET,
        SUBJECT,
        target_dir,
        SNAPSHOT_TAG,
    )

    subject_keys = _list_keys(f"{DATASET}/{SUBJECT}/")
    if not subject_keys:
        raise RuntimeError(
            f"No objects found under {DATASET}/{SUBJECT}/; bucket layout may "
            "have changed."
        )
    for key in subject_keys:
        rel = Path(key).relative_to(DATASET)
        _download(key, target_dir / rel)

    for name in SIDECARS:
        _download(f"{DATASET}/{name}", target_dir / name)

    actual = _sha256(target_dir / "task-fingerfootlips_bold.json")
    if actual != TASK_JSON_SHA256:
        raise RuntimeError(
            "task-fingerfootlips_bold.json sha256 mismatch; upstream may have "
            f"changed.\n  expected: {TASK_JSON_SHA256}\n  actual:   {actual}\n"
            f"  verify and update SNAPSHOT_TAG + TASK_JSON_SHA256."
        )
    _logger.info("%s: download complete", DATASET)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a shell-style exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target_dir",
        nargs="?",
        type=Path,
        default=_DEFAULT_TARGET,
        help=f"Destination directory (default: {_DEFAULT_TARGET}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        download(args.target_dir)
    except urllib.error.URLError as exc:
        _logger.error("%s: network error: %s", DATASET, exc)
        return 2
    except RuntimeError as exc:
        _logger.error("%s: %s", DATASET, exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
