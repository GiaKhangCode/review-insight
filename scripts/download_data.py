"""Download UIT-ViSFD into data/raw/UIT-ViSFD/ and verify its integrity.

Usage:
    uv run python scripts/download_data.py            # no-op if already present
    uv run python scripts/download_data.py --force    # re-download

The dataset is free for research use; cite Phan et al. (2021). It is not redistributed in
this repository (see docs/DATA_CARD.md).
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip"
# Pinned on 2026-11-21. If upstream changes the file, this check fails loudly instead of
# silently evaluating on different data.
EXPECTED_SHA256 = "d3c068814d23b11473c6bf774a81ffe1916b304501ac10e0c99fcd15f11a16a8"
EXPECTED_FILES = frozenset({"Train.csv", "Dev.csv", "Test.csv"})
TIMEOUT_S = 60


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    # Write to a temporary name first: an interrupted download never looks like a valid file.
    partial = destination.with_suffix(".part")
    # URL is a hard-coded https constant, not user input.
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310
        partial.write_bytes(response.read())
    partial.replace(destination)


def extract(archive: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if names != EXPECTED_FILES:
            raise RuntimeError(f"unexpected archive contents: {sorted(names)}")
        # Extract only the expected, flat file names: no path traversal ("../") possible.
        for name in EXPECTED_FILES:
            (target_dir / name).write_bytes(zf.read(name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    target_dir: Path = args.data_dir / "UIT-ViSFD"
    if not args.force and all((target_dir / name).exists() for name in EXPECTED_FILES):
        print(f"already present: {target_dir} (use --force to re-download)")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "UIT-ViSFD.zip"

    print(f"downloading {URL}")
    download(URL, archive)

    actual = sha256_of(archive)
    if actual != EXPECTED_SHA256:
        archive.unlink()
        print(f"checksum mismatch: expected {EXPECTED_SHA256}, got {actual}", file=sys.stderr)
        return 1

    extract(archive, target_dir)
    print(f"ok: {sorted(EXPECTED_FILES)} -> {target_dir}")
    print("Please cite: Phan et al. (2021), SA2SL, KSEM 2021. Research use only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
