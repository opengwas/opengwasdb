"""`pixi run -e dev contract-check`: the Store Reader contract bundle is current (#474).

Two checks too slow for every test run (`CONTRIBUTING.md`, Test tiers):

1. **Regeneration.** `tests/store_reader_contract/generate.py` writes the bundle into a
   temporary directory, and `git diff --no-index --exit-code` compares its `releases`,
   `expected` and `VERSION` with the committed ones. A difference means a builder, a
   planting or an answer changed without the bundle: regenerate and commit it.
2. **Format parity, the other eight releases.** `tests/test_store_reader_contract_parity.py`
   runs three of the eleven releases in every test run; this runs the rest
   (`STORE_READER_CONTRACT_PARITY=rest`).

Exits non-zero, naming the check, if either fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "tests" / "store_reader_contract"
GENERATED = ("releases", "expected", "VERSION")


def regenerated_matches(scratch: Path) -> bool:
    """Whether a fresh run of the generator writes the committed bundle, byte for byte."""
    subprocess.run(
        [sys.executable, str(BUNDLE / "generate.py"), str(scratch)],
        check=True, stdout=subprocess.DEVNULL,
    )
    same = True
    for name in GENERATED:
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--exit-code", "--stat", str(BUNDLE / name),
             str(scratch / name)],
            cwd=ROOT,
        )
        same = same and diff.returncode == 0
    return same


def parity_passes() -> bool:
    """Whether the parity cases `pixi run -e dev test` leaves out pass."""
    environment = {**os.environ, "STORE_READER_CONTRACT_PARITY": "rest"}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_store_reader_contract_parity.py",
         "-n", "6", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, env=environment,
    )
    return run.returncode == 0


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        regenerated = regenerated_matches(Path(scratch))
    print("regeneration:", "the committed bundle" if regenerated else "DIFFERS (above)")
    parity = parity_passes()
    print("parity, the other eight releases:", "passed" if parity else "FAILED")
    return 0 if regenerated and parity else 1


if __name__ == "__main__":
    sys.exit(main())
