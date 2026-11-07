"""Where the scripts' inputs and tools live: documented environment variables, never constants.

Each root has a default for the IEU nodes. The manifest records an input as its root's
variable plus the rest of its path, and by hash, so no host path is committed.

- `OGLD_PHASED_V2`, default `/data/opengwasdb/work/ogld/phased_v2`: the phased HGDP+1kGP v2
  BCFs, each with its `.complete` record (ADR 0002);
- `OGLD_PCS`, default `/data/opengwasdb/work/ogld/pcs`: `scripts/make_pcs.py`'s PC Files and
  reports;
- `OGLD_REFERENCE`, default `/data/opengwasdb/reference`: the panels' members files and gnomAD's
  sample metadata;
- `OGLD_PLINK2`, `OGLD_BCFTOOLS`, `OGLD_BGZIP`, default the program on `PATH`: the tools.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULTS = {
    "OGLD_PHASED_V2": "/data/opengwasdb/work/ogld/phased_v2",
    "OGLD_PCS": "/data/opengwasdb/work/ogld/pcs",
    "OGLD_REFERENCE": "/data/opengwasdb/reference",
}
TOOLS = {"plink2": "OGLD_PLINK2", "bcftools": "OGLD_BCFTOOLS", "bgzip": "OGLD_BGZIP"}


def root(name: str) -> Path:
    return Path(os.environ.get(name, DEFAULTS[name]))


def tool(name: str) -> Path:
    """A tool from its variable, else from `PATH`; refused, naming the variable, if neither."""
    found = os.environ.get(TOOLS[name]) or shutil.which(name)
    if not found:
        raise SystemExit(f"no {name}: set {TOOLS[name]} or put {name} on PATH")
    return Path(found)


def portable(value: str | Path, work: Path | None = None) -> str:
    """`value` with each root replaced by its variable and the work directory by `<work>`."""
    text = str(value)
    for name in DEFAULTS:
        text = text.replace(str(root(name)), f"${name}")
    return text.replace(str(work), "<work>") if work is not None else text
