"""Where the scripts' inputs and tools live: documented environment variables, never constants.

Each root has a default for the IEU nodes. The manifest records an input as its root's
variable plus the rest of its path, and by hash, so no host path is committed.

- `OGLD_PHASED_V2`, default `/data/opengwasdb/work/ogld/phased_v2`: the phased HGDP+1kGP v2
  BCFs, each with its `.complete` record (ADR 0002);
- `OGLD_PCS`, default `/data/opengwasdb/work/ogld/pcs`: `scripts/make_pcs.py`'s PC Files and
  reports;
- `OGLD_REFERENCE`, default `/data/opengwasdb/reference`: the panels' members files and gnomAD's
  sample metadata (`META_NAME` under it);
- `OGLD_KEEP`, default `/data/opengwasdb/work/ogld/keep`: `scripts/make_keeps.py`'s keep files,
  each panel's unrelated samples (ADR 0002);
- `OGLD_PGEN_V2`, default `/data/opengwasdb/work/ogld/phased_v2_pgen`: `scripts/build_hgdp1kgp.py`'s
  conversions, one directory per panel (ogld#7);
- `OGLD_STORES`, default `/data/opengwasdb/work/ogld/stores`: the five HGDP+1kGP LD Stores, staged
  there until ogld#9 chooses where they are published;
- `OGLD_SCRATCH`, default `/data/opengwasdb/work/ogld/scratch/7`: that script's logs and timings;
- `OGLD_403_ANALYSES`, default `/data/opengwasdb/work/ogld/realdata/403-analyses`: opengwasdb#403's
  37 EUR Analyses, `scripts/realdata/make_403_inputs.py`'s output, which the realdata tier, the
  acceptance run and `scripts/bench/latency.py` read (ogld#6, ogld#8);
- `OGLD_LAYOUTS`, default `/data/opengwasdb/work/ogld/scratch/8/layouts`: EUR chr22 built once
  per Near Band layout by `scripts/bench/build_layouts.py`, which `scripts/bench/latency.py`
  times (ogld#8);
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
    "OGLD_KEEP": "/data/opengwasdb/work/ogld/keep",
    "OGLD_PGEN_V2": "/data/opengwasdb/work/ogld/phased_v2_pgen",
    "OGLD_STORES": "/data/opengwasdb/work/ogld/stores",
    "OGLD_SCRATCH": "/data/opengwasdb/work/ogld/scratch/7",
    "OGLD_403_ANALYSES": "/data/opengwasdb/work/ogld/realdata/403-analyses",
    "OGLD_LAYOUTS": "/data/opengwasdb/work/ogld/scratch/8/layouts",
}
#: gnomAD v3.1.2's HGDP+1kGP sample metadata, under `OGLD_REFERENCE`.
META_NAME = "hgdp1kgp-hg38/source/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.tsv.bgz"
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
    # The longest root first: `phased_v2_pgen` begins with `phased_v2`.
    for name in sorted(DEFAULTS, key=lambda n: -len(str(root(n)))):
        text = text.replace(str(root(name)), f"${name}")
    return text.replace(str(work), "<work>") if work is not None else text
