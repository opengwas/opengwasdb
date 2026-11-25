"""Where the scripts' inputs and tools live: documented environment variables, never constants.

Each root has a default for the IEU nodes. The manifest records an input as its root's
variable plus the rest of its path, and by hash, so no host path is committed.

- `OPENGWASDB_LD_PHASED_V2`, default `/data/opengwasdb/work/ogld/phased_v2`: the phased
  HGDP+1kGP v2 BCFs, each with its `.complete` record (ADR 0088);
- `OPENGWASDB_LD_PCS`, default `/data/opengwasdb/work/ogld/pcs`: `scripts/ld/make_pcs.py`'s PC
  Files and reports;
- `OPENGWASDB_LD_REFERENCE`, default `/data/opengwasdb/reference`: the panels' members files
  and gnomAD's sample metadata (`META_NAME` under it);
- `OPENGWASDB_LD_KEEP`, default `/data/opengwasdb/work/ogld/keep`: `scripts/ld/make_keeps.py`'s
  keep files, each panel's unrelated samples (ADR 0088);
- `OPENGWASDB_LD_PGEN_V2`, default `/data/opengwasdb/work/ogld/phased_v2_pgen`:
  `scripts/ld/build_hgdp1kgp.py`'s conversions, one directory per panel (ogld#7);
- `OPENGWASDB_LD_STORES`, default `/data/opengwasdb/work/ogld/stores`: the five HGDP+1kGP LD
  Stores, which opengwas/opengwasdb-stores builds and records (`tools/ld-stores/`, #541);
- `OPENGWASDB_LD_SCRATCH`, default `/data/opengwasdb/work/ogld/scratch/7`: that script's logs
  and timings;
- `OPENGWASDB_LD_403_ANALYSES`, default `/data/opengwasdb/work/ogld/realdata/403-analyses`:
  opengwasdb#403's 37 EUR Analyses, `scripts/ld/realdata/make_403_inputs.py`'s output, which the
  realdata tier, the acceptance run and `scripts/ld/bench/latency.py` read (ogld#6, ogld#8);
- `OPENGWASDB_LD_LAYOUTS`, default `/data/opengwasdb/work/ogld/scratch/8/layouts`: EUR chr22
  built once per Near Band layout by `scripts/ld/bench/build_layouts.py`, which
  `scripts/ld/bench/latency.py` times (ogld#8);
- `OPENGWASDB_LD_PLINK2`, `OPENGWASDB_LD_BCFTOOLS`, `OPENGWASDB_LD_BGZIP`, default the program
  on `PATH`: the tools.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from opengwasdb.ld import env

DEFAULTS = {
    "OPENGWASDB_LD_PHASED_V2": "/data/opengwasdb/work/ogld/phased_v2",
    "OPENGWASDB_LD_PCS": "/data/opengwasdb/work/ogld/pcs",
    "OPENGWASDB_LD_REFERENCE": "/data/opengwasdb/reference",
    "OPENGWASDB_LD_KEEP": "/data/opengwasdb/work/ogld/keep",
    "OPENGWASDB_LD_PGEN_V2": "/data/opengwasdb/work/ogld/phased_v2_pgen",
    "OPENGWASDB_LD_STORES": "/data/opengwasdb/work/ogld/stores",
    "OPENGWASDB_LD_SCRATCH": "/data/opengwasdb/work/ogld/scratch/7",
    "OPENGWASDB_LD_403_ANALYSES": "/data/opengwasdb/work/ogld/realdata/403-analyses",
    "OPENGWASDB_LD_LAYOUTS": "/data/opengwasdb/work/ogld/scratch/8/layouts",
}
#: gnomAD v3.1.2's HGDP+1kGP sample metadata, under `OPENGWASDB_LD_REFERENCE`.
META_NAME = "hgdp1kgp-hg38/source/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.tsv.bgz"
TOOLS = {
    "plink2": "OPENGWASDB_LD_PLINK2",
    "bcftools": "OPENGWASDB_LD_BCFTOOLS",
    "bgzip": "OPENGWASDB_LD_BGZIP",
}


def read(name: str, default: str) -> str:
    """`env.read`, for a script: an old variable that is set is reported on stderr and the script
    exits 2 with no traceback. The one place a script reports it."""
    try:
        return env.read(name, default)
    except env.LegacyVariable as refused:
        print(refused, file=sys.stderr)
        raise SystemExit(2) from None


def root(name: str) -> Path:
    return Path(read(name, DEFAULTS[name]))


def tool(name: str) -> Path:
    """A tool from its variable, else from `PATH`; refused, naming the variable, if neither."""
    found = read(TOOLS[name], "") or shutil.which(name)
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
