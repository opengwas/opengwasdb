"""Build collisions in a Dense Store Release: two source variants on one ALID (#325).

When the hg19 -> hg38 lift puts two source variants on one ALID,
`build-dense-vcf` writes both into that ALID's row. The row keeps one rsid and
one source's statistics, and `_source_alids_by_alid` blanks its `source_alid`
as ambiguous. Nothing fails, and `validate` and `check-observed-view` both pass.
This module finds every such row in a Store from its Observed-Only parent's
blank `source_alid`s. It reconstructs the colliding hg19 variants by lifting a
source GWAS-VCF's records with that Store's own lift, and asks the source VCFs
whose statistics the row holds.

None of this compares two Stores, so it also sees a collision that both Stores
share -- which a cross-Store diff cannot (opengwas/opengwasdb-stores#312).
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pysam

from opengwasdb.build.liftover import lifted_rows, statistics_swap

#: A source variant as a GWAS-VCF spells it: ``(chrom, pos, ref, alt)``.
SourceKey = tuple[str, int, str, str]
#: Where a Store's lift puts a source variant -- ``(hg38 ALID, statistics
#: swapped)`` -- or None when it does not lift.
LiftFn = Callable[[SourceKey], tuple[str, bool] | None]
#: The hg19 positions the chain maps onto one hg38 position, with block strand.
SourcesFn = Callable[[str, int], list[tuple[str, int, str]]]

LIFTOVER = "opengwasdb/build/liftover.py"
#: The files #288's strand-aware lift is made of. A revision whose copies are
#: byte-identical to this checkout's lifts exactly as the installed package does.
LIFT_FILES = (LIFTOVER, "opengwasdb/readers/gwas_vcf.py")
#: Half an `int16_fixed` z code (scale 1024): a stored z reproduces its source
#: statistic to within this, so two candidates cannot both match by rounding.
Z_MATCH = 0.6 / 1024


@dataclass(frozen=True)
class StoreLift:
    """One Store's lift, the revision it came from, and how it is reproduced here."""

    revision: str
    how: str
    blobs: dict[str, str | None]
    lift: LiftFn = field(compare=False, repr=False)

    def record(self) -> dict[str, Any]:
        return {"revision": self.revision, "how": self.how, "blobs": self.blobs}


def _git_blob_id(data: bytes) -> str:
    """The id git gives a file's content (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _blob(repo: Path, revision: str, path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{revision}:{path}"],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def store_lift(repo: Path, revision: str, workdir: Path, chain: Path) -> StoreLift:
    """The lift a Store was built with, reproduced from its build's revision.

    A revision whose lift files are blob-identical to this checkout's is the
    installed package's lift. Any other revision is loaded from a `git archive`
    of its `liftover.py`, which must be self-contained: one that imports the
    package cannot be reproduced on its own, and the run refuses rather than
    lifting with today's code under yesterday's name.
    """
    blobs = {path: _blob(repo, revision, path) for path in LIFT_FILES}
    if blobs == {path: _blob(repo, "HEAD", path) for path in LIFT_FILES}:
        return StoreLift(revision, f"installed package: its lift files are blob-identical to "
                         f"{revision[:7]}'s", blobs, _installed_lift(chain))
    return StoreLift(revision, f"git archive of {LIFTOVER} at {revision}", blobs,
                     archived_lift(repo, revision, workdir, chain))


def _installed_lift(chain: Path) -> LiftFn:
    def lift(key: SourceKey) -> tuple[str, bool] | None:
        rows, _, _ = lifted_rows([key], chain_file=chain)
        row = rows.get(key)
        return None if row is None else (row.alid, statistics_swap(key, row))

    return lift


def archived_lift(repo: Path, revision: str, workdir: Path, chain: Path) -> LiftFn:
    """`build_liftover_lookup` from a revision's own `liftover.py`, archived under `workdir`.

    That lift predates #288: it kept the source's alleles, so it never swapped
    a statistic, which is what the returned flag says.
    """
    expected = _blob(repo, revision, LIFTOVER)
    if expected is None:
        raise SystemExit(f"{revision} has no {LIFTOVER}; its lift cannot be reproduced")
    path = workdir / revision / LIFTOVER
    # A copy left in the workdir is reused only if it is still the revision's
    # blob; an edited one would otherwise run under the revision's name.
    if not path.is_file() or _git_blob_id(path.read_bytes()) != expected:
        (workdir / revision).mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", str(repo), "archive", revision, LIFTOVER],
                                 capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(workdir / revision)], input=archive.stdout,
                       check=True)
    source_bytes = path.read_bytes()
    if _git_blob_id(source_bytes) != expected:
        raise SystemExit(f"{path} is not {revision}:{LIFTOVER} even after re-extraction")
    source = source_bytes.decode("utf-8")
    if "import opengwasdb" in source or "from opengwasdb" in source:
        raise SystemExit(f"{revision}:{LIFTOVER} imports the package, so its lift cannot be "
                         "reproduced on its own; refusing to substitute today's lift")
    spec = importlib.util.spec_from_file_location(f"liftover_{revision}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def lift(key: SourceKey) -> tuple[str, bool] | None:
        # A one-variant batch that fails would breach any failure threshold.
        alid = module.build_liftover_lookup([key], chain_file=chain, failure_threshold=1.0).get(key)
        return None if alid is None else (str(alid), False)

    return lift


def collision_rows(parent: Path) -> pd.DataFrame:
    """The rows of an Observed-Only release whose `source_alid` was blanked."""
    frame = pd.read_csv(parent / "variants.tsv.gz", sep="\t", dtype=str,
                        usecols=["#chromosome", "position", "alid", "rsid", "source_alid"])
    blank = frame["source_alid"].isin([".", ""])
    if blank.all():
        raise SystemExit(f"{parent}: no row records a source_alid, so collisions cannot be read")
    return frame[blank].reset_index(drop=True)


def vcf_records(vcf: Path, chrom: str, pos: int) -> list[dict[str, Any]]:
    """Each allele a GWAS-VCF records at one position, with its z for ALT.

    z is EZ when the file gives one, else ES/SE: the reader's own precedence.
    """
    out: list[dict[str, Any]] = []
    with pysam.VariantFile(str(vcf)) as handle:
        for rec in handle.fetch(chrom, pos - 1, pos):
            if rec.pos != pos:
                continue
            sample = rec.samples[0]
            for k, alt in enumerate(rec.alts or ()):
                ez = sample.get("EZ")
                if ez is not None and ez[k] is not None:
                    z = float(ez[k])
                else:
                    z = float(sample["ES"][k]) / float(sample["SE"][k])
                out.append({"key": (chrom, pos, rec.ref, alt), "rsid": rec.id, "z_alt": z})
    return out


def oriented_z(key: SourceKey, z_alt: float, swapped: bool) -> float:
    """The z a Store holds for a source record: signed to the canonical A1, then swapped."""
    _, _, ref, alt = key
    z = z_alt if alt == min(ref, alt) else -z_alt
    return -z if swapped else z


def describe_collision(
    alid: str, *, lift: LiftFn, sources: SourcesFn, vcfs: dict[str, Path],
    stored_z: Callable[[str, str], float],
) -> dict[str, Any]:
    """The hg19 variants that land on `alid`, and whose statistics the row holds.

    Candidates are the source VCF's records at every hg19 position the chain
    maps onto the ALID's position that this Store's lift puts on the ALID. For
    each Analysis checked, a candidate matches when its oriented z is within
    half a code of the stored z. `statistics_from` names the one candidate
    that matches in every checked Analysis. `matches_neither` is set when the
    stored z is finite and no candidate reproduces it -- the caller stops on it.
    """
    chrom, pos = alid.split(":")[0], int(alid.split(":")[1])
    first = next(iter(vcfs))
    candidates: dict[SourceKey, dict[str, Any]] = {}
    for t_chrom, t_pos, strand in sources(chrom, pos):
        for rec in vcf_records(vcfs[first], t_chrom, t_pos):
            landed = lift(rec["key"])
            if landed is None or landed[0] != alid or rec["key"] in candidates:
                continue
            candidates[rec["key"]] = {"source": ":".join(map(str, rec["key"])), "rsid": rec["rsid"],
                                      "chain_strand": strand, "swapped": landed[1], "z": {}}
    neither = False
    for analysis, vcf in vcfs.items():
        stored = stored_z(alid, analysis)
        hits = 0
        for key, cand in candidates.items():
            rec = next((r for r in vcf_records(vcf, key[0], key[1]) if r["key"] == key), None)
            expected = None if rec is None else oriented_z(key, rec["z_alt"], cand["swapped"])
            match = (None if expected is None or not np.isfinite(stored)
                     else bool(abs(expected - stored) <= Z_MATCH))
            cand["z"][analysis] = {"expected": expected, "matches": match}
            hits += bool(match)
        neither |= bool(np.isfinite(stored)) and hits == 0
    matched = [c for c in candidates.values()
               if c["z"] and all(v["matches"] is True for v in c["z"].values())]
    return {
        "alid": alid,
        "sources": list(candidates.values()),
        "stored_z": {analysis: stored_z(alid, analysis) for analysis in vcfs},
        "statistics_from": matched[0]["source"] if len(matched) == 1 else "not determined",
        "matches_neither": neither,
    }
