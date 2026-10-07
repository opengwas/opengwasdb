"""GWAS-VCF SourceReader (issue #19; updated for #17, #20, #21).

Wraps the existing bcftools-based `opengwasdb.build.vcf_source` logic behind
the `SourceReader` interface. `stream_associations` is a thin re-shaping of
`stream_vcf_associations`'s existing tuples, attaching `stored_effect_scale`
supplied at construction rather than reading it from the VCF -- the source
header is not authoritative for effect scale (issue #17, the ieu-a-7 fix:
`opengwasdb.build.vcf_source` no longer reads it at all). `stream_variants` is
a thin re-shaping of `stream_vcf_variants`, the unfiltered superset the dense
and hybrid builders' union-variant pass needs (issue #20).

`extract_at_sites` is the combined AF+SE-at-sites lookup ancestry assignment
and phenotype-SD estimation both drive (issue #21) -- the single place that
owns the bcftools-query-plus-`orient_to_canonical` site-extraction pattern,
formerly duplicated as `opengwasdb.ancestry.extract.extract_af_at_sites`
(AF-only, no SE). `is_palindromic`, `write_regions_file`, `_parse_af`, and the
liftover helpers live here too now, rather than in the ancestry package, since
they are GWAS-VCF/bcftools-specific, not ancestry-specific.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, KeysView
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.build.vcf_source import (
    _require_bcftools,
    stream_vcf_associations,
    stream_vcf_variants,
)
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import VariantNormalisationError, orient_to_canonical

GWAS_VCF_CAPABILITY = "opengwasdb.gwas-vcf"

# Strand-ambiguous allele pairs: unalignable without strand info, so excluded.
_PALINDROMES = frozenset({frozenset({"A", "T"}), frozenset({"C", "G"})})


def is_palindromic(ref: str, alt: str) -> bool:
    """True for strand-ambiguous A/T or C/G variants."""
    return frozenset({ref.upper(), alt.upper()}) in _PALINDROMES


def write_regions_file(alids: Iterable[str], path: str | Path) -> Path:
    """Write a bcftools ``-R`` regions file (``CHROM\tPOS``) from canonical ALIDs.

    Positions are de-duplicated and sorted so ``bcftools query -R`` reads each
    site once in coordinate order.
    """
    seen: set[tuple[str, int]] = set()
    for alid in alids:
        chrom, pos_str, _a1, _a2 = alid.split(":", 3)
        seen.add((chrom, int(pos_str)))

    def _key(cp: tuple[str, int]) -> tuple[int, str, int]:
        chrom, pos = cp
        return (int(chrom) if chrom.isdigit() else 1000, chrom, pos)

    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        for chrom, pos in sorted(seen, key=_key):
            fh.write(f"{chrom}\t{pos}\n")
    return path


def load_liftover(
    from_build: str = "hg19", to_build: str = "hg38", chain_file: str | None = None
) -> object:
    """Load a pyliftover ``LiftOver`` for orienting study AF onto the reference build."""
    from pyliftover import LiftOver  # type: ignore[import-untyped]

    return LiftOver(chain_file) if chain_file else LiftOver(from_build, to_build)


def _lift(liftover: object, chrom: str, pos_str: str) -> tuple[str, str] | None:
    """Lift ``(chrom, 1-based pos)`` to the target build; None if unmapped/non-standard."""
    bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
    try:
        mapped = liftover.convert_coordinate(f"chr{bare}", int(pos_str) - 1)  # type: ignore[attr-defined]
    except (ValueError, KeyError):
        return None
    if not mapped:
        return None
    new_chrom = mapped[0][0]
    new_chrom = new_chrom[3:] if new_chrom.lower().startswith("chr") else new_chrom
    return new_chrom, str(int(mapped[0][1]) + 1)


def _parse_af(value: str) -> float | None:
    """A usable `AF` from one bcftools `%AF` cell, by the package's one frequency rule.

    Delegated to `opengwasdb.stats.parse_af` rather than re-spelled: a second
    copy of "what is a frequency" is a second answer to it, and the copy here
    disagreed with the shared rule about exactly `0.0` and exactly `1.0` -- the
    values a monomorphic site reports and the placeholder a broken source
    reports on every row (stores #176). bcftools prints a genuinely missing
    value as `.`, which the shared rule already treats as missing.
    """
    return parse_af(value)


def _parse_positive_float(value: str) -> float | None:
    if value in {".", ""}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0.0 else None


@dataclass(frozen=True)
class GwasVcfReader:
    """SourceReader for one GWAS-VCF file.

    `stored_effect_scale` is Analytical Metadata for this file's Analysis,
    resolved by the caller from the build manifest (issue #16's schema) --
    not derived from the file itself. It defaults to `SD` because callers that
    only ever use `extract_at_sites` (ancestry assignment, phenotype-SD
    estimation) have no effect-scale concept to supply and never read it back.

    `liftover`, `regions_file`, and `region` are `extract_at_sites`-only
    concerns (issue #21). bcftools rejects ``-r``/``-R`` together, so `region`
    (bcftools ``-r``, e.g. ``"1"`` -- a cheap chromosome-level restriction)
    takes priority when set, skipping the ``-R`` regions file entirely
    (typical pairing: `region` with `liftover`, since post-lift ALIDs can't
    drive a pre-lift `-R` regions file anyway -- see below). Otherwise, when
    `liftover` is set, `extract_at_sites` does a full unrestricted scan,
    lifting each row before computing its canonical ALID, since a `-R`
    regions file built from post-lift (target-build) ALIDs would not match
    this file's pre-lift coordinates. When neither `region` nor `liftover` is
    set, `regions_file` -- if supplied -- is reused as-is (callers extracting
    from many files against the same site set build it once); otherwise one
    is built internally from the requested ALIDs, as before.
    """

    path: str | Path
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD
    liftover: object | None = None
    regions_file: str | Path | None = None
    region: str | None = None

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        for chrom, pos, ref, alt, z, se, eaf, rsid in stream_vcf_associations(self.path):
            yield ReaderAssociation(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                z=z,
                se=se,
                stored_effect_scale=self.stored_effect_scale,
                eaf=eaf,
                rsid=rsid,
            )

    def stream_variants(self) -> Iterator[SourceVariant]:
        for chrom, pos, ref, alt, rsid in stream_vcf_variants(self.path):
            yield SourceVariant(
                chromosome=chrom, position=pos, ref=ref, alt=alt, rsid=rsid
            )

    def extract_at_sites(self, alids: Iterable[str]) -> dict[str, SiteMetrics]:
        wanted = (
            alids if isinstance(alids, (set, frozenset, dict, KeysView)) else set(alids)
        )
        if not wanted:
            return {}
        bcftools = _require_bcftools()
        cmd = [
            bcftools,
            "query",
            "-f",
            "%CHROM\t%POS\t%REF\t%ALT\t[%AF]\t[%SE]\n",
        ]
        out: dict[str, SiteMetrics] = {}
        with contextlib.ExitStack() as stack:
            # bcftools rejects -r/-R together, so region (a coarser, cheaper
            # restriction) takes priority when set.
            if self.region is not None:
                cmd += ["-r", str(self.region)]
            elif self.liftover is None:
                if self.regions_file is not None:
                    regions_path = Path(self.regions_file)
                else:
                    tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                    regions_path = write_regions_file(wanted, Path(tmp_dir) / "regions.tsv")
                cmd += ["-R", str(regions_path)]
            cmd.append(str(self.path))

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 6:
                        continue
                    chrom_raw, pos_str, ref, alt, af_str, se_str = parts
                    if "," in alt or is_palindromic(ref, alt):
                        continue
                    af = _parse_af(af_str)
                    se = _parse_positive_float(se_str)
                    if af is None or se is None:
                        continue
                    chrom, pos = chrom_raw, pos_str
                    if self.liftover is not None:
                        lifted = _lift(self.liftover, chrom_raw, pos_str)
                        if lifted is None:
                            continue
                        chrom, pos = lifted
                    try:
                        orientation = orient_to_canonical(chrom, pos, alt, ref)
                    except VariantNormalisationError:
                        continue
                    alid = orientation.variant.alid
                    if alid not in wanted:
                        continue
                    out[alid] = SiteMetrics(
                        af=(1.0 - af) if orientation.flipped else af,
                        se=se,
                    )
            finally:
                proc.stdout.close()  # type: ignore[union-attr]
                proc.wait()
        return out
