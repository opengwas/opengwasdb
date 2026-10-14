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

``stream_metric_chunks`` is the bounded one-pass resolver projection (issue
#278). It makes one streaming record query after a permitted header capability
probe, preserves EZ-preferred-otherwise-ES/SE semantics, and retains identity
rows with missing statistics so ancestry and phenotype-SD apply their existing,
different eligibility rules downstream.
"""

from __future__ import annotations

import contextlib
import math
import subprocess
import tempfile
from collections.abc import Generator, Iterable, Iterator, KeysView
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from opengwasdb.build.vcf_source import (
    _derive_z,
    _require_bcftools,
    stream_vcf_associations,
    stream_vcf_variants,
)
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import (
    Orientation,
    VariantNormalisationError,
    orient_to_canonical,
)

if TYPE_CHECKING:
    from opengwasdb.readers.tabular import MetricsChunk, TabularMetricsRow

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


@cache
def load_liftover(
    from_build: str = "hg19", to_build: str = "hg38", chain_file: str | None = None
) -> object:
    """Load a pyliftover ``LiftOver`` for orienting study AF onto the reference build."""
    from pyliftover import LiftOver  # type: ignore[import-untyped]

    return LiftOver(chain_file) if chain_file else LiftOver(from_build, to_build)


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


class LiftoverProjectionError(ValueError):
    """A coordinate lift cannot identify one allele-aligned target variant."""


def _require_snp(effect: str, other: str) -> None:
    alleles = (effect, other)
    if any(len(allele) != 1 or allele.upper() not in "ACGT" for allele in alleles):
        raise LiftoverProjectionError(
            f"liftover supports one biallelic SNP at a time, got {effect!r}/{other!r}"
        )


def _unique_mapping(
    mapped: list[tuple[str, float, str, float]], identity: str
) -> tuple[str, float, str, float] | None:
    if not mapped:
        return None
    if len(mapped) != 1:
        raise LiftoverProjectionError(
            f"ambiguous liftover for {identity}: expected one mapping, found {len(mapped)}"
        )
    return mapped[0]


def _strand_alleles(strand: str, effect: str, other: str, identity: str) -> tuple[str, str]:
    if strand == "+":
        return effect, other
    if strand == "-":
        return effect.translate(_COMPLEMENT), other.translate(_COMPLEMENT)
    raise LiftoverProjectionError(f"unsupported liftover strand {strand!r} for {identity}")


def _lift_variant(
    liftover: object, chrom: str, pos_str: str, effect: str, other: str
) -> tuple[str, str, str, str] | None:
    """Lift one SNP and project its alleles onto one unambiguous mapping strand."""
    _require_snp(effect, other)
    bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
    identity = f"{chrom}:{pos_str}:{effect}:{other}"
    try:
        mapped = liftover.convert_coordinate(f"chr{bare}", int(pos_str) - 1)  # type: ignore[attr-defined]
    except (ValueError, KeyError):
        return None
    mapping = _unique_mapping(mapped, identity)
    if mapping is None:
        return None
    new_chrom, new_pos, strand, *_rest = mapping
    effect, other = _strand_alleles(strand, effect, other, identity)
    bare_target = new_chrom[3:] if new_chrom.lower().startswith("chr") else new_chrom
    return bare_target, str(int(new_pos) + 1), effect, other


def _oriented_variant(
    liftover: object | None, chrom: str, pos: str, effect: str, other: str
) -> Orientation | None:
    if liftover is not None:
        lifted = _lift_variant(liftover, chrom, pos, effect, other)
        if lifted is None:
            return None
        chrom, pos, effect, other = lifted
    try:
        return orient_to_canonical(chrom, pos, effect, other)
    except VariantNormalisationError:
        return None


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
    return parsed if parsed > 0.0 and math.isfinite(parsed) else None


def _format_tags(path: str | Path) -> set[str]:
    """Read FORMAT capability once; this is not a source-record traversal."""
    bcftools = _require_bcftools()
    result = subprocess.run(
        [bcftools, "view", "-h", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "no diagnostic"
        raise OSError(f"bcftools header query failed ({result.returncode}): {message}")
    lines = result.stdout.splitlines()
    columns = next((line.split("\t") for line in lines if line.startswith("#CHROM\t")), [])
    sample_count = max(0, len(columns) - 9)
    if sample_count != 1:
        raise ValueError(
            f"GWAS-VCF resolver requires exactly one sample column, found {sample_count}"
        )
    prefix = "##FORMAT=<ID="
    return {
        line[len(prefix) :].split(",", 1)[0]
        for line in lines
        if line.startswith(prefix)
    }


def _metric_query_format(tags: set[str]) -> str:
    fields = ["%CHROM", "%POS", "%REF", "%ALT"]
    fields.extend(f"[%{tag}]" if tag in tags else "." for tag in ("EZ", "ES", "SE", "AF"))
    return "\t".join(fields) + "\n"


def _validated_metric_cells(lines: Iterable[str]) -> Iterator[tuple[str, ...]]:
    for row_number, line in enumerate(lines, start=1):
        cells = tuple(line.rstrip("\n").split("\t"))
        if len(cells) != 8:
            raise ValueError(
                f"malformed bcftools output at record {row_number}: "
                f"expected 8 fields, got {len(cells)}"
            )
        yield cells


_QUERY_STOP_TIMEOUT_SECONDS = 5.0


def _bounded_wait(proc: subprocess.Popen[str]) -> int:
    try:
        return proc.wait(timeout=_QUERY_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        return proc.wait(timeout=_QUERY_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise OSError("bcftools query could not be reaped after kill") from exc


def _raise_query_failure(returncode: int, errors: TextIO) -> None:
    errors.seek(0)
    message = errors.read().strip() or "no diagnostic"
    raise OSError(f"bcftools record query failed ({returncode}): {message}")


def _finish_metric_query(proc: subprocess.Popen[str], errors: TextIO, completed: bool) -> None:
    if proc.stdout is not None:
        proc.stdout.close()
    cancelled = False
    if not completed and proc.poll() is None:
        proc.terminate()
        cancelled = True
    returncode = _bounded_wait(proc)
    if returncode != 0 and not (cancelled and returncode < 0):
        _raise_query_failure(returncode, errors)


def _stream_metric_cells(path: str | Path) -> Generator[tuple[str, ...], None, None]:
    """Run the one VCF record query used by the bounded resolver (issue #278)."""
    bcftools = _require_bcftools()
    query_format = _metric_query_format(_format_tags(path))
    completed = False
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        proc = subprocess.Popen(
            [bcftools, "query", "-f", query_format, str(path)],
            stdout=subprocess.PIPE, stderr=errors, text=True,
        )
        try:
            assert proc.stdout is not None
            yield from _validated_metric_cells(proc.stdout)
            completed = True
        finally:
            _finish_metric_query(proc, errors, completed)


def _metric_row(
    orientation: Orientation,
    ref: str,
    alt: str,
    af: float | None,
    beta: float | None,
    se: float | None,
) -> TabularMetricsRow:
    from opengwasdb.readers.tabular import TabularMetricsRow

    return TabularMetricsRow(
        orientation.variant.chromosome,
        orientation.variant.position,
        ref,
        alt,
        orientation.variant.alid,
        orientation.flipped,
        af,
        beta,
        se,
    )


@dataclass(frozen=True)
class GwasVcfReader:
    """SourceReader for one GWAS-VCF file.

    `stored_effect_scale` is Analytical Metadata for this file's Analysis,
    resolved by the caller from the build manifest (issue #16's schema) --
    not derived from the file itself. It defaults to `SD` because callers that
    only ever use `extract_at_sites` (ancestry assignment, phenotype-SD
    estimation) have no effect-scale concept to supply and never read it back.

    `regions_file` and `region` are `extract_at_sites`-only concerns (issue
    #21); `liftover` also applies to the resolver metrics stream (issue #278).
    bcftools rejects ``-r``/``-R`` together, so `region`
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
    chunk_rows: int = 50_000

    def __post_init__(self) -> None:
        if self.chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {self.chunk_rows!r}")

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        for chrom, pos, ref, alt, z, se, eaf in stream_vcf_associations(self.path):
            yield ReaderAssociation(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                z=z,
                se=se,
                stored_effect_scale=self.stored_effect_scale,
                eaf=eaf,
            )

    def stream_variants(self) -> Iterator[SourceVariant]:
        for chrom, pos, ref, alt, rsid in stream_vcf_variants(self.path):
            yield SourceVariant(
                chromosome=chrom, position=pos, ref=ref, alt=alt, rsid=rsid
            )

    def stream_metric_chunks(self) -> Generator[MetricsChunk, None, None]:
        """Yield bounded resolver metrics from one streaming record query (#278)."""
        from opengwasdb.readers.tabular import metrics_chunks_from_rows

        rows = self._stream_metrics()
        try:
            yield from metrics_chunks_from_rows(rows, chunk_rows=self.chunk_rows)
        finally:
            rows.close()

    def _stream_metrics(self) -> Generator[TabularMetricsRow, None, None]:
        """Project VCF records without dropping ancestry-only AF+SE rows."""
        cells_stream = _stream_metric_cells(self.path)
        try:
            for cells in cells_stream:
                chrom, pos, ref, alt, ez_str, es_str, se_str, af_str = cells
                orientation = _oriented_variant(self.liftover, chrom, pos, alt, ref)
                if orientation is None:
                    continue
                se = _parse_positive_float(se_str)
                z = None if se is None else _derive_z(ez_str, es_str, se)
                beta = z * se if z is not None and se is not None and math.isfinite(z) else None
                yield _metric_row(orientation, ref, alt, _parse_af(af_str), beta, se)
        finally:
            cells_stream.close()

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
                    orientation = _oriented_variant(
                        self.liftover, chrom_raw, pos_str, alt, ref
                    )
                    if orientation is None:
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
