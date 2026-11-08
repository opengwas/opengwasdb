"""GWAS-VCF SourceReader (issue #19; updated for #17, #20, #21, #278, #56).

One GWAS-VCF holds one Analysis in its one sample column. Every projection but
the variant one reads the file through one streaming ``bcftools query`` --
header-probed for its FORMAT tags and sample count, and exit-checked -- whose
records become `SourceRow`s (`_record_rows`), presented by
`opengwasdb.readers.rows`: the associations, the resolver's metric blocks and
the sites ancestry assignment and phenotype-SD estimation read. ALT is the
effect allele; a record's z is its `EZ` where it reports one, else `ES/SE`.

`stored_effect_scale` is supplied at construction rather than read from the
VCF: the source header is not authoritative for effect scale (issue #17, the
ieu-a-7 fix), and nothing here reads ``##SAMPLE``'s `StudyType`.
`stream_variants` keeps a faster query of the identity columns alone, the
superset the dense and hybrid builders' union-variant pass needs (issue #20).

`write_regions_file` and the liftover helpers live here, since
they are GWAS-VCF/bcftools-specific, not ancestry-specific.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Generator, Iterable, Iterator, KeysView, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TextIO

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import interface
from opengwasdb.readers.interface import (
    ReaderAssociation,
    ReferenceAllele,
    SiteMetrics,
    SourceVariant,
    StreamTally,
)
from opengwasdb.readers.orientation import is_palindromic
from opengwasdb.readers.rows import (
    MetricsChunk,
    ReaderBatch,
    SourceRow,
    extract_at_sites,
    metrics_chunks_from_rows,
    stream_associations,
    stream_batches,
)
from opengwasdb.stats import FrequencyTally, parse_af
from opengwasdb.variants.normalise import (
    IdentityFaults,
    MalformedIdentity,
    Orientation,
    VariantNormalisationError,
    allele_fault,
    identity_allele,
    is_primary_contig,
    normalise_chromosome,
    orient_to_canonical,
    oriented_alid,
)

GWAS_VCF_CAPABILITY = "opengwasdb.gwas-vcf"
#: GWAS-VCF's REF is the genome's reference allele, on every record.
_REFERENCE = ReferenceAllele.REF


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


def project_lifted_alleles(
    strand: str, effect: str, other: str, identity: str = "the lifted variant"
) -> tuple[str, str]:
    """The `(effect, other)` alleles as they read on the target assembly.

    A chain block's orientation says which strand of the target a source region
    landed on, so a reverse-strand mapping's alleles are complemented before any
    identity is formed from them. Every lift in the package projects through
    this one rule -- the resolver's `oriented_variant` (issue #282) and the
    builders' `opengwasdb.build.liftover` (issue #284) -- because two lifts that
    each complement on their own are two answers to the same question, and the
    builder's answer was missing the complement entirely.

    `identity` names the variant in the refusal of an unrecognised strand: the
    resolver passes its own `chrom:pos:effect:other` spelling, so a bad chain
    fails naming the row it was reading.
    """
    if strand == "+":
        return effect, other
    if strand == "-":
        return effect.translate(_COMPLEMENT), other.translate(_COMPLEMENT)
    raise LiftoverProjectionError(f"unsupported liftover strand {strand!r} for {identity}")


def _lift_variant(
    liftover: object, chrom: str, pos_str: str, effect: str, other: str
) -> tuple[str, str, str, str] | None:
    """Lift one SNP and project its alleles onto one unambiguous mapping strand.

    ``None`` when the lift places it nowhere, or on an alt, random or unplaced
    contig: a build drops such a row (#376), and so does this."""
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
    effect, other = project_lifted_alleles(strand, effect, other, identity)
    bare_target = new_chrom[3:] if new_chrom.lower().startswith("chr") else new_chrom
    if not is_primary_contig(bare_target):
        return None
    return bare_target, str(int(new_pos) + 1), effect, other


def oriented_variant(
    liftover: object | None,
    chrom: str,
    pos: str,
    effect: str,
    other: str,
    *,
    other_is_reference: bool,
) -> Orientation | None:
    """One source row's canonical orientation, lifted first when `liftover` is set.

    `None` when the row names no usable variant: unnormalisable alleles, or a
    lift that did not land. An indel or an ambiguous mapping under a lift
    raises `LiftoverProjectionError` instead, because no single position lift
    can establish its target identity (ADR 0044). Shared with the BESD resolver
    (issue #282), which aligns `.esi` rows by the same rule.
    """
    if liftover is not None:
        lifted = _lift_variant(liftover, chrom, pos, effect, other)
        if lifted is None:
            return None
        chrom, pos, effect, other = lifted
    try:
        return orient_to_canonical(
            chrom, pos, effect, other, other_is_reference=other_is_reference
        )
    except VariantNormalisationError:
        return None


def _parse_positive_float(value: str) -> float | None:
    if value in {".", ""}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0.0 and math.isfinite(parsed) else None




def _require_bcftools() -> str:
    path = shutil.which("bcftools")
    if path is None:
        raise RuntimeError(
            "bcftools not found in PATH — install via conda: conda install -c bioconda bcftools"
        )
    return path


def _format_tags(path: str | Path) -> set[str]:
    """Read FORMAT capability once; this is not a source-record traversal.

    A GWAS-VCF holds one Analysis in one sample column, so a file with any
    other number of samples is refused before a record is read.
    """
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
            f"{path}: a GWAS-VCF source requires exactly one sample column, found {sample_count}"
        )
    prefix = "##FORMAT=<ID="
    return {
        line[len(prefix) :].split(",", 1)[0]
        for line in lines
        if line.startswith(prefix)
    }


#: The FORMAT tags a record row reads, in query order. A tag the header does
#: not declare is queried as `.`, a missing value, because bcftools fails the
#: whole query when asked for an undeclared tag (ADR 0036: AF is optional).
_RECORD_TAGS = ("EZ", "ES", "SE", "AF")
_RECORD_FIELDS = 9


def _record_query_format(tags: set[str]) -> str:
    fields = ["%CHROM", "%POS", "%ID", "%REF", "%ALT"]
    fields.extend(f"[%{tag}]" if tag in tags else "." for tag in _RECORD_TAGS)
    return "\t".join(fields) + "\n"


def _validated_cells(lines: Iterable[str], n_fields: int) -> Iterator[list[str]]:
    for row_number, line in enumerate(lines, start=1):
        cells = line.rstrip("\n").split("\t")
        if len(cells) != n_fields:
            raise ValueError(
                f"malformed bcftools output at record {row_number}: "
                f"expected {n_fields} fields, got {len(cells)}"
            )
        yield cells


_QUERY_STOP_TIMEOUT_SECONDS = 5.0


def _bounded_wait(proc: subprocess.Popen[str]) -> tuple[int, bool]:
    try:
        return proc.wait(timeout=_QUERY_STOP_TIMEOUT_SECONDS), False
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        return proc.wait(timeout=_QUERY_STOP_TIMEOUT_SECONDS), True
    except subprocess.TimeoutExpired as exc:
        raise OSError("bcftools query could not be reaped after kill") from exc


def _raise_query_failure(returncode: int, errors: TextIO) -> None:
    errors.seek(0)
    message = errors.read().strip() or "no diagnostic"
    raise OSError(f"bcftools record query failed ({returncode}): {message}")


def _finish_query(proc: subprocess.Popen[str], errors: TextIO, completed: bool) -> None:
    returncode = proc.poll()
    cancelled = not completed and returncode is None
    if cancelled:
        proc.terminate()
    if proc.stdout is not None:
        proc.stdout.close()
    returncode, killed = _bounded_wait(proc)
    expected_signals = {-signal.SIGTERM}
    if killed:
        expected_signals.add(-signal.SIGKILL)
    if returncode != 0 and not (cancelled and returncode in expected_signals):
        _raise_query_failure(returncode, errors)


def _query(
    path: str | Path, query_format: str, n_fields: int, restriction: Sequence[str] = ()
) -> Generator[list[str], None, None]:
    """Every record's cells from one streaming ``bcftools query``, exit-checked.

    A query that fails, or prints a record with the wrong number of fields,
    raises rather than yield a short stream (issue #278). ``restriction`` is
    bcftools' ``-r`` or ``-R`` argument pair, for a site extraction.
    """
    bcftools = _require_bcftools()
    completed = False
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        proc = subprocess.Popen(
            [bcftools, "query", "-f", query_format, *restriction, str(path)],
            stdout=subprocess.PIPE, stderr=errors, text=True,
        )
        try:
            assert proc.stdout is not None
            yield from _validated_cells(proc.stdout, n_fields)
            completed = True
        finally:
            _finish_query(proc, errors, completed)


#: A one-base allele in either case, which nearly every record holds.
_ONE_BASE = frozenset("ACGTacgt")


def _identity(
    chromosomes: dict[str, str], chrom_raw: str, ref: str, alt: str
) -> str | MalformedIdentity:
    """A record's bare chromosome, or why its identity is malformed (#365).

    ``chromosomes`` caches each spelling's normalised name, so a record pays a
    dictionary lookup and two set lookups. A chromosome refused for another
    reason (an unsupported number) still raises.
    """
    chrom = chromosomes.get(chrom_raw)
    if chrom is None:
        try:
            chrom = normalise_chromosome(chrom_raw)
        except MalformedIdentity as missing:
            return missing
        chromosomes[chrom_raw] = chrom
    if ref in _ONE_BASE and alt in _ONE_BASE:
        return chrom
    invalid = allele_fault(ref, alt)
    return chrom if invalid is None else invalid


def _oriented(chrom: str, pos: str, effect: str, other: str) -> tuple[int, str, bool] | None:
    """``(position, alid, flipped)`` of a record whose identity is judged, or ``None``
    for an unparseable position or one allele spelt twice."""
    position = _position(pos)
    if position is None:
        return None
    oriented = oriented_alid(chrom, position, effect, other, other_is_reference=True)
    return None if oriented is None else (position, *oriented)


def _position(cell: str) -> int | None:
    try:
        position = int(cell)
    except ValueError:
        return None
    return position if position > 0 else None


#: How bcftools prints a missing value, and an empty cell.
_MISSING_CELLS = frozenset({".", ""})


def _parse_float(s: str) -> float | None:
    if s in {".", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _derive_z(ez_str: str, es_str: str, se: float | None) -> float | None:
    """A record's z: its EZ when it reports one, else ES/SE; ``None`` for no effect.

    An infinite EZ or ES is a value the record reports, not a missing one, so
    the z stays infinite, for Association Admission to drop as `z_nonfinite`
    (D4), rather than ES/SE standing in for an infinite EZ or an infinite ES
    reading as no effect. A NaN reports nothing. ``se`` is the record's usable
    SE, or ``None``.
    """
    if ez_str not in _MISSING_CELLS:
        try:
            ez = float(ez_str)
        except ValueError:
            ez = math.nan
        if ez == ez:  # not NaN
            return ez
    es = _parse_float(es_str)
    if es is not None and not math.isnan(es) and se is not None:
        return es / se
    return None


def _row(
    chromosome: str, position: int, ref: str, alt: str, alid: str, flipped: bool,
    af: float | None, se: float | None, z: float | None, id_field: str,
) -> SourceRow:
    """A record's `SourceRow`, built without its frozen ``__init__``.

    Pass 2 reads one per record of a genome-wide file, and every value here is
    already validated, so one dictionary update gives the same row. ``beta``
    is ``z * se`` where both are finite, which is what the resolver's metrics
    read; the association reads ``z`` itself, so a reported EZ is stored as it
    is rather than rebuilt.
    """
    row = object.__new__(SourceRow)
    object.__setattr__(
        row,
        "__dict__",
        {
            "chromosome": chromosome,
            "position": position,
            "ref": ref,
            "alt": alt,
            "alid": alid,
            "flipped": flipped,
            "af_alt": af,
            "beta": z * se if z is not None and se is not None and math.isfinite(z) else None,
            "se": se,
            "z": z,
            "rsid": id_field if id_field.startswith("rs") else "",
            "source_id": "" if id_field == "." else id_field,
        },
    )
    return row


def _record_rows(
    path: str | Path,
    *,
    tally: StreamTally | None = None,
    liftover: object | None = None,
    restriction: Sequence[str] = (),
    sites: bool = False,
) -> Generator[SourceRow, None, None]:
    """Every biallelic record of one GWAS-VCF as a `SourceRow`, in file order.

    ALT is the effect allele and REF the other. A record whose identity is
    malformed is named in ``tally`` and left out (#365), as is one with an
    unparseable position or two spellings of one allele. With ``liftover``
    each row is lifted onto the reference build before its identity is formed
    (`oriented_variant`), which refuses an indel or an ambiguous mapping.
    ``sites`` is a site extraction's order: a record with no usable AF or SE,
    or a palindromic pair, is passed over before it is lifted.
    """
    tally = tally or StreamTally()
    query_format = _record_query_format(_format_tags(path))
    with contextlib.closing(_query(path, query_format, _RECORD_FIELDS, restriction)) as records:
        # A site extraction's frequencies are not the stream's to count (D7).
        frequencies = None if sites else tally.frequencies
        rows = filter(_usable_site, records) if sites else records
        if liftover is None:
            yield from _rows_of(rows, tally.faults, frequencies)
        else:
            yield from _lifted_rows_of(rows, tally.faults, frequencies, liftover)


def _rows_of(
    records: Iterable[list[str]], faults: IdentityFaults, frequencies: FrequencyTally | None
) -> Iterator[SourceRow]:
    """Each record's row in its own coordinates: Pass 2's per-record path."""
    chromosomes: dict[str, str] = {}
    for chrom_raw, pos, id_field, ref, alt, ez, es, se_cell, af_cell in records:
        if "," in alt:
            continue
        # A one-base pair on a chromosome already seen is the common record.
        chrom = chromosomes.get(chrom_raw)
        if chrom is None or ref not in _ONE_BASE or alt not in _ONE_BASE:
            identity = _identity(chromosomes, chrom_raw, ref, alt)
            if isinstance(identity, MalformedIdentity):
                faults.record(f"{chrom_raw}:{pos}:{ref}:{alt}", identity)
                continue
            chrom = identity
        oriented = _oriented(chrom, pos, alt, ref)
        if oriented is None:
            continue
        se = _parse_positive_float(se_cell)
        yield _row(
            chrom, oriented[0], ref, alt, oriented[1], oriented[2],
            parse_af(af_cell, frequencies), se, _derive_z(ez, es, se), id_field,
        )


def _lifted_rows_of(
    records: Iterable[list[str]],
    faults: IdentityFaults,
    frequencies: FrequencyTally | None,
    liftover: object,
) -> Iterator[SourceRow]:
    """Each record's row lifted onto the reference build (`oriented_variant`)."""
    chromosomes: dict[str, str] = {}
    for chrom_raw, pos, id_field, ref, alt, ez, es, se_cell, af_cell in records:
        if "," in alt:
            continue
        identity = _identity(chromosomes, chrom_raw, ref, alt)
        if isinstance(identity, MalformedIdentity):
            faults.record(f"{chrom_raw}:{pos}:{ref}:{alt}", identity)
            continue
        orientation = oriented_variant(liftover, chrom_raw, pos, alt, ref, other_is_reference=True)
        if orientation is None:
            continue
        variant = orientation.variant
        se = _parse_positive_float(se_cell)
        yield _row(
            variant.chromosome, variant.position, ref, alt, variant.alid, orientation.flipped,
            parse_af(af_cell, frequencies), se, _derive_z(ez, es, se), id_field,
        )


def _usable_site(cells: list[str]) -> bool:
    """Whether a record can give a site its A1 frequency and SE: both usable, and
    its pair not palindromic. Judged before a record is lifted."""
    _chrom, _pos, _id, ref, alt, _ez, _es, se_cell, af_cell = cells
    return (
        _parse_positive_float(se_cell) is not None
        and parse_af(af_cell) is not None
        and not is_palindromic(ref, alt)
    )


def _variants(path: str | Path) -> Iterator[SourceVariant]:
    """`stream_variants`' fast projection: identity and ID alone, one query.

    Pass 1 reads every record of every source, so this queries the five
    identity columns rather than the FORMAT fields, and yields exactly the
    records `_record_rows` yields, held to it by a parity test.
    """
    chromosomes: dict[str, str] = {}
    _format_tags(path)
    with contextlib.closing(_query(path, "%CHROM\t%POS\t%REF\t%ALT\t%ID\n", 5)) as records:
        yield from _variants_of(records, chromosomes)


def _variants_of(
    records: Iterable[list[str]], chromosomes: dict[str, str]
) -> Iterator[SourceVariant]:
    for chrom_raw, pos, ref, alt, id_field in records:
        if "," in alt:
            continue
        chrom = _identity(chromosomes, chrom_raw, ref, alt)
        if isinstance(chrom, MalformedIdentity):
            continue  # the association stream names it; no axis holds it
        position = _position(pos)
        if position is None or identity_allele(ref) == identity_allele(alt):
            continue
        rsid = id_field if id_field.startswith("rs") else ""
        yield SourceVariant(chromosome=chrom, position=position, ref=ref, alt=alt, rsid=rsid)


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

    The associations and variants are never lifted: they are read in the
    file's own assembly, and Variant Routing lifts them.
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

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        with contextlib.closing(_record_rows(self.path, tally=tally)) as rows:
            yield from stream_associations(rows, self.stored_effect_scale, _REFERENCE)

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        with contextlib.closing(_record_rows(self.path, tally=tally)) as rows:
            yield from stream_batches(rows, _REFERENCE, chunk_rows=self.chunk_rows)

    def stream_variants(self) -> Iterator[SourceVariant]:
        yield from _variants(self.path)

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """Yield bounded resolver metrics from one streaming record query (#278).

        Every record naming a variant is kept, its statistics missing where it
        reports none, so ancestry and phenotype-SD apply their own eligibility
        rules downstream.
        """
        # Closed explicitly, so a query that failed is reported to a caller
        # that stops early (the resolver's row cap) rather than lost (#278).
        with contextlib.closing(_record_rows(self.path, liftover=self.liftover)) as rows:
            yield from metrics_chunks_from_rows(rows, chunk_rows=self.chunk_rows)

    def extract_at_sites(
        self,
        alids: Iterable[str],
        *,
        region: str | None = None,
        regions_file: interface.RegionsFile = None,
    ) -> dict[str, SiteMetrics]:
        """The requested sites, read by bcftools' own restriction.

        ``region`` and ``regions_file`` are the reader's own when not given.
        """
        wanted = (
            alids if isinstance(alids, (set, frozenset, dict, KeysView)) else set(alids)
        )
        if not wanted:
            return {}
        region = self.region if region is None else region
        regions_file = self.regions_file if regions_file is None else regions_file
        with contextlib.ExitStack() as stack:
            # bcftools rejects -r/-R together, so region (a coarser, cheaper
            # restriction) takes priority when set.
            restriction: tuple[str, ...] = ()
            if region is not None:
                restriction = ("-r", str(region))
            elif self.liftover is None:
                if regions_file is not None:
                    regions_path = Path(regions_file)
                else:
                    tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                    regions_path = write_regions_file(wanted, Path(tmp_dir) / "regions.tsv")
                restriction = ("-R", str(regions_path))
            rows = _record_rows(
                self.path, liftover=self.liftover, restriction=restriction, sites=True
            )
            return extract_at_sites(rows, wanted)
