"""Store validation: `validate_store` runs the Validation Rule registry (ADR 0074).

The rules live in `validation/rules/`, one module to a subject; their names, in
the order they run, are spec §20.0's. Beside the registry here are the rules that
stay with code of this module: the manifest, the Dense Component opening (both
through `_load_manifest`), the Indexed Variant Subsets, and source fidelity,
whose read is one function (#56).
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from opengwasdb.associations import Cells, DenseGridReader
from opengwasdb.associations import present as present_cells
from opengwasdb.encoding import (
    StoreCodec,
)
from opengwasdb.encoding.codec import se_float16
from opengwasdb.layouts.build_manifest import apply_se_divisor
from opengwasdb.layouts.dense.indexed_subsets import validate_indexed_subsets
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.model.analyses import (
    read_analyses,
)
from opengwasdb.model.enums import (
    PrimaryStorageLayout,
)
from opengwasdb.store.open import (
    OpenGWASDBStore,
    UnsupportedFormatVersion,
    indels_reference_first,
    open_store,
)
from opengwasdb.top_hits import threshold_key
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Kind,
    Rule,
    Severity,
    Target,
    ValidationOptions,
    ValidationResult,
    check_registry,
)
from opengwasdb.validation.rules import analyses as analyses_rules
from opengwasdb.validation.rules import associations as association_rules
from opengwasdb.validation.rules import by_variant as by_variant_rules
from opengwasdb.validation.rules import completion as completion_rules
from opengwasdb.validation.rules import csr as csr_rules
from opengwasdb.validation.rules import encoding as encoding_rules
from opengwasdb.validation.rules import hybrid as hybrid_rules
from opengwasdb.validation.rules import layout as layout_rules
from opengwasdb.validation.rules import release as release_rules
from opengwasdb.validation.rules import rho as rho_rules
from opengwasdb.validation.rules import sample_size as sample_size_rules
from opengwasdb.validation.rules import sequence as sequence_rules
from opengwasdb.validation.rules import side_tables as side_table_rules
from opengwasdb.validation.rules import top_hits as top_hit_rules
from opengwasdb.validation.rules import variants as variant_rules
from opengwasdb.validation.rules.associations import grid_counts
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.run import evaluate, selected
from opengwasdb.variants import (
    VariantAxis,
    variant_table_path,
)


def validate_store(
    path: str | Path,
    source: str | Path | Sequence[str | Path] | None = None,
    *,
    fidelity_samples: int = 2000,
    fidelity_seed: int = 0,
    source_assembly: str | None = None,
    chain_file: str | Path | None = None,
    rules: Collection[str] | None = None,
) -> ValidationResult:
    """Validate a Store Release directory by running every Validation Rule that applies.

    Internal (structural) validation is always run.  When ``source`` is given —
    the original dataset the store was built from — a **source-fidelity** check
    is also run for dense stores: a random sample of source associations is
    compared against the store, orienting through the stored source-ALID
    provenance (or, for lifted stores, via liftover).

    ``rules`` runs only the named rules and the rules they run after; a name no
    rule has is refused before anything is read.
    """
    if rules is not None:
        selected(RULES, rules)
    options = ValidationOptions(
        source, fidelity_samples, fidelity_seed, source_assembly, chain_file
    )
    errors: list[str] = []
    store = _load_manifest(Path(path), errors)
    if store is None:
        return ValidationResult(
            findings=tuple(Finding(MANIFEST_RULE, Severity.ERROR, e, ".") for e in errors)
        )
    return evaluate(RULES, release_targets(store, options), rules)


def release_targets(store: OpenGWASDBStore, options: ValidationOptions) -> list[Target]:
    """The targets one Store Release is judged as.

    A Dense release is a release and its grid; a Ragged one, a release and its
    association sequence. A Hybrid release is its outer release, its Ragged
    Overflow, and its Dense Component, which is a nested Store Release (spec §16)
    judged as one, by the same rules, when it opens as a Dense release; when it
    does not, the Hybrid rules say why and nothing is judged against it.
    """
    release = Target(Kind.RELEASE, ".", store, options=options)
    layout = store.manifest.primary_layout
    if layout is PrimaryStorageLayout.DENSE:
        return [release, Target(Kind.GRID, "data.zarr", store, parent=release, options=options)]
    sequence = Target(Kind.CSR, "data.zarr/ragged", store, parent=release, options=options)
    if layout is PrimaryStorageLayout.RAGGED:
        return [release, sequence]
    targets = [release, sequence]
    component = _load_manifest(dense_component_path(store.path), [])
    if component is not None and component.manifest.primary_layout is PrimaryStorageLayout.DENSE:
        nested = Target(Kind.RELEASE, "dense", component, parent=release, options=options)
        grid = Target(Kind.GRID, "dense/data.zarr", component, parent=nested, options=options)
        targets += [nested, grid]
    return targets


# --- the registry, and the rules that live beside it ---

#: The rule a release that cannot be opened fails: nothing else can be targeted.
MANIFEST_RULE = "manifest.loads"


def _not_dense(target: Target) -> bool:
    return target.layout is not PrimaryStorageLayout.DENSE


def _source_given(target: Target) -> bool:
    return target.options.source is not None


def _component_opens(target: Target) -> Iterator[str]:
    """A Hybrid release's Dense Component opens, as a Dense release (ADR 0026)."""
    errors: list[str] = []
    dense_store = _load_manifest(dense_component_path(target.store.path), errors)
    yield from errors
    if dense_store is not None and dense_store.manifest.primary_layout is not (
        PrimaryStorageLayout.DENSE
    ):
        yield (
            f"Dense Component manifest must be primary_layout=dense "
            f"(got {dense_store.manifest.primary_layout.value})"
        )


def _indexed_subsets(target: Target) -> Iterator[str]:
    """Every published Indexed Variant Subset is self-consistent and value-equal.

    The rules are the ones the build's publication gate runs on a subset before
    publishing it, there over a deterministic sample and here over every cell
    (#270).
    """
    errors: list[str] = []
    store = target.store
    n_variants, n_analyses = grid_counts(target)
    # The optional indexed-subset namespace (ADR 0061, #264).  A release without it
    # stays valid and unchanged; a release with it must hold only published,
    # self-consistent, value-equal subsets.
    validate_indexed_subsets(
        store.path, target.group, store.manifest, n_variants, n_analyses, errors
    )
    yield from errors


def _top_level_dense_grid(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.DENSE and not target.is_component


def _fidelity_unsupported(_target: Target) -> list[str]:
    return ["source-fidelity check is only supported for dense stores"]


def _source_fidelity(target: Target) -> list[str]:
    errors: list[str] = []
    options = target.options
    assert options.source is not None
    _validate_source_fidelity(
        target.store,
        options.source,
        errors,
        n_samples=options.fidelity_samples,
        seed=options.fidelity_seed,
        source_assembly=options.source_assembly,
        chain_file=options.chain_file,
    )
    return errors


#: The rules before source fidelity, in order: it samples what a Store holds, so it
#: runs only on a release none of them found wrong.
_BEFORE_FIDELITY: tuple[Rule, ...] = (
    Check(name=MANIFEST_RULE, kinds=Kind.RELEASE, run=lambda _target: ()),
    *release_rules.RULES,
    *variant_rules.RULES,
    *analyses_rules.TABLE_RULES,
    *csr_rules.RULES,
    *hybrid_rules.RULES,
    Check(
        name=hybrid_rules.COMPONENT,
        kinds=Kind.RELEASE,
        when=hybrid_rules.is_hybrid,
        after=(REQUIRED,),
        run=_component_opens,
    ),
    *hybrid_rules.COMPONENT_RULES,
    *encoding_rules.RULES,
    *sample_size_rules.CHECKS,
    *by_variant_rules.CHECKS,
    *association_rules.CHECKS,
    *completion_rules.CHECKS,
    *sequence_rules.RULES,
    by_variant_rules.COUNTS,
    *association_rules.PRESENCE_RULES,
    *completion_rules.RULES,
    *association_rules.EAF_RULES,
    *side_table_rules.RULES,
    *sample_size_rules.RULES,
    *top_hit_rules.STRUCTURE_RULES,
    *top_hit_rules.VALUE_RULES,
    by_variant_rules.SAMPLE,
    *completion_rules.METADATA_RULES,
    *rho_rules.RULES,
    *layout_rules.RULES,
    Check(name="subsets.indexed", kinds=Kind.GRID, after=(REQUIRED,), run=_indexed_subsets),
    *analyses_rules.ORIENTATION_RULES,
)

RULES: tuple[Rule, ...] = check_registry(
    [
        *_BEFORE_FIDELITY,
        Check(
            name="fidelity.unsupported",
            kinds=Kind.RELEASE,
            when=lambda target: _source_given(target) and _not_dense(target),
            run=_fidelity_unsupported,
        ),
        Check(
            name="fidelity.source",
            kinds=Kind.GRID,
            when=lambda target: _source_given(target) and _top_level_dense_grid(target),
            # Source fidelity has always run only on a release with no error, so only a
            # failed prerequisite stops it. A rule skipped for having nothing to judge,
            # such as a range check on a component holding no frequencies, is no error.
            after=tuple(r.name for r in _BEFORE_FIDELITY if r.severity is Severity.ERROR),
            only_failures_block=True,
            run=_source_fidelity,
        ),
    ]
)


def _load_manifest(store_path: Path, errors: list[str]) -> OpenGWASDBStore | None:
    try:
        return open_store(store_path)
    except UnsupportedFormatVersion as exc:
        errors.append(str(exc))
    except KeyError as exc:
        errors.append(f"manifest missing required field: {exc.args[0]}")
    except ValueError as exc:
        errors.append(f"manifest has invalid enum value: {exc}")
    except FileNotFoundError:
        errors.append("missing manifest.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manifest is malformed: {exc}")
    return None




# ── Source-fidelity check ────────────────────────────────────────────────────
# Confirm the store faithfully persisted the *values* of the original dataset —
# the one class of bug internal validation structurally cannot catch (a store
# that is self-consistent but systematically wrong: sign flip, allele
# orientation, z/se swap, wrong analysis column, liftover mis-map, quantisation
# corruption). We reuse the build's own readers, so this is a round-trip
# persistence check, not an independent oracle of the normalisation maths.

_FIDELITY_MAX_FILES = 25
_FIDELITY_SCAN_PER_FILE = 100_000
# A sampled cell is compared against the source value put through the store's
# own codec -- the same quantisation the store applied -- with a tolerance
# loose enough for that rounding but far tighter than any sign/scale/column bug.
_FIDELITY_RTOL = 1e-2
_FIDELITY_ATOL = 1e-2
# One source association in canonical form as the build readers yield it:
# (analysis_id, chrom, pos, a1, a2, z, se) -- and one sampled-and-matched store
# cell carrying its source z/se plus the source ALID for diagnostics.
_FidelitySample = tuple[str, str, int, str, str, float, float]
_FidelityCell = tuple[int, int, float, float, str]


def _normalise_assembly(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"grch37", "hg19", "b37", "37"}:
        return "hg19"
    if n in {"grch38", "hg38", "b38", "38"}:
        return "hg38"
    return n


def _is_vcf_path(path: Path) -> bool:
    s = str(path).lower()
    return s.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf"))


class _Unit(NamedTuple):
    """One source file: its Analysis (``None`` where nothing names it), whether it
    is a VCF, and the SD its manifest row declares (1 where it declares none)."""

    analysis_id: str | None
    path: Path
    is_vcf: bool
    se_divisor: float = 1.0


def _source_units(
    source: str | Path | Sequence[str | Path], errors: list[str]
) -> list[_Unit]:
    """Resolve ``source`` into (analysis_id_or_None, path, is_vcf) units.

    A single TSV whose header carries an analysis id and a source file column is
    treated as a build manifest (one analysis per file); any other path is a
    source file that names no Analysis.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            errors.append(f"source path does not exist: {p}")
            return []
        try:
            manifest = _manifest_units(p)
        except ValueError as exc:
            errors.append(f"source-fidelity: {exc}")
            return []
        return manifest if manifest is not None else _unnamed_units([p], errors)
    return _unnamed_units([Path(item) for item in source], errors)


def _header_columns(path: Path) -> list[str]:
    """The tab-separated header of a plain-text file, or none if it is not one."""
    if _is_vcf_path(path) or str(path).lower().endswith(".gz"):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.readline().rstrip("\r\n").split("\t")
    except (OSError, UnicodeDecodeError):
        return []


def _manifest_columns(path: Path) -> tuple[str, str] | None:
    """The (analysis id, source file) columns of a build manifest's header, or
    ``None`` if ``path`` is not one."""
    columns = _header_columns(path)
    id_column = next((c for c in ("analysis_id", "trait_id") if c in columns), None)
    file_column = next((c for c in ("source_file", "file_path") if c in columns), None)
    return None if id_column is None or file_column is None else (id_column, file_column)


def _manifest_units(path: Path) -> list[_Unit] | None:
    """The units of a build manifest, or ``None`` if ``path`` is not one.

    Each carries the SD its row declares, which the build divided the Analysis's
    standard errors by (#18), so the source is compared as the build stored it.
    """
    from opengwasdb.layouts.build_manifest import read_manifest

    if _manifest_columns(path) is None:
        return None
    return [
        _Unit(row.trait_id, Path(row.file_path), _is_vcf_path(Path(row.file_path)), row.se_divisor)
        for row in read_manifest(path)
    ]


def _unnamed_units(
    paths: Sequence[Path], errors: list[str]
) -> list[_Unit]:
    """Units for sources no manifest names. An association TSV holds one Analysis
    and does not say which, so it is refused rather than sampled under none."""
    for path in paths:
        if not _is_vcf_path(path):
            errors.append(
                f"source-fidelity: {path} is an association TSV, which names no Analysis; "
                "give the build manifest that lists it"
            )
    return [_Unit(None, path, _is_vcf_path(path)) for path in paths]


def _stream_unit(
    unit: _Unit, *, reference_first: bool
) -> Iterator[tuple[str, str, int, str, str, float, float]]:
    """Yield canonical (analysis_id, chrom, pos, a1, a2, z, se) from one source.

    ``reference_first`` is whether the Store spells an indel's ALID reference
    first (`indels_reference_first`, ADR 0079). A Store before it sorts an
    indel's alleles, so a deletion is spelt, and its z signed, as that Store
    spells it: alleles sorted, z negated with them.
    """
    from opengwasdb.model.enums import StoredEffectScale
    from opengwasdb.readers.association_tsv import ASSOCIATION_TSV_CAPABILITY
    from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
    from opengwasdb.readers.interface import StreamTally
    from opengwasdb.readers.registry import resolve_reader
    from opengwasdb.variants.normalise import canonical_alid

    analysis_id, path, is_vcf, se_divisor = unit
    if analysis_id is None:
        return  # a source that names no Analysis has been refused by `_source_units`
    # The scale is never read: this check compares stored z/se against the
    # source, and neither the scale nor the EAF plays a part in that.
    capability = GWAS_VCF_CAPABILITY if is_vcf else ASSOCIATION_TSV_CAPABILITY
    reader = resolve_reader(capability, path, StoredEffectScale.SD)
    # Read as the build reads it: a row it dropped as malformed is not a failure.
    for association in reader.stream_associations(StreamTally()):
        _chrom, _pos, a1, a2 = canonical_alid(
            association.chromosome, association.position, association.ref, association.alt
        ).split(":")
        z = association.z
        if not reference_first and a2 < a1:
            a1, a2, z = a2, a1, -z
        se = association.se
        if se_divisor != 1.0:
            se = float(apply_se_divisor(np.array([se], dtype=np.float32), se_divisor)[0])
        yield (analysis_id, association.chromosome, association.position, a1, a2, z, se)


def _sample_sources(
    units: list[_Unit], n_samples: int, rng: np.random.Generator, *, reference_first: bool
) -> list[tuple[str, str, int, str, str, float, float]]:
    """Reservoir-sample up to ``n_samples`` associations across a random subset
    of source files (scan-capped per file so cost stays bounded)."""
    if not units:
        return []
    order = rng.permutation(len(units))[:_FIDELITY_MAX_FILES]
    reservoir: list[tuple[str, str, int, str, str, float, float]] = []
    seen = 0
    for u in order:
        scanned = 0
        for item in _stream_unit(units[int(u)], reference_first=reference_first):
            seen += 1
            scanned += 1
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                j = int(rng.integers(0, seen))
                if j < n_samples:
                    reservoir[j] = item
            if scanned >= _FIDELITY_SCAN_PER_FILE:
                break
    return reservoir


def _resolve_via_origin(store_path: Path, wanted: set[str]) -> dict[str, int] | None:
    """Map wanted source ALIDs → variant_index via the stored source_alid column
    in one streamed pass. Returns None if the store predates that column."""
    import pysam

    out: dict[str, int] = {}
    remaining = set(wanted)
    with pysam.BGZFile(str(variant_table_path(store_path)), "r") as handle:  # type: ignore[call-arg]
        for raw in handle:
            line = raw.decode("utf-8")
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                return None  # no source_alid column → caller falls back to liftover
            src = fields[7]
            if src in remaining:
                out[src] = int(fields[2])
                remaining.discard(src)
                if not remaining:
                    break
    return out


def _draw_fidelity_samples(
    source: str | Path | Sequence[str | Path],
    errors: list[str],
    *,
    n_samples: int,
    seed: int,
    reference_first: bool,
) -> list[_FidelitySample] | None:
    """Sampling phase: deterministically reservoir-sample source associations.

    ``source`` is resolved into per-file units, then up to ``n_samples``
    associations are drawn with a generator seeded by ``seed``, so repeated
    validation of the same store reproduces the same sample. Returns None --
    having already recorded why in ``errors`` -- when the source is unusable
    (missing path) or no associations could be read from it.
    """
    units = _source_units(source, errors)
    if not units and not errors:
        errors.append("source-fidelity: the source names no Analysis to check the store against")
    if errors or not units:
        return None
    samples = _sample_sources(
        units, n_samples, np.random.default_rng(seed), reference_first=reference_first
    )
    if not samples:
        errors.append("source-fidelity: no associations could be read from the source")
        return None
    return samples


def _lookup_variant_rows(store_path: Path, alid_to_identifier: dict[str, str]) -> dict[str, int]:
    """Resolve source ALIDs → variant_index through the Variant Index.

    ``alid_to_identifier`` maps each source ALID to the identifier to look up --
    its own canonical ALID for a same-assembly join, the lifted HG38 ALID for a
    cross-assembly one.
    """
    row_by_alid: dict[str, int] = {}
    va = VariantAxis(store_path)
    try:
        for src_alid, identifier in alid_to_identifier.items():
            rec = va.by_identifier(identifier)
            if rec is not None:
                row_by_alid[src_alid] = rec.variant_index
    finally:
        va.close()
    return row_by_alid


def _resolve_fidelity_rows(
    store_path: Path,
    samples: list[_FidelitySample],
    *,
    src_asm: str,
    store_asm: str,
    chain_file: str | Path | None,
) -> dict[str, int]:
    """Row-resolution phase: map each sampled source variant to a store row.

    Assembly-dependent: same-assembly samples join through the canonical ALID;
    cross-assembly samples first use the stored source-ALID provenance column,
    and fall back to liftover only when the store predates that column.
    """
    src_alids = {f"{c}:{p}:{a1}:{a2}" for (_a, c, p, a1, a2, _z, _s) in samples}
    if src_asm == store_asm:
        return _lookup_variant_rows(store_path, {a: a for a in src_alids})
    origin_map = _resolve_via_origin(store_path, src_alids)
    if origin_map is not None:
        return origin_map  # store carries provenance → no liftover needed
    from opengwasdb.build.liftover import build_liftover_lookup

    tuples = {(c, p, a1, a2) for (_a, c, p, a1, a2, _z, _s) in samples}
    lut = build_liftover_lookup(
        tuples,
        from_build=src_asm,
        to_build=store_asm,
        failure_threshold=1.0,
        chain_file=chain_file,
    )
    wanted = {
        f"{c}:{p}:{a1}:{a2}": hg38
        for c, p, a1, a2 in tuples
        if (hg38 := lut.get((c, p, a1, a2))) is not None
    }
    return _lookup_variant_rows(store_path, wanted)


def _match_fidelity_pairs(
    store_path: Path,
    samples: list[_FidelitySample],
    row_by_alid: dict[str, int],
    errors: list[str],
) -> list[_FidelityCell] | None:
    """Analysis-matching phase: join each sample onto a (row, column) cell.

    A sample whose Analysis the store does not hold, or whose variant row it
    does not carry, is counted rather than silently dropped; when nothing
    matches, one diagnostic names both counts so a wrong source and a wrong
    assembly stay distinguishable. Returns None after recording that error.
    """
    col_by_aid = {
        row["analysis_id"]: int(row["analysis_index"])
        for row in read_analyses(store_path / "analyses.tsv").rows
    }
    pairs: list[_FidelityCell] = []
    n_unmatched_variant = 0
    n_unmatched_analysis = 0
    for aid, c, p, a1, a2, z, se in samples:
        col = col_by_aid.get(aid)
        if col is None:
            n_unmatched_analysis += 1
            continue
        row = row_by_alid.get(f"{c}:{p}:{a1}:{a2}")
        if row is None:
            n_unmatched_variant += 1
            continue
        pairs.append((row, col, z, se, f"{c}:{p}:{a1}:{a2}"))
    if not pairs:
        errors.append(
            f"source-fidelity: none of {len(samples)} sampled source associations could be "
            f"matched to a store cell ({n_unmatched_variant} variants, {n_unmatched_analysis} "
            f"analyses unmatched) — check source_assembly / that this is the right source"
        )
        return None
    return pairs


def _fidelity_blocks(
    store: OpenGWASDBStore, pairs: list[_FidelityCell]
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """Read the matched cells' z/se as one bounded block plus index maps.

    Returns ``(z_block, se_block, row_index, col_index)`` where the index maps
    take each pair's variant/analysis index to its position in the block.
    """
    urows = sorted({pr[0] for pr in pairs})
    ucols = sorted({pr[1] for pr in pairs})
    ri = {r: i for i, r in enumerate(urows)}
    ci = {c: i for i, c in enumerate(ucols)}
    grid = DenseGridReader.open(
        store.arrays(mode="r"), store.manifest.encoding, source=f"Dense Store Release {store.path}"
    )
    # A cross region comes Analysis-major: one run of the rows per column.
    found = grid.inspect(Cells.cross(urows, ucols))
    shape = (len(ucols), len(urows))
    return found.z.reshape(shape).T, found.se.reshape(shape).T, ri, ci


def _compare_fidelity_cells(
    store: OpenGWASDBStore, pairs: list[_FidelityCell]
) -> tuple[int, list[str]]:
    """Comparison phase: compare each matched cell against its source value.

    The store's own codec quantises the source value before comparison, and a
    cell whose store value is missing is skipped, not compared: a store holding
    *no* finite value at any matched cell is a different failure (dropped
    associations), reported by the caller.
    """
    z_blk, se_blk, ri, ci = _fidelity_blocks(store, pairs)
    codec = StoreCodec(store.manifest.encoding)

    n_compared = 0
    mismatches: list[str] = []
    for row, col, z_src, se_src, alid in pairs:
        sz = z_blk[ri[row], ci[col]]
        sse = se_blk[ri[row], ci[col]]
        if not present_cells(np.array([sz]), np.array([sse]))[0]:
            continue
        n_compared += 1
        z_ref = float(codec.quantise_z(np.array([z_src]))[0])
        se_ref = float(se_float16(np.array([se_src]))[0])
        if not (
            np.isclose(sz, z_ref, rtol=_FIDELITY_RTOL, atol=_FIDELITY_ATOL)
            and np.isclose(sse, se_ref, rtol=_FIDELITY_RTOL, atol=_FIDELITY_ATOL)
        ):
            if len(mismatches) < 5:
                mismatches.append(
                    f"{alid}: store z={sz:.4f} se={sse:.4f} vs source z={z_src:.4f} se={se_src:.4f}"
                )
    return n_compared, mismatches


def _report_fidelity_disagreements(
    errors: list[str], pairs: list[_FidelityCell], n_compared: int, mismatches: list[str]
) -> None:
    """Reporting phase: turn the comparison outcome into a hard error.

    Disagreements name the store/source pair; matched cells where the store
    holds no finite value at all are reported as possible dropped associations.
    """
    if mismatches:
        errors.append(
            f"source-fidelity: {len(mismatches)}{'+' if len(mismatches) == 5 else ''} of "
            f"{n_compared} compared associations disagree with the source (e.g. {mismatches[0]})"
        )
    elif n_compared == 0:
        errors.append(
            f"source-fidelity: matched {len(pairs)} source associations but the store held no "
            f"finite value at any of those cells (possible dropped associations)"
        )


def _validate_source_fidelity(
    store: OpenGWASDBStore,
    source: str | Path | Sequence[str | Path],
    errors: list[str],
    *,
    n_samples: int,
    seed: int,
    source_assembly: str | None,
    chain_file: str | Path | None,
) -> None:
    """Cross-check the source's z/se against the store (issue #130).

    A thin orchestrator over five phases -- deterministic sampling, row
    resolution (assembly-dependent), Analysis matching, block comparison, and
    mismatch reporting -- each of which guards the read in front of it.
    """
    samples = _draw_fidelity_samples(
        source,
        errors,
        n_samples=n_samples,
        seed=seed,
        reference_first=indels_reference_first(store.manifest.format_version),
    )
    if samples is None:
        return
    store_asm = _normalise_assembly(store.manifest.reference_assembly)
    src_asm = _normalise_assembly(source_assembly) if source_assembly else store_asm
    row_by_alid = _resolve_fidelity_rows(
        store.path, samples, src_asm=src_asm, store_asm=store_asm, chain_file=chain_file
    )
    pairs = _match_fidelity_pairs(store.path, samples, row_by_alid, errors)
    if pairs is None:
        return
    n_compared, mismatches = _compare_fidelity_cells(store, pairs)
    _report_fidelity_disagreements(errors, pairs, n_compared, mismatches)

def default_top_hit_key(threshold: float = 5e-8) -> str:
    return threshold_key(threshold)
