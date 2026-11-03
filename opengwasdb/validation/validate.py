"""Store validation."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.associations import Cells, DenseGridReader
from opengwasdb.associations import present as present_cells
from opengwasdb.encoding import (
    StoreCodec,
)
from opengwasdb.layouts.dense.indexed_subsets import validate_indexed_subsets
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.model.analyses import (
    read_analyses,
)
from opengwasdb.model.enums import (
    PrimaryStorageLayout,
)
from opengwasdb.store.arrays import (
    array_length,
)
from opengwasdb.store.open import (
    OpenGWASDBStore,
    UnsupportedFormatVersion,
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


# --- the orchestrators rules replace, each run whole until its checks move out ---

#: The rule a release that cannot be opened fails: nothing else can be targeted.
MANIFEST_RULE = "manifest.loads"


@dataclass(frozen=True, kw_only=True)
class _Legacy(Rule):
    """A per-layout orchestrator from before rules had names, run as one rule.

    Each is deleted as the checks it runs move into rules of their own.
    """

    run: Callable[[Target, list[str], list[str]], None]

    def findings(self, target: Target) -> Iterator[Finding]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            self.run(target, errors, warnings)
        finally:
            # Yielded on the way out of an exception too: what the orchestrator
            # found before it failed is reported ahead of the failure.
            yield from (self.finding(target, m, Severity.ERROR) for m in errors)
            yield from (self.finding(target, m, Severity.WARNING) for m in warnings)


def _is(layout: PrimaryStorageLayout) -> Callable[[Target], bool]:
    return lambda target: target.layout is layout


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


def _legacy_rho(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_rho(target, errors)


def _legacy_subsets(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_subsets(target, errors)


def _top_level_dense_grid(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.DENSE and not target.is_component


def _fidelity_unsupported(_target: Target) -> list[str]:
    return ["source-fidelity check is only supported for dense stores"]


def _legacy_fidelity(target: Target) -> list[str]:
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
    _Legacy(name="legacy.rho", kinds=Kind.GRID, after=(REQUIRED,), run=_legacy_rho),
    *layout_rules.RULES,
    _Legacy(name="legacy.subsets", kinds=Kind.GRID, after=(REQUIRED,), run=_legacy_subsets),
    *analyses_rules.ORIENTATION_RULES,
)

RULES: tuple[Rule, ...] = check_registry(
    [
        *_BEFORE_FIDELITY,
        Check(
            name="legacy.fidelity_unsupported",
            kinds=Kind.RELEASE,
            when=lambda target: _source_given(target) and _not_dense(target),
            run=_fidelity_unsupported,
        ),
        Check(
            name="legacy.fidelity",
            kinds=Kind.GRID,
            when=lambda target: _source_given(target) and _top_level_dense_grid(target),
            after=tuple(r.name for r in _BEFORE_FIDELITY if r.severity is Severity.ERROR),
            run=_legacy_fidelity,
        ),
    ]
)


def _dense_rho(target: Target, errors: list[str]) -> None:
    """The optional Rho Matrix against `analyses.tsv`."""
    _validate_rho(target.group, grid_counts(target)[1], errors)


def _dense_subsets(target: Target, errors: list[str]) -> None:
    """Any Indexed Variant Subsets."""
    store = target.store
    n_variants, n_analyses = grid_counts(target)
    # The optional indexed-subset namespace (ADR 0061, #264).  A release without it
    # stays valid and unchanged; a release with it must hold only published,
    # self-consistent, value-equal subsets.
    validate_indexed_subsets(
        store.path, target.group, store.manifest, n_variants, n_analyses, errors
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




_RHO_REQUIRED_ATTRS = (
    "z_thresh",
    "min_nulls",
    "grid_n",
    "window_bp",
    "n_variants_used",
    "observed_only",
    "method",
    "n_analyses",
)


def _validate_rho(root: Any, n_analyses: int, errors: list[str]) -> None:
    """Validate the optional ``data.zarr/rho`` group when present (PRD
    "Validation", issue 049). A store without it stays valid unchanged --
    Rho is opt-in, add-in-place metadata (ADR 0025), not association data.
    Structural/consistency only: this does not re-run the estimator.
    """
    if "rho" not in root:
        return
    group = root["rho"]

    missing_attrs = [a for a in _RHO_REQUIRED_ATTRS if a not in group.attrs]
    if missing_attrs:
        errors.append(f"rho group is missing provenance attrs: {', '.join(missing_attrs)}")
        return
    for name in ("rho", "n_null", "variant_index"):
        if name not in group:
            errors.append(f"rho group is missing data.zarr/rho/{name}")
    if errors:
        return

    attr_n_analyses = int(group.attrs["n_analyses"])
    if attr_n_analyses != n_analyses:
        errors.append(
            f"rho group n_analyses attr {attr_n_analyses} does not match "
            f"analyses.tsv's {n_analyses}"
        )
        return

    expected_len = n_analyses * (n_analyses - 1) // 2
    rho_vals = np.asarray(group["rho"][:], dtype=np.float32)
    n_null_vals = np.asarray(group["n_null"][:], dtype=np.int64)
    if len(rho_vals) != expected_len:
        errors.append(f"rho array has {len(rho_vals)} entries but expected {expected_len}")
    if len(n_null_vals) != expected_len:
        errors.append(f"n_null array has {len(n_null_vals)} entries but expected {expected_len}")
    if errors:
        return

    min_nulls = int(group.attrs["min_nulls"])
    finite = np.isfinite(rho_vals)
    should_be_finite = n_null_vals >= min_nulls
    if not np.array_equal(finite, should_be_finite):
        errors.append(f"rho finiteness is inconsistent with n_null support (min_nulls={min_nulls})")
    if finite.any():
        vals = rho_vals[finite]
        if np.any(vals < -1.0) or np.any(vals > 1.0):
            errors.append("rho contains values outside [-1, 1]")

    n_variants_used = int(group.attrs["n_variants_used"])
    variant_index = group["variant_index"]
    if array_length(variant_index) != n_variants_used:
        errors.append(
            f"rho variant_index has {array_length(variant_index)} entries but "
            f"n_variants_used attr says {n_variants_used}"
        )


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


def _source_units(
    source: str | Path | Sequence[str | Path], errors: list[str]
) -> list[tuple[str | None, Path, bool]]:
    """Resolve ``source`` into (analysis_id_or_None, path, is_vcf) units.

    A single TSV whose header carries ``trait_id``/``file_path`` is treated as a
    build manifest (one analysis per file, analysis_id from ``trait_id``);
    otherwise each path is a self-describing source file (analysis_id per row).
    """
    import csv

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            errors.append(f"source path does not exist: {p}")
            return []
        header = None
        if not _is_vcf_path(p) and not str(p).lower().endswith(".gz"):
            try:
                with open(p, encoding="utf-8") as fh:
                    header = fh.readline()
            except (OSError, UnicodeDecodeError):
                header = None
        if header is not None and "file_path" in header and "trait_id" in header:
            units: list[tuple[str | None, Path, bool]] = []
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    fp = Path(row["file_path"])
                    units.append((row["trait_id"], fp, _is_vcf_path(fp)))
            return units
        return [(None, p, _is_vcf_path(p))]
    return [(None, Path(item), _is_vcf_path(Path(item))) for item in source]


def _stream_unit(
    analysis_id: str | None, path: Path, is_vcf: bool
) -> Iterator[tuple[str, str, int, str, str, float, float]]:
    """Yield canonical (analysis_id, chrom, pos, a1, a2, z, se) from one source."""
    if is_vcf:
        from opengwasdb.build.vcf_source import stream_vcf_associations

        if analysis_id is None:
            return  # a VCF carries no analysis_id; it needs a manifest trait_id
        # `_eaf` is unused here: this check compares stored z/se against the
        # source, and EAF plays no part in that (ADR 0036).
        for chrom, pos, ref, alt, z, se, _eaf in stream_vcf_associations(path):
            a1, a2 = sorted((ref, alt))
            yield (analysis_id, chrom, int(pos), a1, a2, float(z), float(se))
    else:
        from opengwasdb.build.source import read_normalised_associations

        for assoc in read_normalised_associations(path):
            v = assoc.variant
            a1, a2 = sorted((v.effect_allele, v.other_allele))
            yield (
                assoc.analysis_id,
                v.chromosome,
                int(v.position),
                a1,
                a2,
                float(assoc.z),
                float(assoc.se),
            )


def _sample_sources(
    units: list[tuple[str | None, Path, bool]], n_samples: int, rng: np.random.Generator
) -> list[tuple[str, str, int, str, str, float, float]]:
    """Reservoir-sample up to ``n_samples`` associations across a random subset
    of source files (scan-capped per file so cost stays bounded)."""
    if not units:
        return []
    order = rng.permutation(len(units))[:_FIDELITY_MAX_FILES]
    reservoir: list[tuple[str, str, int, str, str, float, float]] = []
    seen = 0
    for u in order:
        analysis_id, path, is_vcf = units[int(u)]
        scanned = 0
        for item in _stream_unit(analysis_id, path, is_vcf):
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
) -> list[_FidelitySample] | None:
    """Sampling phase: deterministically reservoir-sample source associations.

    ``source`` is resolved into per-file units, then up to ``n_samples``
    associations are drawn with a generator seeded by ``seed``, so repeated
    validation of the same store reproduces the same sample. Returns None --
    having already recorded why in ``errors`` -- when the source is unusable
    (missing path) or no associations could be read from it.
    """
    units = _source_units(source, errors)
    if errors or not units:
        return None
    samples = _sample_sources(units, n_samples, np.random.default_rng(seed))
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
        se_ref = float(np.float16(se_src))
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
    samples = _draw_fidelity_samples(source, errors, n_samples=n_samples, seed=seed)
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
