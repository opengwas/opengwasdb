"""Source Reader Capability resolution (issue #19; opengwasdb-stores ADR-0009).

The one place a Source Collection's `source_reader_capability` string becomes
a `SourceReader`. Every builder, Variant Routing, the resolver and the
annotation stages open their sources here: `open_source` from an Analysis's
`SourceSpec`, or `resolve_reader` from a capability and a path, which
`opengwasdb-stores` also calls.

A reader opened with a lift (`resolve_reader`'s ``liftover``, or
`open_source`'s ``alignment``) presents its metric blocks and sites on the
reference build; its variants and associations stay in the source's own
assembly, which Variant Routing lifts. Only a capability that can lift accepts
one, so no caller needs to know which reader class it was handed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.association_tsv import ASSOCIATION_TSV_CAPABILITY, AssociationTsvReader
from opengwasdb.readers.besd import BESD_CAPABILITY, BesdProbeReader, load_besd_source
from opengwasdb.readers.column_map import COLUMN_MAP_CAPABILITY, ColumnMap
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader, load_liftover
from opengwasdb.readers.interface import ImputationScoreDeclaration, SourceReader

if TYPE_CHECKING:
    from opengwasdb.readers.assembly import AssemblyPlan


def _needs_column_map(path: str | Path, _scale: StoredEffectScale) -> SourceReader:
    """`opengwasdb.column-map` without its map: `resolve_reader` builds a mapped
    reader itself, so reaching this factory means no map was given."""
    raise ValueError(f"{COLUMN_MAP_CAPABILITY!r} source {path} needs a Column Map")


def _needs_probe_id(path: str | Path, _scale: StoredEffectScale) -> SourceReader:
    """`opengwasdb.besd` without its probe: `resolve_reader` opens a probe's
    reader itself, so reaching this factory means no probe was named."""
    raise ValueError(f"{BESD_CAPABILITY!r} source {path} needs the .epi probe it reads")


_READERS: dict[str, Callable[[str | Path, StoredEffectScale], SourceReader]] = {
    ASSOCIATION_TSV_CAPABILITY: AssociationTsvReader,
    BESD_CAPABILITY: _needs_probe_id,
    COLUMN_MAP_CAPABILITY: _needs_column_map,
    FINNGEN_R13_CAPABILITY: FinnGenR13Reader,
    GWAS_VCF_CAPABILITY: GwasVcfReader,
    GWAS_SSF_CAPABILITY: GwasSsfReader,
}


#: The capabilities whose readers can lift their annotation projections onto
#: the reference build. A BESD file set's `.esi` rows lift by the GWAS-VCF
#: projection itself (issue #282).
_LIFTING: dict[str, Callable[[str | Path, StoredEffectScale, object], SourceReader]] = {
    GWAS_VCF_CAPABILITY: lambda path, scale, liftover: GwasVcfReader(
        path, scale, liftover=liftover
    ),
    BESD_CAPABILITY: lambda path, scale, _liftover: _needs_probe_id(path, scale),
}


#: The capabilities that read a declared imputation score column (#175): the
#: GWAS-SSF reader, and the Column Map reader that reads by its rules.
_SCORED = frozenset({GWAS_SSF_CAPABILITY, COLUMN_MAP_CAPABILITY})


def requires_probe_id(capability: str) -> bool:
    """Whether a capability's source holds many Analyses, one per named probe.

    A BESD file set holds one Analysis per `.epi` probe, so its Analysis must
    name the probe, never inferred from its `analysis_id`; every other format
    holds one Analysis per file, and a probe there would be silently meaningless.
    """
    return capability == BESD_CAPABILITY


def known_capabilities() -> tuple[str, ...]:
    """Return all registered source reader capabilities in sorted order."""
    return tuple(sorted(_READERS))


def resolve_reader(
    capability: str, path: str | Path, stored_effect_scale: StoredEffectScale | None,
    *, imputation_score_declaration: ImputationScoreDeclaration | None = None,
    column_map: ColumnMap | None = None,
    liftover: object | None = None,
    probe_id: str | None = None,
) -> SourceReader:
    """Return a SourceReader for `path`, given its Source Collection's
    `source_reader_capability` string.

    `stored_effect_scale` is the Analysis's manifest-resolved effect scale
    (issue #17): every reader needs it, since no Source Format's own header
    is authoritative for it.

    `column_map` is the Column Map an `opengwasdb.column-map` Analysis is read
    through (issue #281). It is required for that capability and refused for
    every other, so a map can neither be forgotten nor silently not used.

    `liftover` (a `load_liftover` result) opens the reader lifted onto the
    reference build. A capability that cannot lift refuses it, rather than
    read the source in its own assembly as if it were the reference's.

    `probe_id` is the `.epi` probe an `opengwasdb.besd` Analysis is: required
    there and refused elsewhere (`requires_probe_id`).

    Raises ValueError for an unknown capability, naming the ones this
    package knows about.
    """
    _refuse_unhonoured(
        capability,
        column_map=column_map,
        imputation_score_declaration=imputation_score_declaration,
        liftover=liftover,
        probe_id=probe_id,
    )
    if requires_probe_id(capability):
        return _probe_reader(path, stored_effect_scale, probe_id, liftover)
    if stored_effect_scale is None:
        raise ValueError(f"a {capability!r} source needs its Analysis's stored_effect_scale")
    if liftover is not None:
        return _LIFTING[capability](path, stored_effect_scale, liftover)
    if column_map is not None:
        return ColumnMapReader(
            path, column_map, stored_effect_scale,
            imputation_score_declaration=imputation_score_declaration,
        )
    return _registered_reader(capability, path, stored_effect_scale, imputation_score_declaration)


def _refuse_unhonoured(
    capability: str,
    *,
    column_map: ColumnMap | None,
    imputation_score_declaration: ImputationScoreDeclaration | None,
    liftover: object | None,
    probe_id: str | None,
) -> None:
    """Refuse, before any reader is opened, what ``capability`` cannot honour.

    Each of these is a declaration about how to read the source; a reader that
    went on without it would read the source some other way than declared.
    """
    if capability not in _READERS:
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise ValueError(f"unknown source reader capability {capability!r}; known: {known}")
    declared = {
        "column_map": column_map,
        "imputation_score_declaration": imputation_score_declaration,
        "liftover": liftover,
        "probe_id": probe_id,
    }
    for name, value in declared.items():
        honoured_by, refusal = _HONOURED[name]
        if value is not None and capability not in honoured_by():
            raise ValueError(refusal.format(capability=capability, value=value))


#: Each declaration `resolve_reader` takes: the capabilities that honour it,
#: and how one that cannot is refused.
_HONOURED: dict[str, tuple[Callable[[], frozenset[str]], str]] = {
    "column_map": (
        lambda: frozenset({COLUMN_MAP_CAPABILITY}),
        f"a Column Map reads only {COLUMN_MAP_CAPABILITY!r} sources, not {{capability!r}}",
    ),
    "imputation_score_declaration": (
        lambda: _SCORED,
        "imputation score declaration unsupported by {capability!r}",
    ),
    "liftover": (
        lambda: frozenset(_LIFTING),
        "source reader capability {capability!r} cannot lift its source onto the reference "
        f"build; capabilities that can: {', '.join(sorted(_LIFTING))}",
    ),
    "probe_id": (
        lambda: frozenset({BESD_CAPABILITY}),
        f"{{capability!r}} sources hold one Analysis each; only {BESD_CAPABILITY!r} names a "
        "probe, not {value!r}",
    ),
}


def _registered_reader(
    capability: str,
    path: str | Path,
    stored_effect_scale: StoredEffectScale,
    imputation_score_declaration: ImputationScoreDeclaration | None,
) -> SourceReader:
    """`resolve_reader` for a capability read by its own registered factory."""
    # Built first, so a capability that cannot be read as given (a mapped one
    # without its map) says so before anything else is said about it.
    reader = _READERS[capability](path, stored_effect_scale)
    if imputation_score_declaration is None:
        return reader
    return GwasSsfReader(
        path, stored_effect_scale, imputation_score_declaration=imputation_score_declaration
    )



@dataclass(frozen=True)
class SourceSpec:
    """One Analysis's source: where it is, how to read it, and its assembly.

    Everything `open_source` needs, and nothing a build copies into the Store
    from the manifest, so Variant Routing can read sources without knowing
    what a manifest row is. ``assembly`` is ``"hg19"`` or ``"hg38"``.
    """

    analysis_id: str
    path: str
    capability: str
    assembly: str
    #: ``None`` only for a BESD probe that declares no scale: its batches
    #: carry none, and its association stream refuses to assume one.
    stored_effect_scale: StoredEffectScale | None
    column_map: ColumnMap | None = None
    #: The `.epi` probe a BESD Analysis is; ``None`` for every other format.
    probe_id: str | None = None

    def __post_init__(self) -> None:
        if self.stored_effect_scale is None and not requires_probe_id(self.capability):
            raise ValueError(
                f"Analysis {self.analysis_id!r}: a {self.capability!r} source needs the "
                "Analysis's stored_effect_scale"
            )
        if requires_probe_id(self.capability) != (self.probe_id is not None):
            raise ValueError(
                f"Analysis {self.analysis_id!r}: a {self.capability!r} source "
                + ("must name its .epi probe" if self.probe_id is None
                   else f"is read whole; it names no probe, not {self.probe_id!r}")
            )


def open_source(
    spec: SourceSpec,
    *,
    imputation_score_declaration: ImputationScoreDeclaration | None = None,
    alignment: AssemblyPlan | None = None,
) -> SourceReader:
    """The `SourceReader` for one Analysis's source, as `resolve_reader` resolves it.

    ``alignment`` is the Analysis's `AssemblyPlan`: one that requires a lift
    opens the reader lifted by its chain, as `resolve_reader`'s ``liftover``.
    """
    liftover = None
    if alignment is not None and alignment.requires_liftover:
        liftover = load_liftover(
            alignment.source_build, alignment.reference_build, str(alignment.chain_file)
        )
    return resolve_reader(
        spec.capability,
        spec.path,
        spec.stored_effect_scale,
        imputation_score_declaration=imputation_score_declaration,
        column_map=spec.column_map,
        liftover=liftover,
        probe_id=spec.probe_id,
    )


def lifting_capabilities() -> frozenset[str]:
    """The capabilities whose readers accept a lift onto the reference build."""
    return frozenset(_LIFTING)


def _probe_reader(
    path: str | Path,
    stored_effect_scale: StoredEffectScale | None,
    probe_id: str | None,
    liftover: object | None,
) -> SourceReader:
    """One probe's reader of a multi-Analysis file set, its `.esi` read once.

    Opened lifted, its metric blocks and sites are read from the file set
    lifted onto the reference build; its associations stay in the file set's
    own assembly, as every reader's do.
    """
    if probe_id is None:
        return _needs_probe_id(path, StoredEffectScale.SD)
    return BesdProbeReader(
        path, probe_id, stored_effect_scale,
        aligned=None if liftover is None else load_besd_source(path, liftover=liftover),
    )
