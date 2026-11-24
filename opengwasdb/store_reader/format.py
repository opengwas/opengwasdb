"""A Store Release's `format_version`: which ones exist, and whether this build reads one.

The version tables -- the release series this build reads and the versions it
writes (ADR 0041, spec §21) -- and the parse and check every open runs before it
reads anything else (ADR 0038 §2). What a writer stamps and refuses to write is
`opengwasdb.store.open`'s; it builds on these.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType

log = logging.getLogger(__name__)

#: The highest compatible remainder this build fully understands, per readable
#: **release series** (ADR 0041, spec §21). A series absent from this mapping is
#: rejected; a remainder above the one recorded here is read with a warning,
#: because the compatible axis is defined as one an older reader still reads
#: correctly.
#:
#: Two entries since #245: `0.1` (Zarr v2, read but never written again) and
#: `0.2` (Zarr v3 with sharding, what every builder writes from #247 on), read to
#: 0.2.3: `0.2.1` adds #430's region table, `0.2.2` #381's indel ALIDs spelt
#: reference first (ADR 0079), `0.2.3` #338's `n_missing_se_out_of_range` (ADR
#: 0068). The four pre-reset versions are not in it and are not readable -- see
#: `PRE_RESET_FORMAT_VERSIONS`.
#:
#: `0.3` since #467: 0.2.3's features, with the variable-length ALID sidecar in place
#: of the fixed-width `|S32` index (ADR 0105). A reader of the 0.2 series would find
#: no ALID index and misread every ALID lookup, so 0.3.0 is a new series that an
#: older package refuses by name (ADR 0041). It is in development on the epic #450
#: branch, and released once #468-#472 are in.
SUPPORTED_FORMAT_VERSIONS: Mapping[tuple[int, ...], tuple[int, ...]] = MappingProxyType(
    {(0, 1): (0,), (0, 2): (3,), (0, 3): (0,)}
)

#: The format_versions this build writes, oldest first. A build stamps each release
#: the lowest whose features it uses (`format_version_for`; spec §21.3, ADR 0038 §3
#: as amended by the owner's decision of 2026-11-11, issue #430), and reads every
#: one in `SUPPORTED_FORMAT_VERSIONS`: supporting the *writing* of historical
#: formats would mean keeping every retired encoder alive and tested, for a use
#: case nobody has, but these two are written by the same encoders.
#:
#: `0.2.0` since #247: Zarr v3 with the sharding codec (ADR 0057), at the Dense
#: inner chunk and shard shapes ADR 0058 decided. The builders write it through
#: `opengwasdb.store.arrays`; `0.1.0` stays readable until a later decision deletes
#: the v2 reader (ADR 0057). `0.2.1` since #430: 0.2.0 plus a Ragged release's
#: optional, manifest-declared `regions.tsv`, so a 0.2.0 release is a 0.2.1 one
#: without the file. `0.2.2` since #381: 0.2.1 with an indel's ALID spelt reference
#: first rather than sorted (ADR 0079), `regions.tsv` still optional. `0.2.3`
#: since #338: 0.2.2 whose `analyses.tsv` records `n_missing_se_out_of_range` on
#: every row, and which holds no association whose SE is above 65,504 (ADR 0068).
#: Package v0.9.0 wrote 0.2.0 and 0.2.1 with sorted indels. `0.3.0` since #467: 0.2.3
#: with the variable-length ALID sidecar (ADR 0105). This build writes 0.3.0 for every
#: release it builds, completes or rewrites from 0.2.3 (`opengwasdb.store.rewrite`),
#: and 0.2.0 for one it converts from 0.1.0 (`opengwasdb.store.convert`), whose
#: indels it cannot re-spell. The 0.2 versions stay in this table because a pass
#: that adds arrays to a 0.2 release (`ogdb build-by-variant`) writes them, and a
#: 0.2.2 or 0.2.3 source is completed.
WRITTEN_FORMAT_VERSIONS: tuple[str, ...] = ("0.2.0", "0.2.1", "0.2.2", "0.2.3", "0.3.0")

#: The last 0.2 version: 0.2.2's reference-first indels with 0.2.3's SE count, every
#: feature 0.3.0 keeps. The one version `ogdb rewrite-release` derives 0.3.0 from.
LAST_0_2_FORMAT_VERSION = WRITTEN_FORMAT_VERSIONS[3]

#: The first version whose ALID index is the variable-length sidecar (ADR 0105).
ALID_SIDECAR_FORMAT_VERSION = WRITTEN_FORMAT_VERSIONS[4]

#: The first `format_version` whose indel ALIDs spell the reference allele first
#: (ADR 0079). Every earlier one sorts an indel's alleles, so its deletion reads
#: like the insertion of the same sequence; nothing rewrites such a Store.
REFERENCE_FIRST_INDELS_SINCE = (0, 2, 2)


def indels_reference_first(version: str) -> bool:
    """Whether a Store of ``version`` spells an indel's ALID reference first.

    A Store before `REFERENCE_FIRST_INDELS_SINCE` is read as it was written: its
    indel ALIDs keep their sorted spelling, and a check or a join against it
    spells a source's indel the same way (ADR 0079).
    """
    return parse_format_version(version) >= REFERENCE_FIRST_INDELS_SINCE


def format_version_for(
    *,
    carries_regions: bool,
    reference_first_indels: bool,
    records_se_out_of_range: bool = True,
    alid_sidecar: bool = True,
) -> str:
    """The `format_version` a release this build writes is stamped: the lowest whose
    features it uses (spec §21.3). The one rule every writer stamps by -- the builders,
    completion, the converter -- so a release no older reader can fully read says so,
    and every other one reads exactly as before.

    `carries_regions`: the release carries `regions.tsv`, which 0.2.1 adds.
    `reference_first_indels`: its indel ALIDs spell the reference first, which 0.2.2
    adds (ADR 0079); every release this build builds or completes does, one it
    converts from 0.1.0 does not.
    `records_se_out_of_range`: its `analyses.tsv` counts, on every row, the
    associations left out for an SE above 65,504, which 0.2.3 adds (ADR 0068). Every
    release this build builds or completes does. 0.2.3 is 0.2.2 and the count, so a
    release whose indels are sorted is stamped by its indels, whatever it counts.
    `alid_sidecar`: it carries the variable-length ALID sidecar rather than the
    fixed-width index, which 0.3.0 adds (ADR 0105) on all of 0.2.3's features. Every
    release this build builds, completes or rewrites does; one it converts from 0.1.0
    does not. A release with the sidecar and without one of 0.2.3's features is no
    version, and is refused rather than stamped.
    """
    if alid_sidecar:
        if not (reference_first_indels and records_se_out_of_range):
            raise ValueError(
                "a release carrying the variable-length ALID sidecar is 0.3.0, which also "
                "spells its indels reference first and counts the SEs left out (ADR 0105); "
                f"this one does not (reference_first_indels={reference_first_indels}, "
                f"records_se_out_of_range={records_se_out_of_range})"
            )
        return ALID_SIDECAR_FORMAT_VERSION
    if reference_first_indels:
        return WRITTEN_FORMAT_VERSIONS[3 if records_se_out_of_range else 2]
    return WRITTEN_FORMAT_VERSIONS[1] if carries_regions else WRITTEN_FORMAT_VERSIONS[0]

#: The format_version a release that uses no optional feature is stamped: what
#: every build of this package writes for a release without `regions.tsv`, and
#: what opengwasdb-stores' engine update probes as the format a locked engine
#: writes.
CURRENT_FORMAT_VERSION = format_version_for(carries_regions=False, reference_first_indels=True)

#: The Zarr on-disk format each readable release series is stored in.  The
#: series-to-layout coupling is part of the format contract, so it lives beside
#: the version tables rather than in each caller that needs it (#264 review).
ZARR_FORMAT_BY_SERIES: Mapping[tuple[int, ...], int] = MappingProxyType(
    {(0, 1): 2, (0, 2): 3, (0, 3): 3}
)

#: The first `format_version` whose ALID index is the variable-length sidecar
#: (`opengwasdb.store_reader.alid_sidecar`, ADR 0105) and not the fixed-width pair.
ALID_SIDECAR_SINCE = tuple(int(part) for part in ALID_SIDECAR_FORMAT_VERSION.split("."))


def carries_alid_sidecar(version: str) -> bool:
    """Whether a Store of ``version`` indexes its ALIDs with the variable-length sidecar."""
    return parse_format_version(version) >= ALID_SIDECAR_SINCE


#: The first `format_version` whose `index.sqlite` has no `metadata` table -- the
#: `dense` layout blob, `schema_version`, `n_variants`, `n_analyses` -- (ADR 0102,
#: #471): `manifest.json` is the one copy of the layout. 0.3.0 is one format version,
#: so it is the ALID sidecar's.
INDEX_METADATA_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_index_metadata(version: str) -> bool:
    """Whether a Store of ``version`` carries `index.sqlite`'s `metadata` table.

    Every release of such a version carries `index.sqlite` (a Ragged one's may be
    empty). From 0.3.0 it holds only `completion_quality`, so only a release that holds
    its own carries one (`opengwasdb.store.open.ships_index`).
    """
    return parse_format_version(version) < INDEX_METADATA_DROPPED_SINCE


#: The first `format_version` with no `variant_aliases` (ADR 0102, #471): an rsID the
#: rsID index lacks is a miss, and the Store Reader never opens `index.sqlite`.
VARIANT_ALIASES_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_variant_aliases(version: str) -> bool:
    """Whether a Store of ``version`` may name an rsID in `index.sqlite`'s
    `variant_aliases`, which a Dense or Hybrid query falls back to (#476 retires it)."""
    return parse_format_version(version) < VARIANT_ALIASES_DROPPED_SINCE


#: The first `format_version` whose Top-Hit tiers store `z` but neither `abs_z` nor
#: `p_value`, which the validator recomputes from `z` (ADR 0102, #471).
STORED_P_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_stored_p(version: str) -> bool:
    """Whether a Store of ``version``'s Top-Hit tiers store `abs_z` and `p_value`."""
    return parse_format_version(version) < STORED_P_DROPPED_SINCE


#: The first `format_version` that answers a range from the position sidecar
#: (`opengwasdb.store_reader.position_sidecar`, ADR 0105 §6) and not through tabix.
#: 0.3.0 is one format version: every release that carries the ALID sidecar carries
#: this one too.
POSITION_SIDECAR_SINCE = ALID_SIDECAR_SINCE


def carries_position_sidecar(version: str) -> bool:
    """Whether a Store of ``version`` carries the position sidecar a range is answered from."""
    return parse_format_version(version) >= POSITION_SIDECAR_SINCE


#: The first `format_version` with no `variant_offsets.npy` (ADR 0102, ADR 0105 §7,
#: #469): its Store Reader never reads `variants.tsv.gz`, so nothing seeks a row's BGZF
#: offset; a row's identity and rsID come from the sidecars, and the variant count from
#: the ALID sidecar. 0.3.0 is one format version, so it is the ALID sidecar's.
VARIANT_OFFSETS_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_variant_offsets(version: str) -> bool:
    """Whether a Store of ``version`` carries `variant_offsets.npy`, each row's BGZF
    offset in its Variant Table, and is read through that table: the 0.2 and 0.1 series."""
    return parse_format_version(version) < VARIANT_OFFSETS_DROPPED_SINCE


#: The first `format_version` whose Hybrid Dense Component carries no Variant Axis
#: and no `analyses.tsv` of its own: it reads the shared ones through
#: `dense_to_shared.npy` (#472, ADR 0102 §4). 0.3.0 is one format version.
DENSE_COMPONENT_SHARES_AXIS_SINCE = ALID_SIDECAR_SINCE


def dense_component_shares_axis(version: str) -> bool:
    """Whether a Hybrid Dense Component of ``version`` reads its variants and Analyses
    from the release that nests it, rather than carrying its own copy of each."""
    return parse_format_version(version) >= DENSE_COMPONENT_SHARES_AXIS_SINCE


def zarr_format_for_version(version: str, *, source: str = "release") -> int:
    """The Zarr on-disk format a readable ``format_version`` is stored in.

    Runs the reader's own version check first, so a caller can ask this alone
    and get both "is it readable" and "what layout is it" -- a caller that
    instead pattern-matched a version string would be the second place the
    version-to-layout rule lives, and the two would eventually disagree.
    """
    series, _ = split_format_version(version)
    check_format_version(version, source=source)
    try:
        return ZARR_FORMAT_BY_SERIES[series]
    except KeyError:
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, series "
            f"{_series_text(series)}, which this build reads but has no Zarr on-disk "
            "layout for; the version table and the layout table disagree"
        ) from None

#: The versions the format carried before the reset, and what each one was.
#: Every one is two-component, so the parser rejects it on shape alone; naming
#: them here is what turns that rejection into an instruction rather than a
#: complaint about punctuation (issue #143). No store in any of them was ever
#: published outside the core team, and this build decodes none of them: the
#: bytes of a `3.0` release are the bytes of a `0.1.0` one, but `0.1`, `1.0`
#: and `2.0` are genuinely different encodings and reading them was deleted
#: rather than deprecated.
PRE_RESET_FORMAT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "0.1": "the first pre-release format, float16 throughout",
        "1.0": "fixed-point z (issue #114)",
        "2.0": "residual-coded eaf (issue #116)",
        "3.0": "residual-coded se (issue #118)",
    }
)


class UnsupportedFormatVersion(Exception):
    """A release declares a format_version this build cannot interpret."""


class MalformedFormatVersion(UnsupportedFormatVersion):
    """A release's format_version is not ``MAJOR.MINOR.PATCH``.

    A subclass rather than a `ValueError`: to a caller deciding whether it can
    read a release, "the version is nonsense" and "the version is from the
    future" are the same answer, and both must stop the read.
    """


def parse_format_version(version: str) -> tuple[int, int, int]:
    """``"0.1.0"`` -> ``(0, 1, 0)``. Raises `MalformedFormatVersion` otherwise.

    Three components, not two, and that shape change is the safety mechanism of
    the reset rather than a cosmetic choice (issue #143, ADR 0041). Pre-reset
    releases are stamped ``0.1``, and a reset that reused that string would
    give two different formats one name -- a store that reads as plausible and
    is wrong. A two-component value has no reading here at all, so every such
    release is refused loudly instead.
    """
    text = str(version)
    was = PRE_RESET_FORMAT_VERSIONS.get(text)
    if was is not None:
        raise UnsupportedFormatVersion(
            f"format_version {text!r} is a pre-release format ({was}), which this "
            "build no longer reads: the format was reset to "
            f"{CURRENT_FORMAT_VERSION} and the decoders for every earlier version "
            "were deleted (ADR 0041, issue #143). Rebuild the release from source "
            "-- a store in one of those formats cannot be validated, completed or "
            "queried by this build"
        )
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MalformedFormatVersion(
            f"format_version {text!r} is not MAJOR.MINOR.PATCH (ADR 0041, spec §21)"
        )
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def split_format_version(version: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``"0.1.0"`` -> ``((0, 1), (0,))``: a version's series and its remainder.

    The **series** is the part a compatible change may not move, and the
    **remainder** is the part it may. Which is which follows semantic
    versioning's own rule, that the *leftmost non-zero component is the
    breaking one*: below ``1.0.0`` a format is declaring itself unsettled, so
    ``MINOR`` carries an incompatible change and ``PATCH`` does not; from
    ``1.0.0`` it is ``MAJOR``, and ``MINOR`` joins the remainder.

    Both halves come from one function because they are one rule. A caller that
    took the series here and worked out the remainder for itself would be
    stating that rule a second time, and the two would eventually disagree --
    `check_format_version` compares what this returns and knows nothing about
    which component is which.
    """
    major, minor, patch = parse_format_version(version)
    if major == 0:
        return (0, minor), (patch,)
    return (major,), (minor, patch)


def check_format_version(version: str, *, source: str = "release") -> None:
    """Reject a `format_version` this build cannot interpret (ADR 0041 §2).

    Rejection is on the **release series** alone: a build carries the series it
    implements, not a list of every version it has ever seen. A *newer
    remainder* within a known series is accepted and warned about -- accepting
    it is the definition of a compatible change, and the warning is how a
    mis-classified one (a change that should have moved the series) becomes
    visible instead of silently returning partial data.
    """
    series, remainder = split_format_version(version)
    known = SUPPORTED_FORMAT_VERSIONS.get(series)
    if known is None:
        readable = [_series_text(known_series) for known_series in SUPPORTED_FORMAT_VERSIONS]
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, a release series this "
            f"build does not implement; readable series: {sorted(readable)}"
        )
    if remainder > known:
        log.warning(
            "%s declares format_version=%s, a newer compatible version than this build "
            "knows (%s). Reading it as that: anything added after it is not visible "
            "here, and if any of it changes how existing arrays decode it was "
            "misclassified as compatible.",
            source,
            version,
            _series_text(series) + "." + ".".join(str(part) for part in known),
        )


def _series_text(series: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in series)
