"""Variant Reference artifacts: a routing, frozen, for single-pass builds (issue #185).

A Variant Reference is the precomputed variant axis a build stores against,
plus the mapping from a source's own ``(chrom, pos, ref, alt)`` coordinate to
the canonical hg38 ALID whose row it populates. A build given one routes by it
(`opengwasdb.variants.routing.FromReference`) instead of by Pass 1's lifted
union. This module is the artifact format -- reading, writing, and the rsid
oracles a routing is checked against; extracting one from a manifest's
sources is Variant Routing's (`opengwasdb.variants.routing.extract_variant_reference`).

Three inputs are accepted, distinguished by content rather than by suffix:

* ``*.variant-ref.tsv.gz`` -- the artifact the standalone
  ``extract-variant-reference`` stage writes (issue #187). It carries explicit
  ``source_keys``, so an hg19 source variant resolves to its lifted hg38 ALID
  without re-running Pass 1.
* a plain ALID list -- one canonical GRCh38 ALID per line. Source keys are the
  ALID's own coordinate in either allele order (identity).
* a store ``variants.tsv.gz`` -- the Store Variant Table. Source keys are
  identity, as for a plain list.

Anything that cannot be read as one of these fails loudly. A build must not
proceed against an empty or corrupt axis and silently store nothing -- an empty
result indistinguishable from a real answer is the failure mode this package
exists to prevent.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TypeVar

from opengwasdb.variants.axis import (
    alid_sort_key,
    iter_variant_records,
    parse_canonical_alid,
    variant_table_path,
)
from opengwasdb.variants.windows import DEFAULT_WINDOW_SIZE_MB, window_key, window_size_bp

log = logging.getLogger(__name__)

__all__ = [
    "VariantReference",
    "expected_rsids_by_routed_key",
    "expected_rsids_by_routing",
    "read_variant_reference",
    "require_rsids_match_expected",
    "require_written_rsids_match",
    "warn_reference_left_rows_blank",
    "write_variant_reference",
]

#: A source's own variant identity, exactly as a SourceReader streams it.
SourceKey = tuple[str, int, str, str]

#: Columns the artifact writer emits; the reader requires every one but
#: ``rsid`` (a source that named no rsid writes it blank).
_ARTIFACT_COLUMNS = ("alid", "chromosome", "position", "a1", "a2", "source_keys")

#: The artifact's first line. The streaming extraction writes it as the first
#: gzip member (issue #196).
ARTIFACT_HEADER = "#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n"


@dataclass(frozen=True)
class VariantReference:
    """A precomputed variant axis and the source coordinates resolving to it.

    ``source_lookup`` maps a source's raw ``(chrom, pos, ref, alt)`` (the exact
    tuple a ``SourceReader`` streams) to the canonical hg38 ALID whose row it
    populates. ``explicit_source_keys`` records whether that mapping came from
    an artifact's ``source_keys`` column or was inferred by identity, so a
    caller can tell a genuinely-hg38 panel from one carrying hg19 provenance.
    """

    alids: list[str]
    source_lookup: dict[SourceKey, str]
    rsid_by_alid: dict[str, str]
    explicit_source_keys: bool


def read_variant_reference(path: str | Path) -> VariantReference:
    """Read a variant reference, failing loudly on anything unreadable.

    The format is detected from the first line: a tab-separated header selects
    the artifact or Store Variant Table reader by its columns; anything else is
    a plain ALID list.
    """
    reference_path = Path(path)
    if not reference_path.is_file():
        raise ValueError(f"variant reference {reference_path} does not exist")
    header, lines = _read_header_and_lines(reference_path)
    if header is None:
        return _reference_from_alid_lines(reference_path, lines)
    if "source_keys" in header or ("a1" in header and "a2" in header):
        return _reference_from_artifact(reference_path, header, lines)
    if "alid" in header:
        return _reference_from_store_table(reference_path, header, lines)
    raise ValueError(
        f"variant reference {reference_path} has neither an 'alid' nor a "
        "'source_keys' column"
    )


#: A candidate or routing key: the raw source tuple, or the string form a
#: route table keys it by (`opengwasdb.variants.routing.keys.routing_key`).
_SiteKeyT = TypeVar("_SiteKeyT")


def expected_rsids_by_routed_key(
    rsid_by_site: Mapping[_SiteKeyT, str],
    routing: Mapping[str, str],
    key: Callable[[_SiteKeyT], str],
    *,
    drop_disputed: bool = False,
) -> dict[str, str]:
    """The oracle for a *raw* candidate map routed through a build's key function.

    It walks the raw candidates in their own `(rank, site)` order and applies
    ``key`` itself to each before looking the site up, rather than reading a map
    that was already keyed: a preparation that collapses two raw sites onto one
    key and keeps the wrong name therefore disagrees here, which an oracle
    sharing that preparation by construction could not see (issue #255 round 4).

    Takes the first non-empty rsid per routed ALID, exactly as
    `expected_rsids_by_routing` does for an already-keyed map. With
    ``drop_disputed`` -- a build run under ``--on-rsid-conflict drop`` -- a
    variant that any key whose sites name more than one rsid routes to expects
    no name at all (issue #306).
    """
    if drop_disputed:
        return _expected_without_disputed_keys(rsid_by_site, routing, key)
    expected: dict[str, str] = {}
    for site, rsid in rsid_by_site.items():
        alid = routing.get(key(site))
        if alid and rsid and alid not in expected:
            expected[alid] = rsid
    return expected


def _expected_without_disputed_keys(
    rsid_by_site: Mapping[_SiteKeyT, str],
    routing: Mapping[str, str],
    key: Callable[[_SiteKeyT], str],
) -> dict[str, str]:
    """`expected_rsids_by_routed_key` when a disputed variant names nothing.

    Finds the disputed keys itself rather than taking the build's list, so a
    build that missed one -- and kept a name its sources disagree on -- fails
    the comparison. Any ALID a disputed key routes to expects no name, whatever
    another key routed there says. Each key's first name is kept in the order
    its first named site came, which is the raw `(rank, site)` order restricted
    to named sites, so the first key routed to an ALID is the one the raw walk
    finds.
    """
    first_name, disputed = _first_names_and_disputed_keys(rsid_by_site, key)
    unnamed = {routing.get(routed) for routed in disputed}
    expected: dict[str, str] = {}
    for routed, rsid in first_name.items():
        alid = routing.get(routed)
        if alid and alid not in expected and alid not in unnamed:
            expected[alid] = rsid
    return expected


def expected_rsids_by_routing(
    rsid_by_site: Mapping[_SiteKeyT, str], routing: Mapping[_SiteKeyT, str]
) -> dict[str, str]:
    """An independent oracle for the ALID -> rsid map a harvest should produce.

    Deliberately simple and separate from `_rekey_rsids_to_alids` and from how
    the routing was built: it walks the candidates in the mapping's own order
    (Pass 1's `(rank, site)`, which a dict preserves), takes the first non-empty
    rsid per *routed* ALID, and returns the exact map. Comparing the resolved
    map against this is what makes the issue #255 check value-sensitive -- a run
    that keeps every key but attaches the wrong name to one of them disagrees
    here, which a set-membership check cannot see.
    """
    expected: dict[str, str] = {}
    for site, rsid in rsid_by_site.items():
        alid = routing.get(site)
        if alid and rsid and alid not in expected:
            expected[alid] = rsid
    return expected


def _first_names_and_disputed_keys(
    rsid_by_site: Mapping[_SiteKeyT, str], key: Callable[[_SiteKeyT], str]
) -> tuple[dict[str, str], set[str]]:
    """Each key's first name in raw order, and the keys named more than one way."""
    first_name: dict[str, str] = {}
    disputed: set[str] = set()
    for site, rsid in rsid_by_site.items():
        if not rsid:
            continue
        routed = key(site)
        if first_name.setdefault(routed, rsid) != rsid:
            disputed.add(routed)
    return first_name, disputed


def warn_reference_left_rows_blank(reference: VariantReference) -> None:
    """Warn that a reference naming some rsids leaves its other rows blank.

    `--variant-reference` accepts a reference that already names its variants,
    and both builders then use it as-is: a variant it does not name is stored
    with no rsid even when a source names it, and no source read is spent to
    find out. That behaviour predates this contract and is out of issue #255's
    scope, so the build says so with counts rather than leaving it to be
    discovered (round-4/5 scope ruling). Shared by both builders so the help,
    the spec and this warning cannot disagree.
    """
    unnamed = len(reference.alids) - len(reference.rsid_by_alid)
    if not unnamed:
        return
    log.warning(
        "variant reference names %d of its %d ALIDs and leaves %d blank; this build "
        "uses the reference as given and does not harvest the sources, so a variant "
        "only a source names is stored with no rsid (issue #255 finding 3)",
        len(reference.rsid_by_alid),
        len(reference.alids),
        unnamed,
    )


def require_rsids_match_expected(
    resolved_by_alid: Mapping[str, str], expected_by_alid: Mapping[str, str]
) -> None:
    """Refuse a resolved rsid map that differs from the routed expectation.

    Value-sensitive, unlike a set comparison: an ALID carrying the wrong name
    fails, as does a missing or an extra one. The caller builds
    ``expected_by_alid`` with `expected_rsids_by_routing` from the harvest's
    candidates and the routing the build actually uses, so a rekey or merge that
    misplaces a name fails before publication.

    It cannot catch a *source read* that omitted a variant: a table written from
    the same omitted candidates agrees with this expectation, and so does
    `require_written_rsids_match`. That omission is a reader-level failure and
    only an independent second read could see it.

    A source that names nothing builds an empty expectation and passes: "the
    sources name none" and "every named variant is present" stay different
    answers, but only the second is required.
    """
    wrong = sorted(
        alid
        for alid in sorted(set(resolved_by_alid) | set(expected_by_alid))
        if resolved_by_alid.get(alid) != expected_by_alid.get(alid)
    )
    if not wrong:
        return
    alid = wrong[0]
    raise ValueError(
        f"the resolved rsid map disagrees with the routed harvest for {len(wrong)} "
        f"variant(s) (e.g. {alid!r}: resolved {resolved_by_alid.get(alid)!r}, expected "
        f"{expected_by_alid.get(alid)!r}); refusing to publish identifiers that are not "
        "the ones the sources named for the variants they route to"
    )


def require_written_rsids_match(
    store_path: str | Path,
    alids: Sequence[str],
    rsid_by_alid: Mapping[str, str],
) -> None:
    """Refuse to publish a Store Variant Table that disagrees with its rsid map.

    The second half of the issue #255 rule: after the axis is resolved and the
    table written, the table is read back and compared row for row against the
    resolved map, so a partial loss, an off-reference row dropped from the
    write, or a table written from a stale map fails on the bytes a query will
    read rather than being trusted from the map that produced them.

    It complements, and does not replace, `require_rsids_match_expected`: this
    compares the bytes with the map, that compares the map with the routed
    candidates. A source read that omitted a variant is invisible to both, since
    the table is written from the same omitted map.

    A source that names nothing writes `.` (read back as ``""``) and passes: the
    expected map is built with ``""`` for an absent name, so "the sources name
    none" and "this row has no name" stay the same answer.
    """
    expected = {alid: rsid_by_alid.get(alid, "") for alid in alids}
    seen = 0
    for record in iter_variant_records(variant_table_path(store_path)):
        seen += 1
        wanted = expected.get(record.alid)
        written = record.rsid or ""
        if wanted is None:
            raise ValueError(
                f"{store_path} variant table holds {record.alid!r}, which the resolved "
                "axis does not -- the written table and the rsid map disagree"
            )
        if written != wanted:
            raise ValueError(
                f"{store_path} variant table row {record.alid!r} carries rsid {written!r} "
                f"but the resolved map says {wanted!r}; refusing to publish a variant "
                "table that disagrees with the identifiers the build resolved"
            )
    if seen != len(alids):
        raise ValueError(
            f"{store_path} variant table has {seen} row(s) but the resolved axis has "
            f"{len(alids)}; refusing to publish a variant table that lost rows"
        )


def write_variant_reference(
    path: str | Path,
    alids: Sequence[str],
    source_lookup: Mapping[SourceKey, str],
    rsid_by_alid: Mapping[str, str] | None = None,
    *,
    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB,
) -> None:
    """Write the artifact the inline two-pass Pass 1 would otherwise recompute.

    ``source_lookup`` is exactly the ``(source coord) -> hg38 ALID`` map Pass
    1's lifted union routes by, so a store built through a written
    artifact is identical to one built from the same manifest in two passes.

    Rows are assembled window by window (issue #188): each genomic window's
    ALIDs are sorted and the windows concatenate in genomic order, so the axis
    is never re-sorted as a whole -- only one window at a time.
    """
    size_bp = window_size_bp(window_size_mb)
    keys_by_alid = _source_keys_by_alid(source_lookup)
    rsids = rsid_by_alid or {}
    windows: dict[tuple[tuple[int, str], int], list[str]] = {}
    for alid in set(alids):
        chrom, position, _a1, _a2 = alid.split(":")
        windows.setdefault(window_key(chrom, int(position), size_bp), []).append(alid)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        handle.write(ARTIFACT_HEADER)
        for window in sorted(windows):
            for alid in sorted(windows[window], key=alid_sort_key):
                handle.write(artifact_line(alid, keys_by_alid.get(alid, ()), rsids.get(alid, "")))


def artifact_line(alid: str, source_keys: Sequence[str], rsid: str) -> str:
    """One artifact row: the ALID, its coordinate, its rsid and its source keys."""
    chrom, position, a1, a2 = alid.split(":")
    return (
        f"{alid}\t{chrom}\t{position}\t{a1}\t{a2}\t{rsid}\t{';'.join(source_keys)}\n"
    )


# ── reading ──────────────────────────────────────────────────────────────────


def _open_text(path: Path) -> IO[str]:
    if str(path).lower().endswith((".gz", ".bgz", ".bgzf")):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _read_header_and_lines(path: Path) -> tuple[list[str] | None, list[str]]:
    """The header (when the file is tab-separated) and every data line."""
    with _open_text(path) as handle:
        first = handle.readline()
        if not first.strip():
            raise ValueError(f"variant reference {path} is empty")
        if "\t" in first:
            header = [field.lstrip("#").strip() for field in first.rstrip("\n").split("\t")]
            return header, list(handle)
        return None, [first, *handle]


def _canonical_alid(token: str, path: Path) -> str:
    """Normalise an ALID to ``chrom:pos:a1:a2`` or fail loudly."""
    parsed = parse_canonical_alid(token)
    if parsed is None:
        raise ValueError(f"variant reference {path} contains invalid ALID {token!r}")
    a1, a2 = sorted((parsed.effect_allele, parsed.other_allele))
    return f"{parsed.chromosome}:{parsed.position}:{a1}:{a2}"


def _identity_keys(alid: str) -> Iterator[SourceKey]:
    """The two raw source tuples a canonical ALID can be reported as."""
    chrom, position, a1, a2 = alid.split(":")
    pos = int(position)
    yield (chrom, pos, a1, a2)
    yield (chrom, pos, a2, a1)


def _identity_lookup(alids: Sequence[str]) -> dict[SourceKey, str]:
    """Identity source keys -- for panels whose sources are already hg38."""
    lookup: dict[SourceKey, str] = {}
    for alid in alids:
        for key in _identity_keys(alid):
            lookup[key] = alid
    return lookup


def _require_alids(alids: Sequence[str], path: Path) -> None:
    if not alids:
        raise ValueError(f"variant reference {path} contained no ALIDs")


def _reference_from_alid_lines(path: Path, lines: Sequence[str]) -> VariantReference:
    alids: list[str] = []
    seen: set[str] = set()
    for token in _alid_tokens(lines):
        _append_alid(path, _canonical_alid(token, path), seen, alids)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=_identity_lookup(alids),
        rsid_by_alid={},
        explicit_source_keys=False,
    )


def _alid_tokens(lines: Sequence[str]) -> Iterator[str]:
    for line in lines:
        token = line.strip().split()[0] if line.strip() else ""
        if token and not token.startswith("#"):
            yield token


def _data_rows(lines: Sequence[str]) -> list[list[str]]:
    return [
        line.rstrip("\n").split("\t")
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _append_alid(path: Path, alid: str, seen: set[str], alids: list[str]) -> None:
    if alid in seen:
        raise ValueError(f"variant reference {path} contains duplicate ALID {alid!r}")
    seen.add(alid)
    alids.append(alid)


def _field(fields: Sequence[str], index: int, path: Path) -> str:
    if index >= len(fields):
        raise ValueError(
            f"variant reference {path} row has {len(fields)} fields, expected at least {index + 1}"
        )
    return fields[index]


def _record_rsid(
    path: Path,
    fields: Sequence[str],
    rsid_col: int | None,
    alid: str,
    rsid_by_alid: dict[str, str],
) -> None:
    rsid = _field(fields, rsid_col, path).strip() if rsid_col is not None else ""
    if rsid and rsid != ".":
        rsid_by_alid[alid] = rsid


def _reference_from_store_table(
    path: Path, header: Sequence[str], lines: Sequence[str]
) -> VariantReference:
    alid_col = header.index("alid")
    rsid_col = header.index("rsid") if "rsid" in header else None
    alids: list[str] = []
    rsid_by_alid: dict[str, str] = {}
    seen: set[str] = set()
    for fields in _data_rows(lines):
        alid = _canonical_alid(_field(fields, alid_col, path), path)
        _append_alid(path, alid, seen, alids)
        _record_rsid(path, fields, rsid_col, alid, rsid_by_alid)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=_identity_lookup(alids),
        rsid_by_alid=rsid_by_alid,
        explicit_source_keys=False,
    )


def _reference_from_artifact(
    path: Path, header: Sequence[str], lines: Sequence[str]
) -> VariantReference:
    missing = [name for name in _ARTIFACT_COLUMNS if name not in header]
    if missing:
        raise ValueError(
            f"variant reference {path} is missing the column(s) {', '.join(missing)}"
        )
    alids: list[str] = []
    rsid_by_alid: dict[str, str] = {}
    source_lookup: dict[SourceKey, str] = {}
    seen: set[str] = set()
    rsid_col = header.index("rsid") if "rsid" in header else None
    for fields in _data_rows(lines):
        alid = _artifact_alid(path, header, fields)
        _append_alid(path, alid, seen, alids)
        source_keys = _field(fields, header.index("source_keys"), path)
        _add_source_keys(path, source_keys, alid, source_lookup)
        _record_rsid(path, fields, rsid_col, alid, rsid_by_alid)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=source_lookup,
        rsid_by_alid=rsid_by_alid,
        explicit_source_keys=True,
    )


def _artifact_alid(path: Path, header: Sequence[str], fields: Sequence[str]) -> str:
    """The row's ALID, cross-checked against its own coordinate columns."""
    alid = _canonical_alid(_field(fields, header.index("alid"), path), path)
    described = ":".join(
        _field(fields, header.index(name), path)
        for name in ("chromosome", "position", "a1", "a2")
    )
    constructed = _canonical_alid(described, path)
    if constructed != alid:
        raise ValueError(
            f"variant reference {path}: row columns describe {constructed!r} but "
            f"the alid column says {alid!r}"
        )
    return alid


def _add_source_keys(
    path: Path, raw_keys: str, alid: str, lookup: dict[SourceKey, str]
) -> None:
    tokens = [token.strip() for token in raw_keys.split(";") if token.strip()]
    if not tokens:
        # An external panel row carries no source mapping; identity is the only
        # honest reading (its sources are already on the reference build).
        for key in _identity_keys(alid):
            _bind_source_key(path, key, alid, lookup)
        return
    for token in tokens:
        _bind_source_key(path, _parse_source_key(path, token), alid, lookup)


def _bind_source_key(path: Path, key: SourceKey, alid: str, lookup: dict[SourceKey, str]) -> None:
    existing = lookup.get(key)
    if existing is not None and existing != alid:
        raise ValueError(
            f"variant reference {path}: source key {key!r} maps to both "
            f"{existing!r} and {alid!r}"
        )
    lookup[key] = alid


def _parse_source_key(path: Path, token: str) -> SourceKey:
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError(f"variant reference {path} contains invalid source key {token!r}")
    chrom, position, ref, alt = parts
    try:
        pos = int(position)
    except ValueError as exc:
        raise ValueError(
            f"variant reference {path} contains invalid source key {token!r}"
        ) from exc
    if pos <= 0 or not chrom or not ref or not alt:
        raise ValueError(f"variant reference {path} contains invalid source key {token!r}")
    return (chrom, pos, ref, alt)


def _source_keys_by_alid(
    source_lookup: Mapping[SourceKey, str],
) -> dict[str, list[str]]:
    by_alid: dict[str, list[str]] = {}
    for (chrom, pos, ref, alt), alid in source_lookup.items():
        by_alid.setdefault(alid, []).append(f"{chrom}:{pos}:{ref}:{alt}")
    for keys in by_alid.values():
        keys.sort()
    return by_alid
