"""What a Store Release answers, as digests: the proof that a rewrite changed no answer.

A rewrite (`opengwasdb.store.rewrite`) changes how a release is stored, never what
it answers. Before it publishes, it asks the source and the staged release the same
deterministic set of questions and refuses on any difference. This module is that
set of questions and the digest of each answer. It has one owner, here: the
store-reader golden test (`tests/store_reader_digests.py`) calls it, and so does the
store-comparison benchmark for `digest_array` and `result_digests`.

Every association-returning question the query facade answers is asked: each
Analysis's associations; a PheWAS of variants spread over the axis, by ALID and by
rsID, and of two identifiers no release holds; a range PheWAS; a lookup; and the
top hits at five thresholds, overall and per Analysis, both Observed-Only and with
imputed cells. Every array each answer holds is digested (`result_digests`), with
the `variants_table()` and `analyses_table()` column by column, and the lookup, range
and widest top-hit answers resolved to rows, with variant info. Both name each row's
rsID and `source_alid`, so every row's `source_alid` is compared through
`variants_table()`, and the rows an answer reaches through its resolved rows: a 0.2
release reads them from its Variant Table, a 0.3.0 one from its sidecars (#469, #470).
A resolved row's `log10_p` is kept as a
number, because a test compares it with a tolerance against pins written by code
that computed it otherwise (`tests/test_store_reader_digests.py`); `differences`
compares it exactly, which is right between two releases read by one build.

The **ALID probes** ask about the ALIDs a spread of variants misses: the longest the
release holds, and one at each length where an ALID index changes how it searches
(`PROBED_ALID_LENGTHS`). Spread variants are short -- on OGS-00007 and OGS-00025 the
longest of them is 15 and 16 bytes -- so without the probes no question would reach
the sidecar's full-byte compare (#467 review). They are their own section, so the
store-reader pins, written before it, still name every shape they hold.

The **range probes** ask `range_phewas` where a position index can go wrong
(`range_questions`, #468): a range at the first and at the last variant; either side of
the first chromosome boundary; an empty range between two variants and one on a
chromosome the release lacks; and a range starting and ending on a position several
variants share. Each is chosen from the release's own rows, so it is deterministic, and
each is asked Observed-Only and with imputed cells. They are their own section too.

The **misses** ask about rsIDs no release names, by PheWAS and inside a lookup of the
spread identifiers. A 0.2 Dense or Hybrid release looks such an rsID up in
`index.sqlite`'s `variant_aliases`; a 0.3.0 release has no such table and answers it
as a miss without opening SQLite (ADR 0102, #471). Their own section, as the probes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.store_reader.associations import Associations

#: The arrays every association result holds; a digest that could not see one of them
#: could call two different answers equal.
RESULT_ARRAY_NAMES: tuple[str, ...] = tuple(Associations.empty())

#: The thresholds asked of `top_hits`: the three a Top-Hit Index holds, and two
#: it does not, which a release answers by scanning (`z_critical`).
THRESHOLDS = (5e-8, 5e-6, 5e-4, 1e-2, 0.2)

#: How many variants the PheWAS and lookup shapes name, spread over the axis.
N_VARIANTS_ASKED = 7

#: How many Analyses the per-Analysis shapes name, spread over the release. Every
#: fixture release has fewer, so each is asked about every one of its Analyses.
N_ANALYSES_ASKED = 8

#: ALID lengths, in bytes, where an ALID index changes how it searches: either side of
#: the sidecar's 24-byte key (`alid_sidecar.KEY_WIDTH`) and of the 0.2 series' 32-byte
#: slot (`alid.ALID_WIDTH`). The probes ask one ALID of each length a release holds.
PROBED_ALID_LENGTHS = (23, 24, 25, 32, 33)

#: How many of a release's longest ALIDs the probes ask.
N_LONGEST_PROBED = 3

#: Identifiers no release holds: an absent answer must stay absent.
ABSENT_IDENTIFIERS = ("1:1:A:C", "rs0")

#: rsIDs no release names, which the misses section asks (module docstring).
ABSENT_RSIDS = ("rs0", "rs999999999")

#: The chromosomes a range probe looks for one the release lacks in, in their order.
PROBED_CHROMOSOMES = (*(str(n) for n in range(1, 23)), "X", "Y", "MT")

#: The end of a range that runs to a chromosome's end: past every human position, and
#: within what a tabix index addresses (2**29), so a 0.2 release answers it too.
CHROMOSOME_END = 500_000_000


class IncompleteResultError(ValueError):
    """A query result lacks one of `RESULT_ARRAY_NAMES`, so its digest would not see it."""


def digest_array(values: np.ndarray) -> str:
    """A sha256 over one result array's dtype, shape, order and values.

    NaN is canonicalised as a position: a NaN may carry a different payload in
    two stores and still mean the same missing cell, so the digest records
    where the NaNs are and hashes only the non-NaN bytes.
    """
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    if values.dtype.kind == "O":
        unique, inverse = np.unique(values, return_inverse=True)
        digest.update(np.asarray(inverse, dtype="int64").tobytes())
        for item in unique.tolist():
            digest.update(repr(item).encode())
            digest.update(b"\x1f")
    elif values.dtype.kind == "f":
        missing = np.isnan(values)
        digest.update(b"nan\x1f")
        digest.update(np.packbits(missing).tobytes())
        digest.update(b"values\x1f")
        digest.update(values[~missing].tobytes())
    else:
        digest.update(values.tobytes())
    return digest.hexdigest()


def result_digests(result: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Digest EVERY array the query returned, not only the expected ones.

    Hashing just the known contract would silently discard an array one store
    returns and another does not, so the comparison would call two different
    results identical. The expected arrays must still be present; the digest map's
    keys are otherwise the result's own keys, so a comparison reports a one-sided
    extra array by name.
    """
    missing = [name for name in RESULT_ARRAY_NAMES if name not in result]
    if missing:
        raise IncompleteResultError(
            f"query result is missing array(s) {missing}; refusing an identity check "
            "that cannot see every array it claims to compare"
        )
    return {name: digest_array(values) for name, values in result.items()}


def _spread(values: list[Any], count: int) -> list[Any]:
    """`count` items spread evenly over `values`, first and last included."""
    if len(values) <= count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(i)] for i in positions]


def _table_digests(table: dict[int, dict[str, Any]]) -> dict[str, str]:
    """A `variants_table()` or `analyses_table()` as one digest per column."""
    keys = sorted(table)
    columns = sorted({name for row in table.values() for name in row})
    out = {"__keys__": digest_array(np.asarray(keys, dtype="int64"))}
    for name in columns:
        values = np.empty(len(keys), dtype=object)
        values[:] = [repr(table[key].get(name)) for key in keys]
        out[name] = digest_array(values)
    return out


def query_shapes(
    q: Any, variants: dict[int, dict[str, Any]] | None = None
) -> dict[str, Callable[[], Any]]:
    """Every association-returning question the facade answers, by a stable name.

    `variants` is `q.variants_table()`, passed when the caller already read it.
    """
    variants = q.variants_table() if variants is None else variants
    analyses = q.analyses_table()
    ids = _spread([row["analysis_id"] for _, row in sorted(analyses.items())], N_ANALYSES_ASKED)
    records = [variants[key] for key in sorted(variants)]
    identifiers = _identifiers(_spread(records, N_VARIANTS_ASKED))
    shapes: dict[str, Callable[[], Any]] = {}
    for observed_only, flag in ((False, "all"), (True, "observed_only")):
        asking = _Asking(q, ids, identifiers, records[0], records[-1], observed_only)
        shapes.update({f"{name}/{flag}": shape for name, shape in asking.shapes().items()})
    return shapes


def _identifiers(asked: list[dict[str, Any]]) -> list[str]:
    """The asked variants' ALIDs, then the rsIDs of those that carry one."""
    return [r["alid"] for r in asked] + [r["rsid"] for r in asked if r["rsid"]]


@dataclass(frozen=True)
class _Asking:
    """The questions asked of one release with one `observed_only` flag."""

    q: Any
    ids: list[str]
    identifiers: list[str]
    first: dict[str, Any]
    last: dict[str, Any]
    observed_only: bool

    def shapes(self) -> dict[str, Callable[[], Any]]:
        q, o = self.q, self.observed_only
        shapes: dict[str, Callable[[], Any]] = {
            f"analysis/{a}": partial(q.analysis, a, observed_only=o) for a in self.ids
        }
        for identifier in [*self.identifiers, *ABSENT_IDENTIFIERS]:
            shapes[f"phewas/{identifier}"] = partial(q.phewas, identifier, observed_only=o)
        shapes["range_phewas"] = partial(
            q.range_phewas,
            self.first["chromosome"],
            self.first["position"],
            self.last["position"],
            observed_only=o,
        )
        shapes["lookup"] = partial(q.lookup, self.identifiers, self.ids, observed_only=o)
        shapes.update(self._top_hits())
        if hasattr(q, "range_by_analysis"):
            shapes["range_by_analysis"] = partial(
                q.range_by_analysis, self.first["chromosome"], 1, 10**9, observed_only=o
            )
        return shapes

    def _top_hits(self) -> dict[str, Callable[[], Any]]:
        q, o = self.q, self.observed_only
        shapes: dict[str, Callable[[], Any]] = {}
        for threshold in THRESHOLDS:
            shapes[f"top_hits/{threshold}"] = partial(
                q.top_hits, threshold=threshold, observed_only=o
            )
            for a in self.ids:
                shapes[f"top_hits/{threshold}/{a}"] = partial(
                    q.top_hits, analysis_id=a, threshold=threshold, observed_only=o
                )
        shapes["top_hits/limit"] = partial(q.top_hits, threshold=0.2, limit=3, observed_only=o)
        return shapes


#: The shapes whose answers are also resolved to rows.
RESOLVED_SHAPES = ("lookup/", "top_hits/0.2/all", "range_phewas/")


def _resolved(q: Any, result: Any) -> tuple[dict[str, str], list[float]]:
    """A result's resolved rows: every field digested but `log10_p`, which is returned."""
    rows = list(q.resolve(result, include_variant_info=True))
    names = sorted({name for row in rows for name in row} - {"log10_p"})
    digests = {}
    for name in names:
        values = np.empty(len(rows), dtype=object)
        values[:] = [repr(row.get(name)) for row in rows]
        digests[name] = digest_array(values)
    return digests, [float(row["log10_p"]) for row in rows]


def probed_alids(variants: dict[int, dict[str, Any]]) -> list[str]:
    """The release's longest ALIDs, then the first at each of `PROBED_ALID_LENGTHS`."""
    alids = [variants[key]["alid"] for key in sorted(variants)]
    by_length = sorted(range(len(alids)), key=lambda i: (-len(alids[i].encode()), i))
    chosen = [alids[i] for i in by_length[:N_LONGEST_PROBED]]
    for length in PROBED_ALID_LENGTHS:
        at = next((a for a in alids if len(a.encode()) == length), None)
        if at is not None and at not in chosen:
            chosen.append(at)
    return chosen


#: Every range question `range_questions` can pose, in the order it names them.
RANGE_QUESTIONS: tuple[str, ...] = (
    "first",
    "to_first",
    "last",
    "from_last",
    "boundary/before",
    "boundary/after",
    "empty/between",
    "empty/chromosome",
    "shared",
    "shared/from_before",
    "shared/to_after",
)


def range_questions(variants: dict[int, dict[str, Any]]) -> dict[str, tuple[str, int, int]]:
    """The ranges the range probes ask, by name, as ``(chromosome, start, end)``.

    Chosen from the rows of `variants` (`variants_table()`), in row order, which is
    genomic order. A case the release cannot pose -- one chromosome, so no boundary;
    no shared position; no gap; every listed chromosome held -- is left out, and named
    by `unposed_range_questions`, which the rewrite logs and records in its provenance.
    """
    rows = [
        (str(variants[k]["chromosome"]), int(variants[k]["position"])) for k in sorted(variants)
    ]
    if not rows:
        return {}
    (first_chrom, first_pos), (last_chrom, last_pos) = rows[0], rows[-1]
    asked = {
        "first": (first_chrom, first_pos, first_pos),
        "to_first": (first_chrom, 1, first_pos),
        "last": (last_chrom, last_pos, last_pos),
        "from_last": (last_chrom, last_pos, CHROMOSOME_END),
    }
    asked.update(_boundary_questions(rows))
    asked.update(_empty_questions(rows))
    asked.update(_shared_questions(rows))
    return asked


def unposed_range_questions(asked: Mapping[str, Any]) -> list[str]:
    """The names in `RANGE_QUESTIONS` that `asked` (`range_questions`) leaves out, in order."""
    return [name for name in RANGE_QUESTIONS if name not in asked]


def _boundary_questions(rows: list[tuple[str, int]]) -> dict[str, tuple[str, int, int]]:
    """Either side of the first chromosome boundary: from the last variant before it to
    its chromosome's end, and from its chromosome's start to the first variant after it."""
    for (before_chrom, before_pos), (after_chrom, after_pos) in zip(rows, rows[1:], strict=False):
        if before_chrom != after_chrom:
            return {
                "boundary/before": (before_chrom, before_pos, CHROMOSOME_END),
                "boundary/after": (after_chrom, 1, after_pos),
            }
    return {}


def _empty_questions(rows: list[tuple[str, int]]) -> dict[str, tuple[str, int, int]]:
    """Between the first two variants of one chromosome with a position free between
    them, and on the first of `PROBED_CHROMOSOMES` the release lacks."""
    asked: dict[str, tuple[str, int, int]] = {}
    for (chrom, pos), (next_chrom, next_pos) in zip(rows, rows[1:], strict=False):
        if chrom == next_chrom and next_pos - pos >= 2:
            asked["empty/between"] = (chrom, pos + 1, next_pos - 1)
            break
    held = {chrom for chrom, _ in rows}
    lacked = next((chrom for chrom in PROBED_CHROMOSOMES if chrom not in held), None)
    if lacked is not None:
        asked["empty/chromosome"] = (lacked, 1, CHROMOSOME_END)
    return asked


def _shared_questions(rows: list[tuple[str, int]]) -> dict[str, tuple[str, int, int]]:
    """At the first position several variants share: the position alone, from the
    position before it on its chromosome, and to the position after it."""
    at = next((i for i in range(len(rows) - 1) if rows[i] == rows[i + 1]), None)
    if at is None:
        return {}
    chrom, pos = rows[at]
    before, after = _neighbouring_positions(rows, at)
    return {
        "shared": (chrom, pos, pos),
        "shared/from_before": (chrom, before, pos),
        "shared/to_after": (chrom, pos, after),
    }


def _neighbouring_positions(rows: list[tuple[str, int]], at: int) -> tuple[int, int]:
    """The positions either side of row `at`'s on its chromosome: the last before it
    (1 if none) and the first after it (`CHROMOSOME_END` if none)."""
    chrom, pos = rows[at]
    same = [p for c, p in rows if c == chrom]
    lower = [p for p in same if p < pos]
    higher = [p for p in same if p > pos]
    return max(lower, default=1), min(higher, default=CHROMOSOME_END)


def _range_digests(
    q: Any, asked: dict[str, tuple[str, int, int]]
) -> dict[str, dict[str, str]]:
    """Each range probe's `range_phewas`, Observed-Only and not, digested."""
    out = {}
    for name, (chrom, start, end) in asked.items():
        for observed_only, flag in ((False, "all"), (True, "observed_only")):
            result = q.range_phewas(chrom, start, end, observed_only=observed_only)
            out[f"range_phewas/{name}/{flag}"] = result_digests(dict(result))
    return out


def _probe_digests(q: Any, variants: dict[int, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Each probed ALID's PheWAS, and one lookup of them all, digested."""
    alids = probed_alids(variants)
    ids = _spread(
        [row["analysis_id"] for _, row in sorted(q.analyses_table().items())], N_ANALYSES_ASKED
    )
    out = {f"phewas/{alid}": result_digests(dict(q.phewas(alid))) for alid in alids}
    out["lookup"] = result_digests(dict(q.lookup(alids, ids)))
    return out


def _miss_digests(q: Any, variants: dict[int, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Each absent rsID's PheWAS, and a lookup of the spread identifiers with them, digested."""
    records = [variants[key] for key in sorted(variants)]
    identifiers = _identifiers(_spread(records, N_VARIANTS_ASKED))
    ids = _spread(
        [row["analysis_id"] for _, row in sorted(q.analyses_table().items())], N_ANALYSES_ASKED
    )
    out = {f"phewas/{rsid}": result_digests(dict(q.phewas(rsid))) for rsid in ABSENT_RSIDS}
    out["lookup"] = result_digests(dict(q.lookup([*identifiers, *ABSENT_RSIDS], ids)))
    return out


def release_digests(path: str | Path) -> dict[str, Any]:
    """Every answer the release at `path` gives, digested; `log10_p` per resolved shape."""
    from opengwasdb.query import query_store

    with query_store(path) as q:
        variants = q.variants_table()
        out: dict[str, Any] = {
            "adapter": type(q).__name__,
            "variants_table": _table_digests(variants),
            "analyses_table": _table_digests(q.analyses_table()),
            "shapes": {},
            "resolved": {},
            "log10_p": {},
        }
        for name, shape in query_shapes(q, variants).items():
            result = shape()
            out["shapes"][name] = result_digests(dict(result))
            if name.startswith(RESOLVED_SHAPES):
                out["resolved"][name], out["log10_p"][name] = _resolved(q, result)
        out["alid_probes"] = _probe_digests(q, variants)
        out["misses"] = _miss_digests(q, variants)
        asked = range_questions(variants)
        out["range_probes"] = _range_digests(q, asked)
        out["unposed_range_questions"] = unposed_range_questions(asked)
    return out


def differences(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Every place two `release_digests` disagree, by section, question and array; none if equal.

    `log10_p` is compared exactly, NaN at the same place counting as equal: both sides
    are read by this build, so a difference is a difference in what was stored.
    """
    found: list[str] = []
    if reference["adapter"] != candidate["adapter"]:
        found.append(f"adapter: {reference['adapter']} != {candidate['adapter']}")
    for section in ("variants_table", "analyses_table"):
        found += [f"{section}/{n}" for n in _differing(reference[section], candidate[section])]
    for section in ("shapes", "resolved", "alid_probes", "range_probes", "misses"):
        found += _shape_differences(section, reference[section], candidate[section])
    return found + _log10_p_differences(reference["log10_p"], candidate["log10_p"])


def _shape_differences(
    section: str, ours: dict[str, dict[str, str]], theirs: dict[str, dict[str, str]]
) -> list[str]:
    return [
        f"{section}/{shape}/{name}"
        for shape in sorted(set(ours) | set(theirs))
        for name in _differing(ours.get(shape, {}), theirs.get(shape, {}))
    ]


def _log10_p_differences(
    ours: dict[str, list[float]], theirs: dict[str, list[float]]
) -> list[str]:
    found = []
    for shape in sorted(set(ours) | set(theirs)):
        a = np.asarray(ours.get(shape, []), dtype=np.float64)
        b = np.asarray(theirs.get(shape, []), dtype=np.float64)
        if digest_array(a) != digest_array(b):
            found.append(f"log10_p/{shape}")
    return found


def _differing(reference: Mapping[str, str], candidate: Mapping[str, str]) -> list[str]:
    names = sorted(set(reference) | set(candidate))
    return [name for name in names if reference.get(name) != candidate.get(name)]


__all__ = [
    "IncompleteResultError",
    "RANGE_QUESTIONS",
    "RESULT_ARRAY_NAMES",
    "differences",
    "digest_array",
    "query_shapes",
    "range_questions",
    "release_digests",
    "result_digests",
    "unposed_range_questions",
]
