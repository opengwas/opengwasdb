"""What a Store Release answers, as digests: the proof that a rewrite changed no answer.

A rewrite (`opengwasdb.store.rewrite`) changes how a release is stored, never what
it answers. Before it publishes, it asks the source and the staged release the same
deterministic set of questions and refuses on any difference. This module is that
set of questions and the digest of each answer. It has one owner, here: the Store
Reader contract bundle's expected answers are its output (`tests/store_reader_contract/`,
#474), and the store-comparison benchmark calls it for `digest_array` and
`result_digests`.

Every association-returning question the query facade answers is asked: each
Analysis's associations; a PheWAS of variants spread over the axis, by ALID and by
rsID, and of two identifiers no release holds; a range PheWAS; a lookup; and the
top hits at five thresholds, overall and per Analysis, both Observed-Only and with
imputed cells. A Dense or Hybrid release refuses the two thresholds it stores no tier
for (#483); the digest of that answer is of the refusal's type and message
(`refusal_digest`), so the proof compares the refusal where it cannot compare rows.
Every array each answer holds is digested (`result_digests`), with
the `variants_table()` and `analyses_table()` column by column, and the lookup, range
and widest top-hit answers resolved to rows, with variant info. Both, and the variant-only
reads, name each row's rsID and `source_alid`, so every row's `source_alid` is compared through
`variants_table()`, and the rows an answer reaches through its resolved rows: a 0.2
release reads them from its Variant Table, a 0.3.0 one from its sidecars (#469, #470).

**Values, not digests: `pval` and `log10_p`.** Every answer's `pval`, and a resolved row's
`pval` and `log10_p`, are kept as numbers (`VALUES`), not digested. Both are the libm's
(`store_reader.stats`: `erfc`, `erf`, `log` and `log1p`), which differ in the last ulp
between one C library and another, so answers written on one machine and read on another
are compared with `differences(..., rtol=CROSS_MACHINE_RTOL)`: four float64 ulps, NaN at
the same place counting as equal. They are the only values so derived: every other float
an answer holds is stored, or `beta`, ``z * se``, which IEEE rounds the same everywhere.
Between two releases read by one build -- the rewrite's proof -- `differences` compares
them exactly, its default: there a difference is a difference in what was stored.
`readable_differences` compares readable rows by the same rule.

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

The **refusals** ask what the contract answers with an error (spec §18, §19): `top_hits`,
`analysis()` and `lookup()` of an Analysis the release does not hold
(`UnknownAnalysisError`) and a range on a chromosome no ALID can spell
(`UnsupportedChromosome`). A refusal's digest is of its type and message
(`refusal_digest`); a refusal that answers instead raises, since only the refusal is
expected, so `differences` never has to compare an answer with a refusal.

**One enumeration.** `questions` yields every question once, with the errors that are its
answer; `release_answers` asks each once per open and renders the answer twice: its
digest, and, for a reader in another language that cannot reproduce Python's digests,
its readable rows (`readable_answer`: the row count, the first rows, and the first row of
each `association_status`; a refusal's type and named `fields`). `release_digests`, the
rewrite's proof, is the first rendering alone. A release is opened through `query_store`
unless a caller passes its own `query`: the Store Reader contract test (#474,
`tests/store_reader_contract/`) opens it with the Store Reader alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.store_reader.analyses import UnknownAnalysisError
from opengwasdb.store_reader.associations import Associations
from opengwasdb.store_reader.identity import VariantNormalisationError
from opengwasdb.store_reader.indexed import list_indexed_subsets
from opengwasdb.store_reader.top_hits import UnstoredThresholdError

#: Opens a release for its questions: the query facade over the release at a path.
QueryOpener = Callable[[Path], Any]

#: The arrays every association result holds; a digest that could not see one of them
#: could call two different answers equal.
RESULT_ARRAY_NAMES: tuple[str, ...] = tuple(Associations.empty())

#: The thresholds asked of `top_hits`: the three a Top-Hit Index holds, and two
#: it does not, which a Ragged release answers by scanning (`z_critical`) and a Dense
#: or Hybrid one refuses.
THRESHOLDS = (5e-8, 5e-6, 5e-4, 1e-2, 0.2)

#: How many variants the PheWAS and lookup shapes name, spread over the axis.
N_VARIANTS_ASKED = 7

#: The most variants the range PheWAS spans, and the most cells (variants by Analyses)
#: it resolves (`_range_last`): every fixture release has fewer, so each is asked over
#: the whole span, and so is OGS-00015 (241 Analyses) over 10,000 variants.
RANGE_ROWS_ASKED = 10_000
RANGE_CELLS_ASKED = 2_500_000

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

#: An Analysis no release holds, and a chromosome no ALID can spell: a number outside
#: 1-26 (`identity.normalise_chromosome`). The refusals ask them.
UNKNOWN_ANALYSIS = "absent"
UNSUPPORTED_CHROMOSOME = "27"

#: The errors a refusal question may raise; any other is a failure, not an answer.
REFUSAL_ERRORS = (UnknownAnalysisError, VariantNormalisationError)

#: The fields kept as numbers rather than digested (module docstring): every answer's
#: `pval`, keyed ``<section>/<question>`` (a resolved row's under ``resolved/``), and a
#: resolved row's `log10_p`, keyed by its question.
PVAL = "pval"
LOG10_P = "log10_p"
VALUES = (LOG10_P, PVAL)

#: How far apart two `VALUES` read on different machines may be, relatively: four
#: float64 ulps. The libm differs between C libraries in the last ulp (#474); a
#: relative change of 1e-9 is still a difference.
CROSS_MACHINE_RTOL = 4 * float(np.finfo(np.float64).eps)


class IncompleteResultError(ValueError):
    """A query result lacks one of `RESULT_ARRAY_NAMES`, so its digest would not see it."""


#: The thresholds `THRESHOLDS` asks that no Top-Hit Index stores, and the `top_hits`
#: shapes that ask them.
UNSTORED_THRESHOLDS = (1e-2, 0.2)
UNSTORED_SHAPES = tuple(f"top_hits/{t}" for t in UNSTORED_THRESHOLDS) + ("top_hits/limit",)


def expects_refusal(q: Any, shape: str) -> bool:
    """Whether `shape` is one `q` is expected to refuse: a Dense or Hybrid release asked
    `top_hits` at a threshold it stores no tier for (#483). Any other refusal is a failure."""
    return adapter_name(q) in ("StoreQuery", "HybridStoreQuery") and shape.startswith(
        UNSTORED_SHAPES
    )


def adapter_name(q: Any) -> str:
    """The query facade `q` is: its class's name, or a private subclass's public facade
    (the legacy conversion asks its source through one, `legacy_expected`, #492)."""
    return next(cls.__name__ for cls in type(q).__mro__ if not cls.__name__.startswith("_"))


def refusal_digest(refused: Exception) -> dict[str, str]:
    """A refused question's answer: one `error` digest of the exception's type and message."""
    text = f"{type(refused).__name__}: {refused}"
    return {"error": hashlib.sha256(text.encode()).hexdigest()}


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
        last = _range_last(records, len(analyses))
        asking = _Asking(q, ids, identifiers, records[0], last, observed_only)
        shapes.update({f"{name}/{flag}": shape for name, shape in asking.shapes().items()})
    return shapes


def _range_last(records: list[dict[str, Any]], n_analyses: int) -> dict[str, Any]:
    """Where the range PheWAS ends: the last variant's position, read on the first
    variant's chromosome, unless the range would then hold more variants than
    `RANGE_ROWS_ASKED`, or than `RANGE_CELLS_ASKED` over the release's Analyses, when it
    ends at the last variant within both bounds (#492).

    A range question over a large release's whole first chromosome resolves tens of
    millions of rows on each side (ADR 0105 §3: 844 M cells on OGS-00016); bounded, it
    still asks the range path over many rows, and every fixture release asks it unbounded.
    """
    limit = max(1, min(RANGE_ROWS_ASKED, RANGE_CELLS_ASKED // max(n_analyses, 1)))
    first, last = records[0], records[-1]
    positions = [
        int(r["position"])
        for r in records
        if r["chromosome"] == first["chromosome"] and int(r["position"]) <= int(last["position"])
    ]
    if len(positions) <= limit:
        return last
    return {**last, "position": positions[limit - 1]}


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


def _resolved(q: Any, result: Any) -> tuple[dict[str, str], dict[str, list[float]]]:
    """A result's resolved rows: every field digested but `VALUES`, which are returned."""
    rows = list(q.resolve(result, include_variant_info=True))
    names = sorted({name for row in rows for name in row} - set(VALUES))
    digests = {}
    for name in names:
        values = np.empty(len(rows), dtype=object)
        values[:] = [repr(row.get(name)) for row in rows]
        digests[name] = digest_array(values)
    return digests, {kind: [float(row[kind]) for row in rows] for kind in VALUES}


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


def refusal_questions(q: Any) -> dict[str, Callable[[], Any]]:
    """The questions the contract answers with an error, by a stable name (module
    docstring)."""
    return {
        "top_hits/unknown_analysis": partial(
            q.top_hits, analysis_id=UNKNOWN_ANALYSIS, threshold=THRESHOLDS[0]
        ),
        "analysis/unknown_analysis": partial(q.analysis, UNKNOWN_ANALYSIS),
        "lookup/unknown_analysis": partial(q.lookup, list(ABSENT_IDENTIFIERS), [UNKNOWN_ANALYSIS]),
        "range_phewas/unsupported_chromosome": partial(
            q.range_phewas, UNSUPPORTED_CHROMOSOME, 1, CHROMOSOME_END
        ),
        "variants_in_range/unsupported_chromosome": partial(
            q.variants_in_range, UNSUPPORTED_CHROMOSOME, 1, CHROMOSOME_END
        ),
    }


#: The sections of `release_answers` that `questions` asks into, beside the two tables.
SECTIONS = (
    "shapes", "alid_probes", "variant_reads", "indexed_subsets", "misses", "range_probes",
    "refusals",
)


@dataclass(frozen=True)
class Question:
    """One question asked of a release: the section of the answers it belongs to, its
    name there, how to ask it, the errors that are its answer rather than a failure, and
    whether its rows are also resolved (`RESOLVED_SHAPES`)."""

    section: str
    name: str
    ask: Callable[[], Any]
    refusals: tuple[type[Exception], ...] = ()
    resolved: bool = False


def questions(
    q: Any, variants: dict[int, dict[str, Any]], path: str | Path
) -> Iterator[Question]:
    """Every question asked of the release `q` reads, once, in a fixed order: the one
    enumeration both renderings of `release_answers` -- digests and readable rows -- read,
    so a question added here is in both. `variants` is `q.variants_table()`."""
    ids = _spread(
        [row["analysis_id"] for _, row in sorted(q.analyses_table().items())], N_ANALYSES_ASKED
    )
    records = [variants[key] for key in sorted(variants)]
    identifiers = _identifiers(_spread(records, N_VARIANTS_ASKED))
    asked = range_questions(variants)
    yield from _shape_questions(q, variants)
    yield from _probe_questions(q, variants, ids)
    yield from _variant_read_questions(q, identifiers, asked)
    yield from _subset_questions(q, path, ids)
    yield from _miss_questions(q, identifiers, ids)
    yield from _range_probe_questions(q, asked)
    for name, ask in refusal_questions(q).items():
        yield Question("refusals", name, ask, REFUSAL_ERRORS)


def _shape_questions(q: Any, variants: dict[int, dict[str, Any]]) -> Iterator[Question]:
    """`query_shapes`, with the refusal a Dense or Hybrid release gives an unstored tier."""
    for name, shape in query_shapes(q, variants).items():
        refused = (UnstoredThresholdError,) if expects_refusal(q, name) else ()
        yield Question("shapes", name, shape, refused, name.startswith(RESOLVED_SHAPES))


def _probe_questions(
    q: Any, variants: dict[int, dict[str, Any]], ids: list[str]
) -> Iterator[Question]:
    """Each probed ALID's PheWAS, and one lookup of them all (module docstring)."""
    alids = probed_alids(variants)
    for alid in alids:
        yield Question("alid_probes", f"phewas/{alid}", partial(q.phewas, alid))
    yield Question("alid_probes", "lookup", partial(q.lookup, alids, ids))


def _variant_read_questions(
    q: Any, identifiers: list[str], asked: dict[str, tuple[str, int, int]]
) -> Iterator[Question]:
    """The variant-only reads (#484): the count, the spread and absent identifiers' rows,
    and the rows of every range probe."""
    yield Question("variant_reads", "variant_count", q.variant_count)
    yield Question(
        "variant_reads", "variants", partial(q.variants, [*identifiers, *ABSENT_IDENTIFIERS])
    )
    for name, (chrom, start, end) in asked.items():
        yield Question(
            "variant_reads", f"variants_in_range/{name}",
            partial(q.variants_in_range, chrom, start, end),
        )


def _miss_questions(q: Any, identifiers: list[str], ids: list[str]) -> Iterator[Question]:
    """Each absent rsID's PheWAS, and a lookup of the spread identifiers with them."""
    for rsid in ABSENT_RSIDS:
        yield Question("misses", f"phewas/{rsid}", partial(q.phewas, rsid))
    yield Question("misses", "lookup", partial(q.lookup, [*identifiers, *ABSENT_RSIDS], ids))


def _range_probe_questions(
    q: Any, asked: dict[str, tuple[str, int, int]]
) -> Iterator[Question]:
    """Each range probe's `range_phewas`, Observed-Only and not."""
    for name, (chrom, start, end) in asked.items():
        for observed_only, flag in ((False, "all"), (True, "observed_only")):
            yield Question(
                "range_probes", f"range_phewas/{name}/{flag}",
                partial(q.range_phewas, chrom, start, end, observed_only=observed_only),
            )


def _subset_questions(q: Any, path: str | Path, ids: list[str]) -> Iterator[Question]:
    """Each Indexed Variant Subset's answer for the spread Analyses, Observed-Only and
    not (ADR 0061): a subset records its parent release, so a derived release that
    carries it on must answer through it as its source did."""
    for subset in list_indexed_subsets(path):
        for analysis_id in ids:
            for observed_only, flag in ((False, "all"), (True, "observed_only")):
                yield Question(
                    "indexed_subsets", f"{subset}/{analysis_id}/{flag}",
                    partial(
                        q.analysis, analysis_id, observed_only=observed_only,
                        indexed_subset=subset,
                    ),
                )


def _walk(
    q: Any, variants: dict[int, dict[str, Any]], path: str | Path
) -> Iterator[tuple[Question, Any]]:
    """Each question with its answer: what it returned, or the refusal it is expected to
    raise. Any other error is a failure, raised."""
    for question in questions(q, variants, path):
        try:
            yield question, question.ask()
        except question.refusals as refused:
            yield question, refused


def _digested(answer: Any) -> dict[str, str]:
    """One answer digested: a refusal's type and message, an association answer's
    arrays, a count, or a variant-only read's rows column by column."""
    if isinstance(answer, Exception):
        return refusal_digest(answer)
    if isinstance(answer, Associations):
        digests = result_digests(dict(answer))
        del digests[PVAL]  # kept as values (`_render`)
        return digests
    if isinstance(answer, int):
        return {"count": digest_array(np.asarray([answer], dtype="int64"))}
    return _table_digests(answer)


def _open(path: str | Path, query: QueryOpener | None) -> Any:
    if query is None:
        from opengwasdb.query import query_store

        query = query_store
    return query(Path(path))


def release_answers(
    path: str | Path, *, query: QueryOpener | None = None, readable: bool = True
) -> dict[str, Any]:
    """Every answer the release at `path` gives, asked once (`questions`) and rendered
    twice: ``digests`` (`release_digests`) and, with `readable`, ``readable``
    (`readable_answer`). `query` opens the release (`query_store` by default; module
    docstring)."""
    with _open(path, query) as q:
        variants = q.variants_table()
        analyses = q.analyses_table()
        digests: dict[str, Any] = {
            "adapter": adapter_name(q),
            "variants_table": _table_digests(variants),
            "analyses_table": _table_digests(analyses),
            "resolved": {},
            **{kind: {} for kind in VALUES},
            **{section: {} for section in SECTIONS},
            "unposed_range_questions": unposed_range_questions(range_questions(variants)),
        }
        shown: dict[str, Any] = {
            "variants_table": _readable_variants(variants),
            "analyses_table": readable_answer(analyses, len(analyses), key="analysis_index"),
            **{section: {} for section in SECTIONS},
        }
        for question, answer in _walk(q, variants, path):
            _render(q, question, answer, digests, shown if readable else None)
    return {"digests": digests, "readable": shown} if readable else {"digests": digests}


def _render(
    q: Any, question: Question, answer: Any, digests: dict[str, Any], shown: dict[str, Any] | None
) -> None:
    """Put one answer in both renderings: its digest (and resolved rows, where the question
    asks for them), and its readable rows unless `shown` is None."""
    digests[question.section][question.name] = _digested(answer)
    if isinstance(answer, Associations):
        digests[PVAL][f"{question.section}/{question.name}"] = [float(p) for p in answer[PVAL]]
    if question.resolved and not isinstance(answer, Exception):
        digests["resolved"][question.name], values = _resolved(q, answer)
        digests[LOG10_P][question.name] = values[LOG10_P]
        digests[PVAL][f"resolved/{question.name}"] = values[PVAL]
    if shown is not None:
        shown[question.section][question.name] = _readable(answer)


def release_digests(path: str | Path, *, query: QueryOpener | None = None) -> dict[str, Any]:
    """Every answer the release at `path` gives, digested; `VALUES` kept as numbers.
    The rewrite's proof (`release_answers`, without the readable rows)."""
    digests: dict[str, Any] = release_answers(path, query=query, readable=False)["digests"]
    return digests


#: How many of each answer's first rows `readable_answer` spells out.
READABLE_ROWS = 2


def readable_value(value: Any) -> Any:
    """`value` as JSON can hold it exactly: a float as itself, NaN as ``"NaN"`` and an
    infinity as ``"Infinity"`` or ``"-Infinity"`` (JSON has neither), a numpy scalar as
    the Python value it holds."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _readable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: readable_value(v) for name, v in sorted(row.items())}


def readable_answer(
    answer: Any, rows: int = READABLE_ROWS, *, key: str = "variant_index"
) -> dict[str, Any]:
    """How many rows `answer` holds and its first `rows`, spelt out: an association
    answer's arrays row by row, or a table's rows (a variant-only read's, keyed by
    `variant_index`; `analyses_table()`'s, by `analysis_index`) with their `key`.

    An association answer also spells out, under ``by_status``, the first row of each
    `association_status` it holds -- observed, imputed, missing -- with its place in the
    answer (``at``), so each status is read in every answer that holds it, not only where
    it falls among the first rows."""
    if isinstance(answer, Associations):
        names = sorted(answer)
        n = len(answer["z"])

        def row(i: int) -> dict[str, Any]:
            return _readable_row({name: answer[name][i] for name in names})

        statuses = answer["association_status"]
        firsts = {str(s): int(np.flatnonzero(statuses == s)[0]) for s in np.unique(statuses)}
        return {
            "rows": n,
            "head": [row(i) for i in range(min(n, rows))],
            "by_status": {status: {"at": i, **row(i)} for status, i in sorted(firsts.items())},
        }
    keys = sorted(answer)
    return {"rows": len(keys), "head": [_readable_row({key: k, **answer[k]}) for k in keys[:rows]]}


def _readable_refusal(refused: Exception) -> dict[str, Any]:
    """A refusal as a port compares it: its type and its named fields (`fields`); the
    message is for people."""
    fields = getattr(refused, "fields", {})
    return {
        "error": type(refused).__name__,
        "fields": {name: readable_value(v) for name, v in sorted(fields.items())},
        "message": str(refused),
    }


def _readable(answer: Any) -> Any:
    if isinstance(answer, Exception):
        return _readable_refusal(answer)
    if isinstance(answer, int):
        return {"count": answer}
    return readable_answer(answer)


def _readable_variants(variants: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The Variant Table's first rows and the rows of the probed ALIDs."""
    by_alid = {row["alid"]: key for key, row in variants.items()}
    probed = {by_alid[alid]: variants[by_alid[alid]] for alid in probed_alids(variants)}
    first = {key: variants[key] for key in sorted(variants)[:READABLE_ROWS]}
    return {
        "rows": len(variants),
        "head": readable_answer(first)["head"],
        "probed": readable_answer(probed, len(probed))["head"],
    }


def differences(
    reference: dict[str, Any], candidate: dict[str, Any], *, rtol: float = 0.0
) -> list[str]:
    """Every place two `release_digests` disagree, by section, question and array; none if equal.

    `VALUES` -- `log10_p` and `pval` -- are compared as values, by one rule, NaN at the same
    place counting as equal (`values_agree`). By default exactly: both sides are read by
    this build, so a difference is a difference in what was stored. Answers read on
    another machine pass `rtol=CROSS_MACHINE_RTOL` (module docstring).
    """
    found: list[str] = []
    if reference["adapter"] != candidate["adapter"]:
        found.append(f"adapter: {reference['adapter']} != {candidate['adapter']}")
    for section in ("variants_table", "analyses_table"):
        found += [f"{section}/{n}" for n in _differing(reference[section], candidate[section])]
    for section in (
        "shapes", "resolved", "alid_probes", "range_probes", "misses", "variant_reads", "refusals"
    ):
        found += _shape_differences(section, reference[section], candidate[section])
    found += _shape_differences(
        "indexed_subsets",
        reference.get("indexed_subsets", {}),
        candidate.get("indexed_subsets", {}),
    )
    for kind in VALUES:
        found += _value_differences(kind, reference[kind], candidate[kind], rtol)
    return found


def _shape_differences(
    section: str, ours: dict[str, dict[str, str]], theirs: dict[str, dict[str, str]]
) -> list[str]:
    return [
        f"{section}/{shape}/{name}"
        for shape in sorted(set(ours) | set(theirs))
        for name in _differing(ours.get(shape, {}), theirs.get(shape, {}))
    ]


def _value_differences(
    kind: str, ours: dict[str, list[Any]], theirs: dict[str, list[Any]], rtol: float
) -> list[str]:
    return [
        f"{kind}/{shape}"
        for shape in sorted(set(ours) | set(theirs))
        if not values_agree(ours.get(shape, []), theirs.get(shape, []), rtol=rtol)
    ]


def values_agree(ours: Sequence[Any], theirs: Sequence[Any], *, rtol: float = 0.0) -> bool:
    """Whether two runs of floats agree: as many, NaN at the same places, and each other
    value equal or, with `rtol`, within `rtol` of the larger in magnitude. A value may be
    spelt as `readable_value` spells it (``"NaN"``). With `rtol` 0 the comparison is of
    the bytes (`digest_array`), so 0.0 and -0.0 differ."""
    a = np.asarray([float(v) for v in ours], dtype=np.float64)
    b = np.asarray([float(v) for v in theirs], dtype=np.float64)
    if rtol == 0.0:
        return digest_array(a) == digest_array(b)
    if a.shape != b.shape:
        return False
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(a[~finite], b[~finite], equal_nan=True):
        return False  # an infinity against a finite value, or against the other infinity
    a, b = a[finite], b[finite]
    return bool(np.all(np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b))))


def readable_differences(
    reference: dict[str, Any], candidate: dict[str, Any], *, rtol: float = 0.0
) -> list[str]:
    """Every readable answer two `release_answers` disagree on, as
    ``readable/<section>/<question>``; none if equal. Every value is compared exactly
    but a row's `VALUES` field, which is compared as `differences` compares it."""
    found = []
    for section in sorted(set(reference) | set(candidate)):
        ours, theirs = reference.get(section), candidate.get(section)
        if not (isinstance(ours, dict) and isinstance(theirs, dict)):
            if not _readable_agree(ours, theirs, rtol):
                found.append(f"readable/{section}/(whole section)")
            continue
        found += [
            f"readable/{section}/{name}"
            for name in sorted(set(ours) | set(theirs))
            if not _readable_agree(ours.get(name), theirs.get(name), rtol)
        ]
    return found


def _readable_agree(ours: Any, theirs: Any, rtol: float, key: str | None = None) -> bool:
    """Whether two readable values agree, a `VALUES` field by `values_agree`."""
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return _readable_rows_agree(ours, theirs, rtol)
    if isinstance(ours, list) and isinstance(theirs, list):
        return _readable_lists_agree(ours, theirs, rtol, key)
    if key in VALUES and _is_float(ours) and _is_float(theirs):
        return values_agree([ours], [theirs], rtol=rtol)
    return bool(ours == theirs)


def _readable_lists_agree(ours: list[Any], theirs: list[Any], rtol: float, key: str | None) -> bool:
    if len(ours) != len(theirs):
        return False
    return all(_readable_agree(a, b, rtol, key) for a, b in zip(ours, theirs, strict=True))


def _readable_rows_agree(ours: dict[str, Any], theirs: dict[str, Any], rtol: float) -> bool:
    if set(ours) != set(theirs):
        return False
    return all(_readable_agree(ours[k], theirs[k], rtol, k) for k in ours)


def _is_float(value: Any) -> bool:
    """A float as `readable_value` spells one: itself, or one of JSON's missing names."""
    return isinstance(value, float) or value in ("NaN", "Infinity", "-Infinity")


def _differing(reference: Mapping[str, str], candidate: Mapping[str, str]) -> list[str]:
    names = sorted(set(reference) | set(candidate))
    return [name for name in names if reference.get(name) != candidate.get(name)]


__all__ = [
    "CROSS_MACHINE_RTOL",
    "IncompleteResultError",
    "QueryOpener",
    "RANGE_QUESTIONS",
    "RESULT_ARRAY_NAMES",
    "differences",
    "digest_array",
    "query_shapes",
    "range_questions",
    "Question",
    "SECTIONS",
    "questions",
    "readable_answer",
    "readable_differences",
    "readable_value",
    "refusal_questions",
    "release_answers",
    "release_digests",
    "result_digests",
    "unposed_range_questions",
    "values_agree",
    "VALUES",
]
