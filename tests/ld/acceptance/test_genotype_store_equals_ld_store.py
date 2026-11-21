"""The whole-panel acceptance of #91: the Genotype Store equals the LD Store, code for code.

`pixi run -e dev acceptance-genotypes` (CONTRIBUTING.md, "Acceptance runs"), on the IEU nodes.
It reads two stores over the same HGDP+1kGP v2 EUR panel — the materialised k = 10, × 511 LD
Store and the Genotype Store sidecar built from the same pgens — and compares every answer:

- the 28 frozen #81 comparisons: each #403 Analysis's chr22 hits, and the target Analysis over
  the whole panel;
- all 37 Analyses over the whole panel at r² 0.01 and 0.001 (74 requests; a refusal is compared
  by reason, and within 3 Mb the stores refuse 0.001 at their Partner Threshold);
- #19's chr22 suite: matrices with absent, malformed and duplicate names, a filled matrix, a
  region, proxies and clump at several thresholds, and the refusals;
- per chromosome, a 3 Mb region, and one whole-genome matrix of 1,000 seeded-random variants.

Every comparison is against the LD Store, code for code: the matrices' r and kinds, the regions',
the proxies' columns and the clumps' leads and joined r. An input that is unset or missing fails
the run; it never skips.

Environment (`tests/acceptance/run_genotypes.sh` sets the defaults):

- `OGLD_GENOTYPE_STORE`: the sidecar `ogld build --genotypes` wrote.
- `OGLD_81_EUR_STORE`: the k = 10, × 511 whole-panel LD Store over the same panel.
- `OGLD_403_ANALYSES`: #403's 37 Analyses, each with `hits.tsv` (ALID, z), as
  `scripts/realdata/make_403_inputs.py` writes them.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ogld
from ogld.identity import alid_locus

TARGET = "OGS-00012__GCST90476460"
KB = 3000
REGION_BP = 1_500_000
MATRIX_VARIANTS = 1_000
SEED = 91


def named(variable: str, what: str) -> Path:
    """The path `$variable` names. An acceptance run with an input missing fails: it was asked
    for, and a pass would claim it ran."""
    value = os.environ.get(variable, "")
    assert value, f"${variable} must name {what}; it is unset"
    assert Path(value).exists(), f"${variable} must name {what}; {value} is not there"
    return Path(value)


@pytest.fixture(scope="session")
def stores() -> tuple[ogld.LDStore, ogld.LDStore]:
    genotype = ogld.open(named("OGLD_GENOTYPE_STORE", "the Genotype Store sidecar"))
    ld = ogld.open(named("OGLD_81_EUR_STORE", "the k = 10 EUR LD Store over the same panel"))
    assert ld.provenance["adjustment"]["k"] == 10, "the two stores must be built alike"
    assert genotype.provenance["adjustment"]["k"] == 10
    return genotype, ld


@pytest.fixture(scope="session")
def analyses() -> list[tuple[str, list[str], np.ndarray]]:
    root = named("OGLD_403_ANALYSES", "#403's 37 Analyses' hits")
    paths = sorted(root.glob("*/hits.tsv"))
    assert len(paths) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(paths)}"
    loaded = []
    for path in paths:
        hits = np.loadtxt(path, dtype=str, skiprows=1, ndmin=2)
        assert hits.shape[0], f"{path} is not an ALID/score input"
        assert hits.shape[1] == 2, f"{path} is not an ALID/score input"
        loaded.append((path.parent.name, hits[:, 0].tolist(), np.abs(hits[:, 1].astype(float))))
    return loaded


def outcome(store: ogld.LDStore, method: str, *args: Any, **kwargs: Any) -> Any:
    """A query's answer, or its refusal by reason: a refusal is an answer to compare too."""
    try:
        return getattr(store, method)(*args, **kwargs)
    except ogld.Refusal as refused:
        return ("refusal", refused.reason)


def assert_equal(actual: object, expected: object, label: str) -> None:
    """The LD Store's answer and the sidecar's, field for field, with every code compared."""
    assert type(actual) is type(expected), f"{label}: {type(actual)} vs {type(expected)}"
    if isinstance(actual, tuple) and actual and actual[0] == "refusal":
        assert actual == expected, f"{label}: refusals differ"
        return
    for field in dataclasses.fields(actual):  # type: ignore[arg-type]
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=f"{label}: {field.name}")
        elif dataclasses.is_dataclass(a):
            assert_equal(a, b, f"{label}: {field.name}")
        else:
            assert a == b, f"{label}: {field.name} differs"


def compare(
    genotype: ogld.LDStore,
    ld: ogld.LDStore,
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    label: str,
) -> None:
    theirs = outcome(ld, method, *args, **kwargs)
    ours = outcome(genotype, method, *args, **kwargs)
    assert_equal(ours, theirs, label)
    if (
        method in ("matrix", "region")
        and kwargs.get("fill") is None
        and not isinstance(theirs, tuple)
    ):
        np.testing.assert_array_equal(ours.kinds(), theirs.kinds(), err_msg=f"{label}: kinds")


def chr22(alids: list[str]) -> list[int]:
    """The indexes of the ALIDs on chromosome 22, in request order."""
    out = []
    for i, alid in enumerate(alids):
        locus = alid_locus(alid)
        if locus is not None and locus[0] == "22":
            out.append(i)
    return out


def held(
    store: ogld.LDStore, alids: list[str], scores: np.ndarray, keep: list[int]
) -> tuple[list[str], np.ndarray]:
    """The kept ALIDs the store holds, and their scores, in request order."""
    names = [alids[i] for i in keep]
    absent = set(store.matrix(names).absent)
    present = [(a, s) for a, s in zip(names, scores[keep], strict=True) if a not in absent]
    return [a for a, _ in present], np.array([s for _, s in present])


def test_the_chr22_clumps_and_the_target_equal_the_ld_store(
    stores: tuple[ogld.LDStore, ogld.LDStore],
    analyses: list[tuple[str, list[str], np.ndarray]],
) -> None:
    genotype, ld = stores
    compared = 0
    for name, alids, scores in analyses:
        on_chr22 = chr22(alids)
        if not on_chr22:
            continue
        compare(
            genotype,
            ld,
            "clump",
            held(ld, alids, scores, on_chr22),
            {"r2": 0.01, "kb": KB},
            f"{name}: chr22",
        )
        compared += 1
    # The target over the whole panel, as #81 measured it.
    for name, alids, scores in analyses:
        if name == TARGET:
            compare(
                genotype, ld, "clump", (alids, scores), {"r2": 0.01, "kb": KB}, f"{name}: whole"
            )
            compared += 1
    assert compared == 28, f"#81 froze 28 comparisons; this run made {compared}"


def test_the_37_analyses_equal_the_ld_store_at_both_thresholds(
    stores: tuple[ogld.LDStore, ogld.LDStore],
    analyses: list[tuple[str, list[str], np.ndarray]],
) -> None:
    genotype, ld = stores
    compared = 0
    for r2 in (0.01, 0.001):
        for name, alids, scores in analyses:
            compare(
                genotype,
                ld,
                "clump",
                (alids, scores),
                {"r2": r2, "kb": KB},
                f"{name}: whole r2={r2}",
            )
            compared += 1
    assert compared == 74, (
        f"37 Analyses at two thresholds are 74 requests; this run made {compared}"
    )


def test_the_chr22_suite_equals_the_ld_store(
    stores: tuple[ogld.LDStore, ogld.LDStore],
    analyses: list[tuple[str, list[str], np.ndarray]],
) -> None:
    genotype, ld = stores
    _, alids, scores = next(case for case in analyses if case[0] == TARGET)
    on_chr22 = chr22(alids)
    assert len(on_chr22) > 10, "the target must have chr22 hits for this suite to mean anything"
    names, chr22_scores = held(ld, alids, scores, on_chr22)
    first = names[0]
    requests: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("matrix", (names[:10],), {}),
        ("matrix", (names[:10][::-1],), {}),
        ("matrix", ([first, "not-an-alid", first, names[1]],), {}),
        ("matrix", (["22:1:A:C"],), {}),
        ("matrix", ([],), {}),
        ("matrix", (names[:2],), {"fill": 0.0}),
        ("region", ("22", 25_000_000, REGION_BP), {}),
        ("region", ("22", 25_000_000, REGION_BP), {"fill": 0.0}),
        ("region", ("22", 32_000_000, 5_000_001), {}),
        ("proxies", ([first],), {"r2": 0.01, "kb": KB}),
        ("proxies", ([first],), {"r2": 0.01, "kb": 10_000}),
        ("proxies", ([first],), {"r2": 0.001, "kb": KB}),
        ("proxies", ([first],), {"r2": 0.8, "kb": 10_001}),
        ("clump", ([first], [1.0]), {"r2": 0.01, "kb": KB}),
        ("clump", ([first], [1.0]), {"r2": 0.001, "kb": KB}),
        ("clump", (names, chr22_scores), {"r2": 0.01, "kb": KB}),
        ("clump", (names, chr22_scores), {"r2": 0.01, "kb": 5000}),
    ]
    for method, args, kwargs in requests:
        compare(genotype, ld, method, args, kwargs, f"chr22 {method} {args[1:2]} {kwargs}")


def test_each_chromosome_s_region_and_a_whole_genome_matrix_equal_the_ld_store(
    stores: tuple[ogld.LDStore, ogld.LDStore],
) -> None:
    genotype, ld = stores
    provenance = ld.provenance
    count = provenance["n_variants"]
    reader = ld._reader
    chromosomes = list(reader._ranges)
    assert len(chromosomes) >= 22, f"the EUR panel must hold the autosomes; it holds {chromosomes}"
    for chrom in chromosomes:
        start, stop = reader.chromosome_range(chrom)
        first, last = reader.loci_of(np.array([start, stop - 1], dtype=np.int64))[1]
        centre = int((int(first) + int(last)) // 2)
        compare(
            genotype,
            ld,
            "region",
            (chrom, centre, REGION_BP),
            {},
            f"{chrom}: 3 Mb region at {centre}",
        )
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(count, size=MATRIX_VARIANTS, replace=False))
    names = ld._reader.alids_of(rows).tolist()
    assert len(names) == MATRIX_VARIANTS
    compare(genotype, ld, "matrix", (names,), {}, "whole genome: 1,000 seeded variants")
