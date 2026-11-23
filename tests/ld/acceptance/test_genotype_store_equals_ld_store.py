"""The whole-panel acceptance of #91: the Genotype Store equals the LD Store, code for code.

`pixi run -e dev acceptance-genotypes` (CONTRIBUTING.md, "Acceptance runs"), on the IEU nodes.
It reads two stores over the same HGDP+1kGP v2 EUR panel — the materialised k = 10, × 511 LD
Store and the Genotype Store sidecar built from the same pgens — and compares every answer:

- the 28 frozen #81 comparisons: each #403 Analysis's chr22 hits, and the target Analysis over
  the whole panel;
- all 37 Analyses over the whole panel at r² 0.01 and 0.001 (74 requests; a refusal is compared
  by reason, and within 3 Mb the stores refuse 0.001 at their Partner Threshold);
- #19's 50 frozen chr22 requests (`chr22_workload_19.json`, from
  `9448650:prototypes/19_genotype_store/results/workload.json`): the 500-variant matrix, the
  region, both proxies, the 37 Analyses' chr22 clumps, and the ten edge and refusal requests;
- per chromosome, a 3 Mb region, plus one whole-genome matrix of 1,000 seeded-random variants.

Every comparison is against the LD Store, code for code: the matrices' r and kinds, the regions',
the proxies' columns and the clumps' leads and joined r. A comparison whose answers hold nothing
proves nothing, so every one asserts its answer is non-empty (or, for the deliberately empty edge
requests, exactly what emptiness it expects) before comparing. An input that is unset or missing
fails the run; it never skips.

Environment (`tests/acceptance/run_genotypes.sh` sets the defaults):

- `OPENGWASDB_LD_GENOTYPE_STORE`: the sidecar `ogld build --genotypes` wrote.
- `OPENGWASDB_LD_81_EUR_STORE`: the k = 10, × 511 whole-panel LD Store over the same panel.
- `OPENGWASDB_LD_403_ANALYSES`: #403's 37 Analyses, each with `hits.tsv` (ALID, z), as
  `scripts/realdata/make_403_inputs.py` writes them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import opengwasdb.ld
from opengwasdb.variants.sorted_spelling import sorted_alid_locus

from .named_inputs import named

TARGET = "OGS-00012__GCST90476460"
KB = 3000
REGION_BP = 1_500_000
MATRIX_VARIANTS = 1_000
SEED = 91
FROZEN = Path(__file__).with_name("chr22_workload_19.json")


@pytest.fixture(scope="session")
def stores() -> tuple[opengwasdb.ld.LDStore, opengwasdb.ld.LDStore]:
    genotype = opengwasdb.ld.open(
        named("OPENGWASDB_LD_GENOTYPE_STORE", "the Genotype Store sidecar")
    )
    ld = opengwasdb.ld.open(
        named("OPENGWASDB_LD_81_EUR_STORE", "the k = 10 EUR LD Store over the same panel")
    )
    assert ld.provenance["adjustment"]["k"] == 10, "the two stores must be built alike"
    assert genotype.provenance["adjustment"]["k"] == 10
    return genotype, ld


@pytest.fixture(scope="session")
def analyses() -> list[tuple[str, list[str], np.ndarray]]:
    root = named("OPENGWASDB_LD_403_ANALYSES", "#403's 37 Analyses' hits")
    paths = sorted(root.glob("*/hits.tsv"))
    assert len(paths) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(paths)}"
    loaded = []
    for path in paths:
        hits = np.loadtxt(path, dtype=str, skiprows=1, ndmin=2)
        assert hits.shape[0], f"{path} is not an ALID/score input"
        assert hits.shape[1] == 2, f"{path} is not an ALID/score input"
        loaded.append((path.parent.name, hits[:, 0].tolist(), np.abs(hits[:, 1].astype(float))))
    return loaded


@pytest.fixture(scope="session")
def frozen() -> dict[str, Any]:
    """#19's frozen chr22 workload, and the input hashes it names."""
    spec: dict[str, Any] = json.loads(FROZEN.read_text())
    assert spec["frozen_from"].endswith("results/workload.json")
    assert len(spec["matrix"]) == 500
    assert len(spec["analyses"]) == 37
    return spec


def outcome(store: opengwasdb.ld.LDStore, method: str, *args: Any, **kwargs: Any) -> Any:
    """A query's answer, or its refusal by reason: a refusal is an answer to compare too."""
    try:
        return getattr(store, method)(*args, **kwargs)
    except opengwasdb.ld.Refusal as refused:
        return ("refusal", refused.reason)


def assert_meaningful(answer: Any, method: str, label: str) -> None:
    """An empty pair of answers passes both ways and proves nothing (CLAUDE.md)."""
    if isinstance(answer, tuple):
        return
    if method in ("matrix", "region"):
        assert answer.r.size, f"{label}: the answer has no row"
        assert int(np.isfinite(answer.r).sum()) > 1, f"{label}: the answer holds no pair"
    elif method == "proxies":
        assert len(answer) > 0, f"{label}: the answer holds no proxy"
    else:
        assert len(answer) > 0, f"{label}: the answer holds no variant"
        assert len(answer.leads()) > 0, f"{label}: the answer holds no lead"


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    """The LD Store's answer and the sidecar's, field for field, with every code compared."""
    assert type(actual) is type(expected), f"{label}: {type(actual)} vs {type(expected)}"
    if isinstance(actual, tuple) and actual and actual[0] == "refusal":
        assert actual == expected, f"{label}: refusals differ"
        return
    for field in dataclasses.fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=f"{label}: {field.name}")
        elif dataclasses.is_dataclass(a):
            assert_equal(a, b, f"{label}: {field.name}")
        else:
            assert a == b, f"{label}: {field.name} differs"


def compare(
    genotype: opengwasdb.ld.LDStore,
    ld: opengwasdb.ld.LDStore,
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    label: str,
    *,
    meaningful: bool = True,
) -> None:
    theirs = outcome(ld, method, *args, **kwargs)
    ours = outcome(genotype, method, *args, **kwargs)
    if meaningful:
        assert_meaningful(theirs, method, f"{label} (LD Store)")
        assert_meaningful(ours, method, f"{label} (Genotype Store)")
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
        locus = sorted_alid_locus(alid)
        if locus is not None and locus[0] == "22":
            out.append(i)
    return out


def held(
    store: opengwasdb.ld.LDStore, alids: list[str], scores: np.ndarray, keep: list[int]
) -> tuple[list[str], np.ndarray]:
    """The kept ALIDs the store holds, and their scores, in request order."""
    names = [alids[i] for i in keep]
    absent = set(store.matrix(names).absent)
    present = [(a, s) for a, s in zip(names, scores[keep], strict=True) if a not in absent]
    return [a for a, _ in present], np.array([s for _, s in present])


def test_the_chr22_clumps_and_the_target_equal_the_ld_store(
    stores: tuple[opengwasdb.ld.LDStore, opengwasdb.ld.LDStore],
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
    stores: tuple[opengwasdb.ld.LDStore, opengwasdb.ld.LDStore],
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


def test_19_s_fifty_chr22_requests_equal_the_ld_store(
    stores: tuple[opengwasdb.ld.LDStore, opengwasdb.ld.LDStore],
    analyses: list[tuple[str, list[str], np.ndarray]],
    frozen: dict[str, Any],
) -> None:
    """#19's frozen suite, request for request: its matrix, region, proxies, clumps and edges."""
    genotype, ld = stores
    compared = 0
    compare(genotype, ld, "matrix", (frozen["matrix"],), {}, "19: 500-variant matrix")
    compared += 1
    compare(genotype, ld, "region", tuple(frozen["region"]), {}, "19: region")
    compared += 1
    for settings in frozen["proxies"]:
        compare(
            genotype,
            ld,
            "proxies",
            (frozen["proxy_alids"],),
            settings,
            f"19: proxies {settings}",
        )
        compared += 1
    analyses_by_name = {entry["name"]: entry for entry in frozen["analyses"]}
    for name, alids, scores in analyses:
        entry = analyses_by_name[name]
        source = named("OPENGWASDB_LD_403_ANALYSES", "#403's 37 Analyses' hits") / name / "hits.tsv"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["source_sha256"], (
            f"{source} is not the frozen input #19 measured"
        )
        indexes = chr22(alids)
        assert len(indexes) == entry["hits_chr22"], f"{name}: the frozen chr22 hit count differs"
        names, hits = [alids[i] for i in indexes], scores[indexes]
        settings = {"r2": frozen["clump_r2"], "kb": frozen["clump_kb"]}
        if entry["hits_chr22"]:
            compare(genotype, ld, "clump", (names, hits), settings, f"19: clump {name}")
        else:
            # Ten frozen Analyses have no chr22 hit: their request is empty by input, and the
            # assertion that means something is that the fixture is empty and both answers are.
            assert len(ld.clump(names, hits, **settings)) == 0, f"{name}: the fixture is not empty"
            compare(
                genotype,
                ld,
                "clump",
                (names, hits),
                settings,
                f"19: clump {name}",
                meaningful=False,
            )
            assert len(genotype.clump(names, hits, **settings)) == 0
        compared += 1
    assert compared == 37 + 1 + 1 + len(frozen["proxies"])
    first, last = frozen["matrix"][0], frozen["matrix"][-1]
    edges: list[tuple[str, list[str], dict[str, Any]]] = [
        ("absent_duplicate", [first, "22:1:A:C", first, "not-an-alid"], {}),
        ("all_absent", ["22:1:A:C", "not-an-alid"], {}),
        ("empty", [], {}),
        ("explicit_fill", [first, last], {"fill": 0.0}),
    ]
    for label, names, kwargs in edges:
        compare(genotype, ld, "matrix", (names,), kwargs, f"19: {label}", meaningful=False)
        answer = genotype.matrix(names, **kwargs)
        if label == "absent_duplicate":
            assert answer.absent == ("22:1:A:C", "not-an-alid")
            assert len(answer.r) == 2
            assert np.isfinite(answer.r).all()
        elif label == "all_absent":
            assert answer.absent == ("22:1:A:C", "not-an-alid")
            assert answer.r.shape == (0, 0)
        elif label == "empty":
            assert answer.absent == ()
            assert answer.r.shape == (0, 0)
        else:
            assert answer.fill == 0.0
            assert np.isfinite(answer.r).all()
        compared += 1
    refusals: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("proxies", ([first],), {"r2": 0.001, "kb": KB}),
        ("proxies", ([first],), {"r2": 0.01, "kb": 10_000}),
        ("proxies", ([first],), {"r2": 0.8, "kb": 10_001}),
        ("clump", ([first], [1.0]), {"r2": 0.001, "kb": KB}),
        ("region", ("22", 32_000_000, 5_000_001), {}),
    ]
    for method, args, kwargs in refusals:
        theirs = outcome(ld, method, *args, **kwargs)
        assert isinstance(theirs, tuple), f"19: {method} {kwargs} is not the store's refusal"
        assert theirs[0] == "refusal", f"19: {method} {kwargs} is not the store's refusal"
        assert_equal(outcome(genotype, method, *args, **kwargs), theirs, f"19: refusal {method}")
        compared += 1
    assert compared == 50, f"#19's suite is 50 requests; this run made {compared}"


def test_each_chromosome_s_region_and_a_whole_genome_matrix_equal_the_ld_store(
    stores: tuple[opengwasdb.ld.LDStore, opengwasdb.ld.LDStore],
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
