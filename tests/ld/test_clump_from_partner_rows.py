"""Clump and proxies from the Partner Rows (partner-rows §8, §9.2; ogld#71).

`FROZEN` holds the previous reader's answers — the integration tip's code at `1cf9ef2`, which read
the Near Band and the Proxy Index on this same Test Store — for every case in `CASES`, as a digest
of every column each result carries. They were recorded by running `fingerprint` against that
checkout, never against this one, so a difference is the new reader's.

The rest hold the seams this ticket agreed: the refusal table's new row, decided before any read;
a store with no Partner Rows refused rather than answered from the Near Band; the component cut at
exactly W against W − 1; and the same answer at one, two and eight requested workers, a saturated
pool that still progresses, and a damaged frame that fails the whole request with no partial result.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from store_copies import PARTNER_FLOOR, R_SCALE, T_PARTNER, TEST_STORE, copy_store, store_reads

import opengwasdb.ld
from opengwasdb.ld.format import InputError, Refusal, RefusalReason, t_far
from opengwasdb.ld.identity import alid_locus
from opengwasdb.ld.query import clumping
from opengwasdb.ld.store import StoreReader
from opengwasdb.ld.store.workers import COMPONENT_WORKERS_MAX

#: The Test Store's rows, all of them: a clump of every variant, and a proxies query on each.
ALL = np.arange(1056, dtype=np.int64)
#: The Test Store's Far Threshold, which its own record holds (n = 771 at k = 10; ld-store §3, §7).
T_FAR = t_far(771, 10)
#: Every seventh row, and the first forty: the smaller requests the frozen answers also cover.
SAMPLE = ALL[::7]
SMALL = ALL[:40]

#: |z|-like scores, fixed so the frozen answers mean the same request every run.
SCORE = np.abs(np.random.default_rng(71).normal(4.0, 1.5, len(ALL)))
SMALL_SCORE = np.abs(np.linspace(1.0, 6.0, len(SMALL)))

#: A list that names one row twice with the same score, another twice with different ones, and a
#: tie across two rows: the scatter rule and the strongest-first tie rule (ld-store §9).
DUPLICATES = np.array([SMALL[0], SMALL[5], SMALL[5], SMALL[17], SMALL[0], SMALL[30], SMALL[30]])
DUPLICATE_SCORE = np.array([3.0, 5.0, 5.0, 2.0, 3.0, 9.0, 9.0])

#: Two absent ALIDs — one malformed, one a locus no variant sits on — and two present ones.
ABSENT_SCORE = np.array([4.0, 3.0, 2.0, 1.0])

#: `(name, kind, subset, r2, kb)`, and for proxies `(name, kind, subset, r2, kb)` alike.
CASES: list[tuple[str, str, str, float, float]] = [
    (f"{kind}-{subset}-r2-{r2}-kb-{kb}", kind, subset, r2, kb)
    for kind in ("clump", "proxies")
    for subset in ("all", "sample")
    for r2 in (0.01, 0.1, 0.5)
    for kb in (100, 3000)
] + [
    ("clump-duplicates", "clump", "duplicates", 0.01, 3000),
    ("clump-duplicates-r2-1", "clump", "duplicates", 1.0, 3000),
    ("clump-absent", "clump", "absent", 0.01, 3000),
    ("proxies-absent", "proxies", "absent", 0.5, 3000),
    ("proxies-duplicates", "proxies", "duplicates", 0.5, 3000),
    ("proxies-default", "proxies", "small-default", 0.8, 10_000),
    ("clump-t_far-edge", "clump", "all", T_FAR, 3000),
    ("proxies-t_far-edge", "proxies", "sample", T_FAR, 3000),
    ("clump-kb-0", "clump", "small", 0.01, 0),
    ("clump-one-chromosome", "clump", "chromosome-22", 0.01, 3000),
]


def _subset(name: str, store: opengwasdb.ld.LDStore) -> tuple[list[str], npt.NDArray[np.float64]]:
    """One case's ALIDs and scores, from the Test Store's own rows."""
    reader = store._reader
    if name == "all":
        return reader.alids_of(ALL).tolist(), SCORE
    if name == "sample":
        return reader.alids_of(SAMPLE).tolist(), SCORE[::7]
    if name == "small":
        return reader.alids_of(SMALL).tolist(), SMALL_SCORE
    if name == "small-default":
        return reader.alids_of(SMALL).tolist(), SMALL_SCORE
    if name == "duplicates":
        return reader.alids_of(DUPLICATES).tolist(), DUPLICATE_SCORE
    if name == "absent":
        present = reader.alids_of(np.array([SMALL[0], SMALL[1]])).tolist()
        return [present[0], "not-an-alid", present[1], "22:1:A:C"], ABSENT_SCORE
    if name == "chromosome-22":
        names = reader.alids_of(ALL).tolist()
        keep = np.array([alid_locus(alid)[0] == "22" for alid in names])
        return [name for name, ok in zip(names, keep, strict=True) if ok], SCORE[keep]
    raise AssertionError(f"no subset {name}")


def _digest(columns: dict[str, Any]) -> str:
    """A result's every column, read as bytes: an answer's fingerprint, never a subset of it."""
    hasher = hashlib.sha256()
    for name, value in columns.items():
        if isinstance(value, tuple):
            hasher.update(name.encode())
            for item in value:
                hasher.update(str(item).encode())
                hasher.update(b"\0")
            continue
        array = np.ascontiguousarray(value)
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def fingerprint(store: opengwasdb.ld.LDStore, name: str) -> str:
    """One case's answer on `store`, digested: the frozen answers were recorded with this."""
    case = {case[0]: case for case in CASES}[name]
    _, kind, subset, r2, kb = case
    names, score = _subset(subset, store)
    if kind == "clump":
        got = store.clump(names, score, r2=r2, kb=kb)
        return _digest(
            {
                "alid": got.alid,
                "chrom": got.chrom,
                "position": got.position,
                "score": got.score,
                "lead": got.lead,
                "r": got.r,
                "absent": got.absent,
                "r2": got.r2,
                "window_bp": got.window_bp,
            }
        )
    found = store.proxies(names, r2=r2, kb=kb)
    return _digest(
        {
            "query": found.query,
            "proxy": found.proxy,
            "r": found.r,
            "r2": found.r2,
            "distance": found.distance,
            "absent": found.absent,
            "r2_min": found.r2_min,
            "window_bp": found.window_bp,
            "t_far": found.t_far,
        }
    )


# fmt: off
FROZEN: dict[str, str] = {
    "clump-absent":
        "32989243aca572eee6e904a3be85c710b472a0bf94be1b65a9733eff37b94b32",
    "clump-all-r2-0.01-kb-100":
        "7eb8cf4bb367e2e90f93bbbe70e60d8709b2721616588256c36b8c858a61864d",
    "clump-all-r2-0.01-kb-3000":
        "4249e996f5d031137601a88dea196ff3258a727e07bea3df54d8da5d3931f400",
    "clump-all-r2-0.1-kb-100":
        "1bf2357f964aa286b7fef44387f54de9f3f8b88b163eed9a555d91d72f861f8e",
    "clump-all-r2-0.1-kb-3000":
        "a932b1b91df313dabf58e8c643987d6cd8c17246a860c1f11672c1aae90a9589",
    "clump-all-r2-0.5-kb-100":
        "c1bc98616b42d9e9de7e4b6671eaa53913c3cd3a4c91867bbca3934a8f901989",
    "clump-all-r2-0.5-kb-3000":
        "a84a04b2e7d226e0de05eb4ded6f5d5fc8def03a866205b283033bfa72b6e321",
    "clump-duplicates":
        "d976b1b764957e7964d5c40835142e4aded2c2d9b14a0dc62a1988eb10b3c47b",
    "clump-duplicates-r2-1":
        "f32d343daa5481215535f688117abcade672ed64da5fe89a241ee6aab66893d8",
    "clump-kb-0":
        "2f3788ad04a9524800e2a90d09236cfb31fa617459d12a9457d089d5113529b9",
    "clump-one-chromosome":
        "8e13c0f155c5e226183df7eadc502cda6f005c9da5ed70e44a93e4c8e9d9c353",
    "clump-sample-r2-0.01-kb-100":
        "7abaf3c93984cbd3f6795a0102caf4e8db4cbe0211e29cc9ae7dc545f73da8f0",
    "clump-sample-r2-0.01-kb-3000":
        "53dedcf0ff3fde1490fa0dd867cde6d69833ba501b1c2fda02d780d0c4d62308",
    "clump-sample-r2-0.1-kb-100":
        "27cfe80532ba5cb841958f05f58360b837fab488c3c81cf5c1337620ba6d16af",
    "clump-sample-r2-0.1-kb-3000":
        "c89fb21076df92e0af3cf479a4a2425344d638e058cf71d11abc639c18e404a5",
    "clump-sample-r2-0.5-kb-100":
        "026f03bdbb60045cb4ca2b1ac7be012aec4ef54179ef209363819fd547ea50c2",
    "clump-sample-r2-0.5-kb-3000":
        "2b26885cda48c34181435c140636eee7a5d6177229469a1981ef98a30366ffd4",
    "clump-t_far-edge":
        "3ec545a7aa8ea71dd5e542e4466cb11d12b2b5d5bfbef8e03f7280c726e91a36",
    "proxies-absent":
        "5ec6a6a9124b2fba094331a1146762458397e34050caafbfa1c589e1e7b2c8e5",
    "proxies-all-r2-0.01-kb-100":
        "adb6ddf0c318558ab50b0c7d733e6afd8afc0d3984bd34eae44e596493a39d7c",
    "proxies-all-r2-0.01-kb-3000":
        "de7931919ac18fce5d9994c6b2b295b34d8e9115b7d644e74057030af958eecd",
    "proxies-all-r2-0.1-kb-100":
        "aca34552959e1a54c2ffd3dfc953a49804276622381e40ed51816743c9509bca",
    "proxies-all-r2-0.1-kb-3000":
        "3ab9bb72c171eeafddda5bd00a80e1fa25e8efb52e75a5347e0b37ce0b7de159",
    "proxies-all-r2-0.5-kb-100":
        "24910ab4709cd5e2ecb0d5ab28f27192b58ebebfb460ffd4e7085779fd08206e",
    "proxies-all-r2-0.5-kb-3000":
        "b959681387aa7349d0fc29a04ba56bd9e91640c0d1c2489e8b70f82221ef350c",
    "proxies-default":
        "2d7d8523f3af7bd60ea869ecc9922fb8026241462ad3cbc43efc6dec78e39c1f",
    "proxies-duplicates":
        "527748d8a5ef8a0b7767a865a2c911970e5f2e88c55c8c8573d9605b6d1a3b96",
    "proxies-sample-r2-0.01-kb-100":
        "0327b0b0660d645bee4e9a0a02753dce46c7fb5946367446340720a33b031506",
    "proxies-sample-r2-0.01-kb-3000":
        "e054ee1bbd37e391ff9f58bb40b2c1569cd49b7fadd3190b7ebf02631564a1c2",
    "proxies-sample-r2-0.1-kb-100":
        "08a9bd939b5eae28e137d633b01c312b871f119d7962fb296213f6f2f3847b19",
    "proxies-sample-r2-0.1-kb-3000":
        "f9efe01747ae3d5609b97165b630853b9967dcad086fcc856bd2cb2a40457eb7",
    "proxies-sample-r2-0.5-kb-100":
        "60e7fe7ab8eb45941236f7200e1f0a5ae4bdd4591aa93e0be7476c7cded0a574",
    "proxies-sample-r2-0.5-kb-3000":
        "6783d76c77207bebd769a618755e320a6f1946e3e67da5fe5b00f647e792b3c1",
    "proxies-t_far-edge":
        "975317dc1df0ba779e42549ebd5814bc11906c340a25f49a30c4d25681498745",
}
# fmt: on


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_every_answer_is_the_previous_reader_s(name: str) -> None:
    assert name in FROZEN, f"{name} was never frozen from the previous reader"
    assert fingerprint(opengwasdb.ld.open(TEST_STORE), name) == FROZEN[name]


def test_both_queries_read_nothing_of_the_near_band_or_the_proxy_index() -> None:
    """Their one LD read is the Partner Rows: the ALID index, the positions and the frames, but no
    `near/<c>/indptr`, no Near Band code, no Proxy Index array and none of their metadata
    (partner-rows §8.1, §8.4; ld-store §10)."""
    names, score = _subset("small", opengwasdb.ld.open(TEST_STORE))
    for query in ("clump", "proxies"):
        store = opengwasdb.ld.open(TEST_STORE)
        with store_reads() as keys:
            _ask(store, query, names, score, r2=0.01, kb=3000)
        offending = [key for key in keys if key.startswith(("near/", "index/"))]
        assert offending == [], f"{query} read {offending[:5]}"
        assert store._reader._frame_cache.held > 0, f"{query} read no frame"


# --------------------------------------------------------------- the refusal table


def _ask(
    store: opengwasdb.ld.LDStore,
    query: str,
    names: list[str],
    score: npt.NDArray[np.float64],
    *,
    r2: float,
    kb: float,
) -> Any:
    """The query named, so a test asks both with one statement."""
    if query == "clump":
        return store.clump(names, score, r2=r2, kb=kb)
    return store.proxies(names, r2=r2, kb=kb)


def test_a_threshold_one_code_step_below_the_partner_floor_is_refused() -> None:
    """At the Partner Floor's own code the request is answered; one code step below it is
    refused, W ≤ near_bp, for both queries (ld-store §9; partner-rows §8.1)."""
    store = opengwasdb.ld.open(TEST_STORE)
    floor = PARTNER_FLOOR
    assert store.provenance["partner_code_floor"] == floor
    assert (floor / R_SCALE) ** 2 >= T_PARTNER, "the floor's code reaches t_partner"
    below = ((floor - 1) / R_SCALE) ** 2
    assert below < T_PARTNER, "one code step below the floor does not"
    names, score = _subset("small", store)
    answered = store.clump(names, score, r2=T_PARTNER, kb=3000)
    assert len(answered.leads()) > 1, "t_partner itself is answered"
    for refused in (below, T_PARTNER - 1e-12, 0.001):
        for query in ("clump", "proxies"):
            with pytest.raises(Refusal) as caught:
                _ask(store, query, names, score, r2=refused, kb=3000)
            assert caught.value.reason is RefusalReason.THRESHOLD_BELOW_T_PARTNER
            assert "t_partner" in str(caught.value)


def test_the_t_partner_refusal_names_the_store_both_thresholds_and_the_way_out() -> None:
    """A refusal is actionable: it names the request's r2, the t_partner the store records and
    the store it is, and what to do — ask at or above the store's t_partner, or read a store
    derived at a lower one (ld-store §9; partner-rows §8.1; ogld#64)."""
    store = opengwasdb.ld.open(TEST_STORE)
    names, score = _subset("small", store)
    asked = 0.001
    with pytest.raises(Refusal) as caught:
        store.clump(names, score, r2=asked, kb=3000)
    said = str(caught.value)
    assert caught.value.reason is RefusalReason.THRESHOLD_BELOW_T_PARTNER, "the reason is kept"
    assert f"r2={asked}" in said, f"the requested threshold is not named: {said}"
    assert f"t_partner={T_PARTNER}" in said, f"the store's t_partner is not named: {said}"
    assert str(TEST_STORE) in said, f"the store is not named: {said}"
    assert f"r2 ≥ {T_PARTNER}" in said, f"the supported threshold is not offered: {said}"
    assert "Lower" in said or "lower" in said, f"no lower-threshold store is offered: {said}"


def _nothing_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every read a store or a frame makes, refused or not, as it is made."""
    seen: list[str] = []
    read = StoreReader._read
    pread = StoreReader.partner_row

    def recording(array: Any, coords: Any) -> Any:
        seen.append(str(array.path))
        return read(StoreReader, array, coords)

    def recorded(chrom: str, local_row: int) -> Any:
        seen.append(f"partners/{chrom}/{local_row}")
        return pread(chrom, local_row)

    monkeypatch.setattr(StoreReader, "_read", recording)
    monkeypatch.setattr(StoreReader, "partner_row", recorded)
    return seen


def test_the_refusal_is_decided_before_any_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ALID is resolved and no frame is fetched for a refused request: the judge runs first
    (partner-rows §8.2, step 2)."""
    store = opengwasdb.ld.open(TEST_STORE)
    names, score = _subset("small", store)
    below = ((PARTNER_FLOOR - 1) / R_SCALE) ** 2
    seen = _nothing_read(monkeypatch)
    for query in ("clump", "proxies"):
        with pytest.raises(Refusal):
            _ask(store, query, names, score, r2=below, kb=3000)
    assert seen == [], f"a refused request read {seen}"


def _without_partner_rows(tmp_path: Path) -> Path:
    """A copy of the Test Store made into the 0.1.0 store it is read as: no Partner Rows, and no
    provenance for them, which is what a store built before 0.2.0 holds (ld-store §13)."""
    copy = copy_store(tmp_path / "v010")
    shutil.rmtree(copy / "partners")
    root = copy / "zarr.json"
    document = __import__("json").loads(root.read_text())
    provenance = document["attributes"]["ld_store"]
    provenance["format_version"] = "0.1.0"
    for key in ("t_partner", "partner_code_floor"):
        provenance.pop(key)
    provenance["counts"].pop("partner_entries", None)
    provenance["software"].pop("zstd", None)
    root.write_text(__import__("json").dumps(document))
    return copy


def test_a_store_without_partner_rows_is_refused_naming_the_component_and_the_way_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    where = _without_partner_rows(tmp_path)
    old = opengwasdb.ld.open(where)
    names, score = _subset("small", old)
    seen = _nothing_read(monkeypatch)
    for query in ("clump", "proxies"):
        with pytest.raises(InputError) as refused:
            _ask(old, query, names, score, r2=0.01, kb=3000)
        assert "Partner Rows" in str(refused.value)
        assert "ogld derive-partners" in str(refused.value)
        assert str(where) in str(refused.value), "the store it refused is named"
    assert seen == [], f"a store without Partner Rows fell back to {seen}"


# --------------------------------------------------------------- components


def _gap_of_w(reader: StoreReader) -> tuple[list[str], npt.NDArray[np.float64], int]:
    """Three chr22 hits, gaps (g, smaller), so that W = g cuts between the first two and W = g + 1
    does not: the second hit is the first's nearest stored partner, at d = g (partner-rows §8.2)."""
    rows = reader.rows_of(reader.alids_of(ALL).tolist())
    chrom, position = reader.loci_of(rows)
    on22 = np.flatnonzero(chrom == "22")
    start = reader.chromosome_range("22")[0]
    for first in on22:
        partners, _ = reader.partner_row("22", int(rows[first]) - start)
        above = partners.astype(np.int64) + start
        near = above[(above > rows[first]) & (above - rows[first] <= 3_000_000)]
        if not len(near):
            continue
        later = int(np.searchsorted(rows, near[0]))
        gap = int(position[later] - position[first])
        if gap / 1000 * 1000 != gap:
            continue  # the window must be exactly the gap for the W edge to be at W
        closer = [k for k in on22[on22 > later] if 0 < position[k] - position[later] < gap]
        if not closer:
            continue
        picked = np.array([first, later, closer[0]])
        names = reader.alids_of(rows[picked]).tolist()
        return names, np.array([9.0, 1.0, 0.5]), gap
    raise AssertionError("the Test Store must hold a stored pair with a nearer third hit")


def _recorder(real: Any, tasks: list[Any]) -> Any:
    """`_component`, keeping each component the clump hands over."""

    def recorded(reader: StoreReader, task: Any, **kwargs: Any) -> Any:
        tasks.append(task)
        return real(reader, task, **kwargs)

    return recorded


def test_a_gap_of_exactly_w_cuts_and_one_of_w_minus_one_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive hits W apart are separate components, and W − 1 apart are one: the cut is
    at W, and a pair at d = W − 1 can join where one at d = W cannot (partner-rows §8.2)."""
    reader = StoreReader.open(TEST_STORE)
    reader._set_query_workers(1)
    names, score, gap = _gap_of_w(reader)
    real = clumping._component
    for window, want in (
        (gap, [[0], [1, 2]]),
        (gap + 1, [[0, 1, 2]]),
    ):
        tasks: list[Any] = []
        monkeypatch.setattr(clumping, "_component", _recorder(real, tasks))
        got = clumping.clump(reader, names, score, r2=T_PARTNER, kb=window / 1000)
        monkeypatch.setattr(clumping, "_component", real)
        assert [[int(row) for row in task.rows] for task in tasks] == [
            [int(reader.rows_of([names[i]])[0]) for i in part] for part in want
        ], f"a window of {window} bp splits the hits wrongly"
        assert got.lead[1] == (got.alid[1] if window == gap else got.alid[0]), (
            f"a pair {gap} bp apart must {'not ' if window == gap else ''}join at W = {window}"
        )


# --------------------------------------------------------------- concurrency


def test_one_two_and_eight_requested_workers_give_one_answer() -> None:
    """The answer does not depend on the pool, and eight requested workers are four
    (partner-rows §9.2; #65's measurement)."""
    store = opengwasdb.ld.open(TEST_STORE)
    names, score = _subset("all", store)
    answers = {}
    for requested in (1, 2, 8):
        reader = StoreReader.open(TEST_STORE)
        reader._set_query_workers(requested)
        assert reader._query_workers == min(requested, COMPONENT_WORKERS_MAX), (
            "the reader's budget caps the pool"
        )
        answers[requested] = clumping.clump(reader, names, score, r2=0.01, kb=3000)
    for requested in (2, 8):
        for column in answers[1].columns():
            np.testing.assert_array_equal(
                answers[requested].columns()[column], answers[1].columns()[column]
            )


@pytest.mark.parametrize("requested", [0, -1, 1.5, True, "4"])
def test_the_pool_size_is_a_count_of_threads(requested: Any) -> None:
    """The seam is private (#71 keeps the pool's size internal), and a count below 1 is refused."""
    reader = StoreReader.open(TEST_STORE)
    with pytest.raises(InputError, match="component thread count"):
        reader._set_query_workers(requested)


@pytest.mark.parametrize("requested", [2, 8])
def test_a_saturated_pool_still_progresses_and_never_overruns_its_workers(
    requested: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More components than threads: every task runs, exactly `workers` at once and never more —
    eight requested is four (partner-rows §9.2) — and the answer is the serial one."""
    assert COMPONENT_WORKERS_MAX == 4, "the cap #65's measurement fixed"
    names, score = _subset("all", opengwasdb.ld.open(TEST_STORE))
    once = StoreReader.open(TEST_STORE)
    once._set_query_workers(1)
    serial = clumping.clump(once, names, score, r2=0.01, kb=100)
    never = StoreReader.open(TEST_STORE)
    never._set_query_workers(requested)
    workers = never._query_workers
    assert workers == min(requested, COMPONENT_WORKERS_MAX)
    real = clumping._component
    state = threading.Condition()
    running, peak, done = 0, 0, 0

    def saturated(reader: StoreReader, task: Any, **kwargs: Any) -> Any:
        """Hold each task until the pool's threads are all inside one, so the pool is saturated by
        construction, then let it go."""
        nonlocal running, peak, done
        with state:
            running += 1
            peak = max(peak, running)
            state.notify_all()
            state.wait_for(lambda: running >= workers, timeout=1.0)
        try:
            return real(reader, task, **kwargs)
        finally:
            with state:
                running -= 1
                done += 1
                state.notify_all()

    monkeypatch.setattr(clumping, "_component", saturated)
    got = clumping.clump(never, names, score, r2=0.01, kb=100)
    assert peak == workers, f"the pool ran {peak} tasks at once, not its {workers}"
    assert done > 2 * peak, "the fixture must hold many more components than the pool"
    for column in got.columns():
        np.testing.assert_array_equal(got.columns()[column], serial.columns()[column])


def test_a_damaged_frame_fails_the_whole_request_with_no_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One frame's checksum fails: the request is refused naming it, and no clumps are returned
    from the tasks that finished (partner-rows §9.2)."""
    store = opengwasdb.ld.open(TEST_STORE)
    reader = store._reader
    chr22 = reader.chromosome_range("22")[0]
    # The first hit is chromosome-local row 80, which is frame 5's first row: as the strongest hit
    # of its chromosome it is a lead, so its own row is read, and the rest are chr21 components.
    damaged = 5 * 16
    names = reader.alids_of(np.array([chr22 + damaged, *range(20)])).tolist()
    score = np.array([9.0, *np.linspace(1.0, 2.0, 20)])
    copy = copy_store(tmp_path / "damaged")
    index = np.fromfile(copy / "partners" / "22" / "frames_idx" / "c" / "0", dtype="<u8")
    start, stop = int(index[damaged // 16]), int(index[damaged // 16 + 1])
    frame = bytearray((copy / "partners" / "22" / "frames" / "c" / "0").read_bytes())
    frame[(start + stop) // 2] ^= 0xFF
    (copy / "partners" / "22" / "frames" / "c" / "0").write_bytes(bytes(frame))

    healthy = clumping.clump(StoreReader.open(TEST_STORE), names, score, r2=0.01, kb=3000)
    assert healthy.lead[0] == healthy.alid[0], "the fixture's first hit must be a lead"
    damaged_reader = StoreReader.open(copy)
    damaged_reader._set_query_workers(4)
    with pytest.raises(ValueError, match=rf"frame {damaged // 16} of chromosome 22"):
        clumping.clump(damaged_reader, names, score, r2=0.01, kb=3000)
