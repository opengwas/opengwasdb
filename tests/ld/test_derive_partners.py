"""Deriving the Partner Rows for an existing store (partner-rows §6.2, §6.3; ogld#69).

Every test works on a copy of the committed Test Store that `store_copies.unpartnered` puts back
to 0.1.0 — `partners/` removed, the four keys that describe it dropped — and holds the derived
component against the one the writer built, byte for byte. Nothing here writes into
`tests/ld/data/`.

The interruption test kills a real derivation with a real `SIGKILL`, after its first frame has
reached the file: the child replaces `FramesGroup._append` with itself plus `os.kill`, so the kill
lands mid-chromosome with no clock to wait on and no injected exception for the code to catch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE, copy_store, unpartnered

import opengwasdb.ld
from opengwasdb.ld import cli
from ogquery.ld.format import InputError
from opengwasdb.ld.store import assembly
from opengwasdb.ld.store.assembly import derive_partners
from opengwasdb.ld.store.validation import (
    PARTNER_RULES,
    ValidationFailed,
    validate_at,
    validate_group,
)

#: The Test Store's chromosomes, and the Partner Threshold its writer used (partner-rows §3).
CHROMOSOMES = ("21", "22")
T_PARTNER = 0.01
#: The two chunk files whose bytes a derivation must reproduce: a chromosome's frames and the
#: offsets that address them (partner-rows §5.1).
FILES = ("frames/c/0", "frames_idx/c/0")


def provenance(store: Path) -> dict[str, Any]:
    """The store's `ld_store` provenance as it is stored."""
    root = zarr.open_group(str(store), mode="r")
    recorded: dict[str, Any] = root.attrs["ld_store"]
    return recorded


def chunk(store: Path, chrom: str, name: str) -> bytes:
    """One of a chromosome's two chunk files, as bytes."""
    return (store / "partners" / chrom / name).read_bytes()


def assert_the_builds_bytes(store: Path) -> None:
    """Every chromosome's two chunk files are the committed Test Store's, byte for byte."""
    for chrom in CHROMOSOMES:
        for name in FILES:
            assert chunk(store, chrom, name) == chunk(TEST_STORE, chrom, name), f"{chrom}/{name}"


def failing(store: Path) -> list[str]:
    """The Partner Rows rules this store breaks: the ten a derivation must pass (§10), read
    against the provenance it records — or the one it would claim, since a 0.1.0 store has none."""
    root = zarr.open_group(str(store), mode="r")
    recorded = dict(root.attrs["ld_store"])
    return [
        rule
        for rule, found in validate_group(root, recorded, PARTNER_RULES, threads=1).items()
        if found
    ]


def test_the_committed_store_holds_the_bytes_a_derivation_must_reproduce() -> None:
    """The fixture is meaningful: the committed store holds both chromosomes' frames, so the
    comparisons the other tests make are of bytes and not of two absences."""
    for chrom in CHROMOSOMES:
        assert (TEST_STORE / "partners" / chrom / "frames" / "c" / "0").stat().st_size > 0


def test_a_derived_store_is_the_builds_byte_for_byte(tmp_path: Path) -> None:
    """One producer: the derivation and the writer produce the same frames (partner-rows §6.1)."""
    store = unpartnered(tmp_path / "store")
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None, "every chromosome is in place, so the provenance claims them"
    assert_the_builds_bytes(store)
    assert made["format_version"] == "0.2.0"
    assert made["t_partner"] == T_PARTNER
    assert made["partner_code_floor"] == 52, "0.01 at × 511 is code 52 (partner-rows §3)"
    assert (
        made["counts"]["partner_entries"] == provenance(TEST_STORE)["counts"]["partner_entries"]
    ), "the component's count is every chromosome's own entries"
    assert made["software"]["zstd"] == provenance(TEST_STORE)["software"]["zstd"]
    assert failing(store) == [], "the derived component passes the ten rules it is judged by"


@pytest.mark.parametrize(("threads", "processes"), [(1, 1), (2, 2)])
def test_the_bytes_do_not_depend_on_the_workers(
    tmp_path: Path, threads: int, processes: int
) -> None:
    """Compressor threads and chromosome processes are bounds on time, never on output
    (partner-rows §6.3, §9.1). Eight workers, whose spawn cost the default tier cannot pay for,
    are held by `tests/ld/realdata/test_derive_partners_workers.py`."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER, threads=threads, processes=processes)
    assert_the_builds_bytes(store)


def test_a_store_that_has_its_partner_rows_is_refused(tmp_path: Path) -> None:
    """Nothing to do is said, not done again: a chromosome's Partner Rows are written once."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    with pytest.raises(InputError, match=r"nothing to derive"):
        derive_partners(store, t_partner=T_PARTNER)


def test_a_t_partner_above_the_stores_t_far_is_refused(tmp_path: Path) -> None:
    """A pair between t_far and t_partner within near_bp would be in neither component
    (partner-rows §3), so the two values are named and nothing is written."""
    store = unpartnered(tmp_path / "store")
    with pytest.raises(InputError, match=r"t_partner = 0\.5.*t_far = 0\.0274"):
        derive_partners(store, t_partner=0.5)
    assert not (store / "partners").exists()


def test_t_partner_is_required(tmp_path: Path) -> None:
    """There is no default: a command without one does not run (partner-rows §3; ADR 0097, D3)."""
    with pytest.raises(SystemExit) as stopped:
        cli.main(["derive-partners", str(unpartnered(tmp_path / "store"))])
    assert stopped.value.code == 2, "argparse refuses the missing option"


def test_a_second_t_partner_is_refused_rather_than_mixed_in(tmp_path: Path) -> None:
    """A store derived at one floor has rows at it; a second would put two floors in one store."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    with pytest.raises(InputError, match=r"records Partner Rows at t_partner = 0\.01"):
        derive_partners(store, t_partner=0.005)


def test_chromosomes_derive_part_of_a_store_without_claiming_it(tmp_path: Path) -> None:
    """A subset is derived and the provenance is not written until every chromosome is in place:
    the store reads as the 0.1.0 one it was (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    before = provenance(store)
    assert derive_partners(store, t_partner=T_PARTNER, chromosomes=["21"]) is None
    assert provenance(store) == before
    assert sorted(entry.name for entry in (store / "partners").iterdir()) == ["21"]
    assert failing(store) == []
    derive_partners(store, t_partner=T_PARTNER)
    assert_the_builds_bytes(store)


def test_drop_index_removes_the_index_of_a_derived_store(tmp_path: Path) -> None:
    """`--drop-index` is a step of its own, taken once the Partner Rows hold every entry the
    Proxy Index had (partner-rows §6.2; ADR 0097, D1)."""
    store = unpartnered(tmp_path / "store")
    made = derive_partners(store, t_partner=T_PARTNER, drop_index=True)
    assert made is not None
    assert not (store / "index").exists(), "the Proxy Index is gone"
    assert "index_entries" not in made["counts"], "and so is its count"


def test_drop_index_removes_nothing_from_a_store_whose_component_fails(tmp_path: Path) -> None:
    """A component a rule fails is one the provenance must not claim, so the index it would have
    replaced is still there, whole (partner-rows §6.2, §10)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    frame = store / "partners" / "22" / "frames" / "c" / "0"
    damaged = bytearray(frame.read_bytes())
    damaged[0] ^= 0xFF
    frame.write_bytes(bytes(damaged))
    with pytest.raises(ValidationFailed) as failed:
        derive_partners(store, t_partner=T_PARTNER, drop_index=True)
    assert "partners.frames" in failed.value.findings
    assert (store / "index").exists(), "nothing is removed from a store whose component fails"
    assert "index_entries" in provenance(store)["counts"]


#: The child that derives and kills itself: a real `SIGKILL` of a real derivation, at the first
#: frame that reaches the file, which is one chromosome's frames part-written and nothing
#: published.
KILLED = """
import os, signal
from opengwasdb.ld.store import frames
appended = frames.FramesGroup._append
def append(self):
    appended(self)
    self._file.flush()
    os.kill(os.getpid(), signal.SIGKILL)
frames.FramesGroup._append = append
from opengwasdb.ld.cli import main
raise SystemExit(main())
"""


def run_killed(script: str, store: Path) -> None:
    """`derive-partners` over `store` in a process that runs `script` first, which kills it."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "derive-partners",
            str(store),
            "--t-partner",
            str(T_PARTNER),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == -9, f"the run was killed, not finished: {done.returncode}"


@pytest.fixture(scope="module")
def killed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """A Test Store copy whose derivation was killed mid-chromosome, and its provenance before."""
    store = unpartnered(tmp_path_factory.mktemp("killed") / "store")
    before = provenance(store)
    run_killed(KILLED, store)
    temporaries = list((store / "partners").glob(".*.tmp-*"))
    assert temporaries, "the killed run left the chromosome's group unpublished, as it must"
    written = list(temporaries[0].glob("frames.bin"))
    assert written, "the frame being written is on disk"
    assert written[0].stat().st_size > 0, "a frame reached the file before the kill"
    return store, before


def test_a_killed_derivation_leaves_the_store_as_it_was(
    killed: tuple[Path, dict[str, Any]],
) -> None:
    """An interrupted run publishes no provenance and no `partners/` node, so the store validates
    as the 0.1.0 one it was (partner-rows §6.2)."""
    store, before = killed
    assert provenance(store) == before, "the provenance is byte-for-byte the one it had"
    assert not (store / "partners" / "zarr.json").exists(), "no node for a reader to trust"
    assert [rule for rule, found in validate_at(store, threads=1).items() if found] == []


def test_a_rerun_after_a_kill_completes(killed: tuple[Path, dict[str, Any]]) -> None:
    """A rerun removes what the kill left, skips what it published, and finishes (§6.2)."""
    store, _ = killed
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None
    assert not list((store / "partners").glob(".*.tmp-*")), "the stale temporary is gone"
    assert_the_builds_bytes(store)
    assert json.loads(json.dumps(made))["counts"]["partner_entries"] > 0


def test_a_stale_temporary_left_by_a_kill_is_removed(tmp_path: Path) -> None:
    """A kill leaves a chromosome's group under a dot-prefixed name; the run that claims the
    component removes it, because `partners.files` refuses a store holding one (§5.1, §6.2)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER, chromosomes=["21"])
    stale = store / "partners" / ".22.tmp-killed"
    stale.mkdir()
    (stale / "zarr.json").write_text("{}")
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None
    assert not stale.exists()
    assert sorted(entry.name for entry in (store / "partners").iterdir()) == [
        "21",
        "22",
        "zarr.json",
    ]
    assert_the_builds_bytes(store)


def test_drop_index_refuses_before_a_derivation_that_would_stay_partial(tmp_path: Path) -> None:
    """`--drop-index` never rides on an incomplete derivation, so a subset of chromosomes is
    refused before anything is derived (partner-rows §6.2): the index is the only component a
    0.1.0 store can clump from, and it must not go first."""
    store = unpartnered(tmp_path / "store")
    with pytest.raises(InputError, match=r"--drop-index removes the Proxy Index only from a store"):
        derive_partners(store, t_partner=T_PARTNER, chromosomes=["21"], drop_index=True)
    assert not (store / "partners").exists(), "nothing was derived"
    assert (store / "index").exists()


# ---------------------------------------------------------------- the pool's contract (ogld#69)


def record_worker(job: assembly._Schedule) -> int:
    """A worker that records the process it ran in, and derives nothing. A module-level function,
    so a spawn worker resolves it by name as the pool pickles it."""
    with (job.store.parent / "workers.txt").open("a") as told:
        told.write(f"{os.getpid()}\n")
    return 0


def refuse_work(job: assembly._Schedule) -> int:
    """A worker that fails, as a chromosome the store cannot read does."""
    raise ValueError(f"chromosome {job.chrom} could not be derived: a worker fault")


def test_the_pool_runs_its_work_in_child_processes(tmp_path: Path) -> None:
    """A process pool is the point of `--processes`, so the test holds the pool, not only the
    bytes: one child process a chromosome, none of them this one (partner-rows §6.3)."""
    store = unpartnered(tmp_path / "store")
    plan = assembly._plan(assembly.open_group(store), store, T_PARTNER, None)
    assert len(plan.missing) == 2, "the fixture must have two chromosomes to fill two workers"
    assembly._derive(plan, threads=1, processes=2, work=record_worker)
    workers = [int(line) for line in (tmp_path / "workers.txt").read_text().split()]
    assert len(workers) == 2, workers
    assert os.getpid() not in workers, "the work ran in this process, not in a pool"


def test_a_workers_failure_stops_the_run_and_claims_nothing(tmp_path: Path) -> None:
    """A worker that raises fails the whole run, and a store whose component is not whole keeps
    the provenance it had: no claim for a component one chromosome short (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    before = provenance(store)
    plan = assembly._plan(assembly.open_group(store), store, T_PARTNER, None)
    with pytest.raises(ValueError, match=r"a worker fault"):
        assembly._derive(plan, threads=1, processes=2, work=refuse_work)
    assert provenance(store) == before, "nothing was claimed"
    assert not (store / "partners" / "zarr.json").exists()


def test_a_worker_that_finds_a_lost_chunk_fails_the_run(tmp_path: Path) -> None:
    """The real worker, in a real pool, on a store whose Far Pairs hold the fill value a lost
    chunk reads as: the failure crosses the process boundary and the run publishes nothing."""
    store = unpartnered(tmp_path / "store")
    before = provenance(store)
    r = zarr.open_group(str(store), mode="r+")["far"]["21"]["r"]
    assert isinstance(r, zarr.Array)
    r[0] = r.dtype.type(-32768)  # the reserved code: a value no store holds there
    with pytest.raises(ValueError, match=r"no LD Store holds there"):
        derive_partners(store, t_partner=T_PARTNER, processes=2)
    assert provenance(store) == before
    assert not (store / "partners" / "zarr.json").exists()


# ------------------------------------------------- the claim is one write, the rest is unclaimed


#: The child that derives, publishes the component, and is killed before the claim: the store
#: left is a 0.1.0 one whose `partners/` no provenance names (ogld#69).
UNCLAIMED = """
import os, signal
from opengwasdb.ld.store import assembly
def record(root, made):
    os.kill(os.getpid(), signal.SIGKILL)
assembly._record = record
from opengwasdb.ld.cli import main
raise SystemExit(main())
"""


@pytest.fixture(scope="module")
def unclaimed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Test Store copy killed between `partners/` becoming a node and the provenance that
    claims it, with the provenance it had before."""
    store = unpartnered(tmp_path_factory.mktemp("unclaimed") / "store")
    run_killed(UNCLAIMED, store)
    assert (store / "partners" / "zarr.json").exists(), "the node was published before the kill"
    assert "t_partner" not in provenance(store), "and the claim was not written"
    return store


def first_alid(store: Path) -> str:
    """One variant ALID of a store, as a query asks for it."""
    alid = zarr.open_group(str(store), mode="r")["variants"]["alid"]
    assert isinstance(alid, zarr.Array)
    return str(alid[0])


def test_a_kill_before_the_claim_leaves_an_unclaimed_component(unclaimed: Path) -> None:
    """The node is there and the provenance says 0.1.0, which is a store that does not have the
    Partner Rows: validation judges nothing, and the store is the one it was (partner-rows §6.2,
    ld-store §12)."""
    assert [rule for rule, found in validate_at(unclaimed, threads=1).items() if found] == []
    with pytest.raises(InputError, match=r"no Partner Rows"):
        opengwasdb.ld.open(unclaimed).clump([first_alid(unclaimed)], [1.0], r2=0.01, kb=1000)


def test_a_rerun_claims_an_unclaimed_component_once_the_rules_accept_it(unclaimed: Path) -> None:
    """A rerun finds a whole component nothing claims, judges it, and writes the claim: it never
    trusts it, and never derives it again for nothing (partner-rows §6.2)."""
    made = derive_partners(unclaimed, t_partner=T_PARTNER)
    assert made is not None
    assert made["format_version"] == "0.2.0"
    assert_the_builds_bytes(unclaimed)


def unclaimed_keeping_keys(to: Path) -> Path:
    """A copy of the committed Test Store that keeps its Partner Rows' keys and frames but records
    format 0.1.0: a component no provenance claims, whatever keys a store carries beside it
    (ld-store §7, §13; partner-rows §6.2).

    It is the residue `_settle` exists for, and the one the final review reproduced: a 0.1.0 store
    whose `partners/` is whole and whose `t_partner` is recorded still has no Partner Rows, so
    `derive-partners` settles it and never answers "nothing to derive" (ogld#64).
    """
    store = copy_store(to)
    recorded = json.loads((store / "zarr.json").read_text())
    recorded["attributes"]["ld_store"]["format_version"] = "0.1.0"
    (store / "zarr.json").write_text(json.dumps(recorded))
    assert "t_partner" in provenance(store), "the fixture keeps the keys, or it proves nothing"
    assert (store / "partners" / "21" / "frames" / "c" / "0").exists(), "and the frames"
    return store


def test_a_0_1_0_store_that_keeps_the_keys_is_settled_not_refused(tmp_path: Path) -> None:
    """The recorded **format version** decides whether a store has Partner Rows, so a 0.1.0 store
    that keeps the keys and the frames is unclaimed: `derive-partners` settles the residue rather
    than claiming there is nothing to derive, and the store is the 0.2.0 one it was, byte for
    byte, clump included (ld-store §7, §13; partner-rows §6.2; ogld#64)."""
    store = unclaimed_keeping_keys(tmp_path / "store")
    alid = first_alid(store)
    with pytest.raises(InputError, match=r"format 0\.1\.0.*no Partner Rows"):
        opengwasdb.ld.open(store).clump([alid], [1.0], r2=T_PARTNER, kb=1000)
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None, "the residue is settled and the provenance claims it"
    assert made["format_version"] == "0.2.0"
    assert made["t_partner"] == T_PARTNER
    assert_the_builds_bytes(store)
    assert failing(store) == [], "the claimed component passes the ten rules it is judged by"
    alids = zarr.open_group(str(store), mode="r")["variants"]["alid"]
    assert isinstance(alids, zarr.Array)
    kept = [str(a) for a in alids[:40]]
    score = np.linspace(1.0, 6.0, len(kept))
    settled = opengwasdb.ld.open(store).clump(kept, score, r2=T_PARTNER, kb=1000)
    built = opengwasdb.ld.open(TEST_STORE).clump(kept, score, r2=T_PARTNER, kb=1000)
    assert len(settled) == len(built), "the same variants are present in both"
    np.testing.assert_array_equal(settled.lead, built.lead)
    np.testing.assert_array_equal(settled.r, built.r)


def drop_claim(store: Path) -> None:
    """Make a store's component unclaimed again: the provenance a derivation had not yet written."""
    root = zarr.open_group(str(store), mode="r+")
    prov = dict(root.attrs["ld_store"])
    prov.pop("t_partner", None)
    prov.pop("partner_code_floor", None)
    prov["format_version"] = "0.1.0"
    prov["counts"] = {k: v for k, v in prov["counts"].items() if k != "partner_entries"}
    prov["software"] = {k: v for k, v in prov["software"].items() if k != "zstd"}
    root.attrs.update({"ld_store": prov})


def test_a_rerun_rebuilds_an_unclaimed_component_the_rules_refuse(tmp_path: Path) -> None:
    """An unclaimed component one rule refuses is removed and derived again: a component nothing
    claims is one nothing may trust, however whole it looks (partner-rows §6.2, §10)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    drop_claim(store)
    frame = store / "partners" / "22" / "frames" / "c" / "0"
    damaged = bytearray(frame.read_bytes())
    damaged[0] ^= 0xFF
    frame.write_bytes(bytes(damaged))
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None, "the rebuilt component is claimed"
    assert_the_builds_bytes(store), "and it is the build's, not the damaged one"


def test_a_partners_directory_that_is_a_link_is_refused(tmp_path: Path) -> None:
    """A derivation writes `partners/` and every group in it, so a link there would put a store's
    component in another directory: it is refused before anything is written (ogld#69)."""
    store = unpartnered(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "partners").symlink_to(outside)
    with pytest.raises(InputError, match=r"partners is a symbolic link"):
        derive_partners(store, t_partner=T_PARTNER)
    assert list(outside.iterdir()) == [], "nothing was written outside the store"


def test_a_chromosome_group_that_is_a_link_is_refused(tmp_path: Path) -> None:
    """The same rule a group deeper: `partners/<chrom>` is a path the derivation writes, and a
    link there would publish a chromosome where the store is not (ogld#69)."""
    store = unpartnered(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "partners").mkdir()
    (store / "partners" / "21").symlink_to(outside)
    assert (store / "partners" / "21").is_dir(), "the link is a directory to a reader"
    with pytest.raises(InputError, match=r"partners/21 is a symbolic link"):
        derive_partners(store, t_partner=T_PARTNER)
    assert list(outside.iterdir()) == [], "nothing was written outside the store"


def test_a_symlinked_index_is_unlinked_and_its_target_kept(tmp_path: Path) -> None:
    """A scratch store's `index/` is a link to a published one: `--drop-index` removes the link
    and never what it names (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    published = tmp_path / "published-index"
    (store / "index").rename(published)
    (store / "index").symlink_to(published)
    made = derive_partners(store, t_partner=T_PARTNER, drop_index=True)
    assert made is not None
    assert not (store / "index").is_symlink(), "the link is gone"
    assert not (store / "index").exists(), "and no index is left in the store"
    assert any(published.iterdir()), "the directory it named is untouched"
    assert "index_entries" not in made["counts"]


def test_a_failed_index_removal_keeps_its_count(tmp_path: Path) -> None:
    """A removal that fails is named, and the count that describes the index is kept: the
    provenance never claims a removal that did not happen (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    before = provenance(store)
    index = store / "index"
    index.chmod(0o500)  # its entries cannot be unlinked from it
    try:
        with pytest.raises(InputError, match=r"could not be removed .*: its count is kept"):
            derive_partners(store, t_partner=T_PARTNER, drop_index=True)
        assert provenance(store) == before, "the provenance is untouched"
        assert "index_entries" in provenance(store)["counts"]
        assert index.is_dir(), "the index is still the store's"
    finally:
        index.chmod(0o700)


def test_a_0_1_0_store_that_keeps_the_keys_answers_no_partner_read(tmp_path: Path) -> None:
    """A store's provenance may hold extra keys (ld-store §7), and one of them does not make it a
    store with Partner Rows: the recorded **format version** does. A 0.1.0 store carrying a
    whole component nothing claims is refused by both reads that would use it, rather than
    answered from frames no rule has judged (partner-rows §6.2; ld-store §13; ogld#69)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    recorded = json.loads((store / "zarr.json").read_text())
    recorded["attributes"]["ld_store"]["format_version"] = "0.1.0"
    (store / "zarr.json").write_text(json.dumps(recorded))
    assert "t_partner" in provenance(store), "the fixture keeps the keys, or it proves nothing"
    assert [rule for rule, found in validate_at(store, threads=1).items() if found] == []
    alid = first_alid(store)
    for refused in (
        lambda: opengwasdb.ld.open(store).clump([alid], [1.0], r2=T_PARTNER, kb=1000),
        lambda: opengwasdb.ld.open(store).proxies([alid], r2=T_PARTNER, kb=1000),
    ):
        with pytest.raises(InputError, match=r"format 0\.1\.0.*no Partner Rows") as caught:
            refused()
        assert str(store) in str(caught.value), "the refusal names the store it is about"
        message = str(caught.value)
        assert "opengwasdb ld derive-partners" in message, "and the command that settles it"


# ---- item 3: no link anywhere the derivation writes, and no planted metadata temporary --------


def test_a_partners_link_inside_the_store_is_refused(tmp_path: Path) -> None:
    """`partners/` is the derivation's own directory, so a link at it is refused even when it
    stays inside the store: a cleanup or a claim would then write, and delete, in the directory
    the link names (partner-rows §6.2; ogld#69)."""
    store = unpartnered(tmp_path / "store")
    stash = store / "stash"
    stash.mkdir()
    (stash / ".keep").write_text("not the component's")
    (store / "partners").symlink_to(stash)
    with pytest.raises(InputError, match=r"partners is a symbolic link"):
        derive_partners(store, t_partner=T_PARTNER)
    assert (stash / ".keep").exists(), "nothing the link names was touched"


def test_a_planted_metadata_temporary_is_refused(tmp_path: Path) -> None:
    """Zarr writes a metadata document through `zarr.<uuid>.partial` beside it, and follows a link
    there: a store holding one before a derivation is refused, so a planted or stale temporary is
    never written through (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (store / "zarr.0123456789abcdef0123456789abcdef.partial").symlink_to(outside)
    with pytest.raises(InputError, match=r"zarr\.[0-9a-f]+\.partial is a temporary"):
        derive_partners(store, t_partner=T_PARTNER)
    assert outside.read_text() == "{}", "the file the temporary names is untouched"
    assert not (store / "partners").exists(), "nothing was derived"


def test_a_root_metadata_link_is_refused(tmp_path: Path) -> None:
    """The claim replaces the root metadata, so a link at it would put the store's provenance in
    another file (partner-rows §6.2)."""
    store = unpartnered(tmp_path / "store")
    outside = tmp_path / "zarr.json"
    outside.write_bytes((store / "zarr.json").read_bytes())
    (store / "zarr.json").unlink()
    (store / "zarr.json").symlink_to(outside)
    with pytest.raises(InputError, match=r"zarr.json is a symbolic link"):
        derive_partners(store, t_partner=T_PARTNER)
    assert json.loads(outside.read_text()) == json.loads((store / "zarr.json").read_text())


# ---- item 4: a residue whose own metadata is unreadable is a rejection, not a crash -----------


def test_a_rerun_rebuilds_an_unclaimed_component_that_has_no_count(tmp_path: Path) -> None:
    """A chromosome group with no `entries` cannot be counted, so the residue cannot be claimed:
    the rules reject it and `_settle` derives it again, rather than the run crashing before the
    rules are ever asked (partner-rows §6.2, §10)."""
    store = unpartnered(tmp_path / "store")
    derive_partners(store, t_partner=T_PARTNER)
    drop_claim(store)
    group = zarr.open_group(str(store / "partners" / "21"), mode="r+")
    del group.attrs["entries"]
    made = derive_partners(store, t_partner=T_PARTNER)
    assert made is not None
    assert_the_builds_bytes(store)
