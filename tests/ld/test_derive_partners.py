"""Deriving the Partner Rows for an existing store (partner-rows §6.2, §6.3; ogld#69).

Every test works on a copy of the committed Test Store that `store_copies.unpartnered` puts back
to 0.1.0 — `partners/` removed, the four keys that describe it dropped — and holds the derived
component against the one the writer built, byte for byte. Nothing here writes into
`tests/data/`.

The interruption test kills a real derivation with a real `SIGKILL`, after its first frame has
reached the file: the child replaces `FramesGroup._append` with itself plus `os.kill`, so the kill
lands mid-chromosome with no clock to wait on and no injected exception for the code to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import zarr
from store_copies import TEST_STORE, unpartnered

from ogld import cli
from ogld.format import InputError
from ogld.store.assembly import derive_partners
from ogld.store.validation import PARTNER_RULES, ValidationFailed, validate_at, validate_group

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


@pytest.mark.parametrize(("threads", "processes"), [(1, 1), (2, 2), (8, 8)])
def test_the_bytes_do_not_depend_on_the_workers(
    tmp_path: Path, threads: int, processes: int
) -> None:
    """Compressor threads and chromosome processes are bounds on time, never on output
    (partner-rows §6.3, §9.1). The pool takes one worker a chromosome at most, so eight
    processes on a two-chromosome store derive two at once."""
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
    """There is no default: a command without one does not run (partner-rows §3; ADR 0011, D3)."""
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
    Proxy Index had (partner-rows §6.2; ADR 0011, D1)."""
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
from ogld.store import frames
appended = frames.FramesGroup._append
def append(self):
    appended(self)
    self._file.flush()
    os.kill(os.getpid(), signal.SIGKILL)
frames.FramesGroup._append = append
from ogld.cli import main
raise SystemExit(main())
"""


@pytest.fixture(scope="module")
def killed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """A Test Store copy whose derivation was killed mid-chromosome, and its provenance before."""
    store = unpartnered(tmp_path_factory.mktemp("killed") / "store")
    before = provenance(store)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            KILLED,
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
