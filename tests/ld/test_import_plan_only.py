"""`opengwasdb ld import --plan-only`: the plan and lifted tables, with no network.

opengwas/opengwasdb-stores#542.

A plan-only run writes `WORK/plan.json` and each `<chr>.lifted.npz`, then exits. It lists no
bucket and downloads nothing, and it never creates OUT. Its plan is the plan a full import
makes: a full import of the same command resumes from it.
"""

import json
import socket
from pathlib import Path

import pytest
from import_cli_args import FIXTURE, required_import_args

import opengwasdb.ld
from opengwasdb.ld.build.importing import run
from opengwasdb.ld.cli import main


def import_args(out: Path, work: Path, *extra: str) -> list[str]:
    """The small import, with no ledger and no supplied parts: a full run lists the bucket."""
    return [
        *required_import_args(out, work),
        *extra,
    ]


class NetworkTouched(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Every way the import reaches the bucket raises, and each attempt is counted.

    Making the fetcher touches no network, so `PinnedHttps()` returns a fetch that raises
    when called: a full import whose ledger and parts are all supplied never calls it.
    """
    touched: list[str] = []

    def refuse(what):
        def call(*args, **kwargs):
            touched.append(what)
            raise NetworkTouched(what)

        return call

    monkeypatch.setattr(run, "PinnedHttps", lambda: refuse("fetch"))
    monkeypatch.setattr(run, "inventory", refuse("inventory"))
    monkeypatch.setattr(socket, "create_connection", refuse("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("socket.getaddrinfo"))
    return touched


def test_plan_only_writes_the_plan_and_lifted_tables_and_nothing_else(
    tmp_path, capsys, no_network
) -> None:
    out, work = tmp_path / "import.store", tmp_path / "work"
    assert main(import_args(out, work, "--plan-only")) == 0
    assert no_network == []
    assert not out.exists()
    assert sorted(path.name for path in work.iterdir()) == [
        "1.lifted.npz",
        "2.lifted.npz",
        "lock",
        "plan.json",
    ]
    plan = json.loads((work / "plan.json").read_text())
    assert sorted(plan["chromosomes"]) == ["1", "2"]
    assert plan["far_bp"] == 10_000_000
    assert plan["identity"]["out"] == str(out.resolve())
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "plan": str(work / "plan.json"),
        "far_bp": 10_000_000,
        "chromosomes": ["1", "2"],
    }


def test_plan_only_plan_is_the_plan_a_full_import_makes(tmp_path, no_network) -> None:
    planned = tmp_path / "planned"
    full = tmp_path / "full"
    assert main(import_args(planned / "import.store", planned / "work", "--plan-only")) == 0
    ledger = ["--ledger", str(FIXTURE / "ledger.json"), "--parts", str(FIXTURE)]
    assert main(import_args(full / "import.store", full / "work", *ledger)) == 0
    only = json.loads((planned / "work" / "plan.json").read_text())
    made = json.loads((full / "work" / "plan.json").read_text())
    assert only["chromosomes"] == made["chromosomes"]
    assert only["far_bp"] == made["far_bp"]
    # The two runs differ in their output and ledger alone, which the identity pins.
    for key in ("out", "ledger_sha256"):
        only["identity"].pop(key), made["identity"].pop(key)
    assert only["identity"] == made["identity"]
    assert no_network == []


def test_a_full_import_resumes_from_a_plan_only_work_directory(
    tmp_path, monkeypatch, no_network
) -> None:
    out, work = tmp_path / "import.store", tmp_path / "work"
    supplied = ["--ledger", str(FIXTURE / "ledger.json"), "--parts", str(FIXTURE)]
    assert main(import_args(out, work, *supplied, "--plan-only")) == 0
    written = {
        name: ((work / name).read_bytes(), (work / name).stat().st_mtime_ns)
        for name in ("plan.json", "1.lifted.npz", "2.lifted.npz")
    }

    def replanned(*args, **kwargs):
        raise AssertionError("the full import planned again")

    monkeypatch.setattr(run, "_new_plan", replanned)
    assert main(import_args(out, work, *supplied)) == 0
    for name, (content, mtime) in written.items():
        assert (work / name).read_bytes() == content
        assert (work / name).stat().st_mtime_ns == mtime
    with opengwasdb.ld.open(out) as store:
        assert store.provenance["n_variants"] == 6
    assert no_network == []


def test_plan_only_repeated_reuses_its_plan(tmp_path, monkeypatch, no_network) -> None:
    out, work = tmp_path / "import.store", tmp_path / "work"
    assert main(import_args(out, work, "--plan-only")) == 0
    before = (work / "plan.json").stat().st_mtime_ns

    def replanned(*args, **kwargs):
        raise AssertionError("planned again")

    monkeypatch.setattr(run, "_new_plan", replanned)
    assert main(import_args(out, work, "--plan-only")) == 0
    assert (work / "plan.json").stat().st_mtime_ns == before
    assert no_network == []


def test_plan_only_refuses_a_work_directory_pinning_other_inputs(
    tmp_path, capsys, no_network
) -> None:
    out, work = tmp_path / "import.store", tmp_path / "work"
    assert main(import_args(out, work, "--plan-only")) == 0
    capsys.readouterr()
    assert main(import_args(out, work, "--plan-only", "--far-target-bp", "9000000")) == 2
    assert "pins different inputs" in capsys.readouterr().err
    assert no_network == []


def test_plan_only_refuses_an_output_that_exists(tmp_path, capsys, no_network) -> None:
    out, work = tmp_path / "import.store", tmp_path / "work"
    out.mkdir()
    assert main(import_args(out, work, "--plan-only")) == 2
    assert "exists" in capsys.readouterr().err
    assert not (work / "plan.json").exists()
    assert no_network == []
