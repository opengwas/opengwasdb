"""`opengwasdb ld`: its exit statuses, its place under the engine CLI, and its refusal of the
`OGLD_*` variables it replaces (#457)."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import store_copies
from ld_script_modules import SCRIPTS
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.ld import cli, env
from opengwasdb.store_reader.identity import VariantNormalisationError

STORE = store_copies.TEST_STORE


def test_ld_is_a_group_of_the_engine_cli() -> None:
    result = CliRunner().invoke(app, ["ld", "--help"])
    assert result.exit_code == 0
    for command in ("build", "import", "validate", "info", "derive-partners"):
        assert command in result.output, f"`opengwasdb ld` lacks {command}"


def test_the_engine_validate_is_not_the_ld_validate() -> None:
    # The engine's own top-level `validate` stays; the ld one sits under `ld`.
    top = CliRunner().invoke(app, ["validate", "--help"]).output
    ld = CliRunner().invoke(app, ["ld", "validate", "--help"]).output
    assert top != ld
    assert "location" in ld.lower()


def test_validate_and_info_through_the_engine_cli() -> None:
    runner = CliRunner()
    validated = runner.invoke(app, ["ld", "validate", str(STORE)])
    assert validated.exit_code == 0, validated.output
    assert "ok   partners.symmetric" in validated.output
    info = runner.invoke(app, ["ld", "info", str(STORE)])
    assert info.exit_code == 0
    assert json.loads(info.output)["format_version"]


def test_a_store_that_breaks_a_rule_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = store_copies.copy_store(tmp_path / "broken")
    victim = next(p for p in sorted(broken.rglob("c")) if p.is_dir() and "partners" in p.parts)
    shutil.rmtree(victim)
    assert cli.main(["validate", str(broken)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_a_refusal_exits_2_and_names_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    # k = 10 (the default) with no PC File is refused before anything is read.
    argv = [
        "build", "out", "--pgen", "x.pgen", "--keep", "k", "--panel-id", "p", "--ancestry", "a",
        "--genotype-source", "g", "--r-scale", "511", "--t-partner", "0.01",
    ]  # fmt: skip
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.startswith("opengwasdb ld build: ")


def test_a_variant_normalisation_refusal_exits_2_naming_the_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # #456 moved the identity refusal to the engine's VariantNormalisationError; it is still a
    # refusal, exit 2, in the `opengwasdb ld <command>: <problem>` form.
    def refuse(location: str) -> None:
        raise VariantNormalisationError("unsupported allele 'N'")

    monkeypatch.setattr(cli, "open_store", refuse)
    assert cli.main(["info", str(STORE)]) == 2
    assert capsys.readouterr().err == "opengwasdb ld info: unsupported allele 'N'\n"


@pytest.mark.parametrize("missing", ["--r-scale", "--t-partner", "--keep", "--panel-id", "--pgen"])
def test_a_build_without_a_required_input_is_a_usage_error(
    missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "build", "out", "--pgen", "x.pgen", "--keep", "k", "--panel-id", "p", "--ancestry", "a",
        "--genotype-source", "g", "--r-scale", "511", "--t-partner", "0.01",
    ]  # fmt: skip
    at = argv.index(missing)
    del argv[at : at + 2]
    with pytest.raises(SystemExit) as refused:
        cli.main(argv)
    assert refused.value.code == 2
    assert missing in capsys.readouterr().err


def test_a_thread_count_below_one_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as refused:
        cli.main(["validate", str(STORE), "--threads", "0"])
    assert refused.value.code == 2


# ------------------------------------------------------------------ the old variable names


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("OGLD_KEEP", "OPENGWASDB_LD_KEEP"),
        ("OGLD_STORE_EUR", "OPENGWASDB_LD_STORE_EUR"),
    ],
)
def test_an_old_variable_is_refused_by_the_cli_naming_its_replacement(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(old, "x")
    assert cli.main(["info", str(STORE)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("opengwasdb ld info: ") and f"{old} is now {new}" in err
    assert err.count("is now") == 1 and len(err.splitlines()) == 1, "reported once"


def test_the_api_services_timeout_variable_is_not_an_old_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OGDB_STORE_TIMEOUT_SECONDS is ogdb-api's live setting; `opengwasdb ld` has no say in it.
    monkeypatch.setenv("OGDB_STORE_TIMEOUT_SECONDS", "30")
    assert cli.main(["info", str(STORE)]) == 0


def test_an_old_variable_is_refused_through_the_engine_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OGLD_KEEP", "x")
    result = CliRunner().invoke(app, ["ld", "validate", str(STORE)])
    assert result.exit_code == 2


@pytest.mark.parametrize("old", ["OGLD_KEEP", "OGLD_STORE_EUR", "OGLD_PLINK2"])
def test_the_scripts_roots_refuse_an_old_variable(
    old: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    roots = importlib.import_module("test_panel.roots")
    monkeypatch.setenv(old, "x")
    with pytest.raises(SystemExit) as refused:
        roots.root("OPENGWASDB_LD_KEEP")
    assert refused.value.code == 2, "a refusal, not 'the store breaks a rule' (1)"
    assert f"{old} is now OPENGWASDB_LD_" in capsys.readouterr().err


def test_env_read_raises_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The package reports nothing: only what faces the terminal does."""
    monkeypatch.setenv("OGLD_KEEP", "x")
    with pytest.raises(env.LegacyVariable, match="OGLD_KEEP is now OPENGWASDB_LD_KEEP"):
        env.read("OPENGWASDB_LD_KEEP", "dflt")
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""


def test_the_new_name_is_read_and_the_old_one_is_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENGWASDB_LD_KEEP", raising=False)
    assert env.read("OPENGWASDB_LD_KEEP", "dflt") == "dflt"
    monkeypatch.setenv("OPENGWASDB_LD_KEEP", "mine")
    assert env.read("OPENGWASDB_LD_KEEP", "dflt") == "mine"
    with pytest.raises(ValueError, match="not an OPENGWASDB_LD_"):
        env.read("HOME", "")


@pytest.mark.parametrize(
    "script",
    [
        "acceptance/run.sh",
        "acceptance/run_stores.sh",
        "acceptance/run_genotypes.sh",
        "realdata/run.sh",
    ],
)
def test_the_shell_runners_refuse_an_old_variable(script: str) -> None:
    root = Path(__file__).resolve().parents[2]
    ran = subprocess.run(
        ["bash", str(root / "tests" / "ld" / script)],
        env={**os.environ, "OGLD_KEEP": "x"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert ran.returncode == 2
    assert "OGLD_KEEP is now OPENGWASDB_LD_KEEP" in ran.stderr
