"""`scripts/build_hgdp1kgp.py` (ogld#7): when a conversion counts as current, and every way the
recipe stops rather than go on. plink2 is a fake script that writes small filesets, or fails in
one chosen way (`FAKE_PLINK2`)."""

from __future__ import annotations

import importlib
import json
import stat
import sys
from pathlib import Path

import pytest
from script_modules import SCRIPTS, load_script

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
recipe = load_script("build_hgdp1kgp")
tools = importlib.import_module("test_panel.tools")

FAKE = """#!/bin/sh
# argv goes to $FAKE_ARGV; $FAKE_PLINK2 is ok, short (a sample missing), swap (another sample),
# nopgen (exit 0 without the pgen) or fail (exit 3 after the psam and pvar, as a disk-full would).
if [ "$1" = "--version" ]; then echo "PLINK fake $FAKE_VERSION"; exit 0; fi
echo "$@" >> "$FAKE_ARGV"
while [ $# -gt 0 ]; do
  case "$1" in --keep) keep=$2;; --out) out=$2;; esac
  shift
done
ids=$(tail -n +2 "$keep")
[ "$FAKE_PLINK2" = short ] && ids=$(echo "$ids" | head -n -1)
[ "$FAKE_PLINK2" = swap ] && ids=$(echo "$ids" | head -n -1; echo other)
printf '#IID\\tSEX\\n' > "$out.psam"
echo "$ids" | while read -r i; do printf '%s\\tNA\\n' "$i" >> "$out.psam"; done
echo pvar > "$out.pvar"
[ "$FAKE_PLINK2" = fail ] && exit 3
[ "$FAKE_PLINK2" = nopgen ] && exit 0
echo pgen > "$out.pgen"
"""
#: Stands in for /usr/bin/time: `time -v -o FILE command...`.
FAKE_TIME = """#!/bin/sh
out=$3; shift 3
"$@"; rc=$?
echo "Exit status: $rc" > "$out"
exit $rc
"""


def script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Roots under tmp_path, a keep file of three samples, a source BCF and the fake plink2."""
    for variable, name in (
        ("OGLD_PGEN_V2", "pgen"),
        ("OGLD_PHASED_V2", "bcf"),
        ("OGLD_KEEP", "keep"),
        ("OGLD_SCRATCH", "scratch"),
        ("OGLD_STORES", "stores"),
        ("OGLD_PCS", "pcs"),
    ):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / name))
    (tmp_path / "keep" / "AFR.keep").write_text("#IID\na\nb\nc\n")
    (tmp_path / "bcf" / recipe.BCF.format(chrom=22)).write_text("bcf")
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv"))
    monkeypatch.setenv("FAKE_VERSION", "1")
    monkeypatch.setenv("FAKE_PLINK2", "ok")
    return script(tmp_path / "plink2", FAKE)


def runs(tmp_path: Path) -> int:
    argv = tmp_path / "argv"
    return len(argv.read_text().splitlines()) if argv.exists() else 0


def convert(plink2: Path) -> str:
    return recipe.convert_one(plink2, "AFR", 22, 4, 1000)


def test_a_conversion_is_the_owners_command_and_is_stamped(world, tmp_path):
    assert "converted, 3 samples" in convert(world)
    out = recipe.prefix("AFR", 22)
    expected = tools.conversion_args(
        tmp_path / "bcf" / recipe.BCF.format(chrom=22), tmp_path / "keep" / "AFR.keep", out
    )
    given = (tmp_path / "argv").read_text().split()
    assert given == [*expected, "--threads", "4", "--memory", "1000"]
    stamp = json.loads(Path(f"{out}.stamp.json").read_text())
    assert stamp["command"] == " ".join(expected)


def test_a_fileset_with_a_matching_stamp_is_not_converted_again(world, tmp_path):
    convert(world)
    assert "skipped" in convert(world)
    assert runs(tmp_path) == 1


@pytest.mark.parametrize("change", ["keep", "plink2", "source", "file"])
def test_a_fileset_whose_inputs_changed_is_converted_again(world, tmp_path, monkeypatch, change):
    convert(world)
    if change == "keep":
        (tmp_path / "keep" / "AFR.keep").write_text("#IID\na\nb\n")
    elif change == "plink2":
        monkeypatch.setenv("FAKE_VERSION", "2")
    elif change == "source":
        (tmp_path / "bcf" / recipe.BCF.format(chrom=22)).write_text("another bcf")
    else:
        Path(f"{recipe.prefix('AFR', 22)}.pgen").unlink()
    assert "converted" in convert(world)
    assert runs(tmp_path) == 2


def test_a_conversion_that_failed_leaves_no_stamp_and_is_not_skipped_after(
    world, tmp_path, monkeypatch
):
    """plink2 wrote the psam and pvar and not the pgen, and its log would say `End time`."""
    monkeypatch.setenv("FAKE_PLINK2", "fail")
    with pytest.raises(SystemExit, match="failed"):
        convert(world)
    assert not Path(f"{recipe.prefix('AFR', 22)}.stamp.json").exists()
    monkeypatch.setenv("FAKE_PLINK2", "ok")
    assert "converted" in convert(world)
    assert Path(f"{recipe.prefix('AFR', 22)}.pgen").exists()


def test_a_failed_reconversion_removes_the_stamp_of_the_one_it_replaces(world, monkeypatch):
    convert(world)
    monkeypatch.setenv("FAKE_VERSION", "2")
    monkeypatch.setenv("FAKE_PLINK2", "fail")
    with pytest.raises(SystemExit):
        convert(world)
    monkeypatch.setenv("FAKE_VERSION", "1")
    assert not Path(f"{recipe.prefix('AFR', 22)}.stamp.json").exists()


def test_a_conversion_with_fewer_samples_than_the_keep_file_stops(world, monkeypatch):
    monkeypatch.setenv("FAKE_PLINK2", "short")
    with pytest.raises(SystemExit, match="converted 2 samples; the keep file has 3"):
        convert(world)
    assert not Path(f"{recipe.prefix('AFR', 22)}.stamp.json").exists()


def test_a_conversion_with_other_samples_of_the_same_count_stops(world, monkeypatch):
    monkeypatch.setenv("FAKE_PLINK2", "swap")
    with pytest.raises(SystemExit, match="not the keep file's"):
        convert(world)


def test_a_conversion_that_exits_zero_without_a_pgen_stops(world, monkeypatch):
    monkeypatch.setenv("FAKE_PLINK2", "nopgen")
    with pytest.raises(SystemExit, match=r"exited 0 but .*chr22\.pgen is missing"):
        convert(world)
    assert not Path(f"{recipe.prefix('AFR', 22)}.stamp.json").exists()


def test_a_missing_source_bcf_stops(world, tmp_path):
    (tmp_path / "bcf" / recipe.BCF.format(chrom=22)).unlink()
    with pytest.raises(SystemExit, match="missing"):
        convert(world)


def test_the_keep_file_needs_its_header(tmp_path):
    keep = tmp_path / "x.keep"
    keep.write_text("a\nb\n")
    with pytest.raises(SystemExit, match="#IID"):
        recipe.keep_iids(keep)


def test_the_build_command_is_one_ogld_accepts(tmp_path):
    from ogld.cli import _parser

    pgens = [tmp_path / f"chr{c}.pgen" for c in recipe.CHROMS]
    command = recipe.build_command(
        "AFR", pgens, tmp_path / "s.store", tmp_path / "k", tmp_path / "p", 16
    )
    args = _parser().parse_args(command[1:])
    assert args.pgen == pgens
    assert (args.r_scale, args.k, args.threads) == (recipe.PANEL_SCALE, recipe.K, 16)
    assert args.t_partner == recipe.PANEL_T_PARTNER
    assert (args.panel_id, args.ancestry) == (recipe.panel_id("AFR"), "AFR")


def test_the_validate_command_is_one_ogld_accepts(tmp_path):
    from ogld.cli import _parser

    args = _parser().parse_args(recipe.validate_command(tmp_path / "s", 16)[1:])
    assert (args.location, args.threads) == (str(tmp_path / "s"), 16)


def test_the_arguments_dedupe_panels_and_refuse_an_unknown_one():
    assert recipe.parse(["convert", "--panel", "EAS", "--panel", "EAS"]).panel == ["EAS"]
    with pytest.raises(SystemExit):
        recipe.parse(["build", "--panel", "XYZ"])


@pytest.mark.parametrize("jobs", [0, 9])
def test_conversions_are_capped_at_eight(jobs):
    with pytest.raises(SystemExit, match="--jobs"):
        recipe.convert(["AFR"], jobs, 4, 16000)


@pytest.mark.parametrize(("builds", "threads"), [(3, 16), (0, 16), (2, 17)])
def test_builds_are_capped_at_two_of_sixteen_threads(builds, threads):
    with pytest.raises(SystemExit, match=r"--builds|--threads"):
        recipe.build(["AFR"], builds, threads)


def test_a_build_with_missing_inputs_names_them(world):
    with pytest.raises(SystemExit, match=r"missing inputs.*AFR\.eigenvec"):
        recipe.build_one("AFR", 16)


def test_an_existing_store_is_refused_and_left_alone(world, tmp_path):
    (tmp_path / "pcs" / "AFR.eigenvec").write_text("x")
    for chrom in recipe.CHROMS:
        Path(f"{recipe.prefix('AFR', chrom)}.pgen").parent.mkdir(exist_ok=True)
        Path(f"{recipe.prefix('AFR', chrom)}.pgen").write_text("x")
    store = recipe.store_path("AFR")
    store.mkdir()
    (store / "kept").write_text("mine")
    with pytest.raises(SystemExit, match="exists"):
        recipe.build_one("AFR", 16)
    assert (store / "kept").read_text() == "mine"


def test_a_stage_that_exits_non_zero_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "TIME", str(script(tmp_path / "time", FAKE_TIME)))
    log = tmp_path / "AFR.log"
    recipe.timed(["true"], log, "build")
    with pytest.raises(SystemExit, match=r"validate failed \(1\)"):
        recipe.timed(["false"], log, "validate")
    assert "validate end rc=1" in log.read_text()


def test_a_failed_panel_cancels_the_ones_not_yet_started(monkeypatch):
    started = []

    def build_one(panel: str, threads: int) -> str:
        started.append(panel)
        raise SystemExit(f"{panel} failed")

    monkeypatch.setattr(recipe, "build_one", build_one)
    with pytest.raises(SystemExit, match="AFR failed"):
        recipe.build(["AFR", "EUR", "EAS"], 1, 16)
    assert started == ["AFR"]
