"""`scripts/build_hgdp1kgp.py` (ogld#7): its argument handling, the command lines it forms, and
when a conversion counts as current. plink2 is never run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from script_modules import SCRIPTS, load_script

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
recipe = load_script("build_hgdp1kgp")

PLINK2 = Path("/bin/plink2")
LOG = """PLINK v2.0.0-a.6.9LM 64-bit Intel (29 Jan 2025)
Options in effect:
  --bcf {bcf}
  --keep {keep}
  --make-pgen pvar-cols=filter
  --max-alleles 2
  --memory 16000
  --new-id-max-allele-len 1000
  --out {out}
  --set-all-var-ids @:#:$r:$a
  --threads 4

Hostname: x
{end}"""


def make_conversion(tmp_path: Path, iids: list[str], keep_ids: list[str], finished: bool = True):
    keep = tmp_path / "AFR.keep"
    keep.write_text("#IID\n" + "".join(f"{i}\n" for i in keep_ids))
    out = tmp_path / "pgen" / "chr22"
    out.parent.mkdir()
    bcf = tmp_path / "chr22.bcf"
    command = recipe.conversion_command(PLINK2, bcf, keep, out)
    for ext in ("pgen", "pvar"):
        Path(f"{out}.{ext}").write_text("x")
    Path(f"{out}.psam").write_text("#IID\tSEX\n" + "".join(f"{i}\tNA\n" for i in iids))
    end = "End time: Tue Nov 10 13:04:57 2026\n" if finished else ""
    Path(f"{out}.log").write_text(LOG.format(bcf=bcf, keep=keep, out=out, end=end))
    return command, out, keep


def test_conversion_command_is_build_inputs_section_3():
    command = recipe.conversion_command(PLINK2, Path("a.bcf"), Path("k.keep"), Path("o"))
    assert command == [
        "/bin/plink2", "--bcf", "a.bcf", "--keep", "k.keep", "--max-alleles", "2",
        "--set-all-var-ids", "@:#:$r:$a", "--new-id-max-allele-len", "1000",
        "--make-pgen", "pvar-cols=filter", "--out", "o", "--threads", "4", "--memory", "16000",
    ]  # fmt: skip


def test_build_command_names_panel_pcs_keep_and_scale():
    pgens = [Path(f"c{c}.pgen") for c in (1, 2)]
    command = recipe.build_command(
        "AFR", pgens, Path("s.store"), Path("AFR.keep"), Path("AFR.eigenvec"), 16
    )
    assert command[:3] == ["ogld", "build", "s.store"]
    pairs = {
        command[i]: command[i + 1] for i in range(3, len(command) - 1, 2) if command[i] != "--pgen"
    }
    assert [command[i + 1] for i, t in enumerate(command) if t == "--pgen"] == [
        "c1.pgen",
        "c2.pgen",
    ]
    assert pairs["--r-scale"] == "511"
    assert pairs["--k"] == "10"
    assert pairs["--threads"] == "16"
    assert pairs["--keep"] == "AFR.keep"
    assert pairs["--pcs"] == "AFR.eigenvec"
    assert pairs["--panel-id"] == "hgdp1kgp-v2-AFR"
    assert pairs["--ancestry"] == "AFR"
    assert pairs["--info-min"] == "none"


def test_validate_command():
    assert recipe.validate_command(Path("s"), 16) == ["ogld", "validate", "--threads", "16", "s"]


def test_paths_follow_the_brief(monkeypatch):
    for variable in ("OGLD_PGEN_V2", "OGLD_STORES", "OGLD_KEEP"):
        monkeypatch.delenv(variable, raising=False)
    assert str(recipe.prefix("EUR", 6)) == "/data/opengwasdb/work/ogld/phased_v2_pgen/EUR/chr6"
    assert recipe.store_path("AMR").name == "hgdp1kgp-v2-AMR-k10.store"
    assert recipe.keep_file("SAS") == Path("/data/opengwasdb/work/ogld/keep/SAS.keep")
    assert recipe.PANELS[0] == "AFR"


def test_a_finished_matching_conversion_is_current(tmp_path):
    command, out, keep = make_conversion(tmp_path, ["a", "b"], ["b", "a"])
    assert recipe.conversion_current(command, out, keep)


@pytest.mark.parametrize(
    ("iids", "keep_ids", "finished"),
    [
        (["a"], ["a", "b"], True),  # a sample short
        (["a", "c"], ["a", "b"], True),  # the same count, a different sample
        (["a", "b"], ["a", "b"], False),  # the run never finished
    ],
)
def test_a_conversion_that_does_not_match_is_redone(tmp_path, iids, keep_ids, finished):
    command, out, keep = make_conversion(tmp_path, iids, keep_ids, finished)
    assert not recipe.conversion_current(command, out, keep)


def test_a_conversion_with_other_options_is_redone(tmp_path):
    command, out, keep = make_conversion(tmp_path, ["a"], ["a"])
    other = [*command[:-4], "--max-alleles", "3", *command[-4:]]
    other[other.index("--max-alleles") + 1] = "3"
    assert not recipe.conversion_current(other, out, keep)


def test_threads_and_memory_do_not_make_a_conversion_stale(tmp_path):
    command, out, keep = make_conversion(tmp_path, ["a"], ["a"])
    bcf = Path(command[command.index("--bcf") + 1])
    again = recipe.conversion_command(PLINK2, bcf, keep, out, threads=8, memory_mb=1000)
    assert recipe.conversion_current(again, out, keep)


def test_a_missing_log_or_file_is_not_current(tmp_path):
    command, out, keep = make_conversion(tmp_path, ["a"], ["a"])
    Path(f"{out}.pgen").unlink()
    assert not recipe.conversion_current(command, out, keep)


def test_keep_file_needs_its_header(tmp_path):
    keep = tmp_path / "x.keep"
    keep.write_text("a\nb\n")
    with pytest.raises(SystemExit, match="#IID"):
        recipe.keep_iids(keep)


def test_arguments_default_to_all_five_panels_afr_first():
    args = recipe.parse(["build"])
    assert args.panel == ["AFR", "EUR", "EAS", "SAS", "AMR"]
    assert (args.builds, args.threads) == (2, 16)
    assert recipe.parse(["convert"]).jobs == 8
    assert recipe.parse(["convert", "--panel", "EAS", "--panel", "EAS"]).panel == ["EAS"]


def test_arguments_refuse_an_unknown_panel():
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


def test_a_build_with_missing_inputs_names_them(tmp_path, monkeypatch):
    monkeypatch.setenv("OGLD_PGEN_V2", str(tmp_path / "pgen"))
    monkeypatch.setenv("OGLD_KEEP", str(tmp_path / "keep"))
    monkeypatch.setenv("OGLD_PCS", str(tmp_path / "pcs"))
    with pytest.raises(SystemExit, match=r"missing inputs.*AFR\.keep"):
        recipe.build_one("AFR", 16)
