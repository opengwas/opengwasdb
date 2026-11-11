"""`scripts/make_keeps.py`: each HGDP+1kGP panel's keep file is its unrelated samples (owner,
10 November 2026; ADR 0002, decision 11), on literal rows of gnomAD v3.1.2's sample metadata.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest
from script_modules import SCRIPTS, load_script

# The script imports the scripts' shared helpers (`test_panel`), as it does when run from there.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
make_keeps = load_script("make_keeps")

#: gnomAD's metadata as the script reads it: `s`, then columns, `relatedness_inference` a JSON
#: object whose `related` is the maximal-independent-set flag.
HEADER = ["s", "project_meta", "relatedness_inference", "sample_filters"]


def metadata(tmp_path: Path, related: dict[str, bool]) -> Path:
    out = tmp_path / "meta.tsv.bgz"
    rows = ["\t".join(HEADER)]
    for iid, flag in related.items():
        inference = {"related_samples": [], "related": flag}
        rows.append("\t".join([iid, "{}", json.dumps(inference), "{}"]))
    with gzip.open(out, "wt") as handle:
        handle.write("\n".join(rows) + "\n")
    return out


def pc_file(directory: Path, panel: str, iids: list[str]) -> None:
    lines = ["#IID\tPC1\tPC2", *(f"{iid}\t0.1\t-0.2" for iid in iids)]
    (directory / f"{panel}.eigenvec").write_text("\n".join(lines) + "\n")


def test_a_keep_file_is_the_panel_s_unrelated_samples_in_its_pc_file_s_order(
    tmp_path: Path,
) -> None:
    meta = metadata(tmp_path, {"NA2": False, "NA1": True, "HG3": False, "XX9": False})
    (tmp_path / "pcs").mkdir()
    pc_file(tmp_path / "pcs", "EUR", ["NA2", "NA1", "HG3"])
    argv = ["--pcs", str(tmp_path / "pcs"), "--meta", str(meta), "--out", str(tmp_path / "keep")]
    make_keeps.main([*argv, "--panel", "EUR"])
    keep = tmp_path / "keep" / "EUR.keep"
    # NA1 is related, and XX9, in the metadata, is no EUR sample.
    assert keep.read_text() == "#IID\nNA2\nHG3\n"
    first = keep.read_bytes()
    make_keeps.main([*argv, "--panel", "EUR"])
    assert keep.read_bytes() == first, "the same inputs write the same bytes"


def test_a_sample_missing_from_the_metadata_is_refused(tmp_path: Path) -> None:
    meta = metadata(tmp_path, {"NA2": False})
    (tmp_path / "pcs").mkdir()
    pc_file(tmp_path / "pcs", "AFR", ["NA2", "NOT-IN-GNOMAD"])
    argv = ["--pcs", str(tmp_path / "pcs"), "--meta", str(meta), "--out", str(tmp_path / "keep")]
    with pytest.raises(SystemExit, match="NOT-IN-GNOMAD"):
        make_keeps.main([*argv, "--panel", "AFR"])
    assert not (tmp_path / "keep" / "AFR.keep").exists()


@pytest.mark.parametrize("state", ["missing", "empty"])
def test_a_missing_or_empty_pc_file_is_refused_naming_it(tmp_path: Path, state: str) -> None:
    meta = metadata(tmp_path, {"NA2": False})
    (tmp_path / "pcs").mkdir()
    if state == "empty":
        (tmp_path / "pcs" / "SAS.eigenvec").write_text("")
    argv = ["--pcs", str(tmp_path / "pcs"), "--meta", str(meta), "--out", str(tmp_path / "keep")]
    with pytest.raises(SystemExit, match=r"SAS\.eigenvec"):
        make_keeps.main([*argv, "--panel", "SAS"])
