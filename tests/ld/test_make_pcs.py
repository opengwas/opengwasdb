"""`scripts/ld/make_pcs.py` at its seams (review round 2 of ogld#5, standards a6), with no PCA run and
no plink2: the commit its report records, its refusal of a checkout that is not that commit's
code, the scan's input (every variant, the PC Check choosing the common ones itself) and the
fields of the PC Check record it writes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ld_script_modules import SCRIPTS, load_script

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
make_pcs = load_script("make_pcs")

N = 100
ROOT = SCRIPTS.parents[1]


def settings(tmp_path: Path) -> object:
    return make_pcs.Settings(
        tmp_path / "work", tmp_path / "out", tmp_path, tmp_path, 2_000_000, Path("plink2"), 1, 1, 1
    )


def genotypes(seed: int, rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """`rows` variants every 25 kb, the first quarter rare (MAF < 0.05), for N samples."""
    rng = np.random.default_rng(seed)
    freq = np.where(np.arange(rows) < rows // 4, 0.01, rng.uniform(0.1, 0.5, rows))
    return np.arange(1, rows + 1) * 25_000, rng.binomial(2, freq[:, None], (rows, N)).astype(
        np.int8
    )


def test_the_scan_reads_every_variant_and_the_pc_check_picks_the_common_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pos, g = genotypes(1)
    asked: list[list[str]] = []
    monkeypatch.setattr(make_pcs.Settings, "plink", lambda self, args, log: asked.append(args))
    monkeypatch.setattr(make_pcs, "read_bed", lambda prefix: ([], [f"s{i}" for i in range(N)], g))
    monkeypatch.setattr(make_pcs, "bim_positions", lambda prefix: pos)
    fam, windows = make_pcs._scan_chromosome(settings(tmp_path), "EUR", 7, Path("p"), Path("k"))
    assert "--maf" not in asked[0], "the module selects the common variants, not plink2"
    assert "--make-bed" in asked[0]
    common = make_pcs.pc_check.common(g)
    assert (~common).sum() >= 100, "the fixture must hold rare variants for this to mean anything"
    alone = make_pcs.pc_check.windows(np.full(int(common.sum()), "7"), pos[common], g[common])
    assert windows.where == alone.where
    np.testing.assert_allclose(np.abs(windows.basis), np.abs(alone.basis), atol=1e-12)
    assert fam == [f"s{i}" for i in range(N)]


def test_the_scan_record_holds_every_field_over_every_chromosome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = settings(tmp_path)
    s.out.mkdir()
    rng = np.random.default_rng(2)
    iids = [f"s{i}" for i in range(N)]
    lines = ["#IID\t" + "\t".join(f"PC{k}" for k in range(1, 21))]
    lines += [iid + "\t" + "\t".join(f"{x:.6f}" for x in rng.standard_normal(20)) for iid in iids]
    (s.out / "EUR.eigenvec").write_text("\n".join(lines) + "\n")

    def scan(s: object, panel: str, chrom: int, pgen: Path, keep: Path) -> tuple[list[str], object]:
        pos, g = genotypes(chrom, rows=120)
        return iids, make_pcs.pc_check.windows(np.full(len(pos), str(chrom)), pos, g)

    monkeypatch.setattr(make_pcs, "_scan_chromosome", scan)
    (s.work / "EUR").mkdir(parents=True)
    record = make_pcs._scan_panel(s, "EUR", {c: Path(f"chr{c}") for c in range(1, 23)})
    assert list(record) == [
        "window_bp", "step_bp", "min_share", "max_ratio", "windows", "scan", "dropped", "null",
        "kept", "covariates", "k_target", "k_min", "k",
    ]  # fmt: skip
    # 3 Mb a chromosome, common from 0.775 Mb: windows start there and 2 Mb on, so two each.
    assert record["windows"] == 44, "the windows of all 22 chromosomes, merged"
    assert [e["pc"] for e in record["scan"]] == [f"PC{k}" for k in range(1, 21)]
    fields = {"pc", "window", "window_share", "rest_share", "ratio", "local"}
    assert all(fields | {"window_null_q99", "rest_null_q99"} == set(e) for e in record["scan"])
    assert (record["k_target"], record["k_min"]) == (10, 5)


def head() -> str:
    done = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return done.stdout.strip()


def test_the_report_names_the_commit_and_a_dirty_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        make_pcs.checkout, "problem", lambda root, installed: "a change: opengwasdb/ld/x.py"
    )
    with pytest.raises(SystemExit, match=r"a change: opengwasdb/ld/x\.py"):
        make_pcs._commit()
    monkeypatch.setattr(make_pcs.checkout, "problem", lambda root, installed: "")
    # main re-executes itself with one BLAS thread when its environment lacks it: not here.
    monkeypatch.setattr(make_pcs, "single_threaded_blas", lambda: None)
    monkeypatch.setattr(
        make_pcs, "_convert_all", lambda s: {c: Path(f"c{c}") for c in range(1, 23)}
    )
    monkeypatch.setattr(
        make_pcs,
        "_panel_pcs",
        lambda s, panel, keep, pgens: {
            "n_samples": N,
            "variants_pruned_in": 1,
            "eigenvalues": [1.0],
        },
    )
    monkeypatch.setattr(make_pcs, "_scan_panel", lambda s, panel, pgens: {"pc_check": "done"})
    monkeypatch.setattr(make_pcs, "run", lambda cmd, log: "up")
    monkeypatch.setattr(make_pcs, "version", lambda binary: "PLINK v2 (test)")
    for c in range(1, 23):
        name = make_pcs.BCF.format(chrom=c)
        (tmp_path / f"{name}.complete").write_text(json.dumps({"crc32c": f"crc{c}"}))
    argv = ["--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"), "--bcf-dir", str(tmp_path)]
    argv += ["--meta", str(tmp_path / "m"), "--plink2", "plink2", "--panel", "EUR=keep"]
    make_pcs.main(argv)
    report = json.loads((tmp_path / "o" / "EUR.json").read_text())
    assert report["commit"] == head()
    assert report["pc_check"] == {"pc_check": "done"}
