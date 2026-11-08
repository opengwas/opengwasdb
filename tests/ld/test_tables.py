"""The parts of the skeleton that are built: result tables, BuildInputs, LDStore's facts, the CLI.

These are containers and plumbing, implemented with the skeleton because every ticket
uses them. The queries that fill them are tests/contract/'s.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ogld
from ogld import cli
from ogld.format import t_far
from ogld.store import Partners, StoreReader

EUR_K10 = t_far(771, 10)


def variants(
    n: int, *, chrom: str = "22", start: int = 1_000_000, step: int = 1_000_000
) -> ogld.Variants:
    positions = np.arange(start, start + n * step, step, dtype=np.int64)
    return ogld.Variants(
        chrom=np.array([chrom] * n),
        position=positions,
        alid=np.array([f"{chrom}:{p}:A:G" for p in positions]),
        a1=np.array(["A"] * n),
        a2=np.array(["G"] * n),
        a1_frequency=np.full(n, 0.2),
    )


def test_a_table_refuses_columns_of_different_lengths() -> None:
    good = variants(3)
    assert len(good) == 3
    assert list(good.columns()) == ["chrom", "position", "alid", "a1", "a2", "a1_frequency"]
    with pytest.raises(ValueError, match="equally long"):
        ogld.Variants(**{**good.columns(), "position": np.arange(2, dtype=np.int64)})


def test_partners_refuses_an_indptr_that_does_not_end_at_its_entries() -> None:
    ok = Partners(
        indptr=np.array([0, 1, 1], dtype=np.int64),
        partner=np.array([5], dtype=np.int64),
        r=np.array([0.5]),
    )
    assert len(ok) == 2
    with pytest.raises(ValueError, match="indptr"):
        Partners(
            indptr=np.array([0, 2], dtype=np.int64),
            partner=np.array([5], dtype=np.int64),
            r=np.array([0.5]),
        )


def matrix(r: list[list[float]], fill: float | None = None, step: int = 1_000_000) -> ogld.LDMatrix:
    return ogld.LDMatrix(
        r=np.array(r),
        variants=variants(len(r), step=step),
        absent=(),
        t_far=EUR_K10,
        near_bp=3_000_000,
        far_bp=10_000_000,
        fill=fill,
    )


def test_an_ld_matrix_must_be_square_over_its_variants() -> None:
    with pytest.raises(ValueError, match="variants"):
        ogld.LDMatrix(
            r=np.eye(2),
            variants=variants(3),
            absent=(),
            t_far=EUR_K10,
            near_bp=3_000_000,
            far_bp=10_000_000,
            fill=None,
        )


def test_an_ld_matrix_says_what_each_nan_means() -> None:
    # Variants 4 Mb apart: 0-1 is a Far Pair candidate, 0-2 is 8 Mb, 0-3 is 12 Mb.
    nan = math.nan
    out = matrix(
        [[1, nan, 0.4, nan], [nan, 1, nan, nan], [0.4, nan, 1, nan], [nan, nan, nan, 1]],
        step=4_000_000,
    ).kinds()
    assert out[0].tolist() == ["stored", "bound", "stored", "not_measured"]


@pytest.mark.parametrize("fill", [0.0, 0.5, -9.0])
def test_a_filled_matrix_has_no_kinds(fill: float) -> None:
    # A filled cell cannot be told from a stored one, so kinds() is defined only without a
    # fill: ask for the matrix again with fill=None.
    filled = matrix([[1, fill], [fill, 1]], fill=fill, step=4_000_000)
    with pytest.raises(ValueError, match="fill"):
        filled.kinds()


def build_inputs(**overrides: Any) -> ogld.BuildInputs:
    fields: dict[str, Any] = {
        "pgen": (Path("chr22.pgen"),),
        "keep": Path("EUR.keep"),
        "pcs": Path("pcs.eigenvec"),
        "pc_source": "gnomAD v3.1.2 hgdp_tgp_meta.subcontinental_pca.pca_scores",
        "panel_id": "hgdp1kgp-v2-EUR",
        "ancestry": "EUR",
        "genotype_source": "gnomAD v3.1.2 HGDP+1kG",
    }
    return ogld.BuildInputs(**{**fields, **overrides})


def test_build_inputs_defaults_are_the_owner_s() -> None:
    # Epic #1 and the owner's decisions of 7 November 2026: up to 10 PCs, at least 5.
    inputs = build_inputs()
    assert (inputs.k, inputs.k_min, inputs.mac_min, inputs.info_min) == (10, 5, 20, 0.8)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pgen": ()}, "pgen"),
        ({"k": -1}, "negative"),
        ({"pcs": None}, "PC File"),
        ({"pc_source": None}, "PC File"),
        ({"k_min": 11}, "k_min"),
        ({"k_min": -1}, "k_min"),
    ],
)
def test_build_inputs_refuse_what_cannot_be_built(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ogld.InputError, match=message):
        build_inputs(**overrides)


def test_k_zero_needs_no_pc_file_and_no_minimum() -> None:
    # With k = 0 no PC is read, so the minimum does not apply.
    assert build_inputs(k=0, pcs=None, pc_source=None).pcs is None


class FakeReader(StoreReader):
    """A StoreReader whose provenance is three facts; nothing else is read."""

    @property
    def provenance(self) -> Any:
        return {"t_far": 0.0274, "near_bp": 3_000_000, "far_bp": 8_000_000}


def test_an_ld_store_reports_the_facts_its_store_recorded() -> None:
    opened = ogld.LDStore(FakeReader())
    assert (opened.t_far, opened.near_bp, opened.far_bp) == (0.0274, 3_000_000, 8_000_000)


def test_the_cli_needs_a_command() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2


def test_the_cli_refuses_a_build_it_cannot_run(capsys: pytest.CaptureFixture[str]) -> None:
    # k = 10 with no PC File is refused before anything is read, with exit status 2.
    argv = ["build", "out", "--pgen", "x.pgen", "--keep", "k", "--panel-id", "p"]
    status = cli.main([*argv, "--ancestry", "EUR", "--genotype-source", "g"])
    assert status == 2
    assert "PC File" in capsys.readouterr().err


def test_the_cli_reports_each_rule_and_fails_on_any_finding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing: dict[str, list[ogld.Finding]] = {"near.codes": [], "index.symmetric": []}
    assert cli.report(passing) == 0
    one = ogld.Finding("index.symmetric", "22:5", "one direction only")
    assert cli.report({**passing, "index.symmetric": [one]}) == 1
    out = capsys.readouterr().out
    assert "ok   near.codes" in out
    assert "FAIL index.symmetric" in out
    assert "22:5: one direction only" in out


BUILD_ARGS = ["build", "out", "--pgen", "x.pgen", "--keep", "k", "--pcs", "p.eigenvec"]
BUILD_ARGS += ["--pc-source", "s", "--panel-id", "p", "--ancestry", "EUR", "--genotype-source", "g"]


def test_the_cli_s_defaults_are_build_inputs_own(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ogld.BuildInputs] = []
    monkeypatch.setattr(cli, "build", lambda inputs, out, threads=1: seen.append(inputs) or {})
    assert cli.main(BUILD_ARGS) == 0
    expected = ogld.BuildInputs(
        pgen=(Path("x.pgen"),),
        keep=Path("k"),
        pcs=Path("p.eigenvec"),
        pc_source="s",
        panel_id="p",
        ancestry="EUR",
        genotype_source="g",
    )
    assert seen == [expected]


def test_the_cli_exits_2_on_an_input_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(inputs: object, out: object, threads: int = 1) -> None:
        raise ogld.InputError("chr22:1:A:G has a missing call")

    monkeypatch.setattr(cli, "build", refuse)
    assert cli.main(BUILD_ARGS) == 2


def test_the_cli_lets_any_other_fault_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ValueError from inside the build, such as quantise refusing r = 1.3, is a builder
    # fault: exit 2 ("an input it cannot use") would hide it.
    def fault(inputs: object, out: object, threads: int = 1) -> None:
        raise ValueError("not a correlation, so it has no code: [1.3]")

    monkeypatch.setattr(cli, "build", fault)
    with pytest.raises(ValueError, match="not a correlation"):
        cli.main(BUILD_ARGS)
