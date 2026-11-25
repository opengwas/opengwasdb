"""The parts of the skeleton that are built: result tables, BuildInputs, LDStore's facts, the CLI.

These are containers and plumbing, implemented with the skeleton because every ticket
uses them. The queries that fill them are tests/ld/contract/'s.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import opengwasdb.ld
from opengwasdb.ld import cli
from opengwasdb.ld.format import t_far
from opengwasdb.ld.store import Partners
from ogquery.ld.store import StoreReader

EUR_K10 = t_far(771, 10)


def variants(
    n: int, *, chrom: str = "22", start: int = 1_000_000, step: int = 1_000_000
) -> opengwasdb.ld.Variants:
    positions = np.arange(start, start + n * step, step, dtype=np.int64)
    return opengwasdb.ld.Variants(
        chrom=np.array([chrom] * n),
        position=positions,
        alid=np.array([f"{chrom}:{p}:A:G" for p in positions]),
        a1=np.array(["A"] * n),
        a2=np.array(["G"] * n),
        a1_frequency=np.full(n, 0.2),
    )


def test_partners_refuses_an_indptr_that_does_not_end_at_its_entries() -> None:
    one = np.array([5], dtype=np.int64)
    ok = Partners(
        indptr=np.array([0, 1, 1], dtype=np.int64), partner=one, r=np.array([0.5]), distance=one
    )
    assert len(ok) == 2
    with pytest.raises(ValueError, match="indptr"):
        Partners(
            indptr=np.array([0, 2], dtype=np.int64), partner=one, r=np.array([0.5]), distance=one
        )
    with pytest.raises(ValueError, match="distance"):
        Partners(
            indptr=np.array([0, 1], dtype=np.int64),
            partner=one,
            r=np.array([0.5]),
            distance=np.array([5, 6], dtype=np.int64),
        )


def matrix(
    r: list[list[float]], fill: float | None = None, step: int = 1_000_000
) -> opengwasdb.ld.LDMatrix:
    return opengwasdb.ld.LDMatrix(
        r=np.array(r),
        variants=variants(len(r), step=step),
        absent=(),
        t_far=EUR_K10,
        near_bp=3_000_000,
        far_bp=10_000_000,
        fill=fill,
    )


def build_inputs(**overrides: Any) -> opengwasdb.ld.BuildInputs:
    fields: dict[str, Any] = {
        "pgen": (Path("chr22.pgen"),),
        "keep": Path("EUR.keep"),
        "pcs": Path("pcs.eigenvec"),
        "pc_source": "gnomAD v3.1.2 hgdp_tgp_meta.subcontinental_pca.pca_scores",
        "panel_id": "hgdp1kgp-v2-EUR",
        "ancestry": "EUR",
        "genotype_source": "gnomAD v3.1.2 HGDP+1kG",
        "r_scale": 511,
        "t_partner": 0.01,
    }
    return opengwasdb.ld.BuildInputs(**{**fields, **overrides})


def test_build_inputs_defaults_are_the_owner_s() -> None:
    # Epic opengwas/ogld#1 and the owner's decisions of 7 November 2026: up to 10 PCs, at least 5.
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
    with pytest.raises(opengwasdb.ld.InputError, match=message):
        build_inputs(**overrides)


def test_k_zero_needs_no_pc_file_and_no_minimum() -> None:
    # With k = 0 no PC is read, so the minimum does not apply.
    assert build_inputs(k=0, pcs=None, pc_source=None).pcs is None


class FakeReader(StoreReader):
    """A StoreReader whose provenance is three facts; nothing else is read."""

    @property
    def provenance(self) -> Any:
        return {"t_far": 0.0274, "near_bp": 3_000_000, "far_bp": 8_000_000}


def test_the_cli_needs_a_command() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2


def test_the_cli_refuses_a_build_it_cannot_run(capsys: pytest.CaptureFixture[str]) -> None:
    # k = 10 with no PC File is refused before anything is read, with exit status 2.
    argv = [
        "build",
        "out",
        "--pgen",
        "x.pgen",
        "--keep",
        "k",
        "--panel-id",
        "p",
        "--r-scale",
        "511",
        "--t-partner",
        "0.01",
    ]
    status = cli.main([*argv, "--ancestry", "EUR", "--genotype-source", "g"])
    assert status == 2
    assert "PC File" in capsys.readouterr().err


def test_the_cli_reports_each_rule_and_fails_on_any_finding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing: dict[str, list[opengwasdb.ld.Finding]] = {"near.codes": [], "index.symmetric": []}
    assert cli.report(passing) == 0
    one = opengwasdb.ld.Finding("index.symmetric", "22:5", "one direction only")
    assert cli.report({**passing, "index.symmetric": [one]}) == 1
    out = capsys.readouterr().out
    assert "ok   near.codes" in out
    assert "FAIL index.symmetric" in out
    assert "22:5: one direction only" in out


BUILD_ARGS = ["build", "out", "--pgen", "x.pgen", "--keep", "k", "--pcs", "p.eigenvec"]
BUILD_ARGS += ["--pc-source", "s", "--panel-id", "p", "--ancestry", "EUR", "--genotype-source", "g"]
BUILD_ARGS += ["--r-scale", "511", "--t-partner", "0.01"]


def test_the_cli_s_defaults_are_build_inputs_own(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[opengwasdb.ld.BuildInputs] = []
    monkeypatch.setattr(cli, "build", lambda inputs, out, threads=1: seen.append(inputs) or {})
    assert cli.main(BUILD_ARGS) == 0
    expected = opengwasdb.ld.BuildInputs(
        pgen=(Path("x.pgen"),),
        keep=Path("k"),
        pcs=Path("p.eigenvec"),
        pc_source="s",
        panel_id="p",
        ancestry="EUR",
        genotype_source="g",
        r_scale=511,
        t_partner=0.01,
    )
    assert seen == [expected]


def test_the_cli_exits_2_on_an_input_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(inputs: object, out: object, threads: int = 1) -> None:
        raise opengwasdb.ld.InputError("chr22:1:A:G has a missing call")

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


# ------------------------------------------------------------------ the build's scale (ogld#16)


def test_a_build_with_no_scale_is_refused_naming_what_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Owner, 10 November 2026: a build takes its scale as an input, and there is no default it
    # could fall back on silently (ADR 0090).
    field = next(f for f in dataclasses.fields(opengwasdb.ld.BuildInputs) if f.name == "r_scale")
    assert field.default is dataclasses.MISSING
    given = {
        f.name: getattr(build_inputs(), f.name)
        for f in dataclasses.fields(opengwasdb.ld.BuildInputs)
    }
    del given["r_scale"]
    with pytest.raises(TypeError, match="r_scale"):
        opengwasdb.ld.BuildInputs(**given)
    with pytest.raises(
        opengwasdb.ld.InputError, match="r scale is a whole number from 1 to 32767, not None"
    ):
        build_inputs(r_scale=None)
    unscaled = BUILD_ARGS[: BUILD_ARGS.index("--r-scale")]
    with pytest.raises(SystemExit) as refused:
        cli.main(unscaled)
    assert refused.value.code == 2
    assert "Missing option '--r-scale'" in capsys.readouterr().err


def test_a_build_with_no_partner_threshold_is_refused_naming_what_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A build input with no default (ADR 0097, D3): each Source Panel's build names its own, and
    # the store refuses one above t_far once the panel's n and k give it (partner-rows §3).
    field = next(f for f in dataclasses.fields(opengwasdb.ld.BuildInputs) if f.name == "t_partner")
    assert field.default is dataclasses.MISSING
    given = {
        f.name: getattr(build_inputs(), f.name)
        for f in dataclasses.fields(opengwasdb.ld.BuildInputs)
    }
    del given["t_partner"]
    with pytest.raises(TypeError, match="t_partner"):
        opengwasdb.ld.BuildInputs(**given)
    unset = BUILD_ARGS[: BUILD_ARGS.index("--t-partner")]
    with pytest.raises(SystemExit) as refused:
        cli.main(unset)
    assert refused.value.code == 2
    assert "Missing option '--t-partner'" in capsys.readouterr().err
    with pytest.raises(opengwasdb.ld.InputError, match="t_partner is the r²"):
        opengwasdb.ld.BuildInputs(**{**given, "t_partner": float("nan")})


@pytest.mark.parametrize("scale", [0, -511, 32768, 511.5])
def test_a_build_s_scale_is_one_the_format_admits(scale: object) -> None:
    with pytest.raises(opengwasdb.ld.InputError, match=f"not {scale!r}"):
        build_inputs(r_scale=scale)


@pytest.mark.parametrize("scale", [511, 2047])
def test_a_build_writes_the_encoding_of_its_scale(scale: int) -> None:
    assert build_inputs(r_scale=scale).encoding().record() == {
        "r_dtype": "int16",
        "r_scale": scale,
        "r_reserved_code": -32768,
        "rounding": "half_to_even",
    }
