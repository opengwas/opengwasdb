"""Displaced liftover segments at the plan, CLI and resume seams (#99)."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_import_cli import FIXTURE, import_args

from opengwasdb.ld.build.importing import run
from opengwasdb.ld.build.importing.lift import Lifted, far_distance
from opengwasdb.ld.build.importing.run import ImportInputs, load_lifted
from opengwasdb.ld.cli import main
from opengwasdb.ld.format import NEAR_BP
from opengwasdb.ld.store import Variants


def literal_table() -> Lifted:
    source = [1, 500_000, 1_000_000, 11_000_100, 11_000_200, 12_000_000]
    position = np.array([1, 500_000, 1_000_000, 2_000_000, 2_000_100, 12_000_000])
    count = len(position)
    return Lifted(
        Variants(
            np.array(["1"] * count),
            position,
            np.array([f"1:{p}:A:C" for p in position]),
            np.array(["A"] * count),
            np.array(["C"] * count),
            np.full(count, 0.25),
        ),
        np.array([10, 11, 12, 25, 26, 15]),
        np.array(source),
        np.array([1, -1, 1, 1, -1, -1]),
        np.array([False, True, False, True, True, False]),
    )


def plan_inputs(tmp_path: Path, **options: int) -> ImportInputs:
    work = tmp_path / "work"
    work.mkdir()
    return ImportInputs(
        metadata=FIXTURE / "metadata.json",
        variants=FIXTURE / "variants.tsv",
        chain=FIXTURE / "identity.chain",
        fasta=FIXTURE / "reference.fa.gz",
        work=work,
        **options,
    )


def test_plan_saves_only_kept_rows_and_counts_the_displaced(tmp_path, monkeypatch) -> None:
    table = literal_table()
    assert far_distance(table.source_position, table.variants.position) < NEAR_BP
    monkeypatch.setattr(run, "lift_variants", lambda *_: (table, {"allele_mismatch": 3}))
    source = SimpleNamespace(
        rows=99, size=10, parts={(1, 1): "p11", (2, 2): "p22", (1, 2): "p12", (2, 1): "p21"}
    )
    rows = [SimpleNamespace(index=int(index)) for index in table.index]
    planned = run._plan_chromosome(plan_inputs(tmp_path), source, "1", rows)
    assert planned["parts"] == ["p11"]  # block 2 was needed only by the dropped rows
    saved = load_lifted(tmp_path / "work" / "1.lifted.npz")
    assert saved.index.tolist() == [10, 11, 12, 15]
    assert saved.sign.tolist() == [1, -1, 1, -1]
    assert saved.negative_strand.tolist() == [False, True, False, False]
    assert saved.source_position.tolist() == [1, 500_000, 1_000_000, 12_000_000]
    assert saved.variants.position.tolist() == [1, 500_000, 1_000_000, 12_000_000]
    assert planned["losses"] == {"allele_mismatch": 3, "liftover_displaced": 2}
    assert planned["far_bp"] >= 8_000_000


def test_plan_refuses_a_displaced_segment_over_the_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run, "lift_variants", lambda *_: (literal_table(), {}))
    inputs = plan_inputs(tmp_path, displaced_cap=1)
    with pytest.raises(ValueError, match="chromosome 1: displaced segment has 2 variants"):
        run._plan_chromosome(inputs, SimpleNamespace(rows=99), "1", [SimpleNamespace(index=0)])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--far-target-bp", "2000000"),
        ("--far-target-bp", "10000001"),
        ("--far-target-bp", "x"),
        ("--far-target-bp", "-8000000"),
        ("--displaced-cap", "-1"),
        ("--displaced-cap", "x"),
    ],
)
def test_out_of_range_options_are_refused_at_parse_time(tmp_path, capsys, option, value) -> None:
    with pytest.raises(SystemExit):
        main([*import_args(tmp_path / "a.store", tmp_path / "work"), option, value])
    refusal = capsys.readouterr().err
    assert f"Invalid value for '{option}'" in refusal


def test_options_reach_the_import_and_default_to_the_owner_ruling(tmp_path, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(run, "import_panukb", lambda inputs, out: seen.append(inputs) or {})
    base = import_args(tmp_path / "a.store", tmp_path / "work")
    assert main(base) == 0
    assert main([*base, "--far-target-bp", "10000000", "--displaced-cap", "10000"]) == 0
    assert main([*base, "--far-target-bp", "9000000", "--displaced-cap", "0"]) == 0
    assert [(i.far_target_bp, i.displaced_cap) for i in seen] == [
        (8_000_000, 10_000),
        (10_000_000, 10_000),
        (9_000_000, 0),
    ]


@pytest.mark.parametrize("change", [{"far_target_bp": 9_000_000}, {"displaced_cap": 5}])
def test_resume_with_a_different_target_or_cap_is_refused(tmp_path, change) -> None:
    inputs = plan_inputs(tmp_path)
    metadata = json.loads(inputs.metadata.read_text())
    out = tmp_path / "a.store"
    run._plan(inputs, out, metadata)
    run._plan(inputs, out, metadata)  # the same pins resume
    with pytest.raises(ValueError, match="pins different inputs"):
        run._plan(replace(inputs, **change), out, metadata)
