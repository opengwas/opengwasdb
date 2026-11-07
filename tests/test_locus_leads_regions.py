"""The region figures' data: the window rule, clump membership, and the refusal to write
counts that are not the committed ones (prototypes/locus_leads, #409)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from locus_leads_panel import write_panel
from prototype_modules import load_prototype

blocks, genetic_map, panel, profile_sweep, regions, run, sweep = (
    load_prototype("locus_leads", name)
    for name in ("blocks", "genetic_map", "panel", "profile_sweep", "regions", "run", "sweep"))

SIGNAL = ["0/0", "1/1", "0/0", "1/1"] * 10
OTHER = ["0/0", "0/0", "1/1", "1/1"] * 10  # r^2 = 0 with SIGNAL


def _map(assembly: str) -> Any:
    return genetic_map.LoadedMap(
        genetic_map.GeneticMap({"1": (np.array([1, 10_000_000]), np.array([0.0, 100.0]))}),
        {"assembly": assembly, "map": "test.tsv.gz", "sha256": "0" * 64})


def test_the_window_starts_at_an_index_variant_and_holds_the_most_of_them() -> None:
    position = np.array([1_000_000, 5_000_000, 6_000_000, 10_999_999, 11_000_000, 12_000_000,
                         30_000_000, 30_000_001, 30_000_002, 30_000_003, 30_000_004])
    is_index = np.array([True] * 6 + [False] * 5)
    chromosome = np.full(position.size, "1")
    # From 1 Mb, [1 Mb, 11 Mb) holds four (11,000,000 is outside: the window is half-open);
    # from 5 Mb it holds five. The five rows at 30 Mb are not index variants.
    assert np.sum(is_index & (position >= 1_000_000) & (position < 11_000_000)) == 4

    window = regions.densest_window(chromosome, position, is_index)

    assert window == regions.Window("1", 5_000_000, 5)


def test_a_tie_goes_to_the_first_window_in_genome_order() -> None:
    # Chromosome 10 comes first in the rows and first as text; chromosome 2 comes first in
    # the genome. Each holds two windows of two index variants.
    chromosome = np.array(["10", "10", "2", "2", "2", "2"])
    position = np.array([1_000, 2_000, 50_000_000, 50_000_100, 3_000, 4_000])
    is_index = np.ones(chromosome.size, dtype=bool)

    window = regions.densest_window(chromosome, position, is_index)

    assert window == regions.Window("2", 3_000, 2)


def _profile(records: list[tuple[str, str, int]], named: str) -> dict[str, Any]:
    return {"judgement": {"named_analysis": {"analysis_id": named}},
            "analyses": [{"analysis": {"analysis_id": a}, "selection": {"z_band": band},
                          "index_variants": n} for a, band, n in records]}


def test_one_analysis_per_other_z_band_is_drawn_from_those_with_enough_index_variants() -> None:
    records = [("N", ">=100", 27), ("x", "<38.5", 26), ("y", "<38.5", 27), ("z", "<38.5", 400),
               ("w", "38.5-100", 5), ("v", "38.5-100", 28)]

    picked = regions.choose(_profile(records, "N"))

    assert picked["named"] == "N" and picked["min_index_variants"] == 27
    assert [(d["z_band"], d["eligible"]) for d in picked["draws"]] == [
        ("38.5-100", ["v"]), ("<38.5", ["y", "z"])]
    assert picked["draws"][1]["chosen"] in ("y", "z")
    assert picked == regions.choose(_profile(records, "N")), "the draw is seeded"
    with pytest.raises(ValueError, match="no Analysis in z band 38.5-100"):
        regions.choose(_profile(records[:4] + [("w", "38.5-100", 5)], "N"))


def _tier(abs_z: dict[int, float]) -> Any:
    """Chromosome 1 rows at the given positions, A/G, in Store Variant Index order."""
    at = np.fromiter(abs_z, dtype="int64")
    return run.TierRows(np.full(at.size, "1"), at, np.char.add(np.char.add("1:", at.astype(str)),
                                                               ":A:G"),
                        np.full(at.size, "A"), np.full(at.size, "G"),
                        np.fromiter(abs_z.values(), dtype="float64"), np.arange(at.size))


def test_each_row_is_drawn_in_the_clump_plink2_puts_it_in_not_the_nearest(
        tmp_path: Path) -> None:
    # A and B carry one signal, D and E another, interleaved along the chromosome.
    layout = {100_000: (20.0, SIGNAL), 150_000: (9.0, OTHER), 200_000: (15.0, SIGNAL),
              250_000: (8.0, OTHER)}
    write_panel(tmp_path, "1", [(p, "A", "G") for p in layout], [g for _, g in layout.values()])
    tier = _tier({p: z for p, (z, _) in layout.items()})

    rebuilt = regions.rebuild(tier, _map("GRCh38"), panel.Panel(tmp_path), frozenset(),
                              work_dir=tmp_path / "work", threads=1)
    position = rebuilt.shared.rows.position
    assert position.tolist() == list(layout), "every row must be in the shared set"
    index = position[rebuilt.index_of == np.arange(position.size)]
    assert index.tolist() == [100_000, 150_000]
    # B (200 kb) is nearer D (150 kb) than A (100 kb): a nearest-index rule would misplace it.
    assert abs(200_000 - 150_000) < abs(200_000 - 100_000)

    record = regions.region(rebuilt, {}, blocks.Blocks("EUR", {}, {}))

    assert record["rows"] == {"position": list(layout), "abs_z": [20.0, 9.0, 15.0, 8.0],
                              "clump": [1, 2, 1, 2], "index_variant": [True, True, False, False]}
    assert record["clumps"] == [
        {"clump": 1, "index_position": 100_000, "index_abs_z": 20.0, "rows": 2,
         "start": 100_000, "end": 200_000},
        {"clump": 2, "index_position": 150_000, "index_abs_z": 9.0, "rows": 2,
         "start": 150_000, "end": 250_000}]


HM3 = frozenset({"1:100:A:G"})
BLOCKS = "chr\tstart\tend\tancestry\n1\t0\t200\tEUR\n1\t200\t10000000\tEUR\n"
SELECTION = {"seed": 1, "chosen": [{"analysis_id": "a2"}]}


def _committed(store: Path, panel_dir: Path, work: Path) -> dict[str, Any]:
    """#403's and #409's records of the conftest Store's Analysis a2, made by their own code,
    in files laid out as the committed ones are."""
    on_panel = panel.Panel(panel_dir)
    tier, at_5e4 = run.read_tier(store, "a2", 5e-4)
    reference, at_5e8 = run.read_tier(store, "a2", 5e-8)
    key = store.parent.name
    record_403 = sweep.analyse(reference, at_5e8, _map("GRCh37"), on_panel,
                               work_dir=work / "403", threads=1, spikes=(),
                               configs=sweep.grid_configs(), hm3=HM3) | {"store": key}
    loaded = blocks.read_blocks(work / "blocks.tsv", "EUR")
    record_409 = profile_sweep.score_profile(
        tier, at_5e4, reference, record_403, _map("GRCh37"), on_panel, loaded, HM3,
        store_id=key, work_dir=work / "409", threads=1)
    record_409["selection"] = {"z_band": "<38.5", "max_abs_z": 6.0}
    digest = hashlib.sha256(json.dumps(SELECTION, sort_keys=True).encode()).hexdigest()
    return {"sweep": {"selection": {"sha256": digest}, "genetic_map": {"sha256": "0" * 64},
                      "hm3": {"sha256": genetic_map.file_digest(work / "hm3.txt", "sha256")},
                      "counts": {"named_cm": {"d_cm": 2.5, "w_bp": 10_000_000}},
                      "analyses": [json.loads(json.dumps(record_403))]},
            "profile": {"judgement": {"named_analysis": {"analysis_id": "a2"}},
                        "analyses": [json.loads(json.dumps(record_409))]}}


def _write_regions(committed: dict[str, Any], store: Path, panel_dir: Path, work: Path,
                   output: Path) -> dict[str, Any]:
    paths = {"sweep": work / "eur_sweep.json", "profile": work / "eur_profile.json",
             "hm3": work / "hm3.txt", "blocks": work / "blocks.tsv"}
    paths["sweep"].write_text(json.dumps(committed["sweep"]))
    committed["profile"]["inputs"] = {
        "sweep_403": {"sha256": genetic_map.file_digest(paths["sweep"], "sha256")},
        "blocks": {"sha256": genetic_map.file_digest(paths["blocks"], "sha256")}}
    paths["profile"].write_text(json.dumps(committed["profile"]))
    given = regions.Inputs(store.parent.parent, _map("GRCh37"), panel.Panel(panel_dir),
                           blocks.read_blocks(paths["blocks"], "EUR"), HM3, work / "regions", 1)
    written: dict[str, Any] = regions.regions(paths, SELECTION, given, output)
    return written


@pytest.fixture
def conftest_store(dense_store_path: Path, tmp_path_factory: pytest.TempPathFactory,
                   ) -> dict[str, Any]:
    """The conftest Store, a GRCh37 panel holding its two variants in LD, and the records."""
    panel_dir = tmp_path_factory.mktemp("panel")
    write_panel(panel_dir, "1", [(100, "A", "G"), (300, "A", "G")], [SIGNAL, SIGNAL],
                assembly="GRCh37")
    work = tmp_path_factory.mktemp("work")
    (work / "hm3.txt").write_text("\n".join(sorted(HM3)) + "\n")
    (work / "blocks.tsv").write_text(BLOCKS)
    return {"store": dense_store_path, "panel": panel_dir, "work": work,
            "committed": _committed(dense_store_path, panel_dir, work)}


def test_records_that_match_are_written_with_the_region(conftest_store: dict[str, Any]) -> None:
    rows = conftest_store["committed"]["profile"]["analyses"][0]
    assert rows["shared_set"]["rows"] == 2 and rows["index_variants"] == 1, \
        "a2's two 5e-8 rows must be on the panel and in one clump"
    output = conftest_store["work"] / "out" / "eur_regions.json"

    written = _write_regions(conftest_store["committed"], conftest_store["store"],
                             conftest_store["panel"], conftest_store["work"], output)

    assert json.loads(output.read_text()) == written
    (only,) = written["regions"]
    assert only["rows"]["clump"] == [1, 1] and only["rows"]["index_variant"] == [True, False]
    agreed = {"position": 100, "clump": 1, "kind": "agreed"}
    # The rows straddle a block boundary, so one lead per block splits plink2's one clump.
    assert {m: only["leads"][m]["leads"] for m in only["leads"]} == {
        "bp_only": [agreed], "named_cm": [agreed], "prominence": [agreed],
        "ld_blocks": [agreed, {"position": 300, "clump": 1, "kind": "spurious_split"}]}
    assert only["blocks"] == [[0, 200], [200, 10_000_000]]


@pytest.mark.parametrize(("record", "change", "field"), [
    ("profile", lambda a: a.update(index_variants=a["index_variants"] + 1), "index_variants"),
    ("sweep", lambda a: next(c for c in a["configs"] if c["d_cm"] == "bp_only"
                             and c["w_bp"] == 10_000_000)["by_r2"]["0.01"].update(
                                 locus_leads=2), "locus_leads"),
    ("profile", lambda a: a["rules"][-1]["density"].update(leads_hm3=7), "leads_hm3"),
])
def test_a_count_that_is_not_the_committed_one_refuses_to_write(
        conftest_store: dict[str, Any], record: str, change: Any, field: str) -> None:
    committed = copy.deepcopy(conftest_store["committed"])
    change(committed[record]["analyses"][0])
    assert committed != conftest_store["committed"], "the change must reach the record"
    output = conftest_store["work"] / "out" / "eur_regions.json"

    with pytest.raises(regions.RegionMismatch, match=rf"field \S*{field} differs"):
        _write_regions(committed, conftest_store["store"], conftest_store["panel"],
                       conftest_store["work"], output)

    assert not output.exists()
