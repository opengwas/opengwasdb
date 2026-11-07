"""The region figures' data: the window rule, clump membership, and the refusal to write
counts that are not the committed ones (prototypes/locus_leads, #409)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import SOURCE_HEADER, SOURCE_ROWS
from locus_leads_panel import write_panel
from prototype_modules import load_prototype

from opengwasdb.build.observed import build_dense_observed_from_sources

blocks, genetic_map, panel, profile_sweep, regions, rule, run, sweep = (
    load_prototype("locus_leads", name)
    for name in ("blocks", "genetic_map", "panel", "profile_sweep", "regions", "rule", "run",
                 "sweep"))

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


def _profile(records: list[tuple[str, str, int]], named: str,
             store: str = "S") -> dict[str, Any]:
    """#409's record of Analyses given as (store/id or id, z band, index variants)."""
    def key(name: str) -> dict[str, Any]:
        found, _, analysis = name.rpartition("/")
        return {"store": found or store, "analysis": {"analysis_id": analysis}}
    return {"judgement": {"named_analysis": {"analysis_id": named}},
            "analyses": [key(a) | {"selection": {"z_band": band}, "index_variants": n}
                         for a, band, n in records]}


def _ids(entries: list[dict[str, str]]) -> list[str]:
    return [f"{e['store']}/{e['analysis_id']}" for e in entries]


def test_one_analysis_per_other_z_band_is_drawn_from_those_with_enough_index_variants() -> None:
    records = [("N", ">=100", 27), ("x", "<38.5", 26), ("y", "<38.5", 27), ("z", "<38.5", 400),
               ("w", "38.5-100", 5), ("v", "38.5-100", 28)]

    picked = regions.choose(_profile(records, "N"))

    assert picked["named"] == {"store": "S", "analysis_id": "N"}
    assert picked["min_index_variants"] == 27
    assert [(d["z_band"], _ids(d["eligible"])) for d in picked["draws"]] == [
        ("38.5-100", ["S/v"]), ("<38.5", ["S/y", "S/z"])]
    assert _ids([picked["draws"][1]["chosen"]])[0] in ("S/y", "S/z")
    assert picked == regions.choose(_profile(records, "N")), "the draw is seeded"
    with pytest.raises(ValueError, match="no Analysis in z band 38.5-100"):
        regions.choose(_profile(records[:4] + [("w", "38.5-100", 5)], "N"))


def test_analyses_are_told_apart_by_store_and_a_repeated_pair_is_refused() -> None:
    # An analysis_id is unique only within its Store. The eligible are listed by analysis_id,
    # then Store, the order the committed draw was made in.
    records = [("N", ">=100", 27), ("A/y", "<38.5", 30), ("B/y", "<38.5", 40),
               ("B/a", "<38.5", 50)]

    picked = regions.choose(_profile(records, "N"))

    assert _ids(picked["draws"][0]["eligible"]) == ["B/a", "A/y", "B/y"]
    with pytest.raises(ValueError, match="A/y appears twice"):
        regions.choose(_profile([*records, ("A/y", "<38.5", 31)], "N"))


def _tier(abs_z: dict[int, float]) -> Any:
    """Chromosome 1 rows at the given positions, A/G, in Store Variant Index order."""
    at = np.fromiter(abs_z, dtype="int64")
    return run.TierRows(np.full(at.size, "1"), at, np.char.add(np.char.add("1:", at.astype(str)),
                                                               ":A:G"),
                        np.full(at.size, "A"), np.full(at.size, "G"),
                        np.fromiter(abs_z.values(), dtype="float64"), np.arange(at.size))


# Two signals interleaved along 1 Mb. The weaker one's index variant comes first by position,
# and the row at 700 kb is in its clump though the stronger index variant is nearer.
INTERLEAVED = {100_000: (9.0, OTHER), 500_000: (20.0, SIGNAL), 700_000: (8.0, OTHER),
               1_100_000: (15.0, SIGNAL)}


@pytest.fixture(scope="module")
def interleaved(tmp_path_factory: pytest.TempPathFactory) -> Any:
    directory = tmp_path_factory.mktemp("interleaved")
    write_panel(directory, "1", [(p, "A", "G") for p in INTERLEAVED],
                [g for _, g in INTERLEAVED.values()])
    tier = _tier({p: z for p, (z, _) in INTERLEAVED.items()})
    rebuilt = regions.rebuild(tier, _map("GRCh38"), panel.Panel(directory), frozenset(),
                              work_dir=directory / "work", threads=1)
    assert rebuilt.shared.rows.position.tolist() == list(INTERLEAVED), \
        "every row must be in the shared set"
    return rebuilt


def _index_positions(rebuilt: Any) -> list[int]:
    rows = rebuilt.shared.rows
    return rows.position[rebuilt.index_of == np.arange(rows.position.size)].tolist()


def test_each_row_is_drawn_in_the_clump_plink2_puts_it_in_numbered_by_strength(
        interleaved: Any) -> None:
    index = _index_positions(interleaved)
    assert index == [100_000, 500_000], "two clumps, the weaker one first by position"
    assert INTERLEAVED[100_000][0] < INTERLEAVED[500_000][0]
    # 700 kb is nearer the 500 kb index variant: a nearest-index rule would misplace it.
    assert abs(700_000 - 500_000) < abs(700_000 - 100_000)

    record = regions.region(interleaved, {}, blocks.Blocks("EUR", {}, {}))

    assert record["rows"] == {"position": list(INTERLEAVED), "abs_z": [9.0, 20.0, 8.0, 15.0],
                              "clump": [2, 1, 2, 1], "index_variant": [True, True, False, False]}
    assert record["clumps"] == [
        {"clump": 1, "index_position": 500_000, "index_abs_z": 20.0, "rows": 2,
         "start": 500_000, "end": 1_100_000},
        {"clump": 2, "index_position": 100_000, "index_abs_z": 9.0, "rows": 2,
         "start": 100_000, "end": 700_000}]
    # Every index variant behind the window choice, so the JSON alone can repeat it.
    assert record["genome_index_variants"] == {"chromosome": ["1", "1"],
                                               "position": [100_000, 500_000]}


def test_the_view_pads_its_rows_by_4_percent_at_least_20_kb_and_not_below_0(
        interleaved: Any) -> None:
    # The rows span 1 Mb, so 4% (40 kb) is more than the 20 kb floor.
    record = regions.region(interleaved, {}, blocks.Blocks("EUR", {}, {}))

    assert record["view"] == {"chromosome": "1", "start": 60_000, "end": 1_140_000}
    assert regions.view_bounds(100_000, 250_000) == (80_000, 270_000)
    assert regions.view_bounds(5_000, 10_000) == (0, 30_000)


def test_a_profile_bin_without_a_5e4_row_is_null_not_zero(interleaved: Any) -> None:
    record = regions.region(interleaved, {}, blocks.Blocks("EUR", {}, {}))

    profile = record["profile"]
    # The view's 50 kb bins run from bin 1 (60 kb) to bin 22 (1,140 kb); four hold a row.
    expected: list[float | None] = [None] * 22
    for position, (abs_z, _) in INTERLEAVED.items():
        expected[position // 50_000 - 1] = abs_z
    assert expected.count(None) == 18
    assert profile == {"bin_bp": 50_000, "first_bin": 1, "max_abs_z": expected}


def test_a_method_s_leads_are_agreed_or_split_and_its_merged_index_variants_lost(
        interleaved: Any) -> None:
    rows = interleaved.shared.rows

    def leads_within(w_bp: int) -> np.ndarray:
        found: np.ndarray = rule.locus_leads(rows.chromosome, rows.position, interleaved.shared.cm,
                                             rows.abs_z, rows.variant_index,
                                             d_cm=float("inf"), w_bp=w_bp)
        return found

    record = regions.region(interleaved, {"merged": leads_within(10_000_000),
                                          "apart": leads_within(0)},
                            blocks.Blocks("EUR", {}, {}))

    assert record["leads"]["merged"] == {
        "leads": [{"position": 500_000, "clump": 1, "kind": "agreed"}], "lost": 1}
    assert record["leads"]["apart"] == {"leads": [
        {"position": 100_000, "clump": 2, "kind": "agreed"},
        {"position": 500_000, "clump": 1, "kind": "agreed"},
        {"position": 700_000, "clump": 2, "kind": "spurious_split"},
        {"position": 1_100_000, "clump": 1, "kind": "spurious_split"}], "lost": 0}


def test_5e8_rows_left_out_of_the_shared_set_are_counted_with_their_reasons(
        tmp_path: Path) -> None:
    write_panel(tmp_path, "1", [(100_000, "A", "G"), (300_000, "A", "G"), (500_000, "A", "G")],
                [SIGNAL, SIGNAL, OTHER])
    tier = _tier({100_000: 20.0, 300_000: 12.0, 400_000: 9.0, 500_000: 8.0, 5_000_000: 30.0})
    # 300 kb carries C/T against the panel's A/G; 400 kb and 5 Mb are not on the panel.
    tier = dataclasses.replace(tier, allele_1=np.array(["A", "C", "A", "A", "A"]),
                               allele_2=np.array(["G", "T", "G", "G", "G"]))

    rebuilt = regions.rebuild(tier, _map("GRCh38"), panel.Panel(tmp_path), frozenset(),
                              work_dir=tmp_path / "work", threads=1)
    assert rebuilt.accounting["partition"] | {"shared_set": 0} == {
        "shared_set": 0, "matched_no_cm": 0, "off_panel_chromosome": 0,
        "absent_at_position": 2, "allele_mismatch": 1, "ambiguous": 0}

    record = regions.region(rebuilt, {}, blocks.Blocks("EUR", {}, {}))

    assert record["view"] == {"chromosome": "1", "start": 80_000, "end": 520_000}
    # 5 Mb is out of the view, so its |z| of 30 is not the largest left out.
    assert record["not_drawn"] == {"rows": 2, "max_abs_z": 12.0, "reasons": {
        "matched_no_cm": 0, "off_panel_chromosome": 0, "absent_at_position": 1,
        "allele_mismatch": 1, "ambiguous": 0}}


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


def _write_regions(committed: dict[str, Any], built: dict[str, Any],
                   out: Path) -> dict[str, Any]:
    """`regions` on the conftest Store against `committed`, writing under `out`."""
    committed = copy.deepcopy(committed)
    paths = {"sweep": out / "eur_sweep.json", "profile": out / "eur_profile.json",
             "hm3": built["work"] / "hm3.txt", "blocks": built["work"] / "blocks.tsv"}
    paths["sweep"].write_text(json.dumps(committed["sweep"]))
    committed["profile"]["inputs"] = {
        "sweep_403": {"sha256": genetic_map.file_digest(paths["sweep"], "sha256")},
        "blocks": {"sha256": genetic_map.file_digest(paths["blocks"], "sha256")}}
    paths["profile"].write_text(json.dumps(committed["profile"]))
    given = regions.Inputs(built["store"].parent.parent, _map("GRCh37"),
                           panel.Panel(built["panel"]), blocks.read_blocks(paths["blocks"], "EUR"),
                           HM3, out / "regions", 1)
    written: dict[str, Any] = regions.regions(paths, SELECTION, given, out / "eur_regions.json")
    return written


@pytest.fixture(scope="module")
def conftest_store(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The conftest Store, built once from its source rows, a GRCh37 panel holding its two
    variants in LD, and #403's and #409's records of it."""
    root = tmp_path_factory.mktemp("stores")
    source = root / "associations.tsv"
    source.write_text("\n".join([SOURCE_HEADER, *SOURCE_ROWS, ""]), encoding="utf-8")
    store = root / "fixture" / "store.opengwasdb"
    build_dense_observed_from_sources([source], store, store_id="fixture-store",
                                      release_id="observed-v1", reference_assembly="GRCh37")
    panel_dir = tmp_path_factory.mktemp("panel")
    write_panel(panel_dir, "1", [(100, "A", "G"), (300, "A", "G")], [SIGNAL, SIGNAL],
                assembly="GRCh37")
    work = tmp_path_factory.mktemp("work")
    (work / "hm3.txt").write_text("\n".join(sorted(HM3)) + "\n")
    (work / "blocks.tsv").write_text(BLOCKS)
    return {"store": store, "panel": panel_dir, "work": work,
            "committed": _committed(store, panel_dir, work)}


def test_records_that_match_are_written_with_the_region(conftest_store: dict[str, Any],
                                                        tmp_path: Path) -> None:
    rows = conftest_store["committed"]["profile"]["analyses"][0]
    assert rows["shared_set"]["rows"] == 2 and rows["index_variants"] == 1, \
        "a2's two 5e-8 rows must be on the panel and in one clump"

    written = _write_regions(conftest_store["committed"], conftest_store, tmp_path)

    assert json.loads((tmp_path / "eur_regions.json").read_text()) == written
    (only,) = written["regions"]
    assert only["rows"]["clump"] == [1, 1] and only["rows"]["index_variant"] == [True, False]
    agreed = {"position": 100, "clump": 1, "kind": "agreed"}
    # The rows straddle a block boundary, so one lead per block splits plink2's one clump.
    assert {m: only["leads"][m]["leads"] for m in only["leads"]} == {
        "bp_only": [agreed], "named_cm": [agreed], "prominence": [agreed],
        "ld_blocks": [agreed, {"position": 300, "clump": 1, "kind": "spurious_split"}]}
    assert only["blocks"] == [[0, 200], [200, 10_000_000]]
    assert only["not_drawn"] == {"rows": 0, "max_abs_z": None, "reasons": {
        reason: 0 for reason in ("matched_no_cm", "off_panel_chromosome", "absent_at_position",
                                 "allele_mismatch", "ambiguous")}}


@pytest.mark.parametrize(("record", "change", "field"), [
    ("profile", lambda a: a.update(index_variants=a["index_variants"] + 1), "index_variants"),
    ("sweep", lambda a: next(c for c in a["configs"] if c["d_cm"] == "bp_only"
                             and c["w_bp"] == 10_000_000)["by_r2"]["0.01"].update(
                                 locus_leads=2), "locus_leads"),
    ("profile", lambda a: a["rules"][-1]["density"].update(leads_hm3=7), "leads_hm3"),
])
def test_a_count_that_is_not_the_committed_one_refuses_to_write(
        conftest_store: dict[str, Any], tmp_path: Path, record: str, change: Any,
        field: str) -> None:
    committed = copy.deepcopy(conftest_store["committed"])
    change(committed[record]["analyses"][0])
    assert committed != conftest_store["committed"], "the change must reach the record"

    with pytest.raises(regions.RegionMismatch, match=rf"field \S*{field} differs"):
        _write_regions(committed, conftest_store, tmp_path)

    assert not (tmp_path / "eur_regions.json").exists()
