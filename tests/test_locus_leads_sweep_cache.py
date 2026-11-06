"""A cached Analysis record is reused only for the inputs that made it (#403 fix round 1)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import locus_leads_panel
import pytest
from numpy import array
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
pairs = load_prototype("locus_leads", "pairs")
panel = load_prototype("locus_leads", "panel")
sweep = load_prototype("locus_leads", "sweep")

CODE = {"sha256": "c" * 64, "commit": "abc123"}


def _map(sha256: str) -> Any:
    points = {"1": (array([1, 10]), array([0.0, 1.0]))}
    return genetic_map.LoadedMap(genetic_map.GeneticMap(points),
                                 {"assembly": "GRCh38", "sha256": sha256})


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Any]:
    store = tmp_path / "OGS-1" / "store.opengwasdb"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text('{"store_id": "OGS-1", "release_id": "r1"}')
    directory = tmp_path / "panel"
    directory.mkdir()
    locus_leads_panel.write_panel(directory, "1", [(100, "A", "G"), (200, "A", "G")],
                                  [["0/0", "1/1"] * 4, ["0/0", "1/1"] * 4])
    hm3 = tmp_path / "hm3.txt"
    hm3.write_text("1:100:A:G\n")
    return {"chosen": {"store": "OGS-1", "analysis_id": "a"}, "stores_dir": tmp_path,
            "panel": panel.Panel(directory), "hm3": hm3, "map": _map("a" * 64)}


def _fingerprint(inputs: dict[str, Any], **changes: Any) -> dict[str, Any]:
    settings = {**inputs, **changes}
    fingerprint: dict[str, Any] = sweep.input_fingerprint(
        settings["chosen"], settings["stores_dir"], settings["map"],
        sweep.panel_content_digests(settings["panel"]), settings["hm3"], CODE,
        settings.get("spikes", sweep.EUR_SPIKES))
    return fingerprint


def test_the_fingerprint_names_the_inputs_the_record_depends_on(inputs: dict[str, Any]) -> None:
    fingerprint = _fingerprint(inputs)
    assert fingerprint["analysis_id"] == "a" and fingerprint["genetic_map_sha256"] == "a" * 64
    assert fingerprint["code_sha256"] == CODE["sha256"]
    assert fingerprint["manifest_sha256"] == hashlib.sha256(
        (inputs["stores_dir"] / "OGS-1" / "store.opengwasdb" / "manifest.json").read_bytes()
    ).hexdigest()
    assert fingerprint["tier"] == 5e-8 and fingerprint["r2"] == [0.01, 0.001]
    assert len(fingerprint["grid"]) == 24
    assert fingerprint["hm3_sha256"] == hashlib.sha256(inputs["hm3"].read_bytes()).hexdigest()
    assert fingerprint == _fingerprint(inputs)  # a function of the inputs, nothing else


def test_a_changed_map_a_changed_manifest_or_changed_code_changes_the_fingerprint(
    inputs: dict[str, Any],
) -> None:
    base = _fingerprint(inputs)
    assert _fingerprint(inputs, map=_map("b" * 64)) != base
    (inputs["stores_dir"] / "OGS-1" / "store.opengwasdb" / "manifest.json").write_text("{}")
    assert _fingerprint(inputs)["manifest_sha256"] != base["manifest_sha256"]
    inputs["hm3"].write_text("1:200:A:G\n")
    assert _fingerprint(inputs)["hm3_sha256"] != base["hm3_sha256"]


def test_the_panel_is_fingerprinted_by_the_content_of_each_file_not_its_size(
    inputs: dict[str, Any],
) -> None:
    base = _fingerprint(inputs)
    directory = inputs["panel"].directory
    for suffix in (".pgen", ".psam"):
        target = directory / f"chr1{suffix}"
        original = target.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 0xFF  # one byte flipped: the size is unchanged
        target.write_bytes(bytes(changed))
        try:
            assert target.stat().st_size == len(original)
            assert _fingerprint(inputs)["panel"] != base["panel"], suffix
        finally:
            target.write_bytes(original)
    assert _fingerprint(inputs) == base


def test_a_panel_with_other_variants_has_another_pvar_digest(
    inputs: dict[str, Any], tmp_path: Path,
) -> None:
    other = tmp_path / "other_panel"
    other.mkdir()
    locus_leads_panel.write_panel(other, "1", [(100, "A", "C"), (200, "A", "G")],
                                  [["0/0", "1/1"] * 4, ["0/0", "1/1"] * 4])
    changed = sweep.panel_content_digests(panel.Panel(other))["sha256"]["1"]
    assert changed[".pvar.zst"] != sweep.panel_content_digests(inputs["panel"])["sha256"]["1"][
        ".pvar.zst"]


def test_the_fingerprint_holds_the_spikes_the_computation_used(inputs: dict[str, Any]) -> None:
    assert _fingerprint(inputs, spikes=())["spikes"] == []
    assert _fingerprint(inputs)["spikes"] == [["7", 131_782_269, "TSI"], ["1", 235_056_161, "GBR"]]
    assert _fingerprint(inputs, spikes=())["spikes"] != _fingerprint(inputs)["spikes"]


def test_a_cached_record_is_reused_for_the_same_fingerprint(tmp_path: Path) -> None:
    cache = tmp_path / "results" / "a.json"
    calls: list[int] = []

    def compute() -> dict[str, Any]:
        calls.append(1)
        return {"value": 7}

    fingerprint = {"genetic_map_sha256": "a" * 64}
    assert sweep.cached_record(cache, fingerprint, compute) == {"value": 7}
    assert sweep.cached_record(cache, fingerprint, compute) == {"value": 7}
    assert len(calls) == 1


def test_a_record_made_with_another_map_is_refused_not_relabelled(tmp_path: Path) -> None:
    cache = tmp_path / "a.json"
    sweep.cached_record(cache, {"genetic_map_sha256": "a" * 64, "tier": 5e-8}, lambda: {"v": 1})
    with pytest.raises(sweep.CacheMismatch, match="genetic_map_sha256") as error:
        sweep.cached_record(cache, {"genetic_map_sha256": "b" * 64, "tier": 5e-8},
                            lambda: {"v": 2})
    assert "tier" not in str(error.value)  # names what differs, not everything
    assert json.loads(cache.read_text())["record"] == {"v": 1}  # the old record is left alone


def test_a_record_with_no_fingerprint_is_refused(tmp_path: Path) -> None:
    cache = tmp_path / "a.json"
    cache.write_text(json.dumps({"locus_leads": 1}))
    with pytest.raises(sweep.CacheMismatch, match="no fingerprint"):
        sweep.cached_record(cache, {"tier": 5e-8}, lambda: {"v": 1})


def test_the_pair_records_are_written_deterministically_with_the_digest_of_the_file(
    tmp_path: Path,
) -> None:
    rows = [{"analysis": "a", "store": "S", "d_cm": 0.1, "w_bp": 2_500_000, "kind": "lost_signal",
             "alid": "1:1:A:G", "stronger_alid": "1:2:A:G", "abs_z": 9.5, "stronger_abs_z": 11.0,
             "r2": None, "distance_bp": 1, "distance_cm": 0.0001},
            {"analysis": "a", "store": "S", "d_cm": "bp_only", "w_bp": 1_000_000,
             "kind": "spurious_split", "alid": "1:3:A:G", "stronger_alid": "1:2:A:G",
             "abs_z": 8.0, "stronger_abs_z": 11.0, "r2": 0.25, "distance_bp": 2,
             "distance_cm": 0.5}]
    first, second = tmp_path / "one.tsv.gz", tmp_path / "two.tsv.gz"
    described = pairs.write_pair_records(rows, first)
    pairs.write_pair_records(rows, second)
    assert first.read_bytes() == second.read_bytes()
    assert described["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert described["rows"] == 2
    lines = gzip.decompress(first.read_bytes()).decode().splitlines()
    assert lines[0].split("\t") == described["columns"]
    assert lines[1].split("\t")[described["columns"].index("r2")] == "NA"  # undefined, not 0
    assert lines[2].split("\t")[described["columns"].index("d_cm")] == "bp_only"
    assert lines[2].split("\t")[described["columns"].index("r2")] == "0.25"


def test_a_sweep_rerun_with_another_map_is_refused_rather_than_relabelled(
    inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_read_tier(store: Path, analysis_id: str, threshold: float) -> tuple[Any, Any]:
        return None, None

    def fake_analyse(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("analyse")
        return {"pair_records": [], "chance_ld": {}, "configs": []}

    monkeypatch.setattr(sweep, "read_tier", fake_read_tier)
    monkeypatch.setattr(sweep, "analyse", fake_analyse)
    monkeypatch.setattr(sweep, "evaluate", lambda analyses: {})
    monkeypatch.setattr(sweep, "evaluate_counts", lambda analyses: {})
    monkeypatch.setattr(sweep, "pool_chance", lambda records: {})
    selection = {"rule": {"seed": 1}, "chosen": [
        {"store": "OGS-1", "analysis_id": "a", "observed_rows": 10, "max_abs_z": 9.0}]}
    work = tmp_path / "work"
    options = {"hm3_path": inputs["hm3"], "pairs_path": tmp_path / "pairs.tsv.gz", "threads": 1,
               "code": CODE, "spike_sizes": lambda spikes, loaded: []}
    panel_dir = inputs["panel"].directory
    sweep.sweep(selection, inputs["stores_dir"], inputs["map"], panel_dir, work, **options)
    sweep.sweep(selection, inputs["stores_dir"], inputs["map"], panel_dir, work, **options)
    assert calls == ["analyse"]  # the second run read the cache
    with pytest.raises(sweep.CacheMismatch, match="genetic_map_sha256"):
        sweep.sweep(selection, inputs["stores_dir"], _map("b" * 64), panel_dir, work, **options)
    assert calls == ["analyse"]
    with pytest.raises(sweep.CacheMismatch, match="spikes"):
        sweep.sweep(selection, inputs["stores_dir"], inputs["map"], panel_dir, work,
                    spikes=(), **options)
    assert calls == ["analyse"]


def test_the_panel_is_digested_once_per_sweep_not_once_per_analysis(
    inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    digested: list[int] = []
    real_digests = sweep.panel_content_digests
    def counting(panel: Any) -> dict[str, Any]:
        digested.append(1)
        result: dict[str, Any] = real_digests(panel)
        return result

    monkeypatch.setattr(sweep, "panel_content_digests", counting)
    monkeypatch.setattr(sweep, "read_tier", lambda *a: (None, None))
    monkeypatch.setattr(sweep, "analyse", lambda *a, **k: {"pair_records": [], "chance_ld": {},
                                                           "configs": []})
    for name in ("evaluate", "evaluate_counts", "pool_chance"):
        monkeypatch.setattr(sweep, name, lambda x: {})
    chosen = [{"store": "OGS-1", "analysis_id": name, "observed_rows": 10, "max_abs_z": 9.0}
              for name in ("a", "b", "c")]
    sweep.sweep({"rule": {"seed": 1}, "chosen": chosen}, inputs["stores_dir"], inputs["map"],
                inputs["panel"].directory, tmp_path / "work", hm3_path=inputs["hm3"],
                pairs_path=tmp_path / "pairs.tsv.gz", threads=1, code=CODE,
                spike_sizes=lambda spikes, loaded: [])
    assert len(digested) == 1
