"""A BESD probe and a Column Map row resolved in one manifest (#281 with #282).

Both formats add their own key to a Resolution Record's fingerprint -- the map
`resolution_config.column_map`, a probe `resolution_config.source_probe_id` --
and each key must reach its own rows only, so an ordinary tabular record's
digest stays what it was before either existed.
"""

from __future__ import annotations

from pathlib import Path

from test_resolve_besd import (
    _EUR_MIX,
    _P1,
    _besd_row,
    _record,
    _resolve,
    _two_probe_file,
    _write_manifest,
    _write_ssf,
)


def test_a_besd_probe_and_a_mapped_row_resolve_in_one_manifest(tmp_path: Path) -> None:
    prefix = _two_probe_file(tmp_path)
    ssf = _write_ssf(tmp_path / "eur.tsv", _EUR_MIX, 1.0)
    tabular = ("sd", "estimated_from_source_maf", "500000", "GRCh38")
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", prefix, _P1),
        ("mapped", str(ssf), "opengwasdb.column-map", "", *tabular),
        ("plain", str(ssf), "opengwasdb.gwas-ssf", "", *tabular),
    ])

    records = _resolve(tmp_path, manifest, column_map="gwas-ssf")

    configs = {}
    for analysis_id in ("eqtl_p1", "mapped", "plain"):
        record = _record(records, analysis_id)
        assert record["status"] == "success", (analysis_id, record["error"])
        assert record["ancestry"]["assigned_ancestry"] == "EUR", analysis_id
        configs[analysis_id] = record["fingerprints"]["resolution_config"]
    assert configs["eqtl_p1"]["source_probe_id"] == _P1
    assert "column_map" not in configs["eqtl_p1"]
    assert configs["mapped"]["column_map"]["name"] == "gwas-ssf"
    assert "source_probe_id" not in configs["mapped"]
    assert "column_map" not in configs["plain"] and "source_probe_id" not in configs["plain"]
    mapped, plain = _record(records, "mapped"), _record(records, "plain")
    assert mapped["diagnostics"] == plain["diagnostics"], "one file, read two ways, one answer"
