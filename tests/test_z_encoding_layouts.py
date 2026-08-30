"""Fixed-point `z` end to end, on all three Primary Storage Layouts (#114).

The acceptance criteria of ADR 0037 §1 are about stores, not arithmetic: a
z-score put in by a builder has to come back out of a query to within 0.001,
including the ones too large for the plane to hold, and the store has to say so
in its manifest and validate against it.

`ukb-b`'s largest association is |z| = 137.5 (HERC2/OCA2), so it is the value
used throughout: under `float16` its p-value was wrong by a factor of 5,400 in
the worst case, and an earlier draft of this issue would have failed the build
outright.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.encoding import (
    Z_OVERFLOW,
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
    DenseZPlane,
    StoreCodec,
    StoreEncoding,
    UnsupportedEncoding,
)
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store
from opengwasdb.stats import log10_p_two_sided
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, open_store
from opengwasdb.validation import validate_store

# The FADS1/FADS2 pilot hit, the largest |z| in `ukb-b`, one just inside the
# representable range, and one ordinary value.
BIG_Z = 137.5
FADS_Z = 47.8
NEAR_EDGE_Z = 31.5
ORDINARY_Z = 2.0

ROUND_TRIP_TOL = 0.001


def _p_ratio(stored: float, exact: float) -> float:
    """How far apart two z-scores' two-sided p-values are, as a ratio."""
    log_stored = log10_p_two_sided(np.array([stored]))[0]
    log_exact = log10_p_two_sided(np.array([exact]))[0]
    return float(10 ** abs(log_stored - log_exact))


# ── Dense ───────────────────────────────────────────────────────────────────

_SOURCE_HEADER = "\t".join([
    "analysis_id", "phenotype_id", "phenotype_label", "analysis_label",
    "chromosome", "position", "effect_allele", "other_allele",
    "z", "se", "rsid", "stored_effect_scale",
])


@pytest.fixture
def dense_store(tmp_path: Path) -> Path:
    rows = [
        f"a1\tp1\tHeight\tHeight primary\t1\t100\tA\tG\t{ORDINARY_Z}\t0.1\trs1\tsd",
        f"a1\tp1\tHeight\tHeight primary\t1\t200\tA\tG\t{BIG_Z}\t0.0015\trs2\tsd",
        f"a1\tp1\tHeight\tHeight primary\t1\t300\tA\tG\t{-FADS_Z}\t0.01\trs3\tsd",
        f"a1\tp1\tHeight\tHeight primary\t1\t400\tA\tG\t{NEAR_EDGE_Z}\t0.02\trs4\tsd",
        # a2 leaves 1:200 and 1:300 empty, so the grid carries missing cells.
        f"a2\tp2\tDisease\tDisease primary\t1\t100\tA\tG\t{-ORDINARY_Z}\t0.2\trs1\tlog_or",
        f"a2\tp2\tDisease\tDisease primary\t1\t400\tA\tG\t{-BIG_Z}\t0.003\trs4\tlog_or",
    ]
    source = tmp_path / "associations.tsv"
    source.write_text(_SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "dense.opengwasdb"
    build_dense_observed_from_sources(
        [source], out, store_id="enc", release_id="v1", reference_assembly="GRCh38"
    )
    return out


def _dense_by_position(store: Path) -> dict[tuple[int, str], float]:
    query = query_store(store)
    try:
        variants = query.variants_table()
        analyses = query.analyses_table()
        result = query.analysis("a1")
        other = query.analysis("a2")
    finally:
        query.close()
    out: dict[tuple[int, str], float] = {}
    for res in (result, other):
        for vi, ai, z in zip(res["variant_index"], res["analysis_index"], res["z"], strict=True):
            key = (variants[int(vi)]["position"], analyses[int(ai)]["analysis_id"])
            out[key] = float(z)
    return out


def test_dense_z_round_trips_including_out_of_range_values(dense_store):
    z_by_cell = _dense_by_position(dense_store)
    assert z_by_cell[(100, "a1")] == pytest.approx(ORDINARY_Z, abs=ROUND_TRIP_TOL)
    assert z_by_cell[(200, "a1")] == pytest.approx(BIG_Z, abs=ROUND_TRIP_TOL)
    assert z_by_cell[(300, "a1")] == pytest.approx(-FADS_Z, abs=ROUND_TRIP_TOL)
    assert z_by_cell[(400, "a1")] == pytest.approx(NEAR_EDGE_Z, abs=ROUND_TRIP_TOL)
    assert z_by_cell[(400, "a2")] == pytest.approx(-BIG_Z, abs=ROUND_TRIP_TOL)


def test_dense_p_value_at_the_largest_z_is_accurate_to_two_percent(dense_store):
    """float16 got this cell's p wrong by up to 5,400x (ADR 0037)."""
    stored = _dense_by_position(dense_store)[(200, "a1")]
    assert _p_ratio(stored, BIG_Z) < 1.02


def test_dense_missing_cells_still_read_as_absent(dense_store):
    """An integer plane has no NaN, so "missing" is a reserved code -- and a
    query must not return it as a z-score of -32 (spec §15)."""
    z_by_cell = _dense_by_position(dense_store)
    assert (200, "a2") not in z_by_cell
    assert (300, "a2") not in z_by_cell


def test_dense_declares_its_encoding_and_validates_against_it(dense_store):
    manifest = json.loads((dense_store / "manifest.json").read_text())
    assert manifest["format_version"] == CURRENT_FORMAT_VERSION
    assert manifest["encoding"]["z"] == {"kind": "int16_fixed", "scale": 1024}
    assert manifest["encoding"]["se"] == {"kind": "float16"}
    assert validate_store(dense_store).ok


def test_dense_holds_only_the_out_of_range_cells_in_the_overflow_table(dense_store):
    root = open_store(dense_store).arrays(mode="r")
    raw = np.asarray(root["z"][:])
    assert str(raw.dtype) == "int16"
    expected = np.flatnonzero(raw == Z_OVERFLOW)
    # |z| of 137.5 and 47.8 are outside ±32; 2.0 and 31.5 are not.
    assert len(expected) == 3
    assert np.array_equal(np.asarray(root[Z_OVERFLOW_INDEX][:]), expected)


def test_dense_top_hit_index_agrees_with_the_stored_plane(dense_store):
    """The index holds decoded float32 z (ADR 0037), and it must equal what a
    query reads back -- including for the cells held in the overflow table."""
    opened = open_store(dense_store)
    root = opened.arrays(mode="r")
    plane = DenseZPlane.open(root, opened.manifest.encoding)
    for key in root["top_hits"]:
        group = root["top_hits"][key]
        rows = group["variant_index"][:].astype(np.int64)
        cols = group["analysis_index"][:].astype(np.int64)
        np.testing.assert_array_equal(group["z"][:].astype("float32"), plane.points(rows, cols))


def test_a_z_plane_that_contradicts_the_manifest_is_rejected(dense_store):
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["z"] = {"kind": "float16"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(dense_store)
    assert not result.ok
    assert any("dtype int16" in error for error in result.errors), result.errors


def test_an_overflow_table_that_lost_a_cell_is_rejected(dense_store):
    """The lost value would be the *largest* |z| in the store -- the one thing
    that must never read as something plausible instead."""
    root = zarr.open_group(str(dense_store / "data.zarr"), mode="a")
    kept_index = np.asarray(root[Z_OVERFLOW_INDEX][:])[1:]
    kept_value = np.asarray(root[Z_OVERFLOW_VALUE][:])[1:]
    for name, data, dtype in (
        (Z_OVERFLOW_INDEX, kept_index, "int64"),
        (Z_OVERFLOW_VALUE, kept_value, "float32"),
    ):
        del root[name]
        root.create_dataset(name, data=data, chunks=(max(1, len(data)),), dtype=dtype)

    result = validate_store(dense_store)
    assert not result.ok
    assert any("no entry in the overflow table" in error for error in result.errors), result.errors


def test_a_1_0_release_with_no_encoding_block_is_refused(dense_store):
    """The dangerous direction: falling back to the legacy plan here would
    decode the `int16` plane as `float16` and return z-scores a thousand times
    too large, which is a plausible number and therefore the worst outcome."""
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["encoding"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsupportedEncoding, match="no `encoding` block"):
        open_store(dense_store)
    result = validate_store(dense_store)
    assert not result.ok


def test_reading_a_fixed_point_plane_under_a_float16_plan_raises(dense_store):
    """The same disagreement reached directly, without going through a
    manifest: the codec refuses rather than scaling by 1024."""
    opened = open_store(dense_store)
    raw = opened.arrays(mode="r")["z"][:]
    with pytest.raises(ValueError, match="manifest disagree"):
        StoreCodec(StoreEncoding.legacy()).decode_z(raw)


def test_a_reader_meeting_an_unknown_encoding_kind_rejects_the_release(dense_store):
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["z"] = {"kind": "int12_fixed", "scale": 256}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(dense_store)
    assert not result.ok
    assert any("int12_fixed" in error for error in result.errors), result.errors


def test_a_legacy_float16_store_still_reads_correctly(dense_store, tmp_path):
    """`0.1` stays readable and is never written again (ADR 0038 §2). A release
    that declares no encoding is in the legacy plan -- `float16` throughout --
    and the read path must decode it as such rather than as the plan this build
    happens to write."""
    opened = open_store(dense_store)
    root = opened.arrays(mode="r")
    decoded = DenseZPlane.open(root, opened.manifest.encoding).band(0, root["z"].shape[0])

    legacy = tmp_path / "legacy.opengwasdb"
    shutil.copytree(dense_store, legacy)
    legacy_root = zarr.open_group(str(legacy / "data.zarr"), mode="a")
    chunks = legacy_root["z"].chunks
    del legacy_root["z"]
    del legacy_root[Z_OVERFLOW_INDEX]
    del legacy_root[Z_OVERFLOW_VALUE]
    legacy_root.create_dataset(
        "z", data=decoded.astype(np.float16), chunks=chunks, dtype="float16"
    )
    manifest_path = legacy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["encoding"]
    manifest["format_version"] = "0.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    query = query_store(legacy)
    try:
        result = query.analysis("a1")
        variants = query.variants_table()
    finally:
        query.close()
    z_by_position = {
        variants[int(vi)]["position"]: float(z)
        for vi, z in zip(result["variant_index"], result["z"], strict=True)
    }
    # float16 tolerance, not fixed-point tolerance: this is the accuracy #114
    # exists to fix, and it is what a legacy store genuinely holds.
    assert z_by_position[100] == pytest.approx(ORDINARY_Z, abs=0.01)
    assert z_by_position[300] == pytest.approx(-FADS_Z, abs=0.05)


# ── Ragged ──────────────────────────────────────────────────────────────────


def _ssf_fixture(tmp_path: Path) -> tuple[Path, Path]:
    import gzip

    header = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "beta", "standard_error", "rsid", "variant_id",
    ]
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    # beta/se chosen so z is exactly the value named: z = beta / se.
    rows = [
        {"chromosome": "1", "base_pair_location": 100_000, "effect_allele": "A",
         "other_allele": "G", "beta": ORDINARY_Z * 0.5, "standard_error": 0.5, "rsid": "rs1"},
        {"chromosome": "1", "base_pair_location": 200_000, "effect_allele": "A",
         "other_allele": "G", "beta": BIG_Z * 0.002, "standard_error": 0.002, "rsid": "rs2"},
        {"chromosome": "1", "base_pair_location": 300_000, "effect_allele": "A",
         "other_allele": "G", "beta": NEAR_EDGE_Z * 0.01, "standard_error": 0.01, "rsid": "rs3"},
    ]
    with gzip.open(filtered_dir / "trait_a.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")

    manifest = tmp_path / "manifest.tsv"
    columns = [
        "analysis_index", "analysis_id", "trait_id", "analysis_label",
        "trait_ontology_id", "trait_ontology_label", "trait_chr", "trait_bp",
        "n", "tissue", "context", "mhc", "filtered_file",
    ]
    row = {
        "analysis_index": 0, "analysis_id": "trait_a", "trait_id": "T1",
        "analysis_label": "GENE1", "trait_ontology_id": "ENSEMBL:ENSG00001",
        "trait_ontology_label": "Ensembl", "trait_chr": "1", "trait_bp": 150_000,
        "n": 5000, "tissue": "Liver", "context": "", "mhc": "FALSE",
        "filtered_file": "trait_a.tsv.gz",
    }
    manifest.write_text(
        "\t".join(columns) + "\n" + "\t".join(str(row[c]) for c in columns) + "\n",
        encoding="utf-8",
    )
    return manifest, filtered_dir


@pytest.fixture
def ragged_store(tmp_path: Path) -> Path:
    manifest, filtered_dir = _ssf_fixture(tmp_path)
    out = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered_dir, out, store_id="enc-ragged", release_id="v1",
        allow_unverified_eaf=True,
    )
    return out


def test_ragged_csr_z_round_trips_including_out_of_range_values(ragged_store):
    query = query_store(ragged_store)
    try:
        result = query.analysis("trait_a")
        variants = query.variants_table()
    finally:
        query.close()
    z_by_position = {
        variants[int(vi)]["position"]: float(z)
        for vi, z in zip(result["variant_index"], result["z"], strict=True)
    }
    assert z_by_position[100_000] == pytest.approx(ORDINARY_Z, abs=ROUND_TRIP_TOL)
    assert z_by_position[200_000] == pytest.approx(BIG_Z, abs=ROUND_TRIP_TOL)
    assert z_by_position[300_000] == pytest.approx(NEAR_EDGE_Z, abs=ROUND_TRIP_TOL)
    assert _p_ratio(z_by_position[200_000], BIG_Z) < 1.02


def test_ragged_declares_its_encoding_and_validates_against_it(ragged_store):
    manifest = json.loads((ragged_store / "manifest.json").read_text())
    assert manifest["encoding"]["z"] == {"kind": "int16_fixed", "scale": 1024}
    assert validate_store(ragged_store).ok
    root = zarr.open_group(str(ragged_store / "data.zarr" / "ragged"), mode="r")
    assert str(root["z"].dtype) == "int16"
    assert list(np.asarray(root[Z_OVERFLOW_INDEX][:])) == [
        int(np.flatnonzero(np.asarray(root["z"][:]) == Z_OVERFLOW)[0])
    ]


def test_ragged_top_hit_index_agrees_with_the_stored_csr(ragged_store):
    csr = RaggedCSRReader(ragged_store)
    z_all = csr.z_all()
    offsets = csr._offsets[:]
    vi_all = csr._variant_index[:]
    root = zarr.open_group(str(ragged_store / "data.zarr"), mode="r")
    for key in root["top_hits"]:
        group = root["top_hits"][key]
        for vi, ai, z_hit in zip(
            group["variant_index"][:], group["analysis_index"][:], group["z"][:], strict=True
        ):
            start, end = int(offsets[int(ai)]), int(offsets[int(ai) + 1])
            pos = start + int(np.searchsorted(vi_all[start:end], vi))
            assert float(z_all[pos]) == pytest.approx(float(z_hit), abs=1e-6)


# ── Hybrid ──────────────────────────────────────────────────────────────────
#
# Both of a Hybrid release's components are written under one plan, so the
# test that matters is that an off-panel (Ragged Overflow) association and an
# on-panel (Dense Component) one come back equally accurate.

_HG38_ON_PANEL = "1:100000:A:G"
_HG38_OFF_PANEL = "1:1064620:C:T"


@pytest.fixture
def hybrid_store(tmp_path: Path) -> Path:
    vcf = tmp_path / "trait_a.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
        "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
        # Both rows report against ALT, which is not the canonical A1 (the
        # lexicographically smaller allele, ADR 0004), so both are stored with
        # the sign flipped -- hence the negated expectations below.
        # on-panel (Dense Component), out of range: ES/SE = 137.5
        f"1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t{BIG_Z * 0.002}:0.002\n"
        # off-panel (Ragged Overflow), out of range: ES/SE = -47.8
        f"1\t1000000\t.\tC\tT\t.\tPASS\t.\tES:SE\t{-FADS_Z * 0.01}:0.01\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\n"
        f"trait_a\t{vcf}\tTrait A\t1000\tsd\tdeclared_standardised\t\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{_HG38_ON_PANEL}\n", encoding="utf-8")

    out = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, out, reference_panel=panel, store_id="enc-hybrid", release_id="v1",
        allow_unverified_eaf=True,
    )
    return out


def test_hybrid_both_components_round_trip_out_of_range_z(hybrid_store):
    query = query_store(hybrid_store)
    try:
        result = query.analysis("trait_a")
        variants = query.variants_table()
    finally:
        query.close()
    z_by_alid = {
        variants[int(vi)]["alid"]: float(z)
        for vi, z in zip(result["variant_index"], result["z"], strict=True)
    }
    assert z_by_alid[_HG38_ON_PANEL] == pytest.approx(-BIG_Z, abs=ROUND_TRIP_TOL)
    assert z_by_alid[_HG38_OFF_PANEL] == pytest.approx(FADS_Z, abs=ROUND_TRIP_TOL)
    assert _p_ratio(z_by_alid[_HG38_ON_PANEL], BIG_Z) < 1.02


def test_hybrid_components_share_one_plan_and_validate(hybrid_store):
    manifest = json.loads((hybrid_store / "manifest.json").read_text())
    dense_manifest = json.loads((hybrid_store / "dense" / "manifest.json").read_text())
    assert manifest["encoding"] == dense_manifest["encoding"]
    assert StoreEncoding.from_manifest(manifest["encoding"]).z.scale == 1024
    assert validate_store(hybrid_store).ok


def test_hybrid_components_declaring_different_plans_are_rejected(hybrid_store):
    """The two components partition one Analysis's associations, so a result
    assembled from both is only coherent if they were encoded the same way --
    and each carries its own manifest, so nothing else couples them."""
    component_manifest = hybrid_store / "dense" / "manifest.json"
    manifest = json.loads(component_manifest.read_text())
    manifest["encoding"]["z"]["scale"] = 512
    component_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(hybrid_store)
    assert not result.ok
    assert any("different statistic encodings" in error for error in result.errors), result.errors
