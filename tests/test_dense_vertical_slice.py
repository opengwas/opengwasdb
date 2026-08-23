from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.encoding import DenseZPlane
from opengwasdb.index import connect, get_metadata
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store


def test_dense_build_writes_standard_envelope_and_metadata(dense_store_path):
    assert (dense_store_path / "manifest.json").exists()
    assert (dense_store_path / "index.sqlite").exists()
    assert (dense_store_path / "data.zarr").exists()
    assert (dense_store_path / "variants.tsv.gz").exists()
    assert (dense_store_path / "variants.tsv.gz.tbi").exists()
    assert (dense_store_path / "variant_offsets.npy").exists()
    assert (dense_store_path / "variant_alid_bytes.npy").exists()
    assert (dense_store_path / "variant_alid_rows.npy").exists()

    result = validate_store(dense_store_path)
    assert result.ok, result.errors

    root = open_store(dense_store_path).arrays(mode="r")
    assert root["z"].shape == (3, 2)
    assert root["se"].shape == (3, 2)
    assert root["z"].chunks == (3, 2)  # clipped to array shape

    import math

    z = DenseZPlane.open(root, open_store(dense_store_path).manifest.encoding).band(
        0, root["z"].shape[0]
    )
    se = root["se"][:].astype("float32")
    assert z[0, 0] == 2.0
    assert z[0, 1] == 6.0
    assert z[1, 0] == -3.0
    assert math.isnan(float(z[1, 1]))
    assert math.isnan(float(se[2, 0]))
    assert z[2, 1] == 6.0

    with connect(dense_store_path / "index.sqlite") as connection:
        assert get_metadata(connection, "n_variants") == 3
        assert get_metadata(connection, "n_analyses") == 2
        dense_meta = get_metadata(connection, "dense")
        variant_rows = connection.execute("SELECT COUNT(*) AS n FROM variants").fetchone()["n"]
        aliases = connection.execute(
            "SELECT alias, variant_index FROM variant_aliases ORDER BY alias, variant_index"
        ).fetchall()
    assert variant_rows == 0
    assert [(row["alias"], row["variant_index"]) for row in aliases] == [
        ("rs1", 0),
        ("rs2", 1),
        ("rs3", 2),
    ]
    assert dense_meta["chunk_shape"] == [1000, 1000]
    assert dense_meta["compressor"]["cname"] == "zstd"
    assert dense_meta["compressor"]["shuffle"] == "bitshuffle"
    assert dense_meta["variant_axis"]["format"] == "tabix_tsv_v1"


def test_analyses_tsv_has_real_top_hit_counts(dense_store_path):
    # a1: z=2.0, z=-3.0 -- neither passes any threshold (loosest is 5e-4).
    # a2: z=6.0, z=6.0 -- both pass all three thresholds (ADR 0032).
    rows = {r["analysis_id"]: r for r in read_analyses(dense_store_path / "analyses.tsv").rows}
    for column in ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
        assert rows["a1"][column] == "0"
        assert rows["a2"][column] == "2"


def test_query_facade_supports_variant_range_analysis_phewas_top_hits_and_metadata(
    dense_store_path,
):
    # z matrix (3 variants × 2 analyses):
    #   a1(0)  a2(1)
    #   2.0    6.0   ← variant 0: 1:100:A:G (rs1)
    #  -3.0    NaN   ← variant 1: 1:200:C:T (rs2)
    #   NaN    6.0   ← variant 2: 1:300:A:G (rs3)
    query = query_store(dense_store_path)

    # phewas via alias — variant 0 has data in both analyses
    phewas = query.phewas("rs1")
    assert sorted(phewas["analysis_index"].tolist()) == [0, 1]
    assert len(phewas["z"]) == 2

    # range: variants 0 and 1 are in [1, 250]; variant 1 missing a2 → 3 cells
    range_res = query.range_phewas("1", 1, 250)
    cells = sorted(
        zip(
            range_res["variant_index"].tolist(),
            range_res["analysis_index"].tolist(),
            strict=True,
        )
    )
    assert cells == [(0, 0), (0, 1), (1, 0)]

    # analysis: a1 has cells for variants 0 and 1 only
    a1 = query.analysis("a1")
    assert sorted(a1["variant_index"].tolist()) == [0, 1]
    assert (a1["analysis_index"] == 0).all()
    z_by_vi = {
        v: z for v, z in zip(a1["variant_index"].tolist(), a1["z"].tolist(), strict=True)
    }
    assert z_by_vi[0] == pytest.approx(2.0, rel=5e-3)
    assert z_by_vi[1] == pytest.approx(-3.0, rel=5e-3)

    # lookup: 3 finite cells out of 4 requested (1:300×a1 is NaN)
    lookup = query.lookup(["1:100:A:G", "1:300:A:G"], ["a1", "a2"])
    assert len(lookup["z"]) == 3

    # top_hits: only |z|=6.0 cells pass p < 5e-8 → 2 cells (both a2)
    top_hits = query.top_hits(threshold=5e-8)
    assert len(top_hits["z"]) == 2
    assert all(abs(z) >= 6.0 for z in top_hits["z"])
    assert (top_hits["analysis_index"] == 1).all()

    # Analysis-addressed top hits use the same sparse result contract and
    # return only that analysis's genomic-order slice.
    a2_top_hits = query.top_hits(analysis_id="a2", threshold=5e-8, limit=1)
    assert a2_top_hits["variant_index"].tolist() == [0]
    assert a2_top_hits["analysis_index"].tolist() == [1]
    assert query.top_hits(analysis_id="unknown", threshold=5e-8)["z"].size == 0

    # metadata accessors
    variants = query.variants_table()
    assert len(variants) == 3
    assert variants[0]["alid"] == "1:100:A:G"
    assert variants[0]["rsid"] == "rs1"

    analyses = query.analyses_table()
    assert len(analyses) == 2
    assert analyses[0]["analysis_id"] == "a1"
    assert analyses[1]["stored_effect_scale"] == "log_or"


def test_dense_index_does_not_duplicate_canonical_alids(dense_store_path):
    query = query_store(dense_store_path)
    variants = query.variants_table()
    alids = [v["alid"] for v in variants.values()]
    assert len(alids) == len(set(alids))


def test_query_excludes_missing_dense_cells(dense_store_path):
    query = query_store(dense_store_path)

    result = query.range_phewas("1", 150, 350)

    # variant 1 (pos 200) has a1 only; variant 2 (pos 300) has a2 only → 2 cells
    assert len(result["z"]) == 2
    assert all(np.isfinite(result["z"]))
    assert all(np.isfinite(result["se"]))


def test_range_indices_matches_range_variant_index(dense_store_path):
    from opengwasdb.index import connect
    from opengwasdb.variants import VariantAxis

    with connect(dense_store_path / "index.sqlite") as conn:
        va = VariantAxis(dense_store_path, conn)
        full_records = va.range("1", 1, 500)
        fast_indices = va.range_indices("1", 1, 500)
        va.close()

    expected = np.array([r.variant_index for r in full_records], dtype="int32")
    assert np.array_equal(fast_indices, expected)
    # empty range returns empty array
    with connect(dense_store_path / "index.sqlite") as conn:
        va = VariantAxis(dense_store_path, conn)
        assert len(va.range_indices("1", 999999, 999999)) == 0
        va.close()


def test_lookup_surgical_read_matches_direct_matrix(dense_store_path):
    # issue 052: surgical oindex[rows, cols] must return exactly the finite cells
    # (and the same values) as a direct per-cell matrix read — no over- or
    # under-reporting when rows/cols are scattered and some cells are missing.
    query = query_store(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="r")
    variants = query.variants_table()
    analyses = query.analyses_table()
    alids = [v["alid"] for v in variants.values()]
    aids = [a["analysis_id"] for a in analyses.values()]

    result = query.lookup(alids, aids)
    z = DenseZPlane.open(root, open_store(dense_store_path).manifest.encoding).band(
        0, root["z"].shape[0]
    )
    se = root["se"][:].astype("float32")

    returned = set()
    for r, c, zz, ss in zip(
        result["variant_index"], result["analysis_index"],
        result["z"], result["se"], strict=True,
    ):
        r, c = int(r), int(c)
        assert np.isfinite(z[r, c]) and np.isfinite(se[r, c])
        assert zz == pytest.approx(float(z[r, c]), rel=1e-3, abs=1e-3)
        assert ss == pytest.approx(float(se[r, c]), rel=1e-3, abs=1e-3)
        returned.add((r, c))

    finite = {
        (int(r), int(c))
        for r, c in zip(*np.where(np.isfinite(z) & np.isfinite(se)), strict=True)
    }
    assert returned == finite
