"""Per-association sample size in a Ragged BESD build (issue #302, ADR 0067).

eQTLGen's `NrSamples` varies per (gene, SNP) pair from 99 to 31,684, and a
Store held one `sample_size` per Analysis. These tests pin the opt-in `n` plane
`build-ragged-besd --sample-size-table` writes: each pair's N stored exactly, a
pair the table does not name stored as missing -- never estimated -- and a build
without the table unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from besd_fixtures import write_besd_sparse_3f, write_epi, write_esi, write_sample_size_table

from opengwasdb.encoding import N_MISSING, EncodingMeasurements, SeMeasurements, StoreEncoding
from opengwasdb.encoding.plan import NEncoding, UnsupportedEncoding
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

P1 = "ENSG00000000001"
P2 = "ENSG00000000002"

#: Variant order on the axis is position order: rs1, rs2, rs3, rs4.
SNPS = [
    {"chr": "1", "snp_id": "rs1", "bp": 1_000_000, "a1": "A", "a2": "G"},
    {"chr": "1", "snp_id": "rs2", "bp": 1_100_000, "a1": "C", "a2": "T"},
    {"chr": "1", "snp_id": "rs3", "bp": 1_200_000, "a1": "A", "a2": "C"},
    {"chr": "2", "snp_id": "rs4", "bp": 2_000_000, "a1": "G", "a2": "T"},
]
PROBES = [
    {"chr": "1", "probe_id": P1, "bp": 1_050_000, "gene": "GENE1"},
    {"chr": "1", "probe_id": P2, "bp": 1_150_000, "gene": "GENE2"},
]
#: P1 holds rs1, rs2, rs3; P2 holds rs2 and rs4. z = beta / se.
ASSOCIATIONS = [
    [(0, 0.10, 0.02), (1, -0.20, 0.03), (2, 0.50, 0.05)],
    [(1, 0.30, 0.04), (3, -0.15, 0.025)],
]
#: Grouped by probe, in `.epi` order, any order within a probe. P1's rs2 is
#: absent; P2's rs1 names no association of P2's; ENSG...99 is no probe at all.
TABLE = [
    (P1, "rs3", "1200"),
    (P1, "rs1", "1000"),
    (P2, "rs4", "2400"),
    (P2, "rs2", "2100"),
    (P2, "rs1", "2000"),
    ("ENSG00000000099", "rs1", "50"),
]
#: Each Analysis's N in variant order, NaN where the table names no N.
EXPECTED_N = {P1: [1000.0, np.nan, 1200.0], P2: [2100.0, 2400.0]}


def _fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_esi(root / "test.esi", SNPS)
    write_epi(root / "test.epi", PROBES)
    write_besd_sparse_3f(root / "test.besd", len(PROBES), ASSOCIATIONS)
    return root / "test"


def _table(root: Path, rows: list[tuple[str, str, str]] = TABLE, name: str = "n.tsv.gz") -> Path:
    path = root / name
    write_sample_size_table(path, rows)
    return path


def _build(
    root: Path,
    *,
    table: Path | None,
    name: str = "out.opengwasdb",
    analyses_path: Path | None = None,
    store_esi_eaf: bool = False,
) -> Path:
    out = root / name
    fixture = root / "fixture"
    build_ragged_from_besd(
        fixture / "test" if (fixture / "test.besd").exists() else _fixture(fixture),
        out,
        store_id="test",
        release_id="v1",
        association_coverage="cis",
        sample_size_table=table,
        analyses_path=analyses_path,
        store_esi_eaf=store_esi_eaf,
    )
    return out


@pytest.fixture(scope="module")
def sampled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sampled")
    return _build(root, table=_table(root))


# ── The plan ────────────────────────────────────────────────────────────────


def test_a_plan_without_a_sample_size_input_declares_no_n() -> None:
    plan = StoreEncoding.decide(EncodingMeasurements(n_analyses=2))
    assert plan.n is None
    assert "n" not in plan.to_manifest()


def test_an_n_plan_is_uint32_and_round_trips_through_the_manifest() -> None:
    plan = StoreEncoding.decide(EncodingMeasurements(n_analyses=2, n_plane=True))
    assert plan.n == NEncoding(kind="uint32")
    assert plan.to_manifest()["n"] == {"kind": "uint32"}
    assert plan.to_manifest()["version"] == 3, "an additive plane must not bump the block"
    assert StoreEncoding.from_manifest(plan.to_manifest()) == plan
    assert N_MISSING == 2**32 - 1


def test_an_n_kind_this_build_does_not_implement_is_refused() -> None:
    block = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, n_plane=True)).to_manifest()
    block["n"] = {"kind": "float32"}
    with pytest.raises(UnsupportedEncoding, match="n encoding kind 'float32'"):
        StoreEncoding.from_manifest(block)


def test_setting_reference_eaf_keeps_the_n_plan() -> None:
    plan = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, n_plane=True))
    assert plan.with_eaf_reference(True).n == plan.n


# ── The builder ─────────────────────────────────────────────────────────────


def _analysis_n(store: Path, analysis_id: str) -> np.ndarray:
    from opengwasdb.query import query_store

    result = query_store(store).analysis(analysis_id)
    order = np.argsort(result["variant_index"])
    return np.asarray(result["n"], dtype=np.float64)[order]


def test_each_pair_stores_its_published_n_and_an_absent_pair_is_missing(sampled: Path) -> None:
    for analysis_id, expected in EXPECTED_N.items():
        np.testing.assert_array_equal(_analysis_n(sampled, analysis_id), expected)


def test_the_plane_is_uint32_with_the_reserved_missing_code(sampled: Path) -> None:
    from opengwasdb.store import arrays as store_arrays

    plane = np.asarray(store_arrays.open_group(sampled / "data.zarr" / "ragged")["n"][:])
    assert plane.dtype == np.uint32
    np.testing.assert_array_equal(plane, [1000, N_MISSING, 1200, 2100, 2400])
    manifest = json.loads((sampled / "manifest.json").read_text())
    assert manifest["encoding"]["n"] == {"kind": "uint32"}


def test_the_table_identity_and_its_match_counts_are_recorded(
    sampled: Path, tmp_path: Path
) -> None:
    record = json.loads((sampled / "manifest.json").read_text())["provenance"][
        "sample_size_table"
    ]
    table = Path(record["path"])
    assert record["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert record["n_rows"] == 6
    assert record["n_rows_matched"] == 4
    assert record["n_rows_unmatched"] == 2
    assert record["n_associations"] == 5
    assert record["n_associations_missing"] == 1


def test_an_analysis_given_n_declares_variant_level_scope(sampled: Path) -> None:
    from opengwasdb.model.analyses import read_analyses

    rows = read_analyses(sampled / "analyses.tsv").rows
    assert {r["analysis_id"]: r["sample_size_scope"] for r in rows} == {
        P1: "variant_level",
        P2: "variant_level",
    }


# ── Opt-in only ─────────────────────────────────────────────────────────────


def _tree(store: Path) -> dict[str, bytes]:
    """Every file under a release, keyed by relative path."""
    return {
        str(path.relative_to(store)): path.read_bytes()
        for path in sorted(store.rglob("*"))
        if path.is_file()
    }


def test_without_a_table_the_release_carries_no_trace_of_the_plane(tmp_path: Path) -> None:
    from opengwasdb.query import query_store
    from opengwasdb.store import arrays as store_arrays

    store = _build(tmp_path, table=None)
    manifest = json.loads((store / "manifest.json").read_text())
    assert "n" not in manifest["encoding"]
    assert "sample_size_table" not in manifest["provenance"]
    ragged = store_arrays.open_group(store / "data.zarr" / "ragged")
    assert "n" not in ragged
    assert "n" not in ragged["by_variant"]
    query = query_store(store)
    assert "n" not in query.analysis(P1)
    assert "n" not in query.phewas("rs2")
    assert "n" not in query.top_hits()


def test_a_table_changes_only_the_n_plane_and_its_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With and without the table, every other file's bytes are the same.

    Two builds from one file set differ in the `n` plane, its by-variant copy,
    the manifest and `analyses.tsv` (the scope) -- and in nothing else, so the
    opt-in cannot have perturbed a statistic plane. One Blosc thread, because a
    threaded compressor writes a chunk's blocks in completion order and its
    bytes differ run to run (`opengwasdb.store.arrays`, #243).
    """
    monkeypatch.setenv("BLOSC_NTHREADS", "1")
    plain = _tree(_build(tmp_path, table=None, name="plain.opengwasdb"))
    sampled = _tree(_build(tmp_path, table=_table(tmp_path), name="sampled.opengwasdb"))
    added = sorted(set(sampled) - set(plain))
    assert added, "the table must add the plane for this to mean anything"
    assert all(
        path.startswith(("data.zarr/ragged/n/", "data.zarr/ragged/by_variant/n/"))
        for path in added
    ), added
    assert not set(plain) - set(sampled)
    changed = sorted(path for path in plain if plain[path] != sampled[path])
    assert changed == ["analyses.tsv", "manifest.json"], changed
    before, after = (
        json.loads(tree["manifest.json"]) for tree in (plain, sampled)
    )
    for manifest in (before, after):
        manifest.pop("created_at")
    after["encoding"].pop("n")
    after["provenance"].pop("sample_size_table")
    assert before == after


# ── The by-variant copy and the query facade ────────────────────────────────


def _n_by_pair(result: dict[str, np.ndarray]) -> dict[tuple[int, int], float]:
    return {
        (int(v), int(a)): float(n)
        for v, a, n in zip(
            result["variant_index"], result["analysis_index"], result["n"], strict=True
        )
    }


def _csr_n_by_pair(store: Path) -> dict[tuple[int, int], float]:
    from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader

    csr = RaggedCSRReader(store)
    offsets = np.asarray(csr._offsets[:], dtype=np.int64)
    pairs: dict[tuple[int, int], float] = {}
    for analysis in range(csr.n_analyses):
        start, end = int(offsets[analysis]), int(offsets[analysis + 1])
        vi = np.asarray(csr._variant_index[start:end])
        for v, n in zip(vi, csr.n_slice(start, end), strict=True):
            pairs[(int(v), analysis)] = float(n)
    return pairs


def _same(left: dict[tuple[int, int], float], right: dict[tuple[int, int], float]) -> bool:
    return left.keys() == right.keys() and all(
        (np.isnan(left[k]) and np.isnan(right[k])) or left[k] == right[k] for k in left
    )


def test_every_query_shape_returns_the_csr_n(sampled: Path) -> None:
    from opengwasdb.query import query_store

    csr = _csr_n_by_pair(sampled)
    values = np.array(list(csr.values()))
    assert sorted(values[np.isfinite(values)]) == [1000.0, 1200.0, 2100.0, 2400.0]
    assert np.isnan(values).sum() == 1, "the fixture must hold a missing N to test the marker"
    query = query_store(sampled)
    shapes = {
        "phewas rs2": query.phewas("rs2"),
        "range_phewas": query.range_phewas("1", 900_000, 1_300_000),
        "range_by_analysis": query.range_by_analysis("1", 1_000_000, 1_200_000),
        "lookup": query.lookup(["rs1", "rs2", "rs3", "rs4"], [P1, P2]),
        "top_hits indexed": query.top_hits(threshold=5e-8),
        "top_hits scan": query.top_hits(threshold=1e-9),
        "analysis": query.analysis(P2),
    }
    for name, result in shapes.items():
        got = _n_by_pair(result)
        assert got, f"{name} must return rows for this to mean anything"
        assert _same(got, {pair: csr[pair] for pair in got}), name
    assert np.isnan(_n_by_pair(shapes["phewas rs2"])[(1, 0)]), "P1's rs2 has no N"


def test_an_empty_result_from_a_sampled_release_still_names_n(sampled: Path) -> None:
    from opengwasdb.query import query_store

    query = query_store(sampled)
    assert len(query.analysis("not-an-analysis")["n"]) == 0
    assert len(query.lookup(["rs4"], [P1])["n"]) == 0


def test_the_by_variant_copy_holds_the_csr_codes(sampled: Path) -> None:
    from opengwasdb.layouts.ragged.by_variant import ByVariantReader

    reader = ByVariantReader(sampled)
    first, last = reader.rows_for_variant_range(0, 3)
    block = reader.decode(0, 3, first, last)
    assert _same(_n_by_pair(block), _csr_n_by_pair(sampled))


# ── Validation ──────────────────────────────────────────────────────────────


def _copy(store: Path, tmp_path: Path) -> Path:
    import shutil

    copy = tmp_path / store.name
    shutil.copytree(store, copy)
    return copy


def _errors(store: Path) -> list[str]:
    from opengwasdb.validation import validate_store

    return validate_store(store).errors


def _rewrite(store: Path, name: str, values: np.ndarray) -> None:
    """Replace one `ragged/` (or `ragged/by_variant/`) array whole, keeping its layout."""
    from opengwasdb.store import arrays as store_arrays
    from opengwasdb.store.arrays import ArrayRole

    group, _, leaf = f"data.zarr/ragged/{name}".rpartition("/")
    root = store_arrays.open_group_for_write(store / group, "a")
    store_arrays.create_array(
        root,
        leaf,
        ArrayRole.ASSOCIATION_SEQUENCE,
        data=values,
        dtype=values.dtype,
        compressor=store_arrays.compressor(),
        overwrite=True,
    )


def _plane(store: Path, name: str = "n") -> np.ndarray:
    from opengwasdb.store import arrays as store_arrays

    return np.asarray(store_arrays.open_group(store / "data.zarr" / "ragged")[name][:])


def test_validate_passes_a_release_with_a_good_n_plane(sampled: Path) -> None:
    from opengwasdb.validation import validate_store

    result = validate_store(sampled)
    assert result.ok, result.errors


def test_validate_rejects_an_n_plane_of_the_wrong_length_by_name(
    sampled: Path, tmp_path: Path
) -> None:
    store = _copy(sampled, tmp_path)
    _rewrite(store, "n", np.append(_plane(store), np.uint32(7)))
    errors = _errors(store)
    assert any("data.zarr/ragged/n has 6 entries but offsets imply 5" in e for e in errors), errors


def test_validate_rejects_a_zero_sample_size_by_name(sampled: Path, tmp_path: Path) -> None:
    store = _copy(sampled, tmp_path)
    plane = _plane(store)
    plane[0] = 0
    _rewrite(store, "n", plane)
    _rewrite(store, "by_variant/n", plane[np.argsort([0, 1, 2, 1, 3], kind="stable")])
    errors = _errors(store)
    assert any("data.zarr/ragged/n" in e and "neither" in e for e in errors), errors


def test_validate_rejects_an_n_plane_the_manifest_does_not_declare(
    sampled: Path, tmp_path: Path
) -> None:
    store = _copy(sampled, tmp_path)
    manifest = json.loads((store / "manifest.json").read_text())
    del manifest["encoding"]["n"]
    (store / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    errors = _errors(store)
    assert any("carries an n plane but the manifest declares none" in e for e in errors), errors


def test_validate_rejects_a_declared_n_plane_that_is_missing(
    sampled: Path, tmp_path: Path
) -> None:
    import shutil

    store = _copy(sampled, tmp_path)
    shutil.rmtree(store / "data.zarr" / "ragged" / "n")
    errors = _errors(store)
    assert any("declares an n plane but carries none" in e for e in errors), errors


def test_validate_rejects_analysis_level_scope_on_an_analysis_with_n(
    sampled: Path, tmp_path: Path
) -> None:
    store = _copy(sampled, tmp_path)
    tsv = store / "analyses.tsv"
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    column = header.index("sample_size_scope")
    row = lines[1].split("\t")
    assert row[header.index("analysis_id")] == P1
    row[column] = "analysis_level"
    lines[1] = "\t".join(row)
    tsv.write_text("\n".join(lines) + "\n")
    errors = _errors(store)
    assert any(
        P1 in e and "sample_size_scope='analysis_level'" in e for e in errors
    ), errors


def test_validate_rejects_a_by_variant_n_that_disagrees_with_the_csr(
    sampled: Path, tmp_path: Path
) -> None:
    store = _copy(sampled, tmp_path)
    copied = np.asarray(_plane(store, "by_variant/n"))
    assert copied[0] == 1000, "the first by-variant row must be P1's rs1 for this to mean anything"
    copied[0] = 999
    _rewrite(store, "by_variant/n", copied)
    errors = _errors(store)
    assert any("by_variant" in e and "n code" in e for e in errors), errors


def test_validate_rejects_a_by_variant_index_without_the_n_copy(
    sampled: Path, tmp_path: Path
) -> None:
    import shutil

    store = _copy(sampled, tmp_path)
    shutil.rmtree(store / "data.zarr" / "ragged" / "by_variant" / "n")
    errors = _errors(store)
    assert any("by_variant/n is present on one side only" in e for e in errors), errors


# ── The `.esi` frequency, stored per association when asked ─────────────────

#: 600 variants: enough overlap (>= 500) and spread for the orientation check
#: to read a direction. Odd rows list the alleles against canonical order, so
#: their stored effect allele is the source's `a2` and the cell holds 1 - f.
N_FREQUENCY_SNPS = 600


def _frequency(i: int) -> float:
    return round(0.05 + 0.9 * i / (N_FREQUENCY_SNPS - 1), 6)


def _frequency_snps() -> list[dict]:
    return [
        {
            "chr": "1",
            "snp_id": f"rs{9000 + i}",
            "bp": 1_000_000 + 100 * i,
            "a1": "A" if i % 2 == 0 else "G",
            "a2": "G" if i % 2 == 0 else "A",
            "freq": f"{_frequency(i)}",
        }
        for i in range(N_FREQUENCY_SNPS)
    ]


def _stored_frequency(i: int) -> float:
    return _frequency(i) if i % 2 == 0 else 1.0 - _frequency(i)


def _frequency_fixture(root: Path) -> Path:
    """Three probes over the 600 variants, each probe a third of them."""
    root.mkdir(parents=True, exist_ok=True)
    write_esi(root / "test.esi", _frequency_snps())
    probes = [
        {"chr": "1", "probe_id": f"ENSG0000000010{k}", "bp": 1_000_000 + 20_000 * k}
        for k in range(3)
    ]
    write_epi(root / "test.epi", probes)
    write_besd_sparse_3f(
        root / "test.besd",
        3,
        [[(i, 0.01 * (i % 7 - 3), 0.02) for i in range(k, N_FREQUENCY_SNPS, 3)] for k in range(3)],
    )
    return root / "test"


def _reference(root: Path, *, against_other_allele: bool = False) -> Path:
    """An `alid`/`eaf` reference table, A1-oriented, near the `.esi` frequencies."""
    lines = ["alid\teaf"]
    for i, snp in enumerate(_frequency_snps()):
        frequency = _stored_frequency(i)
        if against_other_allele:
            frequency = 1.0 - frequency
        lines.append(f"1:{snp['bp']}:A:G\t{min(0.99, frequency + 0.01):.6f}")
    path = root / "reference.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


def _frequency_table(root: Path) -> Path:
    """A sample size for every pair of the frequency fixture, varying by pair."""
    rows = [
        (f"ENSG0000000010{k}", f"rs{9000 + i}", str(1000 + 10 * i))
        for k in range(3)
        for i in range(k, N_FREQUENCY_SNPS, 3)
    ]
    return _table(root, rows, name="freq-n.tsv.gz")


def _build_frequencies(
    root: Path,
    *,
    eaf_reference: Path | None = None,
    store_esi_eaf: bool = True,
    sample_size_table: Path | None = None,
    name: str = "freq.opengwasdb",
) -> Path:
    out = root / name
    fixture = root / "freq-fixture"
    build_ragged_from_besd(
        fixture / "test" if (fixture / "test.besd").exists() else _frequency_fixture(fixture),
        out,
        store_id="test",
        release_id="v1",
        association_coverage="cis",
        store_esi_eaf=store_esi_eaf,
        eaf_reference=eaf_reference,
        sample_size_table=sample_size_table,
    )
    return out


@pytest.fixture(scope="module")
def frequencies(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("frequencies")
    return _build_frequencies(root, eaf_reference=_reference(root))


def test_the_esi_frequency_is_stored_and_a_flipped_allele_stores_one_minus_f(
    frequencies: Path,
) -> None:
    from opengwasdb.query import query_store

    result = query_store(frequencies).analysis("ENSG00000000100")
    eaf = dict(zip(result["variant_index"].tolist(), result["eaf"].tolist(), strict=True))
    # Probe 0 holds variants 0, 3, 6, ...: variant 0 lists A/G (canonical), so it
    # stores f = 0.05; variant 3 lists G/A (flipped), so it stores 1 - f.
    assert _frequency(3) == pytest.approx(0.054508)
    assert eaf[0] == pytest.approx(0.05, abs=1e-6)
    assert eaf[3] == pytest.approx(1.0 - 0.054508, abs=1e-6)
    assert len(eaf) == 200
    for variant, value in eaf.items():
        assert value == pytest.approx(_stored_frequency(variant), abs=1e-6), variant


def test_stored_esi_frequencies_with_reference_evidence_pass_validate(frequencies: Path) -> None:
    from opengwasdb.model.analyses import read_analyses
    from opengwasdb.validation import validate_store

    result = validate_store(frequencies)
    assert result.ok, result.errors
    assert not [w for w in result.warnings if "orientation" in w], result.warnings
    rows = read_analyses(frequencies / "analyses.tsv").rows
    assert {(r["eaf_scope"], r["eaf_orientation"]) for r in rows} == {("association", "passed")}
    record = json.loads((frequencies / "manifest.json").read_text())["provenance"][
        "eaf_orientation"
    ]
    assert record["method"] == "reference_panel"
    assert record["scope"] == "source_file"
    assert len(record["analyses"]) == 3
    assert {entry["n_overlap"] for entry in record["analyses"]} == {N_FREQUENCY_SNPS}


def test_without_a_reference_the_probes_are_unverified_not_a_consensus(tmp_path: Path) -> None:
    """Three probes would meet the consensus rule, but they share one `.esi`
    frequency per variant and so agree by construction; they must not certify
    each other."""
    from opengwasdb.model.analyses import read_analyses
    from opengwasdb.validation import validate_store

    store = _build_frequencies(tmp_path)
    rows = read_analyses(store / "analyses.tsv").rows
    assert len(rows) == 3, "three probes are what would trigger the consensus rule"
    assert {r["eaf_orientation"] for r in rows} == {"unverified"}
    record = json.loads((store / "manifest.json").read_text())["provenance"]["eaf_orientation"]
    assert record["method"] == "none"
    result = validate_store(store)
    assert result.ok, result.errors
    assert any("unverified" in w for w in result.warnings), result.warnings


def test_an_esi_reported_against_the_other_allele_fails_the_build(tmp_path: Path) -> None:
    from opengwasdb.build.eaf_orientation import EafOrientationError

    with pytest.raises(EafOrientationError, match="anti-correlated"):
        _build_frequencies(
            tmp_path, eaf_reference=_reference(tmp_path, against_other_allele=True)
        )
    assert not (tmp_path / "freq.opengwasdb").exists()


def test_the_orientation_sample_is_the_whole_axis_sample_without_holding_the_axis() -> None:
    """Sampling a stream of ALIDs, then gathering only the sampled sites, is the
    sample the whole-axis dict gave -- including two variants at one position
    and rows with no frequency."""
    from opengwasdb.build.eaf_orientation import select_sites
    from opengwasdb.layouts.ragged.besd_eaf import sampled_frequencies
    from opengwasdb.variants import orient_to_canonical

    variants = [
        orient_to_canonical("1", str(1_000 + 10 * (i // 2)), "A", "G" if i % 2 else "C").variant
        for i in range(30_000)
    ]
    assert variants[0].position == variants[1].position, "the fixture must share positions"
    esi_to_variant = {row: (row, False) for row in range(len(variants))}
    eaf_by_esi = np.linspace(0.01, 0.99, len(variants)).astype(np.float32)
    eaf_by_esi[::7] = np.nan
    whole_axis = {
        variants[index].alid: float(eaf_by_esi[row])
        for row, (index, _flipped) in esi_to_variant.items()
        if np.isfinite(eaf_by_esi[row])
    }
    expected = {alid: whole_axis[alid] for alid in select_sites(whole_axis)}
    assert len(expected) == 20_000, "the axis must be larger than the sample to test sampling"
    assert sampled_frequencies(variants, esi_to_variant, eaf_by_esi) == expected


def test_without_the_flag_a_frequency_bearing_esi_stores_no_eaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `.esi` frequency is opt-in: an `.esi` that reports one builds the
    release a frequency-less `.esi` builds, byte for byte (ADR 0067)."""
    from opengwasdb.store import arrays as store_arrays

    monkeypatch.setenv("BLOSC_NTHREADS", "1")
    store = _build_frequencies(tmp_path, store_esi_eaf=False)
    assert any(snp["freq"] != "NA" for snp in _frequency_snps()), "the .esi must report some"
    manifest = json.loads((store / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"] == {"kind": "absent"}
    assert "eaf_orientation" not in manifest["provenance"]
    assert "eaf" not in store_arrays.open_group(store / "data.zarr" / "ragged")
    blank = tmp_path / "blank"
    blank.mkdir()
    for suffix in ("epi", "besd"):
        (blank / f"test.{suffix}").write_bytes(
            (tmp_path / "freq-fixture" / f"test.{suffix}").read_bytes()
        )
    write_esi(blank / "test.esi", [{**snp, "freq": "NA"} for snp in _frequency_snps()])
    plain = blank / "plain.opengwasdb"
    build_ragged_from_besd(
        blank / "test", plain, store_id="test", release_id="v1", association_coverage="cis"
    )
    with_frequencies, without = _tree(store), _tree(plain)
    assert with_frequencies.keys() == without.keys()
    differ = [path for path in without if with_frequencies[path] != without[path]]
    assert differ == ["manifest.json"], differ
    before, after = (json.loads(tree["manifest.json"]) for tree in (with_frequencies, without))
    for record in (before, after):
        record.pop("created_at")
        record["provenance"].pop("source_besd_prefix")
    assert before == after


def test_the_flag_on_an_esi_without_frequencies_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store_esi_eaf was requested but the .esi reports no"):
        _build(tmp_path, table=None, store_esi_eaf=True)
    assert not (tmp_path / "out.opengwasdb").exists()


def test_an_eaf_reference_without_the_flag_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only with store_esi_eaf"):
        _build_frequencies(tmp_path, store_esi_eaf=False, eaf_reference=_reference(tmp_path))
    assert not (tmp_path / "freq.opengwasdb").exists()


# ── SE stays float16 beside an `n` plane ────────────────────────────────────


def _se_measurements_that_choose_a_residual() -> SeMeasurements:
    return SeMeasurements(
        eligible=True,
        exception_fraction={0.5: 0.0},
        worst_relative_error={0.5: 0.002},
        compressed_bytes={0.5: 100},
        float16_compressed_bytes=1000,
    )


def test_an_n_plan_keeps_se_float16_whatever_the_measurements_say() -> None:
    measured = _se_measurements_that_choose_a_residual()
    without = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, se=measured))
    assert without.se.is_residual, "the measurements must choose a residual to mean anything"
    with_n = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, se=measured, n_plane=True))
    assert with_n.se.kind == "float16"


def test_a_build_with_n_and_stored_eaf_keeps_se_float16(tmp_path: Path) -> None:
    """The residual model assumes one N per Analysis; per-pair N breaks it (ADR 0067)."""
    reference = _reference(tmp_path)
    without_n = _build_frequencies(tmp_path, eaf_reference=reference, name="eaf.opengwasdb")
    with_n = _build_frequencies(
        tmp_path,
        eaf_reference=reference,
        sample_size_table=_frequency_table(tmp_path),
        name="eaf-n.opengwasdb",
    )
    plans = {
        store.name: json.loads((store / "manifest.json").read_text())["encoding"]
        for store in (without_n, with_n)
    }
    assert plans["eaf.opengwasdb"]["se"]["kind"] == "int8_residual", (
        "the same frequencies without n must choose the residual for this to mean anything"
    )
    assert plans["eaf-n.opengwasdb"]["se"] == {"kind": "float16"}
    assert plans["eaf-n.opengwasdb"]["eaf"]["kind"] != "absent"
    assert "n" in plans["eaf-n.opengwasdb"]


def _residual_se_release_with_n(tmp_path: Path) -> Path:
    """Review round 1's reproduction: a valid residual-SE release given a
    consistent `n` plane -- both copies, the plan, and the scope -- after the fact."""
    from opengwasdb.validation import validate_store

    store = _build_frequencies(tmp_path, eaf_reference=_reference(tmp_path))
    manifest = json.loads((store / "manifest.json").read_text())
    assert manifest["encoding"]["se"]["kind"] == "int8_residual", (
        "the release must code se as a residual for this to mean anything"
    )
    assert validate_store(store).ok, "the release must be valid before n is added"
    codes = np.full(len(_plane(store, "variant_index")), 1000, dtype=np.uint32)
    _rewrite(store, "n", codes)
    _rewrite(store, "by_variant/n", codes)
    manifest["encoding"]["n"] = {"kind": "uint32"}
    (store / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tsv = store / "analyses.tsv"
    header, *rows = (line.split("\t") for line in tsv.read_text().splitlines())
    scope = header.index("sample_size_scope")
    for row in rows:
        row[scope] = "variant_level"
    tsv.write_text("".join("\t".join(row) + "\n" for row in [header, *rows]))
    return store


def test_validate_rejects_residual_se_beside_an_n_plane(tmp_path: Path) -> None:
    errors = _errors(_residual_se_release_with_n(tmp_path))
    assert any("n plane requires float16 se" in e and "int8_residual" in e for e in errors), (
        errors
    )


def test_the_loader_refuses_residual_se_beside_an_n_plan() -> None:
    block = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, n_plane=True)).to_manifest()
    block["se"] = {"kind": "int8_residual", "residual_range": 0.5}
    with pytest.raises(UnsupportedEncoding, match="n plane requires float16 se"):
        StoreEncoding.from_manifest(block)


def test_the_validation_rule_names_residual_se_beside_an_n_plan() -> None:
    """The rule itself, for a plan that reached validation without the loader."""
    from dataclasses import replace

    from opengwasdb.encoding import SeEncoding
    from opengwasdb.validation.sample_size import validate_n_plan

    plan = replace(
        StoreEncoding.decide(EncodingMeasurements(n_analyses=1, n_plane=True)),
        se=SeEncoding("int8_residual", 0.5),
    )
    errors: list[str] = []
    validate_n_plan({"n": np.ones(1, dtype=np.uint32)}, plan, errors, label="data.zarr/ragged")
    assert errors == [
        "data.zarr/ragged declares an n plane beside se int8_residual, but an n plane "
        "requires float16 se (ADR 0067 §4)"
    ]


# ── N is exact at the precision boundaries ──────────────────────────────────

#: Literals, not derived: the first count `float32` cannot hold (2**24 + 1),
#: the largest count the plane holds (2**32 - 2), and the missing code (2**32 - 1).
BEYOND_FLOAT32 = 16_777_217
LARGEST_COUNT = 4_294_967_294
MISSING_CODE = 4_294_967_295


def test_counts_past_float32_round_trip_exactly_through_the_codec() -> None:
    from opengwasdb.encoding import StoreCodec, encode_sample_sizes

    codes = encode_sample_sizes(np.array([BEYOND_FLOAT32, LARGEST_COUNT, np.nan]))
    assert codes.dtype == np.uint32
    assert codes.tolist() == [16_777_217, 4_294_967_294, 4_294_967_295]
    plan = StoreEncoding.decide(EncodingMeasurements(n_analyses=1, n_plane=True))
    decoded = StoreCodec(plan).decode_n(codes)
    assert decoded[:2].tolist() == [16_777_217.0, 4_294_967_294.0]
    assert np.isnan(decoded[2])


def test_counts_past_float32_round_trip_exactly_through_a_build(tmp_path: Path) -> None:
    rows = [(P1, "rs1", "16777217"), (P1, "rs3", "4294967294"), (P2, "rs2", "2100")]
    store = _build(tmp_path, table=_table(tmp_path, rows))
    assert _plane(store).tolist() == [16_777_217, N_MISSING, 4_294_967_294, 2100, N_MISSING]
    stored = _analysis_n(store, P1)
    assert stored[0] == 16_777_217.0 and stored[2] == 4_294_967_294.0, stored.tolist()


def test_the_missing_code_is_refused_as_an_observed_count(tmp_path: Path) -> None:
    from opengwasdb.encoding import encode_sample_sizes
    from opengwasdb.layouts.ragged.sample_size_table import SampleSizeTableError

    with pytest.raises(ValueError, match="4294967295.0"):
        encode_sample_sizes(np.array([MISSING_CODE], dtype=np.float64))
    with pytest.raises(SampleSizeTableError, match="n '4294967295' is not a whole number"):
        _build(tmp_path, table=_table(tmp_path, [(P1, "rs1", "4294967295")]))


# ── The table's format is enforced, not guessed at ──────────────────────────


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(P2, "rs2", "2100"), (P1, "rs1", "1000")],
            "rows for probe 'ENSG00000000001' arrive after the build reached probe "
            "'ENSG00000000002'",
        ),
        (
            [(P1, "rs1", "1000"), (P2, "rs2", "2100"), (P1, "rs3", "1200")],
            "rows for probe 'ENSG00000000001' arrive a second time",
        ),
        ([(P1, "rs1", "1000"), (P1, "rs1", "1000")], "appears twice"),
        ([(P1, "rs1", "0")], "n '0' is not a whole number"),
        ([(P1, "rs1", "31684.5")], "n '31684.5' is not a whole number"),
        ([(P1, "rs1", "NA")], "n 'NA' is not a whole number"),
        ([("ENSG00000000099", "rs1", "50")], "gives none of the build's 5 associations an N"),
    ],
)
def test_a_malformed_table_fails_the_build_by_line(
    tmp_path: Path, rows: list[tuple[str, str, str]], message: str
) -> None:
    from opengwasdb.layouts.ragged.sample_size_table import SampleSizeTableError

    with pytest.raises(SampleSizeTableError, match=message):
        _build(tmp_path, table=_table(tmp_path, rows, name="bad.tsv"))
    assert not (tmp_path / "out.opengwasdb").exists()


def test_a_block_out_of_order_mid_table_is_refused_when_it_is_met(tmp_path: Path) -> None:
    """A block for a probe the build has passed, followed by more probes.

    The end-of-table drain cannot see this one: the rows of the later probe
    are served, so skipping the stray block would leave the table looking
    consumed while one probe's N silently went unread.
    """
    from opengwasdb.layouts.ragged.sample_size_table import (
        SampleSizeTable,
        SampleSizeTableError,
    )

    table = SampleSizeTable(
        _table(tmp_path, [("B", "rs1", "20"), ("A", "rs1", "10"), ("C", "rs1", "30")]),
        ["A", "B", "C"],
    )
    assert table.sizes_for("A") == {}, "A's block comes after B's, so A meets B first"
    assert table.sizes_for("B") == {"rs1": 20}
    with pytest.raises(SampleSizeTableError, match="arrive after the build reached probe 'C'"):
        table.sizes_for("C")


def test_a_manifest_declaring_analysis_level_scope_for_a_sampled_probe_fails(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "analyses.tsv"
    manifest.write_text(
        "analysis_id\tsample_size\tsample_size_scope\n"
        f"{P1}\t31684\tanalysis_level\n{P2}\t31684\t\n"
    )
    with pytest.raises(ValueError, match="declares sample_size_scope='analysis_level'"):
        _build(tmp_path, table=_table(tmp_path), analyses_path=manifest)


# ── Completion and the CLI ──────────────────────────────────────────────────


def test_ragged_completion_refuses_a_source_with_an_n_plane(sampled: Path, tmp_path: Path) -> None:
    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    with pytest.raises(ValueError, match="per-association sample-size plane"):
        complete_ragged_store(sampled, tmp_path / "completed.opengwasdb", tmp_path / "ld")
    assert not (tmp_path / "completed.opengwasdb").exists()


def test_the_cli_builds_the_plane_and_query_output_shows_n(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    runner = CliRunner()
    out = tmp_path / "cli.opengwasdb"
    built = runner.invoke(
        app,
        [
            "build-ragged-besd", str(_fixture(tmp_path / "fixture")), str(out),
            "--association-coverage", "cis", "--store-id", "cli", "--release-id", "v1",
            "--sample-size-table", str(_table(tmp_path)),
        ],
    )
    assert built.exit_code == 0, built.output
    queried = runner.invoke(app, ["query-analysis", str(out), P1])
    assert queried.exit_code == 0, queried.output
    lines = [line.split("\t") for line in queried.output.strip().splitlines()]
    header = lines[0]
    assert header[-3:] == ["eaf", "n", "association_status"]
    n_by_alid = {row[header.index("alid")]: row[header.index("n")] for row in lines[1:]}
    assert n_by_alid == {"1:1000000:A:G": "1000", "1:1100000:C:T": ".", "1:1200000:A:C": "1200"}
    described = runner.invoke(app, ["info", str(out)])
    assert described.exit_code == 0, described.output
    assert ", n=uint32" in described.output


def test_query_output_has_no_n_column_without_the_plane(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    store = _build(tmp_path, table=None)
    queried = CliRunner().invoke(app, ["query-analysis", str(store), P1])
    assert queried.exit_code == 0, queried.output
    assert "\tn\t" not in queried.output.splitlines()[0]
    described = CliRunner().invoke(app, ["info", str(store)])
    assert "encoding: z=" in described.output and "n=" not in described.output


def test_the_cli_stores_the_esi_frequency_only_with_its_flag(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    prefix = _frequency_fixture(tmp_path / "freq-fixture")
    reference = _reference(tmp_path)
    base = [
        "build-ragged-besd", str(prefix), "--association-coverage", "cis",
        "--store-id", "cli", "--release-id", "v1",
    ]
    runner = CliRunner()
    stored = runner.invoke(
        app,
        [*base[:2], str(tmp_path / "eaf.opengwasdb"), *base[2:], "--store-esi-eaf",
         "--eaf-reference", str(reference)],
    )
    assert stored.exit_code == 0, stored.output
    plan = json.loads((tmp_path / "eaf.opengwasdb" / "manifest.json").read_text())["encoding"]
    assert plan["eaf"]["kind"] != "absent"
    refused = runner.invoke(
        app, [*base[:2], str(tmp_path / "ref.opengwasdb"), *base[2:], "--eaf-reference",
              str(reference)],
    )
    assert refused.exit_code != 0
    assert "only with store_esi_eaf" in str(refused.exception)
    assert not (tmp_path / "ref.opengwasdb").exists()
