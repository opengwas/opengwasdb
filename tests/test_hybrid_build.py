"""End-to-end tests for the Hybrid build + unified query (issues 055/056).

Reuses the dense VCF fixture coordinates. The reference panel deliberately holds
only two of the three variants so the third routes to the Ragged Overflow.

Panel (on-panel → Dense Component):
  1:100000:A:G     (HG38_ALID_1)
  1:1564620:A:G    (HG38_ALID_3)
Off-panel (→ Ragged Overflow):
  1:1064620:C:T    (HG38_ALID_2)
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.readers import GWAS_SSF_CAPABILITY
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

HG19_POS_1 = 100_000
HG19_POS_2 = 1_000_000
HG19_POS_3 = 1_500_000

HG38_ALID_1 = "1:100000:A:G"
HG38_ALID_2 = "1:1064620:C:T"
HG38_ALID_3 = "1:1564620:A:G"


def _vcf_header(study_type: str = "Continuous") -> str:
    return (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
        "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
        "##FORMAT=<ID=EZ,Number=A,Type=Float,Description=\"Z-score\">\n"
        "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Alternate allele frequency\">\n"
        f"##SAMPLE=<ID=STUDY1,StudyType={study_type}>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
    )


def _make_vcf(tmp_path: Path, name: str, rows: list[str], study_type: str = "Continuous") -> Path:
    path = tmp_path / f"{name}.vcf"
    path.write_text(_vcf_header(study_type) + "".join(rows), encoding="utf-8")
    return path


def _make_manifest(
    tmp_path: Path,
    entries: list[tuple[str, Path, str]],
    scales: dict[str, str] | None = None,
    sd_methods: dict[str, str] | None = None,
    sds: dict[str, str] | None = None,
) -> Path:
    """Write the build manifest; see the identical helper in
    test_dense_vcf_build.py for the full rationale -- this builder shares
    its manifest reader with the dense one."""
    scales = scales or {}
    sd_methods = sd_methods or {}
    sds = sds or {}
    manifest = tmp_path / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd"
    ]
    for trait_id, file_path, trait_name in entries:
        scale = scales.get(trait_id, "sd")
        sd_method = sd_methods.get(trait_id, "declared_standardised")
        sd = sds.get(trait_id, "")
        lines.append(
            f"{trait_id}\t{file_path}\t{trait_name}\t1000\t{scale}\t{sd_method}\t{sd}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _panel(tmp_path: Path) -> Path:
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{HG38_ALID_1}\n{HG38_ALID_3}\n", encoding="utf-8")
    return panel


@pytest.fixture(params=[1, 2], ids=["serial", "parallel"])
def hybrid_store(tmp_path, request):
    vcf1 = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",  # z -4.0 dense
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:0.3:0.3\n",  # z -5.0 OVERFLOW
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",  # z  3.0 dense
        ],
    )
    vcf2 = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n",  # z -12.0 dense
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t1.2:0.3:0.45\n",  # z   4.0 dense
        ],
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
    )
    store_path = tmp_path / f"store_{request.param}.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest,
        store_path,
        reference_panel=_panel(tmp_path),
        store_id="hybrid-test",
        release_id="v1",
        n_workers=request.param,
    )
    return store_path


def test_manifest_is_hybrid(hybrid_store):
    manifest = StoreManifest.load(hybrid_store)
    assert manifest.primary_layout.value == "hybrid"
    assert manifest.association_coverage.value == "full"


def test_analyses_tsv_has_no_phenotype_columns_and_carries_analysis_label(hybrid_store):
    """ADR 0034/issue #68: the Hybrid builder writes the unified schema at
    both the Dense Component and shared/top-level analyses.tsv."""
    for path in (hybrid_store / "dense" / "analyses.tsv", hybrid_store / "analyses.tsv"):
        table = read_analyses(path)
        assert "phenotype_id" not in table.fieldnames
        assert "phenotype_label" not in table.fieldnames
        rows = {r["analysis_id"]: r for r in table.rows}
        assert rows["trait_a"]["analysis_label"] == "Trait A"
        assert rows["trait_b"]["analysis_label"] == "Trait B"


def test_stored_effect_scale_comes_from_manifest_not_header(tmp_path):
    """The ieu-a-7 fix (issue #17), hybrid path: a VCF header says
    ``StudyType=Continuous`` but the manifest declares ``log_or`` -- the
    built store must record the manifest's value."""
    vcf = _make_vcf(
        tmp_path,
        "trait_a",
        [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
        study_type="Continuous",
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf, "Trait A")], scales={"trait_a": "log_or"}
    )
    store_path = tmp_path / "store.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path),
        store_id="s", release_id="r",
    )

    q = query_store(store_path)
    analyses = q.analyses_table()
    assert analyses[0]["stored_effect_scale"] == "log_or"


def test_missing_required_manifest_field_fails_the_build_loudly(tmp_path):
    """A manifest missing stored_effect_scale must fail the build with a
    clear error before any I/O, not fall back to VCF-header inference or a
    silent default (issue #17)."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\n"
        f"trait_a\t{vcf}\tTrait A\t1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stored_effect_scale"):
        build_hybrid_from_vcf_manifest(
            manifest_path, tmp_path / "store.opengwasdb",
            reference_panel=_panel(tmp_path), store_id="s", release_id="r",
        )


def test_continuous_trait_rescaled_by_manifest_original_sd(tmp_path):
    """issue #18 AC1, hybrid path: a continuous-trait Analysis with a
    manifest-supplied original_sd != 1 has its se divided by that SD, for
    both the on-panel (Dense Component) and off-panel (overflow) associations
    of the same study."""
    vcf = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # on-panel, se=0.5
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.4\n",  # off-panel, se=0.4
        ],
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_a", vcf, "Trait A")],
        sd_methods={"trait_a": "source_provided"},
        sds={"trait_a": "2.0"},
    )
    store_path = tmp_path / "store.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path), store_id="s", release_id="r",
    )

    q = query_store(store_path)
    on_panel = q.lookup([HG38_ALID_1], ["trait_a"])
    off_panel = q.lookup([HG38_ALID_2], ["trait_a"])
    assert on_panel["se"][0] == pytest.approx(0.25, rel=5e-3)  # 0.5 / 2.0
    assert off_panel["se"][0] == pytest.approx(0.2, rel=5e-3)  # 0.4 / 2.0


def test_binary_trait_never_rescaled(tmp_path):
    """issue #18 AC2, hybrid path: original_sd_method=binary_trait is never
    rescaled, for either the on-panel or off-panel associations of the study."""
    vcf = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # on-panel, se=0.5
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.4\n",  # off-panel, se=0.4
        ],
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_b", vcf, "Trait B")],
        scales={"trait_b": "log_or"},
        sd_methods={"trait_b": "binary_trait"},
    )
    store_path = tmp_path / "store.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path), store_id="s", release_id="r",
    )

    q = query_store(store_path)
    on_panel = q.lookup([HG38_ALID_1], ["trait_b"])
    off_panel = q.lookup([HG38_ALID_2], ["trait_b"])
    assert on_panel["se"][0] == pytest.approx(0.5, rel=5e-3)
    assert off_panel["se"][0] == pytest.approx(0.4, rel=5e-3)


def test_original_sd_method_unavailable_fails_the_build_loudly(tmp_path):
    """issue #18 AC3, hybrid path: original_sd_method=unavailable fails the
    build rather than assuming sd=1."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf, "Trait A")], sd_methods={"trait_a": "unavailable"}
    )

    with pytest.raises(ValueError, match="unavailable"):
        build_hybrid_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb",
            reference_panel=_panel(tmp_path), store_id="s", release_id="r",
        )


def test_store_envelope(hybrid_store):
    assert (hybrid_store / "manifest.json").exists()
    assert (hybrid_store / "variants.tsv.gz").exists()
    assert (hybrid_store / "dense" / "data.zarr").exists()
    assert (hybrid_store / "data.zarr" / "ragged").exists()
    assert (hybrid_store / "dense" / "dense_to_shared.npy").exists()


def test_dense_matrix_is_panel_sized(hybrid_store):
    root = open_store(hybrid_store).dense_component().arrays(mode="r")
    # 2 panel variants × 2 analyses (off-panel variant is NOT a dense row).
    assert root["z"].shape == (2, 2)


def test_shared_table_is_union(hybrid_store):
    manifest = StoreManifest.load(hybrid_store)
    assert manifest.provenance["hybrid"]["n_panel"] == 2
    assert manifest.provenance["hybrid"]["n_off_panel"] == 1
    assert manifest.provenance["n_variants"] == 3


def test_lookup_on_panel_hits_dense(hybrid_store):
    q = query_store(hybrid_store)
    r = q.lookup([HG38_ALID_1], ["trait_a"])
    assert len(r["z"]) == 1
    assert r["z"][0] == pytest.approx(-4.0, rel=5e-3)
    assert r["association_status"][0] == "observed"


def test_lookup_off_panel_hits_overflow(hybrid_store):
    q = query_store(hybrid_store)
    r = q.lookup([HG38_ALID_2], ["trait_a"])
    assert len(r["z"]) == 1
    assert r["z"][0] == pytest.approx(-5.0, rel=5e-3)


def test_off_panel_variant_absent_from_dense_matrix(hybrid_store):
    """The overflow variant must NOT occupy a dense row (disjoint partition)."""
    q = query_store(hybrid_store)
    variants = q.variants_table()
    off_shared = next(
        k for k, v in variants.items() if v["alid"] == HG38_ALID_2
    )
    dense_to_shared = np.load(hybrid_store / "dense" / "dense_to_shared.npy")
    assert off_shared not in dense_to_shared.tolist()


def test_phewas_off_panel(hybrid_store):
    q = query_store(hybrid_store)
    r = q.phewas(HG38_ALID_2)
    # Only trait_a observed the off-panel variant.
    assert len(r["z"]) == 1
    assert r["z"][0] == pytest.approx(-5.0, rel=5e-3)


def test_analysis_unions_both_components(hybrid_store):
    q = query_store(hybrid_store)
    r = q.analysis("trait_a")
    # trait_a: 2 dense (ALID_1, ALID_3) + 1 overflow (ALID_2) = 3
    assert len(r["z"]) == 3
    assert set(np.round(r["z"], 1).tolist()) == {-4.0, -5.0, 3.0}


def test_range_phewas_unions_components(hybrid_store):
    q = query_store(hybrid_store)
    # Range covering all three variants (1:100000 .. 1:1564620).
    r = q.range_phewas("1", 1, 2_000_000)
    variants = q.variants_table()
    alids = {variants[int(vi)]["alid"] for vi in r["variant_index"]}
    assert {HG38_ALID_1, HG38_ALID_2, HG38_ALID_3} <= alids


def test_top_hits_merges_components(hybrid_store):
    q = query_store(hybrid_store)
    r = q.top_hits(threshold=5e-4)
    variants = q.variants_table()
    alids = {variants[int(vi)]["alid"] for vi in r["variant_index"]}
    # The off-panel overflow hit (z=-5.0) must appear alongside dense hits.
    assert HG38_ALID_2 in alids
    assert list(zip(r["analysis_index"], r["variant_index"], strict=True)) == sorted(
        zip(r["analysis_index"], r["variant_index"], strict=True)
    )


def test_top_hits_reads_both_component_frequencies_from_indexes(hybrid_store):
    q = query_store(hybrid_store)
    expected = q.top_hits(threshold=5e-4)["eaf"]

    def fail_dense(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Dense top hits should use indexed frequencies")

    def fail_overflow(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Overflow top hits should use indexed frequencies")

    q._dense._eaf_pairs = fail_dense
    q._csr.eaf_pairs = fail_overflow
    result = q.top_hits(threshold=5e-4)
    np.testing.assert_array_equal(result["eaf"], expected)
    q.close()


def test_top_hits_selects_and_merges_one_analysis(hybrid_store):
    q = query_store(hybrid_store)
    global_result = q.top_hits(threshold=5e-4)
    selected = q.top_hits(analysis_id="trait_a", threshold=5e-4)
    expected = global_result["analysis_index"] == 0

    for name in ("variant_index", "analysis_index", "z", "se", "association_status"):
        np.testing.assert_array_equal(selected[name], global_result[name][expected])
    assert len(selected["z"]) == 2  # one Dense hit and one overflow hit
    assert q.top_hits(analysis_id="unknown", threshold=5e-4)["z"].size == 0


def test_top_hit_counts_sum_dense_and_overflow_components(hybrid_store):
    # trait_a: z=-4.0 (dense), z=-5.0 (overflow), z=3.0 (dense).
    #   5e-8: none pass (z_critical=5.45)          -> 0
    #   5e-6: only |5.0| passes (z_critical=4.56)  -> 1
    #   5e-4: |4.0| and |5.0| pass (z_critical=3.48) -> 2
    # trait_b: z=-12.0 (dense), z=4.0 (dense).
    #   5e-8/5e-6: only |12.0| passes -> 1
    #   5e-4: both pass -> 2
    rows = {r["analysis_id"]: r for r in read_analyses(hybrid_store / "analyses.tsv").rows}
    assert rows["trait_a"]["n_hits_5e8"] == "0"
    assert rows["trait_a"]["n_hits_5e6"] == "1"
    assert rows["trait_a"]["n_hits_5e4"] == "2"
    assert rows["trait_b"]["n_hits_5e8"] == "1"
    assert rows["trait_b"]["n_hits_5e6"] == "1"
    assert rows["trait_b"]["n_hits_5e4"] == "2"


# ── Validation (issue 059) ───────────────────────────────────────────────────


def test_validate_passes(hybrid_store):
    result = validate_store(hybrid_store)
    assert result.ok, result.errors


def test_validate_catches_imputed_overflow(hybrid_store):
    """An imputed array on the overflow must fail — the overflow is never imputed."""
    root = open_store(hybrid_store).arrays(mode="a")["ragged"]
    n = int(root["offsets"][:][-1])
    root.create_dataset("imputed", data=np.zeros(max(n, 1), dtype="uint8"))
    result = validate_store(hybrid_store)
    assert not result.ok
    assert any("overflow" in e.lower() and "imputed" in e.lower() for e in result.errors)


def test_validate_catches_disjoint_violation(hybrid_store):
    """Point an overflow entry at an on-panel variant → disjoint partition fails."""
    dense_to_shared = np.load(hybrid_store / "dense" / "dense_to_shared.npy")
    on_panel_shared = int(dense_to_shared[0])
    root = open_store(hybrid_store).arrays(mode="a")["ragged"]
    vi = root["variant_index"][:]
    if len(vi) == 0:
        pytest.skip("no overflow associations to corrupt")
    vi[0] = on_panel_shared
    root["variant_index"][:] = vi
    result = validate_store(hybrid_store)
    assert not result.ok
    assert any("disjoint" in e.lower() for e in result.errors)


def test_validate_catches_overflow_top_hit_offsets(hybrid_store):
    root = open_store(hybrid_store).arrays(mode="r+")
    offsets = root["top_hits/p_5e_04/analysis_offsets"][:]
    offsets[-1] -= 1
    root["top_hits/p_5e_04/analysis_offsets"][:] = offsets

    result = validate_store(hybrid_store)
    assert not result.ok
    assert any("invalid analysis offsets" in error for error in result.errors)


# --- GWAS-SSF source_reader_capability (issue #84) ---

_SSF_HEADER = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
]


def _write_ssf(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _SSF_HEADER) + "\n")


def test_gwas_ssf_capability_builds_a_hybrid_store_alongside_gwas_vcf(tmp_path):
    """issue #84: a manifest row whose source_reader_capability names the new
    GWAS-SSF reader builds through the same Hybrid pipeline as a GWAS-VCF row,
    with no source-format branching in build_hybrid_from_vcf_manifest itself
    -- trait_ssf (GWAS-SSF) and trait_b (GWAS-VCF, default capability) are
    built from the same manifest. Same dense/overflow/dense z pattern as
    `hybrid_store`'s trait_a, sourced from a filtered/harmonised GWAS-SSF
    file instead of a VCF.
    """
    ssf_path = tmp_path / "trait_ssf.tsv.gz"
    _write_ssf(
        ssf_path,
        [
            # effect_allele=G, other_allele=A -> A1=A, effect != A1 -> flip -> z=-4.0, dense
            {
                "chromosome": "1", "base_pair_location": HG19_POS_1,
                "effect_allele": "G", "other_allele": "A",
                "beta": 2.0, "standard_error": 0.5,
            },
            # effect_allele=T, other_allele=C -> A1=C, effect != A1 -> flip -> z=-5.0, overflow
            {
                "chromosome": "1", "base_pair_location": HG19_POS_2,
                "effect_allele": "T", "other_allele": "C",
                "beta": 1.5, "standard_error": 0.3,
            },
            # effect_allele=A, other_allele=G -> A1=A, effect == A1 -> no flip -> z=3.0, dense
            {
                "chromosome": "1", "base_pair_location": HG19_POS_3,
                "effect_allele": "A", "other_allele": "G",
                "beta": 0.6, "standard_error": 0.2,
            },
        ],
    )
    vcf_b = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
        ],
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\n"
        f"trait_ssf\t{ssf_path}\tTrait SSF\t1000\tsd\tdeclared_standardised\t\t{GWAS_SSF_CAPABILITY}\n"
        f"trait_b\t{vcf_b}\tTrait B\t1000\tsd\tdeclared_standardised\t\t\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store_ssf.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path),
        store_id="hybrid-ssf-test", release_id="v1",
    )

    result = validate_store(store_path)
    assert result.ok, result.errors

    q = query_store(store_path)
    on_panel = q.lookup([HG38_ALID_1], ["trait_ssf"])
    off_panel = q.lookup([HG38_ALID_2], ["trait_ssf"])
    analysis = q.analysis("trait_ssf")
    q.close()

    assert on_panel["z"][0] == pytest.approx(-4.0, rel=5e-3)
    assert off_panel["z"][0] == pytest.approx(-5.0, rel=5e-3)
    assert set(np.round(analysis["z"], 1).tolist()) == {-4.0, -5.0, 3.0}


# --- source_assembly (issue #85): per-row declared source genome build ---


def _manifest_with_source_assembly(
    tmp_path: Path, entries: list[tuple[str, Path, str, str]]
) -> Path:
    """Like `_make_manifest`, plus a `source_assembly` column per entry."""
    manifest = tmp_path / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_assembly"
    ]
    for trait_id, file_path, trait_name, source_assembly in entries:
        lines.append(
            f"{trait_id}\t{file_path}\t{trait_name}\t1000\tsd\tdeclared_standardised\t"
            f"\t{source_assembly}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_hg38_source_assembly_is_not_lifted_in_hybrid_build(tmp_path):
    """issue #85: a row declaring source_assembly=hg38 passes through with no
    liftover in the Hybrid builder too -- HG19_POS_2 (1,000,000) is a
    position the real hg19->hg38 chain shifts to 1,064,620 (HG38_ALID_2); if
    it were lifted a second time despite the hg38 declaration, the store
    would show that shifted position instead of the source file's own
    1,000,000. This is the exact bug the real GWAS-Catalog-SSF reproduction
    in issue #85 found: an already-hg38 harmonised source silently re-lifted.
    """
    vcf = _make_vcf(
        tmp_path, "trait_ssf",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],  # z=5.0, flip->-5.0
    )
    manifest = _manifest_with_source_assembly(tmp_path, [("trait_ssf", vcf, "Trait SSF", "hg38")])
    store_path = tmp_path / "store.opengwasdb"

    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path),
        store_id="hybrid-hg38-test", release_id="v1",
    )

    q = query_store(store_path)
    result = q.analysis("trait_ssf")
    vt = q.variants_table()
    q.close()

    assert len(result["z"]) == 1
    variant = vt[int(result["variant_index"][0])]
    assert variant["position"] == HG19_POS_2
    assert result["z"][0] == pytest.approx(-5.0, rel=5e-3)


def test_mixed_hg19_and_hg38_manifest_builds_hybrid_store(tmp_path):
    """issue #85: a Hybrid build mixing a default (hg19) row and an
    explicitly-hg38 row lifts only the hg19 row -- the scenario the bug
    report specifically named (a GWAS-VCF row alongside a harmonised
    GWAS-SSF row in one build)."""
    vcf_hg19 = _make_vcf(
        tmp_path, "trait_vcf",
        [f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n"],  # lifted -> 1564620
    )
    vcf_hg38 = _make_vcf(
        tmp_path, "trait_ssf",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],  # not lifted; stays 1000000
    )
    manifest = _manifest_with_source_assembly(
        tmp_path,
        [("trait_vcf", vcf_hg19, "Trait VCF", ""), ("trait_ssf", vcf_hg38, "Trait SSF", "hg38")],
    )
    store_path = tmp_path / "store.opengwasdb"

    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=_panel(tmp_path),
        store_id="hybrid-mixed-test", release_id="v1",
    )

    q = query_store(store_path)
    vt = q.variants_table()
    vcf_result = q.analysis("trait_vcf")
    ssf_result = q.analysis("trait_ssf")
    q.close()

    assert vt[int(vcf_result["variant_index"][0])]["position"] == 1_564_620
    assert vt[int(ssf_result["variant_index"][0])]["position"] == HG19_POS_2


def test_analyses_table_reports_shared_not_dense_only_hit_counts(hybrid_store):
    """`analyses_table()` must report the shared table's Top-Hit Counts, which
    include the Ragged Overflow Component's hits, not the Dense Component's
    panel-local counts (issue #107).

    The two files legitimately differ: `add_hit_counts()` is applied once
    against the Dense Component's own index and again against the whole
    store's, so the Dense Component's own `analyses.tsv` counts only on-panel
    hits. Delegating to it silently undercounts every Analysis with off-panel
    hits -- on the `gwas-catalog-eur-hybrid` pilot that hides 4,476 real hits,
    one Analysis by 27%.
    """
    shared = read_analyses(hybrid_store / "analyses.tsv").rows
    dense_only = read_analyses(hybrid_store / "dense" / "analyses.tsv").rows
    shared_by_id = {r["analysis_id"]: r for r in shared}
    dense_by_id = {r["analysis_id"]: r for r in dense_only}

    # trait_a's middle variant is off-panel (routed to the Overflow) with
    # z = -5.0 -> p ~ 5.7e-7, so it clears 5e-6/5e-4 but not 5e-8. That makes
    # the shared and dense-only counts genuinely differ for this fixture --
    # without which this test could pass on a store where the bug is invisible.
    assert shared_by_id["trait_a"]["n_hits_5e6"] != dense_by_id["trait_a"]["n_hits_5e6"]

    q = query_store(hybrid_store)
    try:
        table = q.analyses_table()
    finally:
        q.close()

    by_id = {row["analysis_id"]: row for row in table.values()}
    for analysis_id, shared_row in shared_by_id.items():
        for column in ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
            assert by_id[analysis_id][column] == shared_row[column], (
                f"{analysis_id}.{column}: analyses_table() returned "
                f"{by_id[analysis_id][column]!r} (the Dense Component's "
                f"panel-local count) rather than the shared table's "
                f"{shared_row[column]!r}"
            )
