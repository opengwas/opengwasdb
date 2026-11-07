"""An insertion and a deletion at one position are two variants (issue #381).

FinnGen r13 BMI_IRN reports `1:827587 G>GGGGCC` and `GGGGCC>G`, two alleles of a
repeat. A sorted ALID spelt both `1:827587:G:GGGGCC`, so Association Admission
kept the insertion and dropped the deletion as `duplicate_reversed_indel`
(#357). An indel's ALID now spells its reference allele first (ADR 0079): the
pair is `1:827587:G:GGGGCC` and `1:827587:GGGGCC:G`, and a build that reads a
source declaring its reference stores both rows: a GWAS-VCF or FinnGen source,
or a GWAS-SSF row whose harmoniser code is 10 or 11 in the layout that holds
harmonised alleles in its main columns. A source that declares no reference
cannot place either, so both are dropped and logged, under a reason that says
why: `indel_reference_undeclared`, or the GWAS-SSF layout's own.
"""

from __future__ import annotations

from pathlib import Path

import besd_fixtures
import pytest
from admission_helpers import logged, ragged_ssf_store, record_analyses

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.query import query_store
from opengwasdb.readers.column_map import load_column_map
from opengwasdb.variants.axis import iter_variant_records

#: Each reader of a format, and the Column Map that reads it as that reader does.
FINNGEN = [
    pytest.param("opengwasdb.finngen-r13", None, id="finngen-reader"),
    pytest.param("opengwasdb.column-map", "finngen-r13", id="finngen-map"),
]

#: ``(chrom, pos, ref, alt, beta, se, af_alt)`` as BMI_IRN reports them; the
#: effect is the `alt` allele's.
INSERTION = ("1", "827587", "G", "GGGGCC", "-0.00115727", "0.0135908", "0.00828552")
DELETION = ("1", "827587", "GGGGCC", "G", "0.0226233", "0.05058", "0.000487289")
#: An SNV elsewhere, so the Store holds a variant whatever the pair resolves to.
CONTROL = ("1", "900000", "A", "G", "0.2", "0.1", "0.3")
INSERTION_ALID = "1:827587:G:GGGGCC"
DELETION_ALID = "1:827587:GGGGCC:G"
CONTROL_ALID = "1:900000:A:G"
#: Each indel's z on its ALID's alleles. Both are signed for their reference
#: allele, A1, and FinnGen's beta is for the alt, so both are negated.
INSERTION_Z = 0.00115727 / 0.0135908
DELETION_Z = -0.0226233 / 0.05058
CONTROL_Z = -2.0
#: Within the z code's resolution.
Z_RESOLUTION = 1e-3


def _rows(deletion_first: bool) -> list[tuple[str, ...]]:
    pair = [DELETION, INSERTION] if deletion_first else [INSERTION, DELETION]
    return [*pair, CONTROL]


def _finngen(path: Path, rows: list[tuple[str, ...]]) -> Path:
    lines = ["#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt"]
    lines += [f"{c}\t{p}\t{r}\t{a}\t\t\t0.5\t0.3\t{b}\t{s}\t{f}" for c, p, r, a, b, s, f in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _gwas_vcf(path: Path, rows: list[tuple[str, ...]]) -> Path:
    import test_dense_vcf_build as dense_fixture

    path.write_text(
        dense_fixture._vcf_header("Continuous")
        + "".join(
            f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\tES:SE:AF\t{b}:{s}:{f}\n"
            for c, p, r, a, b, s, f in rows
        ),
        encoding="utf-8",
    )
    return path


def _gwas_ssf(path: Path, rows: list[tuple[str, ...]], hm_code: str = "11") -> Path:
    lines = [
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
        "\tstandard_error\teffect_allele_frequency\tp_value\thm_code"
    ]
    lines += [f"{c}\t{p}\t{a}\t{r}\t{b}\t{s}\t{f}\t0.5\t{hm_code}" for c, p, r, a, b, s, f in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _manifest(tmp_path: Path, source: Path, capability: str) -> Path:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"bmi_irn\t{source}\tBMI_IRN\t500000\tsd\tdeclared_standardised\t\t{capability}"
        "\tGRCh38\n",
        encoding="utf-8",
    )
    return manifest


def _z_by_alid(store: Path) -> dict[str, float]:
    query = query_store(store)
    try:
        result = query.analysis("bmi_irn")
        variants = query.variants_table()
    finally:
        query.close()
    return {
        variants[int(index)]["alid"]: float(z)
        for index, z in zip(result["variant_index"], result["z"], strict=True)
    }


def _dense(
    tmp_path: Path, source: Path, capability: str, column_map: str | None = None
) -> tuple[dict[str, float], dict]:
    """A Dense Store of one Analysis from ``source``: its z by ALID, and its record."""
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        _manifest(tmp_path, source, capability), store, store_id="s", release_id="r",
        column_map=None if column_map is None else load_column_map(column_map),
    )
    (analysis,) = record_analyses(store)
    return _z_by_alid(store), analysis


def _looked_up(store: Path, analysis_id: str) -> dict[str, float]:
    """Each indel's z, looked up by its ALID."""
    with query_store(store) as q:
        return {
            alid: float(q.lookup([alid], [analysis_id])["z"][0])
            for alid in (INSERTION_ALID, DELETION_ALID)
        }


def _logged(store: Path) -> list[list[str]]:
    return logged(admission_record_for(store), 0)[1:]


def _assert_two_variants(z_by_alid: dict[str, float], analysis: dict) -> None:
    assert z_by_alid == pytest.approx(
        {INSERTION_ALID: INSERTION_Z, DELETION_ALID: DELETION_Z, CONTROL_ALID: CONTROL_Z},
        abs=Z_RESOLUTION,
    )
    assert analysis["dropped"] == {}
    assert (analysis["examined"], analysis["admitted"]) == (3, 3)


@pytest.mark.parametrize(("capability", "column_map"), FINNGEN)
@pytest.mark.parametrize("deletion_first", [True, False], ids=["deletion-first", "insertion-first"])
def test_a_dense_build_stores_finngen_s_insertion_and_deletion_as_two_variants(
    tmp_path: Path, deletion_first: bool, capability: str, column_map: str | None
):
    source = _finngen(tmp_path / "bmi_irn.tsv", _rows(deletion_first))

    z_by_alid, analysis = _dense(tmp_path, source, capability, column_map)

    _assert_two_variants(z_by_alid, analysis)


@pytest.mark.parametrize("deletion_first", [True, False], ids=["deletion-first", "insertion-first"])
def test_a_dense_vcf_build_stores_the_pair_as_two_variants(tmp_path: Path, deletion_first: bool):
    source = _gwas_vcf(tmp_path / "bmi_irn.vcf", _rows(deletion_first))

    z_by_alid, analysis = _dense(tmp_path, source, "opengwasdb.gwas-vcf")

    _assert_two_variants(z_by_alid, analysis)


@pytest.mark.parametrize("on_panel", [True, False], ids=["dense-component", "overflow"])
def test_a_hybrid_build_stores_the_pair_as_two_variants(tmp_path: Path, on_panel: bool):
    """On the Dense Component's axis, or in the Ragged Overflow, two rows."""
    source = _gwas_vcf(tmp_path / "bmi_irn.vcf", _rows(deletion_first=True))
    panel = tmp_path / "panel.txt"
    alids = [INSERTION_ALID, DELETION_ALID, CONTROL_ALID] if on_panel else [CONTROL_ALID]
    panel.write_text("".join(f"{alid}\n" for alid in alids), encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"

    result = build_hybrid_from_vcf_manifest(
        _manifest(tmp_path, source, "opengwasdb.gwas-vcf"), store, reference_panel=panel,
        store_id="s", release_id="r",
    )

    assert result.n_overflow == (0 if on_panel else 2)
    z_by_alid = _looked_up(store, "bmi_irn")
    assert z_by_alid == pytest.approx(
        {INSERTION_ALID: INSERTION_Z, DELETION_ALID: DELETION_Z}, abs=Z_RESOLUTION
    )
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {}


def _ssf_rows(rows: list[tuple[str, ...]], hm_code: str | None = None) -> list[dict]:
    """Filtered GWAS-SSF rows, with a harmoniser code when one is given."""
    code = {} if hm_code is None else {"hm_code": hm_code}
    return [
        {"chromosome": c, "base_pair_location": int(p), "effect_allele": a,
         "other_allele": r, "beta": float(b), "standard_error": float(s),
         "effect_allele_frequency": float(f), **code}
        for c, p, r, a, b, s, f in rows
    ]


@pytest.mark.parametrize("hm_code", ["10", "11"])
def test_a_ragged_build_stores_a_harmonised_pair_as_two_variants(tmp_path: Path, hm_code: str):
    """A main-column GWAS-SSF row whose harmoniser code is 10 or 11 declares its
    `other_allele` the reference, so a Ragged SSF Store holds the pair as two
    variants, and its record counts both as kept by the code."""
    store, _ = ragged_ssf_store(tmp_path, _ssf_rows(_rows(deletion_first=True), hm_code))

    alids = [record.alid for record in iter_variant_records(store / "variants.tsv.gz")]
    assert sorted(alids) == sorted([INSERTION_ALID, DELETION_ALID, CONTROL_ALID])
    z_by_alid = _looked_up(store, "trait_a")
    assert z_by_alid == pytest.approx(
        {INSERTION_ALID: INSERTION_Z, DELETION_ALID: DELETION_Z}, abs=Z_RESOLUTION
    )
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {}
    assert analysis["indels_kept_by_hm_code"] == 2


@pytest.mark.parametrize("hm_code", ["10", "11"])
def test_a_dense_build_of_a_harmonised_gwas_ssf_stores_the_pair(tmp_path: Path, hm_code: str):
    source = _gwas_ssf(tmp_path / "bmi_irn.tsv", _rows(deletion_first=True), hm_code)

    z_by_alid, analysis = _dense(tmp_path, source, "opengwasdb.gwas-ssf")

    _assert_two_variants(z_by_alid, analysis)
    assert analysis["indels_kept_by_hm_code"] == 2


def _undeclared_log(reason: str) -> list[list[str]]:
    """How each undeclared indel of the pair is logged: by source key, the
    source's `ref` then `alt`, in source order."""
    return [[DELETION_ALID, reason], [INSERTION_ALID, reason]]


UNDECLARED_LOG = _undeclared_log("indel_reference_undeclared")


def _assert_pair_dropped(store: Path, analysis: dict, log: list[list[str]]) -> None:
    alids = [record.alid for record in iter_variant_records(store / "variants.tsv.gz")]
    assert alids == [CONTROL_ALID]
    (reason,) = {reason for _, reason in log}
    assert analysis["dropped"] == {reason: 2}
    assert (analysis["examined"], analysis["admitted"]) == (3, 1)
    assert analysis["indels_kept_by_hm_code"] == 0
    assert _logged(store) == log


@pytest.mark.parametrize("hm_code", ["12", "13", "NA"])
def test_a_gwas_ssf_pair_another_harmoniser_code_names_is_dropped(tmp_path: Path, hm_code: str):
    source = _gwas_ssf(tmp_path / "bmi_irn.tsv", _rows(deletion_first=True), hm_code)

    _z, analysis = _dense(tmp_path, source, "opengwasdb.gwas-ssf")

    _assert_pair_dropped(
        tmp_path / "store.opengwasdb", analysis, _undeclared_log("indel_ssf_hm_code")
    )


def test_a_gwas_ssf_pair_beside_hm_allele_columns_is_dropped(tmp_path: Path):
    """The main columns of such a file are the study's, so none of its indels is
    placed, whatever its code (owner, 7 Nov)."""
    lines = [
        "hm_other_allele\thm_effect_allele\thm_code\tchromosome\tbase_pair_location"
        "\teffect_allele\tother_allele\tbeta\tstandard_error\teffect_allele_frequency\tp_value"
    ]
    lines += [
        f"{r}\t{a}\t10\t{c}\t{p}\t{a}\t{r}\t{b}\t{s}\t{f}\t0.5"
        for c, p, r, a, b, s, f in _rows(deletion_first=True)
    ]
    source = tmp_path / "bmi_irn.tsv"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _z, analysis = _dense(tmp_path, source, "opengwasdb.gwas-ssf")

    _assert_pair_dropped(
        tmp_path / "store.opengwasdb", analysis, _undeclared_log("indel_ssf_hm_columns")
    )


def test_a_gwas_ssf_column_map_declares_no_reference_whatever_the_code(tmp_path: Path):
    """A Column Map's translation declares by its allele convention: GWAS-SSF's
    effect and other alleles name none."""
    source = _gwas_ssf(tmp_path / "bmi_irn.tsv", _rows(deletion_first=True), "10")

    _z, analysis = _dense(tmp_path, source, "opengwasdb.column-map", "gwas-ssf")

    _assert_pair_dropped(tmp_path / "store.opengwasdb", analysis, UNDECLARED_LOG)


def test_a_ragged_ssf_build_drops_and_logs_an_unharmonised_pair(tmp_path: Path):
    store, _ = ragged_ssf_store(tmp_path, _ssf_rows(_rows(deletion_first=True)))

    (analysis,) = record_analyses(store)
    _assert_pair_dropped(store, analysis, _undeclared_log("indel_ssf_unharmonised"))


def test_a_build_dense_source_drops_and_logs_the_pair(tmp_path: Path):
    """An association TSV's effect and other alleles declare no reference."""
    from dense_sources import build_dense_fixture

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tbeta\tse\n"
        + "".join(
            f"A\t{c}\t{p}\t{a}\t{r}\t{b}\t{s}\n" for c, p, r, a, b, s, _f in _rows(True)
        ),
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_fixture(
        [source], store, store_id="s", release_id="r", reference_assembly="GRCh38"
    )

    (analysis,) = record_analyses(store)
    _assert_pair_dropped(store, analysis, UNDECLARED_LOG)


def test_a_besd_build_drops_and_logs_each_probe_s_rows_at_the_pair(tmp_path: Path):
    """A `.esi` row's A1 and A2 declare no reference. Each probe's row at either
    indel is its own drop, named by `.esi` row."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    snps: list[besd_fixtures.EsiRow] = [
        {"chr": c, "snp_id": f"rs{i}", "bp": int(p), "a1": a, "a2": r}
        for i, (c, p, r, a, _b, _s, _f) in enumerate(_rows(deletion_first=True))
    ]
    besd_fixtures.write_esi(fixture / "test.esi", snps)
    besd_fixtures.write_epi(
        fixture / "test.epi", [{"chr": "1", "probe_id": "ENSG0", "bp": 900000}]
    )
    besd_fixtures.write_besd_sparse_3f(
        fixture / "test.besd", 1, [[(i, 0.1, 0.02) for i in range(3)]]
    )
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        fixture / "test", store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )

    (analysis,) = record_analyses(store)
    _assert_pair_dropped(
        store, analysis,
        [["esi:0", "indel_reference_undeclared"], ["esi:1", "indel_reference_undeclared"]],
    )

