"""D5 for reversed indels (issue #357), on FinnGen r13 BMI_IRN's own pair.

`1:827587 G>GGGGCC` and `GGGGCC>G` are two alleles of a repeat, and their
sorted alleles make one ALID, `1:827587:G:GGGGCC`. FinnGen declares `ref` the
reference allele, so Association Admission keeps the row whose reference the
ALID spells first, the insertion, and drops the deletion as
`duplicate_reversed_indel`. GWAS-SSF names an effect and an other allele,
neither declared the reference, so both rows are dropped as conflicting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from admission_helpers import logged, record_analyses

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.query import query_store
from opengwasdb.readers.column_map import load_column_map

#: Each reader of a format, and the Column Map that reads it as that reader does.
FINNGEN = [
    pytest.param("opengwasdb.finngen-r13", None, id="finngen-reader"),
    pytest.param("opengwasdb.column-map", "finngen-r13", id="finngen-map"),
]
GWAS_SSF = [
    pytest.param("opengwasdb.gwas-ssf", None, id="gwas-ssf-reader"),
    pytest.param("opengwasdb.column-map", "gwas-ssf", id="gwas-ssf-map"),
]

#: ``(chrom, pos, ref, alt, beta, se, af_alt)`` as BMI_IRN reports them; the
#: effect is the `alt` allele's.
INSERTION = ("1", "827587", "G", "GGGGCC", "-0.00115727", "0.0135908", "0.00828552")
DELETION = ("1", "827587", "GGGGCC", "G", "0.0226233", "0.05058", "0.000487289")
#: An SNV elsewhere, so the Store holds a variant whatever the pair resolves to.
CONTROL = ("1", "900000", "A", "G", "0.2", "0.1", "0.3")
ALID = "1:827587:G:GGGGCC"
#: The insertion's z on the ALID's alleles: its effect allele, `GGGGCC`, sorts
#: second, so the stored z is the source's negated.
INSERTION_Z = 0.00115727 / 0.0135908


def _finngen(path: Path, rows: list[tuple[str, ...]]) -> Path:
    lines = ["#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt"]
    lines += [f"{c}\t{p}\t{r}\t{a}\t\t\t0.5\t0.3\t{b}\t{s}\t{f}" for c, p, r, a, b, s, f in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _gwas_ssf(path: Path, rows: list[tuple[str, ...]]) -> Path:
    lines = [
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
        "\tstandard_error\teffect_allele_frequency\tp_value"
    ]
    lines += [f"{c}\t{p}\t{a}\t{r}\t{b}\t{s}\t{f}\t0.5" for c, p, r, a, b, s, f in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build(
    tmp_path: Path, source: Path, capability: str, column_map: str | None
) -> tuple[dict[str, float], dict]:
    """A Dense Store of one Analysis from ``source``: its z by ALID, and its record."""
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"bmi_irn\t{source}\tBMI_IRN\t500000\tsd\tdeclared_standardised\t\t{capability}"
        "\tGRCh38\n",
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r",
        column_map=None if column_map is None else load_column_map(column_map),
    )
    query = query_store(store)
    try:
        result = query.analysis("bmi_irn")
        variants = query.variants_table()
    finally:
        query.close()
    z_by_alid = {
        variants[int(index)]["alid"]: float(z)
        for index, z in zip(result["variant_index"], result["z"], strict=True)
    }
    (analysis,) = record_analyses(store)
    return z_by_alid, analysis


def _logged(store: Path) -> list[list[str]]:
    return logged(admission_record_for(store), 0)[1:]


@pytest.mark.parametrize(("capability", "column_map"), FINNGEN)
@pytest.mark.parametrize("deletion_first", [True, False], ids=["deletion-first", "insertion-first"])
def test_finngen_keeps_the_insertion_of_a_reversed_indel(
    tmp_path: Path, deletion_first: bool, capability: str, column_map: str | None
):
    pair = [DELETION, INSERTION] if deletion_first else [INSERTION, DELETION]
    source = _finngen(tmp_path / "bmi_irn.tsv", [*pair, CONTROL])

    z_by_alid, analysis = _build(tmp_path, source, capability, column_map)

    assert set(z_by_alid) == {ALID, "1:900000:A:G"}
    # Within the z code's resolution; the deletion's z is 0.447.
    assert z_by_alid[ALID] == pytest.approx(INSERTION_Z, abs=1e-3)
    assert analysis["dropped"] == {"duplicate_reversed_indel": 1}
    assert (analysis["examined"], analysis["admitted"]) == (3, 2)
    assert _logged(tmp_path / "store.opengwasdb") == [[ALID, "duplicate_reversed_indel"]]


@pytest.mark.parametrize(("capability", "column_map"), GWAS_SSF)
def test_gwas_ssf_declares_no_reference_so_the_pair_is_dropped(
    tmp_path: Path, capability: str, column_map: str | None
):
    source = _gwas_ssf(tmp_path / "bmi_irn.tsv", [DELETION, INSERTION, CONTROL])

    z_by_alid, analysis = _build(tmp_path, source, capability, column_map)

    assert set(z_by_alid) == {"1:900000:A:G"}
    assert analysis["dropped"] == {"duplicate_conflicting": 2}
    assert _logged(tmp_path / "store.opengwasdb") == [
        [ALID, "duplicate_conflicting"], [ALID, "duplicate_conflicting"],
    ]
