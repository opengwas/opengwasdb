"""An indel's ALID spells its reference allele first (issue #381, ADR 0079).

FinnGen R13 BMI_IRN reports two variants at `1:827587`: the insertion
`G>GGGGCC` and the deletion `GGGGCC>G`. A sorted ALID spelt both
`1:827587:G:GGGGCC`, so the deletion read as the insertion flipped. Written
reference first, they are two ALIDs. SNVs, and substitutions whose two alleles
are one length, stay sorted.
"""

from __future__ import annotations

import pytest

from opengwasdb.admission.rules import LOGGED, Reason
from opengwasdb.readers.tabular import NO_REFERENCE
from opengwasdb.variants.normalise import (
    IdentityFaults,
    MalformedIdentity,
    UndeclaredReference,
    VariantNormalisationError,
    canonical_alid,
    identity_fault,
    orient_or_drop,
    orient_to_canonical,
)

INSERTION = ("G", "GGGGCC")
DELETION = ("GGGGCC", "G")


def test_a_deletion_is_spelt_reference_first() -> None:
    assert canonical_alid("1", 827587, *DELETION) == "1:827587:GGGGCC:G"


def test_an_insertion_keeps_its_spelling() -> None:
    assert canonical_alid("1", 827587, *INSERTION) == "1:827587:G:GGGGCC"


def test_the_two_alleles_of_a_repeat_are_two_alids() -> None:
    assert canonical_alid("1", 827587, *INSERTION) != canonical_alid("1", 827587, *DELETION)


#: Allele pairs of every shape: SNVs, substitutions, insertions and deletions
#: (anchored or not), each both ways round.
PAIRS = [
    (ref, alt)
    for one, other in [
        ("A", "G"), ("C", "T"), ("AC", "GT"), ("ACG", "TTT"), ("G", "GGGGCC"),
        ("T", "TA"), ("A", "CT"), ("GAT", "C"), ("AT", "ATTT"),
    ]
    for ref, alt in [(one, other), (other, one)]
]


def test_the_vectorised_alid_rule_gives_alid_alleles_answer() -> None:
    import numpy as np

    from opengwasdb.variants.normalise import alid_allele_arrays, alid_alleles

    ref = np.array([r for r, _a in PAIRS], dtype=object)
    alt = np.array([a for _r, a in PAIRS], dtype=object)

    first, second, indel = alid_allele_arrays(ref, alt)

    assert list(zip(first.tolist(), second.tolist(), strict=True)) == [
        alid_alleles(r, a) for r, a in PAIRS
    ]
    assert indel.tolist() == [len(r) != len(a) for r, a in PAIRS]


@pytest.mark.parametrize(
    ("ref", "alt", "alid"),
    [
        ("G", "A", "1:100:A:G"),
        ("A", "G", "1:100:A:G"),
        ("GT", "AC", "1:100:AC:GT"),
        ("ac", "GT", "1:100:AC:GT"),
    ],
)
def test_alleles_of_one_length_stay_sorted(ref: str, alt: str, alid: str) -> None:
    assert canonical_alid("1", 100, ref, alt) == alid


def test_a_declared_deletion_is_signed_for_its_reference_allele() -> None:
    # FinnGen's effect allele is its alt: the deletion's beta is the effect of
    # `G`, and the ALID's A1 is the reference `GGGGCC`, so the row is flipped.
    ref, alt = DELETION
    orientation = orient_to_canonical("1", 827587, alt, ref, other_is_reference=True)
    assert orientation.variant.alid == "1:827587:GGGGCC:G"
    assert orientation.flipped


def test_a_declared_insertion_is_signed_as_before() -> None:
    ref, alt = INSERTION
    orientation = orient_to_canonical("1", 827587, alt, ref, other_is_reference=True)
    assert orientation.variant.alid == "1:827587:G:GGGGCC"
    assert orientation.flipped


@pytest.mark.parametrize("declared", [True, False])
def test_an_snv_orients_alike_whether_or_not_its_source_declares_a_reference(
    declared: bool,
) -> None:
    orientation = orient_to_canonical("1", 100, "A", "G", other_is_reference=declared)
    assert (orientation.variant.alid, orientation.flipped) == ("1:100:A:G", False)
    orientation = orient_to_canonical("1", 100, "G", "A", other_is_reference=declared)
    assert (orientation.variant.alid, orientation.flipped) == ("1:100:A:G", True)


@pytest.mark.parametrize(("ref", "alt"), [INSERTION, DELETION])
def test_an_indel_whose_source_declares_no_reference_names_no_alid(ref: str, alt: str) -> None:
    with pytest.raises(UndeclaredReference) as raised:
        orient_to_canonical("1", 827587, alt, ref, other_is_reference=False)
    assert isinstance(raised.value, MalformedIdentity)
    assert raised.value.reason == "indel_reference_undeclared"


def test_an_undeclared_indel_is_recorded_as_an_identity_fault() -> None:
    faults = IdentityFaults()
    ref, alt = DELETION
    assert orient_or_drop("1", 827587, alt, ref, faults, declaration=NO_REFERENCE, row={}) is None
    assert faults.rows == [("1:827587:GGGGCC:G", "indel_reference_undeclared")]


def test_identity_fault_names_an_undeclared_indel_after_the_alleles() -> None:
    assert identity_fault("1", "G", "GGGGCC", other_is_reference=True) is None
    fault = identity_fault("1", "G", "GGGGCC", other_is_reference=False)
    assert isinstance(fault, UndeclaredReference)
    # An allele outside the alphabet is judged first: the row names no allele pair.
    invalid = identity_fault("1", "G", "GN", other_is_reference=False)
    assert invalid is not None and invalid.reason == "allele_invalid"
    assert identity_fault("1", "A", "G", other_is_reference=False) is None


def test_admission_logs_an_undeclared_indel_under_its_own_reason() -> None:
    reason = Reason(UndeclaredReference.reason)
    assert reason in LOGGED
    order = list(Reason)
    assert order.index(Reason.ALLELE_INVALID) < order.index(reason) < order.index(
        Reason.SE_UNUSABLE
    )


def test_one_allele_spelt_twice_is_still_refused() -> None:
    with pytest.raises(VariantNormalisationError):
        orient_to_canonical("1", 100, "GA", "ga", other_is_reference=True)


# --- Every Source Reader, on BMI_IRN's pair ---------------------------------

#: ``(chrom, pos, ref, alt, beta)``, beta for `alt`, SE 0.5: the insertion and
#: the deletion at `1:827587`, an SNV whose `alt` sorts first, and a two-base
#: substitution.
PAIR_ROWS = [
    ("1", 827587, "G", "GGGGCC", -0.1),
    ("1", 827587, "GGGGCC", "G", 0.2),
    ("1", 1000, "G", "A", 0.3),
    ("1", 2000, "AC", "GT", 0.4),
]
#: Each row's ALID and stored z when its source declares `ref` the reference:
#: an indel is signed for its reference allele, so its `alt` beta is negated.
DECLARED = [
    ("1:827587:G:GGGGCC", 0.2),
    ("1:827587:GGGGCC:G", -0.4),
    ("1:1000:A:G", 0.6),
    ("1:2000:AC:GT", -0.8),
]
#: What a source that declares no reference yields.
UNDECLARED = DECLARED[2:]


def _faults(reason: str) -> list[list[str]]:
    """The pair's two indels, each named by its source key and ``reason``."""
    return [["1:827587:G:GGGGCC", reason], ["1:827587:GGGGCC:G", reason]]


UNDECLARED_FAULTS = _faults("indel_reference_undeclared")


def _finngen_text() -> str:
    lines = ["#chrom\tpos\tref\talt\trsids\tpval\tbeta\tsebeta\taf_alt"]
    lines += [f"{c}\t{p}\t{r}\t{a}\t\t0.5\t{b}\t0.5\t0.3" for c, p, r, a, b in PAIR_ROWS]
    return "\n".join(lines) + "\n"


def _ssf_text(code: str | None) -> str:
    """A GWAS-SSF file in the main-column layout, every row harmoniser ``code``,
    or with no harmoniser columns at all when ``code`` is None."""
    header = "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error"
    header += "\teffect_allele_frequency"
    tail = "" if code is None else f"\t{code}"
    lines = [header + ("" if code is None else "\thm_code")]
    lines += [f"{c}\t{p}\t{a}\t{r}\t{b}\t0.5\t0.3{tail}" for c, p, r, a, b in PAIR_ROWS]
    return "\n".join(lines) + "\n"


def _ssf_hm_columns_text(code: str = "10") -> str:
    """A GWAS-SSF file in the layout that keeps the study's alleles in the main
    columns and the harmonised ones in `hm_*` columns beside them."""
    lines = [
        "hm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_code"
        "\tchromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
        "\tstandard_error\teffect_allele_frequency"
    ]
    lines += [
        f"{c}\t{p}\t{r}\t{a}\t{b}\t{code}\t{c}\t{p}\t{a}\t{r}\t{b}\t0.5\t0.3"
        for c, p, r, a, b in PAIR_ROWS
    ]
    return "\n".join(lines) + "\n"


def _vcf_text() -> str:
    import test_dense_vcf_build as dense_fixture

    return dense_fixture._vcf_header("Continuous") + "".join(
        f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\tES:SE:AF\t{b}:0.5:0.3\n"
        for c, p, r, a, b in PAIR_ROWS
    )


def _association_tsv_text() -> str:
    lines = ["chromosome\tposition\teffect_allele\tother_allele\tbeta\tse\teaf"]
    lines += [f"{c}\t{p}\t{a}\t{r}\t{b}\t0.5\t0.3" for c, p, r, a, b in PAIR_ROWS]
    return "\n".join(lines) + "\n"


def _reader(tmp_path, kind: str):
    from opengwasdb.model.enums import StoredEffectScale
    from opengwasdb.readers.column_map import load_column_map
    from opengwasdb.readers.registry import resolve_reader

    texts = {
        "finngen": ("finngen.tsv", _finngen_text(), "opengwasdb.finngen-r13", None),
        "finngen-map": ("finngen.tsv", _finngen_text(), "opengwasdb.column-map", "finngen-r13"),
        "gwas-vcf": ("trait.vcf", _vcf_text(), "opengwasdb.gwas-vcf", None),
        "gwas-ssf-code-11": ("ssf.tsv", _ssf_text("11"), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-code-10": ("ssf.tsv", _ssf_text("10"), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-code-12": ("ssf.tsv", _ssf_text("12"), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-code-missing": ("ssf.tsv", _ssf_text("NA"), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-hm-columns": ("ssf.tsv", _ssf_hm_columns_text(), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-unharmonised": ("ssf.tsv", _ssf_text(None), "opengwasdb.gwas-ssf", None),
        "gwas-ssf-map": ("ssf.tsv", _ssf_text("11"), "opengwasdb.column-map", "gwas-ssf"),
        "association-tsv": ("records.tsv", _association_tsv_text(),
                            "opengwasdb.association-tsv", None),
    }
    name, text, capability, column_map = texts[kind]
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return resolve_reader(
        capability, path, StoredEffectScale.SD,
        column_map=None if column_map is None else load_column_map(column_map),
    )


@pytest.mark.parametrize(
    ("kind", "expected", "faults", "kept_by_hm_code"),
    [
        ("finngen", DECLARED, [], 0),
        ("finngen-map", DECLARED, [], 0),
        ("gwas-vcf", DECLARED, [], 0),
        # In the main-column layout, harmoniser codes 10 and 11 declare
        # `other_allele` the reference (owner, 7 Nov); any other code does not.
        ("gwas-ssf-code-11", DECLARED, [], 2),
        ("gwas-ssf-code-10", DECLARED, [], 2),
        ("gwas-ssf-code-12", UNDECLARED, _faults("indel_ssf_hm_code"), 0),
        ("gwas-ssf-code-missing", UNDECLARED, _faults("indel_ssf_hm_code"), 0),
        # Beside `hm_*` allele columns, the main columns are the study's: no
        # row declares the reference, whatever its code.
        ("gwas-ssf-hm-columns", UNDECLARED, _faults("indel_ssf_hm_columns"), 0),
        ("gwas-ssf-unharmonised", UNDECLARED, _faults("indel_ssf_unharmonised"), 0),
        # A Column Map declares by its allele convention, not by a code.
        ("gwas-ssf-map", UNDECLARED, UNDECLARED_FAULTS, 0),
        ("association-tsv", UNDECLARED, UNDECLARED_FAULTS, 0),
    ],
)
def test_every_reader_spells_an_indel_by_its_declared_reference_or_drops_it(
    tmp_path, kind: str, expected: list[tuple[str, float]], faults: list[list[str]],
    kept_by_hm_code: int,
) -> None:
    """Every projection of a reader agrees: the batches, the associations, the
    variants Pass 1 harvests, and the resolver's metric blocks."""
    from opengwasdb.readers.interface import StreamTally
    from opengwasdb.variants.normalise import canonical_alid

    reader = _reader(tmp_path, kind)
    tally = StreamTally()
    batches = list(reader.stream_batches(tally))
    alids = [str(alid) for batch in batches for alid in batch.alid]
    zs = [float(z) for batch in batches for z in batch.z]

    assert alids == [alid for alid, _ in expected]
    assert zs == pytest.approx([z for _, z in expected])
    assert [list(row) for row in tally.faults.rows] == faults
    assert tally.indel_rows_kept_by_hm_code == kept_by_hm_code
    associations = list(reader.stream_associations(StreamTally()))
    assert [
        canonical_alid(a.chromosome, a.position, a.ref, a.alt) for a in associations
    ] == alids
    assert [a.z for a in associations] == pytest.approx(zs)
    assert [
        canonical_alid(v.chromosome, v.position, v.ref, v.alt) for v in reader.stream_variants()
    ] == alids
    chunks = list(reader.stream_metric_chunks())
    assert [str(alid) for chunk in chunks for alid in chunk.alid] == alids


# --- Reference Completion: a panel indel names no ALID ------------------------


def _complete_an_observed_deletion(tmp_path, *, reference: str | None):
    """A 0.2.1 Dense Store from FinnGen holding the deletion `1:950000:GGGGCC:G`,
    completed against a panel naming the same indel by its sorted id: by an
    indel-reference table settling it as ``reference``, or by nothing.
    Returns the observed and completed Stores."""
    import test_dense_completion as tdc
    import test_panel_indels as panel_fixture
    import test_reversed_indels as pair

    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.dense.complete import complete_dense_store

    rows = [
        ("1", "900000", "A", "G", "0.2", "0.1", "0.3"),
        ("1", "950000", "GGGGCC", "G", "0.3", "0.1", "0.2"),
        ("1", "1000000", "A", "G", "-0.25", "0.1", "0.4"),
        ("1", "1100000", "A", "C", "0.15", "0.1", "0.45"),
    ]
    source = pair._finngen(tmp_path / "bmi_irn.tsv", rows)
    observed = tmp_path / "observed.opengwasdb"
    build_dense_from_vcf_manifest(
        pair._manifest(tmp_path, source, "opengwasdb.finngen-r13"), observed,
        store_id="s", release_id="r",
    )
    panel = tmp_path / "ld_panel"
    tdc._write_ld_block(
        panel / "EUR" / "1", "900000-1300000",
        [
            ("1:900000:A:G", 0.35, 900_000),
            ("1:950000:G:GGGGCC", 0.2, 950_000),
            ("1:1000000:A:G", 0.30, 1_000_000),
            ("1:1050000:C:T", 0.40, 1_050_000),
            ("1:1100000:A:C", 0.45, 1_100_000),
        ],
    )
    if reference is not None:
        table = panel / "EUR" / "indel_reference.tsv"
        table.write_text(
            "#chromosome\tposition\tallele1\tallele2\treference\toutcome\n"
            f"1\t950000\tG\tGGGGCC\t{reference}\tsettled_agree\n",
            encoding="utf-8",
        )
        panel_fixture._compress_and_index(table)
    completed = tmp_path / "completed.opengwasdb"
    complete_dense_store(observed, completed, panel, ancestry="EUR", min_cor=0.0, release_id="c")
    return observed, completed


DELETION_AT_950000 = "1:950000:GGGGCC:G"


def _alids_at_950000(store) -> list[str]:
    from opengwasdb.variants.axis import iter_variant_records

    return [
        r.alid for r in iter_variant_records(store / "variants.tsv.gz")
        if r.alid.startswith("1:950000:")
    ]


def _panel_indels_record(store) -> dict:
    import json

    manifest = json.loads((store / "manifest.json").read_text())
    return manifest["provenance"]["completion"]["panel_indels"]


def test_completion_adds_no_second_spelling_of_an_observed_deletion(tmp_path, caplog) -> None:
    """A panel names its indels by a sorted id, so with no declared reference its
    `1:950000:G:GGGGCC` may be the insertion or the deletion. Completion neither
    matches nor fills it, says so in its log and its record, and the observed
    deletion keeps its one reference-first ALID and its cell (ADR 0079)."""
    import logging

    from opengwasdb.query import query_store
    from opengwasdb.validation.validate import validate_store
    from opengwasdb.variants.axis import iter_variant_records

    with caplog.at_level(logging.WARNING, logger="opengwasdb.completion.ld_panel"):
        observed, completed = _complete_an_observed_deletion(tmp_path, reference=None)

    alids = [r.alid for r in iter_variant_records(completed / "variants.tsv.gz")]
    assert "1:1050000:C:T" in alids, "the fixture must complete a panel variant"
    assert _alids_at_950000(completed) == [DELETION_AT_950000]
    with query_store(observed) as before, query_store(completed) as after:
        observed_z = before.lookup([DELETION_AT_950000], ["bmi_irn"])["z"].tolist()
        completed_z = after.lookup([DELETION_AT_950000], ["bmi_irn"])["z"].tolist()
    assert completed_z == observed_z == pytest.approx([-3.0], abs=1e-3)
    assert _panel_indels_record(completed) == {
        "reference_from": "none", "tables": [], "n_indels": 1, "n_placed": 0,
        "unplaced_rows": {
            "no_reference": 1, "no_effect_allele": 0, "effect_allele_not_an_allele": 0,
        },
    }
    assert "names no indel's reference" in caplog.text
    result = validate_store(completed)
    assert result.ok, result.errors


def test_a_settled_panel_deletion_joins_the_observed_deletion(tmp_path) -> None:
    """The panel's table settles its indel as the deletion, so its ALID is the
    observed one: one variant, the observed cell kept, and the record names
    the table it read."""
    from opengwasdb.query import query_store
    from opengwasdb.validation.validate import validate_store

    observed, completed = _complete_an_observed_deletion(tmp_path, reference="GGGGCC")

    assert _alids_at_950000(completed) == [DELETION_AT_950000]
    with query_store(observed) as before, query_store(completed) as after:
        observed_z = before.lookup([DELETION_AT_950000], ["bmi_irn"])["z"].tolist()
        completed_z = after.lookup([DELETION_AT_950000], ["bmi_irn"])["z"].tolist()
    assert completed_z == observed_z
    record = _panel_indels_record(completed)
    assert (record["reference_from"], record["n_indels"], record["n_placed"]) == ("table", 1, 1)
    (table,) = record["tables"]
    assert table["path"].endswith("ld_panel/EUR/indel_reference.tsv.gz")
    assert len(table["sha256"]) == 64
    result = validate_store(completed)
    assert result.ok, result.errors


def test_a_panel_insertion_beside_an_observed_deletion_is_a_new_variant(tmp_path) -> None:
    """Settled as the insertion, the panel's indel is the other variant at the
    site: completion adds it beside the observed deletion."""
    _observed, completed = _complete_an_observed_deletion(tmp_path, reference="G")

    assert _alids_at_950000(completed) == ["1:950000:G:GGGGCC", DELETION_AT_950000]


# --- validate: an indel ALID against the table's own orientation --------------


#: Each layout's 0.2.1 Store of one observed deletion.
LAYOUTS = ["dense", "ragged", "hybrid"]


def _observed_deletion_store(tmp_path, layout: str = "dense"):
    """A 0.2.1 Store whose variant table holds a deletion: Dense from FinnGen,
    Ragged from GWAS-SSF (harmoniser code 10), Hybrid from GWAS-VCF with the
    deletion in its Ragged Overflow, off the Dense Component's axis."""
    import test_reversed_indels as pair
    from admission_helpers import ragged_ssf_store

    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    rows = [
        ("1", "900000", "A", "G", "0.2", "0.1", "0.3"),
        ("1", "950000", "GGGGCC", "G", "0.3", "0.1", "0.2"),
    ]
    if layout == "ragged":
        store, _result = ragged_ssf_store(tmp_path, pair._ssf_rows(rows, hm_code="10"))
        return store
    store = tmp_path / "store.opengwasdb"
    if layout == "hybrid":
        panel = tmp_path / "panel.txt"
        panel.write_text("1:900000:A:G\n", encoding="utf-8")
        build_hybrid_from_vcf_manifest(
            pair._manifest(tmp_path, pair._gwas_vcf(tmp_path / "trait.vcf", rows),
                           "opengwasdb.gwas-vcf"),
            store, reference_panel=panel, store_id="s", release_id="r",
        )
        return store
    build_dense_from_vcf_manifest(
        pair._manifest(tmp_path, pair._finngen(tmp_path / "trait.tsv", rows),
                       "opengwasdb.finngen-r13"),
        store, store_id="s", release_id="r",
    )
    return store


def _rewrite_table(store, edit) -> None:
    """Rewrite `variants.tsv.gz` line by line, keeping its offsets and tabix true."""
    import gzip

    import numpy as np
    import pysam

    path = store / "variants.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)
    offsets = []
    with pysam.BGZFile(str(path), "w") as handle:
        for line in lines:
            if not line.startswith("#"):
                offsets.append(handle.tell())
                line = edit(line)
            handle.write(line.encode("utf-8"))
    np.save(store / "variant_offsets.npy", np.asarray(offsets, dtype=np.uint64))
    pysam.tabix_index(
        str(path), seq_col=0, start_col=1, end_col=1, meta_char="#", zerobased=False,
        force=True,
    )


def _set_format_version(store, version: str) -> None:
    import json

    manifest = json.loads((store / "manifest.json").read_text())
    manifest["format_version"] = version
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _sorted_deletion(line: str, *, source_alid: str | None = None) -> str:
    """The deletion's row with its ALID spelt sorted, as before #381."""
    fields = line.rstrip("\n").split("\t")
    if fields[5] != "1:950000:GGGGCC:G":
        return line
    fields[5] = "1:950000:G:GGGGCC"
    if source_alid is not None:
        fields[3], fields[4], fields[7] = "G", "GGGGCC", source_alid
    return "\t".join(fields) + "\n"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_0_2_1_store_passes_the_indel_order_rule(tmp_path, layout: str) -> None:
    from opengwasdb.validation.validate import validate_store

    store = _observed_deletion_store(tmp_path, layout)
    assert DELETION_AT_950000 in _alids_at_950000(store), "the fixture must hold the deletion"

    result = validate_store(store, rules={"variant_table.indel_order"})

    assert result.ok, result.errors
    assert any(f.rule == "variant_table.indel_order" for f in result.findings) is False
    assert not result.skipped


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_indel_alid_its_effect_allele_contradicts_fails(tmp_path, layout: str) -> None:
    from opengwasdb.validation.validate import validate_store

    store = _observed_deletion_store(tmp_path, layout)
    _rewrite_table(store, _sorted_deletion)

    result = validate_store(store)

    assert result.failed() == {"variant_table.indel_order"}
    (message,) = result.errors
    assert "1:950000:G:GGGGCC but effect_allele 'GGGGCC'" in message


def test_a_sorted_deletion_its_source_alid_spells_reference_first_fails(tmp_path) -> None:
    """A table that spells the deletion sorted throughout, effect and other
    alleles too, still records its source's reference first."""
    from opengwasdb.validation.validate import validate_store

    store = _observed_deletion_store(tmp_path)
    _rewrite_table(store, lambda line: _sorted_deletion(line, source_alid="1:949000:GGGGCC:G"))

    result = validate_store(store)

    assert result.failed() == {"variant_table.indel_order"}
    assert "source_alid 1:949000:GGGGCC:G" in result.errors[0]


def test_a_0_2_0_store_is_not_judged_by_the_indel_order_rule(tmp_path) -> None:
    """A 0.2.0 Store sorts every indel, so its table cannot contradict the rule."""
    from opengwasdb.validation.validate import validate_store

    store = _observed_deletion_store(tmp_path)
    _rewrite_table(store, lambda line: _sorted_deletion(line, source_alid="1:949000:GGGGCC:G"))
    _set_format_version(store, "0.2.0")

    result = validate_store(store)

    assert result.ok, result.errors


@pytest.mark.parametrize(
    "header",
    [
        # Harmonised alleles in `hm_*` columns, with no harmoniser code.
        "hm_other_allele\thm_effect_allele\tchromosome\tbase_pair_location\teffect_allele"
        "\tother_allele\tbeta\tstandard_error",
        # One harmonised allele column of the two.
        "hm_code\thm_other_allele\tchromosome\tbase_pair_location\teffect_allele"
        "\tother_allele\tbeta\tstandard_error",
    ],
    ids=["no-hm-code", "one-hm-allele"],
)
def test_a_gwas_ssf_header_that_fits_no_layout_is_refused(tmp_path, header: str) -> None:
    """The layout says which allele an indel's reference is, so a header that is
    neither harmonised layout, nor free of harmoniser columns, is refused by
    every projection rather than read under one of them."""
    from opengwasdb.model.enums import StoredEffectScale
    from opengwasdb.readers.gwas_ssf import GwasSsfReader
    from opengwasdb.readers.interface import StreamTally

    path = tmp_path / "ssf.tsv"
    width = len(header.split("\t"))
    path.write_text(header + "\n" + "\t".join(["1"] * width) + "\n", encoding="utf-8")
    reader = GwasSsfReader(path, StoredEffectScale.SD)

    for project in (
        lambda: list(reader.stream_batches(StreamTally())),
        lambda: list(reader.stream_variants()),
        lambda: list(reader.stream_metric_chunks()),
        lambda: list(reader.stream_metrics()),
    ):
        with pytest.raises(ValueError, match="fits neither GWAS-SSF harmonised layout"):
            project()


def test_a_ragged_build_records_how_many_indels_the_harmoniser_code_kept(tmp_path) -> None:
    """A Release says how many indels it took on the code rule, beside the ones
    the rule dropped, per Analysis (owner, 7 Nov)."""
    from admission_helpers import ragged_ssf_store, record_analyses

    rows = [
        {"chromosome": c, "base_pair_location": p, "effect_allele": a, "other_allele": r,
         "beta": b, "standard_error": 0.5, "effect_allele_frequency": 0.3, "hm_code": code}
        for (c, p, r, a, b), code in zip(PAIR_ROWS, ["10", "12", "11", "10"], strict=True)
    ]

    store, _ = ragged_ssf_store(tmp_path, rows)

    (analysis,) = record_analyses(store)
    assert analysis["indel_rows_kept_by_hm_code"] == 1
    assert analysis["dropped"] == {"indel_ssf_hm_code": 1}
