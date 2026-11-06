"""A lower-case allele is the same base as an upper-case one (issue #360).

Spec §4: alleles are upper-cased before identity assignment, on every build
path and in extraction. A source spelling `a`/`g` therefore stores its
association under the ALID a source spelling `A`/`G` reaches, with its
statistics oriented on the upper-cased pair, whether the Store is Dense,
Ragged or Hybrid and whether its axis came from the union or from an
extracted Variant Reference.

Each fixture is checked to carry the case it claims before anything is said
about where it lands.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from extraction_fixtures import extract_from_manifest
from liftover_fixtures import write_collision_chain, write_two_strand_chain

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.query import query_store
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis
from opengwasdb.variants.normalise import identity_allele, identity_alleles, identity_base
from opengwasdb.variants.reference import read_variant_reference

#: The one variant every fixture names, in the case-insensitive spelling.
SITE = ("1", 100_000)
ALID = "1:100000:A:G"

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
)
_SSF_COLUMNS = ("chromosome", "base_pair_location", "effect_allele", "other_allele",
                "beta", "standard_error", "rsid")


def _vcf(
    path: Path, ref: str, alt: str, es: float, se: float, rsid: str = ".",
    site: tuple[str, int] = SITE,
) -> Path:
    chrom, pos = site
    path.write_text(
        _VCF_HEADER + f"{chrom}\t{pos}\t{rsid}\t{ref}\t{alt}\t.\tPASS\t.\tES:SE\t{es}:{se}\n",
        encoding="utf-8",
    )
    return path


def _vcf_rows(path: Path, rows: list[tuple[int, str, str, float]]) -> Path:
    """A chr1 GWAS-VCF of ``(position, REF, ALT, ES)`` rows, each with SE 0.5."""
    body = "".join(
        f"1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE\t{es}:0.5\n" for pos, ref, alt, es in rows
    )
    path.write_text(_VCF_HEADER + body, encoding="utf-8")
    return path


def _ssf(path: Path, effect: str, other: str, beta: float, se: float, rsid: str = "") -> Path:
    chrom, pos = SITE
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_COLUMNS) + "\n")
        fh.write(f"{chrom}\t{pos}\t{effect}\t{other}\t{beta}\t{se}\t{rsid}\n")
    return path


def _manifest(
    tmp_path: Path, rows: list[tuple[str, Path, str]], assembly: str = "hg38"
) -> Path:
    """A build manifest: ``(trait_id, path, capability)`` each, declared
    ``assembly`` unless the row names its own as a fourth field."""
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_assembly\tsource_reader_capability"
    ]
    for trait_id, path, capability, *own in rows:
        lines.append(
            f"{trait_id}\t{path}\t{trait_id}\t1000\tsd\tdeclared_standardised\t\t"
            f"{own[0] if own else assembly}\t{capability}"
        )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _z_by_alid(store: Path, analysis_id: str) -> dict[str, float]:
    """Each stored association of ``analysis_id``, by the exact ALID its row carries."""
    with query_store(store) as query:
        result = query.analysis(analysis_id)
        variants = query.variants_table()
    return {
        variants[int(index)]["alid"]: float(z)
        for index, z in zip(result["variant_index"], result["z"], strict=True)
    }


def _artifact_rows(artifact: Path) -> list[list[str]]:
    """The artifact's rows as written: a reader that normalises would hide a
    lower-case ALID, so the bytes are what is checked."""
    with gzip.open(artifact, "rt", encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]


def _axis(store: Path) -> list[str]:
    return [variant.alid for variant in VariantAxis(store).all()]


def _two_spellings(tmp_path: Path, capability: str = GWAS_VCF_CAPABILITY) -> Path:
    """Two Analyses naming one variant, one in each case: z 4.0 and 2.0 on `G`.

    The effect allele is `G` (a GWAS-VCF's ALT, a GWAS-SSF's effect_allele),
    and the upper-cased pair's A1 is `A`, so both store a negated z.
    """
    if capability == GWAS_VCF_CAPABILITY:
        lower = _vcf(tmp_path / "lower.vcf", "a", "g", 2.0, 0.5)
        upper = _vcf(tmp_path / "upper.vcf", "A", "G", 1.0, 0.5)
        assert _alleles(lower) == [("a", "g")] and _alleles(upper) == [("A", "G")]
    else:
        lower = _ssf(tmp_path / "lower.tsv.gz", "g", "a", 2.0, 0.5)
        upper = _ssf(tmp_path / "upper.tsv.gz", "G", "A", 1.0, 0.5)
        assert _alleles(lower) == [("g", "a")] and _alleles(upper) == [("G", "A")]
    return _manifest(tmp_path, [("lower", lower, capability), ("upper", upper, capability)])


def _alleles(path: Path) -> list[tuple[str, str]]:
    """Each data row's alleles as the source spells them: a GWAS-VCF's
    ``(REF, ALT)``, a GWAS-SSF's ``(effect_allele, other_allele)``."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        lines = [line.rstrip("\n").split("\t") for line in fh if not line.startswith("##")]
    header, rows = lines[0], lines[1:]
    if header[0] == "#CHROM":
        return [(row[3], row[4]) for row in rows]
    effect, other = header.index("effect_allele"), header.index("other_allele")
    return [(row[effect], row[other]) for row in rows]


@pytest.mark.parametrize("capability", [GWAS_VCF_CAPABILITY, GWAS_SSF_CAPABILITY])
def test_dense_stores_both_spellings_under_the_upper_case_alid(tmp_path, capability):
    manifest = _two_spellings(tmp_path, capability)
    store = tmp_path / "dense.opengwasdb"

    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    assert _axis(store) == [ALID]
    assert _z_by_alid(store, "lower") == {ALID: pytest.approx(-4.0, rel=5e-3)}
    assert _z_by_alid(store, "upper") == {ALID: pytest.approx(-2.0, rel=5e-3)}
    assert validate_store(store).ok


@pytest.mark.parametrize(
    ("capability", "write"),
    [
        (GWAS_VCF_CAPABILITY, lambda path: _vcf(path.with_suffix(".vcf"), "a", "G", 2.0, 0.5)),
        (GWAS_SSF_CAPABILITY, lambda path: _ssf(path.with_suffix(".tsv.gz"), "G", "a", 2.0, 0.5)),
    ],
    ids=["gwas-vcf", "gwas-ssf"],
)
def test_dense_orients_a_mixed_case_pair_on_its_upper_cased_alleles(tmp_path, capability, write):
    """`a`/`G` sorts `G` first as written but `A` first upper-cased: the effect
    allele `G` is not A1, so z is negated, and the row is `A:G`, not `G:a`."""
    source = write(tmp_path / "mixed")
    assert _alleles(source) == ([("a", "G")] if capability == GWAS_VCF_CAPABILITY else [("G", "a")])
    manifest = _manifest(tmp_path, [("mixed", source, capability)])
    store = tmp_path / "mixed.opengwasdb"

    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    assert _z_by_alid(store, "mixed") == {ALID: pytest.approx(-4.0, rel=5e-3)}


def test_hybrid_two_pass_stores_both_spellings_under_the_upper_case_alid(tmp_path):
    manifest = _two_spellings(tmp_path)
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{ALID}\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"

    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r"
    )

    assert _axis(store) == [ALID]
    with query_store(store) as query:
        hit = query.lookup([ALID], ["lower", "upper"])
    assert sorted(float(z) for z in hit["z"]) == pytest.approx([-4.0, -2.0], rel=5e-3)
    assert validate_store(store).ok


def test_an_extracted_reference_names_both_spellings_by_the_upper_case_alid(tmp_path):
    """Extraction writes the upper-case ALID, so the artifact reads back, and a
    Dense build from it stores both Analyses on that one row."""
    manifest = _two_spellings(tmp_path)
    artifact = tmp_path / "ref.variant-ref.tsv.gz"

    extract_from_manifest(manifest, artifact)
    reference = read_variant_reference(artifact)
    store = tmp_path / "from-ref.opengwasdb"
    assert [row[:5] for row in _artifact_rows(artifact)] == [[ALID, "1", "100000", "A", "G"]]
    build_dense_from_vcf_manifest(
        manifest, store, variant_reference=artifact, store_id="s", release_id="r"
    )

    assert reference.alids == [ALID]
    assert _z_by_alid(store, "lower") == {ALID: pytest.approx(-4.0, rel=5e-3)}
    assert _z_by_alid(store, "upper") == {ALID: pytest.approx(-2.0, rel=5e-3)}


def test_a_lifted_extraction_names_a_lower_case_source_by_the_upper_case_alid(tmp_path):
    """The lifted window writer forms its ALIDs as the hg38 one does: chr3's
    identity chain keeps `3:1000`, and `g`/`a` is named `A:G` there."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    lower = _vcf(tmp_path / "lower.vcf", "g", "a", 2.0, 0.5, site=("3", 1_000))
    assert _alleles(lower) == [("g", "a")]
    manifest = _manifest(tmp_path, [("lower", lower, GWAS_VCF_CAPABILITY)], assembly="hg19")
    artifact = tmp_path / "lifted.variant-ref.tsv.gz"

    extract_from_manifest(manifest, artifact, chain_file=chain)

    assert [row[:5] for row in _artifact_rows(artifact)] == [["3:1000:A:G", "3", "1000", "A", "G"]]
    assert _artifact_rows(artifact)[0][6] == "3:1000:g:a"


def _ragged_manifest(tmp_path: Path, analyses: list[str]) -> Path:
    """A Ragged SSF manifest of ``analyses``, in that order, each read from
    ``<analysis>.tsv.gz`` in the filtered directory."""
    lines = ["analysis_index\tanalysis_id\ttrait_id\tanalysis_label\tn\tfiltered_file"
             "\tsource_genome_build"]
    lines += [f"{i}\t{a}\t{a}\t{a}\t1000\t{a}.tsv.gz\tGRCh38" for i, a in enumerate(analyses)]
    manifest = tmp_path / "ragged.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _ragged_build(manifest: Path, filtered: Path, store: Path) -> None:
    build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r", association_coverage="gwas"
    )


def test_ragged_stores_a_lower_case_source_under_the_upper_case_alid(tmp_path):
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    _ssf(filtered / "lower.tsv.gz", "g", "a", 2.0, 0.5, rsid="rs1")
    assert _alleles(filtered / "lower.tsv.gz") == [("g", "a")]
    store = tmp_path / "ragged.opengwasdb"

    _ragged_build(_ragged_manifest(tmp_path, ["lower"]), filtered, store)

    assert _axis(store) == [ALID]
    assert _z_by_alid(store, "lower") == {ALID: pytest.approx(-4.0, rel=5e-3)}


def _ragged_spellings(tmp_path: Path, lower_rsid: str, upper_rsid: str) -> Path:
    """Two Ragged Analyses, ``lower`` and ``upper``, each spelling the one site
    in its own case and naming it as given."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    _ssf(filtered / "lower.tsv.gz", "g", "a", 2.0, 0.5, rsid=lower_rsid)
    _ssf(filtered / "upper.tsv.gz", "G", "A", 1.0, 0.5, rsid=upper_rsid)
    assert _alleles(filtered / "lower.tsv.gz") == [("g", "a")]
    assert _alleles(filtered / "upper.tsv.gz") == [("G", "A")]
    return filtered


@pytest.mark.parametrize("lower_first", [False, True], ids=["upper-first", "lower-first"])
def test_a_ragged_ssf_build_refuses_two_spellings_named_differently(tmp_path, lower_first):
    """A Ragged SSF build compares a site's spellings as every other harvest
    does (issue #379). `build-ragged-ssf` has no `--on-rsid-conflict`, so it
    refuses whichever Analysis is listed first, and publishes nothing."""
    filtered = _ragged_spellings(tmp_path, lower_rsid="rs2", upper_rsid="rs1")
    analyses = ["upper", "lower"]
    manifest = _ragged_manifest(tmp_path, analyses[::-1] if lower_first else analyses)
    store = tmp_path / "ragged.opengwasdb"

    with pytest.raises(ValueError, match="Ragged SSF build: .* one routing key '1:100000:A:G'"):
        _ragged_build(manifest, filtered, store)
    assert not store.exists()


@pytest.mark.parametrize("blank_case", ["lower", "upper"], ids=["other-spelling", "same-spelling"])
@pytest.mark.parametrize("blank_first", [False, True], ids=["named-first", "blank-first"])
def test_a_ragged_ssf_build_names_a_variant_by_its_first_non_empty_rsid(
    tmp_path, blank_case, blank_first
):
    """Spec §4: a variant takes the first non-empty rsid among the rows a build
    admits for it (#400). A row that names nothing gives way to a later one,
    whether that row spells the site the same way or the other way, and
    whichever Analysis is listed first."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    blank_alleles = ("g", "a") if blank_case == "lower" else ("G", "A")
    _ssf(filtered / "blank.tsv.gz", *blank_alleles, 2.0, 0.5, rsid="")
    _ssf(filtered / "named.tsv.gz", "G", "A", 1.0, 0.5, rsid="rs1")
    assert _alleles(filtered / "blank.tsv.gz") == [blank_alleles]
    analyses = ["named", "blank"]
    manifest = _ragged_manifest(tmp_path, analyses[::-1] if blank_first else analyses)
    store = tmp_path / "ragged.opengwasdb"

    _ragged_build(manifest, filtered, store)

    (variant,) = VariantAxis(store).all()
    assert (variant.alid, variant.rsid) == (ALID, "rs1")


def test_a_ragged_ssf_build_refuses_a_tab_in_a_source_allele(tmp_path):
    """The harvest keys a site as ``alid<TAB>ref<TAB>alt`` (#379), and a quoted
    GWAS-SSF cell can carry a TAB that the ALID's normalisation strips. Such a
    site is refused by name, not keyed under another site's spelling."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    chrom, pos = SITE
    with gzip.open(filtered / "tab.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_COLUMNS) + "\n")
        fh.write(f'{chrom}\t{pos}\tG\t"A\t"\t1.0\t0.5\trs1\n')
    (association,) = GwasSsfReader(filtered / "tab.tsv.gz").stream_associations()
    assert {association.ref, association.alt} == {"A\t", "G"}
    store = tmp_path / "ragged.opengwasdb"

    refusal = f"Ragged SSF build: a source allele of {ALID} holds a TAB"
    with pytest.raises(ValueError, match=refusal):
        _ragged_build(_ragged_manifest(tmp_path, ["tab"]), filtered, store)
    assert not store.exists()


def test_a_dense_build_from_an_alid_list_routes_a_lower_case_source(tmp_path):
    """A plain ALID list names its variants upper-case, and a lower-case source
    key reaches one, so its association is stored, not dropped as unmatched."""
    lower = _vcf(tmp_path / "lower.vcf", "a", "g", 2.0, 0.5)
    assert _alleles(lower) == [("a", "g")]
    manifest = _manifest(tmp_path, [("lower", lower, GWAS_VCF_CAPABILITY)])
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{ALID}\n", encoding="utf-8")
    store = tmp_path / "single.opengwasdb"

    build_dense_from_vcf_manifest(
        manifest, store, variant_reference=panel, store_id="s", release_id="r"
    )

    assert _z_by_alid(store, "lower") == {ALID: pytest.approx(-4.0, rel=5e-3)}


@pytest.mark.parametrize("lower_first", [False, True], ids=["upper-first", "lower-first"])
@pytest.mark.parametrize("mode", ["single-pass", "two-pass"])
def test_a_dense_harvest_refuses_two_spellings_named_differently(tmp_path, mode, lower_first):
    """One site in two cases is one routing key, so a harvest compares the
    names its spellings carry, from a reference naming none or from the
    sources' own union alike (issue #379); `build-dense-vcf` has no
    `--on-rsid-conflict`, and refuses them whichever source is listed first
    (spec §4). The refusal names the build the way it was run."""
    lower = _vcf(tmp_path / "lower.vcf", "a", "g", 2.0, 0.5, rsid="rs2")
    upper = _vcf(tmp_path / "upper.vcf", "A", "G", 1.0, 0.5, rsid="rs1")
    assert _alleles(lower) == [("a", "g")] and _alleles(upper) == [("A", "G")]
    rows = [("upper", upper, GWAS_VCF_CAPABILITY), ("lower", lower, GWAS_VCF_CAPABILITY)]
    manifest = _manifest(tmp_path, rows[::-1] if lower_first else rows)
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{ALID}\n", encoding="utf-8")
    reference = {"single-pass": panel, "two-pass": None}[mode]
    label = {"single-pass": "Single-pass build", "two-pass": "Two-pass build"}[mode]

    with pytest.raises(ValueError, match=f"{label}: .* one routing key '1:100000:A:G'"):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "s.opengwasdb", variant_reference=reference,
            store_id="s", release_id="r",
        )


@pytest.mark.parametrize("lower_first", [False, True], ids=["upper-first", "lower-first"])
@pytest.mark.parametrize("assembly", ["hg38", "hg19"], ids=["hg38", "lifted"])
def test_extraction_refuses_two_spellings_named_differently(tmp_path, assembly, lower_first):
    """A Variant Reference is the union frozen, so its window writers compare
    a site's spellings as the union does (issue #379). Extraction has no
    `--on-rsid-conflict`: it refuses, in either order, rather than freeze the
    first source's name for a single-pass build to use as given. chr3's
    identity chain keeps the lifted site at `3:1000`."""
    site = ("3", 1_000)
    lower = _vcf(tmp_path / "lower.vcf", "a", "g", 2.0, 0.5, rsid="rs2", site=site)
    upper = _vcf(tmp_path / "upper.vcf", "A", "G", 1.0, 0.5, rsid="rs1", site=site)
    assert _alleles(lower) == [("a", "g")] and _alleles(upper) == [("A", "G")]
    rows = [("upper", upper, GWAS_VCF_CAPABILITY), ("lower", lower, GWAS_VCF_CAPABILITY)]
    manifest = _manifest(tmp_path, rows[::-1] if lower_first else rows, assembly=assembly)
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    artifact = tmp_path / "ref.variant-ref.tsv.gz"

    with pytest.raises(
        ValueError, match="Variant Reference extraction: .* one routing key '3:1000:A:G'"
    ):
        extract_from_manifest(manifest, artifact, chain_file=chain)
    assert not artifact.exists()


def _both_assemblies(
    tmp_path: Path,
    hg38_rows: list[tuple[int, str, str, float]],
    hg19_rows: list[tuple[int, str, str, float]],
) -> Path:
    """An hg38 and an hg19 GWAS-VCF source of chr1 rows, in one manifest."""
    hg38 = _vcf_rows(tmp_path / "hg38.vcf", hg38_rows)
    hg19 = _vcf_rows(tmp_path / "hg19.vcf", hg19_rows)
    assert _alleles(hg38) == [(ref, alt) for _pos, ref, alt, _es in hg38_rows]
    assert _alleles(hg19) == [(ref, alt) for _pos, ref, alt, _es in hg19_rows]
    return _manifest(
        tmp_path,
        [("hg38", hg38, GWAS_VCF_CAPABILITY, "hg38"), ("hg19", hg19, GWAS_VCF_CAPABILITY, "hg19")],
    )


@pytest.mark.parametrize("n_workers", [1, 2])
@pytest.mark.parametrize(
    "hg38_spellings",
    [[("G", "A"), ("g", "a")], [("g", "a")]],
    ids=["shared-tuple-and-lower", "lower-only"],
)
def test_every_spelling_of_an_identity_both_assemblies_declare_is_dropped(
    tmp_path, hg38_spellings, n_workers
):
    """hg19 declares `1:501 G/A`, and hg38 declares the same identity as the
    same tuple, as `g/a`, or as both: two loci either way, so every spelling is
    dropped from both assemblies, by the union and by the reference frozen from
    it alike. The collision chain lifts hg19 1:501 to 1:5501 and 1:801 to
    1:5801; a kept `g/a` would draw hg19's 1:501 association onto hg38's
    1:501:A:G."""
    chain = write_collision_chain(tmp_path / "collision.chain")
    manifest = _both_assemblies(
        tmp_path,
        [*((501, ref, alt, 1.0) for ref, alt in hg38_spellings), (701, "C", "T", 1.0)],
        [(501, "G", "A", 2.0), (801, "C", "T", 2.0)],
    )
    artifact = tmp_path / "ref.variant-ref.tsv.gz"
    union, frozen = tmp_path / "union.opengwasdb", tmp_path / "frozen.opengwasdb"

    extract_from_manifest(manifest, artifact, chain_file=chain, n_workers=n_workers)
    build_dense_from_vcf_manifest(manifest, union, chain_file=chain, store_id="s", release_id="r")
    build_dense_from_vcf_manifest(
        manifest, frozen, chain_file=chain, variant_reference=artifact,
        store_id="s", release_id="r",
    )

    assert _axis(union) == ["1:701:C:T", "1:5801:C:T"]
    assert _axis(frozen) == _axis(union)
    assert _z_by_alid(union, "hg19") == {"1:5801:C:T": pytest.approx(-4.0, rel=5e-3)}
    assert _z_by_alid(union, "hg38") == {"1:701:C:T": pytest.approx(-2.0, rel=5e-3)}
    for analysis in ("hg19", "hg38"):
        assert _z_by_alid(frozen, analysis) == _z_by_alid(union, analysis)


#: Every one-byte cell an allele can be written as -- the ALID alphabet, in both
#: cases, among every other printable ASCII byte -- and two non-ASCII letters,
#: one of which upper-cases to two characters.
_CELLS = [bytes([code]) for code in range(0x20, 0x7F)] + ["ä".encode(), "ß".encode()]


def test_the_case_rule_gives_one_answer_in_each_of_its_forms():
    """`identity_allele` is the rule; its batch form and its byte-level fast
    form must answer as it does wherever they answer at all."""
    fast = {cell for cell in _CELLS if identity_base(cell) is not None}
    assert fast == {bytes([code]) for code in b"ACGTacgt"}
    for cell in _CELLS:
        allele = cell.decode("utf-8")
        rule = identity_allele(allele)
        assert identity_alleles([allele]) == [rule], cell
        if cell in fast:
            assert identity_base(cell) == rule, cell
    assert identity_allele("ß") == "SS" and identity_allele("a") == "A"
