"""One rescaling fixture through every builder: the SD standardisation seam (issue #366).

ADR 0029 stores a continuous Analysis's statistics in SD Units by dividing each
source SE by the Analysis's `original_sd`. That divide used to be restated at
every builder's call site -- Dense, Dense-from-VCF, and the Hybrid Dense axis,
Overflow and off-reference spill each had their own -- and the two Ragged
builders had none. Four published Ragged SSF Store Releases (4,443 Analyses)
therefore hold their sources' SEs beside a recorded rescaling method, and no
test noticed, because no test built one rescaling fixture through every builder.

This one does. Each route builds the same small source -- one association of
source SE 0.5 -- declaring `original_sd_method=source_provided` and
`original_sd=2.0`, asserts the association reached the component the route
names, and then that the Store holds SE 0.25 (0.5 / 2.0), z exactly as the
source gave it, and the SD and method in `analyses.tsv`. A method that carries
no magnitude (`declared_standardised`) stores the SE unchanged.

The expected values are the fixture's literals, not the builders' arithmetic.
SE is compared at `rel=5e-3` because a Store codes it to `float16` or a
residual model (spec §6a).
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from besd_fixtures import write_besd_sparse_3f, write_epi, write_esi
from sd_scaling import require_not_rescaling, require_rescaling

from opengwasdb.admission import Gate
from opengwasdb.build.observed import build_dense_observed_from_manifest
from opengwasdb.encoding.sd_units import PhenotypeSd
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.layouts.hybrid.build import _OffReferenceSpill, build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged import build_ssf
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import OriginalSdMethod
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.variants.routing import Unmatched

#: The source SE every route's association carries, and the SD it is divided by.
SOURCE_SE = 0.5
ORIGINAL_SD = "2.0"

#: hg19 1:100000 A/G lifts to hg38 1:100000 (tests/test_dense_vcf_build.py);
#: hg19 1:1000000 C/T lifts to hg38 1:1064620.
HG19_ON_PANEL = 100_000
HG19_OFF_PANEL = 1_000_000
ON_PANEL_ALID = "1:100000:A:G"
OFF_PANEL_ALID = "1:1064620:C:T"


@dataclass(frozen=True)
class Declared:
    """What the manifest row declares about the Analysis's phenotype SD, and how
    the manifest spells the method (the method's own value unless given)."""

    method: OriginalSdMethod
    original_sd: str
    spelling: str | None = None

    @property
    def method_text(self) -> str:
        return self.method.value if self.spelling is None else self.spelling


@dataclass(frozen=True)
class Built:
    """A built Store, its Analysis, and each association to read back -- every
    component the route reaches -- with its source's z."""

    store: Path
    analysis_id: str
    source_z: dict[str, float]


RESCALING = Declared(OriginalSdMethod.SOURCE_PROVIDED, ORIGINAL_SD)
NOT_RESCALING = Declared(OriginalSdMethod.DECLARED_STANDARDISED, "")


# ── Dense, from an association TSV (`build-dense`) ───────────────────────────


def _dense(tmp_path: Path, declared: Declared, _monkeypatch: pytest.MonkeyPatch) -> Built:
    source = tmp_path / "trait.tsv"
    source.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        f"1\t100000\tA\tG\t2.0\t{SOURCE_SE}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analysis_id\tsource_file\tstored_effect_scale\toriginal_sd_method\toriginal_sd\n"
        f"trait\t{source}\tsd\t{declared.method_text}\t{declared.original_sd}\n",
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_observed_from_manifest(
        manifest, store, store_id="s", release_id="r", admission_record=tmp_path / "adm"
    )
    return Built(store, "trait", {ON_PANEL_ALID: 2.0})


# ── GWAS-VCF sources: Dense-from-VCF and the three Hybrid routes ─────────────


def _vcf_manifest(tmp_path: Path, declared: Declared) -> Path:
    """One Analysis whose two associations both carry SE 0.5: an on-panel one
    (z 2.0, stored -2.0 because the canonical ALID flips it) and an off-panel
    one (z 2.0, likewise flipped)."""
    vcf = tmp_path / "trait.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
        f"1\t{HG19_ON_PANEL}\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:{SOURCE_SE}\n"
        f"1\t{HG19_OFF_PANEL}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:{SOURCE_SE}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\n"
        f"trait\t{vcf}\tTrait\t1000\tsd\t{declared.method_text}\t{declared.original_sd}\n",
        encoding="utf-8",
    )
    return manifest


def _panel(tmp_path: Path) -> Path:
    """A panel naming only the on-panel variant: the other is off-panel."""
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{ON_PANEL_ALID}\n", encoding="utf-8")
    return panel


def _dense_vcf(tmp_path: Path, declared: Declared, _monkeypatch: pytest.MonkeyPatch) -> Built:
    store = tmp_path / "dense-vcf.opengwasdb"
    build_dense_from_vcf_manifest(
        _vcf_manifest(tmp_path, declared), store, store_id="s", release_id="r"
    )
    return Built(store, "trait", {ON_PANEL_ALID: -2.0})


def _components_by_alid(store: Path) -> dict[str, str]:
    """Which Hybrid component holds each variant with an association: `dense`
    (the Dense Component's own axis) or `overflow` (the Ragged Overflow)."""
    with query_store(store) as query:
        alid_by_shared = {index: v["alid"] for index, v in query.variants_table().items()}
    dense = {int(index) for index in np.load(store / "dense" / "dense_to_shared.npy")}
    overflow_arrays = open_store(store).arrays(mode="r")["ragged"]
    overflow = {int(index) for index in np.unique(overflow_arrays["variant_index"][:])}
    return {
        alid: ("dense" if index in dense else "overflow")
        for index, alid in alid_by_shared.items()
        if index in dense or index in overflow
    }


def _hybrid_two_pass(
    tmp_path: Path, declared: Declared, _monkeypatch: pytest.MonkeyPatch
) -> Built:
    """One two-pass build reaches both components: the on-panel association
    the Dense Component, the off-panel one the Overflow."""
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _vcf_manifest(tmp_path, declared), store, reference_panel=_panel(tmp_path),
        store_id="s", release_id="r",
    )
    components = _components_by_alid(store)
    assert components.get(ON_PANEL_ALID) == "dense", (
        "the on-panel association must reach the Dense Component, or this route proves nothing"
    )
    assert components.get(OFF_PANEL_ALID) == "overflow", (
        "the off-panel association must reach the Overflow, or this route proves nothing"
    )
    return Built(store, "trait", {ON_PANEL_ALID: -2.0, OFF_PANEL_ALID: -2.0})


def _hybrid_single_pass(
    tmp_path: Path, declared: Declared, monkeypatch: pytest.MonkeyPatch
) -> Built:
    """A single-pass build whose variant reference names only the on-panel
    variant, so the other is unknown to the route table: Pass 2 spills it
    off-reference, and the fold lifts it into the Overflow."""
    spilled: list[int] = []
    real_spill = hybrid_build._off_reference_spill

    def _counting_spill(unmatched: Unmatched, gate: Gate) -> _OffReferenceSpill:
        spill = real_spill(unmatched, gate)
        spilled.append(len(spill[0]))
        return spill

    monkeypatch.setattr(hybrid_build, "_off_reference_spill", _counting_spill)
    store = tmp_path / "hybrid-single-pass.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _vcf_manifest(tmp_path, declared), store, variant_reference=_panel(tmp_path),
        store_id="s", release_id="r", n_workers=1,
    )
    assert sum(spilled) == 1, (
        f"the off-panel association must reach the off-reference spill (spilled {spilled}), "
        "or this route proves nothing"
    )
    components = _components_by_alid(store)
    assert components.get(OFF_PANEL_ALID) == "overflow", (
        "the folded off-reference association must reach the Overflow"
    )
    assert components.get(ON_PANEL_ALID) == "dense"
    return Built(store, "trait", {ON_PANEL_ALID: -2.0, OFF_PANEL_ALID: -2.0})


# ── Ragged, from BESD and from filtered GWAS-SSF ─────────────────────────────


def _ragged_besd(tmp_path: Path, declared: Declared, _monkeypatch: pytest.MonkeyPatch) -> Built:
    """One probe with one association: beta 1.0, SE 0.5, so z 2.0."""
    fixture = tmp_path / "besd"
    fixture.mkdir()
    write_esi(
        fixture / "test.esi", [{"chr": "1", "snp_id": "rs1", "bp": 100_000, "a1": "A", "a2": "G"}]
    )
    write_epi(
        fixture / "test.epi",
        [{"chr": "1", "probe_id": "ENSG00000000001", "bp": 150_000, "gene": "GENE1"}],
    )
    write_besd_sparse_3f(fixture / "test.besd", 1, [[(0, 1.0, SOURCE_SE)]])
    overlay = tmp_path / "analyses.tsv"
    overlay.write_text(
        "analysis_id\tsample_size\tstored_effect_scale\toriginal_sd_method\toriginal_sd\n"
        f"ENSG00000000001\t1000\tsd\t{declared.method_text}\t{declared.original_sd}\n",
        encoding="utf-8",
    )
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        fixture / "test", store, store_id="s", release_id="r", analyses_path=overlay,
        association_coverage="cis_and_signals",
    )
    return Built(store, "ENSG00000000001", {ON_PANEL_ALID: 2.0})


def _ragged_ssf(tmp_path: Path, declared: Declared, _monkeypatch: pytest.MonkeyPatch) -> Built:
    """One association: beta 1.0, SE 0.5, A the canonical first allele, so z 2.0."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    with gzip.open(filtered / "trait.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
            "\tstandard_error\teffect_allele_frequency\trsid\tvariant_id\n"
            f"1\t100000\tA\tG\t1.0\t{SOURCE_SE}\t\trs1\t\n"
        )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analysis_index\tanalysis_id\tanalysis_label\tn\tmhc\tfiltered_file"
        "\tsource_genome_build\tstored_effect_scale\toriginal_sd_method\toriginal_sd\n"
        f"0\ttrait\tTrait\t1000\tFALSE\ttrait.tsv.gz\tGRCh38"
        f"\tsd\t{declared.method_text}\t{declared.original_sd}\n",
        encoding="utf-8",
    )
    store = tmp_path / "ssf.opengwasdb"
    build_ssf.build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )
    return Built(store, "trait", {ON_PANEL_ALID: 2.0})


Route = Callable[[Path, Declared, pytest.MonkeyPatch], Built]

ROUTES: dict[str, Route] = {
    "dense": _dense,
    "dense-vcf": _dense_vcf,
    "hybrid-two-pass": _hybrid_two_pass,
    "hybrid-single-pass": _hybrid_single_pass,
    "ragged-besd": _ragged_besd,
    "ragged-ssf": _ragged_ssf,
}


def _stored(built: Built) -> dict[str, tuple[float, float]]:
    """The Store's z and SE for each of the route's associations."""
    alids = list(built.source_z)
    with query_store(built.store) as query:
        found = query.lookup(alids, [built.analysis_id])
    assert len(found["se"]) == len(alids), (
        f"{alids} must all be stored for {built.analysis_id}, or there is nothing to judge"
    )
    with query_store(built.store) as query:
        alid_of = {index: v["alid"] for index, v in query.variants_table().items()}
    return {
        alid_of[int(index)]: (float(z), float(se))
        for index, z, se in zip(found["variant_index"], found["z"], found["se"], strict=True)
    }


def _recorded(built: Built) -> dict[str, str]:
    rows = {row["analysis_id"]: row for row in read_analyses(built.store / "analyses.tsv").rows}
    return rows[built.analysis_id]


def _assert_standardised(built: Built) -> None:
    """SE 0.5 under a declared SD of 2.0 is stored as 0.25, z as the source gave it."""
    for alid, (z, se) in _stored(built).items():
        assert se == pytest.approx(0.25, rel=5e-3), f"{alid}: stored SE {se}, source 0.5 / SD 2.0"
        assert z == pytest.approx(built.source_z[alid], rel=5e-3), "z is scale-free"
    recorded = _recorded(built)
    assert recorded["original_sd_method"] == RESCALING.method.value
    assert recorded["original_sd"] and float(recorded["original_sd"]) == 2.0, (
        "a standardised Analysis must record the SD it was standardised by, "
        f"not {recorded['original_sd']!r}"
    )


@pytest.mark.parametrize("route", ROUTES.values(), ids=ROUTES.keys())
def test_every_builder_divides_the_se_by_a_rescaling_original_sd_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route
) -> None:
    require_rescaling(RESCALING.method)

    _assert_standardised(route(tmp_path, RESCALING, monkeypatch))


@pytest.mark.parametrize("route", ROUTES.values(), ids=ROUTES.keys())
def test_every_builder_leaves_the_se_alone_under_a_method_without_a_magnitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route
) -> None:
    """`declared_standardised` carries no SD, so SE 0.5 is stored as 0.5."""
    require_not_rescaling(NOT_RESCALING.method)

    built = route(tmp_path, NOT_RESCALING, monkeypatch)

    for alid, (z, se) in _stored(built).items():
        assert se == pytest.approx(SOURCE_SE, rel=5e-3), f"{alid}: stored SE {se}, source 0.5"
        assert z == pytest.approx(built.source_z[alid], rel=5e-3)
    recorded = _recorded(built)
    assert recorded["original_sd_method"] == NOT_RESCALING.method.value
    assert recorded["original_sd"] == ""


def test_a_route_that_hands_its_source_ses_over_as_already_standardised_fails_the_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam test can fail: Ragged SSF made to pass `APPLIED`, as the Ragged
    builders in effect did before #366, stores SE 0.5 and is caught."""

    class _Unstandardised:
        @staticmethod
        def dividing_by(*_divisors: float) -> PhenotypeSd:
            return PhenotypeSd.APPLIED

    monkeypatch.setattr(build_ssf, "PhenotypeSd", _Unstandardised)
    built = _ragged_ssf(tmp_path, RESCALING, monkeypatch)

    with pytest.raises(AssertionError, match="stored SE 0.5"):
        _assert_standardised(built)


#: The routes whose manifest's `original_sd_method` is an optional column.
RAGGED = {"ragged-besd": _ragged_besd, "ragged-ssf": _ragged_ssf}


@pytest.mark.parametrize("route", [*RAGGED.values(), _dense], ids=[*RAGGED, "dense"])
def test_a_padded_method_is_refused_as_dense_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route
) -> None:
    """Ragged validated the method stripped but stored it as spelt, so a padded
    one built and then failed `analyses.vocabularies` (issue #366). Every
    builder refuses it, naming the Analysis and the spelling."""
    padded = Declared(OriginalSdMethod.SOURCE_PROVIDED, ORIGINAL_SD, " source_provided ")

    with pytest.raises(ValueError, match="invalid original_sd_method ' source_provided '"):
        route(tmp_path, padded, monkeypatch)


# ── A row labelled SD Units must have been standardised (issue #366, ADR 0084) ──


@pytest.mark.parametrize(
    "route", [*RAGGED.values(), _dense], ids=[*RAGGED, "dense"]
)
def test_a_row_labelled_sd_units_with_a_method_that_standardises_nothing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route
) -> None:
    """`stored_effect_scale=sd` over `binary_trait` would build an undivided
    Analysis labelled SD Units. Every builder refuses it and names the Analysis."""
    undivided = Declared(OriginalSdMethod.BINARY_TRAIT, "")

    with pytest.raises(
        ValueError, match=r"analysis '(trait|ENSG00000000001)'.*stored_effect_scale='sd'"
    ):
        route(tmp_path, undivided, monkeypatch)


@pytest.mark.parametrize("route", RAGGED.values(), ids=list(RAGGED))
def test_a_ragged_row_labelled_sd_units_with_no_method_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route
) -> None:
    """Ragged read an omitted method as divisor 1.0 and kept the label `sd`;
    Dense already refuses a blank method as an invalid one."""
    omitted = Declared(OriginalSdMethod.BINARY_TRAIT, "", spelling="")

    with pytest.raises(
        ValueError, match=r"analysis '(trait|ENSG00000000001)'.*stored_effect_scale='sd'"
    ):
        route(tmp_path, omitted, monkeypatch)
