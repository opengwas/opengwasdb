"""Allele-flipped EAF is caught before a store can carry it (issue #115).

`GCST003566.h.tsv.gz` reports `effect_allele_frequency` against the other
allele: its A1-oriented EAF correlates at r = -0.9992 against the EUR reference
panel where every other study in the same release correlates at +0.999. Nothing
about that row looks wrong -- 0.42 where the truth is 0.58 is a perfectly
plausible frequency -- so the only way to catch it is to compare a whole
Analysis against something.

The measurements these tests are built to mirror (ADR 0037 §6):

| source                          | r vs EUR panel |
|---------------------------------|----------------|
| GCST003566                      | -0.9992        |
| GCST005076                      | +0.9996        |
| FinnGen (bottlenecked isolate)  | +0.9954        |

The FinnGen row is the one that decides the shape of the check. Its frequencies
differ from EUR reference data by up to 3000x in magnitude, so any test on the
*size* of the difference would reject it; the correlation still reads +0.995
because a bottleneck changes magnitudes, not direction. `test_bottlenecked_
cohort_passes` reproduces that separation on synthetic data whose spread is
asserted to be at least that extreme before anything is concluded from it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.build.eaf_orientation import (
    DEFAULT_MIN_OVERLAP,
    EafOrientationError,
    EafReferenceError,
    apply_orientation_evidence,
    check_eaf_orientation,
    enforce_eaf_orientation,
    load_eaf_reference,
    sample_column,
    select_rows,
    select_sites,
    site_hashes,
)
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import EafOrientationMethod, EafOrientationOutcome

_N_VARIANTS = 4_000


def _alid(i: int) -> str:
    return f"1:{100_000 + i * 37}:A:G"


_ALID_INDEX = {_alid(i): i for i in range(_N_VARIANTS)}


def _panel_frequencies(seed: int = 7) -> dict[str, float]:
    """A panel-shaped set of A1-oriented frequencies: a realistic spread of
    common and rare variants, not a uniform draw."""
    rng = np.random.default_rng(seed)
    freqs = rng.beta(0.6, 0.6, size=_N_VARIANTS)
    return {_alid(i): float(f) for i, f in enumerate(freqs)}


def _observed_like(
    panel: dict[str, float], *, seed: int, flipped: bool = False
) -> dict[str, float]:
    """One study's own frequencies: the panel's, plus sampling noise."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for alid, f in panel.items():
        value = float(np.clip(f + rng.normal(0.0, 0.01), 0.001, 0.999))
        out[alid] = 1.0 - value if flipped else value
    return out


def _bottlenecked(panel: dict[str, float], *, seed: int = 11) -> dict[str, float]:
    """A founder-effect cohort: strong drift on the logit scale, which moves
    frequencies by orders of magnitude without reversing which allele is which."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for alid, f in panel.items():
        logit = np.log(f / (1.0 - f)) + rng.normal(0.0, 3.0)
        out[alid] = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-6, 1 - 1e-6))
    return out


@pytest.fixture
def panel() -> dict[str, float]:
    return _panel_frequencies()


@pytest.fixture
def reference(tmp_path: Path, panel: dict[str, float]):
    return load_eaf_reference(_reference_table(tmp_path, panel), panel.keys())


def _reference_table(tmp_path: Path, freqs: dict[str, float], name: str = "panel.tsv") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("alid\teaf\n")
        for alid, f in freqs.items():
            handle.write(f"{alid}\t{f:.6f}\n")
    return path


# ── The defect itself ────────────────────────────────────────────────────────


def test_flipped_analysis_fails_against_the_panel(panel, reference):
    observed = _observed_like(panel, seed=1, flipped=True)
    report = check_eaf_orientation({"GCST003566": observed}, reference=reference)

    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.FAILED
    assert evidence.r < -0.99
    assert evidence.n_overlap == _N_VARIANTS

    with pytest.raises(EafOrientationError) as excinfo:
        enforce_eaf_orientation(report)
    message = str(excinfo.value)
    assert "GCST003566" in message
    assert "-0.99" in message


def test_correctly_oriented_analysis_passes(panel, reference):
    report = check_eaf_orientation(
        {"GCST005076": _observed_like(panel, seed=2)}, reference=reference
    )

    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.PASSED
    assert evidence.r > 0.99
    enforce_eaf_orientation(report)  # does not raise


def test_bottlenecked_cohort_passes(panel, reference):
    """Magnitude differences of 1000x or more must not read as a flip."""
    observed = _bottlenecked(panel)

    # Assert the fixture is actually extreme before concluding anything from it
    # passing: a mild perturbation would prove nothing about false positives.
    ratios = np.array(
        [max(observed[a], panel[a]) / min(observed[a], panel[a]) for a in panel]
    )
    assert ratios.max() > 1000, f"bottleneck fixture is too mild: max ratio {ratios.max():.0f}"

    report = check_eaf_orientation({"finngen": observed}, reference=reference)
    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.PASSED, evidence.note
    assert evidence.r > 0.5


# ── Gates: what must never be interpreted as a direction ─────────────────────


def test_too_few_overlapping_variants_is_unverified(panel, reference):
    thin = dict(list(_observed_like(panel, seed=3).items())[: DEFAULT_MIN_OVERLAP - 1])
    report = check_eaf_orientation({"tiny": thin}, reference=reference)

    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.UNVERIFIED
    assert evidence.n_overlap == DEFAULT_MIN_OVERLAP - 1
    assert "overlap" in evidence.note


def test_near_monomorphic_sample_is_unverified(tmp_path):
    """A slice with no frequency spread correlates meaninglessly."""
    flat = {_alid(i): 0.99 + (i % 3) * 0.001 for i in range(_N_VARIANTS)}
    reference = load_eaf_reference(_reference_table(tmp_path, flat), flat.keys())
    observed = {alid: f + 0.0005 for alid, f in flat.items()}

    report = check_eaf_orientation({"monomorphic": observed}, reference=reference)
    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.UNVERIFIED
    assert "variance" in evidence.note


def test_analysis_without_eaf_is_unverified_not_passed(panel, reference):
    report = check_eaf_orientation({"no_eaf": {}}, reference=reference)
    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.UNVERIFIED
    assert "stores no EAF" in evidence.note
    assert evidence.stores_eaf is False


def test_an_analysis_with_no_eaf_does_not_block_a_build(panel, reference):
    """A store may legitimately mix sources that report frequency with sources
    that do not (ADR 0036). The one without has no orientation to get wrong, so
    it must not be what stops the build."""
    report = check_eaf_orientation(
        {"has_eaf": _observed_like(panel, seed=8), "no_eaf": {}}, reference=reference
    )

    assert report.unverified == ()  # nothing here failed verification
    enforce_eaf_orientation(report)  # does not raise


def test_a_maf_column_is_refused_as_a_reference(tmp_path, panel):
    """MAF is symmetric about 0.5, so it would certify a flipped study."""
    maf = {alid: min(f, 1.0 - f) for alid, f in panel.items()}
    path = _reference_table(tmp_path, maf, name="maf.tsv")

    with pytest.raises(EafReferenceError) as excinfo:
        load_eaf_reference(path, maf.keys())
    assert "minor allele frequency" in str(excinfo.value)

    # One row above 0.5 -- a rounding artefact, a stray record -- must not be
    # enough to get a MAF column accepted as EAF.
    maf_with_artefact = dict(maf)
    maf_with_artefact[next(iter(maf))] = 0.9
    with pytest.raises(EafReferenceError, match="minor allele frequency"):
        load_eaf_reference(
            _reference_table(tmp_path, maf_with_artefact, name="maf2.tsv"),
            maf_with_artefact.keys(),
        )

    # And the defect it would have hidden: against MAF, a flipped study
    # correlates *positively*, which is the whole reason the gate exists.
    flipped = _observed_like(panel, seed=4, flipped=True)
    shared = list(panel)
    r = np.corrcoef(
        [flipped[a] for a in shared], [min(panel[a], 1 - panel[a]) for a in shared]
    )[0, 1]
    assert r > 0


# ── Unverified outcomes are never silent ─────────────────────────────────────


def test_supplied_reference_that_verifies_nothing_fails_the_build(panel, reference):
    thin = dict(list(_observed_like(panel, seed=5).items())[:10])
    report = check_eaf_orientation({"tiny": thin}, reference=reference)

    with pytest.raises(EafOrientationError) as excinfo:
        enforce_eaf_orientation(report)
    assert "allow_unverified" in str(excinfo.value)

    enforce_eaf_orientation(report, allow_unverified=True)  # deliberate, and recorded
    assert report.provenance(allow_unverified=True)["allow_unverified"] is True


def test_no_reference_and_too_few_analyses_warns_rather_than_passing(panel, caplog):
    report = check_eaf_orientation(
        {"a": _observed_like(panel, seed=6), "b": _observed_like(panel, seed=7)}
    )

    assert report.method is EafOrientationMethod.NONE
    assert {e.outcome for e in report.evidence} == {EafOrientationOutcome.UNVERIFIED}
    with caplog.at_level("WARNING"):
        enforce_eaf_orientation(report)  # does not raise
    assert "unverified" in caplog.text.lower()


# ── Consensus, where there is no panel ───────────────────────────────────────


def test_consensus_of_three_or_more_catches_a_flipped_study(panel):
    observations = {
        "good1": _observed_like(panel, seed=10),
        "good2": _observed_like(panel, seed=11),
        "good3": _observed_like(panel, seed=12),
        "GCST003566": _observed_like(panel, seed=13, flipped=True),
    }
    report = check_eaf_orientation(observations)

    assert report.method is EafOrientationMethod.CONSENSUS
    by_id = {e.analysis_id: e for e in report.evidence}
    assert by_id["GCST003566"].outcome is EafOrientationOutcome.FAILED
    assert by_id["good1"].outcome is EafOrientationOutcome.PASSED

    with pytest.raises(EafOrientationError) as excinfo:
        enforce_eaf_orientation(report)
    assert "disagree" in str(excinfo.value)


def test_consensus_excludes_the_analysis_being_checked(panel):
    """A study included in its own baseline drags that baseline toward itself."""
    observations = {name: _observed_like(panel, seed=20 + i) for i, name in enumerate("abc")}
    report = check_eaf_orientation(observations)
    assert all(e.outcome is EafOrientationOutcome.PASSED for e in report.evidence)

    # Two flipped studies and one correct one. Each flipped study's baseline is
    # the other flipped study and the correct one -- values that average to 0.5
    # everywhere, so the variance gate refuses to read a direction from it. Were
    # a study included in its own baseline, the three-way median would agree
    # with the flipped majority and both flipped studies would read `passed`.
    flipped = _observed_like(panel, seed=30, flipped=True)
    skewed = check_eaf_orientation(
        {
            "flip1": flipped,
            "flip2": {a: min(max(v + 0.002, 0.001), 0.999) for a, v in flipped.items()},
            "correct": _observed_like(panel, seed=31),
        }
    )
    outcomes = {e.analysis_id: e.outcome for e in skewed.evidence}
    assert outcomes["flip1"] is EafOrientationOutcome.UNVERIFIED
    assert outcomes["flip2"] is EafOrientationOutcome.UNVERIFIED


def test_consensus_does_not_pick_a_side_when_the_majority_is_flipped(panel):
    """The majority being wrong is exactly what a consensus cannot adjudicate.

    Two flipped studies and one correct one: the correct study is the one that
    reads `failed`, because it is the one that disagrees with the majority. The
    build stops either way -- which is the point. A consensus can establish that
    the analyses contradict each other; deciding which is right needs a panel.
    """
    flipped = _observed_like(panel, seed=50, flipped=True)
    report = check_eaf_orientation(
        {
            "flip1": flipped,
            "flip2": {a: min(max(v + 0.002, 0.001), 0.999) for a, v in flipped.items()},
            "correct": _observed_like(panel, seed=51),
        }
    )

    outcomes = {e.analysis_id: e.outcome for e in report.evidence}
    assert outcomes["correct"] is EafOrientationOutcome.FAILED
    with pytest.raises(EafOrientationError, match="disagree"):
        enforce_eaf_orientation(report)


# ── Deterministic selection and gathering ────────────────────────────────────


def test_site_selection_is_deterministic_and_order_independent():
    alids = [_alid(i) for i in range(1000)]
    forwards = select_sites(alids, k=50)
    backwards = select_sites(list(reversed(alids)), k=50)

    assert forwards == backwards
    assert len(forwards) == 50
    assert len(set(forwards)) == 50
    # A uniform sample, not a prefix: "the first 50" would be a different list.
    assert forwards != sorted(alids)[:50]
    assert select_sites(alids[:10], k=50) == sorted(alids[:10])


def test_row_selection_agrees_with_alid_selection():
    """The two selection paths must choose the same variants.

    A Ragged builder holds ALIDs and selects with `select_sites`; a Dense or
    Hybrid builder holds row indices into a shared axis and selects with
    `select_rows`. If they disagreed, the same store would be checked over
    different variants depending on how it was built.
    """
    alids = [_alid(i) for i in range(1000)]
    hashes = site_hashes(alids)
    rows = np.arange(1000, dtype=np.int64)

    by_row = select_rows(hashes, rows, k=50)

    assert [alids[r] for r in by_row.tolist()] == select_sites(alids, k=50)


def test_sample_column_draws_from_the_variants_that_have_a_frequency():
    alids = [_alid(i) for i in range(10)]
    hashes = site_hashes(alids)
    rows = np.array([1, 2, 3, 5, 8], dtype=np.int64)
    eaf = np.array([0.1, 0.2, 0.3, np.nan, 0.5], dtype=np.float32)

    got = sample_column(rows, eaf, alids, hashes, k=4)

    # Row 5 has no frequency: NaN is not evidence, so it is absent rather than
    # present as a fabricated value -- and it does not consume the sample
    # budget, which is why four of the four remaining rows come back.
    assert set(got) == {_alid(1), _alid(2), _alid(3), _alid(8)}
    assert got[_alid(2)] == pytest.approx(0.2, rel=1e-6)


def test_a_sparse_analysis_is_sampled_from_its_own_variants(tmp_path, panel, reference):
    """The metabolome-pilot case: an Analysis covering a fraction of a percent
    of the store's variant axis.

    Sampled from the axis, a filtered cis+signals Analysis draws a few dozen
    hits from twenty thousand and reads `unverified`; sampled from its own
    variants it is checked properly. `GCST90199621` in the metabolome pilot
    overlapped 89 of 20,000 axis-sampled sites against the real EUR panel.
    """
    sparse_alids = list(panel)[:800]
    rows = np.array([_ALID_INDEX[a] for a in sparse_alids], dtype=np.int64)
    axis = list(panel)
    hashes = site_hashes(axis)
    values = _observed_like(panel, seed=60)
    eaf = np.array([values[a] for a in sparse_alids], dtype=np.float64)

    # Sampling the axis would have hit a fifth of this Analysis at most; the
    # per-Analysis rule reaches all 800 of the variants it actually carries.
    assert len(select_sites(axis, k=4_000)) == 4_000
    sampled = sample_column(rows, eaf, axis, hashes, k=4_000)
    assert len(sampled) == 800

    report = check_eaf_orientation({"sparse": sampled}, reference=reference)
    (evidence,) = report.evidence
    assert evidence.outcome is EafOrientationOutcome.PASSED, evidence.note


# ── Reference loading ────────────────────────────────────────────────────────


def test_reference_orients_frequencies_to_the_canonical_a1(tmp_path):
    """The panel's EAF describes its EA column, which is not always A1."""
    path = tmp_path / "explicit.tsv"
    path.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\teaf\n"
        "1\t100\tG\tA\t0.25\n"  # G sorts after A, so the stored A1 is A: 0.75
        "1\t200\tA\tG\t0.60\n",
        encoding="utf-8",
    )
    reference = load_eaf_reference(path, ["1:100:A:G", "1:200:A:G"])

    assert reference.eaf["1:100:A:G"] == pytest.approx(0.75)
    assert reference.eaf["1:200:A:G"] == pytest.approx(0.60)


def test_reference_from_an_ld_panel_directory(tmp_path, panel):
    """The layout Reference Completion already reads (`ld_dir/ancestry/chr/block.tsv`)."""
    block_dir = tmp_path / "ukb-hg38" / "EUR" / "1"
    block_dir.mkdir(parents=True)
    rows = list(panel.items())
    for part, start in ((rows[:2000], 0), (rows[2000:], 2000)):
        with (block_dir / f"{start}-{start + 2000}.tsv").open("w", encoding="utf-8") as handle:
            handle.write("CHR\tSNP\tOA\tEA\tEAF\tBP\n")
            for alid, f in part:
                chrom, pos, a1, a2 = alid.split(":")
                # EA is the second allele, so the stored A1 frequency is 1 - EAF.
                handle.write(
                    f"{chrom}\t{chrom}:{pos}_{a1}_{a2}\t{a1}\t{a2}\t{1.0 - f:.6f}\t{pos}\n"
                )

    reference = load_eaf_reference(tmp_path / "ukb-hg38", panel.keys(), ancestry="EUR")

    assert reference.n_variants == _N_VARIANTS
    assert reference.reference_id.endswith("#EUR")
    for alid in list(panel)[:20]:
        assert reference.eaf[alid] == pytest.approx(panel[alid], abs=1e-5)

    # Identity, not a path: the same content read twice gives the same checksum.
    again = load_eaf_reference(tmp_path / "ukb-hg38", panel.keys(), ancestry="EUR")
    assert again.checksum == reference.checksum
    assert reference.checksum.startswith("sha256:")


def test_reference_directory_without_the_requested_ancestry_is_an_error(tmp_path):
    (tmp_path / "panel" / "AFR" / "1").mkdir(parents=True)
    with pytest.raises(EafReferenceError, match="EUR"):
        load_eaf_reference(tmp_path / "panel", [], ancestry="EUR")


def test_gzipped_reference_table_reads(tmp_path, panel):
    path = tmp_path / "panel.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("alid\teaf\n")
        for alid, f in panel.items():
            handle.write(f"{alid}\t{f:.6f}\n")

    reference = load_eaf_reference(path, panel.keys())
    assert reference.n_variants == _N_VARIANTS


# ── Persisting the evidence ──────────────────────────────────────────────────


def test_evidence_is_stamped_onto_the_analyses_it_names(panel, reference):
    report = check_eaf_orientation(
        {"studied": _observed_like(panel, seed=40)}, reference=reference, n_sites=_N_VARIANTS
    )
    analyses = [Analysis(analysis_id="studied"), Analysis(analysis_id="unmentioned")]

    stamped = {a.analysis_id: a for a in apply_orientation_evidence(analyses, report)}

    assert stamped["studied"].eaf_orientation == "passed"
    assert float(stamped["studied"].eaf_orientation_r) > 0.99
    assert stamped["studied"].eaf_orientation_n == str(_N_VARIANTS)
    # An Analysis nobody measured keeps blank columns rather than an outcome.
    assert stamped["unmentioned"].eaf_orientation == ""


def test_provenance_records_what_was_compared_against_what(panel, reference):
    report = check_eaf_orientation(
        {"studied": _observed_like(panel, seed=41)}, reference=reference, n_sites=123
    )
    provenance = report.provenance()

    assert provenance["method"] == "reference_panel"
    assert provenance["reference_checksum"] == reference.checksum
    assert provenance["n_reference_variants"] == _N_VARIANTS
    assert provenance["n_sites"] == 123
    assert provenance["analyses"][0]["outcome"] == "passed"
    assert provenance["analyses"][0]["r"] > 0.99
