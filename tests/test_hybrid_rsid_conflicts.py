"""Sources that name one stored variant differently (issue #306).

A Hybrid routing key upper-cases alleles, so a source written in lower case
lands on the same stored variant as an upper-case one. When the two name that
variant with different rsids, the build cannot tell which name is right. Its
default is to refuse (issue #255 round 4); `--on-rsid-conflict drop` instead
stores the variant with no rsid, so it is found by ALID and position but never
answers to a name its sources disagree on, and says how many keys it dropped.

The fixtures mirror the GWAS Catalog shape that forced the choice (stores
#225): an upper-case harmonised GWAS-SSF file listed first, a lower-case one
after it, both naming the variant in `variant_id` and carrying no `rsid`
column. Every expected value below is written out by hand from the fixture.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from test_hybrid_build import _assert_hybrid_stores_match, _hybrid_axis_rsids
from test_hybrid_checkpoint import _assert_manifests_match
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.checkpoint import (
    BUILD_PARAMS,
    checkpoint_dir_for,
    read_json,
    write_json,
)
from opengwasdb.query import query_store
from opengwasdb.readers import GWAS_SSF_CAPABILITY
from opengwasdb.validation import validate_store
from opengwasdb.variants import VariantAxis

#: On the reference: the variant the two sources name differently.
CONFLICT_ON = "1:100000:A:G"
#: On the reference: a variant both sources name alike, so a drop is seen to
#: touch only the variant whose names disagree.
AGREED = "1:1564620:A:G"
#: Off the reference, so the variant is placed by the fold, not the reference.
CONFLICT_OFF = "1:2000000:C:T"

#: The two competing names, as the upper-case and the lower-case source spell them.
UPPER_NAME = "rs144434394"
LOWER_NAME = "rs780023594"

_HEADER = (
    "chromosome",
    "base_pair_location",
    "other_allele",
    "effect_allele",
    "beta",
    "standard_error",
    "variant_id",
)


def _write_ssf(path: Path, rows: list[tuple[str, int, str, str, float, float, str]]) -> Path:
    """A harmonised GWAS-SSF file with no `rsid` column: the name is `variant_id`."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(_HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell) for cell in row) + "\n")
    return path


def _manifest(tmp_path: Path, sources: list[tuple[str, Path]]) -> Path:
    """Manifest rows in rank order; every source is hg38 GWAS-SSF."""
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_assembly\tsource_reader_capability"
    ]
    for trait_id, path in sources:
        lines.append(
            f"{trait_id}\t{path}\t{trait_id}\t1000\tsd\tdeclared_standardised\t"
            f"\thg38\t{GWAS_SSF_CAPABILITY}"
        )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _reference(tmp_path: Path) -> Path:
    """A plain ALID list: it names no rsids, so the build harvests the sources'."""
    reference = tmp_path / "reference.txt"
    reference.write_text(f"{CONFLICT_ON}\n{AGREED}\n", encoding="utf-8")
    return reference


def _conflict_fixture(tmp_path: Path, conflict: str) -> tuple[Path, Path]:
    """Two sources disagreeing on `conflict` and agreeing on `AGREED`."""
    chrom, pos, ref, alt = conflict.split(":")
    upper = _write_ssf(
        tmp_path / "upper.tsv.gz",
        [
            (chrom, int(pos), ref, alt, 0.5, 0.1, UPPER_NAME),
            ("1", 1564620, "A", "G", 0.2, 0.1, "rs3"),
        ],
    )
    lower = _write_ssf(
        tmp_path / "lower.tsv.gz",
        [
            (chrom, int(pos), ref.lower(), alt.lower(), 0.4, 0.1, LOWER_NAME),
            ("1", 1564620, "a", "g", 0.3, 0.1, "rs3"),
        ],
    )
    return _manifest(tmp_path, [("upper", upper), ("lower", lower)]), _reference(tmp_path)


def _cli(manifest: Path, reference: Path, store: Path, *extra: str):
    return CliRunner().invoke(
        app,
        [
            "build-hybrid", str(manifest), str(store),
            "--variant-reference", str(reference),
            "--store-id", "s", "--release-id", "r", *extra,
        ],
    )


def _table_rsid_cells(store: Path) -> dict[str, str]:
    """ALID -> the `rsid` cell exactly as `variants.tsv.gz` writes it."""
    with gzip.open(store / "variants.tsv.gz", "rt", encoding="utf-8") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if not line.startswith("#")]
    return {row[5]: row[6] for row in rows}


def _summary(output: str) -> dict:
    """The build's JSON summary: the last line it prints."""
    return json.loads(output.strip().splitlines()[-1])


def _assert_one_variant_unnamed(
    result, store: Path, key: str, names: tuple[str, ...], alid: str
) -> None:
    """A `drop` build that met one disputed key reaching one variant: it says
    so, and the variant answers to its ALID, with both sources' associations,
    but to none of the names."""
    assert result.exit_code == 0, result.output
    assert _summary(result.output)["rsid_conflicts"] == {
        "policy": "drop",
        "n_keys": 1,
        "n_variants": 1,
        "sample": [{"key": key, "names": list(names)}],
    }
    assert validate_store(store).ok, validate_store(store).errors
    axis = VariantAxis(store)
    try:
        assert all(axis.by_identifier(name) is None for name in names)
        assert axis.by_identifier(alid) is not None
    finally:
        axis.close()
    with query_store(store) as query:
        hit = query.lookup([alid], ["upper", "lower"])
    assert len(hit["z"]) == 2


@pytest.mark.parametrize("conflict", [CONFLICT_ON, CONFLICT_OFF], ids=["reference", "fold"])
class TestConflictingNames:
    def test_refuse_is_the_default_and_names_the_key_and_both_rsids(self, tmp_path, conflict):
        manifest, reference = _conflict_fixture(tmp_path, conflict)
        store = tmp_path / "refused.opengwasdb"

        result = _cli(manifest, reference, store)

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        message = str(result.exception)
        assert f"case-fold to one routing key {conflict!r} with different rsids" in message
        assert repr(UPPER_NAME) in message and repr(LOWER_NAME) in message
        assert not store.exists()

    def test_drop_stores_the_variant_with_no_rsid_and_reports_it(self, tmp_path, conflict):
        manifest, reference = _conflict_fixture(tmp_path, conflict)
        store = tmp_path / "dropped.opengwasdb"

        result = _cli(manifest, reference, store, "--on-rsid-conflict", "drop")

        _assert_one_variant_unnamed(result, store, conflict, (UPPER_NAME, LOWER_NAME), conflict)
        # The fixture is meaningful: the agreed variant keeps its name, so the
        # store does carry rsids and only the disputed one is missing. The
        # reference's variants are stored whether or not a source observes them.
        expected = {CONFLICT_ON: "", AGREED: "rs3"} | {conflict: ""}
        assert _hybrid_axis_rsids(store) == expected
        # "No rsid" is the table's own `.`, as for a variant no source names.
        assert _table_rsid_cells(store)[conflict] == "."
        axis = VariantAxis(store)
        try:
            assert axis.by_identifier("rs3") == axis.by_identifier(AGREED) is not None
        finally:
            axis.close()


class TestNormalisedNamesDoNotConflict:
    """A compound `rsN:pos:a:b` id and a `;` list name the same variant as
    their plain `rsN`, so a source spelling it that way is no conflict."""

    @pytest.mark.parametrize("policy", ["refuse", "drop"])
    @pytest.mark.parametrize(
        ("upper_id", "lower_id", "stored"),
        [
            # GCST004988 vs GCST010723 at 1:905 (stores #225), on the reference here.
            ("rs199633771:13122834:A:G", "rs199633771", "rs199633771"),
            ("rs34994416;rs199928265", "rs34994416", "rs34994416"),
        ],
        ids=["compound", "list"],
    )
    def test_builds_and_stores_the_plain_name(
        self, tmp_path, policy, upper_id, lower_id, stored
    ):
        upper = _write_ssf(
            tmp_path / "upper.tsv.gz", [("1", 100000, "A", "G", 0.5, 0.1, upper_id)]
        )
        lower = _write_ssf(
            tmp_path / "lower.tsv.gz", [("1", 100000, "a", "g", 0.4, 0.1, lower_id)]
        )
        manifest = _manifest(tmp_path, [("upper", upper), ("lower", lower)])
        store = tmp_path / "normalised.opengwasdb"

        result = _cli(manifest, _reference(tmp_path), store, "--on-rsid-conflict", policy)

        assert result.exit_code == 0, result.output
        assert _summary(result.output)["rsid_conflicts"] == {
            "policy": policy,
            "n_keys": 0,
            "n_variants": 0,
            "sample": [],
        }
        assert _hybrid_axis_rsids(store) == {CONFLICT_ON: stored, AGREED: ""}


#: GCST004988 reports this indel in both allele orders under different names;
#: GCST010723 names the `TTTTG`/`T` order as the other one does (stores #225).
INDEL_ALID = "1:154311889:T:TTTTG"


@pytest.mark.parametrize("on_reference", [True, False], ids=["reference", "fold"])
def test_drop_blanks_a_variant_another_allele_order_still_names(tmp_path, on_reference):
    """The variant, not just the disputed key, stores no rsid: here the
    `TTTTG`/`T` key is disputed (rs376674031 against rs113800458) while the
    `T`/`TTTTG` key, which routes to the same variant, names rs113800458 alone.
    Before the variant was blanked it kept that name."""
    upper = _write_ssf(
        tmp_path / "upper.tsv.gz",
        [
            ("1", 154311889, "T", "TTTTG", 0.5, 0.1, "rs113800458:154284365:T:TTTTG"),
            ("1", 154311889, "TTTTG", "T", -0.5, 0.1, "rs376674031:154284365:TTTTG:T"),
            ("1", 1564620, "A", "G", 0.2, 0.1, "rs3"),
        ],
    )
    lower = _write_ssf(
        tmp_path / "lower.tsv.gz",
        [
            ("1", 154311889, "ttttg", "t", 0.4, 0.1, "rs113800458"),
            ("1", 1564620, "a", "g", 0.3, 0.1, "rs3"),
        ],
    )
    manifest = _manifest(tmp_path, [("upper", upper), ("lower", lower)])
    reference = tmp_path / "reference.txt"
    alids = [INDEL_ALID, AGREED] if on_reference else [AGREED]
    reference.write_text("".join(f"{alid}\n" for alid in alids), encoding="utf-8")
    store = tmp_path / "indel.opengwasdb"

    result = _cli(manifest, reference, store, "--on-rsid-conflict", "drop")

    names = ("rs376674031", "rs113800458")
    _assert_one_variant_unnamed(result, store, "1:154311889:TTTTG:T", names, INDEL_ALID)
    assert _hybrid_axis_rsids(store) == {INDEL_ALID: "", AGREED: "rs3"}
    if on_reference:
        # The disputed order is off the reference, so the fold reaches this
        # Dense row after the skeleton was named: the Dense table is rewritten.
        assert _table_rsid_cells(store / "dense")[INDEL_ALID] == "."


def _agreeing_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Mixed-case sources that name every shared variant alike."""
    upper = _write_ssf(
        tmp_path / "upper.tsv.gz",
        [
            ("1", 100000, "A", "G", 0.5, 0.1, "rs1"),
            ("1", 1564620, "A", "G", 0.2, 0.1, "rs3"),
            ("1", 2000000, "C", "T", 0.6, 0.1, "rs4"),
        ],
    )
    lower = _write_ssf(
        tmp_path / "lower.tsv.gz",
        [
            ("1", 100000, "a", "g", 0.4, 0.1, "rs1"),
            ("1", 1564620, "a", "g", 0.3, 0.1, "."),
            ("1", 2000000, "c", "t", 0.7, 0.1, "rs4"),
        ],
    )
    return _manifest(tmp_path, [("upper", upper), ("lower", lower)]), _reference(tmp_path)


def test_conflict_free_build_is_identical_under_both_policies(tmp_path):
    manifest, reference = _agreeing_fixture(tmp_path)
    refused = tmp_path / "refuse.opengwasdb"
    dropped = tmp_path / "drop.opengwasdb"

    by_refuse = _cli(manifest, reference, refused)
    by_drop = _cli(manifest, reference, dropped, "--on-rsid-conflict", "drop")

    assert by_refuse.exit_code == 0, by_refuse.output
    assert by_drop.exit_code == 0, by_drop.output
    assert _summary(by_refuse.output)["rsid_conflicts"] == {
        "policy": "refuse", "n_keys": 0, "n_variants": 0, "sample": []
    }
    assert _summary(by_drop.output)["rsid_conflicts"] == {
        "policy": "drop", "n_keys": 0, "n_variants": 0, "sample": []
    }
    expected = {CONFLICT_ON: "rs1", AGREED: "rs3", CONFLICT_OFF: "rs4"}
    assert _hybrid_axis_rsids(refused) == expected
    _assert_hybrid_stores_match(refused, dropped)
    _assert_manifests_match(refused, dropped)
    for name in ("variant_rsid_bytes.npy", "variant_rsid_rows.npy"):
        assert (refused / name).read_bytes() == (dropped / name).read_bytes(), name


def test_reference_panel_build_refuses_drop(tmp_path):
    """The legacy two-pass build has no harvest for the policy to govern, so
    asking it to drop is refused rather than silently ignored."""
    manifest, reference = _agreeing_fixture(tmp_path)

    with pytest.raises(ValueError, match="--on-rsid-conflict drop"):
        build_hybrid_from_vcf_manifest(
            manifest, tmp_path / "panel.opengwasdb", reference_panel=reference,
            store_id="s", release_id="r", on_rsid_conflict="drop",
        )


def test_build_that_reads_no_source_names_reports_no_count(tmp_path):
    """Without a harvest nothing was compared, so the count is unknown (null),
    not zero. (A two-pass build refuses mixed-case sources outright, so this
    one source is upper case.)"""
    source = _write_ssf(tmp_path / "upper.tsv.gz", [("1", 100000, "A", "G", 0.5, 0.1, "rs1")])
    manifest = _manifest(tmp_path, [("upper", source)])
    store = tmp_path / "panel.opengwasdb"

    result = build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=_reference(tmp_path), store_id="s", release_id="r"
    )

    assert _hybrid_axis_rsids(store)[CONFLICT_ON] == "rs1"
    assert result.rsid_conflicts.as_dict() == {
        "policy": "refuse", "n_keys": None, "n_variants": None, "sample": []
    }


def test_sample_is_the_first_keys_in_genomic_order(tmp_path):
    """More conflicts than the sample holds: the count is all of them, the
    sample the first `RSID_CONFLICT_SAMPLE_SIZE` in genomic order -- not text
    order, which puts `2:1000000` before `2:100001` and chromosome 10 before 2,
    and not the harvest's `(rank, site)` order, which also puts 10 first."""
    from opengwasdb.variants.rsid import RSID_CONFLICT_SAMPLE_SIZE

    positions = [
        *range(100001, 100001 + RSID_CONFLICT_SAMPLE_SIZE + 2),
        # Sorts before the rest as text, after them on the genome.
        1_000_000,
    ]
    upper = _write_ssf(
        tmp_path / "upper.tsv.gz",
        [("2", pos, "A", "G", 0.5, 0.1, f"rsU{pos}") for pos in positions]
        + [("10", 5, "A", "G", 0.5, 0.1, "rsU10")],
    )
    lower = _write_ssf(
        tmp_path / "lower.tsv.gz",
        [("2", pos, "a", "g", 0.4, 0.1, f"rsL{pos}") for pos in positions]
        + [("10", 5, "a", "g", 0.4, 0.1, "rsL10")],
    )
    manifest = _manifest(tmp_path, [("upper", upper), ("lower", lower)])

    result = build_hybrid_from_vcf_manifest(
        manifest, tmp_path / "many.opengwasdb", variant_reference=_reference(tmp_path),
        store_id="s", release_id="r", on_rsid_conflict="drop",
    )

    report = result.rsid_conflicts.as_dict()
    assert report["n_keys"] == len(positions) + 1
    # Every disputed key is one off-reference variant of its own.
    assert report["n_variants"] == len(positions) + 1
    expected_keys = [f"2:{pos}:A:G" for pos in positions][:RSID_CONFLICT_SAMPLE_SIZE]
    assert [entry["key"] for entry in report["sample"]] == expected_keys
    assert report["sample"][0]["names"] == ["rsU100001", "rsL100001"]


def _build(manifest: Path, reference: Path, store: Path, **overrides: object):
    return build_hybrid_from_vcf_manifest(
        manifest, store, variant_reference=reference, store_id="s", release_id="r", **overrides
    )


def _crash_after_the_fold(
    monkeypatch: pytest.MonkeyPatch, manifest: Path, reference: Path, store: Path,
    **overrides: object,
) -> None:
    """Run a checkpointed build until the joint SE fit, which runs after the
    fold is recorded, and fail it there."""

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash in the joint SE fit")

    monkeypatch.setattr(hybrid_build, "_fit_joint_se", _explode)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _build(manifest, reference, store, checkpoint=True, **overrides)
    monkeypatch.undo()
    assert (checkpoint_dir_for(store) / "fold.done").exists()


class TestCheckpointedConflicts:
    def test_resume_after_the_fold_reports_the_first_run_s_conflicts(
        self, tmp_path, monkeypatch
    ):
        """A resumed run does not harvest again once the fold is recorded, so
        the count must come from the checkpoint, not be lost."""
        manifest, reference = _conflict_fixture(tmp_path, CONFLICT_OFF)
        plain_store = tmp_path / "plain.opengwasdb"
        plain = _build(manifest, reference, plain_store, on_rsid_conflict="drop")
        assert plain.rsid_conflicts.n_keys == 1
        store = tmp_path / "resumed.opengwasdb"
        _crash_after_the_fold(monkeypatch, manifest, reference, store, on_rsid_conflict="drop")

        resumed = _build(
            manifest, reference, store, on_rsid_conflict="drop", checkpoint=True, resume=True
        )

        assert resumed.rsid_conflicts == plain.rsid_conflicts
        assert _hybrid_axis_rsids(store) == _hybrid_axis_rsids(plain_store)

    def test_resume_under_a_different_policy_is_refused(self, tmp_path, monkeypatch):
        manifest, reference = _conflict_fixture(tmp_path, CONFLICT_OFF)
        store = tmp_path / "resumed.opengwasdb"
        _crash_after_the_fold(monkeypatch, manifest, reference, store, on_rsid_conflict="drop")

        with pytest.raises(ValueError, match="Differing: on_rsid_conflict"):
            _build(manifest, reference, store, checkpoint=True, resume=True)

    def test_checkpoint_recorded_before_the_option_resumes_as_refuse(
        self, tmp_path, monkeypatch
    ):
        """A checkpoint written before the option existed records no policy;
        every harvest that compared names until then refused a conflict."""
        manifest, reference = _agreeing_fixture(tmp_path)
        store = tmp_path / "old.opengwasdb"
        _crash_after_the_fold(monkeypatch, manifest, reference, store)
        checkpoint = checkpoint_dir_for(store)
        params = read_json(checkpoint / BUILD_PARAMS)
        assert params.pop("on_rsid_conflict") == "refuse"
        write_json(checkpoint / BUILD_PARAMS, params)

        with pytest.raises(ValueError, match="Differing: on_rsid_conflict"):
            _build(
                manifest, reference, store, on_rsid_conflict="drop", checkpoint=True, resume=True
            )
        resumed = _build(manifest, reference, store, checkpoint=True, resume=True)

        assert resumed.rsid_conflicts.policy == "refuse"
        assert validate_store(store).ok
