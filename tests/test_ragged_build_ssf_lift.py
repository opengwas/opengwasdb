"""`build-ragged-ssf` lifts a source that declares a build other than GRCh38 (#386).

A Ragged SSF Store is one build, GRCh38. A GRCh37 source is oriented in its own
build and lifted through the routing's `place_sites`, whose reverse-strand
blocks re-orient its statistics; the fixtures' chains are `liftover_fixtures`',
so the strand of every block is known by construction.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from liftover_fixtures import (
    STRAND_CHAIN_SIZE,
    STRAND_ROWS,
    assert_strand_statistics,
    write_two_strand_chain,
)
from stored_rows import stored_rows

from opengwasdb.admission import admission_record_for
from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.query import query_store
from opengwasdb.variants.axis import VariantAxis

_HEADER = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
    "standard_error", "effect_allele_frequency", "rsid",
]

#: `STRAND_ROWS` as GWAS-SSF rows on hg19: the VCF's ALT is the effect allele,
#: with beta 0.2, SE 0.1 and frequency 0.2, so each is z = 2 on its own allele.
_GRCH37_ROWS = [
    {
        "chromosome": chrom, "base_pair_location": pos, "effect_allele": alt,
        "other_allele": ref, "beta": 0.2, "standard_error": 0.1,
        "effect_allele_frequency": 0.2, "rsid": f"rs{i}",
    }
    for i, (_name, chrom, pos, ref, alt, _alid, _z, _eaf) in enumerate(STRAND_ROWS, start=1)
]

#: The same three studies as a GRCh38 source reports them, derived by hand from
#: the two-strand chain: chr2 mirrors a position `p` to `size - p + 1` and
#: complements both alleles (the effect allele stays the same study's, now
#: spelled on the other strand), chr3 is the identity.
_GRCH38_ROWS = [
    {**row, **moved}
    for row, moved in zip(
        _GRCH37_ROWS,
        [
            {"base_pair_location": STRAND_CHAIN_SIZE - 1_000 + 1,
             "effect_allele": "T", "other_allele": "C"},
            {"base_pair_location": STRAND_CHAIN_SIZE - 2_000 + 1,
             "effect_allele": "T", "other_allele": "A"},
            {},
        ],
        strict=True,
    )
]


def _write_source(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _HEADER) + "\n")


def _inputs(
    tmp_path: Path, analyses: list[tuple[str, str, list[dict]]]
) -> tuple[Path, Path]:
    """A manifest and source directory for ``(analysis_id, build, rows)`` analyses."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    lines = ["analysis_index\tanalysis_id\tfiltered_file\tsource_genome_build"]
    for index, (analysis_id, build, rows) in enumerate(analyses):
        _write_source(source_dir / f"{analysis_id}.tsv.gz", rows)
        lines.append(f"{index}\t{analysis_id}\t{analysis_id}.tsv.gz\t{build}")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, source_dir


def _build(tmp_path: Path, analyses, name: str = "out", **options) -> Path:
    manifest, source_dir = _stage(tmp_path, name, analyses)
    out = tmp_path / f"{name}.opengwasdb"
    build_ragged_from_ssf(
        manifest, source_dir, out, store_id="t", release_id="v1",
        association_coverage="cis_and_signals", allow_unverified_eaf=True, **options,
    )
    return out


def _stage(tmp_path: Path, name: str, analyses) -> tuple[Path, Path]:
    directory = tmp_path / name
    directory.mkdir()
    return _inputs(directory, analyses)


def _cells(store: Path, analysis: int) -> dict[str, tuple[float, float, float, str]]:
    """Each stored ALID's ``(z, se, eaf, rsid)`` for one Analysis."""
    rsid = {v.variant_index: (v.alid, v.rsid) for v in VariantAxis(store).all()}
    rows = stored_rows(store, analysis)
    return {
        rsid[int(i)][0]: (float(z), float(se), float(eaf), rsid[int(i)][1])
        for i, z, se, eaf in zip(rows.variant_index, rows.z, rows.se, rows.eaf, strict=True)
    }


@pytest.fixture()
def chain(tmp_path: Path) -> Path:
    return write_two_strand_chain(tmp_path / "two-strand.chain")


def _admission(store: Path) -> list[dict[str, Any]]:
    record = admission_record_for(store)
    return json.loads((record / "admission.json").read_text(encoding="utf-8"))["analyses"]


def test_the_fixtures_are_meaningful(tmp_path):
    """The GRCh38 equivalent holds the three ALIDs the chain arithmetic gives, one of
    them on the reverse strand: without this the comparison below could pass on empty."""
    store = _build(tmp_path, [("a", "GRCh38", _GRCH38_ROWS)])
    assert sorted(_cells(store, 0)) == sorted(row[5] for row in STRAND_ROWS)
    assert_strand_statistics(query_store(store), "a")


def test_a_grch37_source_stores_the_cells_of_its_grch38_equivalent(tmp_path, chain):
    lifted = _build(tmp_path, [("a", "GRCh37", _GRCH37_ROWS)], "lifted", chain_file=chain)
    direct = _build(tmp_path, [("a", "GRCh38", _GRCH38_ROWS)], "direct")
    assert _cells(lifted, 0) == _cells(direct, 0)
    assert len(_cells(lifted, 0)) == 3


def test_a_reverse_strand_site_is_re_oriented_after_the_lift(tmp_path, chain):
    """`G/A` at hg19 chr2:1000 is `C/T` on the reverse strand: the effect allele A
    is T there, the larger allele, so z is negated and the frequency complemented."""
    store = _build(tmp_path, [("a", "GRCh37", _GRCH37_ROWS)], chain_file=chain)
    assert_strand_statistics(query_store(store), "a")


def test_a_lifted_store_records_its_lift_and_a_grch38_one_records_none(tmp_path, chain):
    lifted = _build(tmp_path, [("a", "GRCh37", _GRCH37_ROWS)], "lifted", chain_file=chain)
    direct = _build(tmp_path, [("a", "GRCh38", _GRCH38_ROWS)], "direct", chain_file=chain)
    provenance = json.loads((lifted / "manifest.json").read_text(encoding="utf-8"))["provenance"]
    assert provenance["lift"] == {
        "source_build": "hg19",
        "chain_file": str(chain),
        "n_lifted": 3,
        "n_unmapped": 0,
        "n_ambiguous": 0,
        "reference": None,
        "n_alleles_verified": None,
        "n_reference_other_variant": None,
        "n_reference_absent": None,
    }
    assert "lift" not in json.loads((direct / "manifest.json").read_text(encoding="utf-8"))[
        "provenance"
    ]


def test_an_unmapped_site_is_dropped_and_counted_in_the_admission_record(tmp_path, chain):
    """The chain covers chr2 and chr3 only: chr4's site has nowhere to go."""
    unmapped = {**_GRCH37_ROWS[2], "chromosome": "4", "rsid": "rs_unmapped"}
    store = _build(
        tmp_path, [("a", "GRCh37", [*_GRCH37_ROWS, unmapped])],
        chain_file=chain, liftover_failure_threshold=0.5,
    )
    assert sorted(_cells(store, 0)) == sorted(row[5] for row in STRAND_ROWS)
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"lift_unmapped": 1}
    assert (analysis["examined"], analysis["admitted"]) == (4, 3)
    with gzip.open(admission_record_for(store) / "dropped" / "0.tsv.gz", "rt") as fh:
        assert fh.read().splitlines()[1:] == ["4:1000:A:G\tlift_unmapped"]
    provenance = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]
    assert (provenance["lift"]["n_lifted"], provenance["lift"]["n_unmapped"]) == (3, 1)


def test_more_unmapped_sites_than_the_threshold_allows_fail_the_build(tmp_path, chain):
    unmapped = {**_GRCH37_ROWS[2], "chromosome": "4"}
    manifest, source_dir = _stage(tmp_path, "in", [("a", "GRCh37", [*_GRCH37_ROWS, unmapped])])
    with pytest.raises(LiftoverFailureError, match="exceeds threshold 1.0%"):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=chain,
        )


def test_a_site_both_builds_declare_is_ambiguous_and_dropped_from_both(tmp_path, chain, caplog):
    """`3:1000:G:A` spelt by a GRCh38 Analysis and by a GRCh37 one is two loci that
    share a string, as the routing's union treats it: no stored row may be
    attributed to the wrong one, so both Analyses lose the row and say so."""
    other = {**_GRCH37_ROWS[2], "base_pair_location": 5_000, "rsid": "rs9"}
    with caplog.at_level(logging.WARNING):
        store = _build(
            tmp_path,
            [
                ("old", "GRCh37", [_GRCH37_ROWS[2], other, _GRCH37_ROWS[0]]),
                ("new", "GRCh38", [_GRCH37_ROWS[2], other, _GRCH38_ROWS[1]]),
            ],
            chain_file=chain, liftover_failure_threshold=0.5,
        )
    assert "2 raw variant tuple(s) declared both hg38 and hg19" in caplog.text
    # Both sites are declared by both Analyses: each loses both rows.
    assert sorted(_cells(store, 0)) == ["2:999001:C:T"]
    assert sorted(_cells(store, 1)) == ["2:998001:A:T"]
    old, new = _admission(store)
    assert old["dropped"] == new["dropped"] == {"lift_ambiguous": 2}
    assert (old["examined"], old["admitted"]) == (3, 1)
    provenance = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]
    assert provenance["lift"]["n_ambiguous"] == 2
    assert "3:1000:A:G" not in {v.alid for v in VariantAxis(store).all()}


def test_a_grch38_analysis_beside_a_grch37_one_keeps_its_rows_where_no_site_is_shared(
    tmp_path, chain
):
    store = _build(
        tmp_path,
        [("old", "GRCh37", _GRCH37_ROWS[:2]), ("new", "GRCh38", _GRCH38_ROWS[:2])],
        chain_file=chain,
    )
    assert _cells(store, 0) == _cells(store, 1)
    assert len(_cells(store, 0)) == 2
    assert all(not analysis["dropped"] for analysis in _admission(store))


def test_two_source_spellings_lifting_onto_one_variant_collapse_by_the_duplicate_rule(
    tmp_path, chain
):
    """`G/A` and `A/G` at one hg19 site are two routing keys and, lifted, one ALID:
    an Analysis stating both, in agreement, keeps one association."""
    twice = [_GRCH37_ROWS[0], {**_GRCH37_ROWS[0], "effect_allele": "G", "other_allele": "A",
                               "beta": -0.2, "effect_allele_frequency": 0.8}]
    store = _build(tmp_path, [("a", "GRCh37", twice)], chain_file=chain)
    assert list(_cells(store, 0)) == ["2:999001:C:T"]
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"duplicate_collapsed": 1}


def test_a_non_grch38_source_without_a_chain_file_is_refused_before_anything_is_read(tmp_path):
    manifest, source_dir = _stage(tmp_path, "in", [("a", "GRCh38", _GRCH38_ROWS),
                                                    ("old", "GRCh37", _GRCH37_ROWS)])
    (source_dir / "a.tsv.gz").unlink()  # reading it would raise a different error
    with pytest.raises(
        ValueError, match=r"analysis 'old' declares source_genome_build hg19.*--chain-file"
    ):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals",
        )
    assert not (tmp_path / "out.opengwasdb").exists()


def test_a_chain_file_that_does_not_exist_is_refused(tmp_path):
    manifest, source_dir = _stage(tmp_path, "in", [("old", "GRCh37", _GRCH37_ROWS)])
    with pytest.raises(FileNotFoundError, match="analysis 'old'.*not-there.chain"):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=tmp_path / "not-there.chain",
        )


def test_a_source_declaring_no_build_is_refused_whatever_the_chain(tmp_path, chain):
    manifest, source_dir = _stage(tmp_path, "in", [("old", "", _GRCH37_ROWS)])
    with pytest.raises(ValueError, match="analysis 'old' declares no source_genome_build"):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=chain,
        )


def test_a_build_the_lift_cannot_read_is_refused_naming_its_analysis(tmp_path, chain):
    manifest, source_dir = _stage(tmp_path, "in", [("old", "hg18", _GRCH37_ROWS)])
    with pytest.raises(ValueError, match="Unknown genome build 'hg18'"):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=chain,
        )


def test_the_cli_takes_a_chain_file_and_a_failure_threshold(tmp_path, chain):
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    unmapped = {**_GRCH37_ROWS[2], "chromosome": "4"}
    manifest, source_dir = _stage(tmp_path, "in", [("a", "GRCh37", [*_GRCH37_ROWS, unmapped])])
    out = tmp_path / "cli.opengwasdb"
    args = [
        "build-ragged-ssf", str(manifest), str(source_dir), str(out),
        "--admission-record", f"{out}.admission", "--store-id", "s", "--release-id", "v1",
        "--association-coverage", "cis_and_signals", "--allow-unverified-eaf",
        "--chain-file", str(chain),
    ]
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0 and "exceeds threshold" in str(refused.exception)
    built = CliRunner().invoke(app, [*args, "--liftover-failure-threshold", "0.5"])
    assert built.exit_code == 0, built.output
    assert json.loads(built.output.strip().splitlines()[-1])["n_associations"] == 3


def test_a_wholly_unmapped_lift_is_refused_naming_each_analysis(tmp_path, chain):
    rows = [{**_GRCH37_ROWS[2], "chromosome": "4"}, {**_GRCH37_ROWS[0], "chromosome": "5"}]
    manifest, source_dir = _stage(tmp_path, "in", [("a", "GRCh37", rows)])
    with pytest.raises(
        ValueError, match=r"placed no variant.*analysis 'a': 2 lift_unmapped, 0 lift_ambiguous"
    ):
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=chain,
            liftover_failure_threshold=1.0,
        )
    assert _admission(tmp_path / "out.opengwasdb")[0]["dropped"] == {"lift_unmapped": 2}


def test_a_wholly_ambiguous_lift_is_refused_naming_each_analysis(tmp_path, chain):
    manifest, source_dir = _stage(
        tmp_path, "in",
        [("old", "GRCh37", [_GRCH37_ROWS[2]]), ("new", "GRCh38", [_GRCH37_ROWS[2]])],
    )
    with pytest.raises(ValueError) as refused:
        build_ragged_from_ssf(
            manifest, source_dir, tmp_path / "out.opengwasdb", store_id="t", release_id="v1",
            association_coverage="cis_and_signals", chain_file=chain,
        )
    assert "placed no variant" in str(refused.value)
    assert "analysis 'old': 0 lift_unmapped, 1 lift_ambiguous" in str(refused.value)
    assert "analysis 'new': 0 lift_unmapped, 1 lift_ambiguous" in str(refused.value)
