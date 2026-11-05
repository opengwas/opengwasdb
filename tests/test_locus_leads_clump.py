"""plink2 clumping against the Locus Lead rule (prototypes/locus_leads, #402).

A float64 p is 0 past |z| near 39, so plink2 is given -log10 p computed from z.
Stored z is quantised and equal values are common, and plink2 breaks a tie by
panel order whatever order its input is in; so equal values are separated in
the rule's order before plink2 sees them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from prototype_modules import load_prototype

from opengwasdb.stats import p_value_from_z

clump = load_prototype("locus_leads", "clump")
panel = load_prototype("locus_leads", "panel")
rule = load_prototype("locus_leads", "rule")

POSITION = np.array([1_000, 2_000, 3_000])
# Store Variant Indices chosen so the rule's tie winner (5) is not panel-first.
VARIANT_INDEX = np.array([12, 5, 9])


@pytest.fixture(scope="module")
def tiny_panel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three variants on chromosome 1 with identical genotypes in 40 samples (r^2 = 1)."""
    directory = tmp_path_factory.mktemp("panel")
    genotypes = np.random.default_rng(402).choice(["0/0", "0/1", "1/1"], size=40)
    samples = [f"S{i}" for i in range(40)]
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=1,length=10000>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
                        "FORMAT", *samples])]
    for position, (ref, alt) in zip(POSITION, [("A", "G"), ("C", "T"), ("G", "A")], strict=True):
        lines.append("\t".join(["1", str(position), f"1:{position}:{ref}:{alt}", ref, alt,
                                ".", "PASS", ".", "GT", *genotypes]))
    (directory / "tiny.vcf").write_text("\n".join(lines) + "\n")
    subprocess.run([panel.plink2(), "--vcf", str(directory / "tiny.vcf"), "--make-pgen", "vzs",
                    "--out", str(directory / "chr1")], check=True, capture_output=True)
    return directory


def _panel_ids(directory: Path) -> np.ndarray:
    variants = panel.Panel(directory).variants("1")
    match = panel.match_to_panel(np.array(["1"] * 3), POSITION, variants.ref, variants.alt,
                                 {"1": variants})
    assert list(match.status) == ["matched"] * 3
    ids: np.ndarray = match.panel_id
    return ids


def test_neg_log10_p_stays_finite_and_ordered_where_p_is_zero() -> None:
    z = np.array([39.5, -41.25, 40.0])
    assert [p_value_from_z(value) for value in z] == [0.0, 0.0, 0.0]

    neg = clump.neg_log10_p(z)

    # Mills' ratio, Phi(-z) ~ phi(z)/z (1 - 1/z^2 + 3/z^4), gives 349.136 at |z| = 40.
    assert neg[2] == pytest.approx(349.136, abs=1e-3)
    assert neg[1] > neg[2] > neg[0]


def test_plink2_and_the_rule_pick_the_same_lead_among_hits_whose_p_is_zero(
    tiny_panel: Path, tmp_path: Path
) -> None:
    abs_z = np.array([39.5, 41.25, 40.0])  # the strongest is second in panel order
    assert [p_value_from_z(value) for value in abs_z] == [0.0, 0.0, 0.0]
    ids = _panel_ids(tiny_panel)

    lead_of = rule.locus_leads(np.array(["1"] * 3), POSITION, np.zeros(3), abs_z, VARIANT_INDEX,
                               d_cm=0.5, w_bp=1_000_000)
    ranked = clump.ranked_neg_log10_p(abs_z, VARIANT_INDEX)
    index_of = clump.clump_rows(panel.Panel(tiny_panel), np.array(["1"] * 3), ids,
                                ranked.neg_log10_p, r2=0.01, work_dir=tmp_path)
    raw = clump.run_clump(tiny_panel / "chr1", ids, np.array([p_value_from_z(z) for z in abs_z]),
                          log10=False, r2=0.01, out=tmp_path / "raw")

    np.testing.assert_array_equal(lead_of, [1, 1, 1])
    np.testing.assert_array_equal(index_of, [1, 1, 1])
    # Fed raw p, every p is 0 and plink2 takes the panel-first variant instead.
    assert list(raw) == [ids[0]]


def test_plink2_breaks_equal_values_by_panel_order_so_ties_are_separated_in_rule_order(
    tiny_panel: Path, tmp_path: Path
) -> None:
    abs_z = np.full(3, 40.0)
    ids = _panel_ids(tiny_panel)
    rule_first = rule.rule_order(abs_z, VARIANT_INDEX)
    assert list(rule_first) == [1, 2, 0], "the rule's tie winner must not be panel-first"

    # Equal values, written in the rule's order: plink2 still takes panel order.
    equal = clump.run_clump(tiny_panel / "chr1", ids[rule_first],
                            clump.neg_log10_p(abs_z)[rule_first], log10=True, r2=0.01,
                            out=tmp_path / "equal")
    ranked = clump.ranked_neg_log10_p(abs_z, VARIANT_INDEX)
    separated = clump.run_clump(tiny_panel / "chr1", ids, ranked.neg_log10_p, log10=True,
                                r2=0.01, out=tmp_path / "separated")

    assert list(equal) == [ids[0]]
    assert list(separated) == [ids[1]]
    assert ranked.tied_rows == 3 and ranked.tied_groups == 1
