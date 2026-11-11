"""The realdata inputs script's parts (ogld#6): the ranking plink2 is given, how a pvar record is
matched to the store's ALID, and the pins that tie the inputs to the store. The comparison itself
is the realdata tier's, and the acceptance run's."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from script_modules import load_script

script = load_script("realdata/make_403_inputs")
Record = script.Record


def test_minus_log10_p_matches_the_normal_tail_and_does_not_underflow() -> None:
    assert script.minus_log10_p(5.45) == pytest.approx(-math.log10(math.erfc(5.45 / math.sqrt(2))))
    assert 7.2 < script.minus_log10_p(5.45) < 7.4, "5e-8 is z ≈ 5.45"
    assert script.minus_log10_p(342.7) > 25_000, "p = 0 in a double, finite here"


def test_minus_log10_p_is_strictly_increasing_across_the_switch_to_the_asymptotic_form() -> None:
    zs = [29.9, 29.99, 30.0, 30.01, 30.1, 38.0, 38.25, 100.0, 100.25]
    values = [script.minus_log10_p(z) for z in zs]
    assert values == sorted(set(values)), "equal |z| must be equal p and a larger |z| a smaller p"
    assert script.minus_log10_p(30.0) == pytest.approx(
        -math.log10(math.erfc(30.0 / math.sqrt(2))), rel=1e-9
    )


def test_a_hit_the_store_does_not_hold_is_counted_and_dropped() -> None:
    hits = [("1:5:A:C", 6.0, "1"), ("1:9:A:C", 7.0, "1"), ("2:3:A:G", 8.0, "2")]
    kept, unmatched = script.matched(hits, {"1:5:A:C", "2:3:A:G"})
    assert [h[0] for h in kept] == ["1:5:A:C", "2:3:A:G"]
    assert unmatched == 1


PVAR = (
    "##fileformat=PVARv1.0\n#CHROM\tPOS\tID\tREF\tALT\n"
    "1\t5\tsnv\tA\tC\n"
    "1\t10\tins\tC\tCT\n"  # ALID 1:10:C:CT with REF = A1
    "1\t10\tdel\tCT\tC\n"  # the same ALID, REF second
    "1\t20\tsym\tA\t<DEL>\n"  # no ALID: ogld refuses a symbolic allele
    "1\t30\tstar\tA\t*\n"
)


def test_a_record_ogld_gives_no_alid_is_counted_not_dropped_silently(tmp_path: Path) -> None:
    (tmp_path / "chr1.pvar").write_text(PVAR)
    found, skipped = script.records_by_alid(tmp_path / "chr1.pvar", {"1:5:A:C", "1:10:C:CT"})
    assert sorted(found) == ["1:10:C:CT", "1:5:A:C"]
    assert [r.name for r in found["1:10:C:CT"]] == ["ins", "del"]
    assert skipped == 2, "the symbolic and the star allele"


def test_acount_columns_are_read_by_name_not_by_position() -> None:
    usual = "#CHROM\tID\tREF\tALT\tPROVISIONAL_REF?\tALT_CTS\tOBS_CT\n1\tv1\tA\tC\tY\t12\t1330\n"
    moved = "OBS_CT\tALT_CTS\t#CHROM\tID\tALT\tREF\n1330\t12\t1\tv1\tC\tA\n"
    assert script.parse_acount(usual) == {"v1": (12, 1330)}
    assert script.parse_acount(moved) == {"v1": (12, 1330)}, "a reordered table reads the same"
    with pytest.raises(SystemExit, match="OBS_CT"):
        script.parse_acount("#CHROM\tID\tALT_CTS\n1\tv1\t12\n")


def pair() -> list:
    return [Record("ins", "1", 10, "C", "CT", "C"), Record("del", "1", 10, "CT", "C", "C")]


def test_of_an_insertion_and_a_deletion_the_store_s_a1_frequency_names_the_one_it_holds() -> None:
    counts = {"ins": (300, 1330), "del": (40, 1330)}  # A1 = C: REF of ins, ALT of del
    # ins has A1 = C = REF: frequency 1 − 300/1330; del has A1 = C = ALT: frequency 40/1330.
    ins_frequency, del_frequency = 1 - 300 / 1330, 40 / 1330
    assert script.pick("1:10:C:CT", pair(), counts, ins_frequency) == "ins"
    assert script.pick("1:10:C:CT", pair(), counts, del_frequency) == "del", (
        "the REF = A1 record failed a filter, so the builder held the other"
    )
    with pytest.raises(SystemExit, match="no record has the store's A1 frequency"):
        script.pick("1:10:C:CT", pair(), counts, 0.5)


def test_a_record_is_picked_by_frequency_when_both_pass_and_the_rule_prefers_ref_is_a1() -> None:
    same = {"ins": (300, 1330), "del": (300, 1330)}
    assert script.pick("1:10:C:CT", pair(), same, 1 - 300 / 1330) == "ins"


def test_resolve_asks_for_counts_only_of_the_records_of_a_shared_alid() -> None:
    asked: list[list[str]] = []

    def counts_of(names: list[str]) -> dict[str, tuple[int, int]]:
        asked.append(names)
        return {"ins": (300, 1330), "del": (40, 1330)}

    found = {"1:10:C:CT": pair(), "1:5:A:C": [Record("snv", "1", 5, "A", "C", "A")]}
    stored = {"1:10:C:CT": 40 / 1330, "1:5:A:C": 0.1}
    assert script.resolve(found, counts_of, stored) == {"del": "1:10:C:CT", "snv": "1:5:A:C"}
    assert asked == [["ins", "del"]]


def test_a_store_alid_with_no_pgen_record_is_refused(tmp_path: Path) -> None:
    (tmp_path / "chr1.pvar").write_text(PVAR)
    with pytest.raises(SystemExit, match="no pgen record"):
        script.prepare_chromosome(
            tmp_path / "chr1",
            tmp_path / "keep",
            tmp_path / "out",
            "1",
            {"1:5:A:C": 0.1, "1:6:A:C": 0.2},
        )


def test_a_keep_file_is_hashed_as_the_builder_hashes_its_samples(tmp_path: Path) -> None:
    keep = tmp_path / "EUR.keep"
    keep.write_text("#IID\nHG2\nHG1\n")
    assert script.keep_sha256(keep) == hashlib.sha256(b"HG1\nHG2\n").hexdigest()


def test_inputs_that_are_not_the_ones_the_store_was_built_from_are_refused(tmp_path: Path) -> None:
    keep = tmp_path / "EUR.keep"
    keep.write_text("#IID\nHG1\nHG2\n")
    (tmp_path / "chr1.pgen").write_bytes(b"pgen")
    good = {
        "source": {"sha256": {"chr1.pgen": hashlib.sha256(b"pgen").hexdigest()}},
        "panel": {"samples_sha256": script.keep_sha256(keep)},
    }
    assert script.check_pins(good, tmp_path, keep)["keep_sha256"] == script.keep_sha256(keep)
    wrong_file = {**good, "source": {"sha256": {"chr1.pgen": "0" * 64}}}
    with pytest.raises(SystemExit, match="not the file the store was built from"):
        script.check_pins(wrong_file, tmp_path, keep)
    wrong_samples = {**good, "panel": {"samples_sha256": "0" * 64}}
    with pytest.raises(SystemExit, match="not the sample list"):
        script.check_pins(wrong_samples, tmp_path, keep)
