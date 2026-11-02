"""The presence rule and the association record (spec §15, ADR 0070)."""

from __future__ import annotations

import numpy as np
import pytest

import opengwasdb.associations as oa

NAN = np.nan
INF = np.inf


@pytest.mark.parametrize(
    ("z", "se", "flag", "expected"),
    [
        (1.0, 0.1, 0, oa.State.OBSERVED),
        (1.0, 0.1, 1, oa.State.IMPUTED),
        (NAN, NAN, 0, oa.State.MISSING),
        # An infinite value is a value: SE above float16's range is stored +inf (#338).
        (1.0, INF, 0, oa.State.OBSERVED),
        (1.0, -INF, 1, oa.State.IMPUTED),
        (INF, 0.1, 0, oa.State.OBSERVED),
        # The two markers must agree, whatever the flag says.
        (1.0, NAN, 0, oa.State.PARTIAL),
        (NAN, 0.1, 0, oa.State.PARTIAL),
        (NAN, INF, 0, oa.State.PARTIAL),
        (1.0, NAN, 1, oa.State.PARTIAL),
        (1.0, 0.1, 2, oa.State.FLAG_INVALID),
        (NAN, NAN, 1, oa.State.IMPUTED_MISSING),
    ],
)
def test_classify_states_each_cell_by_its_markers_and_flag(z, se, flag, expected) -> None:
    got = oa.classify(np.array([z], np.float32), np.array([se], np.float32), np.array([flag]))
    assert oa.State(int(got[0])) is expected


def test_no_mask_means_every_present_cell_is_observed() -> None:
    got = oa.classify(np.array([1.0, NAN], np.float32), np.array([0.1, NAN], np.float32))
    assert got.tolist() == [oa.State.OBSERVED, oa.State.MISSING]


def test_a_valid_state_is_its_status() -> None:
    for status in oa.Status:
        assert oa.State(status.value).name == status.name


def test_tally_keeps_the_overlaps_a_single_state_hides() -> None:
    z_missing = np.array([False, True, True, False, True, False])
    se_missing = np.array([False, True, False, True, True, False])
    flags = np.array([0, 0, 1, 1, 1, 2])
    tally = oa.Tally.of(z_missing, se_missing, flags)
    assert tally.count(oa.State.OBSERVED) == 1
    assert tally.count(oa.State.MISSING) == 1
    assert tally.count(oa.State.PARTIAL) == 2
    assert tally.count(oa.State.IMPUTED_MISSING) == 1
    assert tally.count(oa.State.FLAG_INVALID) == 1
    assert (tally.z_only_missing, tally.se_only_missing) == (1, 1)
    # The two partial cells are flagged imputed too; the overlaps say so.
    assert (tally.imputed_z_missing, tally.imputed_se_missing) == (2, 2)
    assert tally.invalid == 4
    assert (tally + tally).invalid == 8


def test_states_reads_markers_not_finiteness() -> None:
    assert oa.states(np.array([False]), np.array([False])).tolist() == [oa.State.OBSERVED]


def _record() -> oa.Associations:
    return oa.Associations.from_decoded(
        variant_index=[4, 1, 7],
        analysis_index=[0, 0, 2],
        z=np.array([2.0, NAN, -1.0]),
        se=np.array([INF, NAN, 0.5]),
        eaf=np.array([0.1, NAN, 0.3]),
        imputed=np.array([0, 0, 1]),
        n=np.array([10.0, NAN, 30.0]),
    )


def test_record_reads_as_the_query_result_mapping() -> None:
    record = _record()
    assert list(record) == [
        "variant_index",
        "analysis_index",
        "z",
        "se",
        "eaf",
        "n",
        "association_status",
    ]
    assert record["association_status"].tolist() == ["observed", "missing", "imputed"]
    assert record["association_status"].dtype == object
    assert {k: record[k].dtype.str for k in ("variant_index", "z", "eaf", "n")} == {
        "variant_index": "<i4",
        "z": "<f4",
        "eaf": "<f4",
        "n": "<f8",
    }
    assert "status" not in record and record.count == 3


def test_record_without_n_has_no_n_key() -> None:
    record = oa.Associations.from_decoded([0], 0, np.array([1.0]), np.array([0.1]))
    assert "n" not in record and np.isnan(record["eaf"]).all()
    with pytest.raises(KeyError):
        record["n"]


def test_from_decoded_refuses_an_invalid_cell_naming_it() -> None:
    with pytest.raises(oa.InvalidAssociations, match=r"analysis 3: partial"):
        oa.Associations.from_decoded([9], [3], np.array([1.0]), np.array([NAN]), source="here")


def test_derived_records_keep_columns_in_lockstep() -> None:
    record = _record()
    observed = record.observed_only()
    assert observed["variant_index"].tolist() == [4, 1]
    assert observed["n"].tolist()[0] == 10.0
    assert record.present()["variant_index"].tolist() == [4, 7]
    assert record.head(1)["z"].tolist() == [2.0]
    ordered = record.in_genomic_order()
    assert ordered["variant_index"].tolist() == [1, 4, 7]
    assert ordered["association_status"].tolist() == ["missing", "observed", "imputed"]
    remapped = record.remap_variants(np.arange(10) * 10)
    assert remapped["variant_index"].tolist() == [40, 10, 70]


def test_concat_keeps_n_only_when_asked() -> None:
    record = _record()
    both = oa.Associations.concat([record, record], with_n=True)
    assert both.count == 6 and both["n"].shape == (6,)
    assert "n" not in oa.Associations.concat([record], with_n=False)
    with pytest.raises(ValueError, match="without n"):
        oa.Associations.concat([record, oa.Associations.empty()], with_n=True)
    assert oa.Associations.concat([], with_n=True)["n"].dtype == np.float64


def _float16_bits(bits: int) -> np.ndarray:
    """A decoded cell whose stored float16 had exactly these bits."""
    return np.array([bits], np.uint16).view(np.float16).astype(np.float32)


@pytest.mark.parametrize("bits", [0x7E01, 0xFE00, 0x7C01])
def test_a_nan_other_than_the_canonical_one_is_refused(bits: int) -> None:
    """Spec §15: a non-canonical marker payload is invalid, not missing."""
    z = np.array([NAN], np.float32)
    assert oa.classify(z, _float16_bits(0x7E00)).tolist() == [oa.State.MISSING]
    assert oa.classify(z, _float16_bits(bits)).tolist() == [oa.State.NON_CANONICAL]
    assert oa.Tally.of_decoded(z, _float16_bits(bits)).invalid == 1
    with pytest.raises(oa.InvalidAssociations, match="non_canonical"):
        oa.Associations.from_decoded([0], [0], z, _float16_bits(bits))


def test_an_integer_plane_has_no_payload_to_be_wrong() -> None:
    assert not oa.non_canonical(np.array([-32768, 5], np.int16)).any()
