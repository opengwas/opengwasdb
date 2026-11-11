"""An LD panel places an indel only by a declared reference, and signs it for A1 (#381).

A panel's id sorts its alleles, and its eigenvectors and EAF are for its `EA`,
the sorted-first allele. An indel's ALID spells its reference first (ADR 0079),
so a panel deletion's A1 is not its `EA`: its eigenvector row and its frequency
must be re-signed before they meet a Store's statistics. The reference comes
from the block's own `REF` column (HGDP AMR), or from the indel-reference table
beside the panel's blocks (`scripts/build_panel_indel_reference.py`); without
either the panel places no indel. A table is read only beside its provenance,
whose checksum it must match.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pysam
import pytest

from opengwasdb.completion.block import run_block
from opengwasdb.completion.ld_panel import (
    INDEL_REFERENCE_PROVENANCE,
    INDEL_REFERENCE_TABLE,
    IndelReferenceTableError,
    PanelPlacement,
    load_block,
    panel_indel_frequencies,
)

#: One block: four SNPs observed, and at 1:2500 an indel whose `EA` is the short
#: allele `G`. Its LD is coded on that `EA`: r = +0.8 with the first SNP.
SNPS = [
    ("1:1000:A:G", 1000, 0.3),
    ("1:2000:C:T", 2000, 0.4),
    ("1:2500:G:GGGGCC", 2500, 0.9),
    ("1:3000:A:C", 3000, 0.35),
    ("1:4000:G:T", 4000, 0.25),
]
INDEL = 2
OBSERVED_Z = {"1:1000:A:G": 5.0, "1:2000:C:T": 1.0, "1:3000:A:C": 0.5, "1:4000:G:T": -1.0}
INSERTION = "1:2500:G:GGGGCC"
DELETION = "1:2500:GGGGCC:G"
#: An SNV in the indel's place, for a block whose `EA` there is its A2.
SNV = "1:2500:G:T"
#: `PanelPlacement.record`'s unplaced indels, and SNVs, when every row is placed.
NO_INDEL_UNPLACED = {
    "no_reference": 0, "no_effect_allele": 0, "effect_allele_not_an_allele": 0,
    "identity_invalid": 0,
}
NO_SNV_UNPLACED = {"no_effect_allele": 0, "effect_allele_not_an_allele": 0, "identity_invalid": 0}


def _ld() -> np.ndarray:
    ld = np.eye(len(SNPS))
    ld[0, INDEL] = ld[INDEL, 0] = 0.8
    ld[1, 3] = ld[3, 1] = 0.3
    return ld


def _write_block(
    root: Path,
    *,
    reference: str | None = None,
    with_ref_column: bool = False,
    effect: dict[str, str] | None = None,
    snps: list[tuple[str, int, float]] = SNPS,
    with_ea_column: bool = True,
) -> Path:
    """One flat-layout block, `REF`/`ALT` columns written when asked. Each
    row's `EA` is its sorted-first allele, as the panels write it, unless
    ``effect`` gives the row's `EA` cell; without ``with_ea_column`` the block
    records no `OA` or `EA` at all."""
    block_dir = root / "EUR" / "1"
    block_dir.mkdir(parents=True)
    alleles = "\tOA\tEA" if with_ea_column else ""
    header = f"CHR\tSNP{alleles}\tEAF\tBP" + ("\tREF\tALT" if with_ref_column else "")
    lines = [header]
    for snp, bp, eaf in snps:
        _chrom, _pos, a1, a2 = snp.split(":")
        ea = (effect or {}).get(snp, a1)
        oa = a1 if ea == a2 else a2
        cells = f"\t{oa}\t{ea}" if with_ea_column else ""
        row = f"1\t{snp}{cells}\t{eaf}\t{bp}"
        if with_ref_column:
            ref = reference if snp == INSERTION and reference else a1
            alt = a2 if ref == a1 else a1
            row += f"\t{ref}\t{alt}"
        lines.append(row)
    tsv = block_dir / "1-5000.tsv"
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    values, vectors = np.linalg.eigh(_ld())
    np.savez_compressed(block_dir / "1-5000.ldeig", values=values[::-1], vectors=vectors[:, ::-1])
    return tsv


def _write_table(root: Path, reference: str, outcome: str = "settled_agree") -> None:
    """The indel-reference table beside the blocks, as the builder writes it."""
    path = root / "EUR" / "indel_reference.tsv"
    path.write_text(
        "#chromosome\tposition\tallele1\tallele2\treference\toutcome\n"
        f"1\t2500\tG\tGGGGCC\t{reference}\t{outcome}\n",
        encoding="utf-8",
    )
    _compress_and_index(path)


def _compress_and_index(path: Path) -> None:
    """Bgzip and tabix-index a plain table as `indel_reference.tsv.gz` beside it,
    with the provenance completion requires of it."""
    import hashlib

    table = path.with_name(INDEL_REFERENCE_TABLE)
    pysam.tabix_compress(str(path), str(table), force=True)
    path.unlink()
    pysam.tabix_index(
        str(table), seq_col=0, start_col=1, end_col=1, meta_char="#", zerobased=False,
        force=True,
    )
    path.with_name(INDEL_REFERENCE_PROVENANCE).write_text(
        json.dumps({
            "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            "counts": {"settled_agree": 1},
            "revision": {"commit": None, "script_modified": None},
        }),
        encoding="utf-8",
    )


def _fills(tsv: Path) -> dict[str, float]:
    """Each filled ALID's z, completing one Analysis observed at the four SNPs."""

    def make_reader(block, alids):
        z = np.array([OBSERVED_Z.get(a, np.nan) if a else np.nan for a in alids])
        se = np.where(np.isfinite(z), 0.1, np.nan)
        return [0], lambda _analysis: (z, se)

    result = run_block(tsv, 0.99, 0.0, None, make_reader)
    assert result is not None
    return {fill.alid: fill.z for fill in result.fills}


@pytest.mark.parametrize("declared_by", ["ref-column", "table"])
def test_a_panel_deletion_is_imputed_for_its_reference_allele(tmp_path: Path, declared_by: str):
    """The deletion's A1, `GGGGCC`, is the other allele of its `EA`, so its LD with
    the first SNP changes sign: its imputed z is the insertion's, negated."""
    if declared_by == "ref-column":
        tsv = _write_block(tmp_path, reference="GGGGCC", with_ref_column=True)
    else:
        tsv = _write_block(tmp_path)
        _write_table(tmp_path, "GGGGCC")

    block = load_block(tsv)
    assert block is not None and block.alids[INDEL] == DELETION
    assert block.flipped.tolist() == [False, False, True, False, False]
    fills = _fills(tsv)

    assert set(fills) == {DELETION}, "the fixture must impute the deletion"
    # Left on its EA, the deletion would take the insertion's positive z.
    assert fills[DELETION] < 0


@pytest.mark.parametrize("declared_by", ["ref-column", "table"])
def test_a_panel_insertion_keeps_the_sign_of_its_ea(tmp_path: Path, declared_by: str):
    if declared_by == "ref-column":
        tsv = _write_block(tmp_path, reference="G", with_ref_column=True)
    else:
        tsv = _write_block(tmp_path)
        _write_table(tmp_path, "G")

    fills = _fills(tsv)

    assert set(fills) == {INSERTION}
    assert fills[INSERTION] > 0


def test_the_two_readings_of_one_panel_indel_have_opposite_signs(tmp_path: Path):
    """The same LD and observations give the deletion and the insertion z of
    opposite sign. (Not exact negations: completion rescales imputed z by a
    polynomial fitted at the observed positions, which has a constant term.)"""
    deletion = _fills(_write_block(tmp_path / "d", reference="GGGGCC", with_ref_column=True))
    insertion = _fills(_write_block(tmp_path / "i", reference="G", with_ref_column=True))

    assert np.sign(deletion[DELETION]) == -np.sign(insertion[INSERTION]) != 0


def test_a_panel_with_no_reference_places_no_indel(tmp_path: Path):
    tsv = _write_block(tmp_path)

    block = load_block(tsv)
    assert block is not None
    assert block.alids[INDEL] is None
    assert (block.indel_reference, block.n_indels, block.n_indels_placed) == ("none", 1, 0)
    assert _unplaced(block) == ({**NO_INDEL_UNPLACED, "no_reference": 1}, NO_SNV_UNPLACED)
    assert _fills(tsv) == {}


def _unplaced(block) -> tuple[dict[str, int], dict[str, int]]:
    """The block's unplaced indels and SNVs by reason, as a completion records them."""
    placement = PanelPlacement()
    placement.add(block)
    record = placement.record()
    indels = record["panel_indels"]
    assert sum(indels["unplaced"].values()) == indels["n_indels"] - indels["n_placed"]
    return indels["unplaced"], record["panel_snvs_unplaced"]


def test_a_block_with_no_ea_column_places_no_row(tmp_path: Path):
    """Nothing says which allele any row's LD and EAF are for: an SNV is not
    taken as coded on its A1, and an indel is not placed either."""
    tsv = _write_block(tmp_path, reference="GGGGCC", with_ref_column=True, with_ea_column=False)

    block = load_block(tsv)

    assert block is not None
    assert block.alids == [None] * len(SNPS)
    assert _unplaced(block) == (
        {**NO_INDEL_UNPLACED, "no_effect_allele": 1}, {**NO_SNV_UNPLACED, "no_effect_allele": 4},
    )
    assert _fills(tsv) == {}


@pytest.mark.parametrize(
    ("snp", "indel"), [("1:2500:G:GNNNCC", True), ("1:2500:G:N", False)], ids=["indel", "snv"]
)
def test_a_row_whose_id_no_alid_can_spell_is_counted(tmp_path: Path, snp: str, indel: bool):
    """The indel's reference, `G`, is found, but its other allele is outside the
    ALID alphabet."""
    snps = [*SNPS[:INDEL], (snp, 2500, 0.9), *SNPS[INDEL + 1:]]
    tsv = _write_block(tmp_path, with_ref_column=True, snps=snps)

    block = load_block(tsv)

    assert block is not None and block.alids[INDEL] is None
    reason = {"identity_invalid": 1}
    assert _unplaced(block) == (
        ({**NO_INDEL_UNPLACED, **reason}, NO_SNV_UNPLACED) if indel
        else (NO_INDEL_UNPLACED, {**NO_SNV_UNPLACED, **reason})
    )


def test_a_table_row_it_could_not_settle_places_nothing(tmp_path: Path):
    tsv = _write_block(tmp_path)
    _write_table(tmp_path, ".", outcome="both_ways")

    block = load_block(tsv)
    assert block is not None and block.alids[INDEL] is None
    assert block.indel_reference == "table"


def test_a_panel_deletion_s_frequency_is_its_reference_allele_s(tmp_path: Path):
    """The panel's EAF is for `G`; the deletion's A1 is `GGGGCC`."""
    _write_block(tmp_path, reference="GGGGCC", with_ref_column=True)

    assert panel_indel_frequencies(tmp_path, "EUR") == {DELETION: pytest.approx(0.1)}


def test_a_panel_insertion_s_frequency_is_its_effect_allele_s(tmp_path: Path):
    """The insertion's A1 is its `EA`, `G`: the panel's EAF, as it stands."""
    _write_block(tmp_path, reference="G", with_ref_column=True)

    assert panel_indel_frequencies(tmp_path, "EUR") == {INSERTION: pytest.approx(0.9)}


@pytest.mark.parametrize(
    ("reference", "eaf"),
    [("GGGGCC", [0.3, 0.4, 0.1, 0.35, 0.25]), ("G", [0.3, 0.4, 0.9, 0.35, 0.25])],
    ids=["deletion", "insertion"],
)
def test_eaf_for_a1_complements_only_a_row_whose_a1_is_not_its_ea(
    tmp_path: Path, reference: str, eaf: list[float]
):
    block = load_block(_write_block(tmp_path, reference=reference, with_ref_column=True))

    assert block is not None
    assert block.eaf_for_a1().tolist() == pytest.approx(eaf)


@pytest.mark.parametrize(("effect", "flipped"), [("T", True), ("G", False)])
def test_an_snv_is_signed_for_its_a1_whichever_allele_is_its_ea(
    tmp_path: Path, effect: str, flipped: bool
):
    """No panel on the host codes an SNV on its A2, but the rule holds one that
    does: its LD (+0.8 with the first SNP, coded on `EA`) and its EAF are
    re-signed for its A1, `G`."""
    snps = [*SNPS[:INDEL], (SNV, 2500, 0.9), *SNPS[INDEL + 1:]]
    tsv = _write_block(tmp_path, snps=snps, effect={SNV: effect})

    block = load_block(tsv)
    assert block is not None
    assert block.alids[INDEL] == SNV
    assert bool(block.flipped[INDEL]) is flipped
    assert block.eaf_for_a1()[INDEL] == pytest.approx(0.1 if flipped else 0.9)
    fills = _fills(tsv)
    assert set(fills) == {SNV}, "the fixture must impute the SNV"
    assert np.sign(fills[SNV]) == (-1 if flipped else 1)


@pytest.mark.parametrize(
    ("row", "cell", "reason"),
    [
        (INSERTION, "", "no_effect_allele"),
        ("1:1000:A:G", "", "no_effect_allele"),
        (INSERTION, "C", "effect_allele_not_an_allele"),
        ("1:1000:A:G", "T", "effect_allele_not_an_allele"),
    ],
)
def test_a_row_whose_ea_is_not_one_of_its_alleles_is_not_placed(
    tmp_path: Path, row: str, cell: str, reason: str
):
    """Nothing says which allele such a row's LD and EAF are for, so it names
    no ALID, and the completion record counts it."""
    tsv = _write_block(tmp_path, reference="GGGGCC", with_ref_column=True, effect={row: cell})

    block = load_block(tsv)

    assert block is not None
    index = [snp for snp, _bp, _eaf in SNPS].index(row)
    assert block.alids[index] is None
    assert not block.flipped[index]
    assert _unplaced(block) == (
        ({**NO_INDEL_UNPLACED, reason: 1}, NO_SNV_UNPLACED) if row == INSERTION
        else (NO_INDEL_UNPLACED, {**NO_SNV_UNPLACED, reason: 1})
    )


def test_a_table_without_its_provenance_is_refused(tmp_path: Path):
    tsv = _write_block(tmp_path)
    _write_table(tmp_path, "GGGGCC")
    (tmp_path / "EUR" / INDEL_REFERENCE_PROVENANCE).unlink()

    with pytest.raises(IndelReferenceTableError, match=INDEL_REFERENCE_PROVENANCE):
        load_block(tsv)


def test_a_table_its_provenance_does_not_checksum_is_refused(tmp_path: Path):
    tsv = _write_block(tmp_path)
    _write_table(tmp_path, "GGGGCC")
    path = tmp_path / "EUR" / INDEL_REFERENCE_PROVENANCE
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["table_sha256"] = "0" * 64
    path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(IndelReferenceTableError, match="table_sha256"):
        load_block(tsv)


@pytest.mark.parametrize("key", ["table_sha256", "counts", "revision"])
@pytest.mark.parametrize("absent", ["missing", "null"])
def test_a_provenance_that_records_no_key_completion_needs_is_refused(
    tmp_path: Path, key: str, absent: str
):
    tsv = _write_block(tmp_path)
    _write_table(tmp_path, "GGGGCC")
    path = tmp_path / "EUR" / INDEL_REFERENCE_PROVENANCE
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if absent == "missing":
        del provenance[key]
    else:
        provenance[key] = None
    path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(IndelReferenceTableError, match=f"records no {key}"):
        load_block(tsv)


def test_the_table_sits_beside_the_blocks_and_hides_no_chromosome(tmp_path: Path):
    """The table and its index are files beside the chromosome directories,
    which the panel's chromosome listing never mistakes for one."""
    from opengwasdb.completion.ld_panel import list_chromosomes

    _write_block(tmp_path)
    _write_table(tmp_path, "GGGGCC")

    assert sorted(p.name for p in (tmp_path / "EUR").iterdir()) == [
        "1", "indel_reference.json", "indel_reference.tsv.gz", "indel_reference.tsv.gz.tbi",
    ]
    assert list_chromosomes(tmp_path, "EUR") == ["1"]


def _reads_through_its_own_handle(tsv: Path) -> bool:
    """In a forked worker: load the block, and say whether the table handle it
    read through was opened by this process."""
    from opengwasdb.completion import ld_panel

    load_block(tsv)
    table = ld_panel.indel_reference_table(tsv.parent.parent)
    return ld_panel._open_indel_reference(str(table)).tabix.opened_by == os.getpid()


def test_a_forked_worker_reads_the_table_through_its_own_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A block pool forks after the parent has read the table (Dense's panel
    enumeration loads every block, Ragged's region scan the blocks it plans). A
    worker reading through the parent's handle shares its file offset with every
    other worker, so their fetches read each other's bytes: on OGS-00003 with 16
    workers one raised and another split a line into too many fields."""
    from concurrent.futures import ProcessPoolExecutor

    from opengwasdb.completion import ld_panel

    tsv = _write_block(tmp_path)
    _write_table(tmp_path, "GGGGCC")
    opened = pysam.TabixFile

    class Recorded:
        """A table handle that remembers the process that opened it."""

        def __init__(self, path: str) -> None:
            self._file = opened(path)
            self.contigs = self._file.contigs
            self.opened_by = os.getpid()

        def fetch(self, *args: object) -> object:
            return self._file.fetch(*args)

    monkeypatch.setattr(ld_panel.pysam, "TabixFile", Recorded)
    ld_panel._open_indel_reference.cache_clear()
    try:
        load_block(tsv)
        table = str(ld_panel.indel_reference_table(tsv.parent.parent))
        assert ld_panel._open_indel_reference(table).tabix.opened_by == os.getpid()
        fork = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=1, mp_context=fork) as pool:
            assert pool.submit(_reads_through_its_own_handle, tsv).result()
    finally:
        ld_panel._open_indel_reference.cache_clear()
