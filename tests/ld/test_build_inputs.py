"""ogld#5: what a build refuses, and what its filters drop and count (build-inputs §2, §4–§7).

Each refusal is an `opengwasdb.ld.InputError` naming the panel and what was wrong; each drop is
counted
under the first filter it fails. The inputs are the Test Panel's chr22 records, edited: the
Test Panel itself has no missing call, no failing FILTER, no INFO, no record off the autosomes
and no zero-variance site (its README), so each test makes the one it needs.
"""

from __future__ import annotations

from pathlib import Path

import build_fixtures as fx
import numpy as np
import numpy.typing as npt
import pytest
import testdata

import opengwasdb.ld
from opengwasdb.ld.store import StoreReader

N = 771


def chr22() -> npt.NDArray[np.int64]:
    return np.flatnonzero(testdata.variants().chrom == "22").astype(np.int64)


def passing22() -> npt.NDArray[np.int64]:
    v = testdata.variants()
    return np.flatnonzero((v.chrom == "22") & v.passes_filters).astype(np.int64)


def edited(rows: npt.NDArray[np.int64], edit: dict[int, list[str]]) -> list[str]:
    """The pvar of `rows`, header first, with record `row`'s fields replaced by `edit[row]`."""
    text = fx.pvar_lines()
    lines = [text[0]]
    for row in rows:
        fields = text[1 + int(row)].split("\t")
        lines.append("\t".join(edit.get(int(row), fields)))
    return lines


def fields(row: int) -> list[str]:
    return fx.pvar_lines()[1 + row].split("\t")


# ------------------------------------------------------------------------------ missing calls


def test_missing_calls_are_counted_over_the_whole_input(tmp_path: Path) -> None:
    rows = chr22()
    calls = fx.hard_calls()[rows].copy()
    first, later = 3, len(rows) - 2
    calls[first, 10] = calls[later, 20] = calls[later, 21] = -9
    pgen = fx.write_fileset(tmp_path / "missing", rows, genotypes=calls)
    with pytest.raises(opengwasdb.ld.InputError, match="has 3 missing genotype") as refused:
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    assert fields(int(rows[first]))[2] in str(refused.value), "it names the first by ID"
    assert fx.PANEL_ID in str(refused.value)
    assert not (tmp_path / "store").exists(), "nothing is written for a refused input"


def test_a_missing_call_is_refused_before_a_missing_filter_column(tmp_path: Path) -> None:
    # The shipped gnomAD v3.1.2 pgens have both: no FILTER column and missing calls. The
    # refusal is the missing calls, which a converted pvar cannot fix.
    rows = chr22()
    calls = fx.hard_calls()[rows].copy()
    calls[0, 0] = -9
    lines = [line.rsplit("\t", 1)[0] for line in edited(rows, {})]
    assert lines[0].endswith("ALT")
    pgen = fx.write_fileset(tmp_path / "both", rows, genotypes=calls, lines=lines)
    with pytest.raises(opengwasdb.ld.InputError, match="1 missing genotype"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


# ------------------------------------------------------------------------------ record rules


def test_a_pvar_without_a_filter_column_is_refused(tmp_path: Path) -> None:
    rows = chr22()
    lines = [line.rsplit("\t", 1)[0] for line in edited(rows, {})]
    pgen = fx.write_fileset(tmp_path / "nofilter", rows, lines=lines)
    with pytest.raises(opengwasdb.ld.InputError, match="no FILTER column"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


def test_a_pvar_without_its_columns_is_refused(tmp_path: Path) -> None:
    rows = chr22()
    lines = edited(rows, {})
    lines[0] = lines[0].replace("REF", "ALLELE1")
    pgen = fx.write_fileset(tmp_path / "noref", rows, lines=lines)
    with pytest.raises(opengwasdb.ld.InputError, match="it has no REF"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


@pytest.mark.parametrize("label", ["X", "chrX", "23", "MT", "chrY", "HLA-A"])
def test_a_record_off_the_autosomes_is_refused(tmp_path: Path, label: str) -> None:
    rows = chr22()
    last = int(rows[-1])
    record = fields(last)
    pgen = fx.write_fileset(tmp_path / "x", rows, lines=edited(rows, {last: [label, *record[1:]]}))
    with pytest.raises(opengwasdb.ld.InputError, match="only autosomes") as refused:
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    assert record[2] in str(refused.value)


def test_a_missing_chromosome_label_is_refused(tmp_path: Path) -> None:
    rows = chr22()
    first = int(rows[0])
    pgen = fx.write_fileset(
        tmp_path / "x", rows, lines=edited(rows, {first: [".", *fields(first)[1:]]})
    )
    with pytest.raises(opengwasdb.ld.InputError, match="chromosome is missing"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


def test_a_chr_prefix_is_the_engine_s_label(tmp_path: Path) -> None:
    rows = chr22()
    lines = edited(rows, {int(r): ["chr22", *fields(int(r))[1:]] for r in rows})
    pgen = fx.write_fileset(tmp_path / "chr", rows, lines=lines)
    opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    reader = StoreReader.open(tmp_path / "store")
    assert (reader.rows_of(fx.alids(passing22())) >= 0).all()


@pytest.mark.parametrize(("ref", "alt"), [("N", "A"), ("A", "*"), ("A", "<DEL>"), ("A", ".")])
def test_an_allele_outside_acgt_is_refused(tmp_path: Path, ref: str, alt: str) -> None:
    rows = chr22()
    row = int(rows[5])
    record = fields(row)
    lines = edited(rows, {row: [*record[:3], ref, alt, *record[5:]]})
    pgen = fx.write_fileset(tmp_path / "allele", rows, lines=lines)
    with pytest.raises(opengwasdb.ld.InputError, match="alleles must be A, C, G and T") as refused:
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    assert record[2] in str(refused.value)


def test_lower_case_alleles_are_the_engine_s_upper_case(tmp_path: Path) -> None:
    rows = chr22()
    row = int(passing22()[0])
    record = fields(row)
    lines = edited(rows, {row: [*record[:3], record[3].lower(), record[4].lower(), *record[5:]]})
    opengwasdb.ld.build(
        fx.inputs(fx.write_fileset(tmp_path / "lower", rows, lines=lines)), tmp_path / "store"
    )
    assert StoreReader.open(tmp_path / "store").rows_of(fx.alids([row]))[0] >= 0


def test_two_snvs_sharing_an_alid_are_refused_naming_both(tmp_path: Path) -> None:
    rows = passing22()[:2]
    one = fields(int(rows[0]))
    lines = edited(rows, {int(rows[1]): [one[0], one[1], "copy", one[3], one[4], one[5]]})
    pgen = fx.write_fileset(tmp_path / "twice", rows, lines=lines)
    with pytest.raises(opengwasdb.ld.InputError, match="not a reversed-indel pair") as refused:
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    assert one[2] in str(refused.value)
    assert "copy" in str(refused.value)


def test_a_chromosome_in_two_places_is_refused(tmp_path: Path) -> None:
    v = testdata.variants()
    on21 = np.flatnonzero(v.chrom == "21")
    first = fx.write_fileset(tmp_path / "a", on21)
    second = fx.write_fileset(tmp_path / "b", on21[:50])
    with pytest.raises(opengwasdb.ld.InputError, match="chromosome 21 appears in two places"):
        opengwasdb.ld.build(fx.inputs(first, pgen=(first, second)), tmp_path / "store")


def test_a_pvar_and_pgen_of_different_lengths_are_refused(tmp_path: Path) -> None:
    rows = chr22()
    pgen = fx.write_fileset(tmp_path / "short", rows)
    pvar = pgen.with_suffix(".pvar")
    pvar.write_text("\n".join(pvar.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(opengwasdb.ld.InputError, match="records but"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


# ---------------------------------------------------------------------------------- filters


def test_each_dropped_record_is_counted_under_the_first_filter_it_fails(tmp_path: Path) -> None:
    rows = chr22()
    good = passing22()
    lowqual, het, rare = (
        int(good[0]),
        int(good[1]),
        int(
            np.flatnonzero(
                ~testdata.variants().passes_filters & (testdata.variants().chrom == "22")
            )[0]
        ),
    )
    calls = fx.hard_calls()[rows].copy()
    at = {int(r): i for i, r in enumerate(rows)}
    calls[at[het]] = 1  # every sample heterozygous: MAC 771, and no variance
    record = fields(lowqual)
    lines = edited(rows, {lowqual: [*record[:5], "LowQual"], rare: [*fields(rare)[:5], "LowQual"]})
    pgen = fx.write_fileset(tmp_path / "filters", rows, genotypes=calls, lines=lines)
    provenance = opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    dropped = provenance["variants_dropped"]
    # The rare record fails FILTER first, so it is counted there, not under `mac`.
    assert dropped["filter"] == 2
    assert dropped["mac"] == int((testdata.variants().chrom == "22").sum() - len(good)) - 1
    assert dropped["zero_variance"] == 1
    assert dropped["info"] == 0
    assert provenance["n_variants"] == len(good) - 2
    found = StoreReader.open(tmp_path / "store").rows_of(fx.alids([lowqual, het, int(good[2])]))
    assert found.tolist()[:2] == [-1, -1]
    assert found[2] >= 0


def test_mac_is_counted_on_the_kept_samples(tmp_path: Path) -> None:
    # With mac_min one above a variant's MAC on the 771 samples, that variant is dropped.
    v = testdata.variants()
    row = int(passing22()[0])
    alt = int(fx.hard_calls()[row].sum())
    mac = min(alt, 2 * N - alt)
    rows = chr22()
    pgen = fx.write_fileset(tmp_path / "mac", rows)
    at = opengwasdb.ld.build(fx.inputs(pgen, mac_min=mac), tmp_path / "at")
    above = opengwasdb.ld.build(fx.inputs(pgen, mac_min=mac + 1), tmp_path / "above")
    assert StoreReader.open(tmp_path / "at").rows_of(fx.alids([row]))[0] >= 0
    assert StoreReader.open(tmp_path / "above").rows_of(fx.alids([row]))[0] == -1
    assert above["variant_filters"]["mac_min"] == mac + 1
    assert above["variants_dropped"]["mac"] > at["variants_dropped"]["mac"]
    assert v.chrom[row] == "22"


def with_info(rows: npt.NDArray[np.int64], score: dict[int, str]) -> list[str]:
    """The pvar of `rows` with an INFO column: `score[row]`, or R2=0.95."""
    text = edited(rows, {})
    lines = [text[0] + "\tINFO"]
    for row, line in zip(rows, text[1:], strict=True):
        lines.append(line + "\t" + score.get(int(row), "R2=0.95"))
    return lines


def test_an_info_floor_drops_and_counts_what_is_at_or_below_it(tmp_path: Path) -> None:
    rows, good = chr22(), passing22()
    score = {int(good[0]): "AF=0.1;R2=0.8", int(good[1]): "INFO=0.5", int(good[2]): "INFO=0.81"}
    pgen = fx.write_fileset(tmp_path / "info", rows, lines=with_info(rows, score))
    provenance = opengwasdb.ld.build(fx.inputs(pgen, info_min=0.8), tmp_path / "store")
    assert provenance["variants_dropped"]["info"] == 2, "INFO > info_min: 0.8 itself fails"
    assert provenance["variant_filters"]["info_min"] == 0.8
    found = StoreReader.open(tmp_path / "store").rows_of(fx.alids(good[:3]))
    assert found.tolist()[:2] == [-1, -1]
    assert found[2] >= 0


@pytest.mark.parametrize(
    ("info", "says"),
    [("AF=0.1", "neither"), ("R2=0.9;INFO=0.9", "R2 and INFO"), ("R2=high", "not a number")],
)
def test_an_info_score_must_be_one_number(tmp_path: Path, info: str, says: str) -> None:
    rows = chr22()
    row = int(passing22()[3])
    pgen = fx.write_fileset(tmp_path / "info", rows, lines=with_info(rows, {row: info}))
    with pytest.raises(opengwasdb.ld.InputError, match=says) as refused:
        opengwasdb.ld.build(fx.inputs(pgen, info_min=0.8), tmp_path / "store")
    assert fields(row)[2] in str(refused.value)


def test_a_reversed_indel_pair_keeps_the_record_whose_ref_is_a1(tmp_path: Path) -> None:
    # `TA/T` then `T/TA` at one position share ALID T:TA (A1 = T). The engine keeps the record
    # whose REF is A1, the second, so the store's A1 frequency is that record's REF frequency.
    rows = chr22()
    first, second = (int(r) for r in passing22()[:2])
    chrom, position = fields(first)[0], fields(first)[1]
    lines = edited(
        rows,
        {
            first: [chrom, position, "del", "TA", "T", "."],
            second: [chrom, position, "ins", "T", "TA", "."],
        },
    )
    pgen = fx.write_fileset(tmp_path / "indel", rows, lines=lines)
    provenance = opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")
    assert provenance["variants_dropped"]["duplicate_reversed_indel"] == 1
    reader = StoreReader.open(tmp_path / "store")
    row = reader.rows_of([f"{chrom}:{position}:T:TA"])
    stored = float(reader.variants(row).a1_frequency[0])
    ref_frequency = 1 - fx.hard_calls()[second].mean() / 2
    other = 1 - fx.hard_calls()[first].mean() / 2
    assert abs(ref_frequency - other) > 0.01, (
        "the two records must differ for this to mean anything"
    )
    assert stored == pytest.approx(ref_frequency, abs=1e-6)


# -------------------------------------------------------------------------- keep and PC File


def keep_file(tmp_path: Path, lines: list[str]) -> Path:
    out = tmp_path / "samples.keep"
    out.write_text("\n".join(lines) + "\n")
    return out


def iids() -> list[str]:
    return testdata.panel().keep.read_text().splitlines()[1:]


@pytest.mark.parametrize(
    ("lines", "says"),
    [
        (["HG00096", "HG00097"], "must start with '#IID'"),
        (["#IID"], "lists no sample"),
        (["#IID", "HG00096", "HG00096"], "duplicate IID: HG00096"),
        (["#IID", "HG00096", "NOT-A-SAMPLE"], "lacks 1 keep-file sample"),
    ],
)
def test_a_keep_file_is_refused(tmp_path: Path, lines: list[str], says: str) -> None:
    keep = keep_file(tmp_path, lines)
    with pytest.raises(opengwasdb.ld.InputError, match=says):
        opengwasdb.ld.build(fx.inputs(testdata.panel().pgen, keep=keep), tmp_path / "store")


def test_a_keep_file_subset_is_the_panel_s_n(tmp_path: Path) -> None:
    kept = iids()[:500]
    keep = keep_file(tmp_path, ["#IID", *kept])
    pgen = fx.write_fileset(tmp_path / "sub", chr22())
    provenance = opengwasdb.ld.build(fx.inputs(pgen, keep=keep), tmp_path / "store")
    assert provenance["panel"]["n"] == 500


def pc_file(tmp_path: Path, edit: str) -> Path:
    lines = testdata.panel().pcs.read_text().splitlines()
    if edit == "drop a sample":
        lines = lines[:-1]
    elif edit == "nan":
        cells = lines[5].split("\t")
        cells[3] = "nan"
        lines[5] = "\t".join(cells)
    elif edit == "repeat":
        lines.append(lines[1])
    elif edit == "rename":
        lines[0] = lines[0].replace("PC2\t", "PCtwo\t")
    elif edit == "word":
        lines[2] = lines[2].replace("\t", "\tx", 1)
    elif edit == "short":
        lines[2] = lines[2].rsplit("\t", 1)[0]
    elif edit == "fid":
        lines = ["#FID\t" + lines[0].replace("#IID", "IID"), *("0\t" + line for line in lines[1:])]
    out = tmp_path / "pcs.eigenvec"
    out.write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        ("drop a sample", "no row for 1 kept sample"),
        ("nan", "PC3 is nan"),
        ("repeat", "duplicate IID"),
        ("rename", "named PC1"),
        ("word", "not numbers"),
        ("short", "has 19 values"),
    ],
)
def test_a_pc_file_is_refused(tmp_path: Path, edit: str, says: str) -> None:
    pcs = pc_file(tmp_path, edit)
    with pytest.raises(opengwasdb.ld.InputError, match=says):
        opengwasdb.ld.build(fx.inputs(testdata.panel().pgen, 10, pcs=pcs), tmp_path / "store")


def test_a_pc_file_with_fid_is_read_by_iid(tmp_path: Path) -> None:
    pcs = pc_file(tmp_path, "fid")
    pgen = fx.write_fileset(tmp_path / "chr", np.arange(len(testdata.variants())))
    provenance = opengwasdb.ld.build(fx.inputs(pgen, 10, pcs=pcs), tmp_path / "store")
    assert provenance["adjustment"]["k"] == 10


def test_a_k0_build_reads_no_pc_file(tmp_path: Path) -> None:
    # At k = 0 no PC File is read: one that would be refused is not even opened.
    broken = pc_file(tmp_path, "nan")
    provenance = opengwasdb.ld.build(
        fx.inputs(fx.write_fileset(tmp_path / "c", chr22()), pcs=broken), tmp_path / "s"
    )
    assert provenance["adjustment"]["pc_file"] is None


# ------------------------------------------------------------------------------ the output


def test_a_store_is_never_written_over_another(tmp_path: Path) -> None:
    (tmp_path / "store").mkdir()
    with pytest.raises(opengwasdb.ld.InputError, match="exists"):
        opengwasdb.ld.build(fx.inputs(testdata.panel().pgen), tmp_path / "store")


def test_threads_count_workers(tmp_path: Path) -> None:
    with pytest.raises(opengwasdb.ld.InputError, match="at least 1"):
        opengwasdb.ld.build(fx.inputs(testdata.panel().pgen), tmp_path / "store", threads=0)


def test_a_fileset_needs_its_three_files(tmp_path: Path) -> None:
    pgen = fx.write_fileset(tmp_path / "lonely", chr22())
    pgen.with_suffix(".psam").unlink()
    with pytest.raises(opengwasdb.ld.InputError, match=r"needs its \.pgen, \.pvar"):
        opengwasdb.ld.build(fx.inputs(pgen), tmp_path / "store")


def test_two_filesets_of_one_name_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    v = testdata.variants()
    one = fx.write_fileset(tmp_path / "a" / "same", np.flatnonzero(v.chrom == "21"))
    two = fx.write_fileset(tmp_path / "b" / "same", chr22())
    with pytest.raises(opengwasdb.ld.InputError, match="share a file name"):
        opengwasdb.ld.build(fx.inputs(one, pgen=(one, two)), tmp_path / "store")


def test_a_pvar_zst_is_read(tmp_path: Path) -> None:
    from numcodecs import Zstd

    rows = chr22()
    plain = fx.write_fileset(tmp_path / "plain", rows)
    packed = fx.write_fileset(tmp_path / "packed", rows)
    pvar = packed.with_suffix(".pvar")
    pvar.with_name("packed.pvar.zst").write_bytes(bytes(Zstd().encode(pvar.read_bytes())))
    pvar.unlink()
    one = opengwasdb.ld.build(fx.inputs(plain), tmp_path / "one")
    other = opengwasdb.ld.build(fx.inputs(packed), tmp_path / "other")
    assert one["counts"] == other["counts"]
    assert "packed.pvar.zst" in other["source"]["sha256"]


def test_ogld_build_prints_the_provenance_and_exits_2_on_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from opengwasdb.ld import cli

    pgen = fx.write_fileset(tmp_path / "chr22", chr22())
    argv = ["build", str(tmp_path / "store"), "--pgen", str(pgen)]
    argv += ["--keep", str(testdata.panel().keep), "--panel-id", fx.PANEL_ID, "--ancestry", "EUR"]
    argv += ["--genotype-source", "Test Panel", "--k", "0", "--info-min", "none", "--threads", "2"]
    argv += ["--r-scale", str(fx.R_SCALE), "--t-partner", str(fx.T_PARTNER)]
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["panel"]["id"] == fx.PANEL_ID
    assert printed["build"]["threads"] == 2
    assert cli.main(argv) == 2, "the store exists now, so a second build is refused"
    assert "exists" in capsys.readouterr().err


# ------------------------------------------------------------- malformed and misaligned inputs


def malformed(tmp_path: Path, what: str) -> tuple[opengwasdb.ld.BuildInputs, str]:
    """A chr22 build whose `what` is broken, and the file and line its refusal must name."""
    rows = chr22()
    pgen = fx.write_fileset(tmp_path / "bad", rows)
    k, pcs = 0, None
    if what == "a short pvar line":
        pvar = pgen.with_suffix(".pvar")
        lines = pvar.read_text().splitlines()
        lines[3] = lines[3].rsplit("\t", 2)[0]
        pvar.write_text("\n".join(lines) + "\n")
        return fx.inputs(pgen), r"bad\.pvar line 4"
    if what == "a long pvar line":
        # The review's probe: a record with its REF written twice has one field too many, and
        # was cut to fit and filtered as FILTER, its REF read as the ALT.
        pvar = pgen.with_suffix(".pvar")
        lines = pvar.read_text().splitlines()
        fields = lines[7].split("\t")
        lines[7] = "\t".join([*fields[:4], fields[3], *fields[4:]])
        pvar.write_text("\n".join(lines) + "\n")
        return fx.inputs(pgen), r"bad\.pvar line 8 has 7 fields where its header has 6"
    if what == "a POS that is not an integer":
        pvar = pgen.with_suffix(".pvar")
        lines = pvar.read_text().splitlines()
        fields = lines[5].split("\t")
        lines[5] = "\t".join([fields[0], "12a34", *fields[2:]])
        pvar.write_text("\n".join(lines) + "\n")
        return fx.inputs(pgen), r"bad\.pvar line 6"
    if what == "an empty PC File":
        pcs = tmp_path / "empty.eigenvec"
        pcs.write_text("")
        return fx.inputs(pgen, 10, pcs=pcs), r"empty\.eigenvec is empty"
    if what == "an empty psam":
        pgen.with_suffix(".psam").write_text("")
        return fx.inputs(pgen), r"bad\.psam is empty"
    data = pgen.read_bytes()
    pgen.write_bytes(data[: len(data) // 2])
    return fx.inputs(pgen, k, pcs=pcs), r"bad\.pgen"


@pytest.mark.parametrize(
    "what",
    [
        "a short pvar line",
        "a long pvar line",
        "a POS that is not an integer",
        "an empty PC File",
        "an empty psam",
        "a truncated pgen",
    ],
)
def test_a_malformed_input_is_refused_naming_the_file(tmp_path: Path, what: str) -> None:
    # Review of #5 (standards a2): each was a traceback, an IndexError or a RuntimeError.
    given, names = malformed(tmp_path, what)
    with pytest.raises(opengwasdb.ld.InputError, match=names):
        opengwasdb.ld.build(given, tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_a_psam_missing_one_sample_is_refused(tmp_path: Path) -> None:
    # Review of #5 (standards a1): with row 100 of 771 gone from the psam, every later IID names
    # the genotypes of the sample after it. The keep file lists the 770 the psam still has, so
    # only the psam's count can tell: it must equal the pgen's.
    pgen = fx.write_fileset(tmp_path / "short", chr22())
    psam = pgen.with_suffix(".psam")
    lines = psam.read_text().splitlines()
    gone = lines.pop(101).split("\t")[0]
    psam.write_text("\n".join(lines) + "\n")
    keep = keep_file(tmp_path, ["#IID", *(iid for iid in iids() if iid != gone)])
    with pytest.raises(
        opengwasdb.ld.InputError, match=r"short\.psam lists 770 samples.*short\.pgen.*771"
    ):
        opengwasdb.ld.build(fx.inputs(pgen, keep=keep), tmp_path / "store")


def test_a_build_without_threadpoolctl_refuses_naming_the_extra(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['threadpoolctl'] = None\n"
        "sys.path.insert(0, 'tests/ld')\n"
        "import build_fixtures, testdata, opengwasdb.ld\n"
        "try:\n"
        "    opengwasdb.ld.build(build_fixtures.inputs(testdata.panel().pgen), "
        f"{str(tmp_path / 's')!r})\n"
        "except opengwasdb.ld.InputError as refusal:\n"
        "    print(refusal)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "opengwasdb[ld-build]" in done.stdout
    assert not (tmp_path / "s").exists()
