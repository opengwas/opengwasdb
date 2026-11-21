"""Chromosome restart and deletion at the public import seam (#93)."""

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

import ogld
from ogld.build.importing.run import ImportInputs, import_panukb

FIXTURE = Path(__file__).parent / "data" / "panukb"


@pytest.fixture(scope="module")
def complete_import(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete-import")
    inputs = ImportInputs(
        metadata=FIXTURE / "metadata.json",
        variants=FIXTURE / "variants.tsv",
        chain=FIXTURE / "identity.chain",
        fasta=FIXTURE / "reference.fa.gz",
        ledger=FIXTURE / "ledger.json",
        parts=FIXTURE,
        work=root / "work",
    )
    import_panukb(inputs, root / "complete.store")
    return inputs, root / "complete.store"


def test_restart_does_not_fetch_closed_deleted_chromosome(complete_import, tmp_path) -> None:
    original, complete = complete_import
    inputs = replace(original, parts=None, work=tmp_path / "work", delete_parts=True)
    fetches = []

    def fetch(url):
        name = url.rsplit("/", 1)[-1]
        fetches.append(name)
        return io.BytesIO((FIXTURE / name).read_bytes())

    def interrupt(name):
        assert name == "1"
        raise InterruptedError("after first close")

    out = tmp_path / "resumed.store"
    with pytest.raises(InterruptedError, match="after first close"):
        import_panukb(inputs, out, fetch=fetch, on_chromosome=interrupt)
    assert not out.exists()
    assert fetches == ["part-0"]
    assert not (inputs.work / "parts" / "part-0").exists()
    import_panukb(inputs, out, fetch=fetch)
    assert fetches == ["part-0", "part-1"]
    assert not (inputs.work / "parts" / "part-1").exists()
    alids = [
        f"{chrom}:{position}:{alleles}"
        for chrom in (1, 2)
        for position, alleles in ((1, "A:C"), (2, "A:G"), (4000001, "A:G"))
    ]
    with ogld.open(out) as resumed, ogld.open(complete) as uninterrupted:
        np.testing.assert_array_equal(resumed.matrix(alids).r, uninterrupted.matrix(alids).r)
        assert resumed.provenance["counts"] == uninterrupted.provenance["counts"]
        assert (
            resumed.provenance["source"]["sha256"] == uninterrupted.provenance["source"]["sha256"]
        )


def test_restart_refuses_a_closed_chromosome_claiming_another_panel(
    complete_import, tmp_path
) -> None:
    original, _ = complete_import
    inputs = replace(original, work=tmp_path / "work")

    def interrupt(_name):
        raise InterruptedError("after close")

    out = tmp_path / "resumed.store"
    with pytest.raises(InterruptedError, match="after close"):
        import_panukb(inputs, out, on_chromosome=interrupt)
    root = zarr.open_group(str(inputs.work / "1.store"), mode="r+")
    record = dict(root.attrs["ld_store"])
    record["panel"]["id"] = "another-panel"
    root.attrs["ld_store"] = record
    with pytest.raises(ValueError, match="panel"):
        import_panukb(inputs, out)
    assert not out.exists()


def test_restart_rebuilds_chromosome_interrupted_before_root_metadata(
    complete_import, tmp_path
) -> None:
    original, _ = complete_import
    work = tmp_path / "work"
    unfinished = work / "1.store"
    unfinished.mkdir(parents=True)
    assert not (unfinished / "zarr.json").exists()
    out = tmp_path / "resumed.store"
    import_panukb(replace(original, work=work), out)
    with ogld.open(out) as store:
        assert store.provenance["n_variants"] == 6


def test_restart_refuses_changed_ledger_after_part_deletion(complete_import, tmp_path) -> None:
    original, _ = complete_import
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(original.ledger.read_bytes())
    inputs = replace(original, ledger=ledger, parts=None, work=tmp_path / "work", delete_parts=True)

    def fetch(url):
        return io.BytesIO((FIXTURE / url.rsplit("/", 1)[-1]).read_bytes())

    def interrupt(_name):
        raise InterruptedError("after close")

    out = tmp_path / "resumed.store"
    with pytest.raises(InterruptedError, match="after close"):
        import_panukb(inputs, out, fetch=fetch, on_chromosome=interrupt)
    assert not (inputs.work / "parts" / "part-0").exists()
    changed = json.loads(ledger.read_text())
    changed[0]["sha256"] = "0" * 64
    ledger.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different inputs"):
        import_panukb(inputs, out, fetch=fetch)
    assert not out.exists()
