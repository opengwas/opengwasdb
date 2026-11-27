"""What every layout's completion must do about the LD panel it reads (#327), stated once.

Each layout's test module supplies three callables over its own fixtures and calls the
checks below: ``complete(dst, panel, name=None)`` runs a completion and returns its Store;
``interrupt(dst, panel, after_a_block=False, name=None)`` runs one that dies in its first
block and returns its checkpoint directory; ``resume(checkpoint_dir)`` resumes it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from ld_blocks import add_a_table_with_no_matrix, change_a_block, regenerated_panel

from opengwasdb.completion.checkpoint import checkpoint_dir_for
from opengwasdb.completion.ld_panel import panel_digest
from opengwasdb.store.open import open_store


def record_of(store: Path) -> dict:
    return open_store(store).manifest.provenance["completion"]


def interrupting(complete: Callable[..., Path], layout, monkeypatch) -> Callable[..., Path]:
    """An ``interrupt`` for a layout's `complete` module: its first block raises, or, with
    ``after_a_block``, raises once that block has run and left its checkpoint."""
    run_block = layout._run_block

    def interrupt(dst: Path, panel, after_a_block: bool = False, name=None) -> Path:
        def crash(task):
            if after_a_block:
                run_block(task)
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(layout, "_run_block", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            complete(dst, panel, name)
        monkeypatch.undo()
        return checkpoint_dir_for(dst)

    return interrupt


def check_two_panels_given_no_name_record_two_identities(
    complete: Callable[..., Path], tmp_path: Path, ld_panel: Path
) -> None:
    other = regenerated_panel(ld_panel, tmp_path / "other_panel")
    records = [
        record_of(complete(tmp_path / name, panel))
        for name, panel in (("a.opengwasdb", ld_panel), ("b.opengwasdb", other))
    ]
    ids = [record["ld_panel_id"] for record in records]
    assert ids[0] != ids[1]
    assert "eur-hg38-gpm" not in ids
    assert all(record["ld_panel_id_source"] == "digest" for record in records)
    assert records[0]["ld_panel_path"] == str(ld_panel)
    assert records[0]["ld_panel_ancestry_dir"] == str(ld_panel / "EUR")


def check_a_name_is_recorded_beside_the_digest(
    complete: Callable[..., Path], tmp_path: Path, ld_panel: Path
) -> None:
    # Even the retired constant: given, it is the caller's, and says so.
    record = record_of(complete(tmp_path / "a.opengwasdb", ld_panel, name="eur-hg38-gpm"))
    assert (record["ld_panel_id"], record["ld_panel_id_source"]) == ("eur-hg38-gpm", "caller")
    assert record["ld_panel_digest"] == panel_digest(ld_panel, "EUR")
    with pytest.raises(ValueError, match="empty"):
        complete(tmp_path / "b.opengwasdb", ld_panel, name="")


def check_a_resume_records_the_panel_the_checkpoint_holds(
    interrupt: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path,
    ld_panel: Path,
) -> None:
    dst = tmp_path / "a.opengwasdb"
    checkpoint_dir = interrupt(dst, ld_panel, name="mine")
    held = json.loads((checkpoint_dir / "build_params.json").read_text())
    assert held["ld_panel_digest"] == panel_digest(ld_panel, "EUR")
    resume(checkpoint_dir)
    record = record_of(dst)
    assert (record["ld_panel_id"], record["ld_panel_digest"]) == ("mine", held["ld_panel_digest"])


def check_a_resume_against_a_changed_panel_refuses(
    interrupt: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path,
    ld_panel: Path,
) -> None:
    checkpoint_dir = interrupt(tmp_path / "a.opengwasdb", ld_panel)
    change_a_block(ld_panel)
    with pytest.raises(ValueError, match="digest"):
        resume(checkpoint_dir)
    assert checkpoint_dir.exists()


def check_a_resume_against_an_edited_table_with_no_matrix_refuses(
    interrupt: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path,
    ld_panel: Path,
) -> None:
    """The EAF reader opens every table in a chromosome directory, matrix or not, and what
    it reads becomes the Store's `eaf_reference`: editing one is a different panel."""
    orphan = add_a_table_with_no_matrix(ld_panel)
    checkpoint_dir = interrupt(tmp_path / "a.opengwasdb", ld_panel)
    orphan.write_text(orphan.read_text().replace("\t0.", "\t0.1", 1))
    with pytest.raises(ValueError, match="digest"):
        resume(checkpoint_dir)


def check_a_resume_of_a_checkpoint_with_no_panel_digest_refuses(
    interrupt: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path,
    ld_panel: Path,
) -> None:
    checkpoint_dir = interrupt(tmp_path / "a.opengwasdb", ld_panel, after_a_block=True)
    assert list((checkpoint_dir / "blocks").glob("*.npz")), "the fixture completed a block"
    path = checkpoint_dir / "build_params.json"
    params = json.loads(path.read_text())
    del params["ld_panel_digest"]
    path.write_text(json.dumps(params))
    with pytest.raises(ValueError, match="records no digest"):
        resume(checkpoint_dir)


def check_a_run_interrupted_while_reading_the_panel_resumes(
    complete: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path,
    ld_panel: Path, monkeypatch,
) -> None:
    """The checkpoint is on disk before the panel is read (a FIFO in the panel holds the
    sibling workflow's fault test there until it has seen `build_params.json`), and one
    interrupted there has completed no block: its resume reads the panel and goes on."""
    from opengwasdb.completion import checkpoint as checkpoint_module

    dst = tmp_path / "a.opengwasdb"

    def read_stalled(*args, **kwargs):
        assert (checkpoint_dir_for(dst) / "build_params.json").is_file()
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_module.PanelIdentity, "read", read_stalled)
    with pytest.raises(KeyboardInterrupt):
        complete(dst, ld_panel, name="mine")
    monkeypatch.undo()
    held = json.loads((checkpoint_dir_for(dst) / "build_params.json").read_text())
    assert "ld_panel_digest" not in held
    assert not list((checkpoint_dir_for(dst) / "blocks").glob("*.npz"))
    resume(checkpoint_dir_for(dst))
    record = record_of(dst)
    assert (record["ld_panel_id"], record["ld_panel_digest"]) == (
        "mine", panel_digest(ld_panel, "EUR"),
    )


def check_the_path_as_given_survives_a_resume(
    interrupt: Callable[..., Path], resume: Callable[[Path], object], tmp_path: Path, monkeypatch,
) -> None:
    """Recorded as spelt (here relative, ``ld_panel`` under ``tmp_path``), and as spelt
    after a resume from another directory, which reads it where the first run resolved it."""
    monkeypatch.chdir(tmp_path)
    dst = tmp_path / "a.opengwasdb"
    checkpoint_dir = interrupt(dst, "ld_panel")
    monkeypatch.chdir(tmp_path.parent)
    resume(checkpoint_dir)
    assert record_of(dst)["ld_panel_path"] == "ld_panel"


class PanelIdentityTests:
    """The tests every layout's `TestLdPanelIdentity` runs. A subclass supplies a ``ways``
    fixture: ``(complete, interrupt, resume)``, as described in the module docstring."""

    def test_two_panels_given_no_name_record_two_identities(self, tmp_path, ld_panel, ways):
        check_two_panels_given_no_name_record_two_identities(ways[0], tmp_path, ld_panel)

    def test_a_name_the_caller_gives_is_recorded_beside_the_digest(
        self, tmp_path, ld_panel, ways
    ):
        check_a_name_is_recorded_beside_the_digest(ways[0], tmp_path, ld_panel)

    def test_a_resume_records_the_panel_the_checkpoint_holds(self, tmp_path, ld_panel, ways):
        check_a_resume_records_the_panel_the_checkpoint_holds(ways[1], ways[2], tmp_path, ld_panel)

    def test_a_resume_against_a_changed_panel_refuses(self, tmp_path, ld_panel, ways):
        check_a_resume_against_a_changed_panel_refuses(ways[1], ways[2], tmp_path, ld_panel)

    def test_a_resume_against_an_edited_table_with_no_matrix_refuses(
        self, tmp_path, ld_panel, ways
    ):
        check_a_resume_against_an_edited_table_with_no_matrix_refuses(
            ways[1], ways[2], tmp_path, ld_panel
        )

    def test_a_resume_of_a_checkpoint_with_no_panel_digest_refuses(
        self, tmp_path, ld_panel, ways
    ):
        check_a_resume_of_a_checkpoint_with_no_panel_digest_refuses(
            ways[1], ways[2], tmp_path, ld_panel
        )

    def test_the_path_as_given_survives_a_resume(self, tmp_path, ld_panel, ways, monkeypatch):
        check_the_path_as_given_survives_a_resume(ways[1], ways[2], tmp_path, monkeypatch)

    def test_a_run_interrupted_while_reading_the_panel_resumes(
        self, tmp_path, ld_panel, ways, monkeypatch
    ):
        check_a_run_interrupted_while_reading_the_panel_resumes(
            ways[0], ways[2], tmp_path, ld_panel, monkeypatch
        )
