"""The digest of the LD panel a completion read (#327, spec §12.1).

It is a full-content hash of the files the loader reads, framed by path and size;
these tests state what that is by hand and then change one thing at a time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from opengwasdb.completion.ld_panel import (
    PanelIdentity,
    PanelSource,
    panel_digest,
    panel_input_files,
)

#: The digest of `_tiny_panel`, from `sha256sum` over the text below it: one record per
#: input, a line feed after each field, each file's own `sha256sum` last. Written here
#: by hand so that the code under test is not what says what it should compute.
TINY_PANEL_DIGEST = "b82dbb153dea82f96791c64f0dce01a817c4cc3a2629004720c1217913f446d8"
TINY_PANEL_FRAMED = (
    "1/b1.tsv\n8\n17c4a8f9e9602f7acd0578b6acb25e926053d99f7a482f545038eefa40251b3d\n"
    "1/b1.ldeig.npz\n7\n3d03ffc0b2745d990419242408278f6bab270e8acef898da32b51ecd593b8660\n"
    "1/b3.tsv\n48\n3876a1ffe666f89b6cff12d421fdd91b824177434b5d14206390e3399fe20259\n"
    "2/b2.tsv\n8\n94aea599a616c0edb5428ee08bcb529b866ba78fd5e10d96baf0f99c12ac043c\n"
    "2/b2.unphased.vcor1.gz\n6\n"
    "c94dc30ccba10633b189a935a37fc3fde3bdd300787372810ad4ce458cd014f9\n"
    "indel_reference.json\n2\n44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a\n"
)


def _tiny_panel(root: Path) -> Path:
    """Two blocks, a table with no matrix, an indel-reference provenance and two files
    completion never reads."""
    files = {
        "1/b1.tsv": b"tsv-one\n", "1/b1.ldeig.npz": b"npz-one",
        "2/b2.tsv": b"tsv-two\n", "2/b2.unphased.vcor1.gz": b"gz-two",
        "indel_reference.json": b"{}",
        "1/b3.tsv": b"a table with no matrix: the EAF reader reads it\n",
        "notes.rds": b"not an input", "1/run.log": b"not an input either",
    }
    for name, content in files.items():
        path = root / "EUR" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_the_digest_is_the_sha256_of_the_framed_inputs(tmp_path: Path):
    import hashlib

    # The pin is the framed text's own sha256, so a mistake in either shows here first.
    assert hashlib.sha256(TINY_PANEL_FRAMED.encode()).hexdigest() == TINY_PANEL_DIGEST
    root = _tiny_panel(tmp_path)
    assert [p.relative_to(root / "EUR").as_posix() for p in panel_input_files(root, "EUR")] == [
        "1/b1.tsv", "1/b1.ldeig.npz", "1/b3.tsv", "2/b2.tsv", "2/b2.unphased.vcor1.gz",
        "indel_reference.json",
    ]
    assert panel_digest(root, "EUR") == TINY_PANEL_DIGEST


def test_a_file_completion_never_reads_is_not_in_the_digest(tmp_path: Path):
    root = _tiny_panel(tmp_path)
    (root / "EUR" / "notes.rds").write_bytes(b"changed")
    (root / "EUR" / "1" / "run.log").write_bytes(b"changed")
    assert panel_digest(root, "EUR") == TINY_PANEL_DIGEST


def test_a_table_with_no_matrix_is_in_the_digest(tmp_path: Path):
    """The EAF reader opens every ``*.tsv`` in a chromosome directory, so its frequencies
    reach the Store whether or not the loader loads the block."""
    root = _tiny_panel(tmp_path)
    before = panel_digest(root, "EUR")
    (root / "EUR" / "1" / "b3.tsv").write_bytes(b"changed")
    assert panel_digest(root, "EUR") != before


def test_a_file_that_cannot_be_hashed_raises_not_hashed_as_none(tmp_path: Path, monkeypatch):
    from opengwasdb.completion import ld_panel

    root = _tiny_panel(tmp_path)

    def unreadable(path: Path) -> str:
        raise RuntimeError(f"cannot hash {path.name}")

    monkeypatch.setattr(ld_panel, "_sha256", unreadable)
    with pytest.raises(RuntimeError, match="cannot hash"):
        panel_digest(root, "EUR")


def test_one_byte_in_the_middle_of_a_large_member_changes_the_digest(tmp_path: Path):
    """The head and the tail of an archive are not the archive: a same-size change between
    them is the one a sampled digest would not see."""
    root = _tiny_panel(tmp_path)
    big = root / "EUR" / "1" / "b1.ldeig.npz"
    content = bytearray(os.urandom(3 * 1024 * 1024))
    big.write_bytes(bytes(content))
    before = panel_digest(root, "EUR")
    middle = len(content) // 2
    content[middle] ^= 0xFF
    big.write_bytes(bytes(content))
    assert panel_digest(root, "EUR") != before


def test_a_linked_chromosome_directory_is_read_as_the_loader_reads_it(tmp_path: Path):
    """`list_chromosomes` follows a link, so two panels that differ only in what the link
    points at are two panels; they must not both hash nothing."""
    outside = tmp_path / "outside"
    (outside / "7").mkdir(parents=True)
    (outside / "7" / "b7.tsv").write_bytes(b"tsv-seven\n")
    (outside / "7" / "b7.ldeig.npz").write_bytes(b"npz-seven")
    root = _tiny_panel(tmp_path / "panel")
    (root / "EUR" / "7").symlink_to(outside / "7", target_is_directory=True)
    with_link = panel_digest(root, "EUR")
    assert with_link != TINY_PANEL_DIGEST
    assert panel_input_files(root, "EUR")[-3:-1] == [
        root / "EUR" / "7" / "b7.tsv", root / "EUR" / "7" / "b7.ldeig.npz",
    ]
    (outside / "7" / "b7.ldeig.npz").write_bytes(b"npz-SEVEN")
    assert panel_digest(root, "EUR") != with_link


def test_the_path_is_kept_as_given_and_a_blank_name_is_refused(tmp_path: Path, monkeypatch):
    _tiny_panel(tmp_path / "panel")
    monkeypatch.chdir(tmp_path)
    identity = PanelIdentity.read(PanelSource("panel"))
    assert (identity.path, identity.ancestry_dir) == ("panel", str(Path("panel") / "EUR"))
    assert identity.panel_id == f"sha256:{TINY_PANEL_DIGEST[:16]}"
    assert PanelIdentity.read(PanelSource("panel", name="mine")).panel_id == "mine"
    with pytest.raises(ValueError, match="empty"):
        PanelSource("panel", name="  ")
