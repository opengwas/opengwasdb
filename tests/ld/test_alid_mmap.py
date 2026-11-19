"""Local companion behavior, held to the table and independently decoded canonical bytes (#47)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_alid_index import written


def test_writer_and_derivation_publish_identical_canonical_bytes(tmp_path: Path) -> None:
    from script_modules import load_script

    store = written(tmp_path / "store")
    companion = store / "variants/alid_mmap"
    root = json.loads((store / "zarr.json").read_text())
    pointer = root["attributes"]["alid_mmap"]
    built_generation = companion / pointer["generation"]
    before = {p.name: p.read_bytes() for p in built_generation.iterdir()}
    assert set(before) == {"keys.bin", "rows.i32le", "manifest.json"}
    canonical = zarr.open_group(store, mode="r")
    assert before["keys.bin"] == np.asarray(canonical["variants/alid_index/alid"][:]).tobytes()
    assert (
        before["rows.i32le"]
        == np.asarray(canonical["variants/alid_index/row"][:], dtype="<i4").tobytes()
    )
    assert len(before["rows.i32le"]) == 12, "three fitting rows, two over-width rows"
    del root["attributes"]["alid_mmap"]
    (store / "zarr.json").write_text(json.dumps(root))
    # Retain a completed but unpublished generation, as after a crash before pointer publication.
    script = load_script("derive_alid_mmap")
    assert script.derive(store) == 3
    after_root = json.loads((store / "zarr.json").read_text())
    after = companion / after_root["attributes"]["alid_mmap"]["generation"]
    assert {p.name: p.read_bytes() for p in after.iterdir()} == before
    assert script.derive(store) == 3, "retrying publication is idempotent"
    import shutil

    copied = Path(shutil.copytree(store, tmp_path / "copied"))
    shutil.rmtree(copied / "variants/alid_mmap")
    copied_root = json.loads((copied / "zarr.json").read_bytes())
    del copied_root["attributes"]["alid_mmap"]
    (copied / "zarr.json").write_text(json.dumps(copied_root))
    assert script.derive(copied) == 3
    assert {p.name: p.read_bytes() for p in generation(copied).iterdir()} == before


def generation(store: Path) -> Path:
    root = json.loads((store / "zarr.json").read_bytes())
    return store / "variants/alid_mmap" / root["attributes"]["alid_mmap"]["generation"]


def republish(store: Path, manifest: dict) -> None:
    """An independently edited self-consistent representation, bypassing the real producer."""
    import hashlib

    before = generation(store)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (before / "manifest.json").write_bytes(raw)
    before.rename(before.parent / digest)
    path = store / "zarr.json"
    root = json.loads(path.read_bytes())
    root["attributes"]["alid_mmap"] = {"generation": digest}
    path.write_text(json.dumps(root))


def test_validation_rejects_self_consistent_wrong_rows(tmp_path: Path) -> None:
    import hashlib

    from ogld.store.validation import validate

    store = written(tmp_path / "store")
    directory = generation(store)
    rows = np.frombuffer((directory / "rows.i32le").read_bytes(), dtype="<i4").copy()
    assert len(set(rows)) == 3, "swap real, distinct rows without changing range or uniqueness"
    rows[[0, 1]] = rows[[1, 0]]
    (directory / "rows.i32le").write_bytes(rows.tobytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["payloads"]["rows.i32le"] = hashlib.sha256(rows.tobytes()).hexdigest()
    republish(store, manifest)
    findings = validate(store)
    assert findings["alid_index.count"], "self-consistent hashes do not prove canonical rows"


def test_mapped_lookup_uses_no_locus_or_column_reads(tmp_path: Path, monkeypatch) -> None:
    from ogld.store import StoreReader
    from ogld.store import reader as reader_module

    store = written(tmp_path / "store")
    root = zarr.open_group(store, mode="r")
    names = np.asarray(root["variants/alid"][:]).tolist()
    fitting = [a for a in names if len(a.encode()) <= 32]
    assert len(fitting) == 3
    assert len(fitting[1].encode()) < 32, "NUL alias breaker must fit after appending NUL"
    assert any(len(a.encode()) == 32 for a in fitting)
    reader = StoreReader.open(store)

    def forbidden(*args, **kwargs):
        raise AssertionError("mapped lookup parsed loci or read Zarr payloads")

    monkeypatch.setattr(reader_module, "_loci", forbidden)
    monkeypatch.setattr(reader, "_read", forbidden)
    monkeypatch.setattr(reader, "_runs", forbidden)
    monkeypatch.setattr(reader, "_strings", forbidden)
    asked = [
        *reversed(fitting),
        fitting[0],
        "",
        "nonsense",
        "22:1:A:G",
        fitting[1] + "\0",
        "\ud800",
    ]
    assert reader.rows_of(asked).tolist() == [names.index(a) if a in names else -1 for a in asked]


def test_all_test_store_names_and_width_boundaries_are_exact(tmp_path: Path) -> None:
    from store_copies import copy_store

    from ogld.store import StoreReader
    from ogld.store.assembly import derive_alid_mmap

    store = copy_store(tmp_path / "test-store")
    names = np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()
    assert len(names) > 1000
    assert any(len(a.encode()) == 32 for a in names)
    assert sum(len(a.encode()) > 32 for a in names) == 8
    expected = StoreReader.open(store).rows_of(names[::-1] + names[:3]).tolist()
    assert expected == [*range(len(names) - 1, -1, -1), 0, 1, 2]
    derive_alid_mmap(store)
    assert StoreReader.open(store).rows_of(names[::-1] + names[:3]).tolist() == expected


def test_null_publication_is_corrupt_rather_than_absent(tmp_path: Path) -> None:
    import pytest

    from ogld.format import InputError
    from ogld.store import StoreReader

    store = written(tmp_path / "store")
    path = store / "zarr.json"
    document = json.loads(path.read_bytes())
    document["attributes"]["alid_mmap"] = None
    path.write_text(json.dumps(document))
    with pytest.raises(InputError, match="companion"):
        StoreReader.open(store)


def test_store_without_either_index_keeps_exact_scan(tmp_path: Path) -> None:
    import shutil

    from store_copies import copy_store

    from ogld.store import StoreReader

    store = copy_store(tmp_path / "older")
    root = zarr.open_group(store, mode="r")
    names = np.asarray(root["variants/alid"][:]).tolist()
    shutil.rmtree(store / "variants/alid_index")
    asked = [*names[::37], "22:1:A:G", "nonsense"]
    assert StoreReader.open(store).rows_of(asked).tolist() == [
        names.index(a) if a in names else -1 for a in asked
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "keys-short",
        "rows-short",
        "keys-wrong",
        "rows-wrong",
        "manifest",
        "missing",
        "root",
        "index",
    ],
)
def test_corrupt_publication_refuses_and_fails_validation(tmp_path: Path, damage: str) -> None:
    from ogld.format import InputError
    from ogld.store import StoreReader
    from ogld.store.validation import validate

    store = written(tmp_path / "store")
    directory = generation(store)
    if damage in {"keys-short", "rows-short", "keys-wrong", "rows-wrong"}:
        payload = directory / ("keys.bin" if damage.startswith("keys") else "rows.i32le")
        before = payload.read_bytes()
        assert before
        payload.write_bytes(
            before[:-1] if damage.endswith("short") else bytes([before[0] ^ 1]) + before[1:]
        )
    elif damage == "manifest":
        (directory / "manifest.json").write_text("{")
    elif damage == "missing":
        (directory / "rows.i32le").unlink()
    elif damage == "root":
        path = store / "zarr.json"
        document = json.loads(path.read_bytes())
        document["attributes"]["ld_store"]["software"]["ogld"] = "stale-generation"
        path.write_text(json.dumps(document))
    else:
        path = store / "variants/alid_index/row/zarr.json"
        document = json.loads(path.read_bytes())
        document["attributes"]["new-generation"] = True
        path.write_text(json.dumps(document))
    with pytest.raises(InputError, match="companion"):
        StoreReader.open(store).rows_of(["22:200:A:C"])
    assert validate(store)["alid_index.count"], "corrupt publication cannot validate"


@pytest.mark.parametrize("payload", ["keys.bin", "rows.i32le"])
def test_self_consistent_payload_change_fails_canonical_validation(
    tmp_path: Path, payload: str
) -> None:
    import hashlib

    from ogld.store.validation import validate

    store = written(tmp_path / "store")
    directory = generation(store)
    path = directory / payload
    raw = path.read_bytes()
    assert raw
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["payloads"][payload] = hashlib.sha256(path.read_bytes()).hexdigest()
    republish(store, manifest)
    assert validate(store)["alid_index.count"]


@pytest.mark.parametrize(
    "stage", ["keys.bin", "manifest.json", ".alid-mmap-root.partial", "replace"]
)
def test_failed_publication_is_unpublished_and_restartable(
    tmp_path: Path, monkeypatch, stage: str
) -> None:
    import shutil

    from ogld.store import StoreReader, assembly
    from ogld.store.assembly import derive_alid_mmap

    store = written(tmp_path / "store")
    expected = {p.name: p.read_bytes() for p in generation(store).iterdir()}
    root = zarr.open_group(store, mode="r+")
    del root.attrs["alid_mmap"]
    shutil.rmtree(store / "variants/alid_mmap")
    real_write = assembly._durable_bytes
    real_payloads = assembly._write_alid_payloads
    real_replace = assembly.os.replace

    def fail_write(path, raw):
        if path.name == stage:
            raise OSError("publication interrupted")
        return real_write(path, raw)

    def fail_payloads(root, partial):
        if stage == "keys.bin":
            (partial / "keys.bin").write_bytes(b"partial")
            raise OSError("publication interrupted")
        return real_payloads(root, partial)

    def fail_replace(source, destination):
        if stage == "replace":
            assert source.name == ".alid-mmap-root.partial"
            assert "alid_mmap" not in json.loads((store / "zarr.json").read_bytes())["attributes"]
            raise OSError("publication interrupted")
        return real_replace(source, destination)

    monkeypatch.setattr(assembly, "_durable_bytes", fail_write)
    monkeypatch.setattr(assembly, "_write_alid_payloads", fail_payloads)
    monkeypatch.setattr(assembly.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        derive_alid_mmap(store)
    assert "alid_mmap" not in json.loads((store / "zarr.json").read_bytes())["attributes"]
    assert StoreReader.open(store).rows_of(["22:200:A:C"]).tolist() == [2]
    monkeypatch.undo()
    assert derive_alid_mmap(store) == 3
    assert {p.name: p.read_bytes() for p in generation(store).iterdir()} == expected
    assert StoreReader.open(store).rows_of(["22:200:A:C"]).tolist() == [2]


@pytest.mark.parametrize("schema", [True, 1.0])
def test_schema_must_be_an_integer_version(tmp_path: Path, schema: object) -> None:
    from ogld.format import InputError
    from ogld.store import StoreReader

    store = written(tmp_path / "store")
    manifest = json.loads((generation(store) / "manifest.json").read_bytes())
    manifest["schema"] = schema
    republish(store, manifest)
    with pytest.raises(InputError, match="companion"):
        StoreReader.open(store)


def test_http_ignores_local_companion_even_when_corrupt(tmp_path: Path) -> None:
    from test_store_reads import serving

    from ogld.store import StoreReader

    store = written(tmp_path / "store")
    (generation(store) / "keys.bin").unlink()
    with serving(store) as (url, sent):
        assert StoreReader.open(url).rows_of(["22:200:A:C", "nonsense"]).tolist() == [2, -1]
    assert sent
    assert not any("alid_mmap" in path for path, _ in sent)


def test_exact_wide_lookup_does_not_read_unrelated_near_band_data(tmp_path: Path) -> None:
    from ogld.store import StoreReader

    store = written(tmp_path / "store")
    names = np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()
    assert len(names[1].encode()) == 33, "the request must take the exact wide-key fallback"
    chunks = list((store / "near/22/indptr").rglob("c/*"))
    assert chunks, "remove actual Near Band data rather than an unused path"
    for path in chunks:
        path.unlink()
    reader = StoreReader.open(store)
    assert reader.rows_of([names[1], names[3], names[1]]).tolist() == [1, 3, 1]
    with pytest.raises(ValueError, match="near/22/indptr"):
        reader.r_of_pairs(np.array([1]), np.array([2]))
