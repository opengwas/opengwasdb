"""Local companion behavior, held to the table and independently decoded canonical bytes (#47)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_alid_index import written


def test_a_companion_copied_from_identical_metadata_cannot_change_rows(tmp_path: Path) -> None:
    import shutil
    from unittest.mock import patch

    from test_alid_index import WIDTH_ROWS

    from ogld.format import InputError
    from ogld.store import StoreReader, layout

    a = written(tmp_path / "a")
    changed = list(WIDTH_ROWS)
    changed[2] = ("22", 200, "A", "T")
    b = written(tmp_path / "b", changed)
    for array in ("alid", "row"):
        relative = f"variants/alid_index/{array}/zarr.json"
        assert (a / relative).read_bytes() == (b / relative).read_bytes()
    root_a = json.loads((a / "zarr.json").read_bytes())
    root_b = json.loads((b / "zarr.json").read_bytes())
    pointer = root_a["attributes"].pop("alid_mmap")
    del root_b["attributes"]["alid_mmap"]
    assert root_a == root_b, "only canonical content and the companion differ"
    with patch.object(layout, "alid_companion", lambda root: None):
        canonical = StoreReader.open(b)
        assert canonical.rows_of(["22:200:A:C", "22:200:A:T"]).tolist() == [-1, 2]
    shutil.rmtree(b / "variants/alid_mmap")
    shutil.copytree(a / "variants/alid_mmap", b / "variants/alid_mmap")
    root_b["attributes"]["alid_mmap"] = pointer
    (b / "zarr.json").write_text(json.dumps(root_b))
    with pytest.raises(InputError) as stopped:
        StoreReader.open(b).rows_of(["22:200:A:C", "22:200:A:T"])
    message = str(stopped.value)
    assert str(b) in message
    assert pointer["generation"] in message
    assert "variants/alid_index/alid/c/" in message


def test_manifest_pins_every_canonical_file(tmp_path: Path) -> None:
    import hashlib

    store = written(tmp_path / "store")
    manifest = json.loads((generation(store) / "manifest.json").read_bytes())
    assert manifest["schema"] == 2
    files = [
        p
        for array in ("alid", "row")
        for p in (store / "variants/alid_index" / array).rglob("*")
        if p.is_file()
    ]
    assert any("/c/" in str(p) for p in files), "pin real payloads, not only metadata"
    expected = {
        str(p.relative_to(store)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    assert {k: v for k, v in manifest["sources"].items() if k != "zarr.json"} == expected


@pytest.mark.parametrize(
    "damage", ["nonobject", "negative-count", "float-count", "bool-count", "sources"]
)
def test_malformed_manifest_fields_refuse_with_a_concrete_diagnostic(
    tmp_path: Path, damage: str
) -> None:
    from ogld.format import InputError
    from ogld.store import StoreReader
    from ogld.store.validation import validate

    store = written(tmp_path / "store")
    manifest = json.loads((generation(store) / "manifest.json").read_bytes())
    if damage == "nonobject":
        manifest = []
        reason = "manifest must be an object"
    elif damage == "sources":
        manifest["sources"] = []
        reason = "sources must pin canonical files"
    else:
        manifest["count"] = {"negative-count": -1, "float-count": 3.0, "bool-count": True}[damage]
        reason = "count must be a nonnegative integer"
    republish(store, manifest)
    with pytest.raises(InputError, match=reason) as stopped:
        StoreReader.open(store).rows_of(["22:200:A:C"])
    assert_companion_diagnostic(store, str(stopped.value), "manifest.json")
    findings = validate(store)["alid_index.count"]
    assert findings
    assert_companion_diagnostic(store, findings[0].message, "manifest.json")


def test_schema_one_refuses_with_a_rederivation_instruction(tmp_path: Path) -> None:
    from ogld.format import InputError
    from ogld.store import StoreReader

    store = written(tmp_path / "store")
    manifest = json.loads((generation(store) / "manifest.json").read_bytes())
    manifest["schema"] = 1
    republish(store, manifest)
    with pytest.raises(InputError) as stopped:
        StoreReader.open(store).rows_of(["22:200:A:C"])
    message = str(stopped.value)
    assert str(store) in message
    assert generation(store).name in message
    assert "manifest.json" in message
    assert "schema 1" in message
    assert "scripts/derive_alid_mmap.py" in message


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


def assert_companion_diagnostic(store: Path, message: str, concrete: str) -> None:
    assert str(store) in message
    assert generation(store).name in message
    assert concrete in message


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
    assert_companion_diagnostic(store, findings["alid_index.count"][0].message, "rows.i32le")


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
    with pytest.raises(InputError, match="companion") as stopped:
        StoreReader.open(store).rows_of(["22:200:A:C"])
    assert str(store) in str(stopped.value)
    assert "generation" in str(stopped.value)
    assert "zarr.json" in str(stopped.value)


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
    concrete = {
        "manifest": "manifest.json",
        "missing": "rows.i32le",
        "root": str(store / "zarr.json"),
        "index": "variants/alid_index/row/zarr.json",
    }.get(damage, "keys.bin" if damage.startswith("keys") else "rows.i32le")
    with pytest.raises(InputError, match="companion") as stopped:
        StoreReader.open(store).rows_of(["22:200:A:C"])
    assert_companion_diagnostic(store, str(stopped.value), concrete)
    findings = validate(store)["alid_index.count"]
    assert findings, "corrupt publication cannot validate"
    assert_companion_diagnostic(store, findings[0].message, concrete)


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
    findings = validate(store)["alid_index.count"]
    assert findings
    assert_companion_diagnostic(store, findings[0].message, payload)


def test_a_source_change_during_derivation_names_the_store_generation_and_file(
    tmp_path: Path, monkeypatch
) -> None:
    import shutil

    from ogld.format import InputError
    from ogld.store import assembly

    store = written(tmp_path / "store")
    root = zarr.open_group(store, mode="r+")
    del root.attrs["alid_mmap"]
    shutil.rmtree(store / "variants/alid_mmap")
    write = assembly._write_alid_payloads

    def changing(root, partial):
        count = write(root, partial)
        array = zarr.open_array(store / "variants/alid_index/row", mode="r+")
        array.attrs["changed"] = True
        return count

    monkeypatch.setattr(assembly, "_write_alid_payloads", changing)
    with pytest.raises(InputError) as stopped:
        assembly.derive_alid_mmap(store)
    message = str(stopped.value)
    assert str(store) in message
    generations = [p for p in (store / "variants/alid_mmap").iterdir() if len(p.name) == 64]
    assert len(generations) == 1, "a complete generation exists but is not published"
    assert generations[0].name in message
    assert "variants/alid_index/row/zarr.json" in message
    assert "changed during companion derivation" in message
    assert "alid_mmap" not in zarr.open_group(store, mode="r").attrs


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
    with pytest.raises(InputError, match="companion") as stopped:
        StoreReader.open(store).rows_of(["22:200:A:C"])
    assert_companion_diagnostic(store, str(stopped.value), "manifest.json")


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
