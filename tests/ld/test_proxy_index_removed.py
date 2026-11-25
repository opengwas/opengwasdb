"""The Proxy Index is gone, and a store that still carries one reads and validates alike
(opengwas/ogld#72).

Three seams, each observed to fail against the code before opengwas/ogld#72:

- **No `index/`.** The package source names no array of the component, and a store this code
  writes has no such key: the store's own keys are listed and none begins with `index`. The one
  place that still names it is the removal `opengwasdb ld derive-partners --drop-index` makes, and a
  migration that writes Far Pairs from a store too old to read (ADR 0093).
- **A leftover `index/`.** A 0.2.0 store derived before opengwas/ogld#72 without `--drop-index`
holds the
  component. It opens, answers exactly as the store without one does, and validation ignores it
  rather than reporting it (ld-store §12) — even when every entry of it is gone, which a read of it
  would have to refuse.
- **A 0.1.0 store** still opens and answers `matrix` and `region`, which read the Near Band and
  the Far Pairs; `clump` and `proxies` are refused, as they are now, naming the Partner Rows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import zarr
from store_copies import (
    FLOOR,
    R_SCALE,
    TEST_STORE,
    copy_store,
    snv_variants,
    writer_for_test_store,
)

import opengwasdb.ld
from opengwasdb.ld.format import VALIDATION_RULES, REncoding
from ogquery.ld.store import StoreReader
from opengwasdb.ld.store import layout
from opengwasdb.ld.store.validation import validate_group

#: The package whose source must not name the component.
PACKAGE = Path(opengwasdb.ld.__file__).resolve().parent
#: The names the component had: its writer, the reader's warmer, and its seven rule names.
FORBIDDEN_NAMES = frozenset({"ProxyIndexStream", "clump_warmer"})
FORBIDDEN_RULES = frozenset(
    {
        "index.rows",
        "index.codes",
        "index.distance",
        "index.threshold",
        "index.symmetric",
        "index.near",
        "far.index",
    }
)


def sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def names_in(source: Path) -> set[str]:
    """Every name `source` defines, assigns, calls or reaches for as an attribute."""
    tree = ast.parse(source.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            found.add(node.name)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
    return found


def test_the_package_source_defines_no_proxy_index_stream_or_clump_warmer() -> None:
    named = {
        str(path.relative_to(PACKAGE)): sorted(names_in(path) & FORBIDDEN_NAMES)
        for path in sources()
    }
    assert {where: found for where, found in named.items() if found} == {}
    assert not [rule for rule in VALIDATION_RULES if rule in FORBIDDEN_RULES]


def test_no_source_holds_a_proxy_index_rule_name_as_a_string() -> None:
    # A rule name is only ever a string to a store's provenance or to `validate`; none is left.
    held: dict[str, list[str]] = {}
    for path in sources():
        tree = ast.parse(path.read_text())
        found = sorted(
            {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in FORBIDDEN_RULES
            }
        )
        if found:
            held[str(path.relative_to(PACKAGE))] = found
    assert held == {}


def index_paths(source: Path) -> list[tuple[int, str]]:
    """The string constants of `source` that are the component's directory or a path under it."""
    tree = ast.parse(source.read_text())
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value == "index" or node.value.startswith("index/"))
    ]


def test_the_only_source_that_names_the_component_s_directory_is_its_removal() -> None:
    # A spelling guard, not a proof: the transport test below is what shows no read happens.
    # `opengwasdb/ld/store/assembly.py` is where `--drop-index` removes the directory; the migration
    # helper (`derive_far_pairs`) takes the group as an argument and names no path.
    found = {
        str(path.relative_to(PACKAGE)): index_paths(path) for path in sources() if index_paths(path)
    }
    assert list(found) == ["store/assembly.py"], found


def small_store(out: Path) -> Path:
    """A store this code writes, of four variants on two chromosomes, from a known r."""
    position = np.array([1_000_000, 4_500_000, 2_000_000, 6_000_000], dtype=np.int64)
    chrom = ["1", "1", "2", "2"]
    table = snv_variants(chrom, position)
    writer = writer_for_test_store(out, table)
    r = np.full((2, 2), np.nan)
    r[0, 1] = 0.5
    writer.write(writer.cut(0, r))
    farther = np.full((2, 2), np.nan)
    farther[0, 1] = 0.0
    writer.write(writer.cut(2, farther))
    sourced = {
        key: value
        for key, value in StoreReader.open(TEST_STORE).provenance.items()
        if key not in {"format_version", "encoding", "n_variants", "near_bp", "far_bp", "t_far",
                       "t_far_q", "far_code_floor", "t_partner", "partner_code_floor", "counts"}
    }  # fmt: skip
    writer.finish(sourced)
    return out


def keys_of(store: Path) -> list[str]:
    group = zarr.open_group(str(store), mode="r")
    return sorted(str(path) for path, _ in group.members(max_depth=None))


def test_a_store_this_code_writes_holds_no_index_key(tmp_path: Path) -> None:
    store = small_store(tmp_path / "small")
    keys = keys_of(store)
    assert {"near/1/r", "far/2/partner", "partners/1/frames"} <= set(keys), keys
    assert [key for key in keys if key.startswith("index")] == []
    assert "index_entries" not in zarr.open_group(str(store), mode="r").attrs["ld_store"]["counts"]


def write_a_leftover_index(store: Path) -> None:
    """Give `store` a Proxy Index a store derived before opengwas/ogld#72 still holds: one entry a
    chromosome, valid enough that reading it would answer, and never read by this package."""
    root = zarr.open_group(str(store), mode="r+")
    for chrom, (start, stop) in {
        entry["name"]: (entry["row_start"], entry["row_stop"])
        for entry in root["variants"].attrs["chromosomes"]
    }.items():
        rows = stop - start
        group = root.create_group(f"index/{chrom}")
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indptr[1:] = 1 if rows else 0
        partner = np.minimum(np.arange(rows, dtype=np.int64) + 1, max(0, rows - 1))
        layout.put_numbers(group, "indptr", indptr, layout.INT64_FILL)
        layout.put_numbers(group, "partner", partner.astype(np.int32), layout.PARTNER_FILL)
        codes = np.full(rows, FLOOR, dtype=np.int16)
        layout.new_r_array(group, "r", rows, layout.SIDE, REncoding(R_SCALE))[:] = codes


def answers(store: Path) -> dict[str, np.ndarray]:
    reader = StoreReader.open(store)
    names = list(reader.variants(np.arange(reader.provenance["n_variants"])).alid)
    found: dict[str, np.ndarray] = {"matrix": opengwasdb.ld.open(store).matrix(names).r}
    for chrom in ("21", "22"):
        found[f"region-{chrom}"] = opengwasdb.ld.open(store).region(chrom, 30_000_000, 5_000_000).r
    return found


def test_validation_ignores_a_leftover_index(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    root = zarr.open_group(str(copy), mode="r")
    found = validate_group(root, root.attrs["ld_store"])
    assert {rule: value for rule, value in found.items() if value} == {}, "every rule passes"
    assert set(found) == set(VALIDATION_RULES)


def a_0_1_0_store(to: Path) -> Path:
    """The Test Store as a 0.1.0 store records itself: no Partner Rows claimed, no `t_partner`,
    and the Proxy Index a 0.1.0 store holds with its `counts.index_entries`."""
    copy = copy_store(to)
    write_a_leftover_index(copy)
    root = zarr.open_group(str(copy), mode="r+")
    recorded = dict(root.attrs["ld_store"])
    recorded["format_version"] = "0.1.0"
    for key in ("t_partner", "partner_code_floor"):
        recorded.pop(key, None)
    recorded["counts"] = {
        key: value for key, value in recorded["counts"].items() if key != "partner_entries"
    }
    recorded["counts"]["index_entries"] = 2 * len(np.arange(1056))
    recorded["software"] = {
        key: value for key, value in recorded["software"].items() if key != "zstd"
    }
    root.attrs.update({"ld_store": recorded})
    return copy


def test_a_0_1_0_store_with_every_index_chunk_lost_still_validates_and_answers(
    tmp_path: Path,
) -> None:
    # No rule validates an index of either version since opengwas/ogld#72 (opengwas/ogld#71/#72 read
    # nothing of it), so a 0.1.0 store whose index is gone is still a valid 0.1.0 store, and still
    # answers.
    copy = a_0_1_0_store(tmp_path / "store")
    chunks = sorted(copy.glob("index/*/*/c/*"))
    assert len(chunks) >= 4, "both chromosomes' three arrays must have a chunk to lose"
    for chunk in chunks:
        chunk.unlink()
    root = zarr.open_group(str(copy), mode="r")
    found = validate_group(root, root.attrs["ld_store"])
    assert {rule: value for rule, value in found.items() if value} == {}, "every rule passes"
    assert set(found) == set(VALIDATION_RULES)
    plain, older = answers(TEST_STORE), answers(copy)
    for name, answer in plain.items():
        np.testing.assert_array_equal(older[name], answer, err_msg=name)
