"""The Proxy Index is gone, and a store that still carries one reads and validates alike (#72).

Three seams, each observed to fail against the code before #72:

- **No `index/`.** The package source names no array of the component, and a store this code
  writes has no such key: the store's own keys are listed and none begins with `index`. The one
  place that still names it is the removal `ogld derive-partners --drop-index` makes, and a
  migration that writes Far Pairs from a store too old to read (ADR 0007).
- **A leftover `index/`.** A 0.2.0 store derived before #72 without `--drop-index` holds the
  component. It opens, answers exactly as the store without one does, and validation ignores it
  rather than reporting it (spec §12) — even when every entry of it is gone, which a read of it
  would have to refuse.
- **A 0.1.0 store** still opens and answers `matrix` and `region`, which read the Near Band and
  the Far Pairs; `clump` and `proxies` are refused, as they are now, naming the Partner Rows.
"""

from __future__ import annotations

import ast
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from store_copies import FLOOR, R_SCALE, T_PARTNER, TEST_STORE, copy_store

import ogld
from ogld.format import VALIDATION_RULES, InputError, Refusal, REncoding
from ogld.identity import orient
from ogld.store import StoreReader, StoreWriter, Variants, layout
from ogld.store.validation import validate_group

#: The package whose source must not name the component.
PACKAGE = Path(ogld.__file__).resolve().parent
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
    # `ogld/store/assembly.py` is where `--drop-index` removes it. Nothing reads a path under it:
    # the two migration helpers take the group as an argument (ADR 0007's `derive_far_pairs`).
    found = {
        str(path.relative_to(PACKAGE)): index_paths(path) for path in sources() if index_paths(path)
    }
    assert list(found) == ["store/assembly.py"], found
    lines = {number for number, _ in found["store/assembly.py"]}
    text = (PACKAGE / "store" / "assembly.py").read_text().splitlines()
    for number in lines:
        assert text[number - 1].strip() in {
            '_removed(store / "index")',
            'counts.pop("index_entries", None)',
        }, (
            number,
            text[number - 1],
        )


def small_store(out: Path) -> Path:
    """A store this code writes, of four variants on two chromosomes, from a known r."""
    position = np.array([1_000_000, 4_500_000, 2_000_000, 6_000_000], dtype=np.int64)
    chrom = ["1", "1", "2", "2"]
    oriented = [orient(c, int(p), "A", "C") for c, p in zip(chrom, position, strict=True)]
    table = Variants(
        np.array(chrom),
        position,
        np.array([o.alid for o in oriented]),
        np.array([o.a1 for o in oriented]),
        np.array([o.a2 for o in oriented]),
        np.full(4, 0.3),
    )
    writer = StoreWriter.create(
        out,
        variants=table,
        encoding=REncoding(R_SCALE),
        near_bp=3_000_000,
        far_bp=10_000_000,
        n=771,
        k=10,
        t_partner=T_PARTNER,
    )
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
    """Give `store` a Proxy Index a store derived before #72 still holds: one entry a chromosome,
    valid enough that reading it would answer, and never read by this package."""
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
    found: dict[str, np.ndarray] = {"matrix": ogld.open(store).matrix(names).r}
    for chrom in ("21", "22"):
        found[f"region-{chrom}"] = ogld.open(store).region(chrom, 30_000_000, 5_000_000).r
    return found


def test_a_leftover_index_answers_exactly_as_a_store_without_one(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    held = "index/21/r" in keys_of(copy)
    assert held, "the copy must hold the component for this to mean anything"
    plain, leftover = answers(TEST_STORE), answers(copy)
    assert leftover["matrix"].shape == (1056, 1056), "a matrix of every variant"
    for name, answer in plain.items():
        np.testing.assert_array_equal(leftover[name], answer, err_msg=name)


def test_validation_ignores_a_leftover_index(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    root = zarr.open_group(str(copy), mode="r")
    found = validate_group(root, root.attrs["ld_store"])
    assert {rule: value for rule, value in found.items() if value} == {}, "every rule passes"
    assert set(found) == set(VALIDATION_RULES)


def test_a_leftover_index_whose_entries_are_lost_is_still_not_read(tmp_path: Path) -> None:
    # A read of any of its arrays would refuse them (spec §6): a matrix and a region answer on.
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    for chunk in sorted(copy.glob("index/*/*/c/*")):
        chunk.unlink()
    plain, lost = answers(TEST_STORE), answers(copy)
    for name, answer in plain.items():
        np.testing.assert_array_equal(lost[name], answer, err_msg=name)


def a_0_1_0_store(to: Path) -> Path:
    """The Test Store as a 0.1.0 store records itself: no Partner Rows claimed, no t_partner."""
    copy = copy_store(to)
    root = zarr.open_group(str(copy), mode="r+")
    recorded = dict(root.attrs["ld_store"])
    recorded["format_version"] = "0.1.0"
    for key in ("t_partner", "partner_code_floor"):
        recorded.pop(key, None)
    recorded["counts"] = {
        key: value for key, value in recorded["counts"].items() if key != "partner_entries"
    }
    recorded["software"] = {
        key: value for key, value in recorded["software"].items() if key != "zstd"
    }
    root.attrs.update({"ld_store": recorded})
    return copy


def _clump_or_proxies(query: str, store: Path, names: list[str], scores: np.ndarray) -> None:
    if query == "clump":
        ogld.open(store).clump(names, scores, r2=0.01, kb=3000)
    else:
        ogld.open(store).proxies(names, r2=0.5, kb=3000)


def test_a_0_1_0_store_opens_and_answers_matrix_and_region(tmp_path: Path) -> None:
    copy = a_0_1_0_store(tmp_path / "store")
    plain, older = answers(TEST_STORE), answers(copy)
    for name, answer in plain.items():
        np.testing.assert_array_equal(older[name], answer, err_msg=name)


@pytest.mark.parametrize("query", ["clump", "proxies"])
def test_a_0_1_0_store_is_refused_for_clump_and_proxies_naming_the_partner_rows(
    query: str, tmp_path: Path
) -> None:
    copy = a_0_1_0_store(tmp_path / "store")
    names = list(StoreReader.open(copy).variants(np.arange(20)).alid)
    scores = np.linspace(5.0, 1.0, len(names))
    ask = partial(_clump_or_proxies, query, copy, names, scores)
    with pytest.raises((InputError, Refusal), match="Partner Rows"):
        ask()
