"""ogld#16: each LD Store records its encoding, and the reader, the writer and validation use it.

Stores of the Test Panel's reference r at k = 10 are written at × 511 (the HGDP+1kGP stores'),
× 2047 (UK Biobank EUR's) and × 32767 by the Test Store's own script, once per worker each.
Expected values are the reference r and each scale's literals: a code is computed here as spec §3
writes it (`support.codes`), and each floor is spec §3's, never ogld's arithmetic.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import zarr
from script_modules import load_script
from store_copies import R_SCALE, TEST_STORE

import ogld
from ogld import cli
from ogld.format import RefusalReason

CONTRACT = Path(__file__).resolve().parent / "contract"
REPOSITORY = Path(__file__).resolve().parents[1]
SCALES = (511, 2047, 32767)
#: The far code floor at the Test Panel's t_far(771, 10) = 0.0274211, per scale (spec §3; the
#: Test Panel keeps all 771 EUR members by the owner's ruling).
FLOORS = {511: 85, 2047: 339, 32767: 5426}
SLACK = 1e-9


@cache
def contract(name: str) -> ModuleType:
    """A module of the contract tests: `support`, or `test_store` and its breakers."""
    if str(CONTRACT) not in sys.path:
        sys.path.insert(0, str(CONTRACT))
    return importlib.import_module(name)


_STORES: dict[int, Path] = {}


def store_at(scale: int, factory: pytest.TempPathFactory) -> Path:
    """The Test Panel's reference r at k = 10 at × `scale`: the committed Test Store at its own
    scale, and otherwise written once per worker by the script that wrote it."""
    if scale == R_SCALE:
        return TEST_STORE
    if scale not in _STORES:
        out = factory.mktemp(f"x{scale}") / "store"
        _STORES[scale] = load_script("make_test_store").write_test_store(10, out, scale)
    return _STORES[scale]


@cache
def every_pair(location: Path) -> tuple[ogld.LDMatrix, np.ndarray]:
    """The matrix of every variant of a store at `location`, and the reference r it holds."""
    support = contract("support")
    rows = tuple(int(row) for row in support.passing_rows())
    return ogld.open(location).matrix(support.alids(rows)), support.reference_matrix(10, rows)


# ------------------------------------------------------------------ the round trip


@pytest.mark.parametrize("scale", SCALES)
def test_a_store_reads_back_every_pair_within_half_its_own_step(
    scale: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    m, ref = every_pair(store_at(scale, tmp_path_factory))
    stored = m.kinds() == "stored"
    np.fill_diagonal(stored, False)
    assert stored.sum() > 1000, "the store must hold many pairs for this to mean anything"
    assert np.all(np.abs(m.r[stored] - ref[stored]) <= 1 / (2 * scale) + SLACK)
    # Exactly the code spec §3 gives the reference r, read back at this store's scale.
    np.testing.assert_array_equal(m.r[stored], contract("support").stored_at(ref[stored], scale))


@pytest.mark.parametrize("scale", SCALES)
def test_r_of_one_reads_back_exactly(scale: int, tmp_path_factory: pytest.TempPathFactory) -> None:
    m, ref = every_pair(store_at(scale, tmp_path_factory))
    one = (np.abs(ref) == 1.0) & (m.kinds() == "stored")
    np.fill_diagonal(one, False)
    assert one.sum() > 100, "the Test Panel must hold pairs in perfect LD for this to mean anything"
    np.testing.assert_array_equal(m.r[one], ref[one])


def test_one_process_answers_from_a_511_store_and_a_2047_store(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    support = contract("support")
    hgdp, ukb = (
        ogld.open(store_at(511, tmp_path_factory)),
        ogld.open(store_at(2047, tmp_path_factory)),
    )
    rows = support.passing_rows()[400:520]
    ref = support.reference_matrix(10, tuple(int(row) for row in rows))
    # Interleaved, so neither read can lean on a scale the other left behind.
    first, second, third = (
        hgdp.matrix(support.alids(rows)),
        ukb.matrix(support.alids(rows)),
        (hgdp.matrix(support.alids(rows))),
    )
    held = first.kinds() == "stored"
    assert held.sum() > 1000, "the rows must share many stored pairs for this to mean anything"
    np.testing.assert_array_equal(first.r[held], support.stored_at(ref[held], 511))
    np.testing.assert_array_equal(second.r[held], support.stored_at(ref[held], 2047))
    np.testing.assert_array_equal(third.r, first.r)
    assert (first.r[held] != second.r[held]).sum() > 100, "the two scales must read differently"
    # And each answers a query, from its own codes: proxies of the row with most partners at
    # r² ≥ 0.5 in the reference.
    lead = support.alids(rows[[int(np.argmax((np.nan_to_num(ref) ** 2 >= 0.5).sum(axis=1)))]])
    for store, scale in ((hgdp, 511), (ukb, 2047)):
        found = store.proxies(lead, r2=0.5, kb=500)
        assert len(found.proxy) > 0, "the lead must have proxies for this to mean anything"
        codes = np.rint(found.r * scale)
        np.testing.assert_array_equal(found.r, codes / scale)


# ------------------------------------------------------------------ refused at open


UNIMPLEMENTED = {
    "an int8 container": {"r_dtype": "int8", "r_scale": 127},
    "a scale past int16": {"r_scale": 32768},
    "rounding half away from zero": {"rounding": "half_away_from_zero"},
}


@pytest.mark.parametrize("case", list(UNIMPLEMENTED))
def test_an_encoding_the_reader_does_not_implement_is_refused_at_open_naming_it(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = shutil.copytree(TEST_STORE, tmp_path / "store")
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text())
    recorded: dict[str, Any] = doc["attributes"]["ld_store"]["encoding"]
    recorded.update(UNIMPLEMENTED[case])
    meta.write_text(json.dumps(doc))
    with pytest.raises(ogld.InputError) as refused:
        ogld.open(copy)
    assert repr(recorded) in str(refused.value), refused.value
    assert str(copy) in str(refused.value)
    assert cli.main(["info", str(copy)]) == 2
    assert repr(recorded) in capsys.readouterr().err


# ------------------------------------------------------------------ the floor, the Bound, refusals


@pytest.mark.parametrize("scale", SCALES)
def test_the_far_pairs_and_the_bound_follow_the_store_s_own_floor(
    scale: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    location = store_at(scale, tmp_path_factory)
    support = contract("support")
    assert ogld.open(location).provenance["far_code_floor"] == FLOORS[scale]
    m, ref = every_pair(location)
    pos = m.variants.position
    gap = np.abs(pos[:, None] - pos[None, :])
    same = m.variants.chrom[:, None] == m.variants.chrom[None, :]
    far = same & (gap > 3_000_000) & (gap <= 10_000_000)
    reaches = np.abs(support.codes_at(np.where(far, ref, 0.0), scale)) >= FLOORS[scale]
    kinds = m.kinds()
    assert (far & reaches).any(), "the Test Panel must hold Far Pairs"
    assert (far & ~reaches).any(), "and 3–10 Mb pairs below t_far"
    assert (kinds[far & reaches] == "stored").all()
    assert (kinds[far & ~reaches] == "bound").all()
    assert np.isnan(m.r[far & ~reaches]).all(), "a Bound is never a number"


@pytest.mark.parametrize("scale", SCALES)
def test_every_refusal_holds_at_each_scale(
    scale: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    store = ogld.open(store_at(scale, tmp_path_factory))
    lead = contract("support").alids(contract("support").passing_rows()[:1])
    below = store.t_far / 2
    refusals = [
        lambda: store.proxies(lead, r2=below, kb=5_000),
        lambda: store.proxies(lead, r2=0.5, kb=10_001),
        lambda: store.clump(lead, [1e-9], r2=below, kb=5_000),
        lambda: store.region("22", 30_000_000, 5_000_001),
    ]
    for refused in refusals:
        with pytest.raises(ogld.Refusal) as raised:
            refused()
        assert raised.value.reason in set(RefusalReason)


# ------------------------------------------------------------------ validation at each scale


@pytest.mark.parametrize("scale", SCALES)
def test_every_rule_passes_a_store_of_each_scale(
    scale: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    found = ogld.validate(store_at(scale, tmp_path_factory))
    assert tuple(found) == ogld.format.VALIDATION_RULES
    assert {rule: f for rule, f in found.items() if f} == {}


def failing(store: Path) -> set[str]:
    return {rule for rule, found in ogld.validate(store).items() if found}


@pytest.mark.parametrize("rule", ogld.format.VALIDATION_RULES)
def test_a_breaker_fails_the_same_rules_at_another_scale(
    rule: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The committed Test Store and a × 2047 store, each broken the one way: its own rule fails,
    # and the scale changes no rule's verdict.
    breaker = contract("test_store").BREAKERS[rule]
    here = shutil.copytree(TEST_STORE, tmp_path / "committed")
    there = shutil.copytree(store_at(2047, tmp_path_factory), tmp_path / "x2047")
    breaker(here)
    breaker(there)
    broken = failing(here)
    assert rule in broken, f"{rule} passed a store built to break it"
    assert failing(there) == broken


def test_validation_judges_codes_at_the_store_s_scale(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # 600 is a code of a × 2047 store, and past the scale of a × 511 one: there r̂ would be 1.17.
    for scale, passes in ((2047, True), (511, False)):
        copy = shutil.copytree(store_at(scale, tmp_path_factory), tmp_path / f"x{scale}")
        band = zarr.open_array(copy / "near" / "22" / "r", mode="r+")
        band[7] = 600
        found = ogld.validate(copy)
        assert (not found["near.codes"]) is passes, found["near.codes"]
        if not passes:
            assert "a code past the scale, |code| > 511 at 7" in found["near.codes"][0].message
            with pytest.raises(ValueError, match="600"):
                ogld.open(copy).matrix(contract("support").alids(_chr22_rows()[:20]))


def _chr22_rows() -> np.ndarray:
    support = contract("support")
    rows = support.passing_rows()
    return rows[support.panel_data().variants().chrom[rows] == "22"]


# ------------------------------------------------------------------ one owner


#: What names r's encoding: the scales and the reserved code as numbers, the container's dtype,
#: the constants that held them before ogld#16, and the record's keys read past its owner. A
#: build's `r_scale` input names no scale; it carries the one it is given.
OWNED = re.compile(
    r"\b(?:511|2047|3276[78]|65534)\b"
    r"|\bint16\b|['\"][<>=|]?i2['\"]"
    r"|\b(?:R_SCALE|RESERVED_CODE|R_DTYPE|QUANTISATION_ERROR|ENCODING)\b"
    r"|['\"](?:r_scale|r_dtype|r_reserved_code|rounding)['\"]"
)
#: The encoding's owner (spec §3; docs/architecture/ownership.md).
OWNER = "ogld/format/encoding.py"
#: The phrases that name a scale on purpose, per file: the scale the Test Panel's selection judged
#: Stored r at, and the Test Store script's documentation of the scale it is run at. A literal is
#: let through only inside one of these phrases on its line; any other, such as a code written as
#: `r * 511`, fails.
NAMED_ON_PURPOSE = {
    "scripts/test_panel/params.py": (
        "× 32767, the one encoding there was",
        "The Test Store is × 511 (ogld#16, ADR 0004)",
        "r_scale: int = 32767",
    ),
    "scripts/build_hgdp1kgp.py": (
        "`ogld build --k 10 --r-scale 511` (ADR 0004)",
        "PANEL_SCALE = 511",
        "k = 10, r scale 511, the unrelated keep",
    ),
    "scripts/make_test_store.py": (
        "quantised at × 511, HGDP+1kGP EUR's encoding",
        "make_test_store.py --r-scale 511 [--k 10]",
        "the committed Test Store is × 511, and tests write × 2047 and",
        "× 32767 stores beside it.",
        "511 for the committed Test Store, HGDP+1kGP EUR's",
    ),
}
SCALE_LITERAL = re.compile(r"\b(?:511|2047|32767)\b")


def on_purpose(name: str, line: str, match: re.Match[str]) -> bool:
    """Whether `match` is a scale literal inside one of `name`'s phrases on `line`."""
    if not SCALE_LITERAL.fullmatch(match.group()):
        return False
    for phrase in NAMED_ON_PURPOSE.get(name, ()):
        at = line.find(phrase)
        if at >= 0 and at <= match.start() and match.end() <= at + len(phrase):
            return True
    return False


def test_no_module_outside_the_encoding_names_a_scale_a_dtype_or_a_reserved_code() -> None:
    # Every Python file of the package and of scripts/, docstrings and comments included.
    found = []
    for path in sorted([*REPOSITORY.glob("ogld/**/*.py"), *REPOSITORY.glob("scripts/**/*.py")]):
        name = str(path.relative_to(REPOSITORY))
        if name == OWNER:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            found += [
                f"{name}:{number}: {match.group()}"
                for match in OWNED.finditer(line)
                if not on_purpose(name, line, match)
            ]
    assert found == []
