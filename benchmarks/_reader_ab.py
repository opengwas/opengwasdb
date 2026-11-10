"""Reader A/B: what one source file's read costs, and the floors that bound it.

For each source the harness runs three sides, round by round with the order
alternating and each sample in a fresh interpreter:

* `reader_head` -- the SourceReader this checkout builds, streaming the file's
  associations as Pass 2 reads them: as the reader's association batches where
  it offers them (#318), else one association at a time;
* `reader_base` -- the same, from a detached tree given with
  `--control-module-path` (the epic measured at `d405de0`; both readers changed
  after it, so without this control a slower read cannot be told from a louder
  node);
* `parse_floor` -- the parse alone, by the method recorded in the artifact: a
  `pandas` C-engine read of the seven columns a tabular association row is
  built from plus `z = beta / sebeta`, or the GWAS-VCF reader's own
  `bcftools query` format with no per-record Python. Both floors skip
  orientation and Analysis admission, so they bound the gain rather than
  predict it.

The sides' imported code is fingerprinted (`_artifact.package_fingerprint`), so
an artifact cannot claim a revision it did not run, and every reader side must
stream the same number of associations: a reader that got faster by dropping
rows is a defect, not a speed-up.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import dumps
from math import comb, log
from os import getloadavg
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks._artifact import package_fingerprint
from benchmarks._query_ab import round_order, run_child

#: A direction is named only from this many rounds, and an even number of them,
#: so that head and control each lead half the time. All of `n` rounds falling
#: on one side of 1 happens by chance 2 / 2**n of the time: a quarter at three
#: rounds, which is why three decided nothing.
MIN_ROUNDS = 6
SIGNIFICANCE = 0.05

#: The marker a child prints its one result line under.
CHILD_MARKER = "READER_AB"

#: The entry point each sample re-invokes, so `--child` is this benchmark.
SCRIPT = Path(__file__).resolve().parent / "measure_epic307_evidence.py"

#: The seven source columns a FinnGen association row is built from. `z` is not
#: among them: it is derived from `beta` and `sebeta`, which is what the floor
#: does with its C-engine parse.
FINNGEN_FLOOR_COLUMNS = ("#chrom", "pos", "ref", "alt", "beta", "sebeta", "af_alt")


@dataclass(frozen=True)
class Source:
    """One Source Reader Capability, its reader, and the floor that bounds it."""

    capability: str
    reader_module: str
    reader_class: str
    effect_scale: str
    floor: Callable[[Path], tuple[float, int]]
    floor_detail: Callable[[Path], dict[str, Any]]
    #: The seven-column pandas floor, or the reader's query through bcftools.
    floor_name: str


def _measure_pandas(path: Path) -> tuple[float, int]:
    """The tabular floor: one C-engine parse of the columns, then `z`."""
    import pandas as pd

    started = time.perf_counter()
    frame = pd.read_csv(path, sep="\t", usecols=list(FINNGEN_FLOOR_COLUMNS), engine="c")
    z = frame["beta"].to_numpy(dtype="float64") / frame["sebeta"].to_numpy(dtype="float64")
    elapsed = time.perf_counter() - started
    if z.size != len(frame):
        raise SystemExit(f"{path}: floor derived {z.size} z values for {len(frame)} rows")
    return elapsed, len(frame)


def _measure_bcftools(path: Path) -> tuple[float, int]:
    """The GWAS-VCF floor: the reader's own query, counted and nothing more."""
    from opengwasdb.readers.gwas_vcf import (
        _format_tags,
        _record_query_format,
        _require_bcftools,
    )

    bcftools = _require_bcftools()
    query_format = _record_query_format(_format_tags(path))
    started = time.perf_counter()
    with subprocess.Popen(
        [bcftools, "query", "-f", query_format, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        rows = sum(1 for _ in proc.stdout)
        errors = proc.stderr.read() if proc.stderr is not None else ""
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise OSError(f"bcftools query failed ({proc.returncode}): {errors.strip()[-500:]}")
    return elapsed, rows


def _pandas_detail(path: Path) -> dict[str, Any]:
    import pandas as pd

    return {
        "method": (
            "pandas.read_csv(sep='\\t', engine='c', usecols=<the seven columns the "
            "association row reads>) plus z = beta / sebeta; no orientation and no "
            "row admission"
        ),
        "columns": list(FINNGEN_FLOOR_COLUMNS),
        "pandas_version": pd.__version__,
    }


def _bcftools_detail(path: Path) -> dict[str, Any]:
    from opengwasdb.readers.gwas_vcf import (
        _format_tags,
        _record_query_format,
        _require_bcftools,
    )

    completed = subprocess.run(
        [_require_bcftools(), "--version"], capture_output=True, text=True, check=False
    )
    return {
        "method": (
            "bcftools query with the reader's own record format string, consumed and "
            "counted; no per-record Python, no orientation and no row admission"
        ),
        "query_format": _record_query_format(_format_tags(path)),
        "bcftools": _require_bcftools(),
        "bcftools_version": completed.stdout.splitlines()[0] if completed.stdout else "",
    }


#: Capability -> source. Keyed by the Source Reader Capability strings the build
#: manifest already uses, so an artifact names a source the way a Store does.
SOURCES: dict[str, Source] = {
    "opengwasdb.finngen-r13": Source(
        capability="opengwasdb.finngen-r13",
        reader_module="opengwasdb.readers.finngen",
        reader_class="FinnGenR13Reader",
        effect_scale="LOG_OR",
        floor=_measure_pandas,
        floor_detail=_pandas_detail,
        floor_name="pandas_columns",
    ),
    "opengwasdb.gwas-vcf": Source(
        capability="opengwasdb.gwas-vcf",
        reader_module="opengwasdb.readers.gwas_vcf",
        reader_class="GwasVcfReader",
        effect_scale="SD",
        floor=_measure_bcftools,
        floor_detail=_bcftools_detail,
        floor_name="bcftools_query",
    ),
}


def sides(control: dict[str, Any] | None) -> tuple[str, ...]:
    """The sides one run measures, in the order round 0 runs them."""
    base: tuple[str, ...] = ("reader_head", "parse_floor")
    return ("reader_head", "reader_base", "parse_floor") if control else base


def _reader(capability: str, path: Path) -> Any:
    """The reader for `capability`, imported now so a control tree can win."""
    import importlib

    source = SOURCES[capability]
    module = importlib.import_module(source.reader_module)
    scale = importlib.import_module("opengwasdb.model.enums").StoredEffectScale
    reader_class = getattr(module, source.reader_class)
    return reader_class(path, getattr(scale, source.effect_scale))


def measure(capability: str, path: Path, side: str) -> tuple[float, int]:
    """One measurement: seconds to stream the associations, and how many."""
    source = SOURCES[capability]
    if side in ("reader_head", "reader_base"):
        reader = _reader(capability, path)
        started = time.perf_counter()
        associations = _associations_read(reader)
        return time.perf_counter() - started, associations
    if side == "parse_floor":
        return source.floor(path)
    raise ValueError(f"unknown side {side!r}")


def _associations_read(reader: Any) -> int:
    """How many associations Pass 2 reads from ``reader``, read as it reads them:
    its association batches where it offers them (#318), else its associations
    one at a time. Looked up by name, since a control tree from before #318
    has no `association_batches` to ask."""
    batches = getattr(reader, "stream_association_batches", None)
    if batches is not None:
        return sum(len(batch) for batch in batches())
    return sum(1 for _ in reader.stream_associations())


def child(
    capability: str, path: Path, side: str, module_path: Path | None = None
) -> None:
    """One measurement in this fresh interpreter, printed as one result line.

    `module_path` is inserted ahead of everything before any `opengwasdb`
    import, so `reader_base` runs the tree it names rather than this checkout.
    """
    if module_path is not None:
        sys.path.insert(0, str(module_path))
    import resource

    seconds, rows = measure(capability, path, side)
    package_path, fingerprint = package_fingerprint()
    print(
        f"{CHILD_MARKER} "
        + dumps(
            {
                "side": side,
                "seconds": round(seconds, 4),
                "rows": rows,
                "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "opengwasdb_path": package_path,
                "opengwasdb_fingerprint": fingerprint,
            }
        ),
        flush=True,
    )


def _label(path: Path) -> str:
    name = path.name
    for suffix in (".gz", ".bgz", ".vcf", ".tsv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _one_sample(
    capability: str,
    path: Path,
    side: str,
    round_index: int,
    control: dict[str, Any] | None,
    position: int,
) -> dict[str, Any]:
    """Run one side in a fresh interpreter and record what it cost."""
    load_before = getloadavg()[0]
    argv = ["--child", "--capability", capability, "--path", str(path), "--side", side]
    if side == "reader_base":
        if control is None:
            raise SystemExit("reader_base needs a control tree")
        argv += ["--module-path", str(control["module_path"])]
    child_record = run_child(SCRIPT, argv, CHILD_MARKER)
    return {
        "round": round_index,
        "position": position,
        "side": side,
        "seconds": float(child_record["seconds"]),
        "rows": int(child_record["rows"]),
        "peak_rss_kb": int(child_record["peak_rss_kb"]),
        "opengwasdb_path": child_record["opengwasdb_path"],
        "opengwasdb_fingerprint": child_record["opengwasdb_fingerprint"],
        "load_1m_before": round(load_before, 2),
        "load_1m_after": round(getloadavg()[0], 2),
    }


def _seconds(samples: Sequence[dict[str, Any]], side: str) -> list[float]:
    return [sample["seconds"] for sample in samples if sample["side"] == side]


def _rows(samples: Sequence[dict[str, Any]], side: str) -> int:
    """One row count per side, refusing a run whose own count moved."""
    counts = {int(sample["rows"]) for sample in samples if sample["side"] == side}
    if len(counts) != 1:
        raise SystemExit(f"the {side} side counted different rows between repetitions: {counts}")
    return counts.pop()


def sign_test_p(ratios: Sequence[float]) -> float:
    """The exact two-sided sign test of `ratios` against 1; a tie votes for neither."""
    votes = [ratio > 1 for ratio in ratios if ratio != 1]
    n = len(votes)
    if n == 0:
        return 1.0
    larger = max(sum(votes), n - sum(votes))
    tail = sum(comb(n, k) for k in range(larger, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilcoxon_p(ratios: Sequence[float]) -> float:
    """The exact two-sided Wilcoxon signed-rank p of `ratios` against 1.

    The ranks are of the ratios' logarithms, which are symmetric about zero
    where `ratio - 1` is not. Tied magnitudes share their average rank; ranks
    are doubled so the distribution of rank sums stays in integers and is
    enumerated exactly, which is cheap at these round counts.
    """
    logs = [log(ratio) for ratio in ratios if ratio != 1]
    n = len(logs)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda index: abs(logs[index]))
    doubled = [0] * n
    first = 0
    while first < n:
        last = first
        while last + 1 < n and round(abs(logs[order[last + 1]]), 12) == round(
            abs(logs[order[first]]), 12
        ):
            last += 1
        for place in range(first, last + 1):
            doubled[order[place]] = first + last + 2
        first = last + 1
    positive = sum(rank for rank, value in zip(doubled, logs, strict=True) if value > 0)
    counts: dict[int, int] = {0: 1}
    for rank in doubled:
        grown: dict[int, int] = defaultdict(int)
        for total, count in counts.items():
            grown[total] += count
            grown[total + rank] += count
        counts = grown
    smaller = min(positive, sum(doubled) - positive)
    tail = sum(count for total, count in counts.items() if total <= smaller) / 2**n
    return min(1.0, 2 * tail)


def _verdict(values: Sequence[float], sign_p: float, wilcoxon: float) -> tuple[str, str]:
    """The direction head went, or `not resolved at this load`, and why."""
    n = len(values)
    if n < MIN_ROUNDS or n % 2:
        return (
            "not resolved at this load",
            f"{n} round(s): a direction needs at least {MIN_ROUNDS} and an even number",
        )
    if sign_p >= SIGNIFICANCE or wilcoxon >= SIGNIFICANCE:
        return (
            "not resolved at this load",
            f"sign test p = {sign_p:.3g} and Wilcoxon p = {wilcoxon:.3g}: "
            f"both must be under {SIGNIFICANCE}",
        )
    middle = median(values)
    if middle == 1:
        return "not resolved at this load", "the median ratio is exactly 1"
    return ("slower" if middle > 1 else "faster"), (
        f"sign test p = {sign_p:.3g} and Wilcoxon p = {wilcoxon:.3g} over {n} rounds"
    )


def _paired(
    samples: Sequence[dict[str, Any]], value_side: str, base_side: str
) -> dict[str, Any]:
    """Per-round ratios of `value_side` to `base_side`, with the round's load.

    The sides are interleaved within a round, so their ratio is measured on the
    same minutes of the same night: the load moves by an order of magnitude
    across a run, and an unpaired median can invent a difference that no round
    shows. The verdict is a paired test of those ratios against 1 (see
    `_verdict`): a direction needs six or more rounds, an even number, and
    both the sign test and the Wilcoxon signed-rank test under 0.05.
    """
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        by_round.setdefault(int(sample["round"]), {})[sample["side"]] = sample
    ratios: list[dict[str, Any]] = []
    exact: list[float] = []
    for index in sorted(by_round):
        value = by_round[index].get(value_side)
        base = by_round[index].get(base_side)
        if value is None or base is None:
            continue
        value_seconds = float(value["seconds"])
        base_seconds = float(base["seconds"])
        if not value_seconds or not base_seconds:
            continue
        exact.append(value_seconds / base_seconds)
        positions = (value.get("position"), base.get("position"))
        ratios.append(
            {
                "round": index,
                "ratio": round(value_seconds / base_seconds, 4),
                "seconds": value_seconds,
                "against_seconds": base_seconds,
                "load_1m": base["load_1m_before"],
                "head_first": None if None in positions else positions[0] < positions[1],
            }
        )
    values = [entry["ratio"] for entry in ratios]
    sign_p = sign_test_p(exact)
    wilcoxon = wilcoxon_p(exact)
    verdict, basis = _verdict(exact, sign_p, wilcoxon)
    return {
        "side": value_side,
        "against": base_side,
        "rounds": ratios,
        "n_rounds": len(ratios),
        "above_one": sum(1 for ratio in exact if ratio > 1),
        "head_first_rounds": sum(1 for entry in ratios if entry["head_first"] is True),
        "median_ratio": round(median(values), 4) if values else None,
        "min_ratio": min(values) if values else None,
        "max_ratio": max(values) if values else None,
        "sign_test_p": round(sign_p, 6),
        "wilcoxon_p": round(wilcoxon, 6),
        "verdict": verdict,
        "verdict_basis": basis,
        "medians": {
            value_side: round(median(_seconds(samples, value_side)), 4),
            base_side: round(median(_seconds(samples, base_side)), 4),
        },
    }


def measure_source(
    capability: str,
    path: str | Path,
    *,
    repetitions: int = MIN_ROUNDS,
    control: dict[str, Any] | None = None,
    warm: bool = True,
) -> dict[str, Any]:
    """The sides, interleaved round by round; the row count too.

    Two rounds are enough to run, not to name a direction: `_paired` says "not
    resolved" below six.
    """
    if repetitions < 2:
        raise ValueError("an interleaved A/B needs at least two repetitions per side")
    if capability not in SOURCES:
        raise ValueError(f"unknown Source Reader Capability {capability!r}")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if warm:
        _warm(capability, source)
    measured_sides = sides(control)
    samples: list[dict[str, Any]] = []
    for round_index in range(repetitions):
        for position, side in enumerate(round_order(measured_sides, round_index)):
            samples.append(_one_sample(capability, source, side, round_index, control, position))
    n_rows = _rows(samples, "parse_floor")
    reader_rows = _rows(samples, "reader_head")
    if reader_rows > n_rows:
        raise SystemExit(
            f"{source}: the reader streamed {reader_rows} associations from a {n_rows}-row file"
        )
    comparisons = {"reader_head_vs_parse_floor": _paired(samples, "reader_head", "parse_floor")}
    fingerprints = {side: _fingerprints(samples, side) for side in measured_sides}
    if control is not None:
        _check_control(samples, reader_rows, control, fingerprints, source)
        comparisons["reader_head_vs_reader_base"] = _paired(
            samples, "reader_head", "reader_base"
        )
    return {
        "label": _label(source),
        "capability": capability,
        "path": str(source.resolve()),
        "input_bytes": source.stat().st_size,
        "n_rows": n_rows,
        "reader_rows": reader_rows,
        "parse_floor": SOURCES[capability].floor_detail(source),
        "sides": list(measured_sides),
        "samples": samples,
        "medians": {
            side: round(median(_seconds(samples, side)), 4) for side in measured_sides
        },
        "comparisons": comparisons,
        "fingerprints": fingerprints,
        "control": (
            None
            if control is None
            else {
                "revision": control.get("revision"),
                "fingerprint": control.get("opengwasdb_fingerprint"),
                "fingerprint_basis": control.get("fingerprint_basis"),
            }
        ),
    }


def _fingerprints(samples: Sequence[dict[str, Any]], side: str) -> list[str]:
    """Every fingerprint the `side` samples imported, in round order."""
    return [sample["opengwasdb_fingerprint"] for sample in samples if sample["side"] == side]


def _check_control(
    samples: Sequence[dict[str, Any]],
    reader_rows: int,
    control: dict[str, Any],
    fingerprints: dict[str, list[str]],
    source: Path,
) -> None:
    """Refuse a run whose control did not run the tree it was given.

    `--control-module-path` only works if the child imports it before any
    `opengwasdb` module, and a single top-level import in any harness module
    would silently make control and head the same code -- which would read as
    "unchanged". The fingerprint of the archive is compared with the one each
    control sample reports, and the control must not be this checkout.
    """
    expected = control.get("opengwasdb_fingerprint")
    imported = set(fingerprints.get("reader_base", []))
    if not expected:
        raise SystemExit("the control tree has no fingerprint to check")
    if imported != {expected}:
        raise SystemExit(
            f"{source}: the control side imported {sorted(imported)}, not {expected}; "
            "the control tree named by --control-module-path was not the code that ran"
        )
    if expected in set(fingerprints.get("reader_head", [])):
        raise SystemExit(
            f"{source}: the control's fingerprint equals this checkout's; "
            "--control-module-path did not take effect"
        )
    base_rows = _rows(samples, "reader_base")
    if base_rows != reader_rows:
        raise SystemExit(
            f"{source}: reader_base streamed {base_rows} associations where reader_head "
            f"streamed {reader_rows}; a reader that changes the row count is not a speed-up"
        )


def _warm(capability: str, source: Path) -> None:
    """One untimed floor read, so every sample meets a warm OS cache."""
    run_child(
        SCRIPT,
        ["--child", "--capability", capability, "--path", str(source), "--side", "parse_floor"],
        CHILD_MARKER,
    )
