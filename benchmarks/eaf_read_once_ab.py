"""Interleaved A/B of #253's shared-EAF read: base code against head code.

The committed #242 harness runs one side per invocation, and `_artifact.provenance()`
records the worktree's `HEAD` rather than the `opengwasdb` a run actually
imported. #253's base harness record therefore said `328f536` while the code
that ran was `5cf7f78`, and the supplemental A/B that explained the noisy small
shapes was never preserved. This runner fixes both:

* each side runs in its own interpreter from an explicit `--module-path`
  (`sys.path.insert(0, ...)` before `import opengwasdb`), so the imported code
  is the code named -- not whatever the worktree or a `.pth` happens to resolve;
* each side is recorded by the revision *and* a sha256 fingerprint of the
  `opengwasdb` tree it imported (`benchmarks/_artifact.tree_fingerprint`), so an
  artifact cannot claim a revision it did not run;
* the two sides alternate round by round (`base, head` then `head, base`), so
  machine drift cancels rather than deciding the answer;
* every timed sample, the medians, the commands, the round order and a
  result-identity digest go into one generated JSON artifact.

Usage (needs the heavy-job lock: it queries the store):

    pixi run -e dev python benchmarks/eaf_read_once_ab.py \\
        --store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \\
        --base-rev 5cf7f78 --rounds 3 --reps 25 \\
        --output docs/benchmark-output/opengwasdb_eaf_read_once_ab.json

The child is this same file (`--child`); it prints one `AB_RESULT <json>` line.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks import _query_ab as ab

#: The seven #242 shapes, in a stable order. `bulk` is excluded by default: a
#: whole-Analysis read is ~20 s a sample, which starves the small shapes the
#: A/B exists to settle.
AB_SHAPES = (
    "phewas",
    "regional",
    "regional_one_analysis",
    "random_lookup_10_variants_100_analyses",
    "random_lookup_100_variants_10_analyses",
)
DEFAULT_BASE_REV = "5cf7f78"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _extract_base(repo: Path, rev: str, destination: Path) -> None:
    """A `git archive` of `rev`'s `opengwasdb` into `destination`."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", rev, "opengwasdb"],
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout, check=True)


def _child_main(args: argparse.Namespace) -> int:
    # Before any `opengwasdb` import: the code named is the code that runs.
    # `sys.path` entries must be `str`; a `Path` entry is silently ignored.
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.module_path))
    import opengwasdb

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "module": str(opengwasdb.__file__),
        "shapes": ab.time_shapes(args.store, selection, args.shapes, args.reps),
    }
    print("AB_RESULT " + json.dumps(out), flush=True)
    return 0


def _resolve_selection(repo: Path, store: Path, path: Path) -> dict[str, Any]:
    """The same once-resolved selection the #242 harness uses, saved for the children."""
    from benchmarks.benchmark_store_comparison import StoreSpec, _resolve_selection

    selection = _resolve_selection(StoreSpec(label="ab", path=store))
    path.write_text(json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return selection


def _fingerprint(repo: Path) -> dict[str, str]:
    from benchmarks._artifact import tree_fingerprint

    return {
        "path": str(repo),
        "opengwasdb_fingerprint": tree_fingerprint(repo),
    }


def _child_command(args: argparse.Namespace, module_path: Path, selection: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *ab.child_argv(
            args.store,
            selection,
            args.shapes,
            args.reps,
            module_path=module_path,
            repo_root=args.repo_root,
        ),
    ]


def _run_child(args: argparse.Namespace, module_path: Path, selection: Path) -> dict[str, Any]:
    command = _child_command(args, module_path, selection)
    # `command` is the full argv for the record; `run_child` builds it itself.
    return ab.run_child(Path(__file__).resolve(), command[2:], "AB_RESULT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--selection", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-rev", default=DEFAULT_BASE_REV)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="reuse an already-resolved selection instead of deriving it from the store",
    )
    parser.add_argument(
        "--shapes", nargs="+", default=list(AB_SHAPES), help="which #242 shapes to time"
    )
    args = parser.parse_args(argv)

    if args.child:
        if args.module_path is None or args.selection is None:
            raise SystemExit("--child needs --module-path and --selection")
        return _child_main(args)
    if args.output is None:
        raise SystemExit("--output is required")

    repo = args.repo_root.resolve()
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="eaf_read_once_ab_")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    base_dir = workdir / "base"
    _extract_base(repo, args.base_rev, base_dir)
    selection_path = workdir / "selection.json"
    if args.selection_json is None:
        selection = _resolve_selection(repo, args.store, selection_path)
    else:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selection_path.write_text(
            json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    sides = {
        "base": {"revision": _git(repo, "rev-parse", args.base_rev), **_fingerprint(base_dir)},
        "head": {"revision": _git(repo, "rev-parse", "HEAD"), **_fingerprint(repo)},
    }
    if sides["base"]["opengwasdb_fingerprint"] == sides["head"]["opengwasdb_fingerprint"]:
        raise SystemExit(
            "base and head have the same opengwasdb fingerprint; there is nothing to compare"
        )
    commands = {
        side: _child_command(args, base_dir if side == "base" else repo, selection_path)
        for side in sides
    }

    def run_side(side: str, _index: int) -> dict[str, Any]:
        module_path = base_dir if side == "base" else repo
        child = _run_child(args, module_path, selection_path)
        expected_root = str(module_path / "opengwasdb")
        if not child["module"].startswith(expected_root):
            raise SystemExit(
                f"{side} child imported {child['module']}, not {expected_root}; "
                "the module path did not decide the import"
            )
        return child

    samples, digests, rounds = ab.interleave(list(sides), run_side, args.rounds)
    differing = ab.differing_shapes(list(sides), digests)
    side_medians = ab.medians(samples)
    savings = {
        name: round(1.0 - side_medians["head"][name] / side_medians["base"][name], 4)
        for name in samples["base"]
    }
    artifact = {
        "harness": "eaf_read_once_ab",
        "base_rev_requested": args.base_rev,
        "sides": sides,
        "commands": commands,
        "store": str(args.store),
        **ab.measurement_block(selection, args.shapes, args.reps, args.rounds, samples, rounds),
        "medians_ms": side_medians,
        "head_saving_fraction": savings,
        "identity": ab.identity_block(differing),
        "environment": ab.environment_block(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return ab.identity_verdict(differing)


if __name__ == "__main__":
    raise SystemExit(main())
