"""Pure helpers and refusals of #246's top-hit shard A/B runner.

The runner itself needs the heavy-job lock: it opens two 32 GB Store Releases
and times a Top-Hit Query on each. What is tested here is what can be: how a
side is parsed, what the reported median is, and the refusals that stop a run
that cannot mean anything -- a missing store, one side, or two sides with the
same top-hit layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from benchmarks.top_hit_shard_ab import _median, _parse_config, main


def test_parse_config_requires_a_label_and_a_path():
    assert _parse_config("sharded=/tmp/x") == ("sharded", Path("/tmp/x"))
    for bad in ("sharded", "=/tmp/x", "sharded="):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_config(bad)


def test_median_is_the_middle_of_every_sample():
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert _median([7.5]) == 7.5


def test_a_path_that_is_not_a_store_release_is_refused(tmp_path: Path):
    """A run that cannot open a store must fail before it times anything."""
    with pytest.raises(SystemExit, match="not a Store Release"):
        main(
            [
                "--config",
                f"a={tmp_path / 'a'}",
                "--config",
                f"b={tmp_path / 'b'}",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )


def test_two_sides_with_the_same_label_are_refused(tmp_path: Path):
    with pytest.raises(SystemExit, match="must differ"):
        main(
            [
                "--config",
                f"a={tmp_path / 'a'}",
                "--config",
                f"a={tmp_path / 'b'}",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )


def test_one_side_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit, match="exactly two"):
        main(["--config", f"a={tmp_path / 'a'}", "--output", str(tmp_path / "out.json")])
