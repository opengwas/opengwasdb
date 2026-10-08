#!/usr/bin/env python3
"""Build the committed conversion-cost JSON for a #250 Store from its own log.

Every field is parsed, never typed:

* `wall_seconds`, `peak_rss_kbytes`, `exit_status` come from the
  `/usr/bin/time -v` block in the conversion log;
* `phases_seconds` comes from the converter's printed phase lines;
* `source`/`destination` come from the log's `Command being timed:` line, and
  `published`/`bit_exact` from the converter's own lines;
* the wall-clock window and the 1-minute loads come from the committed
  `epic240_250_monitor.log`, which sampled `p16`/`p11` every 5 minutes;
* the before/after file count and apparent/allocated bytes come from
  `find`/`du` on the two trees.

Run from the repository root:

    python3 scripts/build_conversion_cost.py

Inputs and outputs both live in `docs/benchmark-output/`, so the artifact beside
the log is reproducible from the repository alone.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

OUT = Path("docs/benchmark-output")

#: One JSON per converted Store: its log, the store key in the monitor log, and
#: the output name. The `p16`/`p11` keys are the shell's own job labels, taken
#: from the monitor's first line rather than typed.
STORES = [
    {
        "name": "OGS-00016",
        "log": "opengwasdb_ogs00016_convert_0_2_0.log",
        "monitor_key": "p16",
        "out": "opengwasdb_ogs00016_conversion_0_2_0.json",
    },
    {
        "name": "OGS-00011",
        "log": "opengwasdb_ogs00011_convert_0_2_0.log",
        "monitor_key": "p11",
        "out": "opengwasdb_ogs00011_conversion_0_2_0.json",
    },
]
MONITOR_LOG = "epic240_250_monitor.log"

PHASE_RE = re.compile(r"^  (?P<label>.+): (?P<seconds>[0-9.]+)s$", re.M)
COMMAND_RE = re.compile(r'Command being timed: "(?P<command>[^"]+)"')
MONITOR_RE = re.compile(
    r"^(?P<time>\d\d:\d\d:\d\d) load=(?P<load>[\d.]+) [\d.]+ [\d.]+ "
    r"p16=(?P<p16>\S+) p11=(?P<p11>\S+)$"
)


def footprint(root: Path) -> dict[str, int]:
    n_files = int(
        subprocess.run(
            ["find", str(root), "-type", "f"], capture_output=True, text=True, check=True
        ).stdout.count("\n")
    )
    apparent = int(
        subprocess.run(
            ["du", "-sb", str(root)], capture_output=True, text=True, check=True
        ).stdout.split()[0]
    )
    allocated = int(
        subprocess.run(
            ["du", "-s", "--block-size=1", str(root)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
    )
    return {"n_files": n_files, "apparent_bytes": apparent, "allocated_bytes": allocated}


def _monitor_lines() -> list[dict[str, str]]:
    lines = []
    for line in (OUT / MONITOR_LOG).read_text(encoding="utf-8").splitlines():
        match = MONITOR_RE.match(line)
        if match:
            lines.append(match.groupdict())
    if not lines:
        raise SystemExit(f"{OUT / MONITOR_LOG}: no monitor samples; cannot date the run")
    return lines


def _window(lines: list[dict[str, str]], key: str) -> tuple[dict[str, str], dict[str, str], str]:
    """(first sample, last sample, pid) for the job the monitor labelled `key`."""
    pid = lines[0][key]
    if not pid.isdigit():
        raise SystemExit(f"{MONITOR_LOG}: the first {key} sample is {pid!r}, not a pid")
    alive = [line for line in lines if line[key] == pid]
    if not alive:
        raise SystemExit(f"{MONITOR_LOG}: no sample carries {key}={pid}")
    return alive[0], alive[-1], pid


def _concurrent_load_range(
    lines: list[dict[str, str]], pids: dict[str, str]
) -> tuple[float, float]:
    loads = [
        float(line["load"])
        for line in lines
        if all(line[key] == pid for key, pid in pids.items())
    ]
    if not loads:
        raise SystemExit(f"{MONITOR_LOG}: the two jobs never overlapped")
    return min(loads), max(loads)


def main() -> None:
    lines = _monitor_lines()
    windows = {spec["name"]: _window(lines, spec["monitor_key"]) for spec in STORES}
    pids = {spec["monitor_key"]: windows[spec["name"]][2] for spec in STORES}
    concurrent_min, concurrent_max = _concurrent_load_range(lines, pids)

    for spec in STORES:
        text = (OUT / spec["log"]).read_text(encoding="utf-8")
        maxrss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1))
        wall = re.search(
            r"Elapsed \(wall clock\) time .*: ([0-9]+):([0-9]{2}):([0-9]{2})", text
        )
        wall_s = int(wall.group(1)) * 3600 + int(wall.group(2)) * 60 + int(wall.group(3))
        command = COMMAND_RE.search(text).group("command").split()
        source = Path(command[command.index("scripts/convert_store_to_0_2_0.py") + 1])
        destination = Path(command[command.index("--into") + 1])
        first, last, pid = windows[spec["name"]]
        source_format = json.loads((source / "manifest.json").read_text())["format_version"]
        destination_format = json.loads(
            (destination / "manifest.json").read_text()
        )["format_version"]
        record = {
            "task": "#250",
            "store": spec["name"],
            "source": str(source),
            "destination": str(destination),
            "source_format_version": str(source_format),
            "destination_format_version": str(destination_format),
            "log": spec["log"],
            "wall_seconds": wall_s,
            "peak_rss_kbytes": maxrss,
            "peak_rss_gib": round(maxrss / 1024 / 1024, 2),
            "phases_seconds": {
                m.group("label").strip(): float(m.group("seconds"))
                for m in PHASE_RE.finditer(text)
            },
            "published": f"Published {destination} as 0.2.0" in text,
            "bit_exact": "verified the conversion bit-exact" in text,
            "exit_status": int(re.search(r"Exit status: (\d+)", text).group(1)),
            "monitor": {
                "log": MONITOR_LOG,
                "job_label": spec["monitor_key"],
                "pid": pid,
                "first_sample_utc": first["time"],
                "first_sample_load_1m": float(first["load"]),
                "last_sample_utc": last["time"],
                "last_sample_load_1m": float(last["load"]),
                "both_running_load_1m_min": concurrent_min,
                "both_running_load_1m_max": concurrent_max,
            },
            "before": footprint(source),
            "after": footprint(destination),
        }
        (OUT / spec["out"]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"{spec['out']}: {record['wall_seconds']}s {record['peak_rss_gib']} GiB "
            f"files {record['before']['n_files']}->{record['after']['n_files']} "
            f"window {first['time']}-{last['time']} load {concurrent_min}-{concurrent_max}"
        )


if __name__ == "__main__":
    main()
