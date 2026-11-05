"""Command line for the Locus Lead prototype (#402).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eur_map import build_eur_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m prototypes.locus_leads")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-map", help="average the pyrho EUR maps into one file")
    build.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    build.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
    args = parser.parse_args(argv)
    if args.command == "build-map":
        print(json.dumps(build_eur_map(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
