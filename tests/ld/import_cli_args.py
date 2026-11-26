"""The command line shared by the `ld import` tests: the small Pan-UKB fixture's inputs."""

from pathlib import Path

FIXTURE = Path(__file__).parent / "data" / "panukb"


def required_import_args(out: Path, work: Path) -> list[str]:
    """`import OUT --work WORK` and the four fixture inputs every import needs."""
    return [
        "import",
        str(out),
        "--work",
        str(work),
        "--metadata",
        str(FIXTURE / "metadata.json"),
        "--variants",
        str(FIXTURE / "variants.tsv"),
        "--chain",
        str(FIXTURE / "identity.chain"),
        "--fasta",
        str(FIXTURE / "reference.fa.gz"),
    ]
