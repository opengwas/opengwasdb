"""plink2, htslib and the binary readers the Test Panel is made with."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_panel.params import Params


def conversion_args(source: str | Path, keep: Path | None, out: Path) -> list[str]:
    """The VCF/BCF-to-pgen conversion `docs/spec/build-inputs.md` documents, for hard calls.

    - `--bcf` for a `.bcf`, else `--vcf`; no `dosage=`, because the HGDP+1kGP callsets carry GT
      only, so the pgen holds hard calls.
    - `--keep` is optional: a fileset may hold more samples than one panel.
    - `--max-alleles 2` drops a record that is still multi-allelic; these sources are split.
    - `--set-all-var-ids @:#:$r:$a` names every variant `chrom:pos:REF:ALT`: v2's ID column is
      `.`, and plink2's own outputs (pruning lists, clumps) name variants by ID. A REF or ALT
      over 1,000 bases stops the conversion rather than truncating an ID into a duplicate.
    - `pvar-cols=filter` writes CHROM, POS, ID, REF, ALT and always FILTER, even when every
      value is `.` (v2's are). A build that requires PASS refuses a pvar without the column, so
      it must be there to say the source had nothing to filter. No INFO: the builder reads none
      from a sequenced panel, and gnomAD's runs to several kB a record.
    """
    flag = "--bcf" if str(source).endswith(".bcf") else "--vcf"
    keep_args = ["--keep", str(keep)] if keep is not None else []
    return [
        flag,
        str(source),
        *keep_args,
        "--max-alleles",
        "2",
        "--set-all-var-ids",
        "@:#:$r:$a",
        "--new-id-max-allele-len",
        "1000",
        "--make-pgen",
        "pvar-cols=filter",
        "--out",
        str(out),
    ]


def run(cmd: list[str], log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log.write_text(" ".join(cmd) + "\n\n" + done.stdout + done.stderr)
    if done.returncode != 0:
        raise SystemExit(f"failed ({done.returncode}): {' '.join(cmd)}\n{done.stderr[-2000:]}")
    return done.stdout


def run_plink2(binary: Path, args: list[str], log: Path, threads: int, memory_mb: int) -> str:
    return run([str(binary), *args, "--threads", str(threads), "--memory", str(memory_mb)], log)


def plink2(params: Params, args: list[str], log_name: str) -> None:
    run_plink2(
        params.plink2,
        args,
        params.work / "logs" / f"{log_name}.log",
        params.threads,
        params.memory_mb,
    )


BLAS_ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def single_threaded_blas() -> None:
    """Re-run the calling script with one BLAS thread unless it already has one.

    A threaded matrix product or QR sums in an order that depends on the thread count, which
    moves a result by an ulp and a written file's bytes with it; numpy reads the setting only
    when it loads, so the process starts again.
    """
    if any(os.environ.get(k) != v for k, v in BLAS_ONE_THREAD.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **BLAS_ONE_THREAD})


def stamp_matches(target: Path, inputs: dict[str, str]) -> bool:
    """A cached `target` is reused only when the inputs recorded beside it are today's."""
    stamp = Path(f"{target}.stamp.json")
    return stamp.exists() and json.loads(stamp.read_text()) == inputs


def write_stamp(target: Path, inputs: dict[str, str]) -> None:
    Path(f"{target}.stamp.json").write_text(json.dumps(inputs, indent=1, sort_keys=True) + "\n")


def stamp_digest(target: Path) -> str:
    """The hash of `target`'s stamp, which a cache built from `target` records as its input."""
    return sha256(Path(f"{target}.stamp.json"))


def source_checksum(path: Path) -> str:
    """A downloaded source's verified checksum (`.complete`, ADR 0002), or its SHA-256."""
    complete = Path(f"{path}.complete")
    if complete.exists():
        return "crc32c:" + str(json.loads(complete.read_text())["crc32c"])
    return "sha256:" + sha256(path)


def conversion_inputs(
    binary: Path, args: list[str], source: Path, keep: Path | None
) -> dict[str, str]:
    """What a conversion's output depends on: the command, the source, the samples, plink2."""
    return {
        "command": " ".join(args),
        "source": source_checksum(source),
        "keep": sha256(keep) if keep is not None else "",
        "plink2": version(binary),
    }


def version(binary: Path) -> str:
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    text = (out.stdout or out.stderr).strip()
    if out.returncode != 0 or not text:
        raise SystemExit(f"{binary} --version failed ({out.returncode}): {out.stderr[-500:]}")
    return text.splitlines()[0].strip()


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_rowid_pvar(pvar: Path, out: Path) -> int:
    """A copy of `pvar` whose IDs are `r<row>`, 0-based record order, so plink2's outputs name
    rows unambiguously. The source IDs are rsIDs or `.`, and repeat."""
    n = 0
    with pvar.open() as src, out.open("w") as dst:
        for line in src:
            if line.startswith("#"):
                dst.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            f[2] = f"r{n}"
            dst.write("\t".join(f) + "\n")
            n += 1
    return n


# 2-bit PLINK 1 codes -> count of the .bim column-5 allele, which plink2 --make-bed sets to ALT.
_CODE = np.array([2, -1, 1, 0], dtype=np.int8)
_LUT = np.stack([_CODE[(np.arange(256) >> (2 * k)) & 3] for k in range(4)], axis=1)


def read_bed(prefix: Path) -> tuple[list[str], list[str], np.ndarray]:
    """bim IDs, fam IIDs, and an int8 (variants, samples) matrix of ALT counts, -1 missing."""
    ids = [line.split("\t")[1] for line in Path(f"{prefix}.bim").read_text().splitlines()]
    fam = [line.split()[1] for line in Path(f"{prefix}.fam").read_text().splitlines()]
    nbytes = (len(fam) + 3) // 4
    raw = np.fromfile(f"{prefix}.bed", dtype=np.uint8)
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise SystemExit(f"{prefix}.bed is not a variant-major PLINK 1 bed")
    g = _LUT[raw[3:].reshape(len(ids), nbytes)].reshape(len(ids), nbytes * 4)[:, : len(fam)]
    return ids, fam, np.ascontiguousarray(g)


def bim_positions(prefix: Path) -> np.ndarray:
    return np.array(
        [int(line.split("\t")[3]) for line in Path(f"{prefix}.bim").read_text().splitlines()]
    )


def export_rows(
    params: Params, pfile: Path, rowid_pvar: Path, rows: list[int], out: Path
) -> tuple[list[int], list[str], np.ndarray]:
    """Genotypes of the given 0-based rows of a pgen, as (rows, IIDs, ALT counts)."""
    extract = out.with_suffix(".extract")
    extract.write_text("".join(f"r{r}\n" for r in rows))
    plink2(
        params,
        [
            "--pgen",
            f"{pfile}.pgen",
            "--pvar",
            str(rowid_pvar),
            "--psam",
            f"{pfile}.psam",
            "--extract",
            str(extract),
            "--make-bed",
            "--out",
            str(out),
        ],
        out.name,
    )
    ids, fam, g = read_bed(out)
    got = [int(i[1:]) for i in ids]
    if got != sorted(rows):
        raise SystemExit(f"{out}: plink2 returned other rows than were asked for")
    return got, fam, g


def relatedness(want: set[str], meta: Path) -> dict[str, bool]:
    """Each of `want`'s gnomAD v3.1.2 `relatedness_inference.related` flag, from the sample
    metadata `meta` (gzip TSV, `s` first): True for a sample outside gnomAD's maximal set of
    unrelated samples. A sample the metadata lacks is refused, naming it, rather than taken as
    unrelated."""
    out: dict[str, bool] = {}
    with gzip.open(meta, "rt") as f:
        col = f.readline().rstrip("\n").split("\t").index("relatedness_inference")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] in want:
                out[c[0]] = bool(json.loads(c[col])["related"])
    missing = want - set(out)
    if missing:
        raise SystemExit(
            f"{len(missing)} samples have no gnomAD relatedness, such as {sorted(missing)[:3]}"
        )
    return out
