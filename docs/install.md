# Installing OpenGWASDB

OpenGWASDB is published on PyPI as [`opengwasdb`](https://pypi.org/project/opengwasdb/).
Installing it gives you the Python library and the `opengwasdb` command. You
don't need a clone of this repository, or Pixi.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install opengwasdb
opengwasdb --help
```

## What is supported

| | |
|---|---|
| Operating system | Linux, x86_64 |
| Python | CPython 3.12, 3.13 and 3.14 |

CI installs every release into a clean environment on each of these Pythons
and builds, validates and queries a small Store with it. The package may
install elsewhere (macOS, Linux on ARM), but nothing checks that it works
there yet: see [#441](https://github.com/opengwas/opengwasdb/issues/441).
Native Windows is not supported.

### bcftools is separate

Reading **GWAS-VCF** sources needs [`bcftools`](https://samtools.github.io/bcftools/)
on your `PATH`, and pip cannot install it. Without it, a GWAS-VCF build stops
and says so:

```text
bcftools not found in PATH — install via conda: conda install -c bioconda bcftools
```

`conda install -c bioconda bcftools` and most Linux distributions' `bcftools`
packages both work. Building from TSV, GWAS-SSF or BESD sources, and every
query, needs no bcftools.

### Building LD Stores, and reading Genotype Stores: the `ld-build` extra

Building an LD Store or a Genotype Store from pgen genotypes, and reading a Genotype Store, need
pgenlib and threadpoolctl. They come with [ogquery](https://pypi.org/project/ogquery/)'s `genotype`
extra, which opengwasdb's `ld-build` extra installs (the development and Pixi environments carry it
too; `pip install opengwasdb` alone does not):

```bash
pip install 'opengwasdb[ld-build]'
```

Without them a build that reaches a pgen, or an open of a Genotype Store, stops with ogquery's
`GenotypeExtraMissing` and says to `pip install 'ogquery[genotype]'`; `opengwasdb ld build`
prints that and exits 2. Reading an LD Store needs neither. A Genotype Store is read from a local
path only.

## A first run

This builds the tiny example Store from [Getting started](getting-started.md)
with the installed package. The example's three files come from this
repository at the tag of the version you installed:

```bash
version=$(python -c "import opengwasdb; print(opengwasdb.__version__)")
mkdir -p docs/examples/tiny
for name in manifest.tsv height_eur.tsv ldl_eur.tsv; do
  curl -fsSL -o "docs/examples/tiny/$name" \
    "https://raw.githubusercontent.com/opengwas/opengwasdb/v$version/docs/examples/tiny/$name"
done

opengwasdb build-dense tiny.opengwasdb \
  --manifest docs/examples/tiny/manifest.tsv \
  --store-id tiny --release-id observed-v1 \
  --admission-record tiny.admission
opengwasdb validate tiny.opengwasdb
opengwasdb query-phewas tiny.opengwasdb rs111 --variant-info
```

`validate` prints `valid`, and the query returns one row for each of the two
Analyses. [Getting started](getting-started.md) explains each step and what
the rows mean.

## Upgrading and pinning

```bash
pip install --upgrade opengwasdb      # the newest release
pip install "opengwasdb==0.12.0"      # exactly one release
pip install "opengwasdb~=0.12.0"      # 0.12.x: fixes, nothing breaking
```

Every merge to `main` is a release. Below `1.0.0` a breaking change moves the
minor number (`0.12` → `0.13`), so `~=0.12.0` takes fixes and nothing that
breaks. A Store's `format_version` moves separately from the package's
version. [Store-format compatibility](store-format-compatibility.md) lists the
formats each release reads and writes. Release notes are the
[GitHub Releases](https://github.com/opengwas/opengwasdb/releases), and each
one carries the same two files that are on PyPI.

## Contributing instead

Working on OpenGWASDB itself needs the repository and [Pixi](https://pixi.sh),
which also provides bcftools. See the [README](../README.md) and
[CONTRIBUTING](../CONTRIBUTING.md).

## Licence

OpenGWASDB is released under GPL-3.0-only ([LICENSE](../LICENSE)), and the
University of Bristol holds the copyright. Versions released before this change
keep the Sustainable Use License, version 1.0, they were published under.
