# opengwasdb

Standalone storage and query engine for OpenGWAS-scale summary statistic stores.

The store contract:

- self-contained store releases;
- embedded SQLite for metadata and lookup indexes;
- Zarr for compressed association arrays;
- layout-independent build and query APIs;
- optional reference completion using LD reference panels.

OpenGWASDB implements **Dense**, **Ragged**, and **Hybrid** layouts, with observed-only and reference completion; [docs/store-format-compatibility.md](./docs/store-format-compatibility.md) lists the `format_version`s each release reads and writes. Service and catalogue deployment are not part of this package.

For the broader OpenGWAS platform direction, see [docs/opengwas-roadmap.md](./docs/opengwas-roadmap.md).

## Repository status

Pre-release, working toward a release candidate (see
[Roadmap 1](https://github.com/opengwas/opengwasdb/issues/88)). Dense, Ragged
and Hybrid layouts build, query, validate and reference-complete; the store
format is not yet stable and existing Store Releases will need rebuilding
before the candidate — see [CHANGELOG.md](./CHANGELOG.md) for what changed and
what is known to be missing.

The design baseline lives in:

- [CONTEXT.md](./CONTEXT.md) — domain glossary; the authority on vocabulary
- [docs/spec/store-format.md](./docs/spec/store-format.md) — the store contract
- [docs/adr/](./docs/adr/) — decisions and their consequences
- [CONTRIBUTING.md](./CONTRIBUTING.md) — how work moves through the repository

Work lands through a pull request to `main`; **every merge is a tagged
version**, noted by its `changelog.d/` fragments.

## Getting started

`pip install opengwasdb` installs the library and CLI on Linux; see
[docs/install.md](./docs/install.md). [docs/getting-started.md](./docs/getting-started.md)
builds, validates and queries a tiny Store Release.

## Development

All Python dependencies and native tooling (bcftools) are managed by
[Pixi](https://pixi.sh) from `pyproject.toml`'s `[tool.pixi.*]` tables. A
fresh checkout needs only Pixi installed:

```bash
pixi run -e dev test        # pytest
pixi run -e dev lint        # ruff check .
pixi run -e dev typecheck   # mypy opengwasdb
```

For a one-off script or REPL, use `pixi run -e dev python <script>.py`
rather than invoking a bare interpreter.

`pip install -e ".[dev]"` also works, without `bcftools`. The version comes
from the `v*` tags; clone with them.

## Licence

[Sustainable Use License](./LICENSE), version 1.0: use is limited to internal
business, non-commercial and personal purposes. It is not an OSI open-source
licence.
