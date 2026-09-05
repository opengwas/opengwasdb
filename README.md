# opengwasdb

Standalone storage and query engine for OpenGWAS-scale summary statistic stores.

The project is starting from a clean store contract:

- self-contained store releases;
- embedded SQLite for metadata and lookup indexes;
- Zarr for compressed association arrays;
- layout-independent build and query APIs;
- Dense and Ragged primary layouts;
- optional reference completion using LD reference panels.

The first implementation slice is intentionally narrow: **Dense Observed-Only** stores with `z` and `se` arrays, metadata, validation, and layout-independent queries. Ragged layout, reference completion, and service/catalogue deployment are recorded in the ADRs but are not part of v0.1.

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

Work lands on `dev`; every merge to `main` cuts a tagged version.

## Getting started

Use [docs/getting-started.md](./docs/getting-started.md) for a first run:
install with Pixi, build an in-repository tiny Dense Store Release, validate it,
and run PheWAS, exact-lookup, and top-hit queries. The example needs no external
production data.

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

The package is still a normal `pip install`-able library for downstream
consumers (`pyproject.toml` + hatchling): `pip install -e ".[dev]"` continues
to work if you'd rather manage the environment yourself, but it won't provide
`bcftools` — install that separately in that case.

