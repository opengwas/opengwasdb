"""The Store Reader: the read-only path every reader of a Store Release uses (#464).

Everything the query facade reads -- the release envelope and its format
version, the encoding plan and the decoded planes, the association readers, the
Ragged CSR and By-Variant Index readers, Indexed Variant Subsets, the Top-Hit
Index, the Analyses index, the Variant Axis, and the facade and result
resolution -- lives here, and imports nothing but numpy, zarr, numcodecs, the
standard library (not `sqlite3`) and this package
(`tests/test_store_reader_imports.py`). The rest of opengwasdb imports from it,
never the other way round. It becomes the `ogquery` package (epic #450).

The 0.2.x reads that need pysam or SQLite -- the Store Variant Table's tabix
range fetches and BGZF seeks, and the `variant_aliases` fallback -- are not
here: the Variant Axis calls them through `LegacyVariantTable`, which
opengwasdb implements (`opengwasdb.variants.axis`).
"""
