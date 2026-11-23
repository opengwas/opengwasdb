# Sorted-spelling identity fixtures

`orient.tsv`, `chromosomes.tsv` and `reversed.tsv` are the engine's answers at opengwasdb
`1ab2b09`, before ADR 0079 spelt indels reference first, when every ALID was spelt sorted. ogld
made them with `make_fixture.py` from the engine's `orient_to_canonical` and
`normalise_chromosome`, and committed them as an independent record of the spelling an LD Store
keeps until #462. `tests/ld/test_sorted_spelling_conformance.py` holds
`opengwasdb.variants.sorted_spelling` to them.

They are frozen: nothing regenerates them. `make_fixture.py` was retired in #456: today's engine
spells indels reference first, so re-running it would no longer produce these answers, and an
answer made by the code under test is no check on it. It is in this file's history
(`git log --follow -- tests/ld/data/identity/make_fixture.py`). Each file's first line still
names it.
