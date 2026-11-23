"""opengwasdb.ld's tests, imported from ogld (#455).

A package, so pytest imports its modules and conftests as `ld.…`: as bare names they shared
`sys.modules` with tests/ (`conftest`, `test_store_reads`, `test_validation`), and whichever
was imported first answered for both. tests/ld/conftest.py puts the directories whose
modules are imported by bare name at the end of `sys.path`, so a name tests/ also has
resolves to tests/'s.
"""
