## What and why

<!-- What changes, and what problem it solves. Numbers over adjectives:
     "27% undercount, 4,476 hits hidden" beats "significant improvement". -->

Closes #

## What I deliberately did not do

<!-- Optional but valued. The rejected option is the part a reviewer most
     needs, because they will think of it themselves. Delete if nothing. -->

## Checklist

Delete any line that genuinely does not apply — an unticked box with no
explanation reads as "not done".

**Correctness**

- [ ] Failure modes are loud. Nothing clips, substitutes a default, or skips a
      row silently.
- [ ] Missing data is `NaN`/`None`/an explicit sentinel — never `0`, never a
      stand-in value.
- [ ] If this fixes a silent failure, a validation rule now catches its shape.

**Tests**

- [ ] I ran the new test against the *unfixed* code and watched it fail.
- [ ] The fixture is asserted meaningful before anything is asserted about it.
- [ ] Verified against real data where available, with the numbers in the
      commit message.

**Docs** — see [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md#documentation)

- [ ] `changelog.d/<slug>.<type>.md` added (CI enforces this for changes under
      `opengwasdb/`; the `no-changelog` label is the escape hatch). Its type
      sets the version bump.
- [ ] `docs/store-format-compatibility.md` updated if `format_version`, or the
      formats written or read, moved.
- [ ] Spec and ADRs match the code; superseded ADRs carry a banner.
- [ ] Benchmarks re-run if anything they measure changed — or explicitly dated
      in the document.
- [ ] `opengwasdb-stores` updated, or an issue opened there, if this changes
      the CLI surface or a manifest column.
