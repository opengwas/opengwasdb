# Staged Release work isolation and parent-directory commit locking

The Staged Release contract (CONTEXT.md) says a release under construction is
held at a `.{name}.tmp` sibling of its eventual path, published by rename, and
discarded on error. The name was fixed, so every invocation for a destination
used the *same* directory: `staging()` deleted it on entry before creating its
own. A second invocation for a destination therefore deleted the directory the
first was still writing, and the first failed mid-build with
`variant_offsets.npy` ENOENT — a build that appeared to be running while its
output directory had been removed from under it. Three related holes were in
the same code path: cleanup caught only `Exception`, so `KeyboardInterrupt` and
`SystemExit` left the staging directory behind; the destination existence check
and the two-rename swap were not serialised, so two concurrent commits could
interleave the old-release-aside and new-release-in steps; and a no-overwrite
build could clobber a destination published after its entry check.

## Decision

**One work directory per invocation.** `staging()` creates
`.{name}.tmp.{pid}.{random}` beside the destination with `mkdir`, and never
reuses or deletes a directory it did not itself create. There is no pre-emptive
cleanup of a stale or apparently abandoned `.tmp` directory: deleting one is
only safe when this call created it, which is exactly the assumption that was
wrong. A name collision (random, so effectively impossible) raises rather than
being cleaned up.

**Cleanup removes exactly this invocation's directory.** The `with` body is
wrapped in `except BaseException`, so `KeyboardInterrupt`, `SystemExit` and
ordinary exceptions all discard this invocation's own work directory and leave
the destination exactly as found. Re-raising is unconditional; the handler
never swallows the interruption.

**Publication is serialised per parent directory.** The commit runs under
`_destination_lock(dst)`, an advisory `flock` on the destination's *parent
directory inode*, held open with `os.open(dst.parent, os.O_RDONLY)`. A lock
*file* was rejected because every place it could live is wrong: beside the
destination it is renamed into the published release when the destination is a
Hybrid release's nested Dense Component (staged at `<outer-work>/dense`), and
under the system temp directory its identity moves with `TMPDIR` and private
temp namespaces — two callers that do not share a temp root would not contend —
while a tmp cleaner can unlink it mid-hold, dropping the lock with no process
noticing. A directory inode is a stable filesystem identity, exists by the time
a commit runs, and creates no entry that could be published. `flock` is
released by the kernel when the holding process dies, so a crashed builder
cannot leave a destination permanently locked; that is also why an advisory
lock was preferred over a lock *directory* whose staleness would have to be
guessed at.

**The lock is held for the publication window only, not the build.** Under it,
the destination is re-checked: a no-overwrite commit that finds the destination
now present raises `FileExistsError` and discards its own work rather than
clobbering the winner. The entry-time check is fail-fast only and is never what
makes the decision. The swap itself is unchanged — old release renamed aside,
new release renamed in, old deleted, rollback on failure — which keeps the
destination valid across both renames.

**`overwrite=True` is serialised, but intentionally last writer wins.** Two
concurrent overwrite builds both commit and the later replacement wins, with no
error and no record that a release was superseded. The lock orders the swaps
and keeps each atomic; it does not decide which build *should* win, does not
stop two callers from doing redundant work, and deliberately does not remove a
`.{name}.tmp.*` directory orphaned by a killed build (deleting one is only safe
when the current call created it — see the first decision above). A build
orchestrator that must not schedule two builds for one destination, or that
must reap such an orphan, is therefore **complementary to this lock, not
interchangeable with it**: the lock cannot substitute for that scheduling and
cleanup policy, and that policy cannot make the two-rename swap atomic.

**The lock imposes two environment requirements, both fail-loud.** The parent
directory must be openable for reading — `os.open(O_RDONLY)` needs read/search
permission on it, so a write-and-execute-only parent cannot be locked — and the
filesystem must implement `flock` on directories. An unmet requirement raises
and discards the staged work rather than publishing unserialised; there is
deliberately no fallback that proceeds without the lock, because that would
reintroduce the silent race this ADR exists to remove.

## Considered options

- **A lock file in the system temp directory, keyed by a hash of the resolved
  destination.** This was the first implementation, and it was rejected because
  its identity is not the destination's: callers with different `TMPDIR` values
  or private temp namespaces compute different lock files and do not contend at
  all, and a tmp cleaner may unlink the file while it is held, after which a
  fresh inode can be locked by a third process. Both failure modes are silent.
- **A lock file beside the destination (`.{name}.lock`).** Rejected: a Hybrid
  release's nested Dense Component is staged at `<outer-work>/dense`, so a lock
  file beside that destination would be renamed into the published release
  along with its component.
- **Keep the fixed `.tmp` name and take the lock for the whole build.**
  Rejected: it would make a second same-destination build block or fail for the
  entire (possibly hours-long) build; an `flock` fd is inherited by forked
  worker processes, so a worker that outlived the context could hold the lock
  after the build ended; and it does not by itself fix the `BaseException`
  cleanup hole.
- **A lock directory created with `mkdir`.** Rejected: it does not disappear if
  the builder is killed, so staleness detection would have to guess at live
  owners.
- **`tempfile.mkdtemp` for the work directory.** Rejected on a small but real
  behavioural point: `mkdtemp` creates the directory `0700`, whereas the
  previous `mkdir` gave it the umask-derived mode the published release
  inherited. A plain `mkdir` on a unique name keeps that mode.
- **`renameat2(RENAME_NOREPLACE)` for the no-overwrite case.** Rejected: it is
  Linux-specific, and it does not address the overwrite swap, which is the case
  that can interleave.

## Consequences

- The lock is coarser than one destination: every release published into one
  parent directory shares it, so commits to different releases in the same
  directory are serialised. The critical section is the two renames plus
  deletion of a replaced release, and commits happen once per build, so this is
  accepted in exchange for a lock that cannot be moved or reaped out from under
  it. A directory with many concurrent *completion* rebuilds into it will see
  those commits queue behind each other's old-release deletion.
- The lock is advisory and host-local. `flock` is enforced by the local kernel;
  whether it also serialises a process on another host depends on the
  filesystem. NFS and other network filesystems may not propagate `flock`
  between hosts, so this must not be relied on as a cross-host lock. A
  filesystem that refuses `flock` outright fails the commit loudly rather than
  publishing unserialised. Each individual rename is still atomic by the
  filesystem's own guarantee, which is what keeps the destination valid across
  the swap even if the lock is not honoured.
- The lock creates no filesystem entry and leaves nothing to clean up.
- Two concurrent builds for one destination are allowed to both run. They can
  never touch each other's work. A no-overwrite loser is refused at publication;
  an overwrite pair both succeed, last writer wins, and neither is told it was
  superseded — so an orchestrator that must prevent redundant or racing builds
  is complementary, not interchangeable (see Decision).
- The parent-read and directory-`flock` requirements were verified non-
  destructively on the pilot store parent `/data/opengwasdb/stores` (local XFS,
  mode `0755`): `os.open(O_RDONLY)` succeeded, `LOCK_EX|LOCK_NB` was acquired,
  a second handle on the same inode was refused with `EAGAIN`, and the lock was
  released. Nothing on that filesystem was created, renamed or deleted.
- A staging directory left by `SIGKILL` is no longer removed by the next build.
  It is inert and identifiable by the `.{name}.tmp.*` prefix. A `.{name}.old`
  left by a crash between the two commit renames still holds the previous
  release.
