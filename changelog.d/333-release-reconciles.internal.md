- **The release workflow reconciles instead of trusting one run per merge, and
  publication resumes after a failure (issue #333 review).** GitHub keeps one
  pending run per concurrency group and discards the rest, so three quick merges
  could leave the middle one untagged with nothing to retry it; each run now
  publishes every first-parent commit after the newest `v*` tag that has no tag
  of its own, and any tagged commit whose GitHub Release was never created,
  oldest first. Pushing the tag and creating the Release are separate idempotent
  steps, so a run that dies between them is finished by the next one, and a tag
  is never moved. A push to `main` with no pull request keeps its fragments —
  they set the bump and appear in the notes beside the commit subject, rather
  than being replaced by a patch whose notes claim nothing was read. `git
  describe` is restricted to `v*` as well as the parser, so a stray tag such as
  `99.0.0` on `HEAD` no longer fails a build: verified by building wheels from a
  scratch clone, `0.2.1.dev579+g710b50ff6` with the stray tag and `0.3.0` with
  `v0.3.0` and the stray tag both on `HEAD`.
