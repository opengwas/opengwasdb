- **`opengwasdb ld build | import | validate | info | derive-partners`, and the
  `OPENGWASDB_LD_*` environment variables (issue #457).** The LD Store commands
  that were `ogld …` are a sub-app of the engine CLI, with the same options,
  required inputs and exit codes (0 done, 1 the store breaks a rule, 2 a
  refusal), messages now `opengwasdb ld <command>: <problem>`. An option left
  out takes the default `BuildInputs` and `ImportInputs` declare. The one change
  to a command line: `derive-partners --chromosomes` is repeated once per
  chromosome rather than taking several values. The scripts' and tests' `OGLD_*`
  variables are `OPENGWASDB_LD_*`; one reader refuses, naming the new name, any
  old variable that is still set rather than let its default apply in silence.
