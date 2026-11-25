- **`opengwasdb ld import --plan-only` writes the import's plan and lifted tables, then exits,
  with no network access (opengwas/opengwasdb-stores#542).** It makes `WORK/plan.json` and each
  `WORK/<chr>.lifted.npz` with the same identity and the same planning a full import uses, lists
  no bucket and downloads nothing, and never creates `OUT`. The same command without the option
  resumes from that plan. A Pan-UKB Store can now be re-planned from its recorded inputs and its
  plan compared chromosome by chromosome, without the private `run._plan` driver opengwas/ogld#102
  used.
