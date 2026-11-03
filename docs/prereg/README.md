# Preregistrations

Each registration `<name>` has two files here:

- `<name>.md`: the plan, frozen by the annotated tag `prereg/<name>`. It never changes after the tag.
- `<name>.osf.yaml`: every value entered on OSF, and the registration's URL and date.

## Registering on OSF

One-time setup, already done: the OSF project `state-dependent-twins` exists, with the GitHub add-on linked to
`jerrydzj/state-dependent-twins`.

Before starting, commit `<name>.md`, create the tag `prereg/<name>`, push both, and write `<name>.osf.yaml` from the
previous one.

1. On OSF, go to Registries, then Add New. Answer Yes to "existing OSF project", choose `state-dependent-twins`, choose
   "Open-Ended Registration" and create the draft.
2. Fill in the metadata from `<name>.osf.yaml`. The description is the text of `<name>.md`, pasted verbatim. The
   license defaults to CC0, so change it to the one in the file.
3. Paste the summary from `<name>.osf.yaml`.
4. Download `https://github.com/jerrydzj/state-dependent-twins/archive/refs/tags/prereg/<name>.zip` and upload it as
   the only supplemental file.
5. On the review page, check that every field matches `<name>.osf.yaml`, including the license. Submit, and make the
   registration public immediately.
6. Do not push to GitHub until OSF emails that archiving is complete, because the add-on archives the repository.
7. Add the registration's URL and date to `<name>.osf.yaml` and commit it as `docs: record OSF registration of <name>`.
