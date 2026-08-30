# Working on Kinema

## Never commit to `main`

`main` advances **only** through a reviewed PR. Every task that changes files starts on its
own branch:

```bash
git checkout -b <type>/<short-slug>     # fix/ feat/ docs/ chore/
```

Commit there, push the branch, open a **draft** PR, and wait for review before going
further. Do not merge, and do not switch back to `main` to keep working.

Tag pushes (`git push origin v0.1.0`) are fine on `main` — they cut a release and do not
advance the branch.

This is enforced in two places.

**Locally**, `.claude/hooks/guard-main.ps1` refuses `git commit` and `git push` while HEAD is
`main`, and refuses any push naming `main` as its target from any branch. It fails open: if
it cannot determine the branch, the command proceeds.

**On GitHub**, the `main_protect` ruleset requires a pull request and blocks force-pushes and
deletion on the default branch, with **no bypass actors** — direct pushes are refused for
everyone, including the owner. Merges are limited to rebase and squash. Zero approvals are
required, since GitHub does not allow self-approval; the PR is the gate, not the count.

The absence of a bypass is deliberate. A coding agent working here uses the owner's
credentials, so GitHub cannot distinguish the two — an owner bypass would be an agent bypass.
To push directly in an emergency, disable the ruleset in *Settings → Rules* for the moment it
is needed.

## Where the process docs live

- **Cutting a release** — `RELEASING.md`. Manual by design; the version lives in two
  hand-maintained files and a test asserts they agree.
- **The demo GIFs and the IK measurements behind them** — `tools/demo/README.md`, including
  the rules that keep the Blender-IK comparison fair.
- **Build, test and dependency layout** — `README.md`.

## Two traps worth knowing

- **The dev link goes stale.** `tools/dev.py link` junctions `src/kinema` into Blender's
  extensions directory, and it can revert to a real directory holding old code — after which
  `dev.py test` passes against stale source. After linking, check
  `(Get-Item <path>).LinkType` is `Junction`. `dev.py build` is unaffected; it reads
  `src/kinema` directly.
- **`dist/` is never cleaned.** `dev.py build` writes into whatever is already there, and
  same-version zips collide. Remove `dist/` before a release build.
