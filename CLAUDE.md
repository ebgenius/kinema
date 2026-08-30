# Working on Kinema

## Never commit to `main`

`main` advances **only** through a reviewed PR rebase, done by the maintainer. Every task
that changes files starts on its own branch:

```bash
git checkout -b <type>/<short-slug>     # fix/ feat/ docs/ chore/
```

Then: commit as you go, push the branch, open a **draft** PR with the plan summary as its
body — and **stop there and wait for review**. Do not merge, and do not switch back to
`main` to continue working.

Tag pushes (`git push origin v0.1.0`) are fine on `main`; they cut a release and do not
advance the branch.

This is enforced twice, not just documented.

**Locally**, `.claude/hooks/guard-main.ps1` refuses `git commit` and `git push` when HEAD is
`main`, and refuses any push naming `main` as its target from anywhere. It exists because the
written rule alone did not hold — five commits landed directly on `main` in a single session
before anyone noticed. The hook fails open, so if it cannot determine the branch it gets out
of the way.

**On GitHub**, the `main_protect` ruleset applies `pull_request`, `non_fast_forward` and
`deletion` to the default branch with **no bypass actors** — so direct pushes, force-pushes
and deletion are refused for everyone, the owner included. Zero approvals are required, since
GitHub will not let you approve your own PR and there is one collaborator; the PR itself is
the gate, not the approval count. Merges are restricted to **rebase**.

No bypass is deliberate. An agent working here authenticates with the owner's credentials, so
GitHub cannot tell them apart — an "admin bypass" would be a bypass for the agent too, and
would not have stopped those five commits. To push directly in a genuine emergency, set the
ruleset to Disabled in *Settings → Rules* for the moment it is needed. That is a conscious act
no agent should take on its own.

## Where the process docs live

- **Cutting a release** — `RELEASING.md`. Manual by design; the version lives in two
  hand-maintained files and a test asserts they agree.
- **The demo GIFs and the IK measurements behind them** — `tools/demo/README.md`, including
  the rules that keep the Blender-IK comparison fair.
- **Build, test and dependency layout** — `README.md`.

## Two traps that have cost time here

- **The dev link goes stale.** `tools/dev.py link` junctions `src/kinema` into Blender's
  extensions directory, and it has silently reverted to a real directory holding old code
  more than once — after which `dev.py test` passes against stale source. After linking,
  check `(Get-Item <path>).LinkType` is `Junction`, and re-link if in doubt. `dev.py build`
  is unaffected: it reads `src/kinema` directly.
- **`dist/` is never cleaned.** `dev.py build` writes into whatever is already there, and
  same-version zips collide. `rm -rf dist` before a release build.
