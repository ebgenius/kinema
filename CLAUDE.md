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

Failing open is also how it hides a break — a hook that crashes emits no decision, and no
decision means allow, so a broken guard and a working one look identical from outside. After
touching it, run its cases:

```powershell
pwsh -NoProfile -File .claude/hooks/test-guard-main.ps1
```

They cover both directions, because both have gone wrong: a commit message mentioning *main*
was refused as though it were a push, and `git add` on the line above a `git commit` hid the
commit entirely.

**On GitHub**, the `main_protect` ruleset requires a pull request and blocks force-pushes and
deletion on the default branch, with **no bypass actors** — direct pushes are refused for
everyone, including the owner. Merges are limited to rebase and squash. Zero approvals are
required, since GitHub does not allow self-approval; the PR is the gate, not the count.

The absence of a bypass is deliberate. A coding agent working here uses the owner's
credentials, so GitHub cannot distinguish the two — an owner bypass would be an agent bypass.
To push directly in an emergency, disable the ruleset in *Settings → Rules* for the moment it
is needed.

## Copilot reviews on its own

A second ruleset, `copilot_review`, requests a Copilot code review on every pull request
targeting `main`. It is kept separate from `main_protect` on purpose: that one is load-bearing
and a botched `PUT` to it would quietly reopen the branch.

Drafts are **excluded**, which is the point — a PR is reviewed when it is marked ready, not
while it is being iterated on. Pushes to a ready PR are re-reviewed.

The rule type is not in GitHub's REST documentation, which describes only the UI. It was found
by posting a `disabled` ruleset and reading back what the API echoed:

```jsonc
// PUT /repos/ebgenius/kinema/rulesets/21957417
{"type": "copilot_code_review",
 "parameters": {"review_on_push": true, "review_draft_pull_requests": false}}
```

Set `review_draft_pull_requests` to `true` to get feedback on drafts as well, or delete the
ruleset to go back to requesting reviews by hand. It needs a Copilot Pro, Pro+ or Max plan.

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
