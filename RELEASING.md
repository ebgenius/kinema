# Releasing Kinema

Releases are cut by hand. Building needs a Blender binary plus the ~341 MB wheel payload,
which is a lot to ask of CI for a project with one maintainer, so this file is the process
rather than a workflow file. Automate it when the manual path has been walked a few times.

Two artefacts come out of a release: a **GitHub release** carrying three platform zips, and
an **extensions.blender.org** submission built from the same zips.

## 1. Bump the version

The version lives in **two** hand-maintained places and they must agree:

| File | Field |
|---|---|
| `src/kinema/blender_manifest.toml` | `version` — this is the one that ships |
| `pyproject.toml` | `project.version` — dev-only, never published |

```bash
uv lock                  # propagates the pyproject bump into uv.lock
uv run pytest tests/unit/test_manifest.py::test_version_matches_pyproject
```

That test exists because a release cut from a half-done bump would publish under the old
number, silently.

The extensions platform keys a release on this version and will reject a re-upload of one
it already has, so the number must be new every time.

## 2. Update the changelog

Move items out of `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section in
`CHANGELOG.md`, and add the link reference at the bottom. For anything after 0.1.0 the link
is a compare range:

```markdown
[0.2.0]: https://github.com/ebgenius/kinema/compare/v0.1.0...v0.2.0
```

## 3. Build and verify

```bash
rm -rf dist                                # build/ does NOT clean; same-version zips collide
uv run python tools/vendor.py --check      # offline; exits 1 if a pin is stale
uv run python tools/fetch_wheels.py --clean   # only when the payload should change
uv run pytest                              # PYTHONUTF8=1 on Windows
uv run python tools/dev.py link            # see the warning below
uv run python tools/dev.py test
uv run python tools/dev.py validate
uv run python tools/dev.py build
```

> **Check the dev link before trusting `dev.py test`.** The junction into Blender's
> extensions directory has silently reverted to a stale *copy* more than once; the suite
> then tests old code and passes. After `dev.py link`, confirm the target really is a link:
>
> ```powershell
> (Get-Item "$env:APPDATA\Blender Foundation\Blender\5.2\extensions\user_default\kinema").LinkType
> ```
>
> It must print `Junction`, not blank. `dev.py link` sometimes reports
> `mklink failed: Cannot create a file when that file already exists` while still leaving a
> correct junction — Windows defers the directory delete. Verify by content, not by exit code.
>
> This does not affect the built zips: `dev.py build` reads `src/kinema` directly.

`dev.py build` writes `dist/kinema-<version>-<platform>.zip` with **underscored** platform
suffixes (`windows_x64`, `linux_x64`, `macos_arm64`). Confirm all three exist, that their
timestamps are newer than `git log -1 --format=%ci`, and that none exceeds the extensions
platform's ~200 MB limit.

Then install one zip into a clean profile — point `BLENDER_USER_RESOURCES` at an empty
directory — import a UR5e, and check IK solves. A payload with a missing wheel installs
perfectly and only fails at the first `import jax`, in front of a user.

## 4. Tag and publish

```bash
git tag -a vX.Y.Z -m "Kinema vX.Y.Z"
git push origin vX.Y.Z

gh release create vX.Y.Z --title "Kinema vX.Y.Z" --notes-file <notes.md> \
    dist/kinema-X.Y.Z-windows_x64.zip \
    dist/kinema-X.Y.Z-linux_x64.zip \
    dist/kinema-X.Y.Z-macos_arm64.zip
```

Annotated tags (`-a`), so `git describe` treats them as real release points. On Windows,
`gh` may not be on `PATH`; it installs to `C:\Program Files\GitHub CLI\gh.exe`.

Uploading ~360 MB across three assets takes a while.

## 5. Submit to extensions.blender.org

A web upload followed by a public moderation queue — there is no API for this.

**First ever submission:** upload one zip at <https://extensions.blender.org/submit/>.

**Every release, including the first:** the platform takes one file per upload, so the
other two platforms go up through **"upload new version"** on the *same listing*. They are
matched into a single version when the id and version agree and only `platforms` differs.
The UI does not describe this as "add another platform", which is why it looks like you are
about to create duplicates. You are not.

Paste the release notes into the version's description field.

Reviewers check the manifest, the declared permissions and the licensing. Kinema declares
`network` and `files` with reasons, and dual GPL-3.0-or-later / MIT licensing with the full
texts in `LICENSES/`.

The manifest **cannot be edited on the website** — a change means a new upload, or
converting the extension to a draft. Get `website`, `tagline`, `tags` and `permissions`
right before submitting.

## 6. Update the docs site

The site at <https://ebgenius.github.io/kinema/> lives on the **`github-page` branch**, not
`main`. Nothing links the two, so it does not update itself and it drifts silently — by
v0.1.0 the catalog page still described a fetcher that had been replaced two months earlier.

Work on a branch off `github-page` and open a PR into it, rather than pushing: that branch
deploys on push, so a push publishes unreviewed. Note that the `copilot_review` ruleset
targets the default branch only, so nothing reviews these PRs automatically.

**Always:**

- `docs/getting-started/install.md` — **six** occurrences of the version across three
  filenames, plus the three sizes. Take the sizes from the release assets and note the page
  quotes **decimal MB**, which is what GitHub's own asset listing shows.

**Whenever the sidebar changed** — which is most releases, and the part that actually rots:

| File | What goes stale |
|---|---|
| `docs/reference/sidebar.md` | The panel reference. First to fall behind, and it states the panel *count* in its opening line. |
| `docs/concepts/tcp.md` | Any change to how the TCP is placed, or to what IK aims at. |
| `docs/concepts/ik.md` | Solver behaviour, the budget, what a solve costs. |
| `docs/tutorials/animate-ik.md` | Its "panel while you work" table mirrors the IK panel row for row. |
| `docs/tutorials/bake.md` | Mirrors the bake dialog's options. |
| `docs/getting-started/first-robot.md` | The five-minute tour names each panel in order. |
| `docs/assets/images/*.png` | `kinema_ik_solve.png` and `kinema_viewport_overview.png` both show the sidebar; the second is the site's hero image. |

Two traps:

- `mkdocs build --strict` is what CI runs, and it turns a broken internal link or a dead
  anchor into a build failure. Renaming a heading means updating every inbound link in the
  same commit. Run it locally first — `uv sync --group docs && uv run mkdocs build --strict`.
- The workflow's `paths:` filter is `docs/**`, `mkdocs.yml`, `pyproject.toml`, `uv.lock`,
  `.github/workflows/docs.yml`. A push touching only `README.md`, `src/`, `tests/` or
  `tools/` **does not deploy**. Use `workflow_dispatch` if you need it to.

## 7. After

- Confirm `gh release view vX.Y.Z` lists three assets.
- Confirm the docs site actually rebuilt: the *docs* workflow run should be green and the
  install page should show the new version.
