# Kinema — documentation site

**Animation-ready robot rigs in Blender.** Import any of 186 real robots and get a single
clean armature you can actually animate — with IK that understands singularities, joint
limits and multi-turn joints.

**📖 [Read the docs](https://ebgenius.github.io/kinema/)** — install, tutorials, and a
robotics primer written for Blender artists.

## What this branch is

This branch holds **only the sources for the documentation site**: `docs/`, `mkdocs.yml`,
and the workflow that publishes them. There is no add-on here.

**For the add-on itself — source, tests, and how to build or debug it — see
[`main`](https://github.com/ebgenius/kinema/tree/main#readme).**

The two are deliberately separate: the site is deployed from this branch by GitHub Pages, so
it has its own history and its own (much smaller) dependency list.

## Working on the docs

`docs/README.md` has the serve-and-preview commands. In short:

```powershell
uv run --group docs mkdocs serve
```

Before opening a pull request, build the way CI does:

```powershell
uv run mkdocs build --strict
```

`--strict` turns a broken internal link or a dead anchor into a failure, so it catches a
renamed heading that orphans a link elsewhere in the site.

## How it publishes

`.github/workflows/docs.yml` builds and deploys on every push to this branch, so **open a
pull request rather than pushing** — a push publishes straight away.

The workflow only runs for changes under `docs/`, `mkdocs.yml`, `pyproject.toml`, `uv.lock`
or the workflow itself. A change to anything else does not deploy; use *Run workflow* on the
Actions tab if you need it to.

Note that the automatic Copilot review is configured for the default branch only, so pull
requests into this branch are not reviewed automatically.

## Keeping it accurate

The site describes the **released** version, not `main`. Updating it for a new release is
step 6 of [`RELEASING.md`](https://github.com/ebgenius/kinema/blob/main/RELEASING.md), which
lists the pages that go stale when the sidebar changes.

## License

GPL-3.0-or-later (the add-on links `bpy`). Vendored PyRoki and jaxls remain MIT.
