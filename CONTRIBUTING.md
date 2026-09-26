# Contributing to Kinema

Bug reports, suggestions and pull requests are all welcome.

## Reporting a bug or suggesting something

Open an [issue](https://github.com/ebgenius/kinema/issues). For a bug, these help most:

- the Kinema and Blender versions;
- the robot description, as a link or the file itself;
- what you did, what you expected, and what happened instead.

A suggestion needs nothing more than the idea and what you'd use it for.

## Pull requests

Pull requests from forks are welcome.

1. Fork the repository and branch from `main`.
2. Make the change, with tests where it changes behaviour. The README's
   [Development](README.md#development) section covers setup. `uv run pytest` runs the
   unit tests, and `uv run python tools/dev.py test` runs the full suite inside Blender.
3. Open the pull request against `main`, saying what it changes and why.

`main` only changes through reviewed pull requests, so nothing is merged without one.

## How pull requests are reviewed

- **In Blender, by hand.** The maintainer checks the user-facing side: how the change looks
  and behaves in the sidebar, the dialogs and the viewport.
- **The code, by LLM agents.** Automated reviewers read the code and comment on the pull
  request, and their findings are answered in its threads.

## License

Kinema is GPL-3.0-or-later (see [LICENSE](LICENSE)). By contributing, you agree to your
contribution being licensed the same way.
