"""Review a pull request with a third-party model, through OpenRouter.

Copilot has been the second reviewer on every PR in this repository, and it
earned the place: during the 0.3.x work it caught four tests that passed for
reasons unrelated to what they claimed to check. When its credits run out the
gap is real, so this script puts another model in the same seat.

It is deliberately not an agent. It gathers the diff, the **full post-change
body of every file the diff touches**, and ``CLAUDE.md``, sends them in one
request, and prints what comes back. Sending whole files rather than hunks is
what makes a non-agentic reviewer worth having -- a hunk shows that a line
changed, not what the function around it now does -- and it costs nothing here,
because ``deepseek/deepseek-v4-flash-0731`` has a 1.3M-token context window and
this repository's source files are small. Nothing is fetched at runtime beyond
the single API call.

The prompt carries this repository's own failure classes rather than generic
review advice. A reviewer told to "look for bugs" returns essays; one told that
a test asserting on a file it never created is the specific thing that shipped
here four times returns findings.

The API key comes from ``OPENROUTER_API_KEY``. It is deliberately not read from
``~/.dsh/.credentials.yaml``: that store holds an encrypted *grant*, not a
retrievable key, and reversing it would be both fragile and beside the point.

Usage::

    # review the current branch against main, print only
    uv run python tools/review_pr.py --dry-run

    # review a pull request and post the findings as a comment
    uv run python tools/review_pr.py --pr 35
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"

# gh ships outside PATH on this machine often enough to be worth a fallback.
GH_FALLBACKS = (
    r"C:\Program Files\GitHub CLI\gh.exe",
    r"C:\Program Files (x86)\GitHub CLI\gh.exe",
)

# Vendored source is not under review and would crowd out everything that is.
SKIP_PREFIXES = ("src/kinema/vendor/",)
MAX_FILE_BYTES = 200_000

SYSTEM_PROMPT = """\
You are reviewing a pull request against Kinema, a Blender 5.2 add-on that turns
robot descriptions (URDF, xacro, MJCF) into animation-ready armatures with an IK
solver. You are the only reviewer. Be the reviewer who catches the defect that
ships, not the one who comments on style.

You are given the diff, the complete current contents of every file the diff
touches, and the repository's own CLAUDE.md. Review the change, using the full
files for context: the diff tells you what moved, the files tell you what the
code now actually does.

These are the failure classes that have actually shipped in this repository.
Weight them above generic concerns:

1. VACUOUS TESTS. A test that passes for a reason unrelated to what it claims to
   check. Real examples from this repo: an assertion about a path that was never
   created, so the branch under test was never entered; a leak test using two
   fixtures with different names, so the thing it protected could be deleted with
   no effect; a test that skipped on exactly the condition it was written to
   detect. For every new or changed test, ask concretely: what single edit to the
   production code would make this test fail? If you cannot name one, say so.

2. SILENT FALLBACKS. Code that degrades to a worse path and reports nothing. The
   archetype here: the IK solver silently drops from PyRoki to a NumPy fallback
   whenever a reload fails, so a broken rig looks like a working one. Flag any
   except-branch, default value, or fallback that hides a failure from the user.

3. WINDOWS AND ENCODING TRAPS. This repo is developed on Windows and has been
   bitten by: file:// URIs putting the drive letter in the URI authority rather
   than the path; reading files with the system codec (cp1252) instead of as
   bytes or as declared XML; PowerShell unrolling a single-element array so
   .Count throws. Paths, URIs and file reads deserve a hard look.

4. BLENDER EXTENSION POLICY. The add-on must not write to sys.path, start
   threads, or mutate environment variables, and vendored code must import under
   the add-on's own package namespace. Blender's checker rejects the extension
   otherwise.

5. RELEASE INVARIANTS. The version appears in both src/kinema/blender_manifest.toml
   and pyproject.toml, and a test asserts they agree. A change touching one and
   not the other is a defect.

Also report plain correctness bugs, crashes, and resource leaks you find.

OUTPUT FORMAT. Start with one line: `VERDICT: <one sentence>`. Then, if there are
findings, list them ranked most severe first, each as:

### <short title>
**`<path>:<line>`** — what is wrong, then a concrete failure scenario: the input
or state that triggers it and what the user sees.

FINDING NOTHING IS A VALID AND USEFUL RESULT. If the change is correct, say so in
the verdict and stop. Do not pad a clean diff with observations, style notes,
restatements of what the code does, or suggestions to "consider" things. A
reviewer that always produces a list is one that gets ignored. Only report
something you would be willing to defend as a real problem.\
"""


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def find_gh() -> str:
    found = shutil.which("gh")
    if found:
        return found
    for candidate in GH_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    die("the GitHub CLI (gh) was not found on PATH or in its usual install location")
    raise AssertionError  # unreachable, for type checkers


def run(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    """Run a command and return stdout, failing loudly rather than silently."""
    if args and args[0] == "git":
        # Without this, git escapes any non-ASCII byte in a path as octal inside
        # quotes ("docs/\303\274ber.md"), and the escaped name does not resolve
        # in a later `git show`, so the file vanishes from the review.
        args = [args[0], "-c", "core.quotepath=false", *args[1:]]
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        die(f"{' '.join(args[:3])}... exited {result.returncode}:\n{result.stderr.strip()}")
    return result.stdout


def resolve_refs(args: argparse.Namespace, gh: str) -> tuple[str, str, str]:
    """Return (base, head, description) for the revision range under review.

    The base is always the *remote-tracking* branch, never the local one. A local
    ``main`` that has fallen behind ``origin/main`` would drag every commit
    someone else landed in the meantime into the diff, and the reviewer would be
    told they are part of this change.
    """
    if args.pr is None:
        head = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
        if head == args.base:
            die(f"the current branch is {args.base}; check out the branch you want reviewed")
        base = fetch_base(args.base)
        return base, "HEAD", f"local branch `{head}` against `{args.base}`"

    meta = json.loads(run([gh, "pr", "view", str(args.pr), "--json", "title,body,baseRefName"]))
    # Fetch the PR head so the review works from any checkout, not just the one
    # that happens to be on the branch. Resolve it to a sha immediately: the
    # base fetch below overwrites FETCH_HEAD.
    run(["git", "fetch", "--quiet", "origin", f"refs/pull/{args.pr}/head"])
    head = run(["git", "rev-parse", "FETCH_HEAD"]).strip()
    base = fetch_base(meta.get("baseRefName") or args.base)
    body = (meta.get("body") or "").strip()
    description = f"PR #{args.pr}: {meta.get('title', '')}\n\n{body}".strip()
    return base, head, description


def fetch_base(branch: str) -> str:
    """Update remote-tracking refs and return the tracking ref for ``branch``."""
    run(["git", "fetch", "--quiet", "origin"])
    tracking = f"origin/{branch}"
    check = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{tracking}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if check.returncode != 0:
        die(f"{tracking} does not exist — is {branch} the right base branch?")
    return tracking


def changed_files(base: str, head: str) -> list[str]:
    listing = run(["git", "diff", "--name-only", "--diff-filter=d", f"{base}...{head}"])
    return [line.strip() for line in listing.splitlines() if line.strip()]


def file_body(head: str, path: str) -> tuple[str | None, str]:
    """Return (content, reason) for a file after the change.

    ``reason`` is empty when the content came back. Otherwise it says *which*
    of these happened, because they are not equivalent: skipping a vendored
    file is by design, while failing to read one is a defect in this script,
    and collapsing both into "omitted" is how the second stays invisible.
    """
    if any(path.startswith(prefix) for prefix in SKIP_PREFIXES):
        return None, "vendored — not under review"
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "show", f"{head}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        return None, f"COULD NOT BE READ — `git show` failed: {detail}"
    raw = result.stdout
    if b"\x00" in raw:
        return None, "binary"
    if len(raw) > MAX_FILE_BYTES:
        return None, f"too large ({len(raw):,} bytes)"
    try:
        return raw.decode("utf-8"), ""
    except UnicodeDecodeError as exc:
        return None, f"COULD NOT BE READ — not valid UTF-8 ({exc.reason})"


def build_user_message(description: str, diff: str, head: str, paths: list[str]) -> str:
    parts = [f"## What is under review\n\n{description}\n", f"## The diff\n\n```diff\n{diff}\n```\n"]

    rules = REPO_ROOT / "CLAUDE.md"
    if rules.is_file():
        parts.append(
            "## CLAUDE.md — the repository's own working rules\n\n"
            f"```markdown\n{rules.read_text(encoding='utf-8')}\n```\n"
        )

    parts.append("## Full contents of each changed file, after the change\n")
    for path in paths:
        body, reason = file_body(head, path)
        if body is None:
            parts.append(f"### `{path}`\n\n_(omitted: {reason})_\n")
            # An unreadable file is reviewed from its diff hunk alone. Say so on
            # the terminal as well: buried in the prompt, nobody would ever see
            # that a file quietly did not make it into the review.
            if reason.startswith("COULD NOT BE READ"):
                print(f"warning: {path} — {reason}", file=sys.stderr)
            continue
        fence = "```" + (Path(path).suffix.lstrip(".") or "text")
        parts.append(f"### `{path}`\n\n{fence}\n{body}\n```\n")

    return "\n".join(parts)


def request_review(message: str, model: str, api_key: str) -> str:
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "kinema-pr-review",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The response body explains the failure; the request headers carry the
        # key and are never touched here.
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        die(f"OpenRouter returned HTTP {exc.code}:\n{detail}")
    except urllib.error.URLError as exc:
        die(f"could not reach OpenRouter: {exc.reason}")

    # A refusal, a filtered response or a provider-side error all come back
    # shaped like a completion but with a null content, so reach for it
    # defensively rather than letting it surface as an AttributeError.
    choices = body.get("choices") or []
    content = (choices[0].get("message", {}).get("content") if choices else None) or ""
    if not content.strip():
        die(f"OpenRouter returned no review text:\n{json.dumps(body)[:2000]}")
    return content.strip()


def post_comment(gh: str, pr: int, review: str, model: str) -> None:
    comment = f"{review}\n\n---\n*Automated review by `{model}` via OpenRouter.*\n"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", encoding="utf-8", delete=False
    ) as handle:
        handle.write(comment)
        body_file = handle.name
    try:
        run([gh, "pr", "comment", str(pr), "--body-file", body_file])
    finally:
        os.unlink(body_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pr", type=int, help="pull request number to review and comment on")
    parser.add_argument("--base", default="main", help="base branch (default: main)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the review without posting it"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        die(
            "OPENROUTER_API_KEY is not set.\n"
            '  PowerShell, this session:  $env:OPENROUTER_API_KEY = "sk-or-..."\n'
            "  Permanently:               setx OPENROUTER_API_KEY \"sk-or-...\"\n"
            "The key in ~/.dsh is an encrypted grant and cannot be reused here."
        )

    gh = find_gh()
    base, head, description = resolve_refs(args, gh)

    diff = run(["git", "diff", f"{base}...{head}"])
    if not diff.strip():
        die(f"no changes between {base} and {head} — nothing to review")

    paths = changed_files(base, head)
    message = build_user_message(description, diff, head, paths)
    print(
        f"reviewing {len(paths)} changed file(s), {len(message):,} characters of context, "
        f"with {args.model}...",
        file=sys.stderr,
    )

    review = request_review(message, args.model, api_key)
    print(review)

    if args.dry_run:
        print("\n(dry run — not posted)", file=sys.stderr)
    elif args.pr is None:
        print("\n(no --pr given — not posted)", file=sys.stderr)
    else:
        post_comment(gh, args.pr, review, args.model)
        print(f"\nposted to PR #{args.pr}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
