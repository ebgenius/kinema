<#
.SYNOPSIS
    Refuse `git commit` and `git push` while HEAD is the default branch.

.DESCRIPTION
    main advances only through a reviewed PR rebase. Work belongs on a
    type/short-slug branch; see CLAUDE.md.

    This exists because the written rule alone was not enough: five commits
    landed directly on main in one session before anyone noticed. A hook
    notices every time.

    Reads a Claude Code PreToolUse payload on stdin and, when it decides to
    block, prints a permissionDecision of "deny" with a reason.

    Fails OPEN. If the payload will not parse, or git cannot report a branch,
    the command is allowed. A guard that blocks all shell work when it misfires
    is worse than the problem it solves.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Allow {
    # Silence is consent: emitting nothing leaves the permission decision alone.
    exit 0
}

function Deny([string]$Reason) {
    @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = $Reason
        }
    } | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) { Allow }
    $command = ($raw | ConvertFrom-Json).tool_input.command
    if ([string]::IsNullOrWhiteSpace($command)) { Allow }
} catch {
    Allow
}

# `git commit` / `git push`, allowing for `git -C path push`, `git   push`, and
# the command sitting after a `&&` or `;` in a longer line.
$isCommit = $command -match '(^|[;&|]\s*)git\b[^;&|]*\bcommit\b'
$isPush = $command -match '(^|[;&|]\s*)git\b[^;&|]*\bpush\b'
if (-not ($isCommit -or $isPush)) { Allow }

# Changes nothing, so there is nothing to guard.
if ($command -match '--dry-run') { Allow }

# Tag pushes are how a release is cut and never advance a branch.
# `git push origin v0.1.0`, `git push --tags`, `git push origin refs/tags/...`
$isTagPush = $isPush -and -not $isCommit -and
    ($command -match '(--tags\b|refs/tags/|\sv\d)')
if ($isTagPush) { Allow }

# A push that names main explicitly advances main no matter which branch it is
# run from, so the current-branch check below would miss it. Matched as a whole
# argument or refspec target -- `origin main`, `HEAD:main` -- so a branch merely
# containing the word, like fix/domain-main, is untouched.
if ($isPush -and $command -match '(?:\s|:)main(?:\s|$)') {
    Deny(@"
Pushing to 'main' is blocked. main only advances through a reviewed PR rebase.

Push your branch instead, then open a draft PR and wait for review:
    git push -u origin <type>/<short-slug>

See CLAUDE.md.
"@)
}

# symbolic-ref, not `rev-parse --abbrev-ref HEAD`: rev-parse cannot name the
# branch before the first commit exists (it errors and prints "HEAD"), so a
# fresh repo sitting on main would sail straight through. symbolic-ref reports
# "main" there, and fails on a detached HEAD -- which is not main, so allowing is
# the right answer anyway.
try {
    $branch = (& git symbolic-ref --short --quiet HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) { Allow }
    $branch = $branch.Trim()
} catch {
    Allow
}

if ($branch -ne 'main') { Allow }

$verb = if ($isCommit) { 'Committing' } else { 'Pushing' }
Deny(@"
$verb on 'main' is blocked. main only advances through a reviewed PR rebase.

Create a branch first, then commit there:
    git checkout -b <type>/<short-slug>     # fix/ feat/ docs/ chore/

Then push it and open a draft PR, and wait for review before going further.
See CLAUDE.md. (Tag pushes and --dry-run are allowed on main.)
"@)
