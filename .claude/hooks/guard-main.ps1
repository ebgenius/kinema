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

function Get-Args([string]$Command) {
    # Non-flag tokens, with a leading '+' (force refspec) and any surrounding
    # shell quotes stripped.
    #
    # The quotes matter: `git push origin "main"` is an ordinary thing to type,
    # and comparing the token with them still attached made it read as a ref
    # called `"main"` -- so the guard waved it through. Quoting is shell syntax
    # and never part of a ref name.
    ($Command -split '\s+') |
        Where-Object { $_ -and $_ -notmatch '^-' } |
        ForEach-Object { $_ -replace '^\+', '' } |
        ForEach-Object { $_.Trim('"', "'") } |
        Where-Object { $_ }
}

function Remove-HereStrings([string]$Command) {
    # Blank out here-string bodies before anything reads the command structure.
    #
    # A commit message is data, and in this workflow it arrives as @'...'@ --
    # paragraphs of prose sitting inside the same shell call as the push that
    # follows it. Left in, every word of it is a token: a message mentioning
    # "the main thread" made the push check below see a push to main and refuse
    # a commit that was going nowhere near it.
    #
    # Ordinary quotes are deliberately left alone. They are short enough to hold
    # a real ref, and `git push origin "main"` must still be caught.
    $stripped = $Command -replace "(?s)@'.*?'@", ' '
    return $stripped -replace '(?s)@".*?"@', ' '
}

#: Global git options that consume the following token as their value, so the
#: subcommand scan must step over both.
$script:GitOptionsWithValue = @('-C', '-c', '--git-dir', '--work-tree', '--namespace')

function Get-GitSubcommand([string]$Segment) {
    # The subcommand only, e.g. 'commit' from `git -C repo -c k=v commit -m x`.
    #
    # Matching the word anywhere in the line is not good enough: `\bcommit\b`
    # also fires on `git commit-graph` (a '-' is a word boundary) and on
    # `git log --grep=commit`, blocking read-only commands that merely mention
    # it. Only the subcommand position decides.
    $tokens = @($Segment -split '\s+' | Where-Object { $_ })
    $i = [array]::IndexOf($tokens, 'git')
    if ($i -lt 0) { return $null }
    for ($i++; $i -lt $tokens.Count; $i++) {
        $token = $tokens[$i]
        if ($token -in $script:GitOptionsWithValue) { $i++; continue }
        if ($token -match '^-') { continue }
        return $token
    }
    return $null
}

function Get-Segments([string]$Command) {
    # Each `;`, `&&`, `||`, `|` or *newline* separated part is its own command
    # line. The newline matters: a single tool call routinely holds several
    # statements on separate lines, and without it `git add -A` and
    # `git commit -m x` were one segment whose subcommand read as `add` -- so a
    # commit on main went unnoticed, which is the whole thing this guards.
    $Command -split '(?:&&|\|\||[;|]|\r?\n)'
}

function Get-SubcommandSegments([string]$Command, [string]$Name) {
    # Callers must wrap this in @(). PowerShell unrolls a single-element array
    # on return, so one matching segment comes back as a bare string -- and
    # `.Count` on a string is an error under StrictMode, which would take the
    # whole hook down. A crashed hook fails open, so this is not a cosmetic
    # detail: it is the difference between guarding and only appearing to.
    Get-Segments $Command | Where-Object { (Get-GitSubcommand $_) -eq $Name }
}

function Test-GitSubcommand([string]$Command, [string]$Name) {
    return @(Get-SubcommandSegments $Command $Name).Count -gt 0
}

function Targets-Main([string]$Command) {
    # True if any argument names main as a push destination, in any spelling:
    # `main`, `refs/heads/main`, `HEAD:main`, `HEAD:refs/heads/main`, `+main`.
    #
    # Compared as a whole ref, not a substring, so `fix/domain-main` and
    # `feature/main` are left alone -- only the branch actually called main.
    #
    # Give this the push segments alone, never the whole command. Every word of
    # a commit message is otherwise a candidate ref.
    foreach ($token in Get-Args $Command) {
        $ref = $token
        if ($ref.Contains(':')) { $ref = $ref.Substring($ref.LastIndexOf(':') + 1) }
        $ref = $ref -replace '^refs/heads/', ''
        if ($ref -eq 'main') { return $true }
    }
    return $false
}

function Is-TagPush([string]$Command) {
    # Ask git, rather than guessing from the name. A pattern like `v\d` also
    # matches branches called v2 or v10-experiment, which would hand out a
    # bypass to anything named like a version.
    if ($Command -match '--tags\b') { return $true }
    foreach ($token in Get-Args $Command) {
        $name = $token -replace '^refs/tags/', ''
        if ([string]::IsNullOrWhiteSpace($name)) { continue }
        & git show-ref --verify --quiet "refs/tags/$name" 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    }
    return $false
}

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

# Commit messages are prose, not command structure. Everything below reads
# structure, so the prose goes first.
$command = Remove-HereStrings $command

$isCommit = Test-GitSubcommand $command 'commit'
$pushSegments = @(Get-SubcommandSegments $command 'push') -join ' '
$isPush = -not [string]::IsNullOrWhiteSpace($pushSegments)
if (-not ($isCommit -or $isPush)) { Allow }

# Changes nothing, so there is nothing to guard.
if ($command -match '--dry-run') { Allow }

# A push naming main advances main whatever branch it runs from, so the
# current-branch check below would miss it. Tested before the tag carve-out, so
# `git push origin v1.0 main` cannot ride in on the tag.
if ($isPush -and (Targets-Main $pushSegments)) {
    Deny(@"
Pushing to 'main' is blocked. main only advances through a reviewed PR.

Push your branch instead, then open a draft PR and wait for review:
    git push -u origin <type>/<short-slug>

See CLAUDE.md.
"@)
}

# Tag pushes are how a release is cut and never advance a branch. If git cannot
# confirm the name is a tag the push is simply not exempted, and falls through to
# the branch check below -- so an unconfirmable tag is refused on main rather
# than waved past.
if ($isPush -and -not $isCommit -and (Is-TagPush $pushSegments)) { Allow }

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
$verb on 'main' is blocked. main only advances through a reviewed PR.

Create a branch first, then commit there:
    git checkout -b <type>/<short-slug>     # fix/ feat/ docs/ chore/

Then push it and open a draft PR, and wait for review before going further.
See CLAUDE.md. (Tag pushes and --dry-run are allowed on main.)
"@)
