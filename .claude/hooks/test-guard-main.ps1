<#
.SYNOPSIS
    Cases for guard-main.ps1. Run from anywhere:

        pwsh -NoProfile -File .claude/hooks/test-guard-main.ps1

.DESCRIPTION
    The guard is the only thing standing between a stray `git commit` and an
    unreviewed change on the default branch, and it is easy to break in a way
    that looks fine: a hook that crashes emits nothing, and emitting nothing
    means "allow". So a broken guard and a permissive one are indistinguishable
    from the outside, and both of the bugs these cases were written after had
    exactly that shape.

    Drives the real hook the way Claude Code does -- a PreToolUse payload on
    stdin, a permissionDecision on stdout -- rather than testing its functions
    in isolation, because the wiring is where it went wrong.

    Exits non-zero with a count of misbehaving cases.
#>

param(
    [string]$Hook = (Join-Path $PSScriptRoot 'guard-main.ps1')
)

Set-StrictMode -Version Latest

$script:failures = 0

function Invoke-Hook([string]$Command) {
    <#
        Returns 'allow', 'deny', or a description of why neither.

        The hook exits 0 whatever it decides, and says "deny" by printing JSON.
        So the two failure shapes are a nonzero exit, and output that is neither
        empty nor a decision -- both of which otherwise read as "allow", which
        is the direction that hides a broken guard.

        Checked structurally rather than by matching error wording: a guess at
        the wording misses whatever phrasing was not anticipated, and
        `Cannot find path ...` is one that would have slipped through.
    #>
    $payload = @{ tool_input = @{ command = $Command } } | ConvertTo-Json -Compress
    $out = $payload | & pwsh -NoProfile -File $Hook 2>&1 | Out-String
    $code = $LASTEXITCODE

    if ($code -ne 0) { return "hook exited $code : $(($out -split "`n")[0].Trim())" }
    if ([string]::IsNullOrWhiteSpace($out)) { return 'allow' }
    try {
        $decision = ($out | ConvertFrom-Json).hookSpecificOutput.permissionDecision
    } catch {
        return "unparseable output: $(($out -split "`n")[0].Trim())"
    }
    if ($decision -eq 'deny') { return 'deny' }
    if ($decision -eq 'allow') { return 'allow' }
    return "unexpected decision '$decision'"
}

function Check([string]$Name, [string]$Command, [bool]$ShouldDeny, [string]$In) {
    $expected = if ($ShouldDeny) { 'deny' } else { 'allow' }
    if ($In) { Push-Location $In }
    try {
        $actual = Invoke-Hook $Command
    } finally {
        if ($In) { Pop-Location }
    }
    $ok = ($actual -eq $expected)
    if (-not $ok) { $script:failures++ }
    "{0} {1,-52} expected={2,-5} got={3}" -f `
        $(if ($ok) { '  ok  ' } else { ' FAIL ' }), $Name, $expected, $actual
}

function New-RepoOnBranch([string]$Branch) {
    <#
        A throwaway repository sitting on a given branch.

        The interesting half of this guard only acts when HEAD is the default
        branch, and the workflow it enforces means the suite is always run from
        a feature branch -- where every such case is allowed for the wrong
        reason and proves nothing. `git init -b` gives a real one to test
        against without touching the checkout.

        No commit is needed: symbolic-ref reports the branch of an empty repo,
        which is the case the hook was written to handle.
    #>
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("guardtest-" + [guid]::NewGuid().ToString('N'))
    & git init --quiet -b $Branch $dir 2>&1 | Out-Null
    return $dir
}

$branch = (& git symbolic-ref --short --quiet HEAD 2>$null)
"=== guard-main.ps1, from branch '$branch' ==="
""

# --- refused from any branch, because they advance main ---------------------
Check 'push origin main' 'git push origin main' $true
Check 'push origin HEAD:main' 'git push origin HEAD:main' $true
Check 'push origin HEAD:refs/heads/main' 'git push origin HEAD:refs/heads/main' $true
Check 'force-push +main' 'git push origin +main' $true
Check 'quoted ref: push origin "main"' 'git push origin "main"' $true
Check 'a tag cannot smuggle main alongside it' 'git push origin v0.3.1 main' $true

# --- allowed -----------------------------------------------------------------
Check 'push a feature branch' 'git push -u origin feat/thing' $false
Check 'a branch merely named like main' 'git push origin fix/domain-main' $false
Check 'dry run changes nothing' 'git push --dry-run origin main' $false
Check 'read-only command mentioning main' 'git log --grep=main' $false
Check 'commit-graph is not commit' 'git commit-graph write' $false

# --- a commit message is prose, not command structure ------------------------
# Assembled from arrays: a here-string containing a here-string terminates the
# outer one, which is the same class of confusion the hook itself had.
$marker = '@' + "'"
$endMarker = "'" + '@'

Check 'message mentioning main, while pushing a branch' (@(
    'git add -A'
    "git commit -q -m $marker"
    'Fix the thing'
    ''
    'Now on the main thread, since the import blocks.'
    $endMarker
    'git push'
) -join "`n") $false

Check 'message quoting a push-to-main instruction' (@(
    "git commit -m $marker"
    'Docs: describe releasing'
    ''
    '    git push origin main'
    ''
    'is how a release used to be cut.'
    $endMarker
) -join "`n") $false

# --- quote concatenation, which the shell resolves before git sees it --------
Check 'push origin HEAD:"main"' 'git push origin HEAD:"main"' $true
Check 'push origin m"ai"n' 'git push origin m"ai"n' $true
Check "push origin 'main'" "git push origin 'main'" $true

# --- a continued line is one command, not two --------------------------------
# Splitting on every newline put the destination in a segment of its own, where
# nothing was looking for it. Both shells' continuation characters, since either
# may be what reaches the hook.
$backslash = [char]92
$backtick = [char]96

Check 'push continued with a backslash (bash)' (@(
    "git push origin $backslash"
    'main'
) -join "`n") $true

Check 'push continued with a backtick (PowerShell)' (@(
    "git push origin $backtick"
    'main'
) -join "`n") $true

Check 'continuation with trailing space before the newline' (@(
    "git push origin $backslash  "
    'main'
) -join "`n") $true

# The join must not swallow a real boundary: a backslash inside a token is not
# a continuation unless the newline follows it directly.
Check 'a path containing a backslash is not a continuation' (@(
    'git commit -m "see C:\ws\notes.md"'
    'git push -u origin feat/thing'
) -join "`n") $false

# --- the branch-sensitive half, tested against a repo that is on main --------
# Run from this checkout these are allowed for the wrong reason: the workflow
# keeps HEAD on a feature branch, so every case below would pass no matter what
# the hook did. Throwaway repositories give both branches.
$mainRepo = New-RepoOnBranch 'main'
$featureRepo = New-RepoOnBranch 'feat/thing'
try {
    $addThenCommit = @('git add -A', 'git commit -m "wip"') -join "`n"

    # The headline regression: `git add` on the line above used to hide the
    # commit entirely, because segments did not split on newlines.
    Check 'add then commit, on main' $addThenCommit $true -In $mainRepo
    Check 'add then commit, on a feature branch' $addThenCommit $false -In $featureRepo

    Check 'plain commit on main' 'git commit -m "wip"' $true -In $mainRepo
    Check 'plain commit on a feature branch' 'git commit -m "wip"' $false -In $featureRepo

    # A commit message mentioning the branch is still just prose, on main too.
    Check 'message mentioning main, committing on a feature branch' (@(
        "git commit -m $marker"
        'Now on the main thread, since the import blocks.'
        $endMarker
    ) -join "`n") $false -In $featureRepo

    Check 'push on main with no refspec' 'git push' $true -In $mainRepo
    Check 'dry run on main' 'git push --dry-run' $false -In $mainRepo
} finally {
    Remove-Item -Recurse -Force $mainRepo, $featureRepo -ErrorAction SilentlyContinue
}

""
if ($script:failures -eq 0) {
    'all cases behaved'
    exit 0
}
"$($script:failures) case(s) wrong"
exit $script:failures
