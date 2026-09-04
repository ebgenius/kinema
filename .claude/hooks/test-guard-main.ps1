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

function Check([string]$Name, [string]$Command, [bool]$ShouldDeny) {
    $payload = @{ tool_input = @{ command = $Command } } | ConvertTo-Json -Compress
    $out = $payload | & pwsh -NoProfile -File $Hook 2>&1 | Out-String

    # A crashed hook emits no decision, which reads as "allow" -- the dangerous
    # direction, and how a broken guard passes for a working one. Caught here
    # rather than counted as a pass.
    if ($out -match 'Exception|ParserError|is not recognized|cannot be found') {
        $script:failures++
        $first = ($out -split "`n" | Select-Object -First 1).Trim()
        return "{0} {1,-52} HOOK ERRORED: {2}" -f ' FAIL ', $Name, $first
    }

    $denied = $out -match '"permissionDecision"\s*:\s*"deny"'
    $ok = ($denied -eq $ShouldDeny)
    if (-not $ok) { $script:failures++ }
    "{0} {1,-52} expected={2,-5} got={3}" -f `
        $(if ($ok) { '  ok  ' } else { ' FAIL ' }), $Name,
        $(if ($ShouldDeny) { 'deny' } else { 'allow' }),
        $(if ($denied) { 'deny' } else { 'allow' })
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

# --- statements on separate lines are separate commands ----------------------
# `git add` first used to hide the `git commit` behind it. Only blocked on main,
# so the expectation follows the branch rather than being quietly vacuous.
Check 'add then commit, on separate lines' (@(
    'git add -A'
    'git commit -m "wip"'
) -join "`n") ($branch -eq 'main')

""
if ($script:failures -eq 0) {
    'all cases behaved'
    exit 0
}
"$($script:failures) case(s) wrong"
exit $script:failures
