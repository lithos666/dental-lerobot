# Apply all upstream-compatibility patches in this directory.
# Run from the repository root:  powershell -File patches\apply_patches.ps1
# Add -CheckOnly to dry-run without modifying files.

param([switch]$CheckOnly)

$patches = (Get-ChildItem -Path $PSScriptRoot -Filter "0*.patch" | Sort-Object Name).FullName
if (-not $patches) {
    Write-Error "No patches found under $PSScriptRoot"
    exit 1
}

$flag = if ($CheckOnly) { "--check" } else { $null }
git apply $flag --verbose $patches
if ($LASTEXITCODE -ne 0) {
    Write-Error "git apply failed ($LASTEXITCODE). Resolve conflicts and retry (see patches\README.md)."
    exit $LASTEXITCODE
}
if ($CheckOnly) {
    Write-Host "All $($patches.Count) patches can be applied cleanly."
} else {
    Write-Host "Applied $($patches.Count) patches."
}
