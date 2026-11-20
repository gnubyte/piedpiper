<#
.SYNOPSIS
    Builds PiedPiper's Windows installer: dist\PiedPiper-<version>-x64.msi.

.DESCRIPTION
    Compiles piedpiper-archiver.exe (release), optionally signs it, and packs it with WiX
    Toolset v5 (PiedPiper.wxs). Needs Rust and WiX 5:

        dotnet tool install --global wix --version 5.0.2
        wix extension add -g WixToolset.UI.wixext/5.0.2

    WiX v5 is pinned on purpose: v6 and later need the Open Source Maintenance Fee EULA
    accepted first.

    Code signing is optional. With WINDOWS_CERTIFICATE_PFX (the .pfx file, base64) and
    WINDOWS_CERTIFICATE_PASSWORD set, the program and the MSI are signed with signtool;
    without them both stay unsigned and SmartScreen warns on first run.

.EXAMPLE
    pwsh apps/desktop/packaging/windows/build-msi.ps1 -Version 0.1.0
#>
param(
    [string]$Version = "",
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$OutDir = "dist",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path      # apps\desktop\packaging\windows
$app = (Resolve-Path (Join-Path $here "..\..")).Path         # apps\desktop
$repo = (Resolve-Path (Join-Path $app "..\..")).Path         # repository root

if (-not $Version) {
    $m = Select-String -Path (Join-Path $app "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    $Version = $m.Matches[0].Groups[1].Value
}
# MSI product versions are numeric (major.minor.build): drop a leading v and any -rc / +meta tail.
$msiVersion = ($Version -replace '^v', '') -replace '[-+].*$', ''
if ($msiVersion -notmatch '^\d+\.\d+\.\d+$') { throw "Version must look like 1.2.3, got '$Version'" }

$arch = if ($Target -like "aarch64*") { "arm64" } else { "x64" }
$work = Join-Path $app "target\msi"
New-Item -ItemType Directory -Force $work | Out-Null

function Invoke-Sign([string]$File) {
    if (-not $env:WINDOWS_CERTIFICATE_PFX) { return }
    $pfx = Join-Path $work "signing.pfx"
    [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:WINDOWS_CERTIFICATE_PFX))
    try {
        $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" |
            Sort-Object FullName -Descending | Select-Object -First 1
        if (-not $signtool) { throw "signtool.exe not found (Windows SDK)" }
        & $signtool.FullName sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
            /f $pfx /p $env:WINDOWS_CERTIFICATE_PASSWORD /d "PiedPiper" $File
        if ($LASTEXITCODE) { throw "signing $File failed" }
    } finally {
        Remove-Item -Force $pfx -ErrorAction SilentlyContinue
    }
}

if (-not $SkipBuild) {
    cargo build --release --locked --manifest-path (Join-Path $app "Cargo.toml") --target $Target
    if ($LASTEXITCODE) { throw "cargo build failed" }
}
$exe = Join-Path $app "target\$Target\release\piedpiper-archiver.exe"
if (-not (Test-Path $exe)) { throw "missing $exe" }
Invoke-Sign $exe

# The installer's license page: the MIT license as RTF.
$license = (Get-Content -Raw (Join-Path $repo "LICENSE")).Trim()
$escaped = $license.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}') -replace "`r?`n", "\par`n"
$rtf = "{\rtf1\ansi\deff0{\fonttbl{\f0\fswiss Segoe UI;}}\f0\fs18 $escaped\par}"
$licenseRtf = Join-Path $work "License.rtf"
Set-Content -Path $licenseRtf -Value $rtf -Encoding ascii

New-Item -ItemType Directory -Force $OutDir | Out-Null
$msi = Join-Path (Resolve-Path $OutDir).Path "PiedPiper-$msiVersion-$arch.msi"
wix build (Join-Path $here "PiedPiper.wxs") -arch $arch -ext WixToolset.UI.wixext `
    -d "Version=$msiVersion" -d "ExePath=$exe" -d "RepoRoot=$repo" -d "AppDir=$app" `
    -d "ResourceDir=$(Join-Path $app 'resources\windows')" -d "LicenseRtf=$licenseRtf" `
    -o $msi
if ($LASTEXITCODE) { throw "wix build failed" }
Invoke-Sign $msi
Write-Host "Built $msi"
