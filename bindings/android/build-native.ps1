<#
.SYNOPSIS
    Cross-compiles crates/jni (libpiedpiper_jni.so) for Android with the NDK's clang and copies
    the libraries into piedpiper/src/main/jniLibs/<abi>/ (Windows PowerShell counterpart of
    build-native.sh).

.EXAMPLE
    .\build-native.ps1                      # arm64-v8a, armeabi-v7a, x86_64
    .\build-native.ps1 -Abi x86_64          # only the listed ABIs

.NOTES
    NDK: $env:ANDROID_NDK_HOME / $env:ANDROID_NDK_ROOT, else the newest one under the SDK
    ($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT, local.properties sdk.dir, %LOCALAPPDATA%\Android\Sdk).
    Cargo output goes to $env:CARGO_TARGET_DIR (default <repo>\target\android, separate from the
    desktop builds so concurrent cargo runs do not contend for a lock).
#>
param(
    [ValidateSet('arm64-v8a', 'armeabi-v7a', 'x86_64')]
    [string[]] $Abi = @('arm64-v8a', 'armeabi-v7a', 'x86_64'),
    # Minimum API level baked into the linker target; keep it equal to minSdk.
    [int] $Api = 24,
    # Keep the ELF symbol table (bigger .so, better native stack traces).
    [switch] $KeepSymbols
)
$ErrorActionPreference = 'Stop'

# Runs a native tool, passing its output (stdout and stderr) through as text and failing on a
# non-zero exit code. Windows PowerShell turns redirected native stderr into error records, which
# 'Stop' would treat as fatal even for cargo's normal progress output.
function Invoke-Native([string] $Exe, [string[]] $Arguments) {
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

$Here = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Here '..\..')).Path
$JniLibs = Join-Path $Here 'piedpiper\src\main\jniLibs'

function Find-Sdk {
    foreach ($v in @($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT)) {
        if ($v) { return $v }
    }
    $props = Join-Path $Here 'local.properties'
    if (Test-Path $props) {
        $line = Get-Content $props | Where-Object { $_ -match '^sdk\.dir=' } | Select-Object -Last 1
        if ($line) { return ($line -replace '^sdk\.dir=', '' -replace '\\:', ':' -replace '\\\\', '\') }
    }
    return (Join-Path $env:LOCALAPPDATA 'Android\Sdk')
}

$Ndk = $env:ANDROID_NDK_HOME
if (-not $Ndk) { $Ndk = $env:ANDROID_NDK_ROOT }
if (-not $Ndk) {
    $ndkRoot = Join-Path (Find-Sdk) 'ndk'
    $newest = Get-ChildItem $ndkRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } } |
        Select-Object -Last 1
    if ($newest) { $Ndk = $newest.FullName }
}
$Bin = if ($Ndk) { Join-Path $Ndk 'toolchains\llvm\prebuilt\windows-x86_64\bin' } else { '' }
if (-not $Bin -or -not (Test-Path $Bin)) {
    throw "Android NDK not found (looked for '$Bin'); set ANDROID_NDK_HOME."
}

# ABI -> rust target, clang target prefix
$Specs = @{
    'arm64-v8a'   = @('aarch64-linux-android', 'aarch64-linux-android')
    'armeabi-v7a' = @('armv7-linux-androideabi', 'armv7a-linux-androideabi')
    'x86_64'      = @('x86_64-linux-android', 'x86_64-linux-android')
}

if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $Root 'target\android' }
if (-not $KeepSymbols) { $env:CARGO_PROFILE_RELEASE_STRIP = 'symbols' }
$Ar = Join-Path $Bin 'llvm-ar.exe'
$ReadElf = Join-Path $Bin 'llvm-readelf.exe'
$Installed = @(Invoke-Native 'rustup' @('target', 'list', '--installed'))

Write-Host "NDK:        $Ndk"
Write-Host "API level:  $Api"
Write-Host "target dir: $env:CARGO_TARGET_DIR"

foreach ($a in $Abi) {
    $target = $Specs[$a][0]
    $linker = Join-Path $Bin ($Specs[$a][1] + "$Api-clang.cmd")
    if (-not (Test-Path $linker)) { throw "missing linker $linker (does this NDK support API $Api?)" }
    if ($Installed -notcontains $target) {
        Write-Host "==> rustup target add $target"
        Invoke-Native 'rustup' @('target', 'add', $target) | Write-Host
    }
    $upper = $target.ToUpper().Replace('-', '_')
    $lower = $target.Replace('-', '_')
    Set-Item "env:CARGO_TARGET_${upper}_LINKER" $linker
    Set-Item "env:CARGO_TARGET_${upper}_AR" $Ar
    Set-Item "env:CC_$lower" $linker
    Set-Item "env:AR_$lower" $Ar

    Write-Host "==> $a ($target)"
    Invoke-Native 'cargo' @('build', '--release', '--manifest-path', (Join-Path $Root 'crates\jni\Cargo.toml'), '--target', $target) |
        Write-Host

    $dst = Join-Path $JniLibs $a
    New-Item -ItemType Directory -Force $dst | Out-Null
    $so = Join-Path $dst 'libpiedpiper_jni.so'
    Copy-Item (Join-Path $env:CARGO_TARGET_DIR "$target\release\libpiedpiper_jni.so") $so -Force

    # Android 15+ devices may use 16 KB pages: every LOAD segment must be 16 KB aligned.
    if ($a -ne 'armeabi-v7a' -and (Test-Path $ReadElf)) {
        & $ReadElf -lW $so | Where-Object { $_ -match '^\s*LOAD\s' } | ForEach-Object {
            $align = ($_.Trim() -split '\s+')[-1]
            if ([Convert]::ToInt64($align, 16) -lt 16384) { Write-Warning "$a LOAD segment alignment $align < 16 KB" }
        }
    }
    Write-Host "    -> $so ($((Get-Item $so).Length) bytes)"
}
