<#
.SYNOPSIS
    Installs the MSI, checks what it registered, runs the Explorer commands end to end, and
    uninstalls again. Meant for CI (the runner is an administrator); it changes the machine.

.DESCRIPTION
    -Stage Install    install silently, check files and registry, run the installed program
                      from the command line
    -Stage Explorer   run "Compress" on 21 selected items, then "Extract to folder" and the
                      double-click "Extract", through the same shell code Explorer uses
                      (examples/shell_invoke.exe); COM starts the installed program
    -Stage Uninstall  uninstall silently and check that the registration is gone

.EXAMPLE
    pwsh smoke-test.ps1 -Stage Install -Msi dist\PiedPiper-0.1.0-x64.msi
#>
param(
    [Parameter(Mandatory)][ValidateSet("Install", "Explorer", "Uninstall")][string]$Stage,
    [string]$Msi = "",
    [string]$Harness = ""
)
$ErrorActionPreference = "Stop"

$classes = "HKLM:\Software\Classes"
$exe = Join-Path $env:ProgramFiles "PiedPiper\piedpiper-archiver.exe"
$work = Join-Path ([IO.Path]::GetTempPath()) "piedpiper-smoke"
$keys = @(
    "$classes\CLSID\{27A4092B-ED3D-4569-9C13-DF2ED29481D9}\LocalServer32",
    "$classes\CLSID\{58FBBC3B-004A-4BD4-9932-577D0D2CF1C7}\LocalServer32",
    "$classes\CLSID\{3298D9A7-1F73-4CB7-88D9-BD7515A43736}\LocalServer32",
    "$classes\AllFilesystemObjects\shell\PiedPiper.Compress\DropTarget",
    "$classes\SystemFileAssociations\.pp\shell\PiedPiper.ExtractHere\DropTarget",
    "$classes\SystemFileAssociations\.pp\shell\PiedPiper.ExtractTo\DropTarget",
    "$classes\PiedPiper.Archive\shell\open\command"
)

function Invoke-Msiexec([string[]]$ArgList, [string]$Log) {
    $p = Start-Process msiexec.exe -ArgumentList ($ArgList + @("/qn", "/norestart", "/l*v", "`"$Log`"")) -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        Get-Content $Log -Tail 60
        throw "msiexec $($ArgList -join ' ') failed with exit code $($p.ExitCode)"
    }
}

function Wait-Until([scriptblock]$Condition, [string]$What, [int]$Seconds = 180) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while (-not (& $Condition)) {
        if ((Get-Date) -gt $deadline) { throw "timed out waiting for $What" }
        Start-Sleep -Milliseconds 500
    }
}

function Wait-Idle {
    Wait-Until { -not (Get-Process piedpiper-archiver -ErrorAction SilentlyContinue) } "piedpiper-archiver to exit"
}

function Show-Log {
    $log = Join-Path $env:APPDATA "PiedPiper\archiver.log"
    if (Test-Path $log) { Write-Host "--- archiver.log"; Get-Content $log }
}

switch ($Stage) {
    "Install" {
        Invoke-Msiexec @("/i", "`"$Msi`"") (Join-Path $env:TEMP "piedpiper-install.log")
        if (-not (Test-Path $exe)) { throw "not installed: $exe" }
        foreach ($k in $keys) { if (-not (Test-Path $k)) { throw "missing registry key $k" } }
        $server = (Get-ItemProperty "$classes\CLSID\{27A4092B-ED3D-4569-9C13-DF2ED29481D9}\LocalServer32").'(default)'
        if ($server -ne "`"$exe`" --com compress") { throw "unexpected LocalServer32: $server" }
        foreach ($f in "LICENSE.txt", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.md", "LICENSE-lepton_jpeg.txt", "NOTICE-lepton_jpeg.txt") {
            if (-not (Test-Path (Join-Path $env:ProgramFiles "PiedPiper\$f"))) { throw "missing $f" }
        }

        # The installed program, from the command line.
        Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force "$work\cli\docs" | Out-Null
        Set-Content "$work\cli\docs\a.txt" (("alpha " * 50 + "`n") * 40)
        Set-Content "$work\cli\docs\b.txt" (("beta " * 50 + "`n") * 40)
        $out = & $exe --cli --level 1 compress "$work\cli\docs" | Out-String
        if ($LASTEXITCODE) { throw "compress failed: $out" }
        if (-not (Test-Path "$work\cli\docs.tar.pp")) { throw "no docs.tar.pp ($out)" }
        & $exe --cli extract-to "$work\cli\docs.tar.pp" | Out-Null
        if ($LASTEXITCODE) { throw "extract failed" }
        if ((Get-FileHash "$work\cli\docs\a.txt").Hash -ne (Get-FileHash "$work\cli\docs (2)\docs\a.txt").Hash) {
            throw "round trip changed a.txt"
        }
        Write-Host "Installed and working from the command line."
    }

    "Explorer" {
        if (-not (Test-Path $Harness)) { throw "missing harness $Harness" }
        # No windows: nobody can click a dialog away on a CI runner.
        New-Item -ItemType Directory -Force (Join-Path $env:APPDATA "PiedPiper") | Out-Null
        Set-Content (Join-Path $env:APPDATA "PiedPiper\archiver.ini") "level = 1`nlog = 1`nui = none`n"
        try {
            $sel = Join-Path $work "explorer"
            Remove-Item -Recurse -Force $sel -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force "$sel\sub" | Out-Null
            1..20 | ForEach-Object { Set-Content "$sel\file$_.txt" ("line $_ " * 100) }
            Set-Content "$sel\sub\inner.txt" "inner"

            # 21 items: more than the 15 Explorer allows for verbs without MultiSelectModel=Player.
            $items = (Get-ChildItem $sel).FullName
            & $Harness PiedPiper.Compress @items
            if ($LASTEXITCODE) { throw "Compress verb failed" }
            Wait-Until { Test-Path "$sel\Archive.tar.pp" } "Archive.tar.pp"
            Wait-Idle

            & $Harness PiedPiper.ExtractTo "$sel\Archive.tar.pp"
            if ($LASTEXITCODE) { throw "Extract-to-folder verb failed" }
            Wait-Until { Test-Path "$sel\Archive\sub\inner.txt" } "the extracted folder"
            Wait-Idle
            if ((Get-ChildItem "$sel\Archive").Count -ne 21) { throw "expected 21 items in the extracted folder" }
            if ((Get-FileHash "$sel\file7.txt").Hash -ne (Get-FileHash "$sel\Archive\file7.txt").Hash) { throw "round trip changed file7.txt" }

            # Double-click: the default "open" verb is a plain command line. It is run as
            # registered rather than through the shell: with another program listed for .pp
            # (VS Code on hosted runners), Windows first asks which program to use.
            $open = (Get-ItemProperty "$classes\PiedPiper.Archive\shell\open\command").'(default)'
            if ($open -ne "`"$exe`" extract `"%1`"") { throw "unexpected open command: $open" }
            Move-Item "$sel\Archive.tar.pp" "$work\Moved.tar.pp"
            & $exe extract "$work\Moved.tar.pp" | Out-Null
            Wait-Until { Test-Path "$work\Moved\sub\inner.txt" } "the double-click extraction"
            Wait-Idle
            Write-Host "Explorer commands work end to end."
        } catch {
            Show-Log
            throw
        } finally {
            Remove-Item (Join-Path $env:APPDATA "PiedPiper\archiver.ini") -ErrorAction SilentlyContinue
        }
        Show-Log
    }

    "Uninstall" {
        Invoke-Msiexec @("/x", "`"$Msi`"") (Join-Path $env:TEMP "piedpiper-uninstall.log")
        if (Test-Path $exe) { throw "still installed: $exe" }
        foreach ($k in $keys) { if (Test-Path $k) { throw "left behind: $k" } }
        Write-Host "Uninstalled cleanly."
    }
}
