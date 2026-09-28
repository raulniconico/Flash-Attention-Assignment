<#
.SYNOPSIS
    Create .venv and install requirements.txt for this project (Windows).

.DESCRIPTION
    Finds Python >= 3.10 (installs 3.12 via winget if none is found), creates
    .venv in the project root, and installs requirements.txt into it.
    .venv matches the interpreter configured in .idea/misc.xml.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Recreate
#>
param(
    [switch]$Recreate   # delete an existing .venv first
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$MinMinor = 10

function Test-Python([string[]]$Cmd) {
    # Returns $true if $Cmd runs a real Python >= 3.$MinMinor (skips the Microsoft Store stub).
    try {
        $exe = $Cmd[0]
        $cmdArgs = @($Cmd | Select-Object -Skip 1)
        $out = & $exe @cmdArgs -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $major, $minor = "$out".Trim().Split(' ') | ForEach-Object { [int]$_ }
        return ($major -eq 3 -and $minor -ge $MinMinor)
    } catch {
        return $false
    }
}

function Find-Python {
    $candidates = @(
        @('py', '-3.12'), @('py', '-3.13'), @('py', '-3.14'), @('py', '-3.11'), @('py', '-3.10'), @('py', '-3'),
        @('python3'), @('python')
    )
    # Default per-user install locations (PATH is not refreshed right after winget installs).
    foreach ($v in '312', '313', '314', '311', '310') {
        $candidates += , @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"))
    }
    foreach ($c in $candidates) {
        if (Test-Python $c) { return , $c }
    }
    return $null
}

Write-Host "==> Looking for Python >= 3.$MinMinor"
$Python = Find-Python
if (-not $Python) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "==> No suitable Python found; installing Python 3.12 with winget (user scope)"
        winget install --id Python.Python.3.12 --scope user --exact --silent `
            --accept-package-agreements --accept-source-agreements
        $Python = Find-Python
    }
    if (-not $Python) {
        Write-Error ("Python >= 3.$MinMinor not found. Install it from https://www.python.org/downloads/ " +
                     "(tick 'Add python.exe to PATH'), then re-run this script.")
    }
}
$PyExe = $Python[0]
$PyArgs = @($Python | Select-Object -Skip 1)
Write-Host "    using: $($Python -join ' ') ($(& $PyExe @PyArgs --version))"

if ($Recreate -and (Test-Path $Venv)) {
    Write-Host "==> Removing existing .venv"
    Remove-Item -Recurse -Force $Venv
}

$VenvPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    Write-Host "==> Creating virtual environment in .venv"
    & $PyExe @PyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed" }
}

Write-Host "==> Installing requirements"
& $VenvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed" }
& $VenvPy -m pip install -r (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Write-Error "requirements install failed" }

Write-Host "==> Verifying imports"
& $VenvPy -c "import numpy, matplotlib, PIL, IPython, ipykernel; print('    ok: numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Error "import check failed" }

Write-Host ""
Write-Host "Done. Activate with:   .\.venv\Scripts\Activate.ps1"
Write-Host "Run notebooks with:    .\.venv\Scripts\jupyter notebook"
Write-Host "In PyCharm, the project interpreter is already set to .venv."
