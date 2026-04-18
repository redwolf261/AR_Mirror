param(
    [switch]$SkipPythonSmoke,
    [switch]$SkipFrontendBuild,
    [switch]$SkipBackendBuild
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$results = @()
$hasFailures = $false

function Add-Result {
    param(
        [string]$Check,
        [string]$Status,
        [string]$Details
    )
    $script:results += [PSCustomObject]@{
        Check = $Check
        Status = $Status
        Details = $Details
    }
}

function Invoke-Check {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host "`n==> $Name" -ForegroundColor Cyan
    try {
        & $Action
        Add-Result -Check $Name -Status 'PASS' -Details 'Completed'
        Write-Host "PASS: $Name" -ForegroundColor Green
    }
    catch {
        $script:hasFailures = $true
        Add-Result -Check $Name -Status 'FAIL' -Details $_.Exception.Message
        Write-Host "FAIL: $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor DarkRed
    }
}

function Get-PythonExe {
    $candidates = @(
        (Join-Path $root '.venv\Scripts\python.exe'),
        (Join-Path $root 'ar\Scripts\python.exe')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw 'No Python virtual environment found. Expected .venv or ar.'
}

Invoke-Check -Name 'Workspace sanity' -Action {
    if (-not (Test-Path (Join-Path $root 'app.py'))) {
        throw 'app.py not found at workspace root.'
    }
    if (-not (Test-Path (Join-Path $root 'web-ui\package.json'))) {
        throw 'web-ui/package.json not found.'
    }
}

if (-not $SkipPythonSmoke) {
    Invoke-Check -Name 'Python dependency import smoke' -Action {
        $py = Get-PythonExe
        & $py -c "import flask, flask_cors, cv2, numpy, PIL; print('python-import-ok')"
        if ($LASTEXITCODE -ne 0) {
            throw 'Python dependency import smoke failed.'
        }
    }

    Invoke-Check -Name 'Backend API health smoke (app_cloud.py)' -Action {
        $py = Get-PythonExe
        $proc = Start-Process -FilePath $py -ArgumentList 'app_cloud.py' -WorkingDirectory $root -PassThru
        try {
            $ready = $false
            for ($i = 0; $i -lt 25; $i++) {
                Start-Sleep -Seconds 1
                try {
                    $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:5050/' -UseBasicParsing -TimeoutSec 2
                    if ($resp.StatusCode -eq 200) {
                        $ready = $true
                        break
                    }
                }
                catch {
                    # keep retrying
                }
            }

            if (-not $ready) {
                throw 'Backend API did not become healthy on http://127.0.0.1:5050/ within timeout.'
            }
        }
        finally {
            if ($null -ne $proc -and -not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
else {
    Add-Result -Check 'Python dependency import smoke' -Status 'SKIP' -Details 'Skipped by flag'
    Add-Result -Check 'Backend API health smoke (app_cloud.py)' -Status 'SKIP' -Details 'Skipped by flag'
}

if (-not $SkipFrontendBuild) {
    Invoke-Check -Name 'Frontend build (web-ui)' -Action {
        Push-Location (Join-Path $root 'web-ui')
        try {
            npm run build
            if ($LASTEXITCODE -ne 0) {
                throw 'Frontend build failed.'
            }
        }
        finally {
            Pop-Location
        }
    }
}
else {
    Add-Result -Check 'Frontend build (web-ui)' -Status 'SKIP' -Details 'Skipped by flag'
}

if (-not $SkipBackendBuild) {
    Invoke-Check -Name 'Backend build (NestJS)' -Action {
        Push-Location (Join-Path $root 'backend')
        try {
            npm run build
            if ($LASTEXITCODE -ne 0) {
                throw 'Backend build failed.'
            }
        }
        finally {
            Pop-Location
        }
    }
}
else {
    Add-Result -Check 'Backend build (NestJS)' -Status 'SKIP' -Details 'Skipped by flag'
}

Write-Host "`n=== Stabilization Summary ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize

$failed = ($results | Where-Object { $_.Status -eq 'FAIL' }).Count
if ($hasFailures -or $failed -gt 0) {
    Write-Host "`nStabilization gate FAILED ($failed checks failed)." -ForegroundColor Red
    exit 1
}

Write-Host "`nStabilization gate PASSED." -ForegroundColor Green
exit 0
