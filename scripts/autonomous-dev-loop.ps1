param(
    [int]$BackendPhase = 0,
    [int]$Duration = 0,
    [double]$MonitorIntervalSec = 1.5,
    [string]$OutDir = "logs/live-monitor",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

function Test-Http {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 2
    )

    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        return $true
    }
    catch {
        return $false
    }
}

function Wait-Http {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 60,
        [string]$Name = "service"
    )

    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
        if (Test-Http -Url $Url) {
            Write-Host "[OK] $Name is live at $Url"
            return $true
        }
        Start-Sleep -Milliseconds 800
    }

    Write-Warning "Timed out waiting for $Name at $Url"
    return $false
}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

$frontendDir = Join-Path $root "frontend\ar-mirror-ui"
$monitorScript = Join-Path $root "scripts\live_snapshot_monitor.py"

if (-not (Test-Path $monitorScript)) {
    throw "Monitor script missing: $monitorScript"
}

New-Item -ItemType Directory -Path (Join-Path $root $OutDir) -Force | Out-Null

$backendProc = $null
$frontendProc = $null

try {
    if (-not (Test-Http -Url "http://localhost:5051/api/snapshot")) {
        Write-Host "[start] Launching backend..."
        $backendArgs = "`"$python`" app.py --phase $BackendPhase --duration $Duration"
        if ($BackendPhase -eq 0) {
            $backendArgs = "$backendArgs --headless"
        }
        $backendProc = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; $backendArgs" -PassThru
    }
    else {
        Write-Host "[skip] Backend already live on :5051"
    }

    if (-not (Test-Http -Url "http://localhost:3001")) {
        Write-Host "[start] Launching frontend..."
        $frontendProc = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$frontendDir'; npm run dev" -PassThru
    }
    else {
        Write-Host "[skip] Frontend already live on :3001"
    }

    $null = Wait-Http -Url "http://localhost:5051/api/snapshot" -TimeoutSec 90 -Name "backend"
    $null = Wait-Http -Url "http://localhost:3001" -TimeoutSec 90 -Name "frontend"

    if (-not $NoBrowser) {
        Start-Process "http://localhost:3001"
    }

    Write-Host "[loop] Starting live snapshot monitor (Ctrl+C to stop)..."
    & $python $monitorScript --api-url "http://localhost:5051/api/snapshot" --out-dir (Join-Path $root $OutDir) --interval $MonitorIntervalSec
}
finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host "[stop] Stopping backend process $($backendProc.Id)"
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        Write-Host "[stop] Stopping frontend process $($frontendProc.Id)"
        Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue
    }
}
