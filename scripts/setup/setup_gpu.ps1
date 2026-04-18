# GPU Setup Script for AR Mirror
# This script will:
# 1. Find your Python installation
# 2. Uninstall CPU-only PyTorch
# 3. Install CUDA-enabled PyTorch
# 4. Verify GPU is working

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "AR MIRROR - GPU ACCELERATION SETUP" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Find Python
Write-Host "[1/5] Finding Python installation..." -ForegroundColor Yellow

$pythonPaths = @(
    "C:\Users\shind\AppData\Local\Programs\Python",
    "C:\Program Files\Python*",
    "C:\Python*"
)

$pythonExe = $null
foreach ($path in $pythonPaths) {
    $found = Get-ChildItem -Path $path -Recurse -Filter python.exe -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -notlike "*WindowsApps*" } |
    Select-Object -First 1
    
    if ($found) {
        $pythonExe = $found.FullName
        break
    }
}

if (-not $pythonExe) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure to:" -ForegroundColor Yellow
    Write-Host "  1. Check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "  2. Install for all users (recommended)" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "OK Found Python: $pythonExe" -ForegroundColor Green
& $pythonExe --version
Write-Host ""

# Step 2: Check current PyTorch installation
Write-Host "[2/5] Checking current PyTorch installation..." -ForegroundColor Yellow

$pipList = & $pythonExe -m pip list 2>&1 | Out-String
if ($pipList -match "torch") {
    Write-Host "OK PyTorch is currently installed" -ForegroundColor Green
    
    # Check if it has CUDA
    $cudaCheck = & $pythonExe -c "import torch; print(torch.cuda.is_available())" 2>&1
    if ($cudaCheck -match "True") {
        Write-Host "OK CUDA is already enabled!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your PyTorch installation already has CUDA support." -ForegroundColor Cyan
        Write-Host "Skipping reinstallation." -ForegroundColor Cyan
        Write-Host ""
        $skipReinstall = $true
    }
    else {
        Write-Host "WARNING Current PyTorch is CPU-only" -ForegroundColor Yellow
        Write-Host "Will uninstall and reinstall with CUDA support..." -ForegroundColor Yellow
        $skipReinstall = $false
    }
}
else {
    Write-Host "WARNING PyTorch not installed" -ForegroundColor Yellow
    $skipReinstall = $false
}
Write-Host ""

# Step 3: Uninstall old PyTorch (if needed)
if (-not $skipReinstall) {
    Write-Host "[3/5] Uninstalling old PyTorch..." -ForegroundColor Yellow
    
    & $pythonExe -m pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Host
    
    Write-Host "OK Old PyTorch uninstalled" -ForegroundColor Green
    Write-Host ""
    
    # Step 4: Install CUDA-enabled PyTorch
    Write-Host "[4/5] Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
    Write-Host "This may take 5-10 minutes depending on your internet speed..." -ForegroundColor Cyan
    Write-Host ""
    
    # PyTorch with CUDA 12.1 (for RTX 2050)
    & $pythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK PyTorch with CUDA 12.1 installed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host ""
}
else {
    Write-Host "[3/5] Skipping PyTorch uninstallation (already has CUDA)" -ForegroundColor Green
    Write-Host "[4/5] Skipping PyTorch installation (already has CUDA)" -ForegroundColor Green
    Write-Host ""
}

# Step 5: Verify GPU
Write-Host "[5/5] Verifying GPU setup..." -ForegroundColor Yellow
Write-Host ""

# Create temp Python script
$scriptPath = Join-Path $env:TEMP "verify_gpu_temp.py"
$verifyScript = @'
import torch
import sys

print('='*60)
print('GPU VERIFICATION')
print('='*60)
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_memory / 1024**3:.2f} GB')
    print('='*60)
    print('SUCCESS - GPU READY FOR AR MIRROR!')
    print('='*60)
    sys.exit(0)
else:
    print('='*60)
    print('ERROR - CUDA NOT AVAILABLE')
    print('='*60)
    print('')
    print('Possible issues:')
    print('  1. NVIDIA drivers not installed')
    print('  2. GPU disabled in BIOS')
    print('  3. PyTorch installation failed')
    print('')
    print('Check Task Manager > Performance > GPU to verify GPU is visible')
    sys.exit(1)
'@

Set-Content -Path $scriptPath -Value $verifyScript
& $pythonExe $scriptPath
$exitCode = $LASTEXITCODE
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=====================================================================" -ForegroundColor Green
    Write-Host "SUCCESS! GPU is ready!" -ForegroundColor Green
    Write-Host "=====================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run verification: python scripts\verify_gpu.py" -ForegroundColor White
    Write-Host "  2. Run app: python app.py --phase 2" -ForegroundColor White
    Write-Host "  3. Expected FPS: 200-300 (vs current 21)" -ForegroundColor White
    Write-Host ""
}
else {
    Write-Host "=====================================================================" -ForegroundColor Red
    Write-Host "WARNING: GPU not detected" -ForegroundColor Red
    Write-Host "=====================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. GPU is visible in Task Manager" -ForegroundColor White
    Write-Host "  2. NVIDIA drivers are installed" -ForegroundColor White
    Write-Host "  3. Restart computer if drivers were just installed" -ForegroundColor White
    Write-Host ""
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
