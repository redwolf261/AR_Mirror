# Quick Setup Script for AR Mirror with Conda
# Run this in Anaconda PowerShell Prompt

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "AR MIRROR - CONDA ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
Write-Host "[1/5] Checking for Conda installation..." -ForegroundColor Yellow
$condaCheck = Get-Command conda -ErrorAction SilentlyContinue

if (-not $condaCheck) {
    Write-Host "ERROR: Conda not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Anaconda or Miniconda first:" -ForegroundColor Yellow
    Write-Host "  Anaconda: https://www.anaconda.com/download" -ForegroundColor White
    Write-Host "  Miniconda: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host ""
    Write-Host "After installing, run this script in 'Anaconda PowerShell Prompt'" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "OK Conda found: $(conda --version)" -ForegroundColor Green
Write-Host ""

# Check if environment already exists
Write-Host "[2/5] Checking for existing environment..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "ar_mirror"

if ($envExists) {
    Write-Host "WARNING: Environment 'ar_mirror' already exists" -ForegroundColor Yellow
    $response = Read-Host "Remove and recreate? (y/n)"
    
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        conda env remove -n ar_mirror -y
        Write-Host "OK Environment removed" -ForegroundColor Green
    }
    else {
        Write-Host "Updating existing environment instead..." -ForegroundColor Yellow
        conda activate ar_mirror
        conda env update -f environment.yml --prune
        Write-Host "OK Environment updated" -ForegroundColor Green
        Write-Host ""
        Write-Host "Skipping to verification..." -ForegroundColor Cyan
        $skipCreation = $true
    }
}
else {
    Write-Host "OK No existing environment found" -ForegroundColor Green
}
Write-Host ""

# Create environment
if (-not $skipCreation) {
    Write-Host "[3/5] Creating conda environment from environment.yml..." -ForegroundColor Yellow
    Write-Host "This will take 5-10 minutes (downloading PyTorch + CUDA ~3GB)..." -ForegroundColor Cyan
    Write-Host ""
    
    conda env create -f environment.yml
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Environment creation failed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Try updating conda first:" -ForegroundColor Yellow
        Write-Host "  conda update -n base conda -y" -ForegroundColor White
        pause
        exit 1
    }
    
    Write-Host "OK Environment created successfully!" -ForegroundColor Green
}
else {
    Write-Host "[3/5] Skipping environment creation (already exists)" -ForegroundColor Green
}
Write-Host ""

# Activate environment
Write-Host "[4/5] Activating environment..." -ForegroundColor Yellow
conda activate ar_mirror

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate environment" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "OK Environment activated" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "[5/5] Verifying GPU support..." -ForegroundColor Yellow
Write-Host ""

# Create temp verification script
$verifyScript = @'
import sys
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("\nSUCCESS - GPU READY!")
        sys.exit(0)
    else:
        print("\nWARNING - CUDA NOT AVAILABLE")
        print("GPU may need drivers or BIOS configuration")
        print("System will fallback to CPU (slower)")
        sys.exit(1)
except Exception as e:
    print(f"\nERROR: {e}")
    sys.exit(1)
'@

$scriptPath = Join-Path $env:TEMP "verify_conda_env.py"
Set-Content -Path $scriptPath -Value $verifyScript

python $scriptPath
$verifyResult = $LASTEXITCODE
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
if ($verifyResult -eq 0) {
    Write-Host "SETUP COMPLETE - GPU READY!" -ForegroundColor Green
    Write-Host "=====================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run full verification:" -ForegroundColor White
    Write-Host "     python scripts\verify_gpu.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  2. Run validation suite:" -ForegroundColor White
    Write-Host "     python phase2_validation.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  3. Launch AR Mirror:" -ForegroundColor White
    Write-Host "     python app.py --phase 2" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Expected performance: 200-300 FPS (vs 21 FPS CPU)" -ForegroundColor Green
}
else {
    Write-Host "SETUP COMPLETE - GPU NOT DETECTED" -ForegroundColor Yellow
    Write-Host "=====================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Environment created but GPU not available." -ForegroundColor Yellow
    Write-Host "This could be normal if:" -ForegroundColor Cyan
    Write-Host "  1. This is the first setup (drivers may need restart)" -ForegroundColor White
    Write-Host "  2. GPU is disabled in BIOS" -ForegroundColor White
    Write-Host "  3. NVIDIA drivers need updating" -ForegroundColor White
    Write-Host ""
    Write-Host "System will work on CPU (21 FPS) until GPU is enabled." -ForegroundColor Yellow
}
Write-Host "=====================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Environment 'ar_mirror' is now active." -ForegroundColor Cyan
Write-Host "To use AR Mirror in future, always start with:" -ForegroundColor Cyan
Write-Host "  conda activate ar_mirror" -ForegroundColor Yellow
Write-Host ""
pause
