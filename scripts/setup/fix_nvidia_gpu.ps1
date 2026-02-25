# NVIDIA GPU Fix Script for HP Victus Laptop
# Run as Administrator

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "NVIDIA GeForce RTX 2050 - GPU Fix Script" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Check current status
Write-Host "1. Checking NVIDIA GPU Status..." -ForegroundColor Yellow
$nvidia = Get-PnpDevice -FriendlyName "*NVIDIA GeForce*"
Write-Host "   Current Status: $($nvidia.Status)" -ForegroundColor $(if ($nvidia.Status -eq 'OK') {'Green'} else {'Red'})
Write-Host ""

# Option 1: Enable the device
Write-Host "2. Attempting to enable NVIDIA GPU..." -ForegroundColor Yellow
try {
    Enable-PnpDevice -InstanceId $nvidia.InstanceId -Confirm:$false -ErrorAction Stop
    Write-Host "   ✓ GPU enabled successfully!" -ForegroundColor Green
}
catch {
    Write-Host "   ✗ Failed to enable (this requires Administrator rights)" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Check if CUDA toolkit is installed
Write-Host "3. Checking CUDA Toolkit..." -ForegroundColor Yellow
$cudaPaths = @(
    "$env:CUDA_PATH",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

$cudaFound = $false
foreach ($path in $cudaPaths) {
    if ($path -and (Test-Path "$path\bin\nvcc.exe" -ErrorAction SilentlyContinue)) {
        Write-Host "   ✓ CUDA Toolkit found: $path" -ForegroundColor Green
        $env:CUDA_PATH = $path
        $cudaFound = $true
        break
    }
}

if (-not $cudaFound) {
    Write-Host "   ✗ CUDA Toolkit not found!" -ForegroundColor Red
    Write-Host "   Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive" -ForegroundColor Cyan
}
Write-Host ""

# Check PyTorch CUDA
Write-Host "4. Testing PyTorch CUDA..." -ForegroundColor Yellow
& python -c @"
import torch
print('   PyTorch: ' + torch.__version__)
print('   CUDA Available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('   CUDA Version: ' + torch.version.cuda)
    print('   GPU Name: ' + torch.cuda.get_device_name(0))
    print('   GPU Count: ' + str(torch.cuda.device_count()))
else:
    print('   ⚠ CUDA not detected by PyTorch')
"@
Write-Host ""

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "1. If GPU not enabled: Right-click PowerShell → Run as Administrator" -ForegroundColor White
Write-Host "   Then run: powershell -ExecutionPolicy Bypass -File fix_nvidia_gpu.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. If CUDA not found: Install CUDA Toolkit 12.1" -ForegroundColor White
Write-Host "   Download: https://developer.nvidia.com/cuda-12-1-0-download-archive" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Update NVIDIA drivers from:" -ForegroundColor White
Write-Host "   https://www.nvidia.com/download/index.aspx" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. After fixing, restart computer and run validation:" -ForegroundColor White
Write-Host "   python phase2_validation.py" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
