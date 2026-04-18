@echo off
echo.
echo ========================================
echo Checking NVIDIA GPU after BIOS change
echo ========================================
echo.

echo 1. Checking GPU devices...
wmic path win32_VideoController get name,status,driverversion
echo.

echo 2. Checking NVIDIA GPU status...
powershell -Command "Get-PnpDevice -FriendlyName '*NVIDIA*' | Select-Object FriendlyName, Status, Problem | Format-List"
echo.

echo 3. Testing PyTorch CUDA...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

echo ========================================
echo If CUDA is False, the GPU is still disabled in BIOS
echo ========================================
pause
