# Try to detect OpenCL platforms and AMD GPU
$ErrorActionPreference = "Stop"

Write-Host "Checking OpenCL installation..."

# Check for OpenCL DLLs
$openclPaths = @(
    "C:\Windows\System32\OpenCL.dll",
    "${env:AMDAPPSDKROOT}\bin\x64\OpenCL.dll",
    "${env:ProgramFiles}\AMD\ROCm\bin\OpenCL.dll"
)

foreach ($path in $openclPaths) {
    if (Test-Path $path) {
        Write-Host "Found OpenCL at: $path"
    }
}

# Try to get GPU info using PowerShell
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*Radeon*" }
    if ($gpu) {
        Write-Host "Found AMD GPU: $($gpu.Name)"
        Write-Host "Driver Version: $($gpu.DriverVersion)"
    }
} catch {
    Write-Host "Error detecting GPU: $_"
}

# Check ROCm installation
$rocmPath = "${env:ProgramFiles}\AMD\ROCm"
if (Test-Path $rocmPath) {
    Write-Host "ROCm installation found at: $rocmPath"
    Get-ChildItem "$rocmPath\bin\*.dll" | ForEach-Object {
        Write-Host "Found ROCm DLL: $($_.Name)"
    }
} else {
    Write-Host "ROCm installation not found. Please install ROCm from https://www.amd.com/en/developer/rocm.html"
}