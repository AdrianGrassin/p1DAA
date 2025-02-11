#!/usr/bin/env pwsh
# Windows installer script for Matrix Multiplication project

$ErrorActionPreference = "Stop"

function Write-Status($message) {
    Write-Host "=> $message" -ForegroundColor Cyan
}

function Test-Command($command) {
    try { Get-Command $command -ErrorAction Stop | Out-Null; return $true }
    catch { return $false }
}

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Check/Install Chocolatey
Write-Status "Checking Chocolatey installation..."
if (-not (Test-Command "choco")) {
    Write-Status "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

# Install .NET SDK if not present
Write-Status "Checking .NET SDK installation..."
if (-not (Test-Command "dotnet")) {
    Write-Status "Installing .NET SDK..."
    choco install dotnet-8.0-sdk -y
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

# Check GPU vendor and install appropriate drivers/SDKs
Write-Status "Detecting GPU..."
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.AdapterDACType -ne "Internal" }
$isNvidia = $gpu.Name -match "NVIDIA"
$isAMD = $gpu.Name -match "AMD"

if ($isNvidia) {
    Write-Status "NVIDIA GPU detected. Installing CUDA Toolkit..."
    try {
        choco install cuda --version=12.3.1 -y
    }
    catch {
        Write-Host "Failed to install CUDA via Chocolatey. Please install manually from:" -ForegroundColor Yellow
        Write-Host "https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
}

if ($isAMD) {
    Write-Status "AMD GPU detected. Installing ROCm OpenCL..."
    try {
        # Direct download link to AMD OpenCL SDK installer
        $amdDriverUrl = "https://drivers.amd.com/drivers/installer/23.40/whql/amd-software-adrenalin-edition-23.40-minimalsetup-231116_web.exe"
        $installer = "amd_opencl_runtime.exe"
        Write-Status "Downloading AMD OpenCL Runtime..."
        Invoke-WebRequest -Uri $amdDriverUrl -OutFile $installer -UseBasicParsing
        
        if (Test-Path $installer) {
            Write-Status "Installing AMD OpenCL Runtime..."
            Start-Process -FilePath $installer -ArgumentList "/S" -Wait
            Remove-Item $installer -Force
        }
    }
    catch {
        Write-Host "Failed to download AMD OpenCL Runtime. Please install manually from:" -ForegroundColor Yellow
        Write-Host "https://www.amd.com/en/support" -ForegroundColor Yellow
    }
}

# Install project dependencies
Write-Status "Installing project dependencies..."
try {
    dotnet restore
    dotnet build --configuration Release
}
catch {
    Write-Host "Failed to build project: $_" -ForegroundColor Red
    exit 1
}

Write-Status "Installation completed successfully!"
Write-Host @"

Matrix Multiplication Project is now set up!
To run the program:
- For CPU tests: dotnet run <size> f
- For GPU tests: dotnet run <size> g
- For hybrid tests: dotnet run <size> h
- For benchmarks: dotnet run csv

Replace <size> with matrix size (e.g., 1000)
"@ -ForegroundColor Green