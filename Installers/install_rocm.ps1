$ErrorActionPreference = "Stop"

Write-Host "Installing AMD ROCm for GPU acceleration..."

# Check if running as admin
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs to be run as Administrator. Please restart with admin privileges."
    exit 1
}

# Define download URLs and paths
$rocmUrl = "https://repo.radeon.com/rocm/msi/5.7.1/ROCm-Setup-5.7.1.exe"
$downloadPath = "$env:TEMP\ROCm-Setup.exe"

try {
    # Download ROCm installer
    Write-Host "Downloading ROCm installer..."
    Invoke-WebRequest -Uri $rocmUrl -OutFile $downloadPath

    # Install ROCm
    Write-Host "Installing ROCm (this may take a while)..."
    Start-Process -FilePath $downloadPath -ArgumentList "/S" -Wait

    # Verify installation
    $rocmPath = "${env:ProgramFiles}\AMD\ROCm"
    if (Test-Path $rocmPath) {
        Write-Host "ROCm installation completed successfully."
        Write-Host "Please restart your computer to complete the installation."
    } else {
        Write-Host "ROCm installation may have failed. Please check the logs."
    }
} catch {
    Write-Host "Error during installation: $_"
    exit 1
} finally {
    # Cleanup
    if (Test-Path $downloadPath) {
        Remove-Item $downloadPath -Force
    }
}