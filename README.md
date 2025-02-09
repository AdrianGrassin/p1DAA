# Matrix Multiplication Project

High-performance matrix multiplication implementation using CPU and GPU acceleration.

## System Requirements

### Windows
- Windows 10/11 64-bit
- AMD or NVIDIA GPU
- PowerShell 5.1 or later
- Administrator privileges

### Linux
- Ubuntu 22.04+, Fedora 38+, or Arch Linux
- AMD or NVIDIA GPU
- Root privileges (sudo)

## Quick Installation

### Windows
1. Open PowerShell as Administrator
2. Navigate to the project directory
3. Run:
```powershell
.\install_windows.ps1
```

### Linux
1. Open terminal
2. Navigate to the project directory
3. Run:
```bash
chmod +x install_linux.sh
sudo ./install_linux.sh
```

## Manual Installation

If the automatic installer doesn't work, you can install the dependencies manually:

### Windows
1. Install .NET 8.0 SDK from https://dotnet.microsoft.com/download
2. For NVIDIA GPUs: Install CUDA Toolkit 12.3 from https://developer.nvidia.com/cuda-downloads
3. For AMD GPUs: Install AMD OpenCL SDK

### Linux
1. Install .NET 8.0 SDK
2. For NVIDIA GPUs: Install CUDA Toolkit
3. For AMD GPUs: Install ROCm OpenCL
4. Install OpenCL ICD Loader

## Usage

Run matrix multiplication with different methods:
```bash
# CPU row-based multiplication (size 1000x1000)
dotnet run 1000 f

# CPU column-based multiplication
dotnet run 1000 c

# GPU multiplication
dotnet run 1000 g

# Hybrid CPU-GPU multiplication
dotnet run 1000 h

# Run full benchmark suite and generate CSV
dotnet run csv
```

## Performance Tips

1. For best GPU performance:
   - Use matrix sizes that are multiples of 16
   - Prefer larger matrices (1000x1000 or larger)
   - Keep matrices in GPU memory for multiple operations

2. For best CPU performance:
   - Use the hybrid mode for matrices smaller than 500x500
   - Enable CPU performance mode in your OS
   - Close other CPU-intensive applications

## Troubleshooting

### Windows
- If CUDA initialization fails, verify NVIDIA drivers are up to date
- If OpenCL fails, verify GPU drivers are installed correctly
- Run `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) to verify GPU detection

### Linux
- If CUDA/OpenCL fails, check driver installation with:
  ```bash
  # NVIDIA
  nvidia-smi
  # AMD
  rocm-smi
  ```
- Verify OpenCL installation:
  ```bash
  clinfo
  ```