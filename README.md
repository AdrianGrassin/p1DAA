# Matrix Multiplication Project

High-performance matrix multiplication implementation with automatic GPU detection and optimization.

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

## Project Structure

This project uses a modular approach to handle different GPU implementations:

- **Main Branch**: Contains core functionality
  - Base matrix operations
  - CPU-based implementations
  - GPU detection scripts
  - Auto-download mechanism for GPU-specific code

- **nvidia-gpu Branch**: NVIDIA-specific implementation
  - CUDA-optimized code
  - NVIDIA-specific kernel tuning
  - Required CUDA dependencies

- **amd-gpu Branch**: AMD-specific implementation
  - ROCm/OpenCL optimized code
  - GCN architecture optimizations
  - Required AMD dependencies

## Automatic Installation

The installer:
1. Detects your GPU vendor
2. Downloads only the necessary GPU-specific code
3. Installs required dependencies
4. Builds with vendor-specific optimizations

### Windows Installation
```powershell
.\install_windows.ps1
```

### Linux Installation
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

## Features

- Automatic GPU vendor detection
- Vendor-specific optimizations:
  - NVIDIA: CUDA support, warp-aware scheduling, unified memory
  - AMD: GCN architecture optimizations, vectorized operations
- Hybrid CPU-GPU computation for smaller matrices
- Benchmarking tools

## Usage

```bash
# CPU row-based multiplication (size 1000x1000)
dotnet run 1000 f

# CPU column-based multiplication
dotnet run 1000 c

# GPU multiplication (auto-detects and downloads required code)
dotnet run 1000 g

# Hybrid CPU-GPU multiplication
dotnet run 1000 h

# Run benchmarks and generate CSV
dotnet run csv
```

## Performance Tips

### For NVIDIA GPUs
- Optimal matrix sizes: multiples of 32
- Enable compute mode in NVIDIA settings
- Monitor temperature for thermal throttling

### For AMD GPUs
- Optimal matrix sizes: multiples of 64
- Enable high-performance mode
- Use latest ROCm drivers

### General Tips
- Prefer large matrices (>1000x1000) for GPU
- Use CPU methods for smaller matrices (<500x500)
- Keep matrices in GPU memory when possible

## Performance Characteristics

Current benchmark results show:

- Small matrices (<500x500):
  - CPU outperforms GPU due to data transfer overhead
  - Hybrid mode automatically switches to CPU
  - Best performance with row-based CPU implementation

- Medium matrices (500x500 - 1000x1000):
  - GPU starts showing benefits
  - Hybrid mode provides best performance
  - 2-3x speedup over CPU-only methods

- Large matrices (>1000x1000):
  - GPU provides significant speedup (2.86 GFlops)
  - Hybrid mode slightly edges out pure GPU
  - Up to 25x speedup over CPU methods

### Recommendations

For optimal performance:
- Use CPU methods for matrices smaller than 500x500 
- Use Hybrid mode for matrices 500x500 to 1000x1000
- Use GPU mode for matrices larger than 1000x1000

## Troubleshooting

### Common Issues

1. "GPU multiplication is not available"
   - Run the installer first to download GPU-specific code
   - Check your internet connection
   - Verify GPU drivers are installed

2. "Failed to detect GPU"
   - Update GPU drivers
   - Check if GPU is recognized by system
   - Run with administrator/root privileges

### Debug Commands

#### Windows
```powershell
# Check GPU vendor and driver
wmic path win32_VideoController get name, driverversion

# For NVIDIA:
nvidia-smi
nvcc --version

# For AMD:
rocm-smi
```

#### Linux
```bash
# Check GPU vendor
lspci | grep -i "vga\|3d"

# For NVIDIA:
nvidia-smi
nvcc --version

# For AMD:
rocm-smi
clinfo
```

## Development

To contribute:
1. Identify target GPU architecture
2. Use appropriate branch:
   - NVIDIA optimizations → nvidia-gpu branch
   - AMD optimizations → amd-gpu branch
   - Core functionality → main branch
3. Test with various matrix sizes
4. Submit pull request

## License

MIT License - See LICENSE file for details.