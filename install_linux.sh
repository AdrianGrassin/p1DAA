#!/bin/bash
# Linux installer script for Matrix Multiplication project

set -e # Exit on error

print_status() {
    echo -e "\e[36m=> $1\e[0m"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "Cannot detect Linux distribution"
    exit 1
fi

# Install .NET SDK
print_status "Installing .NET SDK..."
case $DISTRO in
    "ubuntu"|"debian")
        wget https://packages.microsoft.com/config/$DISTRO/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
        dpkg -i packages-microsoft-prod.deb
        rm packages-microsoft-prod.deb
        apt-get update
        apt-get install -y dotnet-sdk-8.0
        ;;
    "fedora")
        dnf install dotnet-sdk-8.0 -y
        ;;
    "arch")
        pacman -S dotnet-sdk --noconfirm
        ;;
    *)
        echo "Unsupported distribution: $DISTRO"
        exit 1
        ;;
esac

# Detect GPU and install appropriate drivers/SDKs
print_status "Detecting GPU..."
if lspci | grep -i nvidia > /dev/null; then
    print_status "NVIDIA GPU detected. Installing CUDA..."
    case $DISTRO in 
        "ubuntu"|"debian")
            wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO$(lsb_release -rs | sed 's/\.//g')/x86_64/cuda-keyring_1.0-1_all.deb
            dpkg -i cuda-keyring_1.0-1_all.deb
            rm cuda-keyring_1.0-1_all.deb
            apt-get update
            apt-get install -y cuda-toolkit-12-3
            ;;
        "fedora")
            dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora$(rpm -E %fedora)/x86_64/cuda-fedora$(rpm -E %fedora).repo
            dnf install -y cuda-toolkit
            ;;
        "arch")
            pacman -S cuda cuda-tools --noconfirm
            ;;
    esac
    
    # Set up CUDA environment
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
fi

if lspci | grep -i amd > /dev/null; then
    print_status "AMD GPU detected. Installing ROCm OpenCL..."
    case $DISTRO in
        "ubuntu"|"debian")
            wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
            echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list
            apt-get update
            apt-get install -y rocm-opencl-dev
            ;;
        "fedora")
            dnf config-manager --add-repo https://repo.radeon.com/rocm/yum/rpm
            dnf install -y rocm-opencl-devel
            ;;
        "arch")
            pacman -S opencl-amd --noconfirm
            ;;
    esac
fi

# Install OpenCL ICD Loader
print_status "Installing OpenCL ICD Loader..."
case $DISTRO in
    "ubuntu"|"debian")
        apt-get install -y ocl-icd-opencl-dev
        ;;
    "fedora")
        dnf install -y ocl-icd-devel
        ;;
    "arch")
        pacman -S opencl-icd-loader --noconfirm
        ;;
esac

# Build the project
print_status "Building the project..."
cd "$(dirname "$0")"
dotnet restore
dotnet build --configuration Release

print_status "Installation completed successfully!"
echo -e "\e[32m
Matrix Multiplication Project is now set up!
To run the program:
- For CPU tests: dotnet run <size> f
- For GPU tests: dotnet run <size> g
- For hybrid tests: dotnet run <size> h
- For benchmarks: dotnet run csv

Replace <size> with matrix size (e.g., 1000)
\e[0m"
