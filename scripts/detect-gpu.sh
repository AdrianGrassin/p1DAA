#!/bin/bash
# Detect GPU and select appropriate git branch

get_gpu_vendor() {
    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA"
    elif lspci | grep -i amd > /dev/null; then
        echo "AMD"
    else
        echo "UNKNOWN"
    fi
}

select_git_branch() {
    local vendor="$1"
    
    # Fetch all branches
    git fetch origin
    
    case "$vendor" in
        "NVIDIA")
            echo "NVIDIA GPU detected. Switching to NVIDIA-optimized branch..."
            git checkout nvidia-gpu
            ;;
        "AMD")
            echo "AMD GPU detected. Switching to AMD-optimized branch..."
            git checkout amd-gpu
            ;;
        *)
            echo "Unknown GPU vendor. Using default branch..."
            git checkout master
            ;;
    esac
}