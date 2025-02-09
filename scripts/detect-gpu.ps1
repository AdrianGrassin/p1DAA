#!/usr/bin/env pwsh
# Detect GPU and select appropriate git branch

function Get-GPUVendor {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.AdapterDACType -ne "Internal" }
    if ($gpu.Name -match "NVIDIA") {
        return "NVIDIA"
    }
    elseif ($gpu.Name -match "AMD") {
        return "AMD"
    }
    return "UNKNOWN"
}

function Select-GitBranch {
    param (
        [string]$vendor
    )
    
    # Fetch all branches
    git fetch origin
    
    switch ($vendor) {
        "NVIDIA" {
            Write-Host "NVIDIA GPU detected. Switching to NVIDIA-optimized branch..."
            git checkout nvidia-gpu
        }
        "AMD" {
            Write-Host "AMD GPU detected. Switching to AMD-optimized branch..."
            git checkout amd-gpu
        }
        default {
            Write-Host "Unknown GPU vendor. Using default branch..."
            git checkout master
        }
    }
}