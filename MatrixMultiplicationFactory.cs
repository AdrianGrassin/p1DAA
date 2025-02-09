namespace MatrixProd;

public interface MatrixMultiplication
{
    Task<Matriz> multiplicar(Matriz m1, Matriz m2);
}

public static class MatrixMultiplicationFactory
{
    public static async Task DownloadAndSetupGPUCode()
    {
        var gpu = await DetectGPU();
        await DownloadGPUSpecificCode(gpu);
    }

    private static async Task<string> DetectGPU()
    {
        if (OperatingSystem.IsWindows())
        {
            // Use PowerShell script
            using var ps = System.Management.Automation.PowerShell.Create();
            ps.AddScript(File.ReadAllText("scripts/detect-gpu.ps1"));
            ps.AddCommand("Get-GPUVendor");
            var result = await ps.InvokeAsync();
            return result[0].ToString();
        }
        else if (OperatingSystem.IsLinux())
        {
            // Use bash script
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "/bin/bash",
                    Arguments = "scripts/detect-gpu.sh",
                    RedirectStandardOutput = true,
                    UseShellExecute = false
                }
            };
            process.Start();
            string output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            return output.Trim();
        }
        return "UNKNOWN";
    }

    private static async Task DownloadGPUSpecificCode(string gpuVendor)
    {
        string repoUrl = "https://github.com/yourusername/MatrixProd.git";
        string branch = gpuVendor switch
        {
            "NVIDIA" => "nvidia-gpu",
            "AMD" => "amd-gpu",
            _ => "master"
        };

        // Clone specific branch
        var process = new System.Diagnostics.Process
        {
            StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "git",
                Arguments = $"clone -b {branch} --single-branch {repoUrl} gpu-code",
                RedirectStandardOutput = true,
                UseShellExecute = false
            }
        };
        await process.WaitForExitAsync();

        // Copy necessary files
        if (Directory.Exists("gpu-code"))
        {
            foreach (var file in Directory.GetFiles("gpu-code", "Matrix*Multiplication.cs"))
            {
                File.Copy(file, Path.Combine(Directory.GetCurrentDirectory(), Path.GetFileName(file)), true);
            }
            Directory.Delete("gpu-code", true);
        }
    }

    public static MatrixMultiplication CreateMultiplier(string method)
    {
        return method switch
        {
            "f" => new MatrixFilMultiplication(),
            "c" => new MatrixColMultiplication(),
            "g" => CreateGPUMultiplier(),
            "h" => new MatrixHybridMultiplication(),
            _ => throw new ArgumentException("Invalid multiplication method")
        };
    }

    private static MatrixMultiplication CreateGPUMultiplier()
    {
        // This will be implemented in the GPU-specific branches
        throw new NotImplementedException("GPU multiplication not available in base version. Please run the installer first.");
    }
}