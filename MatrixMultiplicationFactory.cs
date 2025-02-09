namespace MatrixProd;

public static class MatrixMultiplicationFactory
{
    private static bool? hasNvidia;
    
    public static async Task<MatrixMultiplication> CreateGPUMultiplier()
    {
        // Cache NVIDIA detection
        if (!hasNvidia.HasValue)
        {
            try
            {
                var gpuInfo = MatrixGPUMultiplication.GetBestGPU();
                hasNvidia = gpuInfo.VendorName.Contains("NVIDIA");
            }
            catch
            {
                hasNvidia = false;
            }
        }

        // Try CUDA first for NVIDIA GPUs
        if (hasNvidia.Value)
        {
            try
            {
                return new MatrixCUDAMultiplication();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CUDA initialization failed, falling back to OpenCL: {ex.Message}");
            }
        }

        // Fall back to OpenCL
        return new MatrixGPUMultiplication();
    }
}