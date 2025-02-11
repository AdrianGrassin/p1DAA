using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using MatrixProd.GPU;
using System;
using System.Threading.Tasks;
using MatrixProd.GPU.AMD;
using MatrixProd.GPU.NVIDIA;

namespace MatrixProd;

public class HybridMatrixMultiplication : IMatrixMultiplication, IDisposable
{
    private const int CPU_THRESHOLD = 500;
    private readonly IMatrixMultiplication _cpuMultiplier;
    private IGPUMatrixMultiplication? _gpuMultiplier;
    private bool _gpuFailed;
    private bool _disposed;
    private readonly object _gpuLock = new object();

    public HybridMatrixMultiplication()
    {
        _cpuMultiplier = new RowMatrixMultiplication();
        // Use the already initialized GPU instance from the factory
        _gpuMultiplier = MatrixMultiplicationFactory._gpuMultiplier;
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(HybridMatrixMultiplication));
        
        // Always use CPU for small matrices
        if (m1.GetRows() <= CPU_THRESHOLD || m2.GetCols() <= CPU_THRESHOLD)
        {
            return await _cpuMultiplier.Multiply(m1, m2);
        }

        // If GPU hasn't failed and we have a multiplier, try GPU
        if (!_gpuFailed && _gpuMultiplier != null)
        {
            try
            {
                return await _gpuMultiplier.Multiply(m1, m2);
            }
            catch (Exception ex)
            {
                // Log the error and mark GPU as failed
                Console.WriteLine($"GPU multiplication failed, falling back to CPU: {ex.Message}");
                lock (_gpuLock)
                {
                    _gpuFailed = true;
                }
            }
        }
        
        // Fallback to CPU
        return await _cpuMultiplier.Multiply(m1, m2);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_cpuMultiplier is IDisposable disposableCpuMultiplier)
            {
                disposableCpuMultiplier.Dispose();
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

public static class MatrixMultiplicationFactory
{
    internal static IGPUMatrixMultiplication? _gpuMultiplier;
    internal static bool _gpuInitialized;
    private static readonly SemaphoreSlim _initLock = new SemaphoreSlim(1, 1);

    public static async Task<IMatrixMultiplication> CreateMultiplier(string method)
    {
        try
        {
            // For GPU methods, we'll use the already initialized GPU if available
            if ((method == "g" || method == "h") && !MatrixMultiplicationFactory._gpuInitialized)
            {
                await MatrixMultiplicationFactory._initLock.WaitAsync();
                try
                {
                    if (!MatrixMultiplicationFactory._gpuInitialized)
                    {
                        // Instead of detecting again, we'll check if _gpuMultiplier is already initialized
                        if (MatrixMultiplicationFactory._gpuMultiplier == null)
                        {
                            // If not initialized and it's null, we'll fallback to CPU
                            Console.WriteLine("No GPU initialized. Using CPU implementation.");
                            method = "f"; // Fallback to CPU
                        }
                        MatrixMultiplicationFactory._gpuInitialized = true;
                    }
                }
                finally
                {
                    MatrixMultiplicationFactory._initLock.Release();
                }
            }

            // Create appropriate multiplier based on method
            IMatrixMultiplication multiplier = method switch
            {
                "f" => new MatrixFilMultiplication(),
                "c" => new MatrixColMultiplication(),
                "g" when MatrixMultiplicationFactory._gpuMultiplier != null => new GPUMatrixMultiplier(MatrixMultiplicationFactory._gpuMultiplier),
                "g" => new MatrixFilMultiplication(), // Fallback to CPU if no GPU
                "h" => new HybridMatrixMultiplication(),
                _ => throw new ArgumentException($"Invalid multiplication method: {method}", nameof(method))
            };

            Console.WriteLine($"Created multiplier for method: {method}");
            return multiplier;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating matrix multiplier: {ex.Message}");
            // Final fallback to CPU implementation
            return new MatrixFilMultiplication();
        }
    }

    internal static async Task<IGPUMatrixMultiplication?> DetectAndInitializeGPU()
    {
        // Try AMD first since it was detected in the environment
        try
        {
            Console.WriteLine("Attempting to initialize AMD GPU...");
            var amdImpl = new AmdImplementation();
            await amdImpl.Device.Initialize();
            if (amdImpl.Device.IsAvailable)
            {
                await Task.Delay(100); // Give the GPU initialization time to settle
                Console.WriteLine("AMD GPU initialized successfully");
                return amdImpl;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AMD GPU initialization failed: {ex.Message}");
        }

        // Then try NVIDIA
        try
        {
            Console.WriteLine("Attempting to initialize NVIDIA GPU...");
            var nvidiaImpl = new NvidiaImplementation();
            await nvidiaImpl.Device.Initialize();
            if (nvidiaImpl.Device.IsAvailable)
            {
                await Task.Delay(100); // Give the GPU initialization time to settle
                Console.WriteLine("NVIDIA GPU initialized successfully");
                return nvidiaImpl;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"NVIDIA GPU initialization failed: {ex.Message}");
        }

        await Task.Delay(10); // Small delay before returning null to ensure proper async behavior
        return null;
    }
    
    public static void Cleanup()
    {
        if (MatrixMultiplicationFactory._gpuMultiplier is IDisposable disposable)
        {
            try
            {
                disposable.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during GPU cleanup: {ex.Message}");
            }
        }
        MatrixMultiplicationFactory._gpuMultiplier = null;
        MatrixMultiplicationFactory._gpuInitialized = false;
        MatrixMultiplicationFactory._initLock.Dispose();
    }
}

internal class GPUMatrixMultiplier : IMatrixMultiplication, IDisposable
{
    private readonly IGPUMatrixMultiplication _gpuMultiplier;
    private bool _disposed;

    public GPUMatrixMultiplier(IGPUMatrixMultiplication gpuMultiplier)
    {
        _gpuMultiplier = gpuMultiplier;
    }

    public Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GPUMatrixMultiplier));
        return _gpuMultiplier.Multiply(m1, m2);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            (_gpuMultiplier as IDisposable)?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}