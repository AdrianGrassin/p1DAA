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
    private static readonly SemaphoreSlim _initLock = new(1, 1);
    private static bool _initialized;
    private static readonly Dictionary<string, IMatrixMultiplication> _multipliers = new();

    public static async Task<IMatrixMultiplication> CreateMultiplier(string method)
    {
        try
        {
            await _initLock.WaitAsync();
            try
            {
                // Check if this method is already initialized
                if (_multipliers.TryGetValue(method, out var existingMultiplier))
                {
                    return existingMultiplier;
                }

                // Create appropriate multiplier based on method
                IMatrixMultiplication multiplier = method switch
                {
                    "f" => new MatrixFilMultiplication(),
                    "c" => new MatrixColMultiplication(),
                    "g" when _gpuMultiplier != null => new GPUMatrixMultiplier(_gpuMultiplier),
                    "g" => new MatrixFilMultiplication(), // Fallback to CPU if no GPU
                    "h" => new HybridMatrixMultiplication(),
                    _ => throw new ArgumentException($"Invalid multiplication method: {method}", nameof(method))
                };

                _multipliers[method] = multiplier;
                Console.WriteLine($"Created multiplier for method: {method}");
                return multiplier;
            }
            finally
            {
                _initLock.Release();
            }
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
        if (_initialized) return _gpuMultiplier;

        await _initLock.WaitAsync();
        try
        {
            if (_initialized) return _gpuMultiplier;

            // Try AMD first since it was detected in the environment
            Console.WriteLine("Attempting to initialize AMD GPU...");
            var amdImpl = new AmdImplementation(printWorkGroupInfo: true);
            await amdImpl.Device.Initialize();
            if (amdImpl.Device.IsAvailable)
            {
                await Task.Delay(100);
                Console.WriteLine("AMD GPU initialized successfully");
                _initialized = true;
                return amdImpl;
            }

            // Then try NVIDIA
            Console.WriteLine("Attempting to initialize NVIDIA GPU...");
            var nvidiaImpl = new NvidiaImplementation();
            await nvidiaImpl.Device.Initialize();
            if (nvidiaImpl.Device.IsAvailable)
            {
                await Task.Delay(100);
                Console.WriteLine("NVIDIA GPU initialized successfully");
                _initialized = true;
                return nvidiaImpl;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"GPU initialization failed: {ex.Message}");
        }
        finally
        {
            _initLock.Release();
        }

        _initialized = true;
        return null;
    }
    
    public static void Cleanup()
    {
        if (_gpuMultiplier is IDisposable disposable)
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

        foreach (var multiplier in _multipliers.Values)
        {
            if (multiplier is IDisposable disposableMultiplier)
            {
                try
                {
                    disposableMultiplier.Dispose();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during multiplier cleanup: {ex.Message}");
                }
            }
        }

        _multipliers.Clear();
        _gpuMultiplier = null;
        _gpuInitialized = false;
        _initialized = false;
        _initLock.Dispose();
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