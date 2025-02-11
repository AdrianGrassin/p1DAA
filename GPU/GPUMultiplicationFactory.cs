using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using MatrixProd.GPU.AMD;
using MatrixProd.GPU.NVIDIA;

namespace MatrixProd.GPU;

public static class MatrixMultiplicationFactory
{
    internal static IGPUMatrixMultiplication? _gpuMultiplier;
    internal static bool _gpuInitialized;
    private static readonly SemaphoreSlim _initLock = new(1, 1);
    private static bool _initialized;
    private static bool _initializationInProgress;
    private static readonly Dictionary<string, IMatrixMultiplication> _multipliers = new();

    public static async Task<IMatrixMultiplication> CreateMultiplier(string method)
    {
        await _initLock.WaitAsync();
        try
        {
            // If we already have this multiplier initialized, return it
            if (_multipliers.TryGetValue(method, out var existingMultiplier))
            {
                return existingMultiplier;
            }

            // Initialize multiplier based on method
            IMatrixMultiplication multiplier = method switch
            {
                "f" => new MatrixFilMultiplication(),
                "c" => new ColumnMatrixMultiplication(),
                "g" when _gpuMultiplier != null => new GPUMatrixMultiplier(_gpuMultiplier),
                "h" when _gpuMultiplier != null => new HybridMatrixMultiplication(),
                "g" or "h" => new RowMatrixMultiplication(), // Fallback to CPU if no GPU
                _ => throw new ArgumentException($"Invalid multiplication method: {method}", nameof(method))
            };

            // Cache the multiplier
            _multipliers[method] = multiplier;
            return multiplier;
        }
        finally
        {
            _initLock.Release();
        }
    }

    internal static async Task<IGPUMatrixMultiplication?> DetectAndInitializeGPU()
    {
        if (_initializationInProgress)
            return null;

        _initializationInProgress = true;
        try
        {
            // Try AMD first since it was detected in the environment
            Console.WriteLine("Attempting to initialize AMD GPU...");
            var amdImpl = new AmdImplementation(printWorkGroupInfo: true);
            await amdImpl.Device.Initialize();
            if (amdImpl.Device.IsAvailable)
            {
                await Task.Delay(100); // Give the GPU initialization time to settle
                Console.WriteLine("AMD GPU initialized successfully");
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
                return nvidiaImpl;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"GPU initialization failed: {ex.Message}");
        }
        finally
        {
            _initializationInProgress = false;
        }

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
        _gpuMultiplier = null;
        _gpuInitialized = false;
        _initializationInProgress = false;
        _initLock.Dispose();
    }
}