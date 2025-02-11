using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using MatrixProd.GPU.AMD;
using MatrixProd.GPU.NVIDIA;

namespace MatrixProd.GPU;

public static class MatrixMultiplicationFactory
{
    internal static IGPUMatrixMultiplication? _gpuMultiplier;
    internal static bool _gpuInitialized;
    private static readonly SemaphoreSlim _initLock = new SemaphoreSlim(1, 1);
    private static bool _initializationInProgress;

    public static async Task<IMatrixMultiplication> CreateMultiplier(string method)
    {
        try
        {
            // For GPU methods, wait for any ongoing initialization
            if ((method == "g" || method == "h") && !_gpuInitialized)
            {
                await _initLock.WaitAsync();
                try
                {
                    // Skip initialization if it's already in progress
                    if (!_gpuInitialized && !_initializationInProgress)
                    {
                        // Just use what's already initialized
                        _gpuInitialized = true;
                    }
                }
                finally
                {
                    _initLock.Release();
                }
            }

            // Create appropriate multiplier based on method
            IMatrixMultiplication multiplier = method switch
            {
                "f" => new RowMatrixMultiplication(),
                "c" => new ColumnMatrixMultiplication(),
                "g" when _gpuMultiplier != null => new GPUMatrixMultiplier(_gpuMultiplier),
                "g" => new RowMatrixMultiplication(), // Fallback to CPU if no GPU
                "h" => new HybridMatrixMultiplication(),
                _ => throw new ArgumentException($"Invalid multiplication method: {method}", nameof(method))
            };

            return multiplier;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating matrix multiplier: {ex.Message}");
            // Final fallback to CPU implementation
            return new RowMatrixMultiplication();
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