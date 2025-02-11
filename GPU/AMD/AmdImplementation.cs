using OpenCL.Net;
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using System.Text;
using Event = OpenCL.Net.Event;
using CLProgram = OpenCL.Net.Program;

namespace MatrixProd.GPU.AMD;

public class AmdDevice : GPUDeviceBase
{
    private readonly Device _device;
    private bool _disposed;
    private readonly bool _printWorkGroupInfo;

    public override string DeviceName => GetDeviceName();
    public override string VendorName => "AMD";

    private string GetDeviceName()
    {
        ErrorCode error;
        var info = Cl.GetDeviceInfo(_device, DeviceInfo.Name, out error);
        return error == ErrorCode.Success ? info.ToString() : "Unknown AMD Device";
    }

    public AmdDevice(Device device, bool printWorkGroupInfo)
    {
        _device = device;
        _printWorkGroupInfo = printWorkGroupInfo;
        ValidateDevice();
    }

    private void ValidateDevice()
    {
        try
        {
            ErrorCode error;
            var deviceType = Cl.GetDeviceInfo(_device, DeviceInfo.Type, out error).CastTo<DeviceType>();
            if (error != ErrorCode.Success)
            {
                throw new PlatformNotSupportedException($"Failed to get device type: {error}");
            }
            
            if ((deviceType & DeviceType.Gpu) != DeviceType.Gpu)
            {
                throw new PlatformNotSupportedException("Device is not a GPU");
            }

            var deviceVendor = Cl.GetDeviceInfo(_device, DeviceInfo.Vendor, out error).ToString();
            if (error != ErrorCode.Success)
            {
                throw new PlatformNotSupportedException($"Failed to get device vendor: {error}");
            }

            // Accept any AMD vendor string variant
            if (!(deviceVendor.Contains("AMD", StringComparison.OrdinalIgnoreCase) || 
                  deviceVendor.Contains("Advanced Micro Devices", StringComparison.OrdinalIgnoreCase)))
            {
                throw new PlatformNotSupportedException($"Device is not an AMD GPU (Vendor: {deviceVendor})");
            }

            // Log device capabilities
            var deviceVersion = Cl.GetDeviceInfo(_device, DeviceInfo.Version, out error).ToString();
            if (_printWorkGroupInfo)
            {
                Console.WriteLine($"OpenCL Device: {GetDeviceName()}");
                Console.WriteLine($"OpenCL Version: {deviceVersion}");
            }
        }
        catch (DllNotFoundException ex)
        {
            throw new PlatformNotSupportedException("OpenCL runtime not found. Please install AMD OpenCL SDK.", ex);
        }
    }

    public override async Task Initialize()
    {
        try
        {
            ErrorCode error;
            
            // Get platform info first
            Platform[] platforms = new Platform[10];
            uint numPlatforms;
            error = Cl.GetPlatformIDs(10, platforms, out numPlatforms);
            if (error != ErrorCode.Success)
            {
                throw new PlatformNotSupportedException($"Failed to get OpenCL platforms: {error}");
            }

            // Don't check ROCm DLLs, they might not be needed for OpenCL
            if (_printWorkGroupInfo)
            {
                Console.WriteLine($"Found {numPlatforms} OpenCL platform(s)");
            }

            // Get compute units (CUs)
            var computeUnits = Cl.GetDeviceInfo(_device, DeviceInfo.MaxComputeUnits, out error).CastTo<uint>();
            if (error == ErrorCode.Success && _printWorkGroupInfo)
            {
                Console.WriteLine($"Detected AMD GPU: {DeviceName}");
                Console.WriteLine($"Compute Units: {computeUnits}");
            }

            // Get global memory size
            var globalMemSize = Cl.GetDeviceInfo(_device, DeviceInfo.GlobalMemSize, out error).CastTo<ulong>();
            if (error == ErrorCode.Success && _printWorkGroupInfo)
            {
                Console.WriteLine($"Global Memory: {globalMemSize / (1024*1024)} MB");
            }

            // Try to create a test context to verify the device is working
            using (var testContext = Cl.CreateContext(null, 1, new[] { _device }, null, IntPtr.Zero, out error))
            {
                if (error != ErrorCode.Success)
                    throw new PlatformNotSupportedException("Failed to create test context");
                
                using (var testQueue = Cl.CreateCommandQueue(testContext, _device, CommandQueueProperties.None, out error))
                {
                    if (error != ErrorCode.Success)
                        throw new PlatformNotSupportedException("Failed to create test command queue");
                }
            }

            IsAvailable = true;
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error initializing AMD GPU: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
            }
            IsAvailable = false;
            throw;
        }
    }

    public override void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

// Add small matrix optimization and better memory management
public class AmdImplementation : IGPUMatrixMultiplication, IDisposable
{
    private Context _context;
    private CommandQueue _queue;
    private readonly Device _device;
    private readonly AmdDevice _gpuDevice;
    private bool _disposed;
    private CLProgram? _program;
    
    // Add fields for matrix dimensions to fix lambda capture
    private int _M, _N, _K;

    private const string OPENCL_KERNEL = @"
    __kernel void matrix_multiply(
        __global const int* A,
        __global const int* B,
        __global int* C,
        const int M,
        const int N,
        const int K,
        const int method) // 0 for row-based, 1 for column-based
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        
        // Bounds checking
        if (row >= M || col >= N) {
            return;
        }
        
        int sum = 0;
        
        if (method == 0) {
            // Row-based multiplication
            const int localRow = row * K;
            for (int k = 0; k < K; k++) {
                sum += A[localRow + k] * B[k * N + col];
            }
        } else {
            // Column-based multiplication
            const int localCol = col * K;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[localCol + k];
            }
        }
        
        C[row * N + col] = sum;
    }";

    // Update thresholds for better performance
    private const int SMALL_MATRIX_THRESHOLD = 500; // Switch to CPU below this size
    private const int OPTIMAL_BLOCK_SIZE = 32;  // Increased from 16 for better vectorization
    private const int MAX_WORK_GROUP_SIZE = 64; // Reduced from 128
    private const int MEMORY_CHUNK_SIZE = 16 * 1024 * 1024; // Increased to 16MB chunks
    private const int MAX_RETRIES = 2; // Reduced from 3
    private const int MAX_MATRIX_SIZE = 2000; // Further reduced size limit

    public IGPUDevice Device => _gpuDevice;

    private int _method; // 0 for row-based, 1 for column-based
    private readonly bool _printWorkGroupInfo;

    public AmdImplementation(bool printWorkGroupInfo = false) // added optional flag
    {
        _printWorkGroupInfo = printWorkGroupInfo;
        var platform = GetAmdPlatform();
        _device = GetAmdDevice(platform);
        _gpuDevice = new AmdDevice(_device, printWorkGroupInfo);
        
        ErrorCode error;
        _context = Cl.CreateContext(null, 1, new[] { _device }, null, IntPtr.Zero, out error);
        if (error != ErrorCode.Success)
            throw new PlatformNotSupportedException("Failed to create OpenCL context");
            
        _queue = Cl.CreateCommandQueue(_context, _device, CommandQueueProperties.None, out error);
        if (error != ErrorCode.Success)
            throw new PlatformNotSupportedException("Failed to create command queue");

        InitializeProgram();
    }

    private void InitializeProgram()
    {
        if (_program != null)
            return;  // Already initialized
            
        try
        {
            ErrorCode error;
            
            const string OPTIMIZED_KERNEL = @"
            #define BLOCK_SIZE 32
            #define VECTOR_SIZE 4

            __kernel void matrix_multiply(
                __global const int* A,
                __global const int* B,
                __global int* C,
                const int M,
                const int N,
                const int K,
                const int method)
            {
                const int row = get_global_id(0);
                const int col = get_global_id(1);
                const int localRow = get_local_id(0);
                const int localCol = get_local_id(1);

                __local int4 As[BLOCK_SIZE][BLOCK_SIZE/4];
                __local int4 Bs[BLOCK_SIZE][BLOCK_SIZE/4];
                
                int4 sum = (int4)(0);
                
                // Process matrix in vector chunks
                for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
                    // Load data into local memory using vector types
                    if (row < M && t * BLOCK_SIZE + localCol * 4 < K) {
                        As[localRow][localCol] = vload4(0, A + row * K + t * BLOCK_SIZE + localCol * 4);
                    }
                    
                    if (t * BLOCK_SIZE + localRow < K && col < N) {
                        Bs[localRow][localCol] = vload4(0, B + (t * BLOCK_SIZE + localRow) * N + col);
                    }
                    
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // Compute using vectorized operations
                    if (row < M && col < N) {
                        for (int k = 0; k < BLOCK_SIZE/4; k++) {
                            sum += As[localRow][k] * Bs[k][localCol];
                        }
                    }
                    
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                // Accumulate vector components
                if (row < M && col < N) {
                    C[row * N + col] = sum.x + sum.y + sum.z + sum.w;
                }
            }";

            // Add delay before program creation to let the driver settle
            Thread.Sleep(100);
            
            _program = Cl.CreateProgramWithSource(_context, 1, new[] { OPTIMIZED_KERNEL }, null, out error);
            if (error != ErrorCode.Success)
                throw new PlatformNotSupportedException($"Failed to create OpenCL program: {error}");

            error = Cl.BuildProgram(_program.Value, 1, new[] { _device }, 
                "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros", 
                null, IntPtr.Zero);

            if (error != ErrorCode.Success)
            {
                var log = Cl.GetProgramBuildInfo(_program.Value, _device, ProgramBuildInfo.Log, out error).ToString();
                throw new PlatformNotSupportedException($"Failed to build OpenCL program: {log}");
            }
        }
        catch (Exception ex)
        {
            _program = null;
            throw new InvalidOperationException("Failed to initialize OpenCL program", ex);
        }
    }

    private Platform GetAmdPlatform()
    {
        try
        {
            // Add retry mechanism for platform detection
            int retryCount = 0;
            const int maxRetries = 3;
            Exception? lastException = null;

            while (retryCount < maxRetries)
            {
                try
                {
                    uint numPlatforms;
                    Platform[] platforms = new Platform[10];
                    ErrorCode error = Cl.GetPlatformIDs(10, platforms, out numPlatforms);
                    if (error != ErrorCode.Success)
                    {
                        throw new PlatformNotSupportedException($"Failed to get OpenCL platforms: {error}");
                    }
                    
                    if (_printWorkGroupInfo)
                    {
                        Console.WriteLine($"Found {numPlatforms} OpenCL platform(s)");
                    }
                    
                    // Look for AMD platform first
                    for (int i = 0; i < numPlatforms; i++)
                    {
                        ErrorCode infoError;
                        var info = Cl.GetPlatformInfo(platforms[i], PlatformInfo.Vendor, out infoError);
                        if (infoError == ErrorCode.Success)
                        {
                            var vendor = info.ToString();
                            if (_printWorkGroupInfo)
                            {
                                Console.WriteLine($"Platform {i}: {vendor}");
                            }
                            if (vendor.Contains("AMD", StringComparison.OrdinalIgnoreCase) || 
                                vendor.Contains("Advanced Micro Devices", StringComparison.OrdinalIgnoreCase))
                            {
                                if (_printWorkGroupInfo)
                                {
                                    Console.WriteLine("Found AMD platform");
                                }
                                Thread.Sleep(50); // Small delay to let the platform stabilize
                                return platforms[i];
                            }
                        }
                    }
                    
                    throw new PlatformNotSupportedException("No AMD OpenCL platform found");
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    retryCount++;
                    if (retryCount < maxRetries)
                    {
                        if (_printWorkGroupInfo)
                        {
                            Console.WriteLine($"Attempt {retryCount} failed, retrying...");
                        }
                        Thread.Sleep(100 * retryCount); // Increasing delay between retries
                        continue;
                    }
                    throw;
                }
            }
            
            if (lastException != null)
            {
                throw new PlatformNotSupportedException("Failed to initialize AMD platform after retries", lastException);
            }
            throw new PlatformNotSupportedException("Failed to initialize AMD platform after retries");
        }
        catch (DllNotFoundException)
        {
            throw new PlatformNotSupportedException("OpenCL runtime not found. Please install AMD GPU drivers.");
        }
    }

    private Device GetAmdDevice(Platform platform)
    {
        uint numDevices;
        Device[] devices = new Device[10];
        ErrorCode error = Cl.GetDeviceIDs(platform, DeviceType.Gpu, 10, devices, out numDevices);
        
        if (error != ErrorCode.Success || numDevices == 0)
            throw new PlatformNotSupportedException("No AMD GPU devices found");
            
        // Get and validate the first GPU device
        var device = devices[0];
        
        ErrorCode deviceError;
        var deviceType = Cl.GetDeviceInfo(device, DeviceInfo.Type, out deviceError).CastTo<DeviceType>();
        if (deviceError != ErrorCode.Success || (deviceType & DeviceType.Gpu) != DeviceType.Gpu)
        {
            throw new PlatformNotSupportedException("Selected device is not a GPU");
        }
        
        return device;
    }

    // Implement IGPUMatrixMultiplication.Multiply
    Task<IMatrix> IGPUMatrixMultiplication.Multiply(IMatrix m1, IMatrix m2, bool useColumns)
    {
        return MultiplyImpl(m1, m2, useColumns);
    }

    // Implement IMatrixMultiplication.Multiply
    public Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        return MultiplyImpl(m1, m2, false);
    }

    private async Task<IMatrix> ProcessInChunks(IMatrix m1, IMatrix m2, int chunkSize, bool useColumns)
    {
        var result = new Matrix(m1.GetRows(), m2.GetCols());
        int numChunksM = (m1.GetRows() + chunkSize - 1) / chunkSize;
        int numChunksN = (m2.GetCols() + chunkSize - 1) / chunkSize;

        // Use semaphore to control memory pressure
        using var semaphore = new SemaphoreSlim(2); // Allow 2 concurrent chunks

        for (int i = 0; i < numChunksM; i++)
        {
            int startM = i * chunkSize;
            int rowCount = Math.Min(chunkSize, m1.GetRows() - startM);

            for (int j = 0; j < numChunksN; j++)
            {
                await semaphore.WaitAsync();
                try
                {
                    int startN = j * chunkSize;
                    int colCount = Math.Min(chunkSize, m2.GetCols() - startN);

                    // Pre-allocate sub-matrices with exact sizes
                    var subM1 = new Matrix(rowCount, m1.GetCols());
                    var subM2 = new Matrix(m2.GetRows(), colCount);

                    // Parallel copy for better performance
                    Parallel.For(0, rowCount, r =>
                    {
                        for (int c = 0; c < m1.GetCols(); c++)
                        {
                            subM1.Set(r, c, m1.Get(startM + r, c));
                        }
                    });

                    Parallel.For(0, m2.GetRows(), r =>
                    {
                        for (int c = 0; c < colCount; c++)
                        {
                            subM2.Set(r, c, m2.Get(r, startN + c));
                        }
                    });

                    // Process chunk
                    var subResult = await MultiplyImpl(subM1, subM2, useColumns);

                    // Copy results back
                    Parallel.For(0, rowCount, r =>
                    {
                        for (int c = 0; c < colCount; c++)
                        {
                            result.Set(startM + r, startN + c, subResult.Get(r, c));
                        }
                    });

                    // Immediate cleanup
                    (subM1 as IDisposable)?.Dispose();
                    (subM2 as IDisposable)?.Dispose();
                    (subResult as IDisposable)?.Dispose();

                    // Force GC after each chunk for large matrices
                    if (m1.GetRows() >= 1500)
                    {
                        GC.Collect(1, GCCollectionMode.Forced);
                        await Task.Delay(50);
                    }
                }
                finally
                {
                    semaphore.Release();
                }
            }
        }

        return result;
    }

    private async Task<IMatrix> MultiplyImpl(IMatrix m1, IMatrix m2, bool useColumns = false)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(AmdImplementation));
        if (_program == null) throw new InvalidOperationException("OpenCL program not initialized");

        // Add size validation first
        if (m1.GetRows() > MAX_MATRIX_SIZE || m1.GetCols() > MAX_MATRIX_SIZE || 
            m2.GetRows() > MAX_MATRIX_SIZE || m2.GetCols() > MAX_MATRIX_SIZE)
        {
            throw new ArgumentException($"Matrix dimensions exceed maximum safe size of {MAX_MATRIX_SIZE}");
        }

        // Calculate memory requirements
        long requiredMemory = (long)m1.GetRows() * m1.GetCols() * sizeof(int) +
                            (long)m2.GetRows() * m2.GetCols() * sizeof(int) +
                            (long)m1.GetRows() * m2.GetCols() * sizeof(int);

        // Get available GPU memory
        ErrorCode error;
        var globalMemSize = Cl.GetDeviceInfo(_device, DeviceInfo.GlobalMemSize, out error).CastTo<ulong>();
        if (error != ErrorCode.Success)
        {
            throw new InvalidOperationException("Failed to get GPU memory size");
        }

        // Use only 75% of available memory to be safe
        long availableMemory = (long)(globalMemSize * 0.75);

        // If memory requirements exceed available memory or matrix is large, process in chunks
        if (requiredMemory > availableMemory || m1.GetRows() >= 1500)
        {
            int chunkSize = (int)Math.Sqrt((availableMemory / (3 * sizeof(int))));
            chunkSize = Math.Min(chunkSize, 512); // Cap chunk size
            return await ProcessInChunks(m1, m2, chunkSize, useColumns);
        }

        int retryCount = 0;
        Exception? lastException = null;

        while (retryCount < MAX_RETRIES)
        {
            try
            {
                // Add explicit cleanup before starting new computation
                GC.Collect();
                GC.WaitForPendingFinalizers();

                if (m1.GetCols() != m2.GetRows())
                    throw new ArgumentException("Matrix dimensions mismatch");

                _M = m1.GetRows();
                _N = m2.GetCols();
                _K = m1.GetCols();
                _method = useColumns ? 1 : 0;

                var computeTask = Task.Run(() =>
                {
                    ErrorCode error;
                    IMatrix? result = null;
                    bool success = false;

                    try
                    {
                        // Calculate optimal work group size
                        var maxWorkGroupSize = Cl.GetDeviceInfo(_device, DeviceInfo.MaxWorkGroupSize, out error).CastTo<IntPtr>();
                        if (error != ErrorCode.Success)
                            throw new InvalidOperationException($"Failed to get device work group size: {error}");

                        // Adjust work group size based on matrix dimensions
                        int baseWorkGroupSize = 16;
                        if (_M >= 1000 || _N >= 1000)
                        {
                            baseWorkGroupSize = 8;
                        }
                        else if (_M >= 500 || _N >= 500)
                        {
                            baseWorkGroupSize = 12;
                        }

                        int workGroupSize = Math.Min(baseWorkGroupSize, (int)Math.Sqrt(maxWorkGroupSize.ToInt64()));
                        
                        if (_printWorkGroupInfo)
                        {
                            Console.WriteLine($"Matrix dimensions: {_M}x{_N}");
                            Console.WriteLine($"Work group size: {workGroupSize}x{workGroupSize}");
                        }

                        // Calculate global work size with padding
                        int globalSizeM = ((_M + workGroupSize - 1) / workGroupSize) * workGroupSize;
                        int globalSizeN = ((_N + workGroupSize - 1) / workGroupSize) * workGroupSize;

                        if (_printWorkGroupInfo)
                        {
                            Console.WriteLine($"Global work size: {globalSizeM}x{globalSizeN}");
                        }

                        result = new Matrix(_M, _N);

                        // Create buffers with pinned memory and specific flags
                        IMem? bufferA = null, bufferB = null, bufferC = null;
                        Kernel? kernel = null;

                        try
                        {
                            int[] dataA = new int[_M * _K];
                            int[] dataB = new int[_K * _N];
                            int[] dataC = new int[_M * _N];

                            // Copy input data
                            Parallel.For(0, _M, i =>
                            {
                                for (int j = 0; j < _K; j++)
                                {
                                    dataA[i * _K + j] = m1.Get(i, j);
                                }
                            });

                            Parallel.For(0, _K, i =>
                            {
                                for (int j = 0; j < _N; j++)
                                {
                                    dataB[i * _N + j] = m2.Get(i, j);
                                }
                            });

                            unsafe
                            {
                                fixed (int* ptrA = dataA, ptrB = dataB, ptrC = dataC)
                                {
                                    // Create buffers with optimal flags
                                    bufferA = Cl.CreateBuffer(_context, MemFlags.ReadOnly | MemFlags.UseHostPtr,
                                        (IntPtr)(sizeof(int) * dataA.Length), (IntPtr)ptrA, out error);
                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to create buffer A: {error}");

                                    bufferB = Cl.CreateBuffer(_context, MemFlags.ReadOnly | MemFlags.UseHostPtr,
                                        (IntPtr)(sizeof(int) * dataB.Length), (IntPtr)ptrB, out error);
                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to create buffer B: {error}");

                                    bufferC = Cl.CreateBuffer(_context, MemFlags.WriteOnly | MemFlags.AllocHostPtr,
                                        (IntPtr)(sizeof(int) * dataC.Length), IntPtr.Zero, out error);
                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to create buffer C: {error}");

                                    // Create and set up kernel
                                    kernel = Cl.CreateKernel(_program.Value, "matrix_multiply", out error);
                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to create kernel: {error}");

                                    error = Cl.SetKernelArg(kernel.Value, 0, bufferA);
                                    error |= Cl.SetKernelArg(kernel.Value, 1, bufferB);
                                    error |= Cl.SetKernelArg(kernel.Value, 2, bufferC);
                                    error |= Cl.SetKernelArg(kernel.Value, 3, _M);
                                    error |= Cl.SetKernelArg(kernel.Value, 4, _N);
                                    error |= Cl.SetKernelArg(kernel.Value, 5, _K);
                                    error |= Cl.SetKernelArg(kernel.Value, 6, _method);

                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to set kernel arguments: {error}");

                                    // Execute kernel
                                    Event evnt;
                                    error = Cl.EnqueueNDRangeKernel(_queue, kernel.Value, 2, null,
                                        new[] { (IntPtr)_M, (IntPtr)_N },
                                        new[] { (IntPtr)workGroupSize, (IntPtr)workGroupSize },
                                        0, null, out evnt);

                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to execute kernel: {error}");

                                    // Read back results (using correct Bool initialization)
                                    error = Cl.EnqueueReadBuffer(_queue, bufferC, Bool.True, IntPtr.Zero,
                                        (IntPtr)(sizeof(int) * dataC.Length), (IntPtr)ptrC,
                                        0, null, out evnt);

                                    if (error != ErrorCode.Success)
                                        throw new InvalidOperationException($"Failed to read results: {error}");

                                    // Copy results to output matrix
                                    Parallel.For(0, _M, i =>
                                    {
                                        for (int j = 0; j < _N; j++)
                                        {
                                            result.Set(i, j, dataC[i * _N + j]);
                                        }
                                    });
                                }
                            }
                        }
                        finally
                        {
                            // Cleanup resources
                            if (kernel.HasValue) Cl.ReleaseKernel(kernel.Value);
                            if (bufferC != null) Cl.ReleaseMemObject(bufferC);
                            if (bufferB != null) Cl.ReleaseMemObject(bufferB);
                            if (bufferA != null) Cl.ReleaseMemObject(bufferA);
                        }

                        success = true;
                        return result;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"GPU multiplication failed: {ex.Message}");
                        if (!success && result != null)
                        {
                            (result as IDisposable)?.Dispose();
                        }
                        throw;
                    }
                });

                return await computeTask;
            }
            catch (Exception ex)
            {
                lastException = ex;
                retryCount++;
                if (retryCount < MAX_RETRIES)
                {
                    if (_printWorkGroupInfo)
                    {
                        Console.WriteLine($"Attempt {retryCount} failed, retrying...");
                    }
                    Thread.Sleep(500 * retryCount);
                    continue;
                }
                throw new InvalidOperationException("GPU multiplication failed after multiple attempts", lastException);
            }
        }

        throw new InvalidOperationException("GPU multiplication failed after exhausting retries", lastException);
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2, bool useColumns)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(AmdImplementation));
        if (_program == null) throw new InvalidOperationException("OpenCL program not initialized");
        
        // Validate matrix dimensions
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Matrix dimensions are not compatible for multiplication");

        try 
        {
            return await MultiplyImpl(m1, m2, useColumns);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AMD GPU multiplication failed: {ex.Message}");
            throw;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Cl.ReleaseCommandQueue(_queue);
            if (_program.HasValue)
                Cl.ReleaseProgram(_program.Value);
            Cl.ReleaseContext(_context);
            _gpuDevice.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}