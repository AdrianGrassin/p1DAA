using ManagedCuda;
using ManagedCuda.BasicTypes;
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using System.Runtime.InteropServices;
using System;
using System.Threading.Tasks;
using MatrixProd.GPU;

namespace MatrixProd.GPU.NVIDIA;

public class NvidiaDevice : GPUDeviceBase
{
    private readonly CudaContext _context;
    private bool _disposed;

    public override string DeviceName { get; }
    public override string VendorName => "NVIDIA";

    public NvidiaDevice(CudaContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        var deviceProps = _context.GetDeviceInfo();
        DeviceName = deviceProps.DeviceName ?? "Unknown NVIDIA Device";
    }

    public override async Task Initialize()
    {
        await Task.CompletedTask; // NVIDIA specific initialization if needed
        IsAvailable = true;
    }

    public override void Dispose()
    {
        if (!_disposed)
        {
            _context?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

public class NvidiaImplementation : IGPUMatrixMultiplication, IDisposable
{
    private CudaContext? _context;
    private NvidiaDevice? _gpuDevice;
    private bool _disposed;
    private bool _initialized;

    public IGPUDevice Device => _gpuDevice ?? throw new InvalidOperationException("Device not initialized");

    public NvidiaImplementation()
    {
        InitializeContext();
    }

    private void InitializeContext()
    {
        try
        {
            // First check if NVIDIA GPU is available
            int deviceCount = CudaContext.GetDeviceCount();
            if (deviceCount <= 0)
            {
                throw new PlatformNotSupportedException("No NVIDIA GPU found");
            }

            Console.WriteLine($"Found {deviceCount} NVIDIA GPU(s)");

            // Get the first available GPU
            _context = new CudaContext(0);
            var deviceProperties = _context.GetDeviceInfo();
            
            Console.WriteLine($"Using NVIDIA GPU: {deviceProperties.DeviceName}");
            Console.WriteLine($"Compute Capability: {deviceProperties.ComputeCapability.Major}.{deviceProperties.ComputeCapability.Minor}");
            Console.WriteLine($"Global Memory: {deviceProperties.TotalGlobalMemory / (1024*1024)} MB");
            Console.WriteLine($"CUDA Cores: {deviceProperties.MultiProcessorCount * 128}"); // Approximate for most cards

            _gpuDevice = new NvidiaDevice(_context);
            _initialized = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error initializing NVIDIA platform: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner error: {ex.InnerException.Message}");
            }
            throw;
        }
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        return await Multiply(m1, m2, false);
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2, bool useColumns)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NvidiaImplementation));
        if (!_initialized) throw new InvalidOperationException("NVIDIA implementation not initialized");
        
        return await Task.Run(() =>
        {
            // Base implementation - will be enhanced with CUDA kernels
            var result = new Matrix(m1.GetRows(), m2.GetCols());
            // ... CUDA implementation will go here
            return result;
        });
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _context?.Dispose();
            _gpuDevice?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}