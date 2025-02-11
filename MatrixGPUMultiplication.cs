// MatrixGPUMultiplication.cs
using OpenCL.Net;
using System.Runtime.InteropServices;
using System.Text;
using Event = OpenCL.Net.Event;
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;

namespace MatrixProd;

public class GPUInfo
{
    public Platform Platform { get; }
    public Device Device { get; }
    public string VendorName { get; }
    public string DeviceName { get; }
    
    public GPUInfo(Platform platform, Device device)
    {
        Platform = platform;
        Device = device;
        ErrorCode error;
        VendorName = Cl.GetPlatformInfo(platform, PlatformInfo.Vendor, out error).ToString();
        DeviceName = Cl.GetDeviceInfo(device, DeviceInfo.Name, out error).ToString();
    }
}

public class MatrixGPUMultiplication : IMatrixMultiplication
{
    private readonly IGPUMatrixMultiplication? _gpuImpl;
    private readonly bool _useColumns;
    private bool _disposed;

    public MatrixGPUMultiplication(bool useColumns = false)
    {
        _useColumns = useColumns;
        _gpuImpl = MatrixMultiplicationFactory.DetectAndInitializeGPU().Result;
        if (_gpuImpl == null)
        {
            throw new PlatformNotSupportedException("No GPU implementation available");
        }
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MatrixGPUMultiplication));
        
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");

        if (_gpuImpl == null)
            throw new InvalidOperationException("GPU implementation not initialized");

        return await _gpuImpl.Multiply(m1, m2);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            (_gpuImpl as IDisposable)?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}