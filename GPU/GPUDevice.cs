using MatrixProd.Core.Interfaces;
using MatrixProd.GPU.AMD;
using MatrixProd.GPU.NVIDIA;  // Added to resolve the NVIDIA namespace error
using ManagedCuda; // Added to use ManagedCuda

namespace MatrixProd.GPU;

public abstract class GPUDeviceBase : IGPUDevice
{
    public abstract string VendorName { get; }
    public abstract string DeviceName { get; }
    public bool IsAvailable { get; protected set; }
    
    public abstract Task Initialize();
    public abstract void Dispose();
}

public static class GPUDeviceFactory
{
    public static async Task<IGPUDevice> CreateDevice()
    {
        try
        {
            // Try AMD implementation first
            var amdImpl = new AmdImplementation();
            await amdImpl.Device.Initialize();
            return amdImpl.Device;
        }
        catch (Exception)
        {
            try
            {
                // Fallback to NVIDIA implementation
                var cudaContext = new CudaContext(0);
                var nvidiaDevice = new NvidiaDevice(cudaContext);
                await nvidiaDevice.Initialize();
                return nvidiaDevice;
            }
            catch (Exception nvidiaEx)
            {
                throw new PlatformNotSupportedException("No compatible GPU device found", nvidiaEx);
            }
        }
    }
}