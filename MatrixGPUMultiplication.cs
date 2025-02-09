// MatrixGPUMultiplication.cs
using OpenCL.Net;
using System.Runtime.InteropServices;
using System.Text;
using Event = OpenCL.Net.Event;

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

public class MatrixGPUMultiplication : MatrixMultiplication, IDisposable
{
    private readonly Context context;
    private readonly CommandQueue queue;
    private readonly OpenCL.Net.Kernel kernel;
    private readonly OpenCL.Net.Program program;
    private readonly Device device;
    private bool disposed;
    private static GPUInfo? selectedGPU;

    private const string kernelSource = @"
        #define BLOCK_SIZE 16

        __kernel void matrix_multiply(
            __global const int* restrict a,
            __global const int* restrict b,
            __global int* restrict c,
            const int M,
            const int N,
            const int K,
            __local int* localA,
            __local int* localB)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);
            int localRow = get_local_id(0);
            int localCol = get_local_id(1);

            int sum = 0;
            
            // Loop over blocks of input matrices
            for (int block = 0; block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
                // Collaborative loading of A and B into local memory
                if ((row < M) && (block * BLOCK_SIZE + localCol < K))
                    localA[localRow * BLOCK_SIZE + localCol] = a[row * K + block * BLOCK_SIZE + localCol];
                else
                    localA[localRow * BLOCK_SIZE + localCol] = 0;

                if ((block * BLOCK_SIZE + localRow < K) && (col < N))
                    localB[localRow * BLOCK_SIZE + localCol] = b[(block * BLOCK_SIZE + localRow) * N + col];
                else
                    localB[localRow * BLOCK_SIZE + localCol] = 0;
                
                barrier(CLK_LOCAL_MEM_FENCE);

                // Compute partial dot product
                if (row < M && col < N) {
                    #pragma unroll 8
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        sum += localA[localRow * BLOCK_SIZE + k] * localB[k * BLOCK_SIZE + localCol];
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (row < M && col < N)
                c[row * N + col] = sum;
        }";

    public static GPUInfo GetBestGPU()
    {
        if (selectedGPU != null) return selectedGPU;

        ErrorCode error;
        var platforms = Cl.GetPlatformIDs(out error);
        if (error != ErrorCode.Success)
            throw new Exception("No OpenCL platforms found");

        var gpuInfos = new List<GPUInfo>();
        foreach (var platform in platforms)
        {
            var devices = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error);
            if (error == ErrorCode.Success)
            {
                foreach (var device in devices)
                {
                    gpuInfos.Add(new GPUInfo(platform, device));
                }
            }
        }

        if (!gpuInfos.Any())
            throw new Exception("No GPU devices found");

        // Prefer NVIDIA for CUDA support, then AMD, then others
        selectedGPU = gpuInfos.FirstOrDefault(g => g.VendorName.Contains("NVIDIA")) ??
                     gpuInfos.FirstOrDefault(g => g.VendorName.Contains("AMD")) ??
                     gpuInfos.First();

        Console.WriteLine($"Selected GPU: {selectedGPU.DeviceName} from {selectedGPU.VendorName}");
        return selectedGPU;
    }

    public MatrixGPUMultiplication()
    {
        var gpu = GetBestGPU();
        ErrorCode error;
        
        device = gpu.Device;
        context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
        queue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out error);

        var programTemp = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out error);
        // Use vendor-specific optimizations
        string buildOptions = "-cl-mad-enable -cl-fast-relaxed-math";
        if (gpu.VendorName.Contains("AMD"))
            buildOptions += " -cl-amd-device-kernel";
        else if (gpu.VendorName.Contains("NVIDIA"))
            buildOptions += " -cl-nv-maxrregcount=32";

        error = Cl.BuildProgram(programTemp, 1, new[] { device }, buildOptions, null, IntPtr.Zero);

        if (error != ErrorCode.Success)
        {
            byte[] logBuffer = new byte[4096];
            IntPtr returnSize;
            using (var buffer = new InfoBuffer(logBuffer))
            {
                error = Cl.GetProgramBuildInfo(programTemp, device, ProgramBuildInfo.Log,
                    new IntPtr(logBuffer.Length), buffer, out returnSize);

                if (error == ErrorCode.Success)
                {
                    var log = Encoding.ASCII.GetString(logBuffer, 0, Math.Min(logBuffer.Length, (int)returnSize)).TrimEnd('\0');
                    throw new Exception($"OpenCL Program Build Error: {log}");
                }
            }
        }

        program = programTemp;
        kernel = Cl.CreateKernel(program, "matrix_multiply", out error);
    }

    public async Task<Matriz> multiplicar(Matriz m1, Matriz m2)
    {
        if (m1.getCols() != m2.getRows())
        {
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");
        }

        int M = m1.getRows();
        int N = m2.getCols();
        int K = m1.getCols();

        // Use GC.AllocateUninitializedArray for better memory management
        int[] dataM1 = GC.AllocateUninitializedArray<int>(M * K);
        int[] dataM2 = GC.AllocateUninitializedArray<int>(K * N);
        int[] dataResult = GC.AllocateUninitializedArray<int>(M * N);

        // Copy data to arrays (can still be parallelized, but simpler is often better)
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                dataM1[i * K + j] = m1.get(i, j);
            }
        }

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < N; j++)
            {
                dataM2[i * N + j] = m2.get(i, j);
            }
        }


        return await Task.Run(() => { // await moved OUTSIDE unsafe

            ErrorCode error;
            const int workGroupSize = 16; // Must match BLOCK_SIZE in kernel

            unsafe // unsafe block INSIDE Task.Run
            {
                fixed (int* ptrM1 = dataM1)
                fixed (int* ptrM2 = dataM2)
                fixed (int* ptrResult = dataResult)
                {
                    // Create buffers
                    var bufferA = Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                        sizeof(int) * M * K, (IntPtr)ptrM1, out error);
                    var bufferB = Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                        sizeof(int) * K * N, (IntPtr)ptrM2, out error);
                    var bufferC = Cl.CreateBuffer(context, MemFlags.WriteOnly,
                        sizeof(int) * M * N, IntPtr.Zero, out error);

                    // Set kernel arguments for buffers
                    error = Cl.SetKernelArg(kernel, 0, bufferA);
                    error = Cl.SetKernelArg(kernel, 1, bufferB);
                    error = Cl.SetKernelArg(kernel, 2, bufferC);

                    // Set scalar arguments (CORRECTED AGAIN)
                    IntPtr ptrM = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrN = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrK = Marshal.AllocHGlobal(sizeof(int));
                    try
                    {
                        Marshal.WriteInt32(ptrM, M); // Write the VALUE of M
                        Marshal.WriteInt32(ptrN, N); // Write the VALUE of N
                        Marshal.WriteInt32(ptrK, K); // Write the VALUE of K

                        error = Cl.SetKernelArg(kernel, 3, sizeof(int), ptrM); // Pass IntPtr to allocated memory
                        error = Cl.SetKernelArg(kernel, 4, sizeof(int), ptrN);
                        error = Cl.SetKernelArg(kernel, 5, sizeof(int), ptrK);

                        // Set local memory arguments
                        error = Cl.SetKernelArg(kernel, 6, new IntPtr(sizeof(int) * workGroupSize * workGroupSize), IntPtr.Zero);
                        error = Cl.SetKernelArg(kernel, 7, new IntPtr(sizeof(int) * workGroupSize * workGroupSize), IntPtr.Zero);


                        var globalWorkSize = new[] {
                            new IntPtr(((M + workGroupSize - 1) / workGroupSize) * workGroupSize),
                            new IntPtr(((N + workGroupSize - 1) / workGroupSize) * workGroupSize)
                        };
                        var localWorkSize = new[] { new IntPtr(workGroupSize), new IntPtr(workGroupSize) };


                        Event evt;
                        error = Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize, 0, null, out evt);


                        error = Cl.EnqueueReadBuffer(queue, bufferC, OpenCL.Net.Bool.True, IntPtr.Zero,
                            new IntPtr(sizeof(int) * M * N), (IntPtr)ptrResult, 0, null, out evt);
                        error = Cl.Finish(queue);
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(ptrM);  // Free the unmanaged memory!
                        Marshal.FreeHGlobal(ptrN);
                        Marshal.FreeHGlobal(ptrK);
                    }
                    Cl.ReleaseMemObject(bufferA);
                    Cl.ReleaseMemObject(bufferB);
                    Cl.ReleaseMemObject(bufferC);


                } // fixed block ends here

            } // unsafe block ends here

            var result = new Matriz(M, N);
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    result.set(i, j, dataResult[i * N + j]);
                }
            }
            return result;
        });
    }

    public void Dispose()
    {
        if (!disposed)
        {
            Cl.ReleaseKernel(kernel);
            Cl.ReleaseProgram(program);
            Cl.ReleaseCommandQueue(queue);
            Cl.ReleaseContext(context);
            disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~MatrixGPUMultiplication()
    {
        Dispose();
    }
}