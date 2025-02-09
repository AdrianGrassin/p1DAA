// MatrixHybridMultiplication.cs

using OpenCL.Net;
using System.Runtime.InteropServices;
using System.Text;
using Event = OpenCL.Net.Event;

namespace MatrixProd;

public class MatrixHybridMultiplication : MatrixMultiplication, IDisposable
{
    private readonly Context context;
    private readonly CommandQueue queue;
    private readonly OpenCL.Net.Kernel kernel;
    private readonly OpenCL.Net.Program program;
    private readonly Device device;
    private bool disposed;
    private const int CPU_THRESHOLD = 500; // Size threshold for CPU processing
    private const int BLOCK_SIZE = 32; // Cache-friendly block size for CPU
    private readonly MatrixFilMultiplication cpuMultiplier;

    private const string kernelSource = @"
        #define BLOCK_SIZE 16

        __kernel void matrix_multiply(
            __global const int* a,
            __global const int* b,
            __global int* c,
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
            
            // Loop over all blocks
            for (int block = 0; block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
                // Load data into local memory
                if ((row < M) && (block * BLOCK_SIZE + localCol < K))
                    localA[localRow * BLOCK_SIZE + localCol] = a[row * K + block * BLOCK_SIZE + localCol];
                if ((block * BLOCK_SIZE + localRow < K) && (col < N))
                    localB[localRow * BLOCK_SIZE + localCol] = b[(block * BLOCK_SIZE + localRow) * N + col];
                
                barrier(CLK_LOCAL_MEM_FENCE);

                // Compute partial dot product
                if (row < M && col < N) {
                    for (int k = 0; k < BLOCK_SIZE && (block * BLOCK_SIZE + k) < K; k++)
                        sum += localA[localRow * BLOCK_SIZE + k] * localB[k * BLOCK_SIZE + localCol];
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (row < M && col < N)
                c[row * N + col] = sum;
        }";

    public MatrixHybridMultiplication()
    {
        cpuMultiplier = new MatrixFilMultiplication();

        ErrorCode error;
        var platform = Cl.GetPlatformIDs(out error).First();
        device = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error).First();
        context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
        queue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out error);

        var programTemp = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out error);
        error = Cl.BuildProgram(programTemp, 1, new[] { device }, string.Empty, null, IntPtr.Zero);

        if (error != ErrorCode.Success)
        {
            byte[] logBuffer = new byte[4096];
            var buffer = new InfoBuffer(logBuffer);
            try
            {
                IntPtr returnSize;
                error = Cl.GetProgramBuildInfo(programTemp, device, ProgramBuildInfo.Log,
                    new IntPtr(logBuffer.Length), buffer, out returnSize);

                if (error == ErrorCode.Success)
                {
                    var log = Encoding.ASCII.GetString(logBuffer, 0, Math.Min(logBuffer.Length, (int)returnSize)).TrimEnd('\0');
                    throw new Exception($"OpenCL Program Build Error: {log}");
                }
            }
            finally
            {
                buffer.Dispose();
            }
        }

        program = programTemp;
        kernel = Cl.CreateKernel(program, "matrix_multiply", out error);
    }


    //CS0738 correction
    public async Task<Matriz> multiplicar(Matriz m1, Matriz m2)
    {
        // Use CPU implementation for small matrices
        if (m1.getRows() <= CPU_THRESHOLD || m2.getCols() <= CPU_THRESHOLD)
        {
            return await cpuMultiplier.multiplicar(m1, m2); // Corrected await
        }

        if (m1.getCols() != m2.getRows())
        {
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");
        }

        int M = m1.getRows();
        int N = m2.getCols();
        int K = m1.getCols();

         // Allocate outside the unsafe context
        int[] dataM1 = GC.AllocateUninitializedArray<int>(M * K);
        int[] dataM2 = GC.AllocateUninitializedArray<int>(K * N);
        int[] dataResult = GC.AllocateUninitializedArray<int>(M * N);
         // Use regular loops for data copy (can still be parallelized if needed)
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

        return await Task.Run(() => {  // await moved OUTSIDE unsafe

            ErrorCode error;
            const int workGroupSize = 16; // Must match BLOCK_SIZE in kernel

            unsafe  // unsafe block INSIDE Task.Run
            {
                fixed (int* ptrM1 = dataM1)
                fixed (int* ptrM2 = dataM2)
                fixed (int* ptrResult = dataResult)
                {
                    var bufferA = Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                        sizeof(int) * M * K, (IntPtr)ptrM1, out error);
                    var bufferB = Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                        sizeof(int) * K * N, (IntPtr)ptrM2, out error);
                    var bufferC = Cl.CreateBuffer(context, MemFlags.WriteOnly,
                        sizeof(int) * M * N, IntPtr.Zero, out error);

                    error = Cl.SetKernelArg(kernel, 0, bufferA);
                    error = Cl.SetKernelArg(kernel, 1, bufferB);
                    error = Cl.SetKernelArg(kernel, 2, bufferC);

                    // Pass scalar arguments directly as values, not addresses.
                    IntPtr ptrM = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrN = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrK = Marshal.AllocHGlobal(sizeof(int));
                    try {
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
                    finally {
                        Marshal.FreeHGlobal(ptrM);  // Free the unmanaged memory!
                        Marshal.FreeHGlobal(ptrN);
                        Marshal.FreeHGlobal(ptrK);
                    }

                    Cl.ReleaseMemObject(bufferA);
                    Cl.ReleaseMemObject(bufferB);
                    Cl.ReleaseMemObject(bufferC);
                }

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

    ~MatrixHybridMultiplication()
    {
        Dispose();
    }
}