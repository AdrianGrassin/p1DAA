using OpenCL.Net;
using System.Runtime.InteropServices;
using System.Text;
using Event = OpenCL.Net.Event;
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;

namespace MatrixProd;

public class MatrixHybridMultiplication : IMatrixMultiplication
{
    private readonly Context context;
    private readonly CommandQueue queue;
    private readonly OpenCL.Net.Kernel kernel;
    private readonly OpenCL.Net.Program program;
    private readonly Device device;
    private bool disposed;
    private bool isGPUWarmedUp;
    
    // Adjusted thresholds based on benchmark data
    private const int CPU_THRESHOLD = 300; // Lowered for better hybrid performance
    private const int GPU_THRESHOLD = 1800; // New threshold for large matrices
    private const int BLOCK_SIZE = 64; // Increased for better GPU utilization
    private readonly MatrixFilMultiplication cpuMultiplier;

    private const string kernelSource = @"
        #define BLOCK_SIZE 64
        #define TILE_SIZE 8

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
            const int row = get_global_id(0);
            const int col = get_global_id(1);
            const int localRow = get_local_id(0);
            const int localCol = get_local_id(1);

            __local int As[BLOCK_SIZE][BLOCK_SIZE];
            __local int Bs[BLOCK_SIZE][BLOCK_SIZE];
            
            int acc = 0;
            
            // Loop over all tiles
            for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
                // Collaborative loading of tiles into local memory
                for (int i = 0; i < BLOCK_SIZE; i += TILE_SIZE) {
                    if (row < M && (t * BLOCK_SIZE + localCol + i) < K)
                        As[localRow][localCol + i] = a[row * K + t * BLOCK_SIZE + localCol + i];
                    if ((t * BLOCK_SIZE + localRow + i) < K && col < N)
                        Bs[localRow + i][localCol] = b[(t * BLOCK_SIZE + localRow + i) * N + col];
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);

                // Compute partial dot product using tiling
                if (row < M && col < N) {
                    for (int k = 0; k < BLOCK_SIZE && (t * BLOCK_SIZE + k) < K; k += TILE_SIZE) {
                        #pragma unroll
                        for (int i = 0; i < TILE_SIZE; i++) {
                            acc += As[localRow][k + i] * Bs[k + i][localCol];
                        }
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (row < M && col < N)
                c[row * N + col] = acc;
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
        error = Cl.BuildProgram(programTemp, 1, new[] { device }, "-cl-mad-enable -cl-fast-relaxed-math", null, IntPtr.Zero);

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

    private async Task WarmupGPU()
    {
        if (isGPUWarmedUp) return;

        // Create small matrices for warmup
        var warmupSize = 32;
        var m1 = new Matrix(warmupSize, warmupSize);
        var m2 = new Matrix(warmupSize, warmupSize);
        m1.SetRandoms();
        m2.SetRandoms();

        // Run a quick multiplication to initialize GPU
        await MultiplyGPU(m1, m2);
        isGPUWarmedUp = true;
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (m1 is not Matrix matrix1) throw new ArgumentException("Matrix must be of type Matrix", nameof(m1));
        if (m2 is not Matrix matrix2) throw new ArgumentException("Matrix must be of type Matrix", nameof(m2));
        
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");

        int size = m1.GetRows();

        // Warmup GPU on first use
        if (!isGPUWarmedUp)
            await WarmupGPU();

        // Choose multiplication strategy based on matrix size
        if (size <= CPU_THRESHOLD)
            return await cpuMultiplier.Multiply(m1, m2);
        else if (size >= GPU_THRESHOLD)
            return await MultiplyGPU(matrix1, matrix2);
        else
            return await MultiplyHybrid(matrix1, matrix2);
    }

    private async Task<IMatrix> MultiplyGPU(Matrix m1, Matrix m2)
    {
        int M = m1.GetRows();
        int N = m2.GetCols();
        int K = m1.GetCols();

        int[] dataM1 = GC.AllocateUninitializedArray<int>(M * K);
        int[] dataM2 = GC.AllocateUninitializedArray<int>(K * N);
        int[] dataResult = GC.AllocateUninitializedArray<int>(M * N);

        // Optimize data transfer by using parallel copy
        Parallel.For(0, M, i =>
        {
            for (int j = 0; j < K; j++)
                dataM1[i * K + j] = m1.Get(i, j);
        });

        Parallel.For(0, K, i =>
        {
            for (int j = 0; j < N; j++)
                dataM2[i * N + j] = m2.Get(i, j);
        });

        return await Task.Run(() =>
        {
            ErrorCode error;
            const int workGroupSize = BLOCK_SIZE;

            unsafe
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

                    IntPtr ptrM = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrN = Marshal.AllocHGlobal(sizeof(int));
                    IntPtr ptrK = Marshal.AllocHGlobal(sizeof(int));
                    try
                    {
                        Marshal.WriteInt32(ptrM, M);
                        Marshal.WriteInt32(ptrN, N);
                        Marshal.WriteInt32(ptrK, K);

                        error = Cl.SetKernelArg(kernel, 3, sizeof(int), ptrM);
                        error = Cl.SetKernelArg(kernel, 4, sizeof(int), ptrN);
                        error = Cl.SetKernelArg(kernel, 5, sizeof(int), ptrK);

                        error = Cl.SetKernelArg(kernel, 6, new IntPtr(sizeof(int) * workGroupSize * workGroupSize), IntPtr.Zero);
                        error = Cl.SetKernelArg(kernel, 7, new IntPtr(sizeof(int) * workGroupSize * workGroupSize), IntPtr.Zero);

                        var globalWorkSize = new[]
                        {
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
                        Marshal.FreeHGlobal(ptrM);
                        Marshal.FreeHGlobal(ptrN);
                        Marshal.FreeHGlobal(ptrK);

                        Cl.ReleaseMemObject(bufferA);
                        Cl.ReleaseMemObject(bufferB);
                        Cl.ReleaseMemObject(bufferC);
                    }
                }
            }

            var result = new Matrix(M, N);
            Parallel.For(0, M, i =>
            {
                for (int j = 0; j < N; j++)
                    result.Set(i, j, dataResult[i * N + j]);
            });

            return result;
        });
    }

    private async Task<IMatrix> MultiplyHybrid(Matrix m1, Matrix m2)
    {
        int M = m1.GetRows();
        int N = m2.GetCols();
        int K = m1.GetCols();
        
        // Split the work between CPU and GPU
        int splitPoint = M / 2;
        
        // Create sub-matrices
        var m1Upper = new Matrix(splitPoint, K);
        var m1Lower = new Matrix(M - splitPoint, K);
        
        // Copy data to sub-matrices in parallel
        Parallel.For(0, splitPoint, i =>
        {
            for (int j = 0; j < K; j++)
                m1Upper.Set(i, j, m1.Get(i, j));
        });
                
        Parallel.For(0, M - splitPoint, i =>
        {
            for (int j = 0; j < K; j++)
                m1Lower.Set(i, j, m1.Get(i + splitPoint, j));
        });

        // Process in parallel
        var gpuTask = MultiplyGPU(m1Upper, m2);
        var cpuTask = cpuMultiplier.Multiply(m1Lower, m2);
        
        await Task.WhenAll(gpuTask, cpuTask);
        
        var gpuResult = await gpuTask;
        var cpuResult = await cpuTask;
        
        // Combine results
        var result = new Matrix(M, N);
        Parallel.For(0, splitPoint, i =>
        {
            for (int j = 0; j < N; j++)
                result.Set(i, j, gpuResult.Get(i, j));
        });
                
        Parallel.For(0, M - splitPoint, i =>
        {
            for (int j = 0; j < N; j++)
                result.Set(i + splitPoint, j, cpuResult.Get(i, j));
        });
        
        return result;
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