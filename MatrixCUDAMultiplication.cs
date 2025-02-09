using System.Runtime.InteropServices;
using CUDA;

namespace MatrixProd;

public class MatrixCUDAMultiplication : MatrixMultiplication, IDisposable
{
    private readonly CudaContext context;
    private readonly CudaKernel kernel;
    private bool disposed;

    private const string cudaKernelSource = @"
        extern ""C"" __global__ void matrix_multiply(
            const int* A,
            const int* B,
            int* C,
            const int M,
            const int N,
            const int K)
        {
            const int BLOCK_SIZE = 16;
            __shared__ int sharedA[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ int sharedB[BLOCK_SIZE][BLOCK_SIZE];
            
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            int row = by * BLOCK_SIZE + ty;
            int col = bx * BLOCK_SIZE + tx;
            
            int sum = 0;
            
            for (int block = 0; block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
                if (row < M && block * BLOCK_SIZE + tx < K)
                    sharedA[ty][tx] = A[row * K + block * BLOCK_SIZE + tx];
                else
                    sharedA[ty][tx] = 0;
                
                if (block * BLOCK_SIZE + ty < K && col < N)
                    sharedB[ty][tx] = B[(block * BLOCK_SIZE + ty) * N + col];
                else
                    sharedB[ty][tx] = 0;
                
                __syncthreads();
                
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; k++)
                    sum += sharedA[ty][k] * sharedB[k][tx];
                
                __syncthreads();
            }
            
            if (row < M && col < N)
                C[row * N + col] = sum;
        }";

    public MatrixCUDAMultiplication()
    {
        try
        {
            context = new CudaContext();
            var module = context.LoadModuleData(cudaKernelSource);
            kernel = module.GetKernel("matrix_multiply");
        }
        catch (Exception ex)
        {
            throw new Exception("Failed to initialize CUDA. Make sure NVIDIA drivers are installed.", ex);
        }
    }

    public async Task<Matriz> multiplicar(Matriz m1, Matriz m2)
    {
        if (m1.getCols() != m2.getRows())
            throw new ArgumentException("Matrix dimensions are not compatible for multiplication");

        int M = m1.getRows();
        int N = m2.getCols();
        int K = m1.getCols();

        return await Task.Run(() =>
        {
            // Allocate device memory and copy data
            using var deviceA = new CudaDeviceVariable<int>(M * K);
            using var deviceB = new CudaDeviceVariable<int>(K * N);
            using var deviceC = new CudaDeviceVariable<int>(M * N);

            // Copy input matrices to device
            var hostA = new int[M * K];
            var hostB = new int[K * N];
            
            // Copy matrix data to linear arrays
            for (int i = 0; i < M; i++)
                for (int j = 0; j < K; j++)
                    hostA[i * K + j] = m1.get(i, j);

            for (int i = 0; i < K; i++)
                for (int j = 0; j < N; j++)
                    hostB[i * N + j] = m2.get(i, j);

            deviceA.CopyToDevice(hostA);
            deviceB.CopyToDevice(hostB);

            // Configure kernel execution
            const int BLOCK_SIZE = 16;
            var blockSize = new dim3(BLOCK_SIZE, BLOCK_SIZE);
            var gridSize = new dim3(
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            // Launch kernel
            kernel.Launch(
                gridSize, blockSize,
                deviceA.DevicePointer, deviceB.DevicePointer, deviceC.DevicePointer,
                M, N, K
            );

            // Copy result back to host
            var hostC = deviceC.ToArray();

            // Create result matrix
            var result = new Matriz(M, N);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result.set(i, j, hostC[i * N + j]);

            return result;
        });
    }

    public void Dispose()
    {
        if (!disposed)
        {
            kernel?.Dispose();
            context?.Dispose();
            disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~MatrixCUDAMultiplication()
    {
        Dispose();
    }
}