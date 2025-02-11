using MatrixProd.Core.Interfaces;
using System.Runtime.CompilerServices;

namespace MatrixProd.Core.Matrix;

public class RowMatrixMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;
    private const int L1_CACHE_SIZE = 32 * 1024; // Typical L1 cache size
    private const int VECTOR_SIZE = 4; // For potential auto-vectorization
    private const int BLOCK_SIZE = 64; // Increased from 32 for better cache line utilization

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(RowMatrixMultiplication));

        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");

        int M = m1.GetRows();
        int N = m2.GetCols();
        int K = m1.GetCols();
        var result = new Matriz(M, N);

        // Calculate optimal block sizes based on L1 cache
        int blockM = Math.Min(BLOCK_SIZE, M);
        int blockN = Math.Min(BLOCK_SIZE, N);
        int blockK = Math.Min(BLOCK_SIZE, K);

        await Task.Run(() =>
        {
            // Pre-cache the second matrix in blocks for better memory access
            var blockB = new int[blockK * N];

            // Process matrix in blocks to maximize cache utilization
            for (int ii = 0; ii < M; ii += blockM)
            {
                int maxI = Math.Min(ii + blockM, M);
                
                // Cache entire rows of first matrix for current block
                var blockA = new int[(maxI - ii) * K];
                Parallel.For(ii, maxI, i =>
                {
                    for (int k = 0; k < K; k++)
                    {
                        blockA[(i - ii) * K + k] = m1.Get(i, k);
                    }
                });

                for (int jj = 0; jj < N; jj += blockN)
                {
                    int maxJ = Math.Min(jj + blockN, N);

                    for (int kk = 0; kk < K; kk += blockK)
                    {
                        int maxK = Math.Min(kk + blockK, K);

                        // Cache block of second matrix
                        for (int k = kk; k < maxK; k++)
                        {
                            for (int j = jj; j < maxJ; j++)
                            {
                                blockB[(k - kk) * blockN + (j - jj)] = m2.Get(k, j);
                            }
                        }

                        // Process block with optimized inner loop
                        Parallel.For(ii, maxI, i =>
                        {
                            int rowIdx = (i - ii) * K;
                            for (int j = jj; j < maxJ; j++)
                            {
                                int sum = 0;
                                int idx = 0;
                                
                                // Manual loop unrolling for better vectorization
                                for (int k = kk; k < maxK - (VECTOR_SIZE - 1); k += VECTOR_SIZE)
                                {
                                    sum += blockA[rowIdx + k] * blockB[idx + (j - jj)]
                                        + blockA[rowIdx + k + 1] * blockB[idx + blockN + (j - jj)]
                                        + blockA[rowIdx + k + 2] * blockB[idx + 2 * blockN + (j - jj)]
                                        + blockA[rowIdx + k + 3] * blockB[idx + 3 * blockN + (j - jj)];
                                    idx += VECTOR_SIZE * blockN;
                                }

                                // Handle remaining elements
                                for (int k = maxK - ((maxK - kk) % VECTOR_SIZE); k < maxK; k++)
                                {
                                    sum += blockA[rowIdx + k] * blockB[(k - kk) * blockN + (j - jj)];
                                }

                                // Accumulate into result
                                if (kk == 0)
                                    result.Set(i, j, sum);
                                else
                                    result.Set(i, j, result.Get(i, j) + sum);
                            }
                        });
                    }
                }
            }
        });

        return result;
    }

    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}