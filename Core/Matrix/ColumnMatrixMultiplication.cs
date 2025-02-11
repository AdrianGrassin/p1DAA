using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using MatrixProd.Core.Interfaces;

namespace MatrixProd.Core.Matrix;

public class ColumnMatrixMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;
    private const int BLOCK_SIZE = 32;
    private const int MIN_PARALLEL_SIZE = 64;

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ColumnMatrixMultiplication));
        
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Matrix dimensions are not compatible for multiplication");

        var result = new Matrix(m1.GetRows(), m2.GetCols());
        int M = m1.GetRows();
        int N = m2.GetCols();
        int K = m1.GetCols();

        // For small matrices, use simple sequential multiplication
        if (M < MIN_PARALLEL_SIZE || N < MIN_PARALLEL_SIZE)
        {
            await Task.Run(() => MultiplySequential(m1, m2, result));
            return result;
        }

        // For larger matrices, use blocked parallel multiplication with column-major access
        await Task.Run(() =>
        {
            int numThreads = Math.Min(Environment.ProcessorCount,
                Math.Max(1, Math.Min(M, Math.Min(N, K)) / BLOCK_SIZE));

            // Process columns in parallel
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = numThreads }, j =>
            {
                for (int kk = 0; kk < K; kk += BLOCK_SIZE)
                {
                    for (int ii = 0; ii < M; ii += BLOCK_SIZE)
                    {
                        int kEnd = Math.Min(kk + BLOCK_SIZE, K);
                        int iEnd = Math.Min(ii + BLOCK_SIZE, M);

                        // Process column blocks
                        for (int k = kk; k < kEnd; k++)
                        {
                            int m2kj = m2.Get(k, j); // Cache this value as it's used multiple times
                            
                            // Use SIMD for inner loop when possible
                            if (Avx2.IsSupported && (iEnd - ii) >= 8)
                            {
                                for (int i = ii; i <= iEnd - 8; i += 8)
                                {
                                    var va = Vector256.Create(
                                        m1.Get(i, k),
                                        m1.Get(i + 1, k),
                                        m1.Get(i + 2, k),
                                        m1.Get(i + 3, k),
                                        m1.Get(i + 4, k),
                                        m1.Get(i + 5, k),
                                        m1.Get(i + 6, k),
                                        m1.Get(i + 7, k)
                                    );
                                    var vb = Vector256.Create(m2kj);
                                    var current = Vector256.Create(
                                        result.Get(i, j),
                                        result.Get(i + 1, j),
                                        result.Get(i + 2, j),
                                        result.Get(i + 3, j),
                                        result.Get(i + 4, j),
                                        result.Get(i + 5, j),
                                        result.Get(i + 6, j),
                                        result.Get(i + 7, j)
                                    );
                                    var product = Avx2.Add(current, Avx2.MultiplyLow(va, vb));
                                    
                                    // Update results
                                    result.Set(i, j, product.GetElement(0));
                                    result.Set(i + 1, j, product.GetElement(1));
                                    result.Set(i + 2, j, product.GetElement(2));
                                    result.Set(i + 3, j, product.GetElement(3));
                                    result.Set(i + 4, j, product.GetElement(4));
                                    result.Set(i + 5, j, product.GetElement(5));
                                    result.Set(i + 6, j, product.GetElement(6));
                                    result.Set(i + 7, j, product.GetElement(7));
                                }
                            }
                            else if (Sse2.IsSupported && (iEnd - ii) >= 4)
                            {
                                for (int i = ii; i <= iEnd - 4; i += 4)
                                {
                                    var va = Vector128.Create(
                                        m1.Get(i, k),
                                        m1.Get(i + 1, k),
                                        m1.Get(i + 2, k),
                                        m1.Get(i + 3, k)
                                    );
                                    var vb = Vector128.Create(m2kj);
                                    var current = Vector128.Create(
                                        result.Get(i, j),
                                        result.Get(i + 1, j),
                                        result.Get(i + 2, j),
                                        result.Get(i + 3, j)
                                    );
                                    var product = Sse2.Add(current, Sse41.MultiplyLow(va, vb));
                                    
                                    // Update results
                                    result.Set(i, j, product.GetElement(0));
                                    result.Set(i + 1, j, product.GetElement(1));
                                    result.Set(i + 2, j, product.GetElement(2));
                                    result.Set(i + 3, j, product.GetElement(3));
                                }
                            }
                            
                            // Handle remaining elements
                            for (int i = ii; i < iEnd; i++)
                            {
                                int current = result.Get(i, j);
                                result.Set(i, j, current + m1.Get(i, k) * m2kj);
                            }
                        }
                    }
                }
            });
        });

        return result;
    }

    private void MultiplySequential(IMatrix m1, IMatrix m2, Matrix result)
    {
        int N = m2.GetCols();
        int M = m1.GetRows();
        int K = m1.GetCols();
        
        // Pre-calculate columns of m2 to improve cache locality
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                int m2kj = m2.Get(k, j);
                for (int i = 0; i < M; i++)
                {
                    int currentValue = result.Get(i, j);
                    result.Set(i, j, currentValue + m1.Get(i, k) * m2kj);
                }
            }
        }
    }

    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}