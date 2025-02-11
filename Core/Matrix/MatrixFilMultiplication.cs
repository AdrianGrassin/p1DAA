// MatrixFilMultiplication.cs
using System.Threading.Tasks;
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace MatrixProd.Core.Matrix;

public class MatrixFilMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;
    private const int BLOCK_SIZE = 32;  // Optimal block size for L1 cache
    private const int MIN_PARALLEL_SIZE = 64;  // Minimum matrix size for parallelization

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MatrixFilMultiplication));
        
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

        // For larger matrices, use blocked parallel multiplication
        await Task.Run(() =>
        {
            // Calculate optimal number of threads based on matrix size and CPU cores
            int numThreads = Math.Min(Environment.ProcessorCount,
                Math.Max(1, Math.Min(M, Math.Min(N, K)) / BLOCK_SIZE));

            Parallel.For(0, M, new ParallelOptions { MaxDegreeOfParallelism = numThreads }, i =>
            {
                for (int jj = 0; jj < N; jj += BLOCK_SIZE)
                {
                    for (int kk = 0; kk < K; kk += BLOCK_SIZE)
                    {
                        int jEnd = Math.Min(jj + BLOCK_SIZE, N);
                        int kEnd = Math.Min(kk + BLOCK_SIZE, K);

                        for (int j = jj; j < jEnd; j++)
                        {
                            int sum = 0;
                            int k;

                            // Use SIMD if available and aligned
                            if (Avx2.IsSupported && (kEnd - kk) >= 8)
                            {
                                var vsum = Vector256<int>.Zero;
                                for (k = kk; k <= kEnd - 8; k += 8)
                                {
                                    var va = Vector256.Create(
                                        m1.Get(i, k),
                                        m1.Get(i, k + 1),
                                        m1.Get(i, k + 2),
                                        m1.Get(i, k + 3),
                                        m1.Get(i, k + 4),
                                        m1.Get(i, k + 5),
                                        m1.Get(i, k + 6),
                                        m1.Get(i, k + 7)
                                    );
                                    var vb = Vector256.Create(
                                        m2.Get(k, j),
                                        m2.Get(k + 1, j),
                                        m2.Get(k + 2, j),
                                        m2.Get(k + 3, j),
                                        m2.Get(k + 4, j),
                                        m2.Get(k + 5, j),
                                        m2.Get(k + 6, j),
                                        m2.Get(k + 7, j)
                                    );
                                    vsum = Avx2.Add(vsum, Avx2.MultiplyLow(va, vb));
                                }
                                var lowSum = vsum.GetLower();
                                var highSum = vsum.GetUpper();
                                sum += lowSum.GetElement(0) + lowSum.GetElement(1) + lowSum.GetElement(2) + lowSum.GetElement(3) +
                                      highSum.GetElement(0) + highSum.GetElement(1) + highSum.GetElement(2) + highSum.GetElement(3);
                            }
                            else if (Sse2.IsSupported && (kEnd - kk) >= 4)
                            {
                                var vsum = Vector128<int>.Zero;
                                for (k = kk; k <= kEnd - 4; k += 4)
                                {
                                    var va = Vector128.Create(
                                        m1.Get(i, k),
                                        m1.Get(i, k + 1),
                                        m1.Get(i, k + 2),
                                        m1.Get(i, k + 3)
                                    );
                                    var vb = Vector128.Create(
                                        m2.Get(k, j),
                                        m2.Get(k + 1, j),
                                        m2.Get(k + 2, j),
                                        m2.Get(k + 3, j)
                                    );
                                    vsum = Sse2.Add(vsum, Sse41.MultiplyLow(va, vb));
                                }
                                sum += vsum.GetElement(0) + vsum.GetElement(1) + vsum.GetElement(2) + vsum.GetElement(3);
                            }
                            else
                            {
                                k = kk;
                            }

                            // Handle remaining elements
                            for (; k < kEnd; k++)
                            {
                                sum += m1.Get(i, k) * m2.Get(k, j);
                            }

                            // Atomic add to result
                            int currentValue;
                            do
                            {
                                currentValue = result.Get(i, j);
                            } while (!result.CompareAndSet(i, j, currentValue, currentValue + sum));
                        }
                    }
                }
            });
        });

        return result;
    }

    private void MultiplySequential(IMatrix m1, IMatrix m2, Matrix result)
    {
        for (int i = 0; i < m1.GetRows(); i++)
        {
            for (int j = 0; j < m2.GetCols(); j++)
            {
                int sum = 0;
                for (int k = 0; k < m1.GetCols(); k++)
                {
                    sum += m1.Get(i, k) * m2.Get(k, j);
                }
                result.Set(i, j, sum);
            }
        }
    }

    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}