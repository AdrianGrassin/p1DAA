using MatrixProd.Core.Interfaces;

namespace MatrixProd.Core.Matrix;

public class ColumnMatrixMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ColumnMatrixMultiplication));

        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");

        int rows = m1.GetRows();
        int cols = m2.GetCols();
        int innerDim = m1.GetCols();
        Matrix resultMatrix = new(rows, cols);
        IMatrix result = resultMatrix;

        await Task.Run(() =>
        {
            // Pre-cache columns of second matrix for better memory access
            var colCache = new int[cols][];
            Parallel.For(0, cols, j =>
            {
                colCache[j] = new int[innerDim];
                for (int k = 0; k < innerDim; k++)
                {
                    colCache[j][k] = m2.Get(k, j);
                }
            });

            // Column-wise multiplication with blocking
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j += BLOCK_SIZE)
                {
                    int endCol = Math.Min(j + BLOCK_SIZE, cols);
                    for (int b = j; b < endCol; b++)
                    {
                        int sum = 0;
                        var col = colCache[b];
                        for (int k = 0; k < innerDim; k++)
                        {
                            sum += m1.Get(i, k) * col[k];
                        }
                        result.Set(i, b, sum);
                    }
                }
            });
        });

        return result;
    }

    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private const int BLOCK_SIZE = 32;
}