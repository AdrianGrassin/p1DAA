// MatrixFilMultiplication.cs
using System.Threading.Tasks;
using MatrixProd.Core.Interfaces;

namespace MatrixProd.Core.Matrix;

public class MatrixFilMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;

    public Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MatrixFilMultiplication));
        
        if (m1.GetCols() != m2.GetRows())
        {
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");
        }

        int rows = m1.GetRows();
        int cols = m2.GetCols();
        int innerDim = m1.GetCols();
        var result = new Matrix(rows, cols);

        return Task.Run(() =>
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < innerDim; k++)
                    {
                        sum += m1.Get(i, k) * m2.Get(k, j);
                    }
                    result.Set(i, j, sum);
                }
            }
            return result as IMatrix;
        });
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}