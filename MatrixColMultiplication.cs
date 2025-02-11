// MatrixColMultiplication.cs
using System.Threading.Tasks;
using MatrixProd.Core.Matrix;
using MatrixProd.Core.Interfaces;

namespace MatrixProd;

public class MatrixColMultiplication : IMatrixMultiplication, IDisposable
{
    private bool _disposed;

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MatrixColMultiplication));

        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");

        var result = new Matrix(m1.GetRows(), m2.GetCols());

        await Task.Run(() =>
        {
            for (int j = 0; j < m2.GetCols(); j++)
            {
                for (int i = 0; i < m1.GetRows(); i++)
                {
                    int sum = 0;
                    for (int k = 0; k < m1.GetCols(); k++)
                    {
                        sum += m1.Get(i, k) * m2.Get(k, j);
                    }
                    result.Set(i, j, sum);
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