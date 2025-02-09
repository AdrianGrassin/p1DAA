// MatrixColMultiplication.cs
using System.Threading.Tasks;

namespace MatrixProd;

public class MatrixColMultiplication : MatrixMultiplication
{
    public async Task<Matriz> multiplicar(Matriz m1, Matriz m2)
    {
        if (m1.getCols() != m2.getRows())
        {
            throw new ArgumentException("Las dimensiones de las matrices no son compatibles para la multiplicación");
        }

        int rows = m1.getRows();
        int cols = m2.getCols();
        int innerDim = m1.getCols();
        Matriz result = new(rows, cols);

        const int BLOCK_SIZE = 32; // Cache-friendly block size

        await Task.Run(() =>
        {
            // Pre-cache columns of second matrix for better memory access
            var colCache = new int[cols][];
            Parallel.For(0, cols, j =>
            {
                colCache[j] = new int[innerDim];
                for (int k = 0; k < innerDim; k++)
                {
                    colCache[j][k] = m2.get(k, j);
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
                            sum += m1.get(i, k) * col[k];
                        }
                        result.set(i, b, sum);
                    }
                }
            });
        });

        return result;
    }
}