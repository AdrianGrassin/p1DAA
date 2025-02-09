// MatrixFilMultiplication.cs
using System.Threading.Tasks;

namespace MatrixProd;

public class MatrixFilMultiplication : MatrixMultiplication
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
            Parallel.For(0, rows, i =>
            {
                var rowCache = new int[innerDim];
                for (int k = 0; k < innerDim; k++)
                {
                    rowCache[k] = m1.get(i, k);
                }

                for (int j = 0; j < cols; j += BLOCK_SIZE)
                {
                    for (int k = 0; k < innerDim; k++)
                    {
                        int rowVal = rowCache[k];
                        int endCol = Math.Min(j + BLOCK_SIZE, cols);

                        for (int b = j; b < endCol; b++)
                        {
                            if (k == 0) result.set(i, b, 0);
                            result.set(i, b, result.get(i, b) + rowVal * m2.get(k, b));
                        }
                    }
                }
            });
        });
        return result;
    }
}