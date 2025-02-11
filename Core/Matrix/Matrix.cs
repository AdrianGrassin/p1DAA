using MatrixProd.Core.Interfaces;
using System.Runtime.CompilerServices;
using System.Threading;

namespace MatrixProd.Core.Matrix;

public class Matrix : IMatrix
{
    private readonly int[] _data;
    private readonly int _rows;
    private readonly int _cols;

    private const int CHUNK_SIZE = 512;

    public int GetRows() => _rows;
    public int GetCols() => _cols;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Get(int row, int col)
    {
        if ((uint)row >= _rows || (uint)col >= _cols)
        {
            throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
        }
        return _data[row * _cols + col];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int row, int col, int value)
    {
        if ((uint)row >= _rows || (uint)col >= _cols)
        {
            throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
        }
        _data[row * _cols + col] = value;
    }

    public bool CompareAndSet(int row, int col, int expected, int value)
    {
        return Interlocked.CompareExchange(ref _data[row * _cols + col], value, expected) == expected;
    }

    public Matrix(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Las dimensiones de la matriz deben ser mayores a cero");
        }
        
        if ((long)rows * cols > Array.MaxLength)
        {
            throw new ArgumentException("Las dimensiones de la matriz son demasiado grandes");
        }
        
        _rows = rows;
        _cols = cols;
        _data = GC.AllocateUninitializedArray<int>(rows * cols);
    }

    public void ProcessInChunks(Action<int, int, int, int> processor)
    {
        int numRowChunks = (_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int numColChunks = (_cols + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (int i = 0; i < numRowChunks; i++)
        {
            int startRow = i * CHUNK_SIZE;
            int rowCount = Math.Min(CHUNK_SIZE, _rows - startRow);

            for (int j = 0; j < numColChunks; j++)
            {
                int startCol = j * CHUNK_SIZE;
                int colCount = Math.Min(CHUNK_SIZE, _cols - startCol);

                processor(startRow, startCol, rowCount, colCount);
            }
        }
    }

    public void SetRandomsInChunks()
    {
        Random rand = new Random();
        ProcessInChunks((startRow, startCol, rowCount, colCount) =>
        {
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < colCount; j++)
                {
                    _data[(startRow + i) * _cols + (startCol + j)] = rand.Next(-100, 100);
                }
            }
        });
    }

    public void SetRandoms()
    {
        if (_rows * _cols > 2250000) // Approximately 1500x1500
        {
            SetRandomsInChunks();
            return;
        }

        Random rand = new Random();
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                _data[i * _cols + j] = rand.Next(-100, 100);
            }
        }
    }

    public override string ToString()
    {
        using var writer = new StringWriter();
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                writer.Write($"{Get(i, j),4} ");
            }
            writer.WriteLine();
        }
        return writer.ToString();
    }
}