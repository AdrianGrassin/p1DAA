using MatrixProd.Core.Matrix;

namespace MatrixProd.Core.Interfaces;

public interface IMatrixMultiplication : IDisposable
{
    Task<IMatrix> Multiply(IMatrix m1, IMatrix m2);
}

public interface IMatrix
{
    int GetRows();
    int GetCols();
    int Get(int row, int col);
    void Set(int row, int col, int value);
    void SetRandoms();
}