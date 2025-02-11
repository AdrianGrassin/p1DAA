namespace MatrixProd.Core.Interfaces;

public interface IGPUDevice : IDisposable
{
    string VendorName { get; }
    string DeviceName { get; }
    bool IsAvailable { get; }
    Task Initialize();
}

public interface IGPUMatrixMultiplication : IMatrixMultiplication, IDisposable
{
    IGPUDevice Device { get; }
    new Task<IMatrix> Multiply(IMatrix m1, IMatrix m2);
    Task<IMatrix> Multiply(IMatrix m1, IMatrix m2, bool useColumns);
}