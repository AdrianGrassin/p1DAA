using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;

namespace MatrixProd.GPU.AMD;

public class AmdMatrixMultiplication : IGPUMatrixMultiplication
{
    private readonly AmdDevice _device;
    private bool _disposed;

    public IGPUDevice Device => _device;

    public AmdMatrixMultiplication(AmdDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(AmdMatrixMultiplication));
        return await Multiply(m1, m2, false); // Default to row-based multiplication
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2, bool useColumns)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(AmdMatrixMultiplication));
        
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Matrix dimensions are not compatible for multiplication");
            
        // Use the AMD implementation
        using var amdImpl = new AmdImplementation();
        return await amdImpl.Multiply(m1, m2, useColumns);
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