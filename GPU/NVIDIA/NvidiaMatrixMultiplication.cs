using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;

namespace MatrixProd.GPU.NVIDIA;

public class NvidiaMatrixMultiplication : IGPUMatrixMultiplication
{
    private readonly NvidiaDevice _device;
    private bool _disposed;

    public IGPUDevice Device => _device;

    public NvidiaMatrixMultiplication(NvidiaDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NvidiaMatrixMultiplication));
        return await Multiply(m1, m2, false); // Default to row-based multiplication
    }

    public async Task<IMatrix> Multiply(IMatrix m1, IMatrix m2, bool useColumns)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NvidiaMatrixMultiplication));
        
        if (m1.GetCols() != m2.GetRows())
            throw new ArgumentException("Matrix dimensions are not compatible for multiplication");
            
        // Use the NVIDIA implementation
        using var nvidiaImpl = new NvidiaImplementation();
        return await nvidiaImpl.Multiply(m1, m2);
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