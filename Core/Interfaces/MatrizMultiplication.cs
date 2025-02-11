using MatrixProd.Core.Matrix;

namespace MatrixProd.Core.Interfaces;

public abstract class MatrixMultiplicationBase : IMatrixMultiplication
{
    private bool _disposed;

    public abstract Task<IMatrix> Multiply(IMatrix m1, IMatrix m2);

    public void Dispose()
    {
        if (!_disposed)
        {
            // Release any unmanaged resources here
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    protected virtual void DisposeUnmanaged()
    {
        // Override in derived classes if they need to dispose unmanaged resources
    }

    protected void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().Name);
        }
    }
}
