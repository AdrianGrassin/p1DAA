/**
 * Clase que representa una matriz de enteros
 * @author: Adrian Grassin Luis
 * @version: 2.1.0
 * @mail: alu0101349480@ull.edu.es
 * 
 * @brief: Clase optimizada que representa una matriz de enteros usando un array unidimensional
 *         para mejor localidad de memoria y rendimiento.
 **/
using System.Runtime.CompilerServices;
using MatrixProd.Core.Interfaces;

namespace MatrixProd.Core.Matrix;

public class Matriz : IMatrix
{
  private readonly int[] _data;
  private readonly int _rows;
  private readonly int _cols;
  private const int CHUNK_SIZE = 512; // Reduced from 1024
  private const int MAX_PARALLEL_DEGREE = 2; // Limit parallel operations
  private const int MAX_SAFE_DIMENSION = 5000; // Maximum safe matrix dimension

  public int GetRows() => _rows;
  public int GetCols() => _cols;

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public int Get(int row, int col) {
    if ((uint)row >= _rows || (uint)col >= _cols) {
      throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
    }
    var index = row * _cols + col;
    // Add prefetch hint for the next elements without using deprecated CER
    if (index + 64 < _data.Length) {
      RuntimeHelpers.EnsureSufficientExecutionStack();
      _ = _data[index + 64]; // Simple prefetch hint
    }
    return _data[index];
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void Set(int row, int col, int value) {
    if ((uint)row >= _rows || (uint)col >= _cols) {
      throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
    }
    _data[row * _cols + col] = value;
  }

  public Matriz(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
      throw new ArgumentException("Las dimensiones de la matriz deben ser mayores a cero");
    }
    
    if (rows > MAX_SAFE_DIMENSION || cols > MAX_SAFE_DIMENSION) {
      throw new ArgumentException($"Matrix dimension exceeds safe limit of {MAX_SAFE_DIMENSION}");
    }
    
    if ((long)rows * cols > Array.MaxLength) {
      throw new ArgumentException("Las dimensiones de la matriz son demasiado grandes");
    }
    
    _rows = rows;
    _cols = cols;
    // Use AllocateUninitializedArray for better memory alignment
    _data = GC.AllocateUninitializedArray<int>(rows * cols, true);
  }

  public void SetRandoms() {
    // Use hardware random number generator if available
    var random = Random.Shared;
    
    // For small matrices, use sequential initialization
    if (_data.Length <= CHUNK_SIZE * 2) {
      for (int i = 0; i < _data.Length; i++) {
        _data[i] = random.Next(0, 100);
      }
      return;
    }

    // For larger matrices, use parallel initialization with cache-friendly access
    int numThreads = Math.Min(Environment.ProcessorCount * 3/4, 8); // Limit max threads
    var options = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
    
    // Calculate chunk size based on cache line size (typical 64 bytes = 16 ints)
    const int CACHE_LINE_INTS = 16;
    int chunkSize = Math.Max(CHUNK_SIZE, ((_data.Length + numThreads - 1) / numThreads + CACHE_LINE_INTS - 1) & ~(CACHE_LINE_INTS - 1));
    
    Parallel.For(0, (_data.Length + chunkSize - 1) / chunkSize, options, chunkIndex => {
      int start = chunkIndex * chunkSize;
      int end = Math.Min(start + chunkSize, _data.Length);
      var localRandom = new Random(random.Next());
      
      // Process chunk with cache line alignment
      for (int i = start; i < end; i += CACHE_LINE_INTS) {
        int count = Math.Min(CACHE_LINE_INTS, end - i);
        for (int j = 0; j < count; j++) {
          _data[i + j] = localRandom.Next(0, 100);
        }
      }
    });
  }

  public override string ToString() {
    using var writer = new StringWriter();
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        writer.Write($"{Get(i, j),4} ");
      }
      writer.WriteLine();
    }
    return writer.ToString();
  }
}
