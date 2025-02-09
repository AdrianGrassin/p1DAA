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

namespace MatrixProd;

public class Matriz {
  private readonly int[] _data;
  private readonly int _rows;
  private readonly int _cols;

  public int getRows() => _rows;
  public int getCols() => _cols;

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public int get(int row, int col) {
    if ((uint)row >= _rows || (uint)col >= _cols) {
      throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
    }
    return _data[row * _cols + col];
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void set(int row, int col, int value) {
    if ((uint)row >= _rows || (uint)col >= _cols) {
      throw new ArgumentOutOfRangeException($"Índices fuera de rango: [{row}, {col}]");
    }
    _data[row * _cols + col] = value;
  }

  public Matriz(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
      throw new ArgumentException("Las dimensiones de la matriz deben ser mayores a cero");
    }
    
    if ((long)rows * cols > Array.MaxLength) {
      throw new ArgumentException("Las dimensiones de la matriz son demasiado grandes");
    }
    
    _rows = rows;
    _cols = cols;
    _data = GC.AllocateUninitializedArray<int>(rows * cols);
  }

  public void setRandoms() {
    const int chunkSize = 1024;
    var random = new Random();
    
    Parallel.For(0, (_data.Length + chunkSize - 1) / chunkSize, chunkIndex => {
      int start = chunkIndex * chunkSize;
      int length = Math.Min(chunkSize, _data.Length - start);
      var localRandom = new Random(random.Next()); // Thread-safe random number generation
      
      Span<int> chunk = _data.AsSpan(start, length);
      for (int i = 0; i < length; i++) {
        chunk[i] = localRandom.Next(0, 100);
      }
    });
  }

  public override string ToString() {
    using var writer = new StringWriter();
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        writer.Write($"{get(i, j),4} ");
      }
      writer.WriteLine();
    }
    return writer.ToString();
  }
}
