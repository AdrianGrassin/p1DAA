/**
 * Interfaz para la multiplicacion de matrices de enteros
 * @author: Adrian Grassin Luis
 * @version: 2.0.0
 * @mail: alu0101349480@ull.edu.es
 */
namespace MatrixProd;

public interface MatrixMultiplication
{
    Task<Matriz> multiplicar(Matriz m1, Matriz m2);
}
