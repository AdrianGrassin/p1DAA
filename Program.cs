// Program.cs

using System.Diagnostics;
using System.Globalization;
using OpenCL.Net;
using System.Threading.Tasks;

class Program
{
    static readonly int[] CSV_SIZES = { 100, 500, 800, 1000, 1500, 2000, 2500 };
    static readonly int ITERATIONS = 5;

    static async Task<double> RunBenchmark(int size, string method)
    {
        MatrixProd.Matriz m1 = new(size, size);
        MatrixProd.Matriz m2 = new(size, size);

        m1.setRandoms();
        m2.setRandoms();

        long totalTime = 0;
        Stopwatch stopwatch = new Stopwatch();

        using (var gpuMultiplication = method == "g" ? new MatrixProd.MatrixGPUMultiplication() : null)
        using (var hybridMultiplication = method == "h" ? new MatrixProd.MatrixHybridMultiplication() : null)
        {
            for (int i = 0; i < ITERATIONS; i++)
            {
                stopwatch.Restart();
                //The await must be included here
                switch (method)
                {
                    case "f":
                        var matrixFilMultiplication = new MatrixProd.MatrixFilMultiplication();
                        await matrixFilMultiplication.multiplicar(m1, m2);
                        break;
                    case "c":
                        var matrixColMultiplication = new MatrixProd.MatrixColMultiplication();
                        await matrixColMultiplication.multiplicar(m1, m2);
                        break;
                    case "g":
                        if (gpuMultiplication != null)
                            await gpuMultiplication.multiplicar(m1, m2);
                        break;
                    case "h":
                        if (hybridMultiplication != null)
                            await hybridMultiplication.multiplicar(m1, m2);
                        break;
                }

                stopwatch.Stop();
                totalTime += stopwatch.ElapsedMilliseconds;
            }
        }

        return totalTime / (double)ITERATIONS;
    }

    static async Task GenerateCSVFiles()
    {
        string filename = "results_comparison.csv";
        var culture = CultureInfo.InvariantCulture;

        using (StreamWriter writer = new StreamWriter(filename))
        {
            writer.WriteLine("Tamaño,Tiempo Filas (ms),Tiempo Columnas (ms),Tiempo GPU (ms), Tiempo Hibrido (ms)");

            foreach (int size in CSV_SIZES)
            {
                double timeRows = await RunBenchmark(size, "f");
                double timeCols = await RunBenchmark(size, "c");
                double timeGPU = await RunBenchmark(size, "g");
                double timeHybrid = await RunBenchmark(size, "h");

                writer.WriteLine($"{size},{timeRows.ToString("F2", culture)},{timeCols.ToString("F2", culture)},{timeGPU.ToString("F2", culture)},{timeHybrid.ToString("F2", culture)}");
                Console.WriteLine($"Completado tamaño {size}x{size}:");
                Console.WriteLine($"  Filas: {timeRows.ToString("F2", culture)}ms");
                Console.WriteLine($"  Columnas: {timeCols.ToString("F2", culture)}ms");
                Console.WriteLine($"  GPU: {timeGPU.ToString("F2", culture)}ms");
                Console.WriteLine($"  Hybrid: {timeHybrid.ToString("F2", culture)}ms");
            }
        }

        Console.WriteLine($"\nResultados comparativos guardados en {filename}");
    }

    static async Task RunTest(int size, string method)
    {
        double avgTime = await RunBenchmark(size, method);
        string methodName = method switch
        {
            "f" => "CPU Optimized",
            "c" => "Columns",
            "g" => "GPU Optimized",
            "h" => "Hybrid CPU-GPU",
            _ => throw new ArgumentException("Método no válido")
        };
        Console.WriteLine($"Size: {size}x{size}, Method: {methodName}, Average time: {avgTime:F2}ms");
    }

    static async Task Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.WriteLine("Usage: dotnet run <size/csv> <f/c/g/h>");
            Console.WriteLine("     size: integer for matrix size");
            Console.WriteLine("     csv: generates comparative results for multiple sizes");
            Console.WriteLine("     f: CPU optimized multiplication");
            Console.WriteLine("     c: column multiplication");
            Console.WriteLine("     g: GPU optimized multiplication");
            Console.WriteLine("     h: hybrid CPU-GPU multiplication");
            return;
        }

        try
        {
            if (args[0].ToLower() == "csv")
            {
                await GenerateCSVFiles();
            }
            else
            {
                string method = args[1].ToLower();
                if (!new[] { "f", "c", "g", "h" }.Contains(method))
                {
                    Console.WriteLine("Method must be 'f' (CPU), 'c' (columns), 'g' (GPU) or 'h' (hybrid)");
                    return;
                }

                if (!int.TryParse(args[0], out int size) || size <= 0)
                {
                    Console.WriteLine("Size must be a positive integer");
                    return;
                }
                await RunTest(size, method);
            }
        }
        catch (Exception ex) when (ex.Message.Contains("OpenCL"))
        {
            Console.WriteLine($"OpenCL Error: {ex.Message}");
            Console.WriteLine("Make sure you have AMD drivers and OpenCL runtime installed");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}