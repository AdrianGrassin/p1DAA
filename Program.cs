// Program.cs

using System.Diagnostics;
using System.Globalization;
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

        using (var multiplier = MatrixProd.MatrixMultiplicationFactory.CreateMultiplier(method))
        {
            for (int i = 0; i < ITERATIONS; i++)
            {
                stopwatch.Restart();
                await multiplier.multiplicar(m1, m2);
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

                writer.WriteLine($"{size},{timeRows:F2},{timeCols:F2},{timeGPU:F2},{timeHybrid:F2}");
                Console.WriteLine($"Completado tamaño {size}x{size}:");
                Console.WriteLine($"  Filas: {timeRows:F2}ms");
                Console.WriteLine($"  Columnas: {timeCols:F2}ms");
                Console.WriteLine($"  GPU: {timeGPU:F2}ms");
                Console.WriteLine($"  Hybrid: {timeHybrid:F2}ms");
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
        try
        {
            // Download GPU-specific code if needed
            if (!File.Exists("MatrixGPUMultiplication.cs"))
            {
                Console.WriteLine("Downloading GPU-specific optimizations...");
                await MatrixProd.MatrixMultiplicationFactory.DownloadAndSetupGPUCode();
            }

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
        catch (NotImplementedException)
        {
            Console.WriteLine("GPU multiplication is not available. Please run the installer first to download GPU-specific code.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}