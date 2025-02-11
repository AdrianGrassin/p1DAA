// Program.cs

using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks; // Add missing using
using MatrixProd; // Ensures MatrixMultiplicationFactory is in scope.
using MatrixProd.Core.Interfaces;
using MatrixProd.Core.Matrix;

namespace MatrixProd
{
    public interface IMatrixMultiplication : IDisposable
    {
        Task<IMatrix> Multiply(IMatrix m1, IMatrix m2);
    }

    class Program
    {
        // Reduce matrix sizes and increase delays for stability
        static readonly int[] CSV_SIZES = { 100, 250, 500, 750, 1000, 1250, 1500, 1850 }; // Added larger sizes
        static readonly int ITERATIONS = 2;
        static readonly int[] BENCHMARK_SIZES = { 100, 250, 500, 750, 1000, 1250, 1500, 1850 }; // Added larger sizes
        static readonly int BENCHMARK_ITERATIONS = 2;
        private static readonly Dictionary<string, IMatrixMultiplication> _multipliers = new();

        // Adjusted delays for better performance
        private const int MIN_DELAY_MS = 500; // Reduced from 2000ms
        private const int COOLDOWN_FACTOR = 2; // For larger matrices
        private const int MAX_MATRIX_SIZE = 2000;
        private const int MAX_PARALLEL_TASKS = 1; // Reduced to 1 for stability
        private static readonly SemaphoreSlim _throttle = new(MAX_PARALLEL_TASKS);
        private static readonly SemaphoreSlim _initLock = new(1, 1);
        private static bool _initialized = false;

        static async Task<(double computeTime, double totalTime)> RunBenchmark(int matrixSize, string method)
        {
            await _throttle.WaitAsync(); // Throttle parallel operations
            try
            {
                var m1 = new Matriz(matrixSize, matrixSize);
                var m2 = new Matriz(matrixSize, matrixSize);

                // Sequential initialization to reduce memory pressure
                m1.SetRandoms();
                m2.SetRandoms();

                long computeTimeTotal = 0;
                var totalStopwatch = new Stopwatch();
                var computeStopwatch = new Stopwatch();

                totalStopwatch.Start();

                if (!_multipliers.ContainsKey(method))
                {
                    _multipliers[method] = await MatrixMultiplicationFactory.CreateMultiplier(method);
                }
                var multiplier = _multipliers[method];

                // Single warmup iteration
                await multiplier.Multiply(m1, m2);
                await Task.Delay(MIN_DELAY_MS);

                // Sequential benchmark runs
                for (int i = 0; i < ITERATIONS; i++)
                {
                    if (i > 0)
                    {
                        GC.Collect(0, GCCollectionMode.Forced);
                        await Task.Delay(MIN_DELAY_MS);
                    }

                    computeStopwatch.Restart();
                    await multiplier.Multiply(m1, m2);
                    computeStopwatch.Stop();
                    computeTimeTotal += computeStopwatch.ElapsedMilliseconds;
                }

                totalStopwatch.Stop();
                return (computeTimeTotal / (double)ITERATIONS, totalStopwatch.ElapsedMilliseconds / (double)ITERATIONS);
            }
            finally
            {
                _throttle.Release();
            }
        }

        static async Task<(double computeTime, double totalTime, double memory)> RunDetailedBenchmark(int matrixSize, string method)
        {
            // Increase timeout for larger matrices
            int timeoutMinutes = matrixSize >= 1500 ? 10 : 5;
            using var timeoutTokenSource = new CancellationTokenSource(TimeSpan.FromMinutes(timeoutMinutes));
            try
            {
                // Adjust cooldown based on matrix size exponentially
                int cooldownDelay = Math.Min(MIN_DELAY_MS * 2, 
                    100 + (int)(Math.Pow(matrixSize / 200.0, 1.5) * 100));
                
                // More aggressive GC for larger matrices
                if (matrixSize >= 1500)
                {
                    GC.Collect(2, GCCollectionMode.Forced, true, true);
                    await Task.Delay(cooldownDelay * 2);
                }
                else
                {
                    GC.Collect(2, GCCollectionMode.Forced);
                    await Task.Delay(cooldownDelay);
                }

                var m1 = new Matriz(matrixSize, matrixSize);
                var m2 = new Matriz(matrixSize, matrixSize);

                // Sequential initialization with progress tracking
                Console.Write("Initializing matrices... ");
                m1.SetRandoms();
                await Task.Delay(50);
                m2.SetRandoms();
                await Task.Delay(50);
                Console.WriteLine("Done");

                long computeTimeTotal = 0;
                var totalStopwatch = new Stopwatch();
                var computeStopwatch = new Stopwatch();
                long memoryBefore = GC.GetTotalMemory(true);

                totalStopwatch.Start();

                var multiplier = _multipliers[method];
                if (multiplier == null)
                    throw new InvalidOperationException($"Multiplier for method {method} not initialized");

                // Warmup with size-based cooling
                Console.Write("Warming up... ");
                await multiplier.Multiply(m1, m2);
                Console.WriteLine("Done");
                
                GC.Collect(1, GCCollectionMode.Forced);
                await Task.Delay(cooldownDelay);

                // Actual benchmark runs
                for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
                {
                    if (timeoutTokenSource.Token.IsCancellationRequested)
                        throw new TimeoutException("Benchmark timed out");

                    computeStopwatch.Restart();
                    var result = await multiplier.Multiply(m1, m2);
                    computeStopwatch.Stop();
                    computeTimeTotal += computeStopwatch.ElapsedMilliseconds;

                    if (result is IDisposable disposableResult)
                    {
                        disposableResult.Dispose();
                    }
                    result = null;
                    
                    // Adaptive cleanup between iterations
                    if (matrixSize >= 1500)
                    {
                        GC.Collect(2, GCCollectionMode.Forced, true, true);
                        await Task.Delay(cooldownDelay * 2);
                    }
                    else if (matrixSize > 500)
                    {
                        GC.Collect(1, GCCollectionMode.Forced);
                        await Task.Delay(cooldownDelay);
                    }
                    else
                    {
                        await Task.Delay(cooldownDelay / 2);
                    }
                }

                totalStopwatch.Stop();
                long memoryAfter = GC.GetTotalMemory(false);
                double memoryUsed = (memoryAfter - memoryBefore) / (1024.0 * 1024.0);

                return (
                    computeTimeTotal / (double)BENCHMARK_ITERATIONS,
                    totalStopwatch.ElapsedMilliseconds / (double)BENCHMARK_ITERATIONS,
                    memoryUsed
                );
            }
            catch (Exception)
            {
                // Ensure thorough cleanup on error
                GC.Collect(2, GCCollectionMode.Forced, true, true);
                throw;
            }
            finally
            {
                timeoutTokenSource.Dispose();
            }
        }

        static async Task GenerateCSVFiles()
        {
            string filename = "results_comparison.csv";
            var progressCount = 0;
            var totalTests = CSV_SIZES.Length * 4; // 4 methods per size
            
            Console.WriteLine("Generating comparative benchmark results...\n");
            
            using (StreamWriter writer = new StreamWriter(filename))
            {
                writer.WriteLine("Tamaño,Tiempo Computación (ms),Tiempo Total (ms),Método");

                // Group sizes into batches for parallel processing
                var batchedSizes = CSV_SIZES
                    .Select((size, index) => new { Size = size, Index = index })
                    .GroupBy(x => x.Size <= 500 ? 0 : x.Index)
                    .ToList();

                foreach (var batch in batchedSizes)
                {
                    var tasks = batch.SelectMany(item => 
                        new[] { "f", "c", "g", "h" }.Select(async method =>
                        {
                            var (computeTime, totalTime) = await RunBenchmark(item.Size, method);
                            string methodName = method switch
                            {
                                "f" => "CPU Optimized",
                                "c" => "Columns",
                                "g" => "GPU Optimized",
                                "h" => "Hybrid CPU-GPU",
                                _ => throw new ArgumentException("Invalid method")
                            };

                            return new { Size = item.Size, MethodName = methodName, ComputeTime = computeTime, TotalTime = totalTime };
                        })).ToList();

                    var results = await Task.WhenAll(tasks);

                    // Write results in order
                    foreach (var result in results.OrderBy(x => x.Size).ThenBy(x => x.MethodName))
                    {
                        writer.WriteLine($"{result.Size},{result.ComputeTime:F2},{result.TotalTime:F2},{result.MethodName}");
                        progressCount++;
                        Console.Write($"\rProgress: {progressCount}/{totalTests} ({(progressCount * 100.0 / totalTests):F0}%)");
                    }

                    // Force GC between batches to avoid memory pressure
                    GC.Collect();
                    await Task.Delay(50);
                }
            }

            Console.WriteLine($"\n\nResults saved to {filename}");
        }

        static async Task InitializeMultipliers()
        {
            await _initLock.WaitAsync();
            try
            {
                if (_initialized) return;

                // Initialize GPU first if available
                if (!MatrixMultiplicationFactory._gpuInitialized)
                {
                    var gpu = await MatrixMultiplicationFactory.DetectAndInitializeGPU();
                    if (gpu != null)
                    {
                        MatrixMultiplicationFactory._gpuMultiplier = gpu;
                        MatrixMultiplicationFactory._gpuInitialized = true;
                    }
                }

                // Pre-initialize all multipliers
                foreach (var method in new[] { "f", "c", "g", "h" })
                {
                    try 
                    {
                        var multiplier = await MatrixMultiplicationFactory.CreateMultiplier(method);
                        _multipliers[method] = multiplier;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Failed to initialize multiplier for method {method}: {ex.Message}");
                    }
                }

                _initialized = true;
            }
            finally
            {
                _initLock.Release();
            }
        }

        static async Task RunDetailedBenchmarks()
        {
            try
            {
                await InitializeMultipliers();

                string filename = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "benchmark_results_detailed.csv");
                var progressCount = 0;
                var totalBenchmarks = BENCHMARK_SIZES.Length * 4;
                
                Console.WriteLine($"Starting detailed benchmarks, saving to: {filename}");
                
                using (StreamWriter writer = new StreamWriter(filename))
                {
                    writer.WriteLine("Size,Method,Compute Time (ms),Total Time (ms),Memory Usage (MB),GFlops");
                    writer.Flush();

                    foreach (var size in BENCHMARK_SIZES)
                    {
                        if (size > MAX_MATRIX_SIZE)
                        {
                            Console.WriteLine($"Skipping size {size} as it exceeds safety limit");
                            continue;
                        }

                        // Exponential delay scaling for larger matrices
                        int currentDelay = (int)(200 * Math.Pow(1.5, Math.Max(0, (size - 500) / 250)));
                        
                        Console.WriteLine($"\nRunning benchmarks for {size}x{size} matrices... ({progressCount}/{totalBenchmarks})");
                        
                        // More thorough cleanup for larger matrices
                        if (size >= 1500)
                        {
                            GC.Collect(2, GCCollectionMode.Forced, true, true);
                            await Task.Delay(currentDelay * 3);
                        }
                        else
                        {
                            GC.Collect(2, GCCollectionMode.Forced);
                            await Task.Delay(currentDelay * 2);
                        }

                        foreach (var method in new[] { "f", "c", "g", "h" })
                        {
                            if (!_multipliers.ContainsKey(method))
                            {
                                Console.WriteLine($"Warning: Multiplier for method {method} not available, skipping...");
                                continue;
                            }

                            string methodName = method switch
                            {
                                "f" => "CPU Row-based",
                                "c" => "CPU Column-based",
                                "g" => "GPU",
                                "h" => "Hybrid CPU-GPU",
                                _ => throw new ArgumentException("Invalid method")
                            };

                            try
                            {
                                Console.WriteLine($"Running {methodName} benchmark...");
                                
                                // Extra cooldown for GPU methods or large matrices
                                if (method is "g" or "h" || size >= 1500)
                                {
                                    await Task.Delay(currentDelay * 2);
                                }

                                var (computeTime, totalTime, memory) = await RunDetailedBenchmark(size, method);
                                double operations = 2.0 * Math.Pow(size, 3);
                                double gflops = (operations / computeTime) / 1_000_000.0;

                                var resultLine = $"{size},{methodName},{computeTime:F2},{totalTime:F2},{memory:F2},{gflops:F2}";
                                writer.WriteLine(resultLine);
                                writer.Flush();
                                
                                Console.WriteLine($"{methodName,-15}: Compute: {computeTime,8:F2}ms, Total: {totalTime,8:F2}ms, {gflops,6:F2} GFlops");
                                progressCount++;
                                
                                // Adaptive cooldown between methods
                                if (size >= 1500)
                                {
                                    GC.Collect(2, GCCollectionMode.Forced, true, true);
                                    await Task.Delay(currentDelay * 2);
                                }
                                else
                                {
                                    GC.Collect(1, GCCollectionMode.Forced);
                                    await Task.Delay(currentDelay);
                                }
                            }
                            catch (Exception ex)
                            {
                                var errorLine = $"{size},{methodName},Error,Error,Error,Error";
                                writer.WriteLine(errorLine);
                                writer.Flush();
                                Console.WriteLine($"Error running {methodName}: {ex.Message}");
                                
                                // Extra cleanup and delay after error
                                GC.Collect(2, GCCollectionMode.Forced, true, true);
                                await Task.Delay(currentDelay * 3);
                            }
                        }

                        // Thorough cleanup between sizes
                        GC.Collect(2, GCCollectionMode.Forced, true, true);
                        await Task.Delay(currentDelay * 4);
                    }
                }

                Console.WriteLine($"\nDetailed benchmark results saved to {filename}");
                if (File.Exists(filename))
                {
                    Console.WriteLine("\nSummary of best performers by matrix size:");
                    AnalyzeBenchmarkResults(filename);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Benchmark failed: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner error: {ex.InnerException.Message}");
                }
                throw;
            }
        }

        static async Task RunTest(int matrixSize, string method, int iterations)
        {
            await InitializeMultipliers();
            var multiplier = await MatrixMultiplicationFactory.CreateMultiplier(method);
            Console.Write($"Running test for {matrixSize}x{matrixSize} matrix... ");
            var (computeTime, totalTime) = await RunBenchmark(matrixSize, method);
            string methodName = method switch
            {
                "f" => "CPU Optimized",
                "c" => "Columns",
                "g" => "GPU Optimized",
                "h" => "Hybrid CPU-GPU",
                _ => throw new ArgumentException("Invalid method")
            };
            
            Console.WriteLine("Done!");
            Console.WriteLine($"\nResults for {methodName}:");
            Console.WriteLine($"Average compute time: {computeTime:F2}ms");
            Console.WriteLine($"Average total time:  {totalTime:F2}ms");
            Console.WriteLine($"Average overhead:    {totalTime - computeTime:F2}ms");
        }

        private static void AnalyzeBenchmarkResults(string filename)
        {
            try
            {
                var results = new Dictionary<int, List<(string method, double computeTime, double gflops)>>();
                
                // Read and parse CSV file
                var lines = File.ReadAllLines(filename).Skip(1); // Skip header
                foreach (var line in lines)
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 6 && !parts.Contains("Error"))
                    {
                        int size = int.Parse(parts[0]);
                        string method = parts[1];
                        double computeTime = double.Parse(parts[2], CultureInfo.InvariantCulture);
                        double gflops = double.Parse(parts[5], CultureInfo.InvariantCulture);

                        if (!results.ContainsKey(size))
                        {
                            results[size] = new List<(string, double, double)>();
                        }
                        results[size].Add((method, computeTime, gflops));
                    }
                }

                // Analyze results for each matrix size
                foreach (var size in results.Keys.OrderBy(k => k))
                {
                    var bestResult = results[size]
                        .OrderBy(r => r.computeTime)
                        .FirstOrDefault();

                    Console.WriteLine($"Size {size}x{size}: Best method = {bestResult.method} ({bestResult.computeTime:F2}ms, {bestResult.gflops:F2} GFlops)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error analyzing results: {ex.Message}");
            }
        }

        static async Task Main(string[] args)
        {
            // Optimize thread pool and CPU affinity
            ThreadPool.GetMinThreads(out int workerThreads, out int completionPortThreads);
            int processorCount = Environment.ProcessorCount;
            // Use 75% of available cores for better CPU utilization while preventing system freeze
            int optimalThreads = Math.Max(4, (processorCount * 3) / 4);
            ThreadPool.SetMinThreads(optimalThreads, completionPortThreads);
            
            // Set process priority to above normal for benchmark runs
            using (var currentProcess = System.Diagnostics.Process.GetCurrentProcess())
            {
                try
                {
                    currentProcess.PriorityClass = System.Diagnostics.ProcessPriorityClass.AboveNormal;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Could not set process priority: {ex.Message}");
                }
            }

            // Setup cancellation support
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => {
                e.Cancel = true; // Prevent immediate termination
                Console.WriteLine("\nCancellation requested. Cleaning up...");
                cts.Cancel();
            };

            try
            {
                // Single initialization point
                await InitializeMultipliers();

                if (args.Length == 0 || args[0].ToLower() == "benchmark")
                {
                    await RunDetailedBenchmarks();
                    return;
                }

                if (args.Length != 3)
                {
                    Console.WriteLine("Usage:");
                    Console.WriteLine("  MatrixProd benchmark           - Run comprehensive benchmarks");
                    Console.WriteLine("  MatrixProd <size> <method> <iterations>");
                    Console.WriteLine("Methods: f (rows), c (columns), g (GPU), h (hybrid)");
                    Console.WriteLine("\nPress Ctrl+C at any time to safely stop the benchmarks.");
                    return;
                }

                if (int.TryParse(args[0], out int size) && int.TryParse(args[2], out int iterations))
                {
                    await RunTest(size, args[1].ToLower(), iterations);
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("\nBenchmark cancelled by user.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner error: {ex.InnerException.Message}");
                }
            }
            finally
            {
                // Ensure cleanup happens even on cancellation
                foreach (var multiplier in _multipliers.Values)
                {
                    try
                    {
                        multiplier?.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Cleanup error: {ex.Message}");
                    }
                }
                _multipliers.Clear();
                MatrixMultiplicationFactory.Cleanup();
                _initLock.Dispose();
                
                // Final cleanup message
                Console.WriteLine("Cleanup completed. Program terminated safely.");
            }
        }
    }
}