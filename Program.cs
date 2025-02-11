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
        // Reduce sizes to prevent memory pressure
        static readonly int[] CSV_SIZES = { 100, 250, 500, 1000 };
        static readonly int ITERATIONS = 2;
        static readonly int[] BENCHMARK_SIZES = { 100, 250, 500, 750, 1000 };
        static readonly int WARMUP_ITERATIONS = 1;
        static readonly int BENCHMARK_ITERATIONS = 2;
        private static readonly Dictionary<string, IMatrixMultiplication> _multipliers = new();

        // Safety limits
        private const int MIN_DELAY_MS = 500; // Increased delay between operations
        private const int MAX_MATRIX_SIZE = 2500;
        private const int MAX_PARALLEL_TASKS = 2; // Limit parallel operations
        private static readonly SemaphoreSlim _throttle = new(MAX_PARALLEL_TASKS);

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
            var m1 = new Matriz(matrixSize, matrixSize);
            var m2 = new Matriz(matrixSize, matrixSize);
            var timeoutTokenSource = new CancellationTokenSource(TimeSpan.FromMinutes(5)); // 5-minute timeout

            try
            {
                // Initialize matrices in parallel
                Parallel.Invoke(
                    () => m1.SetRandoms(),
                    () => m2.SetRandoms()
                );

                long computeTimeTotal = 0;
                var totalStopwatch = new Stopwatch();
                var computeStopwatch = new Stopwatch();
                long memoryBefore = GC.GetTotalMemory(true);

                totalStopwatch.Start();

                var multiplier = _multipliers[method];
                if (multiplier == null)
                    throw new InvalidOperationException($"Multiplier for method {method} not initialized");

                // Warmup phase with timeout protection
                for (int i = 0; i < WARMUP_ITERATIONS; i++)
                {
                    await Task.WhenAny(
                        multiplier.Multiply(m1, m2),
                        Task.Delay(-1, timeoutTokenSource.Token)
                    );

                    if (timeoutTokenSource.Token.IsCancellationRequested)
                        throw new TimeoutException("Warmup phase timed out");
                }

                // Actual benchmark runs
                for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
                {
                    if (timeoutTokenSource.Token.IsCancellationRequested)
                        throw new TimeoutException("Benchmark timed out");

                    GC.Collect(0, GCCollectionMode.Forced);
                    await Task.Delay(10); // Short delay between iterations

                    computeStopwatch.Restart();
                    await multiplier.Multiply(m1, m2);
                    computeStopwatch.Stop();
                    computeTimeTotal += computeStopwatch.ElapsedMilliseconds;
                }

                totalStopwatch.Stop();
                long memoryAfter = GC.GetTotalMemory(false);
                double memoryUsed = (memoryAfter - memoryBefore) / (1024.0 * 1024.0); // MB

                return (
                    computeTimeTotal / (double)BENCHMARK_ITERATIONS,
                    totalStopwatch.ElapsedMilliseconds / (double)BENCHMARK_ITERATIONS,
                    memoryUsed
                );
            }
            catch (Exception)
            {
                timeoutTokenSource.Cancel(); // Ensure cleanup
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

        static async Task RunDetailedBenchmarks()
        {
            try
            {
                Console.WriteLine("Initializing GPU support...");
                
                // Pre-initialize multipliers one at a time
                var methods = new[] { "f", "c", "g", "h" };
                foreach (var method in methods)
                {
                    try
                    {
                        if (!_multipliers.ContainsKey(method))
                        {
                            _multipliers[method] = await MatrixMultiplicationFactory.CreateMultiplier(method);
                            await Task.Delay(MIN_DELAY_MS);
                            GC.Collect();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Failed to initialize {method} multiplier: {ex.Message}");
                    }
                }

                string filename = "benchmark_results_detailed.csv";
                var progressCount = 0;
                var totalBenchmarks = BENCHMARK_SIZES.Length;
                
                using (StreamWriter writer = new StreamWriter(filename))
                {
                    writer.WriteLine("Size,Method,Compute Time (ms),Total Time (ms),Memory Usage (MB),GFlops");

                    // Process one size at a time
                    foreach (var size in BENCHMARK_SIZES)
                    {
                        if (size > MAX_MATRIX_SIZE)
                        {
                            Console.WriteLine($"Skipping size {size} as it exceeds safety limit");
                            continue;
                        }

                        progressCount++;
                        Console.WriteLine($"\nRunning benchmarks for {size}x{size} matrices... ({progressCount}/{totalBenchmarks})");

                        foreach (var method in methods)
                        {
                            if (!_multipliers.ContainsKey(method))
                                continue;

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
                                var (computeTime, totalTime, memory) = await RunDetailedBenchmark(size, method);
                                double operations = 2.0 * Math.Pow(size, 3);
                                double gflops = (operations / computeTime) / 1_000_000.0;

                                writer.WriteLine($"{size},{methodName},{computeTime:F2},{totalTime:F2},{memory:F2},{gflops:F2}");
                                Console.WriteLine($"{methodName,-15}: Compute: {computeTime,8:F2}ms, Total: {totalTime,8:F2}ms, {gflops,6:F2} GFlops");
                                
                                // Increased cooldown between methods
                                GC.Collect();
                                await Task.Delay(MIN_DELAY_MS * 2);
                            }
                            catch (Exception ex)
                            {
                                writer.WriteLine($"{size},{methodName},Error,Error,Error,Error");
                                Console.WriteLine($"{methodName,-15}: Error - {ex.Message}");
                                
                                // Extra delay after error
                                await Task.Delay(MIN_DELAY_MS * 3);
                            }
                        }

                        // Much longer cooldown between sizes
                        GC.Collect(2, GCCollectionMode.Forced);
                        await Task.Delay(MIN_DELAY_MS * 4);
                    }
                }

                Console.WriteLine($"\nDetailed benchmark results saved to {filename}");
                Console.WriteLine("\nSummary of best performers by matrix size:");
                AnalyzeBenchmarkResults(filename);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Benchmark failed: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner error: {ex.InnerException.Message}");
                }
            }
        }

        static void AnalyzeBenchmarkResults(string filename)
        {
            var results = File.ReadAllLines(filename)
                .Skip(1) // Skip header
                .Where(line => !line.Contains("Error"))
                .Select(line =>
                {
                    var parts = line.Split(',');
                    return new
                    {
                        Size = int.Parse(parts[0]),
                        Method = parts[1],
                        ComputeTime = double.Parse(parts[2], CultureInfo.InvariantCulture),
                        GFlops = double.Parse(parts[5], CultureInfo.InvariantCulture)
                    };
                })
                .GroupBy(r => r.Size)
                .OrderBy(g => g.Key);

            foreach (var sizeGroup in results)
            {
                var best = sizeGroup.OrderBy(r => r.ComputeTime).First();
                Console.WriteLine($"Size {sizeGroup.Key}x{sizeGroup.Key}: Best method = {best.Method} ({best.ComputeTime:F2}ms, {best.GFlops:F2} GFlops)");
            }
        }

        static async Task RunTest(int matrixSize, string method, int iterations)
        {
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
                // Initialize GPU with conservative timeout
                if (!MatrixMultiplicationFactory._gpuInitialized)
                {
                    Console.WriteLine("Attempting to initialize AMD GPU...");
                    using var gpuInitTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                    using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(gpuInitTimeout.Token, cts.Token);
                    
                    try
                    {
                        var gpuTask = MatrixMultiplicationFactory.DetectAndInitializeGPU();
                        var timeoutTask = Task.Delay(30000, combinedCts.Token);
                        
                        var completedTask = await Task.WhenAny(gpuTask, timeoutTask);
                        if (completedTask == gpuTask && !gpuTask.IsFaulted)
                        {
                            var gpu = await gpuTask;
                            if (gpu != null)
                            {
                                MatrixMultiplicationFactory._gpuMultiplier = gpu;
                                MatrixMultiplicationFactory._gpuInitialized = true;
                                Console.WriteLine("AMD GPU initialized successfully");
                            }
                        }
                        else
                        {
                            Console.WriteLine("GPU initialization timed out, will fall back to CPU methods");
                        }
                    }
                    catch (OperationCanceledException) when (gpuInitTimeout.Token.IsCancellationRequested)
                    {
                        Console.WriteLine("GPU initialization timed out, will fall back to CPU methods");
                    }
                }

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
                
                // Final cleanup message
                Console.WriteLine("Cleanup completed. Program terminated safely.");
            }
        }
    }
}