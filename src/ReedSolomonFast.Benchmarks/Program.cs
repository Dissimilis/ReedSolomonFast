extern alias Baseline;

using System.Diagnostics;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Perfolizer.Mathematics.OutlierDetection;

namespace ReedSolomonFast.Benchmarks;

/// <summary>
/// Entry point for the optimization loop.
///
/// Usage:
///   dotnet run -c Release                     A/B: frozen baseline vs current. The decision run.
///   dotnet run -c Release -- --competitive    where we stand against ReedSolomon.NET and Witteborn
///   dotnet run -c Release -- --tiers          every kernel tier this CPU has, in one run
///   dotnet run -c Release -- --dispatch       sweep MaxInputGroup x ChunkBytes, one process per pair
///   dotnet run -c Release -- --batch          N separate 4 KiB stripes vs one N x 4 KiB call
///   dotnet run -c Release -- --api            our own API surfaces against each other
///   dotnet run -c Release -- --fixed          fixed cost of a small call: checks, pin, kernel
///   dotnet run -c Release -- --streaming      streaming stores against stripe footprint and output count
///   dotnet run -c Release -- --incremental    EncodeShard, EncodeShards and Update per layout
///   dotnet run -c Release -- --layout         windowed encoding of a DRAM-sized payload, and shard-base skew
///   dotnet run -c Release -- --all            everything
///   dotnet run -c Release -- --report         faster job for publishing tables (README numbers)
///   dotnet run -c Release -- --quick          smoke test only; never quote its numbers
///   dotnet run -c Release -- --filter "*10+4*"   any BenchmarkDotNet filter still works
///
/// HOW TO READ A RESULT. The default run compares the frozen snapshot against the current build.
/// Read the per-case verdict printed after the table, not the table alone. Do NOT compare a Mean
/// from one session against a Mean from another: this machine throttles roughly 2x under sustained
/// load, and a 10+4 encode at 1 MiB has been seen at 20 GB/s and 11 GB/s in consecutive probes
/// with no code change. Only ratios measured inside one run mean anything.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        var flags = new HashSet<string>(args.Where(a => a.StartsWith("--")), StringComparer.OrdinalIgnoreCase);
        var passthrough = args.Where(a => !IsOurFlag(a)).ToArray();
        bool quick = flags.Contains("--quick");
        bool report = flags.Contains("--report");

        PrintProvenance();

        // A faster wrong answer is not an improvement. Gate every session on correctness and fail
        // loudly rather than reporting numbers for a broken coder.
        try
        {
            Correctness.Run();
        }
        catch (InvalidOperationException ex)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        if (quick)
        {
            Console.WriteLine();
            Console.WriteLine("WARNING: --quick is a smoke test. Its error margins are far too wide to");
            Console.WriteLine("decide whether an optimization helped. Never quote a --quick number.");
        }

        Console.WriteLine();
        var config = BuildConfig(quick, report);

        Summary[] summaries = flags.Contains("--all")
            ? BenchmarkRunner.Run(
                [typeof(OptimizationBenchmarks), typeof(CompetitiveBenchmarks), typeof(TierBenchmarks), typeof(ApiSurfaceBenchmarks)],
                config, passthrough)
            : flags.Contains("--competitive") ? [BenchmarkRunner.Run<CompetitiveBenchmarks>(config, passthrough)]
            : flags.Contains("--tiers") ? [BenchmarkRunner.Run<TierBenchmarks>(config, passthrough)]
            : flags.Contains("--dispatch") ? [BenchmarkRunner.Run<DispatchBenchmarks>(SweepConfig(config), passthrough)]
            : flags.Contains("--tails") ? [BenchmarkRunner.Run<TailBenchmarks>(config, passthrough)]
            : flags.Contains("--batch") ? [BenchmarkRunner.Run<BatchBenchmarks>(config, passthrough)]
            : flags.Contains("--api") ? [BenchmarkRunner.Run<ApiSurfaceBenchmarks>(config, passthrough)]
            : flags.Contains("--fixed") ? [BenchmarkRunner.Run<FixedCostBenchmarks>(config, passthrough)]
            : flags.Contains("--streaming") ? [BenchmarkRunner.Run<StreamingSweepBenchmarks>(config, passthrough), BenchmarkRunner.Run<OutputStreamSweepBenchmarks>(config, passthrough)]
            : flags.Contains("--incremental") ? [BenchmarkRunner.Run<IncrementalBenchmarks>(config, passthrough)]
            : flags.Contains("--layout") ? [BenchmarkRunner.Run<WindowBenchmarks>(config, passthrough), BenchmarkRunner.Run<SkewBenchmarks>(config, passthrough)]
            : [BenchmarkRunner.Run<OptimizationBenchmarks>(config, passthrough)];

        foreach (var summary in summaries) PrintAbVerdicts(summary);
        return ReportOutcome(summaries);
    }

    /// <summary>
    /// Turns the A/B table into an explicit answer. Without this, a run that never reached its
    /// precision target still prints a crisp-looking Ratio, and a tired reader accepts a 3% "win"
    /// that sits well inside a 6% error margin.
    /// </summary>
    private static void PrintAbVerdicts(Summary summary)
    {
        var pairs = summary.Reports
            .Where(r => r.ResultStatistics is not null)
            .Select(r => new
            {
                Report = r,
                Shape = r.BenchmarkCase.Parameters.Items.FirstOrDefault(p => p.Name == "Shape")?.Value?.ToString(),
                Size = r.BenchmarkCase.Parameters.Items.FirstOrDefault(p => p.Name == "Shard_Size")?.Value,
                Method = r.BenchmarkCase.Descriptor.WorkloadMethod.Name,
            })
            .Where(x => x.Size is not null && (x.Method.StartsWith("Before", StringComparison.Ordinal) || x.Method.StartsWith("After", StringComparison.Ordinal)))
            .GroupBy(x => (
                x.Shape,
                Size: Convert.ToInt64(x.Size),
                Variant: x.Method.StartsWith("Before", StringComparison.Ordinal) ? x.Method["Before".Length..] : x.Method["After".Length..]))
            .Where(g => g.Count() == 2)
            .OrderBy(g => g.Key.Variant).ThenBy(g => g.Key.Shape).ThenBy(g => g.Key.Size)
            .ToList();

        if (pairs.Count == 0) return;

        Console.WriteLine();
        Console.WriteLine("A/B verdict (before = frozen baseline, after = current build)");
        Console.WriteLine("-------------------------------------------------------------");

        foreach (var pair in pairs)
        {
            var before = pair.First(x => x.Method.StartsWith("Before", StringComparison.Ordinal)).Report.ResultStatistics!;
            var after = pair.First(x => x.Method.StartsWith("After", StringComparison.Ordinal)).Report.ResultStatistics!;

            double ratio = after.Mean / before.Mean;
            double changePercent = (ratio - 1.0) * 100.0;
            double worstError = Math.Max(before.StandardError / before.Mean, after.StandardError / after.Mean) * 100.0;
            bool overlap = before.ConfidenceInterval.Lower <= after.ConfidenceInterval.Upper
                           && after.ConfidenceInterval.Lower <= before.ConfidenceInterval.Upper;

            string verdict = worstError > 2.0 ? $"NO RESULT (noisy: +/-{worstError:F1}%)"
                : overlap ? "NO RESULT (confidence intervals overlap)"
                : changePercent < 0 ? $"IMPROVED {-changePercent:F1}%"
                : $"REGRESSED {changePercent:F1}%";

            Console.WriteLine($"  {pair.Key.Variant,-12} {pair.Key.Shape,-6} {pair.Key.Size,10:N0} B  ratio {ratio:F3}  {verdict}");
        }

        Console.WriteLine();
        Console.WriteLine("  NO RESULT means the run cannot support a claim either way; it does not mean");
        Console.WriteLine("  'no change'. Re-run with clocks pinned, or accept that the effect is below");
        Console.WriteLine("  this machine's resolution.");
        Console.WriteLine();
    }

    private static bool IsOurFlag(string arg) =>
        new[] { "--competitive", "--tiers", "--dispatch", "--tails", "--batch", "--api", "--fixed", "--layout", "--incremental", "--streaming", "--all", "--quick", "--report" }
            .Any(f => arg.Equals(f, StringComparison.OrdinalIgnoreCase));

    /// <summary>A benchmark run that silently failed but exited 0 looks like a clean result. Surface it.</summary>
    private static int ReportOutcome(Summary[] summaries)
    {
        int problems = 0;
        foreach (var summary in summaries)
        {
            if (summary.HasCriticalValidationErrors)
            {
                Console.Error.WriteLine($"{summary.Title}: critical validation errors.");
                problems++;
            }

            foreach (var report in summary.Reports.Where(r => !r.Success))
            {
                Console.Error.WriteLine($"FAILED: {report.BenchmarkCase.DisplayInfo}");
                problems++;
            }
        }

        if (problems > 0)
        {
            Console.Error.WriteLine($"{problems} benchmark problem(s); results are not trustworthy.");
            return 1;
        }

        return 0;
    }

    private static IConfig BuildConfig(bool quick, bool report)
    {
        Job job;
        if (quick)
        {
            job = Job.ShortRun.WithWarmupCount(1).WithIterationCount(3).WithId("Smoke");
        }
        else if (report)
        {
            // For publishing a table across many cases, where the useful precision is "which is
            // faster and roughly by how much" rather than "did this commit move it 5%".
            job = Job.ShortRun.WithWarmupCount(3).WithIterationCount(5).WithId("Report");
        }
        else
        {
            // Not a fixed iteration count: pinning it disables the adaptive stopping rule, and five
            // iterations cannot separate a 5% change from noise on a throttling laptop. Sized so the
            // default 24-case run finishes in about five minutes: one launch, 100 ms iterations,
            // and a 2% relative-error target the verdict printer also enforces.
            job = Job.Default
                .WithMinWarmupCount(4)
                .WithMinIterationCount(10)
                .WithMaxIterationCount(30)
                .WithIterationTime(Perfolizer.Horology.TimeInterval.FromMilliseconds(100))
                .WithMaxRelativeError(0.02)
                .WithLaunchCount(1)
                // A run that slowed because the package got hot is evidence, not an outlier.
                .WithOutlierMode(OutlierMode.DontRemove)
                .WithId("Decide");
        }

        // The optimizations validator is off because a competitor package ships a Debug build and
        // would abort every run. Our own two assemblies are checked explicitly in PrintProvenance.
        return ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(job)
            .AddDiagnoser(MemoryDiagnoser.Default);
    }

    /// <summary>
    /// One job per (group, chunk) pair, each a separate process with the environment variables
    /// VectorKernel reads at type initialization. The base job's timing settings are kept.
    /// </summary>
    private static IConfig SweepConfig(IConfig baseConfig)
    {
        Job baseJob = baseConfig.GetJobs().First();
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddDiagnoser(MemoryDiagnoser.Default);
        foreach (int group in new[] { 8, 12, 16, 24 })
            foreach (int chunkKiB in new[] { 32, 64, 128, 256 })
            {
                config = config.AddJob(baseJob
                    .WithEnvironmentVariables(
                        new EnvironmentVariable("REEDSOLOMONFAST_GROUP", group.ToString()),
                        new EnvironmentVariable("REEDSOLOMONFAST_CHUNK", (chunkKiB * 1024).ToString()))
                    .WithId($"g{group}-c{chunkKiB}K"));
            }

        return config;
    }

    /// <summary>Stamps the commit under test; an unattributable number is worse than no number.</summary>
    private static void PrintProvenance()
    {
        Console.WriteLine($"ReedSolomonFast benchmark suite  |  commit {GitDescribe()}  |  {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"Best kernel on this CPU: {ReedSolomon.BestSupportedKernel}");
        Console.WriteLine("Decision metric: current build vs the frozen baseline build, in this same run.");
        Console.WriteLine("Absolute times are NOT comparable across sessions (this machine throttles ~2x).");
        VerifyBaselineIsDistinct();
        Console.WriteLine();
    }

    /// <summary>
    /// Proves the two A/B rows really are two different builds. If both rows resolved to the same
    /// assembly every ratio would pin near 1.00 and read as "the optimization did nothing".
    /// </summary>
    private static void VerifyBaselineIsDistinct()
    {
        var current = typeof(ReedSolomon).Assembly;
        var baseline = typeof(Baseline::ReedSolomonFast.ReedSolomon).Assembly;
        string? currentName = current.GetName().Name;
        string? baselineName = baseline.GetName().Name;
        if (currentName == baselineName || ReferenceEquals(current, baseline))
        {
            throw new InvalidOperationException(
                $"A/B control is broken: both rows resolve to assembly '{currentName}'. "
                + "The comparison would report ~1.00 regardless of any optimization.");
        }

        Console.WriteLine($"A/B control: current='{currentName}' vs baseline='{baselineName}'");

        foreach (var assembly in new[] { current, baseline })
        {
            var debuggable = assembly.GetCustomAttributes(typeof(DebuggableAttribute), false).OfType<DebuggableAttribute>().FirstOrDefault();
            if (debuggable is not null && debuggable.IsJITOptimizerDisabled)
                throw new InvalidOperationException($"{assembly.GetName().Name} is a Debug build; benchmark numbers would be meaningless. Build with -c Release.");
        }
    }

    private static string GitDescribe()
    {
        try
        {
            var psi = new ProcessStartInfo("git", "describe --always --dirty --tags")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using var process = Process.Start(psi);
            if (process is null) return "unknown";
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(3000);
            return string.IsNullOrEmpty(output) ? "unknown" : output;
        }
        catch
        {
            return "unknown";
        }
    }
}
