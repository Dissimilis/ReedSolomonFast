using System.Diagnostics;

namespace ReedSolomonFast.Benchmarks;

/// <summary>Alternates baseline and current Update calls on the same buffers to reduce process-layout and timing drift.</summary>
internal static class PairedUpdateBenchmarks
{
    public static int Run()
    {
        Console.WriteLine("Paired Update: 30 alternating ABBA/BAAB samples, 99.9% intervals. Check controls against the measured noise floor.");
        string[] shapes = (Environment.GetEnvironmentVariable("RSF_PAIRED_SHAPES") ?? "10+4,50+20").Split(',');
        int[] sizes = (Environment.GetEnvironmentVariable("RSF_PAIRED_SIZES") ?? "65536,1048576").Split(',').Select(int.Parse).ToArray();
        string? encodeControlShape = Environment.GetEnvironmentVariable("RSF_PAIRED_ENCODE_CONTROL_SHAPE");
        foreach (string shape in shapes)
        foreach (int size in sizes)
        foreach (string layout in new[] { "arrays", "padded" })
        foreach (int changes in new[] { 1, 2, 3, 4, 5, 8 })
        {
            var bench = new UpdateBatchBenchmarks { Shape = shape, Shard_Size = size, Layout = layout, Changes = changes };
            bench.Setup();
            Measure("Update", bench.BeforeUpdate, bench.AfterUpdate, shape, size, layout, changes);
            if (changes == 1 && (encodeControlShape is null || encodeControlShape == shape))
                Measure("Encode", bench.BeforeEncode, bench.AfterEncode, shape, size, layout, changes);
        }

        return 0;
    }

    internal static void Measure(string operation, Func<byte> before, Func<byte> after, string shape, int size, string layout, int changes)
    {
        // One pair of coder instances and the exact same buffers throughout. Equal call counts toggle
        // Update parity back to its starting value; reverse the order on every sample (ABBA / BAAB).
        var warmup = Stopwatch.StartNew();
        while (warmup.ElapsedMilliseconds < 300) { before(); after(); }
        int iterations = 1;
        while (Time(before, iterations) < Stopwatch.Frequency / 50.0) iterations *= 2;
        var ratios = new double[30];
        double beforeTotal = 0, afterTotal = 0;
        for (int sample = 0; sample < ratios.Length; sample++)
        {
            double b, a;
            if ((sample & 1) == 0)
            {
                b = Time(before, iterations);
                a = Time(after, iterations);
                a += Time(after, iterations);
                b += Time(before, iterations);
            }
            else
            {
                a = Time(after, iterations);
                b = Time(before, iterations);
                b += Time(before, iterations);
                a += Time(after, iterations);
            }
            beforeTotal += b;
            afterTotal += a;
            ratios[sample] = Math.Log(a / b);
        }
        double mean = ratios.Average();
        double se = Math.Sqrt(ratios.Sum(x => (x - mean) * (x - mean)) / (ratios.Length - 1) / ratios.Length);
        // Two-sided 99.9% Student-t interval, 29 degrees of freedom, on paired log ratios.
        double lower = Math.Exp(mean - 3.659 * se), upper = Math.Exp(mean + 3.659 * se);
        string verdict = se > .02 ? "NOISY" : upper < 1 ? "IMPROVED" : lower > 1 ? "REGRESSED" : "NO RESULT";
        double scale = 1e6 / Stopwatch.Frequency / (2.0 * ratios.Length * iterations);
        Console.WriteLine($"{operation} {shape} {size} {layout} changes={changes} before_us={beforeTotal * scale:F3} after_us={afterTotal * scale:F3} ratio={Math.Exp(mean):F4} ci999=[{lower:F4},{upper:F4}] {verdict}");
        Console.Out.Flush();
    }

    private static long Time(Func<byte> action, int iterations)
    {
        byte result = 0;
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < iterations; i++) result ^= action();
        long elapsed = Stopwatch.GetTimestamp() - start;
        GC.KeepAlive(result);
        return elapsed;
    }
}
