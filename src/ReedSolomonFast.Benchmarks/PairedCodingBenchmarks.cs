extern alias Baseline;

using BaselineCoder = Baseline::ReedSolomonFast.ReedSolomon;

namespace ReedSolomonFast.Benchmarks;

/// <summary>Paired Encode/Reconstruct comparisons on identical caller buffers, including padded layouts.</summary>
internal static class PairedCodingBenchmarks
{
    public static int Run()
    {
        Console.WriteLine("Paired coding: 30 alternating ABBA/BAAB samples, 99.9% intervals. Establish identical-code noise before interpreting ratios.");
        string[] shapes = (Environment.GetEnvironmentVariable("RSF_PAIRED_SHAPES") ?? "5+2,10+4,8+8,50+20").Split(',');
        int[] sizes = (Environment.GetEnvironmentVariable("RSF_PAIRED_SIZES") ?? "65536,1048576").Split(',').Select(int.Parse).ToArray();
        foreach (string shape in shapes)
        foreach (int size in sizes)
        foreach (string layout in new[] { "arrays", "padded" })
        {
            var (dataCount, parityCount) = Geometry.Parse(shape);
            ReedSolomon current;
            BaselineCoder baseline;
            // Reversing construction order helps detect table/allocation-layout artifacts
            // even though the timed operations already alternate on identical shard buffers.
            if (Environment.GetEnvironmentVariable("RSF_PAIRED_BASELINE_FIRST") == "1")
            {
                baseline = new BaselineCoder(dataCount, parityCount);
                current = new ReedSolomon(dataCount, parityCount);
            }
            else
            {
                current = new ReedSolomon(dataCount, parityCount);
                baseline = new BaselineCoder(dataCount, parityCount);
            }
            Memory<byte>[] shards = layout == "padded"
                ? ReedSolomon.AllocateShards(dataCount + parityCount, size)
                : Enumerable.Range(0, dataCount + parityCount).Select(_ => new Memory<byte>(new byte[size])).ToArray();
            var random = new Random(143);
            foreach (var shard in shards) random.NextBytes(shard.Span);
            var data = shards.Take(dataCount).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            var parity = shards.Skip(dataCount).ToArray();
            var present = Enumerable.Range(0, shards.Length).Select(i => i >= parityCount).ToArray();
            baseline.Encode(data, parity);
            var expected = shards.Select(m => m.ToArray()).ToArray();
            current.Encode(data, parity);
            Check();
            foreach (var shard in shards.Take(parityCount)) shard.Span.Clear();
            current.Reconstruct(shards, present);
            Check();
            foreach (var shard in shards.Take(parityCount)) shard.Span.Clear();
            baseline.Reconstruct(shards, present);
            Check();

            PairedUpdateBenchmarks.Measure("Encode", () => { baseline.Encode(data, parity); return parity[0].Span[0]; },
                () => { current.Encode(data, parity); return parity[0].Span[0]; }, shape, size, layout, 0);
            PairedUpdateBenchmarks.Measure("Reconstruct", () => { baseline.Reconstruct(shards, present); return shards[0].Span[0]; },
                () => { current.Reconstruct(shards, present); return shards[0].Span[0]; }, shape, size, layout, 0);
            Check();

            void Check()
            {
                for (int i = 0; i < shards.Length; i++)
                    if (!shards[i].Span.SequenceEqual(expected[i]))
                        throw new InvalidOperationException($"Paired coding mismatch: {shape} {size} {layout} shard {i}.");
            }
        }
        return 0;
    }
}
