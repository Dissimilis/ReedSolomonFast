namespace ReedSolomonFast.Tests;

public class EncodeShardsTableTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var tier in Enum.GetValues<KernelTier>())
        {
            if (!Internal.Kernel.IsSupported(tier)) continue;
            // 96 and 104 entries straddle the 4 KiB temporary-table limit; 128+128
            // exercises the largest possible table with the total-shard limit of 256.
            foreach (var shape in new[] { (25, 8, 12), (25, 8, 13), (128, 128, 128) })
                yield return [tier, shape.Item1, shape.Item2, shape.Item3];
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SelectedColumns_AccumulateCorrectlyAcrossStorageSizes(KernelTier tier, int dataCount, int parityCount, int count)
    {
        foreach (int length in new[] { 1, 63, 64, 65, 129 })
        {
            var coder = new ReedSolomon(dataCount, parityCount, new ReedSolomonOptions { Kernel = tier });
            var reference = new ReedSolomon(dataCount, parityCount, new ReedSolomonOptions { Kernel = KernelTier.Scalar });
            var shards = TestData.RandomShards(dataCount + parityCount, length + 3, 73);
            int[] indices = Enumerable.Range(0, dataCount).Reverse().Take(count).ToArray();
            var selected = indices.Select(i => (ReadOnlyMemory<byte>)shards[i].AsMemory(1, length)).ToArray();
            var parity = shards.Skip(dataCount).Select(s => s.AsMemory(1, length)).ToArray();
            var expected = parity.Select(p => (Memory<byte>)p.ToArray()).ToArray();

            for (int i = 0; i < count; i++) reference.EncodeShard(indices[i], selected[i].Span, expected);
            coder.EncodeShards(indices, selected, parity);
            for (int p = 0; p < parityCount; p++) Assert.Equal(expected[p].ToArray(), parity[p].ToArray());
        }
    }

    [Theory]
    [InlineData(12)]
    [InlineData(13)]
    public void WarmSerialCalls_DoNotAllocate(int count)
    {
        var coder = new ReedSolomon(25, 8);
        var shards = TestData.RandomShards(33, 65, 91);
        int[] indices = Enumerable.Range(0, count).Reverse().ToArray();
        var data = indices.Select(i => (ReadOnlyMemory<byte>)shards[i]).ToArray();
        var parity = TestData.AsMemory(shards, 25, 8);
        for (int i = 0; i < 20; i++) coder.EncodeShards(indices, data, parity);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) coder.EncodeShards(indices, data, parity);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}
