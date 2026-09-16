namespace ReedSolomonFast.Tests;

/// <summary>Checks changed-shard batches against fresh scalar encoding across vector and batch boundaries, layouts and parallel splits.</summary>
public class UpdateBatchTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (KernelTier tier in Enum.GetValues<KernelTier>())
            foreach (int changes in new[] { 1, 2, 3, 4, 5, 6, 7, 8, 10 })
                yield return [tier, changes];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Update_BatchesMatchFreshEncode(KernelTier tier, int changes)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;
        var random = new Random(701 + changes);
        int[] lengths = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 961,
            65_535, 65_536, 65_537, 65_551, 65_567, 65_599, 65_663];
        foreach (int length in lengths)
        foreach (bool padded in new[] { false, true })
        foreach (int workers in new[] { 1, 3 })
        {
            var coder = new ReedSolomon(10, 4, new ReedSolomonOptions
            {
                Kernel = tier,
                MaxDegreeOfParallelism = workers,
                ParallelThresholdBytes = 256,
            });
            Memory<byte>[] storage = padded
                ? ReedSolomon.AllocateShards(14 + changes, length + 3)
                : Enumerable.Range(0, 14 + changes).Select(_ => new Memory<byte>(new byte[length + 3])).ToArray();
            foreach (var m in storage) random.NextBytes(m.Span);
            byte[] guards = storage.SelectMany(m => m.Span[..3].ToArray()).ToArray();
            Memory<byte>[] buffers = storage.Select(m => m.Slice(3, length)).ToArray();
            var data = buffers.Take(10).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            var parity = buffers.Skip(10).Take(4).ToArray();
            int[] indices = Enumerable.Range(0, changes).Select(i => 9 - i).ToArray();
            var oldData = indices.Select(i => data[i]).ToArray();
            var newData = buffers.Skip(14).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            // A zero delta in a batch must not affect the other terms.
            if (changes > 2) newData[1] = oldData[1];
            coder.Encode(data, parity);
            coder.Update(indices, oldData, newData, parity);
            for (int i = 0; i < changes; i++) data[indices[i]] = newData[i];
            var expected = Enumerable.Range(0, 4).Select(_ => new Memory<byte>(new byte[length])).ToArray();
            new ReedSolomon(10, 4, new ReedSolomonOptions { Kernel = KernelTier.Scalar }).Encode(data, expected);
            for (int p = 0; p < 4; p++)
                Assert.True(expected[p].Span.SequenceEqual(parity[p].Span), $"{tier}, changes={changes}, length={length}, padded={padded}, workers={workers}, parity={p}");
            Assert.Equal(guards, storage.SelectMany(m => m.Span[..3].ToArray()).ToArray());
        }
    }
}
