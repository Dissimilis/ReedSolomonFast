using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

/// <summary>Tests that read the process-wide <see cref="KernelStats"/> counters run alone.</summary>
[CollectionDefinition("KernelStats", DisableParallelization = true)]
public class KernelStatsCollection;

/// <summary>
/// The streaming-store path is taken exactly when the option asks, the outputs share a line
/// offset, the shard is at least 512 KiB and the inputs form one group; the scalar peel before
/// the first aligned byte and the vector tail after the last must both be right, on every tier.
/// </summary>
[Collection("KernelStats")]
public class StreamingStoresTests
{
    private const int Threshold = 512 * 1024;

    public static IEnumerable<object[]> VectorTiers() =>
        KernelTestHelpers.AllTiers.Where(t => t != KernelTier.Scalar).Select(t => new object[] { t });

    [Theory]
    [MemberData(nameof(VectorTiers))]
    public void StreamsAtEveryPeelLength_AndMatchesReference(KernelTier tier)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;
        const int len = Threshold + 1;
        var rs = new ReedSolomon(4, 4, new ReedSolomonOptions { Kernel = tier, StreamingStores = true });
        foreach (int offset in new[] { 0, 1, 17, 63 })
        {
            var backing = ReedSolomon.AllocateShards(8, len + 64);
            foreach (var m in backing) m.Span.Fill(0xA5);
            var shards = backing.Select(m => m.Slice(offset, len)).ToArray();
            var rng = new Random(offset);
            for (int i = 0; i < 4; i++) rng.NextBytes(shards[i].Span);
            var plain = shards.Select(m => m.ToArray()).ToArray();
            var expected = TestData.ReferenceParity(rs, plain);

            long before = KernelStats.StreamCalls;
            rs.Encode(shards.Take(4).Select(m => (ReadOnlyMemory<byte>)m).ToArray(), shards.Skip(4).ToArray());
            Assert.True(KernelStats.StreamCalls == before + 1, $"tier={tier} offset={offset}: streaming path not taken");
            for (int p = 0; p < 4; p++)
                Assert.True(expected[p].AsSpan().SequenceEqual(shards[4 + p].Span), $"tier={tier} offset={offset} parity {p}");
            for (int i = 0; i < 8; i++)
            {
                Assert.True(backing[i].Span[..offset].IndexOfAnyExcept((byte)0xA5) < 0, $"tier={tier} offset={offset}: prefix of shard {i} written");
                Assert.True(backing[i].Span[(offset + len)..].IndexOfAnyExcept((byte)0xA5) < 0, $"tier={tier} offset={offset}: suffix of shard {i} written");
            }
        }
    }

    [Fact]
    public void DoesNotStream_BelowThreshold_WithUnequalOffsets_OrAcrossGroups()
    {
        if (ReedSolomon.BestSupportedKernel == KernelTier.Scalar) return;
        long before = KernelStats.StreamCalls;

        // One byte short of the threshold.
        var rs = new ReedSolomon(4, 4, new ReedSolomonOptions { StreamingStores = true });
        var below = ReedSolomon.AllocateShards(8, Threshold - 1);
        rs.Encode(below.Take(4).Select(m => (ReadOnlyMemory<byte>)m).ToArray(), below.Skip(4).ToArray());
        Assert.Equal(before, KernelStats.StreamCalls);

        // Parity buffers at different offsets within the cache line.
        var uneven = ReedSolomon.AllocateShards(8, Threshold + 64);
        var parity = uneven.Skip(4).Select((m, i) => m.Slice(i == 1 ? 1 : 0, Threshold + 1)).ToArray();
        rs.Encode(uneven.Take(4).Select(m => (ReadOnlyMemory<byte>)m.Slice(0, Threshold + 1)).ToArray(), parity);
        Assert.Equal(before, KernelStats.StreamCalls);

        // Two input groups (13 inputs): later groups reload the outputs, so no streaming.
        var wide = new ReedSolomon(13, 4, new ReedSolomonOptions { StreamingStores = true });
        var many = ReedSolomon.AllocateShards(17, Threshold + 1);
        wide.Encode(many.Take(13).Select(m => (ReadOnlyMemory<byte>)m).ToArray(), many.Skip(13).ToArray());
        Assert.Equal(before, KernelStats.StreamCalls);

        // Option off: never.
        var off = new ReedSolomon(4, 4);
        var aligned = ReedSolomon.AllocateShards(8, Threshold + 1);
        off.Encode(aligned.Take(4).Select(m => (ReadOnlyMemory<byte>)m).ToArray(), aligned.Skip(4).ToArray());
        Assert.Equal(before, KernelStats.StreamCalls);

        // Every one of those must still be correct.
        foreach (var (coder, buffers, data) in new[] { (rs, below, 4), (wide, many, 13), (off, aligned, 4) })
        {
            var expected = TestData.ReferenceParity(coder, buffers.Select(m => m.ToArray()).ToArray());
            for (int p = 0; p < 4; p++) Assert.True(expected[p].AsSpan().SequenceEqual(buffers[data + p].Span), $"{data}+4 parity {p}");
        }
    }

    [Fact]
    public void EveryEncodeOverload_MatchesPlain()
    {
        // Above the 512 KiB threshold, AllocateShards layout, so the streaming path is taken on
        // x86; every overload must give the same parity as the plain coder.
        const int len = 512 * 1024 + 100;
        var plain = new ReedSolomon(4, 4);
        var streaming = new ReedSolomon(4, 4, new ReedSolomonOptions { StreamingStores = true });
        var shards = TestData.RandomShards(8, len, 77);
        var expected = TestData.Clone(shards);
        plain.Encode(expected);

        var padded = ReedSolomon.AllocateShards(8, len);
        for (int i = 0; i < 4; i++) shards[i].CopyTo(padded[i]);
        long before = Internal.KernelStats.StreamCalls;
        streaming.Encode(padded.AsSpan(0, 4).ToArray().Select(m => (ReadOnlyMemory<byte>)m).ToArray(), padded.AsSpan(4, 4));
        for (int p = 0; p < 4; p++) Assert.True(expected[4 + p].AsSpan().SequenceEqual(padded[4 + p].Span), $"memories parity {p}");
        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && plain.Kernel != KernelTier.Scalar) Assert.True(Internal.KernelStats.StreamCalls > before, "streaming path not taken");

        var arrays = TestData.Clone(shards);
        streaming.Encode(arrays);
        for (int p = 0; p < 4; p++) Assert.Equal(expected[4 + p], arrays[4 + p]);

        var stripe = TestData.Stripe(shards);
        streaming.Encode(stripe.AsSpan(0, 4 * len), stripe.AsSpan(4 * len), len);
        for (int p = 0; p < 4; p++) Assert.True(expected[4 + p].AsSpan().SequenceEqual(stripe.AsSpan((4 + p) * len, len)), $"stripe parity {p}");

        // Not for reconstruct or incremental parity: those outputs are read again at once.
        var present = Enumerable.Repeat(true, 8).ToArray();
        present[0] = present[2] = false;
        var missing = expected.Select((s, i) => present[i] ? s.ToArray() : new byte[len]).ToArray();
        streaming.Reconstruct(missing, present);
        for (int i = 0; i < 8; i++) Assert.True(expected[i].AsSpan().SequenceEqual(missing[i]), $"shard {i}");
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(-1, 1)]
    [InlineData(200, 57)]
    [InlineData(256, 1)]
    public void Constructor_RejectsBadCounts(int data, int parity) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(data, parity));

}
