using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

/// <summary>
/// Every kernel tier must produce exactly what the naive reference produces, for every shape that
/// exercises a different code path: register-blocking remainders (1..5 outputs, 9 to force two
/// groups), single and many inputs, lengths on both sides of every vector width, and buffers
/// offset by 1..3 bytes so no load or store is aligned.
/// </summary>
public unsafe class KernelTests
{
    public static readonly int[] Lengths = [0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 4096, 65537];

    public static IEnumerable<object[]> Shapes()
    {
        foreach (var tier in KernelTestHelpers.AllTiers)
            foreach (int srcCount in new[] { 1, 2, 3, 7, 16 })
                foreach (int dstCount in new[] { 1, 2, 3, 4, 5, 9 })
                    yield return [tier, srcCount, dstCount];
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void DotProduct_MatchesReference(KernelTier tier, int srcCount, int dstCount)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;

        var rng = new Random(srcCount * 100 + dstCount);
        foreach (int len in Lengths)
        {
            foreach (int offset in new[] { 0, 1, 3 })
            {
                var coef = KernelTestHelpers.RandomBytes(dstCount * srcCount, rng);
                // A zero coefficient must be handled (skipped or multiplied, either way correct).
                if (srcCount > 1) coef[1] = 0;
                var srcs = new byte[srcCount][];
                for (int s = 0; s < srcCount; s++) srcs[s] = KernelTestHelpers.RandomBytes(len + offset, rng);
                var dsts = new byte[dstCount][];
                for (int d = 0; d < dstCount; d++) dsts[d] = KernelTestHelpers.RandomBytes(len + offset, rng);

                var expected = KernelTestHelpers.ReferenceDotProduct(srcs, coef, dstCount, len, offset);
                KernelTestHelpers.Invoke(tier, srcs, dsts, coef, len, offset);

                for (int d = 0; d < dstCount; d++)
                    Assert.True(expected[d].AsSpan().SequenceEqual(dsts[d].AsSpan(offset, len)),
                        $"tier={tier} srcs={srcCount} dsts={dstCount} len={len} offset={offset} dst={d}");
            }
        }
    }

    /// <summary>
    /// EncodeShard and Update run the kernel with the destination aliasing source 0. That is only
    /// safe while every source sits in the first input group; this pins the contract for every tier.
    /// </summary>
    [Theory]
    [MemberData(nameof(Tiers))]
    public void DotProduct_InPlace_WithinFirstGroup_IsCorrect(KernelTier tier)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;
        var rng = new Random(77);
        foreach (int srcCount in new[] { 1, 2, 3, Kernel.MaxInPlaceSources })
            foreach (int len in new[] { 1, 64, 1000, 70_000 })
            {
                var coef = KernelTestHelpers.RandomBytes(srcCount, rng);
                var srcs = new byte[srcCount][];
                for (int s = 0; s < srcCount; s++) srcs[s] = KernelTestHelpers.RandomBytes(len, rng);
                var expected = KernelTestHelpers.ReferenceDotProduct(srcs, coef, 1, len, 0);

                // The destination is the same array as source 0.
                KernelTestHelpers.Invoke(tier, srcs, [srcs[0]], coef, len, 0);
                Assert.True(expected[0].AsSpan().SequenceEqual(srcs[0]), $"tier={tier} srcs={srcCount} len={len}");
            }
    }

    public static IEnumerable<object[]> Tiers() => KernelTestHelpers.AllTiers.Select(t => new object[] { t });

    [Fact]
    public void EveryTier_HasAnIsSupportedAnswer()
    {
        foreach (var tier in KernelTestHelpers.AllTiers)
            _ = KernelTestHelpers.IsSupported(tier);
        Assert.True(KernelTestHelpers.IsSupported(KernelTier.Scalar));
        Assert.True(KernelTestHelpers.IsSupported(Kernel.Best()));
        Assert.Equal(Kernel.Best(), ReedSolomon.BestSupportedKernel);
    }

    /// <summary>
    /// Accumulating into existing outputs, with the inputs spanning one, two and more groups
    /// (12 is one full group, 13 and 16 are two), for every output block size including the
    /// eight-output body: dst ^= sum over sources, from a misaligned offset.
    /// </summary>
    [Theory]
    [MemberData(nameof(Tiers))]
    public void DotProduct_Accumulate_AcrossGroups_MatchesReference(KernelTier tier)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;
        var rng = new Random(77);
        const int offset = 3, len = 1000;
        foreach (int srcCount in new[] { 1, 12, 13, 16 })
            foreach (int dstCount in new[] { 1, 4, 8, 9 })
            {
                var srcs = Enumerable.Range(0, srcCount).Select(_ => KernelTestHelpers.RandomBytes(offset + len, rng)).ToArray();
                var dsts = Enumerable.Range(0, dstCount).Select(_ => KernelTestHelpers.RandomBytes(offset + len, rng)).ToArray();
                var initial = dsts.Select(d => d.ToArray()).ToArray();
                var coef = KernelTestHelpers.RandomBytes(srcCount * dstCount, rng);
                var expected = KernelTestHelpers.ReferenceDotProduct(srcs, coef, dstCount, len, offset);

                KernelTestHelpers.Invoke(tier, srcs, dsts, coef, len, offset, accumulate: true);
                for (int d = 0; d < dstCount; d++)
                {
                    Assert.True(initial[d].AsSpan(0, offset).SequenceEqual(dsts[d].AsSpan(0, offset)), $"tier={tier} srcs={srcCount} dsts={dstCount}: prefix written");
                    for (int i = 0; i < len; i++)
                        Assert.True((byte)(initial[d][offset + i] ^ expected[d][i]) == dsts[d][offset + i], $"tier={tier} srcs={srcCount} dsts={dstCount} dst={d} byte={i}");
                }
            }
    }

    /// <summary>
    /// Every coefficient through the real SIMD path: one source holding every byte value, each of
    /// the 256 multipliers, checked against shift-and-add. A table built wrong for one coefficient
    /// (nibble split, GFNI matrix) shows here and nowhere else; the guard bytes catch an overrun.
    /// </summary>
    [Theory]
    [MemberData(nameof(Tiers))]
    public void DotProduct_EveryCoefficient_MatchesReference(KernelTier tier)
    {
        if (!KernelTestHelpers.IsSupported(tier)) return;
        const int offset = 1, len = 257, guard = 16;
        var src = new byte[offset + len + guard];
        for (int i = 0; i < src.Length; i++) src[i] = (byte)i;
        for (int c = 0; c < 256; c++)
        {
            var dst = new byte[offset + len + guard];
            Array.Fill(dst, (byte)0xA5);
            KernelTestHelpers.Invoke(tier, [src], [dst], [(byte)c], len, offset);
            Assert.Equal((byte)0xA5, dst[0]);
            for (int i = 0; i < guard; i++) Assert.True(dst[offset + len + i] == 0xA5, $"tier={tier} coef={c}: guard byte {i} overwritten");
            for (int i = 0; i < len; i++)
                Assert.True(Gf256Reference.Multiply((byte)c, src[offset + i]) == dst[offset + i], $"tier={tier} coef={c} byte={i}");
        }
    }
}
