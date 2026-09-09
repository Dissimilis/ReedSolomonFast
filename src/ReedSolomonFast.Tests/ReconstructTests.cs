namespace ReedSolomonFast.Tests;

public class ReconstructTests
{
    public static IEnumerable<object[]> Geometries() =>
        from g in new[] { (1, 1), (2, 1), (3, 2), (5, 3), (10, 4), (8, 8), (17, 3), (200, 56) }
        from kind in new[] { MatrixKind.Vandermonde, MatrixKind.Cauchy }
        select new object[] { g.Item1, g.Item2, kind };

    private static IEnumerable<bool[]> Patterns(int total, int parity) =>
        total <= 8 ? TestData.AllErasurePatterns(total, parity) : TestData.RandomErasurePatterns(total, parity, 60, total);

    [Theory]
    [MemberData(nameof(Geometries))]
    public void Reconstruct_PresentFlags_AllContainers_RoundTrip(int data, int parity, MatrixKind kind)
    {
        var rs = new ReedSolomon(data, parity, new ReedSolomonOptions { Matrix = kind });
        const int len = 257;
        var original = TestData.RandomShards(data + parity, len, data * 7 + parity);
        rs.Encode(original);

        foreach (bool[] present in Patterns(data + parity, parity))
        {
            var arrays = Erase(original, present);
            rs.Reconstruct(arrays, present);
            AssertShardsEqual(original, arrays);

            var memories = Erase(original, present);
            rs.Reconstruct(TestData.AsMemory(memories, 0, data + parity), present);
            AssertShardsEqual(original, memories);

            var stripe = TestData.Stripe(Erase(original, present));
            rs.Reconstruct(stripe, len, present);
            Assert.Equal(TestData.Stripe(original), stripe);
        }
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    public void Reconstruct_Allocating_RoundTrip(int data, int parity, MatrixKind kind)
    {
        var rs = new ReedSolomon(data, parity, new ReedSolomonOptions { Matrix = kind });
        const int len = 100;
        var original = TestData.RandomShards(data + parity, len, data * 3 + parity);
        rs.Encode(original);

        foreach (bool[] present in Patterns(data + parity, parity))
        {
            byte[]?[] arrays = original.Select((s, i) => present[i] ? s.ToArray() : null).ToArray();
            rs.Reconstruct(arrays);
            AssertShardsEqual(original, arrays!);

            var memories = original.Select((s, i) => present[i] ? (Memory<byte>)s.ToArray() : Memory<byte>.Empty).ToArray();
            rs.Reconstruct(memories.AsSpan());
            for (int i = 0; i < original.Length; i++) Assert.Equal(original[i], memories[i].ToArray());
        }
    }

    [Fact]
    public void ReconstructData_LeavesParityAlone()
    {
        var rs = new ReedSolomon(6, 3);
        var original = TestData.RandomShards(9, 64, 1);
        rs.Encode(original);
        bool[] present = [false, true, true, false, true, true, true, true, false];

        var arrays = Erase(original, present);
        rs.ReconstructData(arrays, present);
        for (int i = 0; i < 6; i++) Assert.Equal(original[i], arrays[i]);
        Assert.All(arrays[8], b => Assert.Equal(0, b));

        byte[]?[] nullable = original.Select((s, i) => present[i] ? s.ToArray() : null).ToArray();
        rs.ReconstructData(nullable);
        for (int i = 0; i < 6; i++) Assert.Equal(original[i], nullable[i]);
        Assert.Null(nullable[8]);
    }

    [Fact]
    public void ReconstructSome_RebuildsOnlyRequested()
    {
        var rs = new ReedSolomon(4, 3);
        var original = TestData.RandomShards(7, 33, 2);
        rs.Encode(original);
        bool[] present = [false, true, false, true, true, false, true];
        bool[] required = [true, false, false, false, false, true, false];

        var arrays = Erase(original, present);
        rs.ReconstructSome(arrays, present, required);
        Assert.Equal(original[0], arrays[0]);
        Assert.Equal(original[5], arrays[5]);
        Assert.All(arrays[2], b => Assert.Equal(0, b));

        // A DataShards-long required array is accepted too.
        var arrays2 = Erase(original, present);
        rs.ReconstructSome(arrays2, present, [false, false, true, false]);
        Assert.Equal(original[2], arrays2[2]);
        Assert.All(arrays2[0], b => Assert.Equal(0, b));
    }

    [Fact]
    public void Reconstruct_NothingMissing_IsNoOp()
    {
        var rs = new ReedSolomon(3, 2);
        var shards = TestData.RandomShards(5, 10, 3);
        var copy = TestData.Clone(shards);
        rs.Reconstruct(copy, Enumerable.Repeat(true, 5).ToArray());
        AssertShardsEqual(shards, copy);
        Assert.Equal(0, rs.InversionCacheMisses);
    }

    [Fact]
    public void Reconstruct_TooFewPresent_ThrowsAndWritesNothing()
    {
        var rs = new ReedSolomon(4, 2);
        var shards = TestData.RandomShards(6, 20, 4);
        var copy = TestData.Clone(shards);
        bool[] present = [true, false, false, true, true, false];
        var ex = Assert.Throws<InsufficientShardsException>(() => rs.Reconstruct(copy, present));
        Assert.Equal(3, ex.Present);
        Assert.Equal(4, ex.Required);
        AssertShardsEqual(shards, copy);

        Assert.False(rs.TryReconstruct(copy, present));
        Assert.False(rs.TryReconstruct(shards.Select((s, i) => present[i] ? s : null).ToArray()));
        Assert.False(rs.TryReconstruct(TestData.Stripe(copy), 20, present));
    }

    [Fact]
    public void Reconstruct_WrongPresentLength_Throws()
    {
        var rs = new ReedSolomon(2, 2);
        var shards = TestData.RandomShards(4, 8, 5);
        Assert.Throws<ArgumentException>(() => rs.Reconstruct(shards, new bool[3]));
        Assert.Throws<ArgumentException>(() => rs.ReconstructSome(shards, new bool[4], new bool[3]));
    }

    [Fact]
    public void Reconstruct_SamePattern_HitsCache()
    {
        var rs = new ReedSolomon(5, 2);
        var original = TestData.RandomShards(7, 40, 6);
        rs.Encode(original);
        bool[] present = [true, false, true, true, false, true, true];

        var a = Erase(original, present);
        rs.Reconstruct(a, present);
        Assert.Equal(1, rs.InversionCacheMisses);
        var b = Erase(original, present);
        rs.Reconstruct(b, present);
        Assert.Equal(1, rs.InversionCacheMisses);
        AssertShardsEqual(original, b);

        var uncached = new ReedSolomon(5, 2, new ReedSolomonOptions { InversionCache = false });
        var c = Erase(original, present);
        uncached.Reconstruct(c, present);
        AssertShardsEqual(original, c);
        Assert.Equal(-1, uncached.InversionCacheMisses);
    }

    [Fact]
    public void Reconstruct_CachedPattern_DoesNotAllocate()
    {
        var rs = new ReedSolomon(10, 4);
        var original = TestData.RandomShards(14, 4096, 7);
        rs.Encode(original);
        bool[] present = Enumerable.Repeat(true, 14).ToArray();
        present[2] = present[7] = present[12] = false;
        var shards = Erase(original, present);
        rs.Reconstruct(shards, present);

        long before = GC.GetAllocatedBytesForCurrentThread();
        rs.Reconstruct(shards, present);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        AssertShardsEqual(original, shards);
    }

    /// <summary>
    /// 255 and 256 shards, with erasures on both sides of every 64-bit word of the shard mask
    /// (bits 63/64, 127/128, 191/192), data and parity mixed; and one erasure too many, which
    /// must throw and leave every buffer as it was.
    /// </summary>
    [Theory]
    [InlineData(199, 56)]
    [InlineData(200, 56)]
    public void Reconstruct_MaxShards_AcrossEveryMaskWord(int data, int parity)
    {
        const int len = 65;
        int total = data + parity;
        var rs = new ReedSolomon(data, parity);
        var original = TestData.RandomShards(total, len, data);
        rs.Encode(original);

        int[] missing = [0, 63, 64, 127, 128, 191, 192, data - 1, data, total - 1];
        bool[] present = Enumerable.Repeat(true, total).ToArray();
        foreach (int i in missing) present[i] = false;
        var shards = Erase(original, present);
        rs.Reconstruct(shards, present);
        AssertShardsEqual(original, shards);

        bool[] tooMany = Enumerable.Repeat(true, total).ToArray();
        for (int i = 0; i <= parity; i++) tooMany[i] = false;
        var untouched = Erase(original, tooMany);
        var snapshot = TestData.Clone(untouched);
        var ex = Assert.Throws<InsufficientShardsException>(() => rs.Reconstruct(untouched, tooMany));
        Assert.Equal(total - parity - 1, ex.Present);
        Assert.Equal(data, ex.Required);
        Assert.False(rs.TryReconstruct(untouched, tooMany));
        AssertShardsEqual(snapshot, untouched);
    }

    [Fact]
    public void Reconstruct_ConcurrentUse_IsCorrect()
    {
        var rs = new ReedSolomon(8, 4);
        var original = TestData.RandomShards(12, 2048, 8);
        rs.Encode(original);
        var patterns = TestData.RandomErasurePatterns(12, 4, 32, 99).ToArray();

        Parallel.For(0, 64, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            bool[] present = patterns[i % patterns.Length];
            var shards = Erase(original, present);
            rs.Reconstruct(shards, present);
            AssertShardsEqual(original, shards);

            var copy = TestData.Clone(original);
            rs.Encode(copy);
            AssertShardsEqual(original, copy);
        });
    }

    [Fact]
    public void Reconstruct_ZeroLengthShards_IsNoOp()
    {
        var rs = new ReedSolomon(2, 2);
        var shards = new byte[4][] { [], [], [], [] };
        rs.Reconstruct(shards, [true, false, true, false]);
    }

    private static byte[][] Erase(byte[][] shards, bool[] present) =>
        shards.Select((s, i) => present[i] ? s.ToArray() : new byte[s.Length]).ToArray();

    private static void AssertShardsEqual(byte[][] expected, byte[][] actual)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].AsSpan().SequenceEqual(actual[i]), $"shard {i} differs");
    }
}
