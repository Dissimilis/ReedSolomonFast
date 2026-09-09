namespace ReedSolomonFast.Tests;

/// <summary>Regressions for defects found in review before the first release: parallel chunk rounding, integer overflow on large stripes, Update bounds and duplicate indices, Join length rules, aliasing checks.</summary>
public class ReviewRegressionTests
{
    [Theory]
    [InlineData(961, 3)]
    [InlineData(1000, 7)]
    [InlineData(65_537, 16)]
    [InlineData(131_009, 5)]
    public void Parallel_CoversEveryByte_ForAwkwardLengths(int length, int workers)
    {
        // Floor division rounded up to 64 left bytes at the end of every shard untouched for 961 / 3.
        var serial = new ReedSolomon(6, 3);
        var parallel = new ReedSolomon(6, 3, new ReedSolomonOptions { MaxDegreeOfParallelism = workers, ParallelThresholdBytes = 1 });
        var shards = TestData.RandomShards(9, length, length);
        var a = TestData.Clone(shards);
        var b = TestData.Clone(shards);
        serial.Encode(a);
        parallel.Encode(b);
        for (int p = 0; p < 3; p++) Assert.True(a[6 + p].AsSpan().SequenceEqual(b[6 + p]), $"parity {p}");

        bool[] present = [false, true, true, false, true, true, true, false, true];
        var c = shards.Select((s, i) => present[i] ? a[i].ToArray() : new byte[length]).ToArray();
        parallel.Reconstruct(c, present);
        for (int i = 0; i < 9; i++) Assert.True(a[i].AsSpan().SequenceEqual(c[i]), $"shard {i}");
    }

    [Fact]
    public void Constructor_HugeCounts_DoNotOverflowTheLimitCheck()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(int.MaxValue, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, int.MaxValue));
    }

    [Fact]
    public void Update_RejectsDuplicateAndExcessIndices()
    {
        var rs = new ReedSolomon(3, 2);
        var parity = new Memory<byte>[] { new byte[4], new byte[4] };
        Assert.Throws<ArgumentException>(() => rs.Update([1, 1], [new byte[4], new byte[4]], [new byte[4], new byte[4]], parity));
        Assert.Throws<ArgumentException>(() => rs.Update([0, 1, 2, 0], new ReadOnlyMemory<byte>[4], new ReadOnlyMemory<byte>[4], parity));
    }

    [Fact]
    public void Update_AllIndicesAtOnce_MatchesEncode()
    {
        var rs = new ReedSolomon(12, 3);
        var shards = TestData.RandomShards(15, 500, 3);
        rs.Encode(shards);
        var changed = TestData.Clone(shards);
        for (int i = 0; i < 12; i++) new Random(100 + i).NextBytes(changed[i]);
        rs.Update(
            Enumerable.Range(0, 12).ToArray(),
            TestData.AsReadOnly(shards, 0, 12),
            TestData.AsReadOnly(changed, 0, 12),
            TestData.AsMemory(changed, 12, 3));
        var expected = TestData.ReferenceParity(rs, changed);
        for (int p = 0; p < 3; p++) Assert.Equal(expected[p], changed[12 + p]);
    }

    [Fact]
    public void Join_RejectsOutputLongerThanTheShardsHold()
    {
        var rs = new ReedSolomon(3, 1);
        byte[]?[] shards = [new byte[1], new byte[1], new byte[1], new byte[1]];
        Assert.Throws<ArgumentException>(() => rs.Join(shards, 4));
        Assert.Throws<ArgumentException>(() => rs.Join(TestData.AsReadOnly(shards!, 0, 3), 4, new byte[4]));
        var writer = new System.Buffers.ArrayBufferWriter<byte>();
        Assert.Throws<ArgumentException>(() => rs.Join(TestData.AsReadOnly(shards!, 0, 3), 4, writer));
    }

    [Fact]
    public void Join_NeedsOnlyTheShardsItReads()
    {
        var rs = new ReedSolomon(3, 1);
        byte[]?[] shards = [[1, 2, 3, 4], null, null, null];
        Assert.Equal(new byte[] { 1, 2, 3 }, rs.Join(shards, 3));
        var ex = Assert.Throws<InsufficientShardsException>(() => rs.Join(shards, 5));
        Assert.Equal(1, ex.Present);
        Assert.Equal(2, ex.Required);
    }

    [Fact]
    public void Encode_RejectsOutputsAliasingInputs()
    {
        var rs = new ReedSolomon(2, 1);
        var shared = new byte[8];
        Assert.Throws<ArgumentException>(() => rs.Encode(new[] { shared, new byte[8], shared }));

        var backing = new byte[24];
        Assert.Throws<ArgumentException>(() => rs.Encode(
            new ReadOnlyMemory<byte>[] { backing.AsMemory(0, 8), backing.AsMemory(8, 8) },
            new Memory<byte>[] { backing.AsMemory(4, 8) }));
        Assert.Throws<ArgumentException>(() => rs.Encode(backing.AsSpan(0, 16), backing.AsSpan(8, 8), 8));
        Assert.Throws<ArgumentException>(() => rs.Verify(
            new ReadOnlyMemory<byte>[] { backing.AsMemory(0, 8), backing.AsMemory(8, 8), backing.AsMemory(16, 8) },
            backing.AsSpan(16, 8)));

        // Disjoint slices of one buffer are fine.
        rs.Encode(backing.AsSpan(0, 16), backing.AsSpan(16, 8), 8);
    }

    [Fact]
    public void Reconstruct_RejectsTheSameArrayTwice()
    {
        var rs = new ReedSolomon(2, 1);
        var shared = new byte[8];
        Assert.Throws<ArgumentException>(() => rs.Reconstruct(new[] { shared, new byte[8], shared }, [true, true, false]));
    }

    [Fact]
    public void AllocateShards_RejectsOversizedStripe()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ReedSolomon.AllocateShards(256, int.MaxValue / 2));
    }

    [Fact]
    public void MatrixKind_DefaultIsVandermonde()
    {
        Assert.Equal(MatrixKind.Vandermonde, new ReedSolomonOptions().Matrix);
        Assert.Equal(0, (int)MatrixKind.Vandermonde);
    }
}
