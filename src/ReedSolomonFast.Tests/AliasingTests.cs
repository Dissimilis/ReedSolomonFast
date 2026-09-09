namespace ReedSolomonFast.Tests;

/// <summary>
/// Outputs must never overlap inputs, or each other: the kernel writes each output while other
/// outputs still read the inputs, and a later input group reloads the outputs. Every entry point
/// that takes caller buffers must refuse the overlap before writing anything.
/// </summary>
public class AliasingTests
{
    [Fact]
    public void EncodeShard_RejectsShardParityAndParityParityOverlap()
    {
        const int len = 65;
        var rs = new ReedSolomon(3, 2);
        var b = new byte[4 * len];

        Assert.Throws<ArgumentException>(() => rs.EncodeShard(0, b.AsSpan(0, len), new Memory<byte>[] { b.AsMemory(33, len), b.AsMemory(2 * len, len) }));
        Assert.Throws<ArgumentException>(() => rs.EncodeShard(0, new byte[len], new Memory<byte>[] { b.AsMemory(0, len), b.AsMemory(32, len) }));
        Assert.Throws<ArgumentException>(() => rs.EncodeShard(0, b.AsSpan(0, len), b.AsSpan(33, 2 * len), len));

        // Adjacent, not overlapping: fine.
        rs.EncodeShard(0, b.AsSpan(0, len), new Memory<byte>[] { b.AsMemory(len, len), b.AsMemory(2 * len, len) });
        rs.EncodeShard(0, b.AsSpan(0, len), b.AsSpan(len, 2 * len), len);
    }

    [Fact]
    public void Update_RejectsParityOverlap_AllowsOldEqualsNew()
    {
        const int len = 65;
        var rs = new ReedSolomon(3, 2);
        var b = new byte[5 * len];

        Assert.Throws<ArgumentException>(() => rs.Update([1], [b.AsMemory(0, len)], [b.AsMemory(len, len)], [b.AsMemory(32, len), b.AsMemory(3 * len, len)]));
        Assert.Throws<ArgumentException>(() => rs.Update([1], [b.AsMemory(0, len)], [b.AsMemory(len, len)], [b.AsMemory(len + 1, len), b.AsMemory(3 * len, len)]));
        Assert.Throws<ArgumentException>(() => rs.Update([1], [b.AsMemory(0, len)], [b.AsMemory(len, len)], [b.AsMemory(2 * len, len), b.AsMemory(2 * len + 7, len)]));

        // Identical old and new is a valid no-op.
        var parity = TestData.RandomShards(2, len, 8);
        var before = TestData.Clone(parity);
        ReadOnlyMemory<byte> same = new byte[len];
        rs.Update([1], [same], [same], TestData.AsMemory(parity, 0, 2));
        TestData.AssertShardsEqual(before, parity);
    }

    [Fact]
    public void Reconstruct_CallerBuffers_RejectMissingOverlappingPresent()
    {
        // Thirteen inputs make two groups, so an aliased source in the second group would be
        // overwritten before it is read; the check must fire regardless.
        const int len = 65;
        var rs = new ReedSolomon(13, 2);
        var original = TestData.RandomShards(15, len, 1);
        rs.Encode(original);

        var memories = TestData.AsMemory(TestData.Clone(original), 0, 15);
        memories[0] = memories[13];
        bool[] present = Enumerable.Repeat(true, 15).ToArray();
        present[0] = false;
        Assert.Throws<ArgumentException>(() => rs.Reconstruct(memories, present));
        Assert.Throws<ArgumentException>(() => rs.ReconstructData(memories, present));
        Assert.Throws<ArgumentException>(() => rs.TryReconstruct(memories, present));

        // Two missing shards sharing a buffer.
        var backing = new byte[2 * len];
        var two = TestData.AsMemory(TestData.Clone(original), 0, 15);
        two[0] = backing.AsMemory(0, len);
        two[1] = backing.AsMemory(len - 1, len);
        present[1] = false;
        Assert.Throws<ArgumentException>(() => rs.Reconstruct(two, present));

        // The rule covers shards the plan reads. A missing shard aliasing a present parity shard
        // the plan does not need (shard 14 here: 13 inputs suffice) is allowed and harmless, and
        // the result is still right.
        var spare = TestData.AsMemory(TestData.Clone(original), 0, 15);
        bool[] one = Enumerable.Repeat(true, 15).ToArray();
        one[0] = false;
        spare[0] = spare[14];
        rs.Reconstruct(spare, one);
        Assert.True(original[0].AsSpan().SequenceEqual(spare[0].Span));
    }

    [Fact]
    public void Join_RejectsDestinationOverlappingAShard()
    {
        var rs = new ReedSolomon(3, 1);
        var b = new byte[20];
        new Random(2).NextBytes(b);
        var shards = new ReadOnlyMemory<byte>[] { b.AsMemory(0, 4), b.AsMemory(4, 4), b.AsMemory(8, 4) };
        Assert.Throws<ArgumentException>(() => rs.Join(shards, 12, b.AsSpan(2, 12)));
        Assert.Throws<ArgumentException>(() => rs.Join(shards, 12, b.AsSpan(6, 12)));

        var dest = new byte[12];
        rs.Join(shards, 12, dest);
        Assert.True(b.AsSpan(0, 12).SequenceEqual(dest));
    }
}
