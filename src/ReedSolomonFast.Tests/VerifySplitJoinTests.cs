using System.Buffers;

namespace ReedSolomonFast.Tests;

public class VerifySplitJoinTests
{
    [Fact]
    public void Verify_TrueAfterEncode_FalseAfterAnyBitFlip()
    {
        var rs = new ReedSolomon(5, 3);
        var shards = TestData.RandomShards(8, 500, 1);
        rs.Encode(shards);

        Assert.True(rs.Verify(shards));
        Assert.True(rs.Verify(TestData.AsReadOnly(shards, 0, 8)));
        Assert.True(rs.Verify(TestData.AsReadOnly(shards, 0, 8), new byte[3 * 500]));
        Assert.True(rs.Verify(TestData.Stripe(shards), 500));
        Assert.True(rs.Verify(TestData.Stripe(shards), 500, new byte[3 * 500]));

        for (int i = 0; i < 8; i++)
        {
            var corrupt = TestData.Clone(shards);
            corrupt[i][i * 37 % 500] ^= 0x10;
            Assert.False(rs.Verify(corrupt));
            Assert.False(rs.Verify(TestData.AsReadOnly(corrupt, 0, 8)));
            Assert.False(rs.Verify(TestData.Stripe(corrupt), 500));
        }
    }

    [Fact]
    public void Verify_LongShards_CorruptionInAnyPiece_IsFound()
    {
        // Verification runs in 64 KiB pieces; a flipped byte in the first, a middle and the last
        // piece, and in the last byte, must all be found, on every overload.
        var rs = new ReedSolomon(4, 2);
        const int len = 200_001;
        var shards = TestData.RandomShards(6, len, 3);
        rs.Encode(shards);
        Assert.True(rs.Verify(shards));
        Assert.True(rs.Verify(TestData.AsReadOnly(shards, 0, 6)));
        Assert.True(rs.Verify(TestData.Stripe(shards), len));

        foreach (int position in new[] { 0, 70_000, 131_072, len - 1 })
            foreach (int shard in new[] { 0, 3, 4, 5 })
            {
                var corrupt = TestData.Clone(shards);
                corrupt[shard][position] ^= 1;
                Assert.False(rs.Verify(corrupt), $"arrays shard {shard} at {position}");
                Assert.False(rs.Verify(TestData.AsReadOnly(corrupt, 0, 6)), $"memories shard {shard} at {position}");
                Assert.False(rs.Verify(TestData.Stripe(corrupt), len), $"stripe shard {shard} at {position}");
            }
    }

    [Fact]
    public void Verify_ScratchTooSmall_Throws()
    {
        var rs = new ReedSolomon(2, 2);
        var shards = TestData.RandomShards(4, 10, 2);
        Assert.Throws<ArgumentException>(() => rs.Verify(TestData.AsReadOnly(shards, 0, 4), new byte[19]));
        Assert.Throws<ArgumentException>(() => rs.Verify(TestData.Stripe(shards), 10, new byte[19]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12345)]
    public void Split_Encode_Erase_Reconstruct_Join_RoundTrips(int length)
    {
        var rs = new ReedSolomon(5, 2);
        var file = new byte[length];
        new Random(length).NextBytes(file);

        byte[][] shards = rs.Split(file);
        Assert.Equal(7, shards.Length);
        Assert.Equal(rs.GetShardLength(length), shards[0].Length);
        Assert.Equal(rs.GetShardLength(length) * 5 - length, rs.GetPaddingLength(length));
        rs.Encode(shards);

        shards[1] = null!;
        shards[6] = null!;
        rs.Reconstruct(shards!);
        Assert.Equal(file, rs.Join(shards, length));

        var dest = new byte[length];
        rs.Join(TestData.AsReadOnly(shards, 0, 5), length, dest);
        Assert.Equal(file, dest);

        var writer = new ArrayBufferWriter<byte>();
        rs.Join(TestData.AsReadOnly(shards, 0, 7), length, writer);
        Assert.Equal(file, writer.WrittenSpan.ToArray());
    }

    [Fact]
    public void Split_IntoStripe_UsesCallerMemory()
    {
        var rs = new ReedSolomon(3, 2);
        var file = new byte[1000];
        new Random(5).NextBytes(file);
        var stripe = ReedSolomon.AllocateShards(1, 5 * 334)[0];
        Memory<byte>[] shards = rs.Split(file, stripe);
        Assert.Equal(5, shards.Length);
        Assert.Equal(334, shards[0].Length);
        Assert.Equal(file.AsSpan(0, 334).ToArray(), shards[0].ToArray());
        Assert.Equal(0, shards[2].Span[333]);
        rs.Encode(TestData.AsReadOnly(shards.Select(m => m.ToArray()).ToArray(), 0, 3), shards[3..]);
        Assert.True(rs.Verify(shards.Select(m => (ReadOnlyMemory<byte>)m).ToArray()));
    }

    [Fact]
    public void Join_MissingDataShard_Throws()
    {
        var rs = new ReedSolomon(3, 1);
        byte[]?[] shards = [new byte[4], null, new byte[4], new byte[4]];
        var ex = Assert.Throws<InsufficientShardsException>(() => rs.Join(shards, 12));
        Assert.Equal(1, ex.Present);   // shard 0 is there
        Assert.Equal(2, ex.Required);  // Join needed shards 0 and 1 before it could stop
        Assert.Throws<ArgumentException>(() => rs.Join(new byte[2][], 1));
    }

    [Fact]
    public void AllocateShards_AreAlignedAndDistinct()
    {
        var shards = ReedSolomon.AllocateShards(4, 1000);
        Assert.Equal(4, shards.Length);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(1000, shards[i].Length);
            unsafe
            {
                using var h = shards[i].Pin();
                Assert.Equal(0, (long)((nuint)h.Pointer & 63));
            }

            shards[i].Span.Fill((byte)i);
        }

        for (int i = 0; i < 4; i++) Assert.All(shards[i].ToArray(), b => Assert.Equal((byte)i, b));
    }
}
