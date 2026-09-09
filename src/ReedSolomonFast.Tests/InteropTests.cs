using ReedSolomonFast.Internal;
using Egbakou = global::ReedSolomon.NET.ReedSolomon;
using EgbakouGalois = global::ReedSolomon.NET.Galois;
using WittebornCoder = global::Witteborn.ReedSolomon.ReedSolomon;
using WittebornTables = global::Witteborn.ReedSolomon.GaloisTables;

namespace ReedSolomonFast.Tests;

/// <summary>
/// Cross-verification against two independent implementations of the same code: ReedSolomon.NET
/// (Egbakou) and the Witteborn package, both ports of Backblaze's JavaReedSolomon. The field
/// tables, the parity of every geometry, the reconstruction of every erasure pattern tried, and
/// the verify verdicts must all agree byte for byte. A disagreement here is a real defect in
/// one of the three, and the shift-and-add reference in <see cref="Gf256Reference"/> decides.
/// </summary>
public class InteropTests
{
    public static IEnumerable<object[]> Geometries() =>
        new[] { (1, 1), (2, 1), (3, 2), (4, 2), (10, 4), (8, 8), (17, 3), (64, 64), (200, 56), (255, 1), (1, 255) }
            .Select(g => new object[] { g.Item1, g.Item2 });

    [Fact]
    public void FieldTables_MatchBothPackages()
    {
        byte[] egExp = EgbakouGalois.GenerateExpTable();
        short[] egLog = EgbakouGalois.GenerateLogTable();
        sbyte[] wiExp = WittebornTables.EXP_TABLE;
        short[] wiLog = WittebornTables.LOG_TABLE;

        for (int i = 0; i < 255; i++)
        {
            Assert.Equal(egExp[i], Gf256.Exp[i]);
            Assert.Equal((byte)wiExp[i], Gf256.Exp[i]);
        }

        for (int a = 1; a < 256; a++)
        {
            Assert.Equal(egLog[a], Gf256.Log[a]);
            Assert.Equal(wiLog[a], Gf256.Log[a]);
        }

        for (int a = 0; a < 256; a++)
            for (int b = 0; b < 256; b++)
                Assert.Equal(EgbakouGalois.Multiply((byte)a, (byte)b), Gf256.Multiply((byte)a, (byte)b));
    }

    [Fact]
    public void FieldTables_MatchBackblazeGoldenValues()
    {
        // The first entries of Backblaze's EXP_TABLE and LOG_TABLE for polynomial 0x11D, and the
        // spot values its GaloisTest checks.
        byte[] exp = [1, 2, 4, 8, 16, 32, 64, 128, 29, 58, 116, 232, 205, 135, 19, 38];
        short[] log = [-1, 0, 1, 25, 2, 50, 26, 198, 3, 223, 51, 238, 27, 104, 199, 75];
        for (int i = 0; i < exp.Length; i++) Assert.Equal(exp[i], Gf256.Exp[i]);
        for (int i = 1; i < log.Length; i++) Assert.Equal(log[i], Gf256.Log[i]);

        Assert.Equal(12, Gf256.Multiply(3, 4));
        Assert.Equal(21, Gf256.Multiply(7, 7));
        Assert.Equal(41, Gf256.Multiply(23, 45));
        Assert.Equal(128, Power(2, 7));
        Assert.Equal(235, Power(5, 20));
        Assert.Equal(43, Power(13, 7));
    }

    [Fact]
    public void Encode_MatchesBackblazePublishedVector()
    {
        // From Backblaze's JavaReedSolomon ReedSolomonTest: 5+5, two bytes per shard.
        byte[][] shards = [[0, 1], [4, 5], [2, 3], [6, 7], [8, 9], new byte[2], new byte[2], new byte[2], new byte[2], new byte[2]];
        new ReedSolomon(5, 5).Encode(shards);
        Assert.Equal(new byte[] { 12, 13 }, shards[5]);
        Assert.Equal(new byte[] { 10, 11 }, shards[6]);
        Assert.Equal(new byte[] { 14, 15 }, shards[7]);
        Assert.Equal(new byte[] { 90, 91 }, shards[8]);
        Assert.Equal(new byte[] { 94, 95 }, shards[9]);
    }

    private static byte Power(byte a, int n)
    {
        byte r = 1;
        for (int i = 0; i < n; i++) r = Gf256.Multiply(r, a);
        return r;
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    public void Encode_ByteIdentical_ToBothPackages(int data, int parity)
    {
        var egbakou = Egbakou.Create(data, parity);
        var witteborn = new WittebornCoder(data, parity);
        var rs = new ReedSolomon(data, parity);
        foreach (int len in Lengths(data + parity))
        {
            var shards = TestData.RandomShards(data + parity, len, len ^ (data << 8) ^ parity);
            var ours = TestData.Clone(shards);
            var theirs = TestData.Clone(shards);
            var others = TestData.Clone(shards);
            rs.Encode(ours);
            egbakou.EncodeParity(theirs, 0, len);
            witteborn.EncodeParity(others, 0, len);
            for (int p = 0; p < parity; p++)
            {
                Assert.True(theirs[data + p].AsSpan().SequenceEqual(ours[data + p]), $"{data}+{parity} len {len} parity {p}: ReedSolomon.NET differs");
                Assert.True(others[data + p].AsSpan().SequenceEqual(ours[data + p]), $"{data}+{parity} len {len} parity {p}: Witteborn differs");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Geometries))]
    public void Reconstruct_MatchesBothPackages_OnRandomErasures(int data, int parity)
    {
        var egbakou = Egbakou.Create(data, parity);
        var witteborn = new WittebornCoder(data, parity);
        var rs = new ReedSolomon(data, parity);
        int total = data + parity;
        int len = total > 100 ? 100 : 1001;
        var shards = TestData.RandomShards(total, len, 99 + total);
        rs.Encode(shards);

        var patterns = TestData.RandomErasurePatterns(total, parity, 12, total).ToList();
        patterns.Add(ParityOnly(data, parity));
        patterns.Add(DataOnly(data, parity));
        foreach (bool[] present in patterns)
        {
            var ours = Erase(shards, present);
            var theirs = Erase(shards, present);
            var others = Erase(shards, present);
            rs.Reconstruct(ours, present);
            egbakou.DecodeMissing(theirs, (bool[])present.Clone(), 0, len);
            witteborn.DecodeMissing(others, (bool[])present.Clone(), 0, len);
            for (int i = 0; i < total; i++)
            {
                Assert.True(shards[i].AsSpan().SequenceEqual(ours[i]), $"{data}+{parity} shard {i}: ours differs from the original");
                Assert.True(shards[i].AsSpan().SequenceEqual(theirs[i]), $"{data}+{parity} shard {i}: ReedSolomon.NET differs from the original");
                Assert.True(shards[i].AsSpan().SequenceEqual(others[i]), $"{data}+{parity} shard {i}: Witteborn differs from the original");
            }
        }
    }

    [Theory]
    [InlineData(3, 2)]
    [InlineData(10, 4)]
    [InlineData(8, 8)]
    [InlineData(17, 3)]
    public void Verify_AgreesWithBothPackages(int data, int parity)
    {
        var egbakou = Egbakou.Create(data, parity);
        var witteborn = new WittebornCoder(data, parity);
        var rs = new ReedSolomon(data, parity);
        const int len = 2003;
        var shards = TestData.RandomShards(data + parity, len, 5);
        rs.Encode(shards);
        Assert.True(rs.Verify(shards));
        Assert.True(egbakou.IsParityCorrect(shards, 0, len));
        Assert.True(witteborn.IsParityCorrect(shards, 0, len));

        var rng = new Random(6);
        for (int i = 0; i < data + parity; i++)
        {
            var corrupt = TestData.Clone(shards);
            corrupt[i][rng.Next(len)] ^= (byte)(1 + rng.Next(255));
            Assert.False(rs.Verify(corrupt), $"shard {i}: ours accepted a corrupt stripe");
            Assert.False(egbakou.IsParityCorrect(corrupt, 0, len), $"shard {i}: ReedSolomon.NET accepted a corrupt stripe");
            Assert.False(witteborn.IsParityCorrect(corrupt, 0, len), $"shard {i}: Witteborn accepted a corrupt stripe");
        }
    }

    [Theory]
    [InlineData(4, 2)]
    [InlineData(10, 4)]
    public void Reconstruct_AcceptsParityWrittenByOtherPackages(int data, int parity)
    {
        // Parity produced by another implementation must reconstruct here, and ours must
        // reconstruct there: the wire format is the coding matrix, and it is shared.
        const int len = 777;
        var shards = TestData.RandomShards(data + parity, len, 8);
        var theirs = TestData.Clone(shards);
        Egbakou.Create(data, parity).EncodeParity(theirs, 0, len);
        var rs = new ReedSolomon(data, parity);

        var present = Enumerable.Repeat(true, data + parity).ToArray();
        for (int i = 0; i < parity; i++) present[i * 2 % (data + parity)] = false;
        var ours = Erase(theirs, present);
        rs.Reconstruct(ours, present);
        for (int i = 0; i < data + parity; i++) Assert.True(theirs[i].AsSpan().SequenceEqual(ours[i]), $"shard {i}");

        var mine = TestData.Clone(shards);
        rs.Encode(mine);
        var back = Erase(mine, present);
        new WittebornCoder(data, parity).DecodeMissing(back, (bool[])present.Clone(), 0, len);
        for (int i = 0; i < data + parity; i++) Assert.True(mine[i].AsSpan().SequenceEqual(back[i]), $"shard {i} through Witteborn");
    }

    private static IEnumerable<int> Lengths(int total) =>
        total > 100 ? [1, 64, 65, 200] : [1, 31, 64, 65, 1000, 4097];

    private static bool[] ParityOnly(int data, int parity)
    {
        var present = Enumerable.Repeat(true, data + parity).ToArray();
        for (int i = data; i < data + parity; i++) present[i] = false;
        return present;
    }

    private static bool[] DataOnly(int data, int parity)
    {
        var present = Enumerable.Repeat(true, data + parity).ToArray();
        for (int i = 0; i < Math.Min(data, parity); i++) present[i] = false;
        return present;
    }

    private static byte[][] Erase(byte[][] shards, bool[] present) =>
        shards.Select((s, i) => present[i] ? s.ToArray() : new byte[s.Length]).ToArray();
}
