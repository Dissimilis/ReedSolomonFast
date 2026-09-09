using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

internal static class TestData
{
    public static byte[][] RandomShards(int count, int length, int seed)
    {
        var rng = new Random(seed);
        var shards = new byte[count][];
        for (int i = 0; i < count; i++)
        {
            shards[i] = new byte[length];
            rng.NextBytes(shards[i]);
        }

        return shards;
    }

    /// <summary>Parity computed by the naive reference for the coder's own matrix.</summary>
    public static byte[][] ReferenceParity(ReedSolomon rs, byte[][] data)
    {
        var matrix = CodingMatrix.Create(rs.Options.Matrix, rs.DataShards, rs.ParityShards);
        if (rs.Options.CustomParityRows is { } rows) matrix = CodingMatrix.FromParityRows(rows, rs.DataShards, rs.ParityShards);
        int len = data[0].Length;
        var parity = new byte[rs.ParityShards][];
        for (int p = 0; p < rs.ParityShards; p++)
        {
            parity[p] = new byte[len];
            var row = matrix.RowSpan(rs.DataShards + p);
            for (int b = 0; b < len; b++)
            {
                byte acc = 0;
                for (int d = 0; d < rs.DataShards; d++) acc ^= Gf256Reference.Multiply(row[d], data[d][b]);
                parity[p][b] = acc;
            }
        }

        return parity;
    }

    public static byte[][] Clone(byte[][] shards) => shards.Select(s => s.ToArray()).ToArray();

    /// <summary>Copies the present shards and replaces the missing ones with zero-filled buffers of the same length.</summary>
    public static byte[][] Erase(byte[][] shards, bool[] present) =>
        shards.Select((s, i) => present[i] ? s.ToArray() : new byte[s.Length]).ToArray();

    public static void AssertShardsEqual(byte[][] expected, byte[][] actual, string context = "")
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].AsSpan().SequenceEqual(actual[i]), $"{context} shard {i} differs");
    }

    public static ReadOnlyMemory<byte>[] AsReadOnly(byte[][] shards, int from, int count) =>
        shards.Skip(from).Take(count).Select(s => (ReadOnlyMemory<byte>)s).ToArray();

    public static Memory<byte>[] AsMemory(byte[][] shards, int from, int count) =>
        shards.Skip(from).Take(count).Select(s => (Memory<byte>)s).ToArray();

    public static byte[] Stripe(byte[][] shards)
    {
        int len = shards[0].Length;
        var stripe = new byte[shards.Length * len];
        for (int i = 0; i < shards.Length; i++) shards[i].CopyTo(stripe, i * len);
        return stripe;
    }

    /// <summary>Every way to choose up to <paramref name="maxErasures"/> of <paramref name="total"/> indices, as present flags.</summary>
    public static IEnumerable<bool[]> AllErasurePatterns(int total, int maxErasures)
    {
        for (int k = 1; k <= maxErasures; k++)
            foreach (var combo in Combinations(total, k))
            {
                var present = Enumerable.Repeat(true, total).ToArray();
                foreach (int i in combo) present[i] = false;
                yield return present;
            }
    }

    public static IEnumerable<bool[]> RandomErasurePatterns(int total, int maxErasures, int count, int seed)
    {
        var rng = new Random(seed);
        for (int n = 0; n < count; n++)
        {
            var present = Enumerable.Repeat(true, total).ToArray();
            int erase = rng.Next(1, maxErasures + 1);
            foreach (int i in Enumerable.Range(0, total).OrderBy(_ => rng.Next()).Take(erase)) present[i] = false;
            yield return present;
        }
    }

    private static IEnumerable<int[]> Combinations(int n, int k)
    {
        var indices = Enumerable.Range(0, k).ToArray();
        while (true)
        {
            yield return indices.ToArray();
            int i = k - 1;
            while (i >= 0 && indices[i] == n - k + i) i--;
            if (i < 0) yield break;
            indices[i]++;
            for (int j = i + 1; j < k; j++) indices[j] = indices[j - 1] + 1;
        }
    }
}
