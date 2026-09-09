using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

public class CodingMatrixTests
{
    [Theory]
    [InlineData(MatrixKind.Cauchy, 5, 3)]
    [InlineData(MatrixKind.Cauchy, 200, 56)]
    [InlineData(MatrixKind.Vandermonde, 5, 3)]
    [InlineData(MatrixKind.Vandermonde, 200, 56)]
    public void TopSquare_IsIdentity(MatrixKind kind, int data, int parity)
    {
        var m = CodingMatrix.Create(kind, data, parity);
        Assert.Equal(data + parity, m.Rows);
        Assert.Equal(data, m.Cols);
        Assert.True(m.SubMatrix(Enumerable.Range(0, data).ToArray()).IsIdentity());
    }

    [Fact]
    public void Cauchy_ParityEntries_AreInverses()
    {
        var m = CodingMatrix.Create(MatrixKind.Cauchy, 7, 4);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 7; c++)
                Assert.Equal(Gf256Reference.Inverse((byte)((7 + r) ^ c)), m[7 + r, c]);
    }

    [Fact]
    public void Vandermonde_3x2_MatchesHandComputation()
    {
        // Vandermonde rows for 3 data shards are [1,0,0],[1,1,1],[1,2,4],[1,3,5],[1,4,16].
        // Row 3 times the inverse of the top square solves x * top = [1,3,5], which is [1,1,1].
        var m = CodingMatrix.Create(MatrixKind.Vandermonde, 3, 2);
        Assert.Equal(new byte[] { 1, 1, 1 }, m.Row(3));
    }

    [Theory]
    [InlineData(2, 1)]
    [InlineData(4, 2)]
    [InlineData(10, 4)]
    [InlineData(17, 3)]
    [InlineData(100, 50)]
    public void Vandermonde_ParityMatchesReedSolomonNet(int data, int parity)
    {
        const int len = 37;
        var rng = new Random(data * 31 + parity);
        var shards = new byte[data + parity][];
        for (int i = 0; i < shards.Length; i++)
        {
            shards[i] = new byte[len];
            if (i < data) rng.NextBytes(shards[i]);
        }
        global::ReedSolomon.NET.ReedSolomon.Create(data, parity).EncodeParity(shards, 0, len);

        var m = CodingMatrix.Create(MatrixKind.Vandermonde, data, parity);
        for (int p = 0; p < parity; p++)
        {
            var row = m.RowSpan(data + p);
            for (int b = 0; b < len; b++)
            {
                byte acc = 0;
                for (int d = 0; d < data; d++) acc ^= Gf256Reference.Multiply(row[d], shards[d][b]);
                Assert.Equal(shards[data + p][b], acc);
            }
        }
    }

    [Theory]
    [InlineData(MatrixKind.Cauchy, 10, 4)]
    [InlineData(MatrixKind.Vandermonde, 10, 4)]
    [InlineData(MatrixKind.Cauchy, 100, 100)]
    [InlineData(MatrixKind.Vandermonde, 128, 128)]
    public void RandomSquareSubmatrix_IsInvertible(MatrixKind kind, int data, int parity)
    {
        var m = CodingMatrix.Create(kind, data, parity);
        var rng = new Random(1234);
        for (int trial = 0; trial < 50; trial++)
        {
            int[] rows = Enumerable.Range(0, data + parity).OrderBy(_ => rng.Next()).Take(data).ToArray();
            var sub = m.SubMatrix(rows);
            Assert.True(sub.Multiply(sub.Invert()).IsIdentity());
        }
    }

    [Fact]
    public void Invert_MatchesBackblazePublishedInverse()
    {
        // From Backblaze's JavaReedSolomon MatrixTest: a 3x3 over GF(2^8) and its inverse. An
        // inversion that is merely self-consistent (M * M^-1 = I) would pass the other tests
        // with a wrong field; this pins the actual numbers.
        byte[,] input = { { 56, 23, 98 }, { 3, 100, 200 }, { 45, 201, 123 } };
        byte[,] expected = { { 175, 133, 33 }, { 130, 13, 245 }, { 112, 35, 126 } };
        var m = new CodingMatrix(3, 3);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) m[r, c] = input[r, c];
        var inv = m.Invert();
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Assert.True(expected[r, c] == inv[r, c], $"[{r},{c}]: {inv[r, c]}");
    }

    [Fact]
    public void Invert_Singular_Throws()
    {
        var m = new CodingMatrix(2, 2);
        m[0, 0] = 1; m[0, 1] = 2; m[1, 0] = 1; m[1, 1] = 2;
        Assert.Throws<InvalidOperationException>(() => m.Invert());
    }

    [Fact]
    public void Multiply_MatchesReference()
    {
        var rng = new Random(7);
        var a = new CodingMatrix(4, 6);
        var b = new CodingMatrix(6, 3);
        for (int r = 0; r < 4; r++) for (int c = 0; c < 6; c++) a[r, c] = (byte)rng.Next(256);
        for (int r = 0; r < 6; r++) for (int c = 0; c < 3; c++) b[r, c] = (byte)rng.Next(256);
        var p = a.Multiply(b);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 3; c++)
            {
                byte acc = 0;
                for (int k = 0; k < 6; k++) acc ^= Gf256Reference.Multiply(a[r, k], b[k, c]);
                Assert.Equal(acc, p[r, c]);
            }
    }
}
