using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

public class MulTablesTests
{
    [Fact]
    public void FillShuffle_NibbleTables_ReproduceProduct()
    {
        Span<byte> t = stackalloc byte[32];
        for (int c = 0; c < 256; c++)
        {
            MulTables.FillShuffle((byte)c, t);
            for (int x = 0; x < 256; x++)
            {
                byte expected = Gf256Reference.Multiply((byte)c, (byte)x);
                Assert.Equal(expected, (byte)(t[x & 15] ^ t[16 + (x >> 4)]));
            }
        }
    }

    [Fact]
    public void GfniMatrix_SoftwareModel_ReproducesProduct()
    {
        for (int c = 0; c < 256; c++)
        {
            ulong m = MulTables.GfniMatrix((byte)c);
            for (int x = 0; x < 256; x++)
                Assert.Equal(Gf256Reference.Multiply((byte)c, (byte)x), MulTables.ApplyGfniMatrix(m, (byte)x));
        }
    }

    [Fact]
    public unsafe void Layout_RowMajor_WithGfniRegionAfterShuffle()
    {
        byte[] coef = [3, 7, 11, 13, 17, 19];
        var t = new MulTables(coef, rows: 2, cols: 3);
        Assert.Equal(2 * 3 * 32, t.GfniOffset);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
            {
                byte expected = coef[r * 3 + c];
                Assert.Equal(expected, t.Coefficient(r, c));
                byte* shuffle = t.Pointer + (r * 3 + c) * MulTables.ShuffleEntrySize;
                Assert.Equal(expected, shuffle[1]);
                Assert.Equal(Gf256Reference.Multiply(expected, 0x20), shuffle[18]);
                ulong gfni = *(ulong*)(t.Pointer + t.GfniOffset + (r * 3 + c) * MulTables.GfniEntrySize);
                Assert.Equal(MulTables.GfniMatrix(expected), gfni);
            }
    }

    [Fact]
    public void WrongCoefficientCount_Throws() =>
        Assert.Throws<ArgumentException>(() => new MulTables(new byte[5], rows: 2, cols: 3));
}
