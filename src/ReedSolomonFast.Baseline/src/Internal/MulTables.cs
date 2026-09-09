using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ReedSolomonFast.Internal;

/// <summary>
/// Precomputed per-coefficient tables for a set of output rows over a set of inputs, in the layout
/// every kernel reads directly.
/// </summary>
/// <remarks>
/// One pinned buffer holds two regions. The shuffle region has 32 bytes per (row, col): the products
/// of the coefficient with 0..15 (low nibble table) followed by the products with 0x00, 0x10, ..., 0xF0
/// (high nibble table). The GFNI region has 8 bytes per (row, col): the 8x8 GF(2) matrix that
/// <c>VGF2P8AFFINEQB</c> applies. Both regions are indexed <c>row * cols + col</c> so a kernel walking
/// the inputs of one output reads sequential memory.
///
/// The buffer is allocated on the pinned object heap so kernels can hold a raw pointer for the
/// lifetime of the coder without a <c>fixed</c> block per call.
/// </remarks>
internal sealed unsafe class MulTables
{
    public const int ShuffleEntrySize = 32;
    public const int GfniEntrySize = 8;

    private readonly byte[] _data;

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>Start of the pinned buffer. Valid for the lifetime of this instance.</summary>
    public byte* Pointer { get; }

    /// <summary>Offset of the GFNI region from <see cref="Pointer"/>.</summary>
    public int GfniOffset => Rows * Cols * ShuffleEntrySize;

    /// <summary>Builds tables for a row-major <c>rows x cols</c> coefficient matrix.</summary>
    public MulTables(ReadOnlySpan<byte> coefficients, int rows, int cols)
    {
        if (coefficients.Length != rows * cols)
            throw new ArgumentException("Coefficient count does not match rows * cols.", nameof(coefficients));

        Rows = rows;
        Cols = cols;
        _data = GC.AllocateArray<byte>(rows * cols * (ShuffleEntrySize + GfniEntrySize), pinned: true);
        Pointer = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_data));

        Span<byte> shuffle = _data.AsSpan(0, GfniOffset);
        Span<ulong> gfni = MemoryMarshal.Cast<byte, ulong>(_data.AsSpan(GfniOffset));
        for (int i = 0; i < rows * cols; i++)
        {
            FillShuffle(coefficients[i], shuffle.Slice(i * ShuffleEntrySize, ShuffleEntrySize));
            gfni[i] = GfniMatrix(coefficients[i]);
        }
    }

    /// <summary>The coefficient stored at (row, col); the low nibble table's entry for 1.</summary>
    public byte Coefficient(int row, int col) => _data[(row * Cols + col) * ShuffleEntrySize + 1];

    /// <summary>Writes low[i] = c * i and high[i] = c * (i &lt;&lt; 4) for i in 0..15.</summary>
    public static void FillShuffle(byte c, Span<byte> dst32)
    {
        for (int i = 0; i < 16; i++)
        {
            dst32[i] = Gf256.Multiply(c, (byte)i);
            dst32[16 + i] = Gf256.Multiply(c, (byte)(i << 4));
        }
    }

    /// <summary>
    /// The 8x8 bit matrix of "multiply by c" in the byte order <c>VGF2P8AFFINEQB</c> expects:
    /// output bit b is the parity of (matrix byte 7 - b) AND input.
    /// </summary>
    /// <remarks>
    /// Multiplication by a constant is GF(2)-linear in the input bits, so bit b of c * x is the XOR
    /// over set bits i of x of bit b of c * 2^i. Row b therefore has bit i set when c * 2^i has bit b
    /// set. The affine instruction itself is polynomial-agnostic; only this matrix knows about 0x11D.
    /// </remarks>
    public static ulong GfniMatrix(byte c)
    {
        ulong matrix = 0;
        for (int b = 0; b < 8; b++)
        {
            byte row = 0;
            for (int i = 0; i < 8; i++)
            {
                if (((Gf256.Multiply(c, (byte)(1 << i)) >> b) & 1) != 0) row |= (byte)(1 << i);
            }

            matrix |= (ulong)row << (8 * (7 - b));
        }

        return matrix;
    }

    /// <summary>Software model of the affine transform, used by tests to pin the bit layout.</summary>
    public static byte ApplyGfniMatrix(ulong matrix, byte x)
    {
        int result = 0;
        for (int b = 0; b < 8; b++)
        {
            byte row = (byte)(matrix >> (8 * (7 - b)));
            if ((System.Numerics.BitOperations.PopCount((uint)(row & x)) & 1) != 0) result |= 1 << b;
        }

        return (byte)result;
    }
}
