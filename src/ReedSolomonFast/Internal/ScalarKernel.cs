using System.Runtime.CompilerServices;

namespace ReedSolomonFast.Internal;

/// <summary>
/// Portable multiply-add kernel: one 64 KiB table lookup per byte per (output, input) pair.
/// The fallback everywhere, and the tail handler behind every vector kernel.
/// </summary>
internal static unsafe class ScalarKernel
{
    public static bool IsSupported => true;

    /// <summary>
    /// dsts[d] = XOR over s of tables(d, s) * srcs[s], for <paramref name="len"/> bytes. Outputs are
    /// overwritten, not accumulated. <paramref name="tables"/> is <see cref="MulTables.Pointer"/> for a
    /// table set of exactly <paramref name="dstCount"/> rows and <paramref name="srcCount"/> columns.
    /// Work starts <paramref name="offset"/> bytes into every buffer, which is how vector kernels hand over their tails.
    /// </summary>
    public static void DotProduct(byte** srcs, int srcCount, byte** dsts, int dstCount, byte* tables, nuint offset, nuint len, bool accumulate = false)
    {
        for (int d = 0; d < dstCount; d++)
        {
            byte* dst = dsts[d] + offset;
            byte* entry = tables + (nuint)(d * srcCount) * MulTables.ShuffleEntrySize;

            // First input assigns instead of XORs, so the output needs no zeroing pass; unless the
            // caller asked to accumulate into what is already there.
            if (accumulate) MulAdd(dst, srcs[0] + offset, Gf256.MulPointer + (entry[1] << 8), len);
            else MulSet(dst, srcs[0] + offset, Gf256.MulPointer + (entry[1] << 8), len);
            for (int s = 1; s < srcCount; s++)
            {
                entry += MulTables.ShuffleEntrySize;
                byte c = entry[1];
                if (c != 0) MulAdd(dst, srcs[s] + offset, Gf256.MulPointer + (c << 8), len);
            }
        }
    }

    /// <summary>dst[i] = row[src[i]] where <paramref name="row"/> is the 256-byte product row of a coefficient.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MulSet(byte* dst, byte* src, byte* row, nuint len)
    {
        nuint i = 0;
        for (; i + 8 <= len; i += 8)
        {
            dst[i + 0] = row[src[i + 0]];
            dst[i + 1] = row[src[i + 1]];
            dst[i + 2] = row[src[i + 2]];
            dst[i + 3] = row[src[i + 3]];
            dst[i + 4] = row[src[i + 4]];
            dst[i + 5] = row[src[i + 5]];
            dst[i + 6] = row[src[i + 6]];
            dst[i + 7] = row[src[i + 7]];
        }

        for (; i < len; i++) dst[i] = row[src[i]];
    }

    /// <summary>dst[i] ^= row[src[i]].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MulAdd(byte* dst, byte* src, byte* row, nuint len)
    {
        nuint i = 0;
        for (; i + 8 <= len; i += 8)
        {
            dst[i + 0] ^= row[src[i + 0]];
            dst[i + 1] ^= row[src[i + 1]];
            dst[i + 2] ^= row[src[i + 2]];
            dst[i + 3] ^= row[src[i + 3]];
            dst[i + 4] ^= row[src[i + 4]];
            dst[i + 5] ^= row[src[i + 5]];
            dst[i + 6] ^= row[src[i + 6]];
            dst[i + 7] ^= row[src[i + 7]];
        }

        for (; i < len; i++) dst[i] ^= row[src[i]];
    }
}
