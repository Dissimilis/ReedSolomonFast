using System.Runtime.CompilerServices;

namespace ReedSolomonFast.Internal;

/// <summary>
/// The multiply-add engine shared by every SIMD tier. Three nested loops bound three resources:
/// the outer loop walks the byte range in <see cref="ChunkBytes"/> chunks so the streams touched by
/// one pass fit in L2; inside a chunk, inputs are taken in balanced groups of at most
/// <see cref="MaxInputGroup"/> so no more than sixteen memory streams are live (Zen 4's L1 and L2 are
/// 8-way and its prefetcher tracks about sixteen streams); and inside a group the loop is Intel
/// ISA-L's <c>gf_Nvect_dot_prod</c>: one
/// vector position at a time, every input loaded once, up to four output accumulators in registers.
/// Multipliers are re-broadcast from the L1-resident tables per input per position; the register
/// file cannot hold them for more than a handful of inputs.
/// </summary>
/// <remarks>
/// After the first group each output is loaded and folded into the accumulator, one extra vector load
/// per output per group. Before grouping, a 50+20 encode at 1 MiB shards ran at 1.4 GB/s against
/// 10 GB/s at 64 KiB shards: seventy streams at the same page offset evicted each other from every
/// cache set. Groups are balanced because a short trailing group costs nearly a full pass: with a
/// fixed group of eight, 10+4 encode measured 1.48x slower than a single pass of ten.
/// One- and two-output bodies process two vectors per iteration; three- and four-output
/// bodies do not, because on 16-register targets the extra accumulators and multipliers would spill.
/// Bytes past the last whole vector go through <see cref="ScalarKernel"/>.
///
/// In-place use: a destination may alias an input only when all inputs fit in one group, because
/// every position's loads precede its store within a group but a later group would read an input the
/// first group already overwrote. <see cref="Kernel.MaxInPlaceSources"/> states the limit for callers.
/// </remarks>
internal static unsafe class VectorKernel<T> where T : unmanaged, IGfVector<T>
{
    /// <summary>Largest input group; see the class remarks. Twelve inputs plus four outputs is sixteen streams.</summary>
    /// <remarks>Static readonly rather than const so the benchmark sweep can override it per process; the JIT still folds it in tier-1 code.</remarks>
    public static readonly int MaxInputGroup = KernelTuning.Read("REEDSOLOMONFAST_GROUP", 12, 1, 256);

    /// <summary>
    /// Bytes per outer chunk. Twelve streams of 64 KiB are 768 KiB, inside a 1 MiB L2, and the
    /// 64 KiB shard case is where the 512-bit GFNI tier measured 60 GB/s; at 1 MiB shards without
    /// chunking it measured 26 GB/s. Must be a multiple of twice the widest vector.
    /// </summary>
    public static readonly int ChunkBytes = KernelTuning.Read("REEDSOLOMONFAST_CHUNK", 64 * 1024, 128, 1 << 30) & ~127;

    private static int Width
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => sizeof(T);
    }

    /// <summary>
    /// dsts[d] = XOR over s of tables(d, s) * srcs[s] for len bytes. With <c>accumulate</c> every
    /// output starts from its current contents instead of being overwritten (dsts[d] ^= ...), which
    /// is how incremental parity (EncodeShard, EncodeShards, Update) works; the outputs are then
    /// never inputs, so no aliasing rule applies.
    /// </summary>
    public static void DotProduct(byte** srcs, int srcCount, byte** dsts, int dstCount, byte* tables, byte* gfni, nuint len, bool accumulate = false)
    {
        nuint vecLen = len & ~(nuint)(Width - 1);

        nuint chunkBytes = (nuint)ChunkBytes;
        for (nuint start = 0; start < vecLen; start += chunkBytes)
        {
            nuint end = Math.Min(start + chunkBytes, vecLen);

            int groups = (srcCount + MaxInputGroup - 1) / MaxInputGroup;
            int groupSize = (srcCount + groups - 1) / groups;
            for (int g = 0; g < srcCount; g += groupSize)
            {
                int count = Math.Min(groupSize, srcCount - g);
                bool accumulateGroup = accumulate || g > 0;
                byte** groupSrcs = srcs + g;
                byte* groupTables = tables + (nuint)g * MulTables.ShuffleEntrySize;
                byte* groupGfni = gfni + (nuint)g * MulTables.GfniEntrySize;

                int d = 0;
                for (; d + 4 <= dstCount; d += 4)
                {
                    if (T.Unroll4 == 2)
                        Block4x2(groupSrcs, count, srcCount, accumulateGroup, dsts + d, Row(groupTables, d, srcCount), RowGfni(groupGfni, d, srcCount), start, end);
                    else
                        Block4(groupSrcs, count, srcCount, accumulateGroup, dsts + d, Row(groupTables, d, srcCount), RowGfni(groupGfni, d, srcCount), start, end);
                }

                switch (dstCount - d)
                {
                    case 1:
                        Block1(groupSrcs, count, srcCount, accumulateGroup, dsts + d, Row(groupTables, d, srcCount), RowGfni(groupGfni, d, srcCount), start, end);
                        break;
                    case 2:
                        Block2(groupSrcs, count, srcCount, accumulateGroup, dsts + d, Row(groupTables, d, srcCount), RowGfni(groupGfni, d, srcCount), start, end);
                        break;
                    case 3:
                        Block3(groupSrcs, count, srcCount, accumulateGroup, dsts + d, Row(groupTables, d, srcCount), RowGfni(groupGfni, d, srcCount), start, end);
                        break;
                }
            }
        }

        if (vecLen < len)
        {
            ScalarKernel.DotProduct(srcs, srcCount, dsts, dstCount, tables, vecLen, len - vecLen, accumulate);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* Row(byte* tables, int row, int stride) =>
        tables + (nuint)(row * stride) * MulTables.ShuffleEntrySize;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* RowGfni(byte* gfni, int row, int stride) =>
        gfni + (nuint)(row * stride) * MulTables.GfniEntrySize;

    /// <summary>One output, two vectors per iteration. <paramref name="count"/> inputs of a group whose table row stride is <paramref name="stride"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block1(byte** srcs, int count, int stride, bool accumulate, byte** dsts, byte* t0, byte* g0, nuint start, nuint end)
    {
        int w = Width;
        byte* dst0 = dsts[0];
        nuint i = start;

        for (; i + (nuint)(2 * w) <= end; i += (nuint)(2 * w))
        {
            T lo = T.LoadLo(t0, g0);
            T hi = T.LoadHi(t0);
            T a = T.Multiply(T.Load(srcs[0] + i), lo, hi);
            T b = T.Multiply(T.Load(srcs[0] + i + w), lo, hi);
            if (accumulate)
            {
                a = T.Xor(a, T.Load(dst0 + i));
                b = T.Xor(b, T.Load(dst0 + i + w));
            }

            for (int s = 1; s < count; s++)
            {
                lo = T.LoadLo(t0 + s * MulTables.ShuffleEntrySize, g0 + s * MulTables.GfniEntrySize);
                hi = T.LoadHi(t0 + s * MulTables.ShuffleEntrySize);
                byte* src = srcs[s] + i;
                a = T.Xor(a, T.Multiply(T.Load(src), lo, hi));
                b = T.Xor(b, T.Multiply(T.Load(src + w), lo, hi));
            }

            T.Store(dst0 + i, a);
            T.Store(dst0 + i + w, b);
        }

        if (i < end)
        {
            T a = T.Multiply(T.Load(srcs[0] + i), T.LoadLo(t0, g0), T.LoadHi(t0));
            if (accumulate) a = T.Xor(a, T.Load(dst0 + i));
            for (int s = 1; s < count; s++)
            {
                T lo = T.LoadLo(t0 + s * MulTables.ShuffleEntrySize, g0 + s * MulTables.GfniEntrySize);
                T hi = T.LoadHi(t0 + s * MulTables.ShuffleEntrySize);
                a = T.Xor(a, T.Multiply(T.Load(srcs[s] + i), lo, hi));
            }

            T.Store(dst0 + i, a);
        }
    }

    /// <summary>Two outputs, two vectors per iteration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block2(byte** srcs, int count, int stride, bool accumulate, byte** dsts, byte* t0, byte* g0, nuint start, nuint end)
    {
        int w = Width;
        byte* dst0 = dsts[0];
        byte* dst1 = dsts[1];
        byte* t1 = t0 + stride * MulTables.ShuffleEntrySize;
        byte* g1 = g0 + stride * MulTables.GfniEntrySize;
        nuint i = start;

        for (; i + (nuint)(2 * w) <= end; i += (nuint)(2 * w))
        {
            T va = T.Load(srcs[0] + i);
            T vb = T.Load(srcs[0] + i + w);
            T lo0 = T.LoadLo(t0, g0);
            T hi0 = T.LoadHi(t0);
            T lo1 = T.LoadLo(t1, g1);
            T hi1 = T.LoadHi(t1);
            T a0 = T.Multiply(va, lo0, hi0);
            T b0 = T.Multiply(vb, lo0, hi0);
            T a1 = T.Multiply(va, lo1, hi1);
            T b1 = T.Multiply(vb, lo1, hi1);
            if (accumulate)
            {
                a0 = T.Xor(a0, T.Load(dst0 + i));
                b0 = T.Xor(b0, T.Load(dst0 + i + w));
                a1 = T.Xor(a1, T.Load(dst1 + i));
                b1 = T.Xor(b1, T.Load(dst1 + i + w));
            }

            for (int s = 1; s < count; s++)
            {
                byte* src = srcs[s] + i;
                va = T.Load(src);
                vb = T.Load(src + w);
                int so = s * MulTables.ShuffleEntrySize;
                int go = s * MulTables.GfniEntrySize;
                lo0 = T.LoadLo(t0 + so, g0 + go);
                hi0 = T.LoadHi(t0 + so);
                lo1 = T.LoadLo(t1 + so, g1 + go);
                hi1 = T.LoadHi(t1 + so);
                a0 = T.Xor(a0, T.Multiply(va, lo0, hi0));
                b0 = T.Xor(b0, T.Multiply(vb, lo0, hi0));
                a1 = T.Xor(a1, T.Multiply(va, lo1, hi1));
                b1 = T.Xor(b1, T.Multiply(vb, lo1, hi1));
            }

            T.Store(dst0 + i, a0);
            T.Store(dst0 + i + w, b0);
            T.Store(dst1 + i, a1);
            T.Store(dst1 + i + w, b1);
        }

        if (i < end)
        {
            T v = T.Load(srcs[0] + i);
            T a0 = T.Multiply(v, T.LoadLo(t0, g0), T.LoadHi(t0));
            T a1 = T.Multiply(v, T.LoadLo(t1, g1), T.LoadHi(t1));
            if (accumulate)
            {
                a0 = T.Xor(a0, T.Load(dst0 + i));
                a1 = T.Xor(a1, T.Load(dst1 + i));
            }

            for (int s = 1; s < count; s++)
            {
                v = T.Load(srcs[s] + i);
                int so = s * MulTables.ShuffleEntrySize;
                int go = s * MulTables.GfniEntrySize;
                a0 = T.Xor(a0, T.Multiply(v, T.LoadLo(t0 + so, g0 + go), T.LoadHi(t0 + so)));
                a1 = T.Xor(a1, T.Multiply(v, T.LoadLo(t1 + so, g1 + go), T.LoadHi(t1 + so)));
            }

            T.Store(dst0 + i, a0);
            T.Store(dst1 + i, a1);
        }
    }

    /// <summary>Three outputs, one vector per iteration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block3(byte** srcs, int count, int stride, bool accumulate, byte** dsts, byte* t0, byte* g0, nuint start, nuint end)
    {
        int w = Width;
        byte* dst0 = dsts[0];
        byte* dst1 = dsts[1];
        byte* dst2 = dsts[2];
        byte* t1 = t0 + stride * MulTables.ShuffleEntrySize;
        byte* t2 = t1 + stride * MulTables.ShuffleEntrySize;
        byte* g1 = g0 + stride * MulTables.GfniEntrySize;
        byte* g2 = g1 + stride * MulTables.GfniEntrySize;

        for (nuint i = start; i < end; i += (nuint)w)
        {
            T v = T.Load(srcs[0] + i);
            T a0 = T.Multiply(v, T.LoadLo(t0, g0), T.LoadHi(t0));
            T a1 = T.Multiply(v, T.LoadLo(t1, g1), T.LoadHi(t1));
            T a2 = T.Multiply(v, T.LoadLo(t2, g2), T.LoadHi(t2));
            if (accumulate)
            {
                a0 = T.Xor(a0, T.Load(dst0 + i));
                a1 = T.Xor(a1, T.Load(dst1 + i));
                a2 = T.Xor(a2, T.Load(dst2 + i));
            }

            for (int s = 1; s < count; s++)
            {
                v = T.Load(srcs[s] + i);
                int so = s * MulTables.ShuffleEntrySize;
                int go = s * MulTables.GfniEntrySize;
                a0 = T.Xor(a0, T.Multiply(v, T.LoadLo(t0 + so, g0 + go), T.LoadHi(t0 + so)));
                a1 = T.Xor(a1, T.Multiply(v, T.LoadLo(t1 + so, g1 + go), T.LoadHi(t1 + so)));
                a2 = T.Xor(a2, T.Multiply(v, T.LoadLo(t2 + so, g2 + go), T.LoadHi(t2 + so)));
            }

            T.Store(dst0 + i, a0);
            T.Store(dst1 + i, a1);
            T.Store(dst2 + i, a2);
        }
    }

    /// <summary>
    /// Four outputs, two vectors per iteration, for NEON: eight accumulators, two data vectors and
    /// eight table vectors stay in registers, and every table load serves two vectors. Falls back to <see cref="Block4"/> for the odd trailing vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block4x2(byte** srcs, int count, int stride, bool accumulate, byte** dsts, byte* t0, byte* g0, nuint start, nuint end)
    {
        int w = Width;
        byte* dst0 = dsts[0];
        byte* dst1 = dsts[1];
        byte* dst2 = dsts[2];
        byte* dst3 = dsts[3];
        byte* t1 = t0 + stride * MulTables.ShuffleEntrySize;
        byte* t2 = t1 + stride * MulTables.ShuffleEntrySize;
        byte* t3 = t2 + stride * MulTables.ShuffleEntrySize;
        byte* g1 = g0 + stride * MulTables.GfniEntrySize;
        byte* g2 = g1 + stride * MulTables.GfniEntrySize;
        byte* g3 = g2 + stride * MulTables.GfniEntrySize;
        nuint i = start;

        for (; i + (nuint)(2 * w) <= end; i += (nuint)(2 * w))
        {
            T va = T.Load(srcs[0] + i);
            T vb = T.Load(srcs[0] + i + w);
            T lo0 = T.LoadLo(t0, g0);
            T hi0 = T.LoadHi(t0);
            T lo1 = T.LoadLo(t1, g1);
            T hi1 = T.LoadHi(t1);
            T lo2 = T.LoadLo(t2, g2);
            T hi2 = T.LoadHi(t2);
            T lo3 = T.LoadLo(t3, g3);
            T hi3 = T.LoadHi(t3);
            T a0 = T.Multiply(va, lo0, hi0);
            T b0 = T.Multiply(vb, lo0, hi0);
            T a1 = T.Multiply(va, lo1, hi1);
            T b1 = T.Multiply(vb, lo1, hi1);
            T a2 = T.Multiply(va, lo2, hi2);
            T b2 = T.Multiply(vb, lo2, hi2);
            T a3 = T.Multiply(va, lo3, hi3);
            T b3 = T.Multiply(vb, lo3, hi3);
            if (accumulate)
            {
                a0 = T.Xor(a0, T.Load(dst0 + i));
                b0 = T.Xor(b0, T.Load(dst0 + i + w));
                a1 = T.Xor(a1, T.Load(dst1 + i));
                b1 = T.Xor(b1, T.Load(dst1 + i + w));
                a2 = T.Xor(a2, T.Load(dst2 + i));
                b2 = T.Xor(b2, T.Load(dst2 + i + w));
                a3 = T.Xor(a3, T.Load(dst3 + i));
                b3 = T.Xor(b3, T.Load(dst3 + i + w));
            }

            for (int s = 1; s < count; s++)
            {
                byte* src = srcs[s] + i;
                va = T.Load(src);
                vb = T.Load(src + w);
                int so = s * MulTables.ShuffleEntrySize;
                int go = s * MulTables.GfniEntrySize;
                lo0 = T.LoadLo(t0 + so, g0 + go);
                hi0 = T.LoadHi(t0 + so);
                lo1 = T.LoadLo(t1 + so, g1 + go);
                hi1 = T.LoadHi(t1 + so);
                lo2 = T.LoadLo(t2 + so, g2 + go);
                hi2 = T.LoadHi(t2 + so);
                lo3 = T.LoadLo(t3 + so, g3 + go);
                hi3 = T.LoadHi(t3 + so);
                a0 = T.Xor(a0, T.Multiply(va, lo0, hi0));
                b0 = T.Xor(b0, T.Multiply(vb, lo0, hi0));
                a1 = T.Xor(a1, T.Multiply(va, lo1, hi1));
                b1 = T.Xor(b1, T.Multiply(vb, lo1, hi1));
                a2 = T.Xor(a2, T.Multiply(va, lo2, hi2));
                b2 = T.Xor(b2, T.Multiply(vb, lo2, hi2));
                a3 = T.Xor(a3, T.Multiply(va, lo3, hi3));
                b3 = T.Xor(b3, T.Multiply(vb, lo3, hi3));
            }

            T.Store(dst0 + i, a0);
            T.Store(dst0 + i + w, b0);
            T.Store(dst1 + i, a1);
            T.Store(dst1 + i + w, b1);
            T.Store(dst2 + i, a2);
            T.Store(dst2 + i + w, b2);
            T.Store(dst3 + i, a3);
            T.Store(dst3 + i + w, b3);
        }

        if (i < end) Block4(srcs, count, stride, accumulate, dsts, t0, g0, i, end);
    }

    /// <summary>Four outputs, one vector per iteration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block4(byte** srcs, int count, int stride, bool accumulate, byte** dsts, byte* t0, byte* g0, nuint start, nuint end)
    {
        int w = Width;
        byte* dst0 = dsts[0];
        byte* dst1 = dsts[1];
        byte* dst2 = dsts[2];
        byte* dst3 = dsts[3];
        byte* t1 = t0 + stride * MulTables.ShuffleEntrySize;
        byte* t2 = t1 + stride * MulTables.ShuffleEntrySize;
        byte* t3 = t2 + stride * MulTables.ShuffleEntrySize;
        byte* g1 = g0 + stride * MulTables.GfniEntrySize;
        byte* g2 = g1 + stride * MulTables.GfniEntrySize;
        byte* g3 = g2 + stride * MulTables.GfniEntrySize;

        for (nuint i = start; i < end; i += (nuint)w)
        {
            T v = T.Load(srcs[0] + i);
            T a0 = T.Multiply(v, T.LoadLo(t0, g0), T.LoadHi(t0));
            T a1 = T.Multiply(v, T.LoadLo(t1, g1), T.LoadHi(t1));
            T a2 = T.Multiply(v, T.LoadLo(t2, g2), T.LoadHi(t2));
            T a3 = T.Multiply(v, T.LoadLo(t3, g3), T.LoadHi(t3));
            if (accumulate)
            {
                a0 = T.Xor(a0, T.Load(dst0 + i));
                a1 = T.Xor(a1, T.Load(dst1 + i));
                a2 = T.Xor(a2, T.Load(dst2 + i));
                a3 = T.Xor(a3, T.Load(dst3 + i));
            }

            for (int s = 1; s < count; s++)
            {
                v = T.Load(srcs[s] + i);
                int so = s * MulTables.ShuffleEntrySize;
                int go = s * MulTables.GfniEntrySize;
                a0 = T.Xor(a0, T.Multiply(v, T.LoadLo(t0 + so, g0 + go), T.LoadHi(t0 + so)));
                a1 = T.Xor(a1, T.Multiply(v, T.LoadLo(t1 + so, g1 + go), T.LoadHi(t1 + so)));
                a2 = T.Xor(a2, T.Multiply(v, T.LoadLo(t2 + so, g2 + go), T.LoadHi(t2 + so)));
                a3 = T.Xor(a3, T.Multiply(v, T.LoadLo(t3 + so, g3 + go), T.LoadHi(t3 + so)));
            }

            T.Store(dst0 + i, a0);
            T.Store(dst1 + i, a1);
            T.Store(dst2 + i, a2);
            T.Store(dst3 + i, a3);
        }
    }
}
