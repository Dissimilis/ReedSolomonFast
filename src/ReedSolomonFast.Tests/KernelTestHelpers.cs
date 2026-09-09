using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

internal static unsafe class KernelTestHelpers
{
    public static readonly KernelTier[] AllTiers = Enum.GetValues<KernelTier>();

    public static bool IsSupported(KernelTier tier) => Kernel.IsSupported(tier);

    public static byte[] RandomBytes(int n, Random rng)
    {
        var b = new byte[n];
        rng.NextBytes(b);
        return b;
    }

    /// <summary>Naive dot product over the reference field, from byte <paramref name="offset"/> for <paramref name="len"/> bytes.</summary>
    public static byte[][] ReferenceDotProduct(byte[][] srcs, byte[] coef, int dstCount, int len, int offset)
    {
        int srcCount = srcs.Length;
        var result = new byte[dstCount][];
        for (int d = 0; d < dstCount; d++)
        {
            result[d] = new byte[len];
            for (int i = 0; i < len; i++)
            {
                byte acc = 0;
                for (int s = 0; s < srcCount; s++)
                    acc ^= Gf256Reference.Multiply(coef[d * srcCount + s], srcs[s][offset + i]);
                result[d][i] = acc;
            }
        }

        return result;
    }

    /// <summary>Runs one tier's DotProduct on the given buffers, starting at <paramref name="offset"/>.</summary>
    public static void Invoke(KernelTier tier, byte[][] srcs, byte[][] dsts, byte[] coef, int len, int offset, bool accumulate = false)
    {
        var tables = new MulTables(coef, dsts.Length, srcs.Length);
        var handles = new System.Runtime.InteropServices.GCHandle[srcs.Length + dsts.Length];
        byte** srcPtrs = stackalloc byte*[srcs.Length];
        byte** dstPtrs = stackalloc byte*[dsts.Length];
        try
        {
            for (int s = 0; s < srcs.Length; s++)
            {
                handles[s] = System.Runtime.InteropServices.GCHandle.Alloc(srcs[s], System.Runtime.InteropServices.GCHandleType.Pinned);
                srcPtrs[s] = (byte*)handles[s].AddrOfPinnedObject() + offset;
            }

            for (int d = 0; d < dsts.Length; d++)
            {
                handles[srcs.Length + d] = System.Runtime.InteropServices.GCHandle.Alloc(dsts[d], System.Runtime.InteropServices.GCHandleType.Pinned);
                dstPtrs[d] = (byte*)handles[srcs.Length + d].AddrOfPinnedObject() + offset;
            }

            Kernel.DotProduct(tier, srcPtrs, srcs.Length, dstPtrs, dsts.Length, tables.Pointer, tables.Pointer + tables.GfniOffset, (nuint)len, accumulate);
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        GC.KeepAlive(tables);
    }
}
