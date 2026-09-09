using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Benchmarks;

/// <summary>
/// Correctness gate that runs before every benchmark session.
///
/// A kernel that multiplies by the wrong constant, skips a tail, or mixes up two output rows still
/// produces parity-shaped bytes at full speed, and a faster wrong answer looks exactly like a win.
/// This aborts the run rather than reporting numbers for a broken coder.
///
/// Two anchors: the shift-and-add reference field, which cannot drift, checked against every kernel
/// tier the CPU supports over shapes that straddle every vector width; and full round trips through
/// the public API (encode, verify, erase, reconstruct) for every tier and several geometries.
/// </summary>
internal static unsafe class Correctness
{
    // Every vector-width boundary plus one chunk boundary; the xUnit suite covers more shapes.
    private static readonly int[] Lengths = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097, 65_537];

    public static void Run()
    {
        var supported = Enum.GetValues<KernelTier>().Where(Kernel.IsSupported).ToArray();
        Console.WriteLine($"Correctness gate: kernels {string.Join(", ", supported)}");

        int checks = 0;
        checks += CheckField();
        foreach (var tier in supported)
        {
            checks += CheckKernel(tier);
            checks += CheckRoundTrips(tier);
        }

        Console.WriteLine($"Correctness gate passed: {checks:N0} checks.");
    }

    private static byte ReferenceMultiply(byte a, byte b)
    {
        int result = 0, x = a, y = b;
        while (y != 0)
        {
            if ((y & 1) != 0) result ^= x;
            x <<= 1;
            if ((x & 0x100) != 0) x ^= 0x11D;
            y >>= 1;
        }

        return (byte)result;
    }

    private static int CheckField()
    {
        int checks = 0;
        for (int a = 0; a < 256; a++)
            for (int b = 0; b < 256; b++, checks++)
                if (Gf256.Multiply((byte)a, (byte)b) != ReferenceMultiply((byte)a, (byte)b))
                    throw new InvalidOperationException($"CORRECTNESS FAILURE: Gf256.Multiply({a}, {b})");
        return checks;
    }

    private static int CheckKernel(KernelTier tier)
    {
        int checks = 0;
        var rng = new Random(42);
        foreach (int srcCount in new[] { 1, 3, 5, 16 })
            foreach (int dstCount in new[] { 1, 2, 3, 4, 5 })
                foreach (int len in Lengths)
                    foreach (int offset in new[] { 0, 3 })
                    {
                        var coef = new byte[dstCount * srcCount];
                        rng.NextBytes(coef);
                        var srcs = new byte[srcCount][];
                        var dsts = new byte[dstCount][];
                        for (int s = 0; s < srcCount; s++) { srcs[s] = new byte[len + offset]; rng.NextBytes(srcs[s]); }
                        for (int d = 0; d < dstCount; d++) dsts[d] = new byte[len + offset];

                        RunKernel(tier, srcs, dsts, coef, len, offset);

                        for (int d = 0; d < dstCount; d++)
                            for (int i = 0; i < len; i++, checks++)
                            {
                                byte expected = 0;
                                for (int s = 0; s < srcCount; s++) expected ^= ReferenceMultiply(coef[d * srcCount + s], srcs[s][offset + i]);
                                if (dsts[d][offset + i] != expected)
                                    throw new InvalidOperationException($"CORRECTNESS FAILURE: {tier} srcs={srcCount} dsts={dstCount} len={len} offset={offset} dst={d} byte={i}");
                            }

                    }

        return checks;
    }

    private static void RunKernel(KernelTier tier, byte[][] srcs, byte[][] dsts, byte[] coef, int len, int offset)
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

            Kernel.DotProduct(tier, srcPtrs, srcs.Length, dstPtrs, dsts.Length, tables.Pointer, tables.Pointer + tables.GfniOffset, (nuint)len);
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        GC.KeepAlive(tables);
    }

    private static int CheckRoundTrips(KernelTier tier)
    {
        int checks = 0;
        var rng = new Random(7);
        foreach (var (data, parity) in new[] { (1, 1), (3, 2), (10, 4), (8, 8), (50, 20) })
            foreach (var kind in new[] { MatrixKind.Vandermonde, MatrixKind.Cauchy })
            {
                var rs = new ReedSolomon(data, parity, new ReedSolomonOptions { Matrix = kind, Kernel = tier });
                foreach (int len in new[] { 63, 4097 })
                {
                    var original = new byte[data + parity][];
                    for (int i = 0; i < original.Length; i++) { original[i] = new byte[len]; rng.NextBytes(original[i]); }
                    rs.Encode(original);
                    if (!rs.Verify(original)) throw new InvalidOperationException($"CORRECTNESS FAILURE: Verify after Encode, {tier} {data}+{parity} len={len}");
                    checks++;

                    for (int trial = 0; trial < 4; trial++)
                    {
                        var present = Enumerable.Repeat(true, data + parity).ToArray();
                        foreach (int i in Enumerable.Range(0, data + parity).OrderBy(_ => rng.Next()).Take(rng.Next(1, parity + 1))) present[i] = false;
                        var shards = original.Select((s, i) => present[i] ? s.ToArray() : new byte[len]).ToArray();
                        rs.Reconstruct(shards, present);
                        for (int i = 0; i < shards.Length; i++, checks++)
                            if (!shards[i].AsSpan().SequenceEqual(original[i]))
                                throw new InvalidOperationException($"CORRECTNESS FAILURE: Reconstruct, {tier} {kind} {data}+{parity} len={len} shard={i}");
                    }
                }
            }

        return checks;
    }
}
