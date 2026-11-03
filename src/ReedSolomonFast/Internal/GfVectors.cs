using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace ReedSolomonFast.Internal;

/// <summary>
/// One vector width and instruction set, as seen by <see cref="VectorKernel{T}"/>. Each implementation
/// is a single-field struct wrapping a hardware vector; the JIT promotes it to a register, so the
/// generic kernel compiles to the same code a hand-written one would. Multipliers are passed as two
/// plain vectors (low and high nibble tables) rather than a two-field struct, which the JIT would
/// not enregister; GFNI tiers use only the first and their <see cref="LoadHi"/> is a no-op.
/// </summary>
internal unsafe interface IGfVector<TSelf> where TSelf : unmanaged, IGfVector<TSelf>
{
    static abstract bool IsSupported { get; }

    /// <summary>
    /// Vectors per iteration in the four-output body. 2 on NEON, where 32 registers hold eight
    /// accumulators, two data vectors and eight tables (ISA-L's NEON kernels unroll further still);
    /// 1 elsewhere: on the 512-bit x86 tiers a 2x body compiled cleanly and measured nothing.
    /// </summary>
    static abstract int Unroll4 { get; }

    /// <summary>Eight outputs per pass: eight accumulators, eight multipliers and one data vector. Requires a one-register multiplier and 32 vector registers.</summary>
    static abstract bool Block8 { get; }

    static abstract TSelf Load(byte* p);
    static abstract void Store(byte* p, TSelf v);

    /// <summary>Non-temporal store; <paramref name="p"/> must be aligned to the vector width. Plain store where the ISA has none.</summary>
    static abstract void StoreStream(byte* p, TSelf v);
    static abstract TSelf Xor(TSelf a, TSelf b);

    /// <summary>Loads the primary multiplier for one table entry, broadcast to every lane: the low nibble table, or the GFNI matrix.</summary>
    static abstract TSelf LoadLo(byte* shuffleEntry, byte* gfniEntry);

    /// <summary>Loads the high nibble table for one table entry; unused (and free) on GFNI tiers.</summary>
    static abstract TSelf LoadHi(byte* shuffleEntry);

    /// <summary>Multiplies every byte of <paramref name="v"/> by the multiplier's constant.</summary>
    static abstract TSelf Multiply(TSelf v, TSelf lo, TSelf hi);

    /// <summary>XORs a product into the accumulator; implementations may fuse the three XOR operands.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static virtual TSelf MultiplyAdd(TSelf acc, TSelf v, TSelf lo, TSelf hi) => TSelf.Xor(acc, TSelf.Multiply(v, lo, hi));
}

/// <summary>SSSE3: 16-byte <c>PSHUFB</c> lookups on the low and high nibbles.</summary>
internal readonly unsafe struct Ssse3Vec : IGfVector<Ssse3Vec>
{
    public readonly Vector128<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Ssse3Vec(Vector128<byte> v) => V = v;

    public static bool IsSupported => Ssse3.IsSupported;
    public static int Unroll4 => 1;
    public static bool Block8 => false;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Ssse3Vec Load(byte* p) => new(Vector128.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, Ssse3Vec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, Ssse3Vec v) => Sse2.StoreAlignedNonTemporal(p, v.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Ssse3Vec Xor(Ssse3Vec a, Ssse3Vec b) => new(a.V ^ b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Ssse3Vec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Vector128.Load(shuffleEntry));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Ssse3Vec LoadHi(byte* shuffleEntry) => new(Vector128.Load(shuffleEntry + 16));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Ssse3Vec Multiply(Ssse3Vec v, Ssse3Vec lo, Ssse3Vec hi)
    {
        Vector128<byte> mask = Vector128.Create((byte)0x0F);
        Vector128<byte> low = v.V & mask;
        Vector128<byte> high = Sse2.ShiftRightLogical(v.V.AsUInt16(), 4).AsByte() & mask;
        return new(Ssse3.Shuffle(lo.V, low) ^ Ssse3.Shuffle(hi.V, high));
    }
}

/// <summary>AVX2: 32-byte <c>VPSHUFB</c> lookups with the 16-byte tables broadcast to both lanes.</summary>
internal readonly unsafe struct Avx2Vec : IGfVector<Avx2Vec>
{
    public readonly Vector256<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Avx2Vec(Vector256<byte> v) => V = v;

    public static bool IsSupported => Avx2.IsSupported;
    public static int Unroll4 => 1;
    public static bool Block8 => false;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx2Vec Load(byte* p) => new(Vector256.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, Avx2Vec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, Avx2Vec v) => Avx.StoreAlignedNonTemporal(p, v.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx2Vec Xor(Avx2Vec a, Avx2Vec b) => new(a.V ^ b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx2Vec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Avx2.BroadcastVector128ToVector256(shuffleEntry));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx2Vec LoadHi(byte* shuffleEntry) => new(Avx2.BroadcastVector128ToVector256(shuffleEntry + 16));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx2Vec Multiply(Avx2Vec v, Avx2Vec lo, Avx2Vec hi)
    {
        Vector256<byte> mask = Vector256.Create((byte)0x0F);
        Vector256<byte> low = v.V & mask;
        Vector256<byte> high = Avx2.ShiftRightLogical(v.V.AsUInt16(), 4).AsByte() & mask;
        return new(Avx2.Shuffle(lo.V, low) ^ Avx2.Shuffle(hi.V, high));
    }
}

/// <summary>AVX-512BW: 64-byte shuffles with the 16-byte tables broadcast to all four lanes.</summary>
internal readonly unsafe struct Avx512Vec : IGfVector<Avx512Vec>
{
    public readonly Vector512<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Avx512Vec(Vector512<byte> v) => V = v;

    public static bool IsSupported => Avx512BW.IsSupported;
    public static int Unroll4 => 1;
    public static bool Block8 => false;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx512Vec Load(byte* p) => new(Vector512.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, Avx512Vec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, Avx512Vec v) => Avx512F.StoreAlignedNonTemporal(p, v.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx512Vec Xor(Avx512Vec a, Avx512Vec b) => new(a.V ^ b.V);

    // Vector512.Create(Vector128) did not fold to a single broadcast; the dedicated intrinsic
    // (VBROADCASTI32X4) took this tier from below AVX2 to above it in the same probe.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx512Vec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Avx512F.BroadcastVector128ToVector512((uint*)shuffleEntry).AsByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx512Vec LoadHi(byte* shuffleEntry) => new(Avx512F.BroadcastVector128ToVector512((uint*)(shuffleEntry + 16)).AsByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Avx512Vec Multiply(Avx512Vec v, Avx512Vec lo, Avx512Vec hi)
    {
        Vector512<byte> mask = Vector512.Create((byte)0x0F);
        Vector512<byte> low = v.V & mask;
        Vector512<byte> high = Avx512BW.ShiftRightLogical(v.V.AsUInt16(), 4).AsByte() & mask;
        return new(Avx512BW.Shuffle(lo.V, low) ^ Avx512BW.Shuffle(hi.V, high));
    }
}

/// <summary>GFNI on 256-bit vectors: one affine transform per 32 bytes, no nibble tables.</summary>
internal readonly unsafe struct GfniAvx2Vec : IGfVector<GfniAvx2Vec>
{
    public readonly Vector256<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private GfniAvx2Vec(Vector256<byte> v) => V = v;

    public static bool IsSupported => Gfni.V256.IsSupported;
    public static int Unroll4 => 1;
    // AVX-512VL provides the extra registers needed to avoid spilling the eight-output body.
    public static bool Block8 => Environment.Is64BitProcess && Avx512BW.VL.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx2Vec Load(byte* p) => new(Vector256.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, GfniAvx2Vec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, GfniAvx2Vec v) => Avx.StoreAlignedNonTemporal(p, v.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx2Vec Xor(GfniAvx2Vec a, GfniAvx2Vec b) => new(a.V ^ b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx2Vec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Vector256.Create(*(ulong*)gfniEntry).AsByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx2Vec LoadHi(byte* shuffleEntry) => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx2Vec Multiply(GfniAvx2Vec v, GfniAvx2Vec lo, GfniAvx2Vec hi) =>
        new(Gfni.V256.GaloisFieldAffineTransform(v.V, lo.V, 0));
}

/// <summary>GFNI on 512-bit vectors: one affine transform per 64 bytes, no nibble tables.</summary>
internal readonly unsafe struct GfniAvx512Vec : IGfVector<GfniAvx512Vec>
{
    public readonly Vector512<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private GfniAvx512Vec(Vector512<byte> v) => V = v;

    public static bool IsSupported => Gfni.V512.IsSupported;
    public static int Unroll4 => 1;
    public static bool Block8 => true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx512Vec Load(byte* p) => new(Vector512.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, GfniAvx512Vec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, GfniAvx512Vec v) => Avx512F.StoreAlignedNonTemporal(p, v.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx512Vec Xor(GfniAvx512Vec a, GfniAvx512Vec b) => new(a.V ^ b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx512Vec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Vector512.Create(*(ulong*)gfniEntry).AsByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx512Vec LoadHi(byte* shuffleEntry) => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GfniAvx512Vec Multiply(GfniAvx512Vec v, GfniAvx512Vec lo, GfniAvx512Vec hi) =>
        new(Gfni.V512.GaloisFieldAffineTransform(v.V, lo.V, 0));
}

/// <summary>ARM64 NEON: 16-byte <c>TBL</c> lookups on the low and high nibbles.</summary>
internal readonly unsafe struct AdvSimdVec : IGfVector<AdvSimdVec>
{
    public readonly Vector128<byte> V;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private AdvSimdVec(Vector128<byte> v) => V = v;

    public static bool IsSupported => AdvSimd.Arm64.IsSupported;
    public static int Unroll4 => 2;
    public static bool Block8 => false;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec Load(byte* p) => new(Vector128.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(byte* p, AdvSimdVec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStream(byte* p, AdvSimdVec v) => v.V.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec Xor(AdvSimdVec a, AdvSimdVec b) => new(a.V ^ b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec LoadLo(byte* shuffleEntry, byte* gfniEntry) => new(Vector128.Load(shuffleEntry));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec LoadHi(byte* shuffleEntry) => new(Vector128.Load(shuffleEntry + 16));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec Multiply(AdvSimdVec v, AdvSimdVec lo, AdvSimdVec hi)
    {
        Vector128<byte> mask = Vector128.Create((byte)0x0F);
        Vector128<byte> low = v.V & mask;
        Vector128<byte> high = AdvSimd.ShiftRightLogical(v.V, 4);
        return new(AdvSimd.Arm64.VectorTableLookup(lo.V, low) ^ AdvSimd.Arm64.VectorTableLookup(hi.V, high));
    }

#pragma warning disable SYSLIB5003
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static AdvSimdVec MultiplyAdd(AdvSimdVec acc, AdvSimdVec v, AdvSimdVec lo, AdvSimdVec hi)
    {
        // The managed vector used by SVE2 must match NEON's 16-byte width.
        if (Sve2.IsSupported && System.Numerics.Vector<byte>.Count == 16)
        {
            var low = v.V & Vector128.Create((byte)15);
            var high = AdvSimd.ShiftRightLogical(v.V, 4);
            var a = AdvSimd.Arm64.VectorTableLookup(lo.V, low);
            var b = AdvSimd.Arm64.VectorTableLookup(hi.V, high);
            return new(Vector128.AsVector128(Sve2.Xor(Vector128.AsVector(acc.V), Vector128.AsVector(a), Vector128.AsVector(b))));
        }
        return Xor(acc, Multiply(v, lo, hi));
    }
#pragma warning restore SYSLIB5003
}
