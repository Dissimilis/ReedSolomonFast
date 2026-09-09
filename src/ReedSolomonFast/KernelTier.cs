namespace ReedSolomonFast;

/// <summary>
/// The instruction set a coder uses for its GF(2^8) multiply-add kernels. Higher values are faster
/// where supported; <see cref="ReedSolomon.BestSupportedKernel"/> reports the best one on this machine.
/// </summary>
public enum KernelTier
{
    /// <summary>Portable C# using a 64 KiB multiplication table. Always available.</summary>
    Scalar = 0,

    /// <summary>ARM64 NEON: 16-byte table lookups (<c>TBL</c>) on the low and high nibbles.</summary>
    AdvSimd = 1,

    /// <summary>x86 SSSE3: 16-byte shuffles (<c>PSHUFB</c>) on the low and high nibbles.</summary>
    Ssse3 = 2,

    /// <summary>x86 AVX2: 32-byte shuffles (<c>VPSHUFB</c>) on the low and high nibbles.</summary>
    Avx2 = 3,

    /// <summary>x86 AVX-512BW: 64-byte shuffles on the low and high nibbles.</summary>
    Avx512 = 4,

    /// <summary>x86 GFNI with 256-bit vectors: one <c>VGF2P8AFFINEQB</c> per 32 bytes, no tables.</summary>
    GfniAvx2 = 5,

    /// <summary>x86 GFNI with 512-bit vectors: one <c>VGF2P8AFFINEQB</c> per 64 bytes, no tables.</summary>
    GfniAvx512 = 6,
}
