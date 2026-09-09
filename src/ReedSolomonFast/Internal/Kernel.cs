namespace ReedSolomonFast.Internal;

/// <summary>
/// Kernel selection. <see cref="Best"/> is the fastest tier this CPU supports; a coder may pin any
/// supported tier through <see cref="ReedSolomonOptions.Kernel"/>. The environment variable
/// <c>REEDSOLOMONFAST_KERNEL</c> overrides the default for a whole process, which is how the test
/// matrix and the benchmark jobs exercise every tier.
/// </summary>
internal static unsafe class Kernel
{
    public const string EnvironmentVariable = "REEDSOLOMONFAST_KERNEL";

    /// <summary>
    /// A destination may alias one of the sources only when the call has at most this many sources.
    /// The vector kernels take inputs in groups of at most this size and store each group's partial
    /// result before reading the next group, so an aliased input in a later group would be read after
    /// it was overwritten. Within one group every position is fully read before it is stored.
    /// </summary>
    public static int MaxInPlaceSources => VectorKernel<Ssse3Vec>.MaxInputGroup;

    private static KernelTier? s_default;
    private static Exception? s_defaultError;

    /// <summary>
    /// The default tier: the environment override when set, else the best supported. Resolved on
    /// first use rather than in a static initializer, so a bad override surfaces as the intended
    /// exception from the coder's constructor instead of a TypeInitializationException that poisons
    /// the type for the rest of the process.
    /// </summary>
    public static KernelTier Default
    {
        get
        {
            if (s_default is { } tier) return tier;
            if (s_defaultError is { } error) throw error;
            try
            {
                tier = ResolveDefault();
            }
            catch (Exception ex)
            {
                s_defaultError = ex;
                throw;
            }

            s_default = tier;
            return tier;
        }
    }

    public static KernelTier Best()
    {
        if (GfniAvx512Vec.IsSupported) return KernelTier.GfniAvx512;
        if (GfniAvx2Vec.IsSupported) return KernelTier.GfniAvx2;
        if (Avx512Vec.IsSupported) return KernelTier.Avx512;
        if (Avx2Vec.IsSupported) return KernelTier.Avx2;
        if (Ssse3Vec.IsSupported) return KernelTier.Ssse3;
        if (AdvSimdVec.IsSupported) return KernelTier.AdvSimd;
        return KernelTier.Scalar;
    }

    public static bool IsSupported(KernelTier tier) => tier switch
    {
        KernelTier.Scalar => true,
        KernelTier.AdvSimd => AdvSimdVec.IsSupported,
        KernelTier.Ssse3 => Ssse3Vec.IsSupported,
        KernelTier.Avx2 => Avx2Vec.IsSupported,
        KernelTier.Avx512 => Avx512Vec.IsSupported,
        KernelTier.GfniAvx2 => GfniAvx2Vec.IsSupported,
        KernelTier.GfniAvx512 => GfniAvx512Vec.IsSupported,
        _ => false,
    };

    /// <summary>Validates a requested tier, returning the tier a coder should use.</summary>
    public static KernelTier Resolve(KernelTier? requested)
    {
        if (requested is null) return Default;
        if (!Enum.IsDefined(requested.Value))
            throw new ArgumentOutOfRangeException(nameof(requested), requested, "Unknown kernel tier.");
        if (!IsSupported(requested.Value))
            throw new PlatformNotSupportedException($"Kernel tier {requested.Value} is not supported on this CPU.");
        return requested.Value;
    }

    /// <summary>
    /// dsts[d] = XOR over s of tables(d, s) * srcs[s] for <paramref name="len"/> bytes, on the given tier.
    /// <paramref name="tables"/> and <paramref name="gfni"/> point at the first shuffle and GFNI entries of a
    /// <c>dstCount x srcCount</c> region (see <see cref="MulTables"/>); a row range of a larger set works too.
    /// Every branch below is a static call to a JIT-specialized generic, so once <paramref name="tier"/>
    /// is known the body is the same code a per-tier class would have produced.
    /// </summary>
    public static void DotProduct(KernelTier tier, byte** srcs, int srcCount, byte** dsts, int dstCount, byte* tables, byte* gfni, nuint len, bool accumulate = false, bool stream = false)
    {
        if (len == 0 || dstCount == 0) return;

        switch (tier)
        {
            case KernelTier.GfniAvx512:
                VectorKernel<GfniAvx512Vec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            case KernelTier.GfniAvx2:
                VectorKernel<GfniAvx2Vec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            case KernelTier.Avx512:
                VectorKernel<Avx512Vec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            case KernelTier.Avx2:
                VectorKernel<Avx2Vec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            case KernelTier.Ssse3:
                VectorKernel<Ssse3Vec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            case KernelTier.AdvSimd:
                VectorKernel<AdvSimdVec>.DotProduct(srcs, srcCount, dsts, dstCount, tables, gfni, len, accumulate, stream);
                break;
            default:
                ScalarKernel.DotProduct(srcs, srcCount, dsts, dstCount, tables, 0, len, accumulate);
                break;
        }
    }

    private static KernelTier ResolveDefault()
    {
        string? name = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(name)) return Best();

        if (!Enum.TryParse(name, ignoreCase: true, out KernelTier tier) || !Enum.IsDefined(tier))
            throw new InvalidOperationException($"{EnvironmentVariable}='{name}' is not a kernel tier name.");
        if (!IsSupported(tier))
            throw new PlatformNotSupportedException($"{EnvironmentVariable}={tier} is not supported on this CPU.");
        return tier;
    }
}
