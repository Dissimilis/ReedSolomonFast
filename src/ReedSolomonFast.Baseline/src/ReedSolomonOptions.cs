namespace ReedSolomonFast;

/// <summary>
/// Settings for a <see cref="ReedSolomon"/> coder. The defaults are right for almost everyone;
/// every property exists so the remaining callers can take control.
/// </summary>
public sealed class ReedSolomonOptions
{
    /// <summary>Shared instance of the defaults.</summary>
    public static ReedSolomonOptions Default { get; } = new();

    /// <summary>
    /// Coding matrix construction. <see cref="MatrixKind.Vandermonde"/> (the default) produces parity
    /// byte-identical to Backblaze JavaReedSolomon, klauspost/reedsolomon and reed-solomon-erasure.
    /// </summary>
    public MatrixKind Matrix { get; init; } = MatrixKind.Vandermonde;

    /// <summary>
    /// Parity rows of a custom coding matrix, <c>ParityShards</c> rows of <c>DataShards</c> bytes each.
    /// Overrides <see cref="Matrix"/>. The caller is responsible for choosing rows whose every square
    /// submatrix with the identity rows is invertible; reconstruction throws if a chosen subset is not.
    /// </summary>
    public byte[][]? CustomParityRows { get; init; }

    /// <summary>
    /// Whether to remember the inverted matrix and kernel tables for each erasure pattern seen by
    /// reconstruction, so repeated recoveries of the same pattern allocate nothing. Default true.
    /// </summary>
    public bool InversionCache { get; init; } = true;

    /// <summary>Number of erasure patterns the cache holds before it is cleared. Default 256.</summary>
    public int InversionCacheSize { get; init; } = 256;

    /// <summary>
    /// Pins a kernel tier. Null (the default) selects the best supported tier, or the tier named by the
    /// <c>REEDSOLOMONFAST_KERNEL</c> environment variable. A tier the CPU lacks throws
    /// <see cref="PlatformNotSupportedException"/> from the coder's constructor.
    /// </summary>
    public KernelTier? Kernel { get; init; }

    /// <summary>
    /// Maximum threads one call may use. 1 (the default) never leaves the calling thread; -1 uses
    /// every processor. Above 1, calls whose shards are at least <see cref="ParallelThresholdBytes"/>
    /// long split their byte range across the thread pool; calls with eight or more outputs on
    /// shards of 256 KiB or less split by output instead, once the bytes they move per shard pass
    /// the same threshold. The result is identical at every setting.
    /// </summary>
    public int MaxDegreeOfParallelism { get; init; } = 1;

    /// <summary>
    /// Work below which a call stays on the calling thread regardless of <see cref="MaxDegreeOfParallelism"/>:
    /// the shard length for a byte-range split, the bytes moved per shard for an output split. Default 1 MiB.
    /// </summary>
    public int ParallelThresholdBytes { get; init; } = 1 << 20;
}
