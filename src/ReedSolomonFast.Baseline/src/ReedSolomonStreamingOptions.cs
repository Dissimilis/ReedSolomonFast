namespace ReedSolomonFast;

/// <summary>Settings for <see cref="ReedSolomonStreams"/>. The defaults suit files on local disks.</summary>
public sealed class ReedSolomonStreamingOptions
{
    /// <summary>Shared instance of the defaults.</summary>
    public static ReedSolomonStreamingOptions Default { get; } = new();

    /// <summary>
    /// Bytes read from each shard stream per window, and so the shard length the coder sees per
    /// call. Default 64 KiB, the chunk size at which the kernel measured fastest. The stripe a call
    /// allocates is <c>TotalShards</c> times the window (this value, or the shard length if that is
    /// shorter) rounded up to 64, plus 256 bytes of padding per shard; <c>VerifyAsync</c> adds
    /// <c>ParityShards</c> times the window of scratch.
    /// </summary>
    public int WindowSizeBytes { get; init; } = 64 * 1024;

    /// <summary>
    /// Maximum stream reads or writes in flight at once within a window. Default 4. Each stream has
    /// at most one operation in flight regardless of this value.
    /// </summary>
    public int MaxConcurrentIoOperations { get; init; } = 4;
}
