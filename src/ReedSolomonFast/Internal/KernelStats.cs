namespace ReedSolomonFast.Internal;

/// <summary>Counters for tests and probes; never read by the library.</summary>
internal static class KernelStats
{
    /// <summary>Kernel calls that took the non-temporal store path.</summary>
    public static long StreamCalls;
}
