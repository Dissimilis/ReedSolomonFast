namespace ReedSolomonFast.Internal;

/// <summary>
/// Reads a kernel tuning constant from the environment, for the benchmark sweep that compares
/// values in one session. Absent or malformed values give the default; the library never sets these.
/// </summary>
internal static class KernelTuning
{
    // Initialize ARM's multi-group setting only when that path is used. The generic vector
    // kernel's existing initialization and the explicit override stay unchanged on x86.
    internal static class Arm
    {
        internal static readonly int MultiGroupChunkBytes = Read("REEDSOLOMONFAST_CHUNK", 16 * 1024, 128, 1 << 30) & ~127;
    }

    public static int Read(string name, int defaultValue, int min, int max)
    {
        string? text = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(text) || !int.TryParse(text, out int value)) return defaultValue;
        return Math.Clamp(value, min, max);
    }
}
