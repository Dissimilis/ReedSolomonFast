using System.Text.Json;

namespace ReedSolomonFast.Demo;

/// <summary>What recovery needs besides the shard files; metadata for the demo, not a storage format.</summary>
/// <remarks>
/// Split does not record the original length, so the manifest keeps it for Join. The geometry
/// and matrix kind are here because a decoder must be built exactly like the encoder was. The
/// per-shard SHA-256 digests let the decoder tell a damaged shard from a good one, which parity
/// alone cannot; they are only as trustworthy as the file that holds them.
/// </remarks>
internal sealed record Manifest(
    string FileName,
    int OriginalLength,
    int DataShards,
    int ParityShards,
    int ShardLength,
    MatrixKind Matrix,
    string[] ShardDigests)
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));

    public static Manifest Load(string path)
    {
        Manifest manifest;
        try
        {
            manifest = JsonSerializer.Deserialize<Manifest>(File.ReadAllText(path))
                ?? throw new DemoException(ExitCode.Io, $"manifest \"{path}\" is empty.");
        }
        catch (JsonException e)
        {
            throw new DemoException(ExitCode.Io, $"manifest \"{path}\" is not valid: {e.Message}");
        }

        int total = manifest.DataShards + manifest.ParityShards;
        if (manifest.ShardDigests is null || manifest.ShardDigests.Length != total
            || manifest.ShardDigests.Any(d => d is null || d.Length != 64 || !d.All(Uri.IsHexDigit)))
            throw new DemoException(ExitCode.Io, $"manifest \"{path}\" must list one SHA-256 digest per shard, {total} in all.");
        // Compared as strings against lowercase hex, so a hand-edited uppercase digest must not condemn a good shard.
        return manifest with { ShardDigests = manifest.ShardDigests.Select(d => d.ToLowerInvariant()).ToArray() };
    }
}
