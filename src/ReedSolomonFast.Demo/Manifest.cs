using System.Text.Json;

namespace ReedSolomonFast.Demo;

/// <summary>What recovery needs besides the shard files; metadata for the demo, not a storage format.</summary>
/// <remarks>
/// Split does not record the original length, so the manifest keeps it for Join. The geometry
/// and matrix kind are here because a decoder must be built exactly like the encoder was.
/// </remarks>
internal sealed record Manifest(
    string FileName,
    int OriginalLength,
    int DataShards,
    int ParityShards,
    int ShardLength,
    MatrixKind Matrix)
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));

    public static Manifest Load(string path)
    {
        try
        {
            return JsonSerializer.Deserialize<Manifest>(File.ReadAllText(path))
                ?? throw new DemoException(ExitCode.Io, $"manifest \"{path}\" is empty.");
        }
        catch (JsonException e)
        {
            throw new DemoException(ExitCode.Io, $"manifest \"{path}\" is not valid: {e.Message}");
        }
    }
}
