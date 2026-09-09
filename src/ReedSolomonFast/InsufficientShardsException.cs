namespace ReedSolomonFast;

/// <summary>
/// Thrown when fewer shards are present than the coder needs: reconstruction requires at least
/// <c>DataShards</c> of any kind, and <c>Join</c> requires every data shard it reads.
/// </summary>
public sealed class InsufficientShardsException : InvalidOperationException
{
    /// <summary>Shards that were present.</summary>
    public int Present { get; }

    /// <summary>Shards that were needed.</summary>
    public int Required { get; }

    public InsufficientShardsException(int present, int required)
        : base($"Not enough shards present: {present} of the {required} required.")
    {
        Present = present;
        Required = required;
    }

    public InsufficientShardsException(string message, int present, int required)
        : base(message)
    {
        Present = present;
        Required = required;
    }
}
