namespace ReedSolomonFast.Internal;

/// <summary>A set of up to 256 shard indices as four 64-bit words. Value-equal, so it can key a dictionary.</summary>
internal readonly record struct ShardMask(ulong W0, ulong W1, ulong W2, ulong W3)
{
    public ShardMask With(int index)
    {
        ulong bit = 1UL << (index & 63);
        return (index >> 6) switch
        {
            0 => this with { W0 = W0 | bit },
            1 => this with { W1 = W1 | bit },
            2 => this with { W2 = W2 | bit },
            _ => this with { W3 = W3 | bit },
        };
    }
}

/// <summary>
/// Everything reconstruction needs for one erasure pattern: which present shards feed the kernel,
/// which missing shards it produces, and the tables for that product.
/// </summary>
internal sealed class DecodeSet
{
    public required int[] Inputs { get; init; }
    public required int[] Outputs { get; init; }
    public required MulTables Tables { get; init; }
}

/// <summary>
/// Remembers decode sets by (inputs, outputs) pattern. Real workloads see a handful of patterns, so
/// a bounded dictionary that is simply cleared when full is enough; an LRU would cost more than it saves.
/// </summary>
internal sealed class InverseCache
{
    private readonly Dictionary<(ShardMask Inputs, ShardMask Outputs), DecodeSet> _entries = new();
    private readonly int _capacity;
    private int _misses;

    public InverseCache(int capacity)
    {
        _capacity = capacity;
    }

    /// <summary>Cache misses so far; tests use it to prove the second reconstruction of a pattern is free.</summary>
    public int Misses => Volatile.Read(ref _misses);

    public bool TryGet(ShardMask inputs, ShardMask outputs, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out DecodeSet? set)
    {
        lock (_entries)
        {
            return _entries.TryGetValue((inputs, outputs), out set);
        }
    }

    /// <summary>
    /// Stores a set built outside the lock (inverting a 200x200 matrix takes milliseconds). Two
    /// threads racing on one pattern build two identical sets; the first one stored wins and is returned.
    /// </summary>
    public DecodeSet Add(ShardMask inputs, ShardMask outputs, DecodeSet set)
    {
        Interlocked.Increment(ref _misses);
        lock (_entries)
        {
            if (_entries.TryGetValue((inputs, outputs), out DecodeSet? existing)) return existing;
            if (_entries.Count >= _capacity) _entries.Clear();
            _entries[(inputs, outputs)] = set;
        }

        return set;
    }
}
