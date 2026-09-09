using System.Buffers;
using System.Runtime.CompilerServices;
using ReedSolomonFast.Internal;

namespace ReedSolomonFast;

/// <summary>
/// A Reed-Solomon erasure coder over GF(2^8): <c>DataShards</c> shards of data produce
/// <c>ParityShards</c> shards of parity, and any <c>DataShards</c> of the total can rebuild the rest.
/// </summary>
/// <remarks>
/// Shards are laid out data first, then parity, in one index space of <see cref="TotalShards"/>
/// entries, and every shard in a call has the same length. <c>Encode</c> overwrites the parity
/// shards and never touches data; <c>Reconstruct</c> overwrites only the shards it rebuilds and does
/// not verify anything. Instances are immutable after construction and safe to share between
/// threads; build one per geometry and keep it.
/// </remarks>
public sealed unsafe partial class ReedSolomon
{
    private const int MaxTotalShards = 256;

    private readonly CodingMatrix _matrix;
    private readonly MulTables _encodeTables;
    private readonly InverseCache? _inverseCache;
    private readonly MulTables?[] _shardTables;
    private readonly MulTables?[] _updateTables;
    private readonly int _maxDegreeOfParallelism;
    private readonly nuint _parallelThreshold;

    /// <summary>Creates a coder with default options.</summary>
    public ReedSolomon(int dataShards, int parityShards)
        : this(dataShards, parityShards, ReedSolomonOptions.Default)
    {
    }

    /// <summary>Creates a coder.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Fewer than one data or parity shard, or more than 256 in total.</exception>
    /// <exception cref="ArgumentException"><see cref="ReedSolomonOptions.CustomParityRows"/> has the wrong shape.</exception>
    /// <exception cref="PlatformNotSupportedException"><see cref="ReedSolomonOptions.Kernel"/> names a tier this CPU lacks.</exception>
    public ReedSolomon(int dataShards, int parityShards, ReedSolomonOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dataShards, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(parityShards, 1);
        if ((long)dataShards + parityShards > MaxTotalShards)
            throw new ArgumentOutOfRangeException(nameof(parityShards), $"dataShards + parityShards must not exceed {MaxTotalShards}.");
        ArgumentNullException.ThrowIfNull(options);
        if (options.InversionCache) ArgumentOutOfRangeException.ThrowIfLessThan(options.InversionCacheSize, 1);
        if (options.MaxDegreeOfParallelism == 0 || options.MaxDegreeOfParallelism < -1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxDegreeOfParallelism must be -1 or a positive number.");
        ArgumentOutOfRangeException.ThrowIfLessThan(options.ParallelThresholdBytes, 1);

        DataShards = dataShards;
        ParityShards = parityShards;
        TotalShards = dataShards + parityShards;
        Options = options;
        Kernel = Internal.Kernel.Resolve(options.Kernel);

        _matrix = options.CustomParityRows is { } rows
            ? CodingMatrix.FromParityRows(rows, dataShards, parityShards)
            : CodingMatrix.Create(options.Matrix, dataShards, parityShards);
        _encodeTables = new MulTables(_matrix.Data.Slice(dataShards * dataShards), parityShards, dataShards);
        _inverseCache = options.InversionCache ? new InverseCache(options.InversionCacheSize) : null;
        _shardTables = new MulTables?[dataShards];
        _updateTables = new MulTables?[dataShards];
        _maxDegreeOfParallelism = options.MaxDegreeOfParallelism == -1 ? Environment.ProcessorCount : options.MaxDegreeOfParallelism;
        _parallelThreshold = (nuint)options.ParallelThresholdBytes;
    }

    /// <summary>Number of data shards.</summary>
    public int DataShards { get; }

    /// <summary>Number of parity shards.</summary>
    public int ParityShards { get; }

    /// <summary>Data plus parity shards.</summary>
    public int TotalShards { get; }

    /// <summary>The options this coder was built with.</summary>
    public ReedSolomonOptions Options { get; }

    /// <summary>The kernel tier this coder runs.</summary>
    public KernelTier Kernel { get; }

    /// <summary>The fastest kernel tier this CPU supports.</summary>
    public static KernelTier BestSupportedKernel => Internal.Kernel.Best();

    /// <summary>Cache misses of the decode-set cache; diagnostics only.</summary>
    internal int InversionCacheMisses => _inverseCache?.Misses ?? -1;

    // ------------------------------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Computes parity. <paramref name="shards"/> holds <see cref="TotalShards"/> arrays of equal length;
    /// the first <see cref="DataShards"/> are read and the rest are overwritten.
    /// </summary>
    public void Encode(byte[][] shards)
    {
        int len = CheckArrays(shards, TotalShards);
        if (len == 0) return;
        CheckDistinctArrays(shards);

        byte** ptrs = stackalloc byte*[TotalShards];
        var op = new DotOperation(this, null, DataShards, null, ParityShards, _encodeTables, 0, (nuint)len);
        Pinner.Run(shards, 0, ptrs, ref op);
    }

    /// <summary>
    /// Computes parity from <paramref name="data"/> (<see cref="DataShards"/> entries) into
    /// <paramref name="parity"/> (<see cref="ParityShards"/> entries). All entries have the same length.
    /// </summary>
    public void Encode(ReadOnlySpan<ReadOnlyMemory<byte>> data, ReadOnlySpan<Memory<byte>> parity)
    {
        int len = CheckMemories(data, parity, DataShards, ParityShards);
        if (len == 0) return;
        CheckNoOverlap(data, parity);

        byte** ptrs = stackalloc byte*[TotalShards];
        var op = new DotOperation(this, null, DataShards, null, ParityShards, _encodeTables, 0, (nuint)len);
        Pinner.Run(data, default, parity, ptrs, ref op);
    }

    /// <summary>
    /// Computes parity for shards stored back to back: <paramref name="data"/> is <see cref="DataShards"/>
    /// consecutive shards of <paramref name="shardLength"/> bytes, <paramref name="parity"/> receives
    /// <see cref="ParityShards"/> of them.
    /// </summary>
    /// <remarks>
    /// Convenient, not faster: shards exactly a power of two apart share cache sets, and a stripe of
    /// 1 MiB shards measured 4.5x slower than the same shards in separate buffers. Prefer
    /// <see cref="AllocateShards(int, int)"/>, which pads the stride, when the layout is yours to choose.
    /// </remarks>
    public void Encode(ReadOnlySpan<byte> data, Span<byte> parity, int shardLength)
    {
        CheckStripe(data.Length, DataShards, shardLength, nameof(data));
        CheckStripe(parity.Length, ParityShards, shardLength, nameof(parity));
        if (shardLength == 0) return;
        if (data.Overlaps(parity)) throw new ArgumentException("Parity must not overlap the data.", nameof(parity));

        byte** srcs = stackalloc byte*[DataShards];
        byte** dsts = stackalloc byte*[ParityShards];
        fixed (byte* d = data)
        fixed (byte* p = parity)
        {
            for (int i = 0; i < DataShards; i++) srcs[i] = d + (nint)i * shardLength;
            for (int i = 0; i < ParityShards; i++) dsts[i] = p + (nint)i * shardLength;
            Run(srcs, DataShards, dsts, ParityShards, _encodeTables.Pointer, _encodeTables.Pointer + _encodeTables.GfniOffset, (nuint)shardLength);
        }
    }

    /// <summary>
    /// Adds one data shard's contribution to <paramref name="parity"/>, for building parity
    /// incrementally: zero the parity shards, then call this once for every data index, in any order.
    /// Several shards at once through <see cref="EncodeShards"/> read and write the parity only once
    /// per batch and run several times faster than one call per shard.
    /// </summary>
    public void EncodeShard(int dataIndex, ReadOnlySpan<byte> shard, ReadOnlySpan<Memory<byte>> parity)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dataIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(dataIndex, DataShards);
        int len = CheckMemories(default, parity, 0, ParityShards);
        if (len != shard.Length) throw new ArgumentException("Shards are different sizes.", nameof(shard));
        if (len == 0) return;

        byte** ptrs = stackalloc byte*[ParityShards];
        fixed (byte* s = shard)
        {
            var op = new AccumulateOperation(this, s, ShardTables(dataIndex), 2, (nuint)len);
            Pinner.Run(default, default, parity, ptrs, ref op);
        }
    }

    /// <summary>
    /// Adds one data shard's contribution to a contiguous <paramref name="parity"/> stripe of
    /// <see cref="ParityShards"/> shards of <paramref name="shardLength"/> bytes. See <see cref="EncodeShard(int, ReadOnlySpan{byte}, ReadOnlySpan{Memory{byte}})"/>.
    /// </summary>
    public void EncodeShard(int dataIndex, ReadOnlySpan<byte> shard, Span<byte> parity, int shardLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dataIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(dataIndex, DataShards);
        CheckStripe(parity.Length, ParityShards, shardLength, nameof(parity));
        if (shard.Length != shardLength) throw new ArgumentException("Shards are different sizes.", nameof(shard));
        if (shardLength == 0) return;

        byte** ptrs = stackalloc byte*[ParityShards];
        fixed (byte* s = shard)
        fixed (byte* p = parity)
        {
            for (int i = 0; i < ParityShards; i++) ptrs[i] = p + (nint)i * shardLength;
            var op = new AccumulateOperation(this, s, ShardTables(dataIndex), 2, (nuint)shardLength);
            op.Execute(ptrs);
        }
    }

    /// <summary>
    /// Adds several data shards' contributions to <paramref name="parity"/> in one pass: the parity
    /// is read and written once for the whole batch instead of once per shard. <paramref name="dataIndices"/>
    /// are distinct data shard indices; <paramref name="shards"/> holds one entry per index.
    /// Zero the parity first, feed every data index exactly once across any number of calls.
    /// </summary>
    public void EncodeShards(ReadOnlySpan<int> dataIndices, ReadOnlySpan<ReadOnlyMemory<byte>> shards, ReadOnlySpan<Memory<byte>> parity)
    {
        if (shards.Length != dataIndices.Length) throw new ArgumentException("One shard per data index is required.", nameof(shards));
        CheckDistinctDataIndices(dataIndices);
        int len = CheckMemories(shards, parity, shards.Length, ParityShards);
        if (len == 0 || dataIndices.Length == 0) return;
        CheckNoOverlap(shards, parity);

        int n = dataIndices.Length;
        MulTables tables = ColumnTables(dataIndices, repeat: 1);
        byte** ptrs = stackalloc byte*[n + ParityShards];
        var op = new DotOperation(this, null, n, null, ParityShards, tables, 0, (nuint)len) { Accumulate = true };
        Pinner.Run(shards, default, parity, ptrs, ref op);
    }

    /// <summary>
    /// Updates parity after some data shards changed, without re-reading the unchanged ones:
    /// parity ^= M * (old ^ new) for each changed shard. <paramref name="oldData"/> and
    /// <paramref name="newData"/> hold one entry per changed index, in the order of <paramref name="changedIndices"/>.
    /// </summary>
    public void Update(
        ReadOnlySpan<int> changedIndices,
        ReadOnlySpan<ReadOnlyMemory<byte>> oldData,
        ReadOnlySpan<ReadOnlyMemory<byte>> newData,
        ReadOnlySpan<Memory<byte>> parity)
    {
        if (oldData.Length != changedIndices.Length) throw new ArgumentException("One old shard per changed index is required.", nameof(oldData));
        if (newData.Length != changedIndices.Length) throw new ArgumentException("One new shard per changed index is required.", nameof(newData));
        CheckDistinctDataIndices(changedIndices);
        int len = CheckMemories(oldData, parity, oldData.Length, ParityShards);
        foreach (var m in newData)
            if (m.Length != len) throw new ArgumentException("Shards are different sizes.", nameof(newData));
        if (len == 0 || changedIndices.Length == 0) return;

        // Bounded by DataShards, so the pointer array and the pin recursion stay within 3 * 256 frames.
        int n = changedIndices.Length;
        byte** ptrs = stackalloc byte*[2 * n + ParityShards];
        fixed (int* indices = changedIndices)
        {
            var op = new UpdateOperation(this, indices, n, (nuint)len);
            Pinner.Run(oldData, newData, parity, ptrs, ref op);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Verify
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Returns true when the parity shards match the data shards. Never throws for a mismatch.
    /// Works in 64 KiB pieces through a small pooled scratch, so the data and parity are each read
    /// once from memory whatever the shard length, and a mismatch stops at the first bad piece.
    /// </summary>
    public bool Verify(byte[][] shards)
    {
        int len = CheckArrays(shards, TotalShards);
        if (len == 0) return true;

        var op = new VerifyOperation(this, (nuint)len);
        byte** ptrs = stackalloc byte*[TotalShards];
        try
        {
            Pinner.Run(shards, 0, ptrs, ref op);
        }
        finally
        {
            op.Release();
        }

        return op.Matches;
    }

    /// <summary>Returns true when the parity shards match the data shards. Works in 64 KiB pieces through a small pooled scratch.</summary>
    public bool Verify(ReadOnlySpan<ReadOnlyMemory<byte>> shards)
    {
        int len = CheckMemories(shards, default, TotalShards, 0);
        if (len == 0) return true;

        var op = new VerifyOperation(this, (nuint)len);
        byte** ptrs = stackalloc byte*[TotalShards];
        try
        {
            Pinner.Run(shards, default, default, ptrs, ref op);
        }
        finally
        {
            op.Release();
        }

        return op.Matches;
    }

    /// <summary>
    /// Returns true when the parity shards match the data shards, using caller-supplied scratch of at
    /// least <see cref="ParityShards"/> times the shard length. On return the scratch holds the expected parity.
    /// </summary>
    public bool Verify(ReadOnlySpan<ReadOnlyMemory<byte>> shards, Span<byte> scratch)
    {
        int len = CheckMemories(shards, default, TotalShards, 0);
        if (len == 0) return true;
        if (scratch.Length < ScratchLength(len)) throw new ArgumentException("Scratch is too small.", nameof(scratch));
        CheckNoOverlap(shards, scratch);

        byte** ptrs = stackalloc byte*[TotalShards];
        fixed (byte* s = scratch)
        {
            var op = new DotOperation(this, null, DataShards, null, 0, _encodeTables, 0, (nuint)len) { ContiguousDst = s, ContiguousStride = (nuint)len, DstCount = ParityShards };
            Pinner.Run(shards, default, default, ptrs, ref op);
        }

        for (int p = 0; p < ParityShards; p++)
            if (!scratch.Slice(p * len, len).SequenceEqual(shards[DataShards + p].Span)) return false;
        return true;
    }

    /// <summary>Returns true when the parity in a contiguous stripe of <see cref="TotalShards"/> shards matches its data. Works in 64 KiB pieces through a small pooled scratch.</summary>
    public bool Verify(ReadOnlySpan<byte> shards, int shardLength)
    {
        CheckStripe(shards.Length, TotalShards, shardLength, nameof(shards));
        if (shardLength == 0) return true;

        var op = new VerifyOperation(this, (nuint)shardLength);
        byte** ptrs = stackalloc byte*[TotalShards];
        try
        {
            fixed (byte* p = shards)
            {
                for (int i = 0; i < TotalShards; i++) ptrs[i] = p + (nint)i * shardLength;
                op.Execute(ptrs);
            }
        }
        finally
        {
            op.Release();
        }

        return op.Matches;
    }

    /// <summary>Contiguous-stripe verify with caller-supplied scratch of at least <see cref="ParityShards"/> times <paramref name="shardLength"/>.</summary>
    public bool Verify(ReadOnlySpan<byte> shards, int shardLength, Span<byte> scratch)
    {
        CheckStripe(shards.Length, TotalShards, shardLength, nameof(shards));
        if (shardLength == 0) return true;
        if (scratch.Length < ScratchLength(shardLength)) throw new ArgumentException("Scratch is too small.", nameof(scratch));
        if (scratch.Overlaps(shards)) throw new ArgumentException("Scratch must not overlap the shards.", nameof(scratch));

        Encode(shards[..(DataShards * shardLength)], scratch[..(ParityShards * shardLength)], shardLength);
        return scratch[..(ParityShards * shardLength)].SequenceEqual(shards.Slice(DataShards * shardLength, ParityShards * shardLength));
    }

    // ------------------------------------------------------------------------------------------
    // Reconstruct
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Rebuilds every null entry of <paramref name="shards"/> (<see cref="TotalShards"/> entries),
    /// allocating a new array for each and storing it in place.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <see cref="DataShards"/> entries are non-null.</exception>
    public void Reconstruct(byte[]?[] shards) => ReconstructArrays(shards, Target.All, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][])"/> but rebuilds only missing data shards.</summary>
    public void ReconstructData(byte[]?[] shards) => ReconstructArrays(shards, Target.DataOnly, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][])"/> but rebuilds only missing shards whose <paramref name="required"/> flag is set (length <see cref="TotalShards"/> or <see cref="DataShards"/>).</summary>
    public void ReconstructSome(byte[]?[] shards, ReadOnlySpan<bool> required) => ReconstructArrays(shards, Target.Some, required, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][])"/> but returns false instead of throwing when too few shards are present.</summary>
    public bool TryReconstruct(byte[]?[] shards) => ReconstructArrays(shards, Target.All, default, throwIfInsufficient: false);

    /// <summary>
    /// Rebuilds every empty entry of <paramref name="shards"/> (<see cref="TotalShards"/> entries),
    /// allocating a new buffer for each and storing it in the span slot.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <see cref="DataShards"/> entries are non-empty.</exception>
    public void Reconstruct(Span<Memory<byte>> shards) => ReconstructMemories(shards, Target.All, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{Memory{byte}})"/> but rebuilds only missing data shards.</summary>
    public void ReconstructData(Span<Memory<byte>> shards) => ReconstructMemories(shards, Target.DataOnly, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{Memory{byte}})"/> but rebuilds only missing shards whose <paramref name="required"/> flag is set.</summary>
    public void ReconstructSome(Span<Memory<byte>> shards, ReadOnlySpan<bool> required) => ReconstructMemories(shards, Target.Some, required, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{Memory{byte}})"/> but returns false instead of throwing when too few shards are present.</summary>
    public bool TryReconstruct(Span<Memory<byte>> shards) => ReconstructMemories(shards, Target.All, default, throwIfInsufficient: false);

    /// <summary>
    /// Rebuilds every shard whose <paramref name="present"/> flag is false into the buffer already at
    /// that index. Nothing is allocated after the first call for a given erasure pattern.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <see cref="DataShards"/> flags are set.</exception>
    public void Reconstruct(byte[][] shards, ReadOnlySpan<bool> present) => ReconstructArrays(shards, present, Target.All, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][], ReadOnlySpan{bool})"/> but rebuilds only missing data shards.</summary>
    public void ReconstructData(byte[][] shards, ReadOnlySpan<bool> present) => ReconstructArrays(shards, present, Target.DataOnly, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][], ReadOnlySpan{bool})"/> but rebuilds only missing shards whose <paramref name="required"/> flag is set.</summary>
    public void ReconstructSome(byte[][] shards, ReadOnlySpan<bool> present, ReadOnlySpan<bool> required) => ReconstructArrays(shards, present, Target.Some, required, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(byte[][], ReadOnlySpan{bool})"/> but returns false instead of throwing when too few shards are present.</summary>
    public bool TryReconstruct(byte[][] shards, ReadOnlySpan<bool> present) => ReconstructArrays(shards, present, Target.All, default, throwIfInsufficient: false);

    /// <summary>
    /// Rebuilds every shard whose <paramref name="present"/> flag is false into the memory already at
    /// that index. Nothing is allocated after the first call for a given erasure pattern.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <see cref="DataShards"/> flags are set.</exception>
    public void Reconstruct(ReadOnlySpan<Memory<byte>> shards, ReadOnlySpan<bool> present) => ReconstructMemories(shards, present, Target.All, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(ReadOnlySpan{Memory{byte}}, ReadOnlySpan{bool})"/> but rebuilds only missing data shards.</summary>
    public void ReconstructData(ReadOnlySpan<Memory<byte>> shards, ReadOnlySpan<bool> present) => ReconstructMemories(shards, present, Target.DataOnly, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(ReadOnlySpan{Memory{byte}}, ReadOnlySpan{bool})"/> but rebuilds only missing shards whose <paramref name="required"/> flag is set.</summary>
    public void ReconstructSome(ReadOnlySpan<Memory<byte>> shards, ReadOnlySpan<bool> present, ReadOnlySpan<bool> required) => ReconstructMemories(shards, present, Target.Some, required, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(ReadOnlySpan{Memory{byte}}, ReadOnlySpan{bool})"/> but returns false instead of throwing when too few shards are present.</summary>
    public bool TryReconstruct(ReadOnlySpan<Memory<byte>> shards, ReadOnlySpan<bool> present) => ReconstructMemories(shards, present, Target.All, default, throwIfInsufficient: false);

    /// <summary>
    /// Rebuilds missing shards inside a contiguous stripe of <see cref="TotalShards"/> shards of
    /// <paramref name="shardLength"/> bytes, in place, for every index whose <paramref name="present"/> flag is false.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <see cref="DataShards"/> flags are set.</exception>
    public void Reconstruct(Span<byte> shards, int shardLength, ReadOnlySpan<bool> present) => ReconstructStripe(shards, shardLength, present, Target.All, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{byte}, int, ReadOnlySpan{bool})"/> but rebuilds only missing data shards.</summary>
    public void ReconstructData(Span<byte> shards, int shardLength, ReadOnlySpan<bool> present) => ReconstructStripe(shards, shardLength, present, Target.DataOnly, default, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{byte}, int, ReadOnlySpan{bool})"/> but rebuilds only missing shards whose <paramref name="required"/> flag is set.</summary>
    public void ReconstructSome(Span<byte> shards, int shardLength, ReadOnlySpan<bool> present, ReadOnlySpan<bool> required) => ReconstructStripe(shards, shardLength, present, Target.Some, required, throwIfInsufficient: true);

    /// <summary>Like <see cref="Reconstruct(Span{byte}, int, ReadOnlySpan{bool})"/> but returns false instead of throwing when too few shards are present.</summary>
    public bool TryReconstruct(Span<byte> shards, int shardLength, ReadOnlySpan<bool> present) => ReconstructStripe(shards, shardLength, present, Target.All, default, throwIfInsufficient: false);

    private enum Target
    {
        All,
        DataOnly,
        Some,
    }

    private bool ReconstructArrays(byte[]?[] shards, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient)
    {
        ArgumentNullException.ThrowIfNull(shards);
        if (shards.Length != TotalShards) throw new ArgumentException($"Wrong number of shards: {shards.Length}.", nameof(shards));

        Span<bool> present = stackalloc bool[TotalShards];
        int len = -1;
        for (int i = 0; i < TotalShards; i++)
        {
            byte[]? s = shards[i];
            present[i] = s is not null;
            if (s is null) continue;
            if (len < 0) len = s.Length;
            else if (s.Length != len) throw new ArgumentException("Shards are different sizes.", nameof(shards));
        }

        DecodeSet? set = PlanReconstruction(present, target, required, throwIfInsufficient, out bool ok);
        if (set is null) return ok;

        foreach (int i in set.Outputs) shards[i] = new byte[len];
        if (len == 0) return true;

        byte** ptrs = stackalloc byte*[TotalShards];
        fixed (int* inputs = set.Inputs)
        fixed (int* outputs = set.Outputs)
        {
            var op = new DotOperation(this, inputs, set.Inputs.Length, outputs, set.Outputs.Length, set.Tables, 0, (nuint)len);
            Pinner.Run(shards, 0, ptrs, ref op);
        }

        return true;
    }

    private bool ReconstructArrays(byte[][] shards, ReadOnlySpan<bool> present, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient)
    {
        int len = CheckArrays(shards, TotalShards);
        if (present.Length != TotalShards) throw new ArgumentException($"present must have {TotalShards} entries.", nameof(present));

        DecodeSet? set = PlanReconstruction(present, target, required, throwIfInsufficient, out bool ok);
        if (set is null) return ok;
        if (len == 0) return true;

        // Empty arrays may legitimately share Array.Empty; from here every array is written or read.
        for (int i = 1; i < TotalShards; i++)
            for (int j = 0; j < i; j++)
                if (ReferenceEquals(shards[i], shards[j]))
                    throw new ArgumentException($"Shard {i} is the same array as shard {j}.", nameof(shards));

        byte** ptrs = stackalloc byte*[TotalShards];
        fixed (int* inputs = set.Inputs)
        fixed (int* outputs = set.Outputs)
        {
            var op = new DotOperation(this, inputs, set.Inputs.Length, outputs, set.Outputs.Length, set.Tables, 0, (nuint)len);
            Pinner.Run(shards, 0, ptrs, ref op);
        }

        return true;
    }

    private bool ReconstructMemories(Span<Memory<byte>> shards, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient)
    {
        if (shards.Length != TotalShards) throw new ArgumentException($"Wrong number of shards: {shards.Length}.", nameof(shards));

        Span<bool> present = stackalloc bool[TotalShards];
        int len = -1;
        for (int i = 0; i < TotalShards; i++)
        {
            int l = shards[i].Length;
            present[i] = l != 0;
            if (l == 0) continue;
            if (len < 0) len = l;
            else if (l != len) throw new ArgumentException("Shards are different sizes.", nameof(shards));
        }

        DecodeSet? set = PlanReconstruction(present, target, required, throwIfInsufficient, out bool ok);
        if (set is null) return ok;

        foreach (int i in set.Outputs) shards[i] = new byte[len];
        if (len == 0) return true;

        RunDecode(set, shards, (nuint)len);
        return true;
    }

    private bool ReconstructMemories(ReadOnlySpan<Memory<byte>> shards, ReadOnlySpan<bool> present, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient)
    {
        int len = CheckMemories(default, shards, 0, TotalShards);
        if (present.Length != TotalShards) throw new ArgumentException($"present must have {TotalShards} entries.", nameof(present));

        DecodeSet? set = PlanReconstruction(present, target, required, throwIfInsufficient, out bool ok);
        if (set is null) return ok;
        if (len == 0) return true;

        RunDecode(set, shards, (nuint)len);
        return true;
    }

    private void RunDecode(DecodeSet set, ReadOnlySpan<Memory<byte>> shards, nuint len)
    {
        byte** ptrs = stackalloc byte*[TotalShards];
        fixed (int* inputs = set.Inputs)
        fixed (int* outputs = set.Outputs)
        {
            var op = new DotOperation(this, inputs, set.Inputs.Length, outputs, set.Outputs.Length, set.Tables, 0, len);
            Pinner.Run(default, default, shards, ptrs, ref op);
        }
    }

    private bool ReconstructStripe(Span<byte> shards, int shardLength, ReadOnlySpan<bool> present, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient)
    {
        CheckStripe(shards.Length, TotalShards, shardLength, nameof(shards));
        if (present.Length != TotalShards) throw new ArgumentException($"present must have {TotalShards} entries.", nameof(present));

        DecodeSet? set = PlanReconstruction(present, target, required, throwIfInsufficient, out bool ok);
        if (set is null) return ok;
        if (shardLength == 0) return true;

        byte** srcs = stackalloc byte*[set.Inputs.Length];
        byte** dsts = stackalloc byte*[set.Outputs.Length];
        fixed (byte* p = shards)
        {
            for (int i = 0; i < set.Inputs.Length; i++) srcs[i] = p + (nint)set.Inputs[i] * shardLength;
            for (int i = 0; i < set.Outputs.Length; i++) dsts[i] = p + (nint)set.Outputs[i] * shardLength;
            Run(srcs, set.Inputs.Length, dsts, set.Outputs.Length, set.Tables.Pointer, set.Tables.Pointer + set.Tables.GfniOffset, (nuint)shardLength);
        }

        return true;
    }

    /// <summary>
    /// Decides what a reconstruction has to do. Returns null when there is nothing to do or (with
    /// <paramref name="throwIfInsufficient"/> false) when it cannot be done; <paramref name="ok"/> tells which.
    /// </summary>
    private DecodeSet? PlanReconstruction(ReadOnlySpan<bool> present, Target target, ReadOnlySpan<bool> required, bool throwIfInsufficient, out bool ok)
    {
        if (target == Target.Some && required.Length != TotalShards && required.Length != DataShards)
            throw new ArgumentException($"required must have {TotalShards} or {DataShards} entries.", nameof(required));

        int presentCount = 0;
        ShardMask inputs = default;
        for (int i = 0; i < TotalShards && presentCount < DataShards; i++)
        {
            if (!present[i]) continue;
            inputs = inputs.With(i);
            presentCount++;
        }

        if (presentCount < DataShards)
        {
            presentCount = 0;
            for (int i = 0; i < TotalShards; i++) if (present[i]) presentCount++;
            if (throwIfInsufficient) throw new InsufficientShardsException(presentCount, DataShards);
            ok = false;
            return null;
        }

        ShardMask outputs = default;
        int outputCount = 0;
        int limit = target switch
        {
            Target.DataOnly => DataShards,
            Target.Some => required.Length,
            _ => TotalShards,
        };
        for (int i = 0; i < limit; i++)
        {
            if (present[i]) continue;
            if (target == Target.Some && !required[i]) continue;
            outputs = outputs.With(i);
            outputCount++;
        }

        ok = true;
        if (outputCount == 0) return null;

        if (_inverseCache is null) return BuildDecodeSet(inputs, outputs);
        if (_inverseCache.TryGet(inputs, outputs, out DecodeSet? cached)) return cached;
        return _inverseCache.Add(inputs, outputs, BuildDecodeSet(inputs, outputs));
    }

    private DecodeSet BuildDecodeSet(ShardMask inputMask, ShardMask outputMask)
    {
        int[] inputs = Indices(inputMask, DataShards);
        int[] outputs = Indices(outputMask, TotalShards);

        CodingMatrix inverse = _matrix.SubMatrix(inputs).Invert();
        var rows = new byte[outputs.Length * DataShards];
        for (int o = 0; o < outputs.Length; o++)
        {
            int shard = outputs[o];
            Span<byte> row = rows.AsSpan(o * DataShards, DataShards);
            if (shard < DataShards)
            {
                inverse.RowSpan(shard).CopyTo(row);
            }
            else
            {
                // A missing parity shard is its coding row applied to the data, and the data is the
                // inverse applied to the inputs, so the row we need is parityRow * inverse.
                ReadOnlySpan<byte> parityRow = _matrix.RowSpan(shard);
                for (int c = 0; c < DataShards; c++)
                {
                    byte acc = 0;
                    for (int k = 0; k < DataShards; k++) acc ^= Gf256.Multiply(parityRow[k], inverse[k, c]);
                    row[c] = acc;
                }
            }
        }

        return new DecodeSet
        {
            Inputs = inputs,
            Outputs = outputs,
            Tables = new MulTables(rows, outputs.Length, DataShards),
        };
    }

    private static int[] Indices(ShardMask mask, int capacity)
    {
        var list = new List<int>(capacity);
        Span<ulong> words = [mask.W0, mask.W1, mask.W2, mask.W3];
        for (int w = 0; w < 4; w++)
        {
            ulong bits = words[w];
            while (bits != 0)
            {
                int bit = System.Numerics.BitOperations.TrailingZeroCount(bits);
                list.Add(w * 64 + bit);
                bits &= bits - 1;
            }
        }

        return list.ToArray();
    }

    // ------------------------------------------------------------------------------------------
    // Split and Join
    // ------------------------------------------------------------------------------------------

    /// <summary>Bytes per shard needed to hold <paramref name="dataLength"/> bytes: <c>ceil(dataLength / DataShards)</c>.</summary>
    public int GetShardLength(int dataLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dataLength);
        return (int)(((long)dataLength + DataShards - 1) / DataShards);
    }

    /// <summary>Zero bytes appended to the last data shard by <see cref="Split(ReadOnlySpan{byte})"/> for <paramref name="dataLength"/> bytes.</summary>
    public int GetPaddingLength(int dataLength) => GetShardLength(dataLength) * DataShards - dataLength;

    /// <summary>
    /// Copies <paramref name="data"/> into <see cref="DataShards"/> equal shards (the last one zero-padded)
    /// and appends <see cref="ParityShards"/> zeroed parity shards, ready for <see cref="Encode(byte[][])"/>.
    /// The same layout as klauspost's <c>Split</c>; the original length is not stored, keep it for <c>Join</c>.
    /// </summary>
    public byte[][] Split(ReadOnlySpan<byte> data)
    {
        int shardLength = GetShardLength(data.Length);
        var shards = new byte[TotalShards][];
        for (int i = 0; i < TotalShards; i++)
        {
            shards[i] = new byte[shardLength];
            int start = i * shardLength;
            if (i < DataShards && start < data.Length)
                data.Slice(start, Math.Min(shardLength, data.Length - start)).CopyTo(shards[i]);
        }

        return shards;
    }

    /// <summary>
    /// Splits <paramref name="data"/> into a caller-owned <paramref name="stripe"/> of at least
    /// <see cref="TotalShards"/> times <see cref="GetShardLength(int)"/> bytes, zeroing the padding and the
    /// parity region, and returns one slice per shard. Use with <see cref="AllocateShards(int, int)"/> for aligned, pooled layouts.
    /// </summary>
    public Memory<byte>[] Split(ReadOnlySpan<byte> data, Memory<byte> stripe)
    {
        int shardLength = GetShardLength(data.Length);
        long needed = (long)TotalShards * shardLength;
        if (stripe.Length < needed) throw new ArgumentException($"Stripe needs at least {needed} bytes.", nameof(stripe));

        Span<byte> span = stripe.Span;
        data.CopyTo(span);
        span[data.Length..(int)needed].Clear();

        var shards = new Memory<byte>[TotalShards];
        for (int i = 0; i < TotalShards; i++) shards[i] = stripe.Slice(i * shardLength, shardLength);
        return shards;
    }

    /// <summary>
    /// Concatenates the data shards into a new array of exactly <paramref name="outputLength"/> bytes.
    /// Parity shards are ignored; <paramref name="shards"/> may hold <see cref="DataShards"/> or <see cref="TotalShards"/> entries.
    /// </summary>
    /// <exception cref="InsufficientShardsException">A data shard the output needs is null.</exception>
    public byte[] Join(byte[]?[] shards, int outputLength)
    {
        ArgumentNullException.ThrowIfNull(shards);
        ArgumentOutOfRangeException.ThrowIfNegative(outputLength);
        var result = new byte[outputLength];
        JoinCore(shards, outputLength, result);
        return result;
    }

    /// <summary>Concatenates the data shards into <paramref name="destination"/>, writing exactly <paramref name="outputLength"/> bytes.</summary>
    public void Join(ReadOnlySpan<ReadOnlyMemory<byte>> shards, int outputLength, Span<byte> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(outputLength);
        if (destination.Length < outputLength) throw new ArgumentException("Destination is too small.", nameof(destination));
        JoinCore(shards, outputLength, destination[..outputLength]);
    }

    /// <summary>Concatenates the data shards into <paramref name="destination"/>, writing exactly <paramref name="outputLength"/> bytes.</summary>
    public void Join(ReadOnlySpan<ReadOnlyMemory<byte>> shards, int outputLength, IBufferWriter<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentOutOfRangeException.ThrowIfNegative(outputLength);
        int[] lengths = new int[Math.Min(shards.Length, DataShards)];
        for (int i = 0; i < lengths.Length; i++) lengths[i] = shards[i].Length;
        CheckJoinInputs(shards.Length, i => lengths[i], outputLength);

        int remaining = outputLength;
        for (int i = 0; i < DataShards && remaining > 0; i++)
        {
            ReadOnlySpan<byte> shard = shards[i].Span;
            int n = Math.Min(shard.Length, remaining);
            destination.Write(shard[..n]);
            remaining -= n;
        }
    }

    private void JoinCore(ReadOnlySpan<ReadOnlyMemory<byte>> shards, int outputLength, Span<byte> destination)
    {
        int[] lengths = new int[Math.Min(shards.Length, DataShards)];
        for (int i = 0; i < lengths.Length; i++) lengths[i] = shards[i].Length;
        CheckJoinInputs(shards.Length, i => lengths[i], outputLength);

        int written = 0;
        for (int i = 0; i < DataShards && written < outputLength; i++)
        {
            ReadOnlySpan<byte> shard = shards[i].Span;
            int n = Math.Min(shard.Length, outputLength - written);
            shard[..n].CopyTo(destination[written..]);
            written += n;
        }
    }

    private void JoinCore(byte[]?[] shards, int outputLength, Span<byte> destination)
    {
        CheckJoinInputs(shards.Length, i => shards[i]?.Length ?? 0, outputLength);

        int written = 0;
        for (int i = 0; i < DataShards && written < outputLength; i++)
        {
            ReadOnlySpan<byte> s = shards[i];
            int n = Math.Min(s.Length, outputLength - written);
            s[..n].CopyTo(destination[written..]);
            written += n;
        }
    }

    /// <summary>
    /// Join reads data shards in order until it has <paramref name="outputLength"/> bytes. Every shard
    /// it would read must be present, and together they must hold that many bytes; shards past the
    /// last one read may be missing.
    /// </summary>
    private void CheckJoinInputs(int count, Func<int, int> lengthOf, int outputLength)
    {
        if (count != DataShards && count != TotalShards)
            throw new ArgumentException($"Expected {DataShards} or {TotalShards} shards, got {count}.", "shards");
        if (outputLength == 0) return;

        long available = 0;
        int needed = 0;
        int present = 0;
        for (int i = 0; i < DataShards && available < outputLength; i++)
        {
            needed++;
            int length = lengthOf(i);
            if (length == 0)
            {
                // Present counts the needed shards that are there; Required is how many Join must read.
                throw new InsufficientShardsException($"Reconstruction is required: data shard {i} is needed by Join and is missing.", present, needed);
            }

            present++;
            available += length;
        }

        if (available < outputLength)
            throw new ArgumentException($"The data shards hold {available} bytes, fewer than the {outputLength} requested.", nameof(outputLength));
    }

    // ------------------------------------------------------------------------------------------
    // Buffers
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Allocates <paramref name="shardCount"/> shards of <paramref name="shardLength"/> bytes in one
    /// pinned buffer, each starting on a 64-byte boundary, with consecutive shards offset by one extra
    /// cache line so that shards a power of two long do not all map to the same cache sets. Alignment
    /// is optional for every kernel; the padding is what makes this layout fast.
    /// </summary>
    public static Memory<byte>[] AllocateShards(int shardCount, int shardLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(shardCount);
        ArgumentOutOfRangeException.ThrowIfNegative(shardLength);

        long strideLong = (((long)shardLength + 63) & ~63L) + 64;
        long total = shardCount * strideLong + 64;
        if (total > Array.MaxLength) throw new ArgumentOutOfRangeException(nameof(shardLength), "The stripe would exceed the maximum array length.");
        int stride = (int)strideLong;
        byte[] buffer = GC.AllocateUninitializedArray<byte>((int)total, pinned: true);
        int skew = 0;
        fixed (byte* p = buffer) skew = (int)((64 - ((nuint)p & 63)) & 63);

        var shards = new Memory<byte>[shardCount];
        for (int i = 0; i < shardCount; i++) shards[i] = buffer.AsMemory(skew + i * stride, shardLength);
        return shards;
    }

    // ------------------------------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------------------------------

    private MulTables ShardTables(int dataIndex)
    {
        MulTables? t = Volatile.Read(ref _shardTables[dataIndex]);
        if (t is not null) return t;

        // Rows [1, c_p]: parity_p = 1 * parity_p ^ c_p * shard, run in place.
        var rows = new byte[ParityShards * 2];
        for (int p = 0; p < ParityShards; p++)
        {
            rows[p * 2] = 1;
            rows[p * 2 + 1] = _matrix[DataShards + p, dataIndex];
        }

        t = new MulTables(rows, ParityShards, 2);
        return Interlocked.CompareExchange(ref _shardTables[dataIndex], t, null) ?? t;
    }

    private MulTables UpdateTables(int dataIndex)
    {
        MulTables? t = Volatile.Read(ref _updateTables[dataIndex]);
        if (t is not null) return t;

        // Rows [1, c_p, c_p]: parity_p ^= c_p * old ^ c_p * new, without a scratch buffer for old ^ new.
        var rows = new byte[ParityShards * 3];
        for (int p = 0; p < ParityShards; p++)
        {
            rows[p * 3] = 1;
            rows[p * 3 + 1] = _matrix[DataShards + p, dataIndex];
            rows[p * 3 + 2] = rows[p * 3 + 1];
        }

        t = new MulTables(rows, ParityShards, 3);
        return Interlocked.CompareExchange(ref _updateTables[dataIndex], t, null) ?? t;
    }

    /// <summary>
    /// Tables for a batch of data indices: ParityShards rows over the chosen columns of the coding
    /// matrix, each column repeated <paramref name="repeat"/> times (twice for Update's old and new).
    /// Built per call; a batch is at most DataShards columns and the tables are a few KiB.
    /// </summary>
    private MulTables ColumnTables(ReadOnlySpan<int> dataIndices, int repeat)
    {
        int n = dataIndices.Length;
        var rows = new byte[ParityShards * n * repeat];
        for (int p = 0; p < ParityShards; p++)
            for (int r = 0; r < repeat; r++)
                for (int i = 0; i < n; i++)
                    rows[p * n * repeat + r * n + i] = _matrix[DataShards + p, dataIndices[i]];
        return new MulTables(rows, ParityShards, n * repeat);
    }

    private void CheckDistinctDataIndices(ReadOnlySpan<int> indices)
    {
        if (indices.Length > DataShards)
            throw new ArgumentException($"At most {DataShards} data indices are possible.", nameof(indices));
        Span<bool> seen = stackalloc bool[DataShards];
        foreach (int index in indices)
        {
            if ((uint)index >= (uint)DataShards) throw new ArgumentOutOfRangeException(nameof(indices), index, "Not a data shard index.");
            if (seen[index]) throw new ArgumentException($"Index {index} is listed twice.", nameof(indices));
            seen[index] = true;
        }
    }

    /// <summary>
    /// Runs the kernel over the whole byte range, splitting it across the thread pool when the
    /// options allow and the range is long enough to pay for the hand-off.
    /// </summary>
    /// <summary>Shard length up to which a many-output call is split by outputs rather than by byte range.</summary>
    private const int OutputSplitMaxLength = 256 * 1024;

    private void Run(byte** srcs, int srcCount, byte** dsts, int dstCount, byte* shuffle, byte* gfni, nuint len, bool accumulate = false)
    {
        int workers = _maxDegreeOfParallelism;

        // Many outputs on shards that fit L3: split by blocks of four outputs, so each worker owns
        // whole outputs, no worker reloads another's partial sums, and the inputs are shared through
        // L3. Gated on work (bytes touched), not shard length: 50+20 at 64 KiB is a millisecond.
        if (workers > 1 && dstCount >= 8 && len <= OutputSplitMaxLength)
        {
            int blocks = (dstCount + 3) / 4;
            nuint work = len * (nuint)(srcCount * blocks + dstCount);
            if (work >= _parallelThreshold * (nuint)(DataShards + ParityShards))
            {
                int outputWorkers = Math.Min(workers, blocks);
                int blocksPerWorker = (blocks + outputWorkers - 1) / outputWorkers;
                var split = new ParallelJob
                {
                    Tier = Kernel,
                    Srcs = (nint)srcs,
                    SrcCount = srcCount,
                    Dsts = (nint)dsts,
                    DstCount = dstCount,
                    Shuffle = (nint)shuffle,
                    Gfni = (nint)gfni,
                    Len = len,
                    Accumulate = accumulate,
                    OutputsPerWorker = blocksPerWorker * 4,
                };
                Parallel.For(0, outputWorkers, new ParallelOptions { MaxDegreeOfParallelism = outputWorkers }, split.ExecuteOutputs);
                return;
            }
        }

        if (workers > 1)
        {
            nuint byThreshold = len / _parallelThreshold;
            if (byThreshold < (nuint)workers) workers = (int)byThreshold;
        }

        if (workers <= 1)
        {
            Internal.Kernel.DotProduct(Kernel, srcs, srcCount, dsts, dstCount, shuffle, gfni, len, accumulate);
            return;
        }

        // Chunks are multiples of 64 so every worker's vector loop starts where the previous one's
        // ended, and no worker gets a scalar tail except the last. Ceiling division: with a floor,
        // 961 bytes over three workers gave chunks of 320 and never touched byte 960.
        nuint chunk = (((len + (nuint)workers - 1) / (nuint)workers) + 63) & ~(nuint)63;
        var job = new ParallelJob
        {
            Tier = Kernel,
            Srcs = (nint)srcs,
            SrcCount = srcCount,
            Dsts = (nint)dsts,
            DstCount = dstCount,
            Shuffle = (nint)shuffle,
            Gfni = (nint)gfni,
            Len = len,
            Chunk = chunk,
            Accumulate = accumulate,
        };
        Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, job.Execute);
    }

    private sealed class ParallelJob
    {
        public KernelTier Tier;
        public nint Srcs;
        public int SrcCount;
        public nint Dsts;
        public int DstCount;
        public nint Shuffle;
        public nint Gfni;
        public nuint Len;
        public nuint Chunk;
        public bool Accumulate;
        public int OutputsPerWorker;

        /// <summary>Output-block split: worker <paramref name="worker"/> computes its own outputs over the whole length.</summary>
        public void ExecuteOutputs(int worker)
        {
            int d0 = worker * OutputsPerWorker;
            if (d0 >= DstCount) return;
            int count = Math.Min(OutputsPerWorker, DstCount - d0);
            byte* shuffle = (byte*)Shuffle + (nuint)(d0 * SrcCount) * MulTables.ShuffleEntrySize;
            byte* gfni = (byte*)Gfni + (nuint)(d0 * SrcCount) * MulTables.GfniEntrySize;
            Internal.Kernel.DotProduct(Tier, (byte**)Srcs, SrcCount, (byte**)Dsts + d0, count, shuffle, gfni, Len, Accumulate);
        }

        public void Execute(int worker)
        {
            nuint offset = (nuint)worker * Chunk;
            if (offset >= Len) return;
            nuint n = Math.Min(Chunk, Len - offset);

            byte** srcs = stackalloc byte*[SrcCount];
            byte** dsts = stackalloc byte*[DstCount];
            for (int i = 0; i < SrcCount; i++) srcs[i] = ((byte**)Srcs)[i] + offset;
            for (int i = 0; i < DstCount; i++) dsts[i] = ((byte**)Dsts)[i] + offset;
            Internal.Kernel.DotProduct(Tier, srcs, SrcCount, dsts, DstCount, (byte*)Shuffle, (byte*)Gfni, n, Accumulate);
        }
    }

    /// <summary>
    /// Verification over pinned shards in pieces: encode one piece of every parity shard into a pooled
    /// scratch of ParityShards x <see cref="VerifyPieceBytes"/>, compare with the real parity, stop at
    /// the first mismatch. The scratch stays in L2, so memory traffic is the data and the parity, once.
    /// </summary>
    private struct VerifyOperation : IPinnedOperation
    {
        private readonly ReedSolomon _coder;
        private readonly nuint _len;
        private readonly nuint _piece;
        private byte[]? _scratch;

        public bool Matches;

        public VerifyOperation(ReedSolomon coder, nuint len)
        {
            _coder = coder;
            _len = len;
            _piece = Math.Min(len, (nuint)VerifyPieceBytes);
            _scratch = ArrayPool<byte>.Shared.Rent(coder.ParityShards * (int)_piece);
            Matches = false;
        }

        public void Release()
        {
            if (_scratch is { } s)
            {
                _scratch = null;
                ArrayPool<byte>.Shared.Return(s);
            }
        }

        public void Execute(byte** ptrs)
        {
            int data = _coder.DataShards;
            int parity = _coder.ParityShards;
            MulTables tables = _coder._encodeTables;
            byte** srcs = stackalloc byte*[data];
            byte** dsts = stackalloc byte*[parity];

            fixed (byte* scratch = _scratch)
            {
                for (nuint offset = 0; offset < _len; offset += _piece)
                {
                    nuint n = Math.Min(_piece, _len - offset);
                    for (int i = 0; i < data; i++) srcs[i] = ptrs[i] + offset;
                    for (int p = 0; p < parity; p++) dsts[p] = scratch + (nuint)p * _piece;
                    _coder.Run(srcs, data, dsts, parity, tables.Pointer, tables.Pointer + tables.GfniOffset, n);

                    for (int p = 0; p < parity; p++)
                    {
                        if (!new ReadOnlySpan<byte>(dsts[p], (int)n).SequenceEqual(new ReadOnlySpan<byte>(ptrs[data + p] + offset, (int)n)))
                        {
                            Matches = false;
                            return;
                        }
                    }
                }
            }

            GC.KeepAlive(tables);
            Matches = true;
        }
    }

    /// <summary>Bytes of each parity shard verified per piece; ParityShards pieces of scratch stay inside L2.</summary>
    private const int VerifyPieceBytes = 64 * 1024;

    /// <summary>
    /// A dot product over pinned shards. Source and destination pointers are picked out of the pinned
    /// array by index (null index arrays mean "the first srcCount" and "the next dstCount"), or the
    /// destinations are a contiguous scratch region when <see cref="ContiguousDst"/> is set.
    /// </summary>
    private struct DotOperation : IPinnedOperation
    {
        private readonly ReedSolomon _coder;
        private readonly int* _srcIndex;
        private readonly int _srcCount;
        private readonly int* _dstIndex;
        private readonly MulTables _tables;
        private readonly int _tableRow;
        private readonly nuint _len;

        public int DstCount;
        public byte* ContiguousDst;
        public nuint ContiguousStride;
        public bool Accumulate;

        public DotOperation(ReedSolomon coder, int* srcIndex, int srcCount, int* dstIndex, int dstCount, MulTables tables, int tableRow, nuint len)
        {
            _coder = coder;
            _srcIndex = srcIndex;
            _srcCount = srcCount;
            _dstIndex = dstIndex;
            DstCount = dstCount;
            _tables = tables;
            _tableRow = tableRow;
            _len = len;
        }

        public void Execute(byte** ptrs)
        {
            byte** srcs = stackalloc byte*[_srcCount];
            byte** dsts = stackalloc byte*[DstCount];
            for (int i = 0; i < _srcCount; i++) srcs[i] = _srcIndex is null ? ptrs[i] : ptrs[_srcIndex[i]];
            for (int i = 0; i < DstCount; i++)
            {
                dsts[i] = ContiguousDst is not null ? ContiguousDst + (nuint)i * ContiguousStride
                    : _dstIndex is null ? ptrs[_srcCount + i]
                    : ptrs[_dstIndex[i]];
            }

            byte* shuffle = _tables.Pointer + (nuint)(_tableRow * _tables.Cols) * MulTables.ShuffleEntrySize;
            byte* gfni = _tables.Pointer + _tables.GfniOffset + (nuint)(_tableRow * _tables.Cols) * MulTables.GfniEntrySize;
            _coder.Run(srcs, _srcCount, dsts, DstCount, shuffle, gfni, _len, Accumulate);
            GC.KeepAlive(_tables);
        }
    }

    /// <summary>
    /// Update over pinned old, new and parity memories: for each changed index, one in-place kernel
    /// call per parity shard with rows [1, c, c]. Pins everything once instead of once per index.
    /// </summary>
    private struct UpdateOperation : IPinnedOperation
    {
        private readonly ReedSolomon _coder;
        private readonly int* _indices;
        private readonly int _count;
        private readonly nuint _len;

        public UpdateOperation(ReedSolomon coder, int* indices, int count, nuint len)
        {
            _coder = coder;
            _indices = indices;
            _count = count;
            _len = len;
        }

        public void Execute(byte** ptrs)
        {
            for (int i = 0; i < _count; i++)
            {
                var op = new AccumulateOperation(_coder, null, _coder.UpdateTables(_indices[i]), 3, _len)
                {
                    OldIndex = i,
                    NewIndex = _count + i,
                    ParityOffset = 2 * _count,
                };
                op.Execute(ptrs);
            }
        }
    }

    /// <summary>
    /// parity_p = 1 * parity_p ^ c_p * shard (two sources) or ^ c_p * old ^ c_p * new (three sources),
    /// in place, one kernel call per parity shard. In-place is safe only because the source count
    /// stays within <see cref="Internal.Kernel.MaxInPlaceSources"/>; see the note there. Measured
    /// (experiments 19): at 64 KiB with eight or more parity shards this per-parity form beats a
    /// single multi-output accumulate pass by 15-70%, while the batch <see cref="EncodeShards"/>
    /// beats both several times over.
    /// </summary>
    private struct AccumulateOperation : IPinnedOperation
    {
        private readonly ReedSolomon _coder;
        private readonly byte* _shard;
        private readonly MulTables _tables;
        private readonly int _srcCount;
        private readonly nuint _len;

        public int OldIndex;
        public int NewIndex;
        public int ParityOffset;

        public AccumulateOperation(ReedSolomon coder, byte* shard, MulTables tables, int srcCount, nuint len)
        {
            System.Diagnostics.Debug.Assert(srcCount <= Internal.Kernel.MaxInPlaceSources, "In-place accumulate would read an overwritten source.");
            _coder = coder;
            _shard = shard;
            _tables = tables;
            _srcCount = srcCount;
            _len = len;
        }

        public void Execute(byte** ptrs)
        {
            byte** srcs = stackalloc byte*[3];
            byte** dsts = stackalloc byte*[1];
            for (int p = 0; p < _tables.Rows; p++)
            {
                byte* parity = ptrs[ParityOffset + p];
                srcs[0] = parity;
                if (_srcCount == 2)
                {
                    srcs[1] = _shard;
                }
                else
                {
                    srcs[1] = ptrs[OldIndex];
                    srcs[2] = ptrs[NewIndex];
                }

                dsts[0] = parity;
                byte* shuffle = _tables.Pointer + (nuint)(p * _srcCount) * MulTables.ShuffleEntrySize;
                byte* gfni = _tables.Pointer + _tables.GfniOffset + (nuint)(p * _srcCount) * MulTables.GfniEntrySize;
                _coder.Run(srcs, _srcCount, dsts, 1, shuffle, gfni, _len);
            }

            GC.KeepAlive(_tables);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------------------------------

    private static int CheckArrays(byte[][] shards, int expectedCount)
    {
        ArgumentNullException.ThrowIfNull(shards);
        if (shards.Length != expectedCount) throw new ArgumentException($"Wrong number of shards: {shards.Length}, expected {expectedCount}.", nameof(shards));
        byte[] first = shards[0] ?? throw new ArgumentException("Shard 0 is null.", nameof(shards));
        int len = first.Length;
        for (int i = 1; i < shards.Length; i++)
        {
            byte[] s = shards[i] ?? throw new ArgumentException($"Shard {i} is null.", nameof(shards));
            if (s.Length != len) throw new ArgumentException("Shards are different sizes.", nameof(shards));
        }

        return len;
    }

    private static int CheckMemories(ReadOnlySpan<ReadOnlyMemory<byte>> inputs, ReadOnlySpan<Memory<byte>> outputs, int expectedInputs, int expectedOutputs)
    {
        if (inputs.Length != expectedInputs) throw new ArgumentException($"Wrong number of shards: {inputs.Length}, expected {expectedInputs}.", "data");
        if (outputs.Length != expectedOutputs) throw new ArgumentException($"Wrong number of shards: {outputs.Length}, expected {expectedOutputs}.", "parity");

        int len = inputs.Length > 0 ? inputs[0].Length : outputs[0].Length;
        foreach (var m in inputs)
            if (m.Length != len) throw new ArgumentException("Shards are different sizes.", "data");
        foreach (var m in outputs)
            if (m.Length != len) throw new ArgumentException("Shards are different sizes.", "parity");
        return len;
    }

    /// <summary>ParityShards * len as an int, or a clear exception instead of a wrapped size feeding raw pointer writes.</summary>
    private int ScratchLength(int len)
    {
        long scratch = (long)ParityShards * len;
        if (scratch > Array.MaxLength) throw new ArgumentOutOfRangeException(nameof(len), "Parity scratch would exceed the maximum array length.");
        return (int)scratch;
    }

    /// <summary>
    /// Outputs must not alias inputs (except through the documented in-place operations): the kernel
    /// writes each output while other outputs still read the inputs. Array identity is the cheap
    /// check that catches the realistic mistake, passing the same buffer twice.
    /// </summary>
    private void CheckDistinctArrays(byte[][] shards)
    {
        for (int p = DataShards; p < TotalShards; p++)
            for (int i = 0; i < p; i++)
                if (ReferenceEquals(shards[p], shards[i]))
                    throw new ArgumentException($"Shard {p} is the same array as shard {i}; outputs must not overlap inputs.", nameof(shards));
    }

    private static void CheckNoOverlap(ReadOnlySpan<ReadOnlyMemory<byte>> inputs, ReadOnlySpan<Memory<byte>> outputs)
    {
        for (int o = 0; o < outputs.Length; o++)
        {
            ReadOnlySpan<byte> output = outputs[o].Span;
            for (int i = 0; i < inputs.Length; i++)
                if (output.Overlaps(inputs[i].Span))
                    throw new ArgumentException($"Output {o} overlaps input {i}; outputs must not overlap inputs.", "parity");
            for (int j = 0; j < o; j++)
                if (output.Overlaps(outputs[j].Span))
                    throw new ArgumentException($"Outputs {o} and {j} overlap.", "parity");
        }
    }

    private static void CheckNoOverlap(ReadOnlySpan<ReadOnlyMemory<byte>> inputs, ReadOnlySpan<byte> scratch)
    {
        for (int i = 0; i < inputs.Length; i++)
            if (scratch.Overlaps(inputs[i].Span))
                throw new ArgumentException($"Scratch overlaps shard {i}.", "scratch");
    }

    private static void CheckStripe(int length, int shards, int shardLength, string paramName)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(shardLength);
        if (length != (long)shards * shardLength)
            throw new ArgumentException($"Expected {shards} shards of {shardLength} bytes ({(long)shards * shardLength} bytes), got {length}.", paramName);
    }
}
