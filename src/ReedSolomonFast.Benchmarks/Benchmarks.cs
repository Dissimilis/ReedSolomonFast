extern alias Baseline;

using BenchmarkDotNet.Attributes;
using Current = ReedSolomonFast.ReedSolomon;
using CurrentOptions = ReedSolomonFast.ReedSolomonOptions;
using BaselineCoder = Baseline::ReedSolomonFast.ReedSolomon;
using Egbakou = ReedSolomon.NET.ReedSolomon;
using WittebornCoder = global::Witteborn.ReedSolomon.ReedSolomon;

namespace ReedSolomonFast.Benchmarks;

/// <summary>
/// Shard geometries shared across the suite, as "data+parity" strings so BenchmarkDotNet shows
/// them as one column. 5+2 and 10+4 are the common storage settings, 8+8 is the high-parity case
/// where the four-output blocking matters, 50+20 is the many-input case where table loads dominate.
/// </summary>
internal static class Geometry
{
    public static (int Data, int Parity) Parse(string geometry)
    {
        string[] parts = geometry.Split('+');
        return (int.Parse(parts[0]), int.Parse(parts[1]));
    }

    public static byte[][] Shards(int count, int length, int seed)
    {
        var rng = new Random(seed);
        var shards = new byte[count][];
        for (int i = 0; i < count; i++)
        {
            shards[i] = new byte[length];
            rng.NextBytes(shards[i]);
        }

        return shards;
    }
}

/// <summary>
/// THE DECISION HARNESS: frozen pre-optimization code versus current code, in one process.
///
/// Read the Ratio column against the "before" row of the same geometry and size. Below 1.00 is an
/// improvement. The run prints an explicit per-case verdict afterwards; prefer that to eyeballing
/// the table. BenchmarkDotNet runs each case in its own process sequentially, so the earlier row of
/// a pair runs on a slightly cooler package: the bias understates wins and overstates regressions.
/// Pin clocks for any run you intend to act on.
/// </summary>
public class OptimizationBenchmarks
{
    private byte[][] _shards = null!;
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private bool[] _present = null!;
    private bool[] _oneData = null!;
    private bool[] _parityOnly = null!;
    private int[] _indices = null!;
    private Current _current = null!;
    private BaselineCoder _baseline = null!;
    private Current _currentCold = null!;
    private BaselineCoder _baselineCold = null!;
    private Current _currentParallel = null!;
    private Current _currentStreaming = null!;
    private ReadOnlyMemory<byte>[] _paddedData = null!;
    private Memory<byte>[] _paddedParity = null!;
    private ReadOnlyMemory<byte>[] _paddedDataBefore = null!;
    private Memory<byte>[] _paddedParityBefore = null!;
    private ReadOnlyMemory<byte>[] _paddedAll = null!;
    private ReadOnlyMemory<byte>[] _paddedAllBefore = null!;
    private ReadOnlyMemory<byte>[] _changedOld = null!;
    private ReadOnlyMemory<byte>[] _changedNew = null!;
    private ReadOnlyMemory<byte>[][] _ringData = null!;
    private Memory<byte>[][] _ringParity = null!;
    private int _ring;
    private BaselineCoder _baselineParallel = null!;

    // Four shapes and two sizes keep the decision run under five minutes. 5+2 is the common
    // small case, 10+4 the common storage case, 8+8 the high-parity case, 50+20 the many-input
    // case; 64 KiB fits L2 and 1 MiB does not. Add --filter or edit here for a targeted campaign.
    [Params("5+2", "10+4", "8+8", "50+20")]
    public string Shape = "10+4";

    [Params(65_536, 1_048_576)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        _shards = Geometry.Shards(data + parity, Shard_Size, 1);
        _data = _shards.Take(data).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = _shards.Skip(data).Select(s => (Memory<byte>)s).ToArray();
        _present = Enumerable.Repeat(true, data + parity).ToArray();
        // Worst case: as many data shards missing as there is parity.
        for (int i = 0; i < parity; i++) _present[i] = false;
        _indices = Enumerable.Range(0, data).ToArray();
        _oneData = Enumerable.Repeat(true, data + parity).ToArray();
        _oneData[1] = false;
        _parityOnly = Enumerable.Repeat(true, data + parity).ToArray();
        for (int i = data; i < data + parity; i++) _parityOnly[i] = false;
        _current = new Current(data, parity);
        _baseline = new BaselineCoder(data, parity);
        _currentCold = new Current(data, parity, new CurrentOptions { InversionCache = false });
        _currentParallel = new Current(data, parity, new CurrentOptions { MaxDegreeOfParallelism = -1, ParallelThresholdBytes = 65_536 });
        _currentStreaming = new Current(data, parity, new CurrentOptions { StreamingStores = true });
        _baselineParallel = new BaselineCoder(data, parity, new Baseline::ReedSolomonFast.ReedSolomonOptions { MaxDegreeOfParallelism = -1, ParallelThresholdBytes = 65_536 });
        _baselineCold = new BaselineCoder(data, parity, new Baseline::ReedSolomonFast.ReedSolomonOptions { InversionCache = false });
        _current.Encode(_data, _parity);

        // AllocateShards layout: one buffer, padded stride (each side's own AllocateShards), so every shard shares
        // its offset within the line but no two share a cache set.
        var padded = Current.AllocateShards(data + parity, Shard_Size);
        for (int i = 0; i < data + parity; i++) _shards[i].CopyTo(padded[i]);
        _paddedData = padded.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _paddedParity = padded.Skip(data).ToArray();
        var paddedBefore = BaselineCoder.AllocateShards(data + parity, Shard_Size);
        for (int i = 0; i < data + parity; i++) _shards[i].CopyTo(paddedBefore[i]);
        _paddedDataBefore = paddedBefore.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _paddedParityBefore = paddedBefore.Skip(data).ToArray();
        _paddedAll = padded.Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _paddedAllBefore = paddedBefore.Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        var changed = Geometry.Shards(8, Shard_Size, 7);
        _changedOld = changed.Take(4).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _changedNew = changed.Skip(4).Select(s => (ReadOnlyMemory<byte>)s).ToArray();

        // A ring of padded stripes totalling about three times a 16 MB L3, one per call, so
        // inputs and outputs are cold every time, as they are for a caller streaming data through.
        long stripeBytes = (long)(data + parity) * Shard_Size;
        int ring = (int)Math.Max(1, (48L << 20) / stripeBytes);
        _ringData = new ReadOnlyMemory<byte>[ring][];
        _ringParity = new Memory<byte>[ring][];
        for (int r = 0; r < ring; r++)
        {
            var stripe = Current.AllocateShards(data + parity, Shard_Size);
            for (int i = 0; i < data; i++) _shards[i].CopyTo(stripe[i]);
            _ringData[r] = stripe.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            _ringParity[r] = stripe.Skip(data).ToArray();
        }
    }

    // The consumer a caller has right after Encode: every parity byte read once, sequentially.
    private static byte ReadAll(Memory<byte>[] parity)
    {
        var acc = System.Runtime.Intrinsics.Vector256<byte>.Zero;
        foreach (var m in parity)
        {
            var span = m.Span;
            int i = 0;
            for (; i + 32 <= span.Length; i += 32) acc ^= System.Runtime.Intrinsics.Vector256.Create<byte>(span.Slice(i, 32));
            for (; i < span.Length; i++) acc ^= System.Runtime.Intrinsics.Vector256.Create(span[i]);
        }

        return acc[0];
    }

    [Benchmark(Description = "before: Encode cold + read parity")]
    public byte BeforeEncodeColdRead() { int r = _ring = (_ring + 1) % _ringData.Length; _baseline.Encode(_ringData[r], _ringParity[r]); return ReadAll(_ringParity[r]); }

    [Benchmark(Description = "after:  Encode cold + read parity")]
    public byte AfterEncodeColdRead() { int r = _ring = (_ring + 1) % _ringData.Length; _current.Encode(_ringData[r], _ringParity[r]); return ReadAll(_ringParity[r]); }

    // Opt-in streaming stores against the plain coder, same cold ring, parity read once after.
    [Benchmark(Description = "before: Encode cold, streaming off")]
    public byte BeforeEncodeColdStreaming() { int r = _ring = (_ring + 1) % _ringData.Length; _current.Encode(_ringData[r], _ringParity[r]); return ReadAll(_ringParity[r]); }

    [Benchmark(Description = "after:  Encode cold, streaming on")]
    public byte AfterEncodeColdStreaming() { int r = _ring = (_ring + 1) % _ringData.Length; _currentStreaming.Encode(_ringData[r], _ringParity[r]); return ReadAll(_ringParity[r]); }

    [Benchmark(Description = "before: Encode cold")]
    public byte BeforeEncodeCold() { int r = _ring = (_ring + 1) % _ringData.Length; _baseline.Encode(_ringData[r], _ringParity[r]); return _ringParity[r][0].Span[0]; }

    [Benchmark(Description = "after:  Encode cold")]
    public byte AfterEncodeCold() { int r = _ring = (_ring + 1) % _ringData.Length; _current.Encode(_ringData[r], _ringParity[r]); return _ringParity[r][0].Span[0]; }

    [Benchmark(Description = "before: Encode padded")]
    public byte BeforeEncodePadded() { _baseline.Encode(_paddedDataBefore, _paddedParityBefore); return _paddedParityBefore[0].Span[0]; }

    [Benchmark(Description = "after:  Encode padded")]
    public byte AfterEncodePadded() { _current.Encode(_paddedData, _paddedParity); return _paddedParity[0].Span[0]; }

    [Benchmark(Baseline = true, Description = "before: Encode")]
    public byte BeforeEncode()
    {
        _baseline.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "after:  Encode")]
    public byte AfterEncode()
    {
        _current.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "before: Reconstruct")]
    public byte BeforeReconstruct()
    {
        _baseline.Reconstruct(_shards, _present);
        return _shards[0][0];
    }

    [Benchmark(Description = "after:  Reconstruct")]
    public byte AfterReconstruct()
    {
        _current.Reconstruct(_shards, _present);
        return _shards[0][0];
    }

    // Realistic erasure patterns. "Cold" disables the inversion cache, so planning (submatrix
    // inversion, table building) is paid on every call, as it would be across many distinct patterns.
    [Benchmark(Description = "before: Reconstruct one data")]
    public byte BeforeReconstructOneData() { _baseline.Reconstruct(_shards, _oneData); return _shards[1][0]; }

    [Benchmark(Description = "after:  Reconstruct one data")]
    public byte AfterReconstructOneData() { _current.Reconstruct(_shards, _oneData); return _shards[1][0]; }

    [Benchmark(Description = "before: Reconstruct parity only")]
    public byte BeforeReconstructParity() { _baseline.Reconstruct(_shards, _parityOnly); return _shards[^1][0]; }

    [Benchmark(Description = "after:  Reconstruct parity only")]
    public byte AfterReconstructParity() { _current.Reconstruct(_shards, _parityOnly); return _shards[^1][0]; }

    [Benchmark(Description = "before: Reconstruct parity only, cold")]
    public byte BeforeReconstructParityCold() { _baselineCold.Reconstruct(_shards, _parityOnly); return _shards[^1][0]; }

    [Benchmark(Description = "after:  Reconstruct parity only, cold")]
    public byte AfterReconstructParityCold() { _currentCold.Reconstruct(_shards, _parityOnly); return _shards[^1][0]; }

    [Benchmark(Description = "before: Reconstruct one data, cold")]
    public byte BeforeReconstructOneDataCold() { _baselineCold.Reconstruct(_shards, _oneData); return _shards[1][0]; }

    [Benchmark(Description = "after:  Reconstruct one data, cold")]
    public byte AfterReconstructOneDataCold() { _currentCold.Reconstruct(_shards, _oneData); return _shards[1][0]; }

    // Opt-in parallelism, all cores, 64 KiB threshold.
    [Benchmark(Description = "before: Encode parallel")]
    public byte BeforeEncodeParallel() { _baselineParallel.Encode(_data, _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "after:  Encode parallel")]
    public byte AfterEncodeParallel() { _currentParallel.Encode(_data, _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "before: Reconstruct parallel")]
    public byte BeforeReconstructParallel() { _baselineParallel.Reconstruct(_shards, _present); return _shards[0][0]; }

    [Benchmark(Description = "after:  Reconstruct parallel")]
    public byte AfterReconstructParallel() { _currentParallel.Reconstruct(_shards, _present); return _shards[0][0]; }

    [Benchmark(Description = "before: Verify")]
    public bool BeforeVerify() => _baseline.Verify(_shards);

    [Benchmark(Description = "after:  Verify")]
    public bool AfterVerify() => _current.Verify(_shards);

    // The same on the AllocateShards layout: what a caller who followed the README sees.
    [Benchmark(Description = "before: Verify padded")]
    public bool BeforeVerifyPadded() => _baseline.Verify(_paddedAllBefore);

    [Benchmark(Description = "after:  Verify padded")]
    public bool AfterVerifyPadded() => _current.Verify(_paddedAll);

    // Incremental parity: one EncodeShard per data shard. Clearing parity first is part of the
    // protocol and costs the same on both sides.
    [Benchmark(Description = "before: EncodeShard x data")]
    public byte BeforeEncodeShard()
    {
        foreach (var p in _parity) p.Span.Clear();
        for (int i = 0; i < _data.Length; i++) _baseline.EncodeShard(i, _data[i].Span, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "after:  EncodeShard x data")]
    public byte AfterEncodeShard()
    {
        foreach (var p in _parity) p.Span.Clear();
        for (int i = 0; i < _data.Length; i++) _current.EncodeShard(i, _data[i].Span, _parity);
        return _parity[0].Span[0];
    }

    // Update: one changed shard, then four, old and new as separate arrays like the data. The
    // parity drifts from call to call, which changes nothing about the work.
    [Benchmark(Description = "before: Update 1")]
    public byte BeforeUpdate1() { _baseline.Update([2], _changedOld.AsSpan(0, 1), _changedNew.AsSpan(0, 1), _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "after:  Update 1")]
    public byte AfterUpdate1() { _current.Update([2], _changedOld.AsSpan(0, 1), _changedNew.AsSpan(0, 1), _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "before: Update 4")]
    public byte BeforeUpdate4() { _baseline.Update([0, 1, 2, 3], _changedOld, _changedNew, _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "after:  Update 4")]
    public byte AfterUpdate4() { _current.Update([0, 1, 2, 3], _changedOld, _changedNew, _parity); return _parity[0].Span[0]; }

    // The batch form has no baseline counterpart; compare it with the row above by eye.
    [Benchmark(Description = "after:  EncodeShards batch")]
    public byte AfterEncodeShardsBatch()
    {
        foreach (var p in _parity) p.Span.Clear();
        _current.EncodeShards(_indices, _data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Competitive context: this library against the managed field. Useful for reporting and for
/// choosing what to work on next, NOT the signal for whether a commit helped. Every row is
/// single-threaded. The two competitors are scalar Backblaze ports; ReedSolomon.NET keeps a full
/// 256x256 table, Witteborn's package does log/exp lookups per byte.
/// </summary>
public class CompetitiveBenchmarks
{
    private byte[][] _shards = null!;
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private bool[] _present = null!;
    private Current _ours = null!;
    private Egbakou _egbakou = null!;
    private WittebornCoder _witteborn = null!;

    [Params("5+2", "10+4", "8+8", "50+20")]
    public string Shape = "10+4";

    [Params(4_096, 65_536, 1_048_576)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        _shards = Geometry.Shards(data + parity, Shard_Size, 1);
        _data = _shards.Take(data).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = _shards.Skip(data).Select(s => (Memory<byte>)s).ToArray();
        _present = Enumerable.Repeat(true, data + parity).ToArray();
        for (int i = 0; i < parity; i++) _present[i] = false;
        _ours = new Current(data, parity);
        _egbakou = Egbakou.Create(data, parity);
        _witteborn = new WittebornCoder(data, parity);
        _ours.Encode(_shards);
    }

    [Benchmark(Baseline = true, Description = "ReedSolomon.NET Encode")]
    public byte EgbakouEncode()
    {
        _egbakou.EncodeParity(_shards, 0, Shard_Size);
        return _shards[^1][0];
    }

    [Benchmark(Description = "Witteborn Encode")]
    public byte WittebornEncode()
    {
        _witteborn.EncodeParity(_shards, 0, Shard_Size);
        return _shards[^1][0];
    }

    [Benchmark(Description = "ours Encode")]
    public byte OursEncode()
    {
        _ours.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "ReedSolomon.NET Reconstruct")]
    public byte EgbakouReconstruct()
    {
        _egbakou.DecodeMissing(_shards, _present, 0, Shard_Size);
        return _shards[0][0];
    }

    [Benchmark(Description = "Witteborn Reconstruct")]
    public byte WittebornReconstruct()
    {
        _witteborn.DecodeMissing(_shards, _present, 0, Shard_Size);
        return _shards[0][0];
    }

    [Benchmark(Description = "ours Reconstruct")]
    public byte OursReconstruct()
    {
        _ours.Reconstruct(_shards, _present);
        return _shards[0][0];
    }
}

/// <summary>
/// Sweep of the two engine constants that were chosen by reasoning, in one run: each job is a
/// separate process with REEDSOLOMONFAST_GROUP and REEDSOLOMONFAST_CHUNK set, so every
/// combination is measured under the same thermal state and against the same data.
/// </summary>
public class DispatchBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;

    [Params("10+4", "8+8", "50+20")]
    public string Shape = "10+4";

    [Params(65_536, 1_048_576)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        var shards = Geometry.Shards(data + parity, Shard_Size, 1);
        _data = shards.Take(data).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = shards.Skip(data).Select(s => (Memory<byte>)s).ToArray();
        _coder = new Current(data, parity);
    }

    [Benchmark]
    public byte Encode()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Batching small stripes. Coding is byte-position independent, so N independent stripes laid out
/// shard-major (each shard buffer holds the N stripes back to back) can be encoded as one call of
/// N times the stripe length. Same padded backing on both sides; only the number of calls differs.
/// </summary>
public class BatchBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;
    private int _dataShards;

    [Params("5+2", "10+4")]
    public string Shape = "10+4";

    [Params(4, 16, 64)]
    public int Batch;

    public const int StripeBytes = 4096;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        _dataShards = data;
        var buffers = Current.AllocateShards(data + parity, Batch * StripeBytes);
        var rng = new Random(1);
        foreach (var b in buffers) rng.NextBytes(b.Span);
        _data = buffers.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = buffers.Skip(data).ToArray();
        _coder = new Current(data, parity);
    }

    [Benchmark(Baseline = true, Description = "N calls of 4 KiB")]
    public byte Separate()
    {
        var data = new ReadOnlyMemory<byte>[_dataShards];
        var parity = new Memory<byte>[_parity.Length];
        for (int k = 0; k < Batch; k++)
        {
            int offset = k * StripeBytes;
            for (int i = 0; i < data.Length; i++) data[i] = _data[i].Slice(offset, StripeBytes);
            for (int i = 0; i < parity.Length; i++) parity[i] = _parity[i].Slice(offset, StripeBytes);
            _coder.Encode(data, parity);
        }

        return _parity[0].Span[0];
    }

    [Benchmark(Description = "one call of N x 4 KiB")]
    public byte Batched()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// The scalar tail: lengths one and 63 bytes past a vector multiple against the exact multiple.
/// Real shard sizes (a file split ten ways) are almost never multiples of 64.
/// </summary>
public class TailBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;

    [Params(65_536, 65_537, 65_599, 1_048_576, 1_048_577)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var shards = Geometry.Shards(14, Shard_Size, 1);
        _data = shards.Take(10).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = shards.Skip(10).Select(s => (Memory<byte>)s).ToArray();
        _coder = new Current(10, 4);
    }

    [Benchmark]
    public byte Encode()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Every kernel tier this CPU supports, in one run, so the tiers are compared under the same
/// thermal state. This is how the default tier order in Kernel.Best() gets justified per machine,
/// and how a kernel change is checked for not regressing the tiers it did not target.
/// </summary>
public class TierBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;

    [Params("5+2", "10+4", "50+20")]
    public string Shape = "10+4";

    [Params(4_096, 65_536, 1_048_576)]
    public int Shard_Size;

    // A string, not the enum: BenchmarkDotNet's generated project references both our assembly and
    // the baseline snapshot, and cannot tell the two KernelTier types apart.
    [ParamsSource(nameof(SupportedTiers))]
    public string Tier = "Scalar";

    public static IEnumerable<string> SupportedTiers => Enum.GetValues<KernelTier>().Where(Internal.Kernel.IsSupported).Select(t => t.ToString());

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        var shards = Geometry.Shards(data + parity, Shard_Size, 1);
        _data = shards.Take(data).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = shards.Skip(data).Select(s => (Memory<byte>)s).ToArray();
        _coder = new Current(data, parity, new CurrentOptions { Kernel = Enum.Parse<KernelTier>(Tier) });
    }

    [Benchmark]
    public byte Encode()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Our own API surfaces against each other: the three container shapes, incremental encode, and
/// the parallel option. Shows what each convenience costs relative to the contiguous stripe.
/// </summary>
public class ApiSurfaceBenchmarks
{
    private byte[][] _shards = null!;
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private byte[] _stripe = null!;
    private ReadOnlyMemory<byte>[] _alignedData = null!;
    private Memory<byte>[] _alignedParity = null!;
    private Current _serial = null!;
    private Current _parallel = null!;

    [Params("10+4")]
    public string Shape = "10+4";

    [Params(4_096, 65_536, 1_048_576, 8_388_608)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        _shards = Geometry.Shards(data + parity, Shard_Size, 1);
        _data = _shards.Take(data).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _parity = _shards.Skip(data).Select(s => (Memory<byte>)s).ToArray();
        _stripe = new byte[(data + parity) * Shard_Size];
        for (int i = 0; i < _shards.Length; i++) _shards[i].CopyTo(_stripe, i * Shard_Size);
        var aligned = Current.AllocateShards(data + parity, Shard_Size);
        for (int i = 0; i < _shards.Length; i++) _shards[i].CopyTo(aligned[i]);
        _alignedData = aligned.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _alignedParity = aligned.Skip(data).ToArray();
        _serial = new Current(data, parity);
        _parallel = new Current(data, parity, new CurrentOptions { MaxDegreeOfParallelism = -1, ParallelThresholdBytes = 65_536 });
    }

    [Benchmark(Baseline = true, Description = "stripe")]
    public byte Stripe()
    {
        int d = _serial.DataShards * Shard_Size;
        _serial.Encode(_stripe.AsSpan(0, d), _stripe.AsSpan(d), Shard_Size);
        return _stripe[d];
    }

    [Benchmark(Description = "memories")]
    public byte Memories()
    {
        _serial.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "arrays")]
    public byte Arrays()
    {
        _serial.Encode(_shards);
        return _shards[^1][0];
    }

    [Benchmark(Description = "AllocateShards (padded stride)")]
    public byte Aligned()
    {
        _serial.Encode(_alignedData, _alignedParity);
        return _alignedParity[0].Span[0];
    }

    [Benchmark(Description = "EncodeShard x data")]
    public byte Incremental()
    {
        foreach (var p in _parity) p.Span.Clear();
        for (int i = 0; i < _data.Length; i++) _serial.EncodeShard(i, _data[i].Span, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "memories, parallel")]
    public byte Parallel()
    {
        _parallel.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Where the fixed cost of a small call goes: the checked public call over one padded backing,
/// against the pin recursion alone (a no-op pinned operation) and the bare kernel on pointers
/// pinned once. The differences are validation plus dispatch, the pin, and the arithmetic.
/// </summary>
public unsafe class FixedCostBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private byte[] _backing = null!;
    private int[] _offsets = null!;
    private Current _coder = null!;
    private ReedSolomonFast.Internal.MulTables _tables = null!;
    private ReedSolomonFast.KernelTier _tier;

    [Params("10+4")]
    public string Shape = "10+4";

    [Params(4_096, 65_536)]
    public int Shard_Size;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        var shards = Geometry.Shards(data + parity, Shard_Size, 1);
        var padded = Current.AllocateShards(data + parity, Shard_Size);
        for (int i = 0; i < data; i++) shards[i].CopyTo(padded[i]);
        _data = padded.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = padded.Skip(data).ToArray();
        _offsets = new int[data + parity];
        for (int i = 0; i < data + parity; i++)
        {
            System.Runtime.InteropServices.MemoryMarshal.TryGetArray<byte>(padded[i], out var segment);
            _backing = segment.Array!;
            _offsets[i] = segment.Offset;
        }

        _coder = new Current(data, parity);
        _tier = _coder.Kernel;
        var matrix = ReedSolomonFast.Internal.CodingMatrix.Create(ReedSolomonFast.MatrixKind.Vandermonde, data, parity);
        _tables = new ReedSolomonFast.Internal.MulTables(matrix.Data.Slice(data * data), parity, data);
    }

    [Benchmark(Baseline = true, Description = "public Encode (checked, pinned per shard)")]
    public byte Public()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "kernel only (pinned once, no checks)")]
    public byte KernelOnly()
    {
        int data = _coder.DataShards, parity = _coder.ParityShards;
        byte** srcs = stackalloc byte*[data];
        byte** dsts = stackalloc byte*[parity];
        fixed (byte* p = _backing)
        {
            for (int i = 0; i < data; i++) srcs[i] = p + _offsets[i];
            for (int i = 0; i < parity; i++) dsts[i] = p + _offsets[data + i];
            ReedSolomonFast.Internal.Kernel.DotProduct(_tier, srcs, data, dsts, parity, _tables.Pointer, _tables.Pointer + _tables.GfniOffset, (nuint)Shard_Size);
            return dsts[0][0];
        }
    }

    [Benchmark(Description = "pin recursion only (no-op operation)")]
    public nint PinOnly()
    {
        byte** ptrs = stackalloc byte*[_data.Length + _parity.Length];
        var op = new NoOp();
        ReedSolomonFast.Internal.Pinner.Run(_data, default, _parity, ptrs, ref op);
        return (nint)ptrs[0];
    }

    private struct NoOp : ReedSolomonFast.Internal.IPinnedOperation
    {
        public void Execute(byte** ptrs) { }
    }
}

/// <summary>
/// Windowed encoding: a 16 MiB-per-shard payload (224 MiB at 10+4, far beyond L3) encoded as one
/// call, or as a sequence of Memory-slice windows of 64 KiB or 1 MiB over the same buffers. Same
/// bytes every row, so the times compare directly; the question is whether windows hold the
/// L2-resident rate on a DRAM-resident payload.
/// </summary>
public class WindowBenchmarks
{
    private const int PerShard = 16 << 20;
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;

    [Params("10+4")]
    public string Shape = "10+4";

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        var padded = Current.AllocateShards(data + parity, PerShard);
        var rng = new Random(3);
        for (int i = 0; i < data; i++) rng.NextBytes(padded[i].Span);
        _data = padded.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = padded.Skip(data).ToArray();
        _coder = new Current(data, parity);
    }

    [Benchmark(Baseline = true, Description = "one call, 16 MiB shards")]
    public byte Whole()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "256 calls, 64 KiB windows")]
    public byte Windows64K() => Windows(64 << 10);

    [Benchmark(Description = "16 calls, 1 MiB windows")]
    public byte Windows1M() => Windows(1 << 20);

    private byte Windows(int window)
    {
        var data = new ReadOnlyMemory<byte>[_data.Length];
        var parity = new Memory<byte>[_parity.Length];
        for (int offset = 0; offset < PerShard; offset += window)
        {
            for (int i = 0; i < data.Length; i++) data[i] = _data[i].Slice(offset, window);
            for (int i = 0; i < parity.Length; i++) parity[i] = _parity[i].Slice(offset, window);
            _coder.Encode(data, parity);
        }

        return _parity[0].Span[0];
    }
}

/// <summary>
/// Shard-base skew: consecutive shards at stride length + skew in one backing. Skew 0 puts every
/// shard in the same cache sets at power-of-two lengths; AllocateShards uses 256.
/// </summary>
public class SkewBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _coder = null!;

    [Params("10+4")]
    public string Shape = "10+4";

    [Params(65_536, 1_048_576)]
    public int Shard_Size;

    [Params(0, 64, 128, 256, 4096)]
    public int Skew;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        int stride = Shard_Size + Skew;
        var backing = new byte[(data + parity) * stride + 64];
        new Random(4).NextBytes(backing);
        _data = Enumerable.Range(0, data).Select(i => (ReadOnlyMemory<byte>)backing.AsMemory(i * stride, Shard_Size)).ToArray();
        _parity = Enumerable.Range(0, parity).Select(i => backing.AsMemory((data + i) * stride, Shard_Size)).ToArray();
        _coder = new Current(data, parity);
    }

    [Benchmark(Description = "Encode")]
    public byte Encode()
    {
        _coder.Encode(_data, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Incremental parity, per layout: one EncodeShard call per data shard (the per-parity in-place
/// loop, two sources, one output) against EncodeShards with one index per call (one source, all
/// parity as accumulating outputs), and Update with one and four changed shards. "arrays" is
/// separate power-of-two arrays, "padded" the AllocateShards layout.
/// </summary>
public class IncrementalBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private ReadOnlyMemory<byte>[] _changedOld = null!;
    private ReadOnlyMemory<byte>[] _changedNew = null!;
    private Current _coder = null!;

    [Params("8+8", "50+20")]
    public string Shape = "8+8";

    [Params(65_536, 1_048_576)]
    public int Shard_Size;

    [Params("arrays", "padded")]
    public string Layout = "arrays";

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        var shards = Geometry.Shards(data + parity, Shard_Size, 1);
        Memory<byte>[] buffers;
        if (Layout == "padded")
        {
            buffers = Current.AllocateShards(data + parity, Shard_Size);
            for (int i = 0; i < data + parity; i++) shards[i].CopyTo(buffers[i]);
        }
        else
        {
            buffers = shards.Select(s => (Memory<byte>)s).ToArray();
        }

        _data = buffers.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = buffers.Skip(data).ToArray();
        var changed = Geometry.Shards(8, Shard_Size, 2);
        _changedOld = changed.Take(4).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _changedNew = changed.Skip(4).Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        _coder = new Current(data, parity);
        _coder.Encode(_data, _parity);
    }

    [Benchmark(Baseline = true, Description = "EncodeShard x data (in place, per parity)")]
    public byte PerParity()
    {
        for (int i = 0; i < _data.Length; i++) _coder.EncodeShard(i, _data[i].Span, _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "EncodeShards([i]) x data (multi-output accumulate)")]
    public byte MultiOutput()
    {
        Span<int> index = stackalloc int[1];
        for (int i = 0; i < _data.Length; i++)
        {
            index[0] = i;
            _coder.EncodeShards(index, _data.AsSpan(i, 1), _parity);
        }

        return _parity[0].Span[0];
    }

    [Benchmark(Description = "Update, 1 changed")]
    public byte Update1()
    {
        _coder.Update([2], _changedOld.AsSpan(0, 1), _changedNew.AsSpan(0, 1), _parity);
        return _parity[0].Span[0];
    }

    [Benchmark(Description = "Update, 4 changed")]
    public byte Update4()
    {
        _coder.Update([1, 3, 5, 7], _changedOld, _changedNew, _parity);
        return _parity[0].Span[0];
    }
}

/// <summary>
/// Streaming stores, explained: does the crossover follow the stripe footprint (one reused
/// padded stripe of 7 to 32 MiB at three shapes), and does the loss with few outputs come from
/// write-combining pressure (more output streams worse) or from the missing cache benefit (more
/// outputs better)? Five inputs, cold ring, one to eight outputs.
/// </summary>
public class StreamingSweepBenchmarks
{
    private ReadOnlyMemory<byte>[] _data = null!;
    private Memory<byte>[] _parity = null!;
    private Current _plain = null!;
    private Current _streaming = null!;

    [Params("5+2", "10+4", "8+8")]
    public string Shape = "5+2";

    [Params(7, 10, 14, 18, 24, 32)]
    public int Stripe_MiB;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        // Not a power of two, rounded to 4 KiB.
        int len = (int)(((long)Stripe_MiB << 20) / (data + parity)) & ~4095;
        var padded = Current.AllocateShards(data + parity, len);
        var rng = new Random(6);
        for (int i = 0; i < data; i++) rng.NextBytes(padded[i].Span);
        _data = padded.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = padded.Skip(data).ToArray();
        _plain = new Current(data, parity);
        _streaming = new Current(data, parity, new CurrentOptions { StreamingStores = true });
    }

    [Benchmark(Baseline = true, Description = "plain")]
    public byte Plain() { _plain.Encode(_data, _parity); return _parity[0].Span[0]; }

    [Benchmark(Description = "streaming")]
    public byte Streaming() { _streaming.Encode(_data, _parity); return _parity[0].Span[0]; }
}

public class OutputStreamSweepBenchmarks
{
    private const int Len = 1 << 20;
    private ReadOnlyMemory<byte>[][] _data = null!;
    private Memory<byte>[][] _parity = null!;
    private int _ring;
    private Current _plain = null!;
    private Current _streaming = null!;

    [Params(1, 2, 4, 8)]
    public int Outputs;

    [GlobalSetup]
    public void Setup()
    {
        const int data = 5;
        int shards = data + Outputs;
        int ring = Math.Max(1, (int)((48L << 20) / ((long)shards * Len)));
        _data = new ReadOnlyMemory<byte>[ring][];
        _parity = new Memory<byte>[ring][];
        var rng = new Random(7);
        for (int r = 0; r < ring; r++)
        {
            var stripe = Current.AllocateShards(shards, Len);
            for (int i = 0; i < data; i++) rng.NextBytes(stripe[i].Span);
            _data[r] = stripe.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            _parity[r] = stripe.Skip(data).ToArray();
        }

        _plain = new Current(data, Outputs);
        _streaming = new Current(data, Outputs, new CurrentOptions { StreamingStores = true });
    }

    [Benchmark(Baseline = true, Description = "plain, cold")]
    public byte Plain() { int r = _ring = (_ring + 1) % _data.Length; _plain.Encode(_data[r], _parity[r]); return _parity[r][0].Span[0]; }

    [Benchmark(Description = "streaming, cold")]
    public byte Streaming() { int r = _ring = (_ring + 1) % _data.Length; _streaming.Encode(_data[r], _parity[r]); return _parity[r][0].Span[0]; }
}
