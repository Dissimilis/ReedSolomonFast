using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Attributes;
using Current = ReedSolomonFast.ReedSolomon;
using CurrentOptions = ReedSolomonFast.ReedSolomonOptions;

namespace ReedSolomonFast.Benchmarks;

/// <summary>
/// How Encode and worst-case Reconstruct scale with worker count, against two references that
/// move the same bytes without field arithmetic. Every case runs on the cold stripe ring, so
/// inputs come from DRAM as they do for a caller streaming data through.
///
/// The XOR reference walks the kernel's own traversal (64 KiB chunks, every input once per block
/// of four outputs, accumulators in registers) and replaces the multiply with nothing. It is the
/// tightest upper bound the coder can be held to at the same worker count and geometry. The copy
/// reference copies every data shard once, the plain memcpy rate of the payload. Both split by
/// byte range across exactly the requested workers, with no threshold gate, so the 64 KiB rows
/// show what the coder's gate is protecting it from.
///
/// Throughput is data bytes per second, matching the README tables; the run prints a scaling
/// table (each worker count over one worker) and the bandwidth fraction (coder over XOR
/// reference) after the BenchmarkDotNet table.
/// </summary>
public unsafe class ParallelScalingBenchmarks
{
    private const int ChunkBytes = 64 * 1024;
    private const int MaxInputGroup = 12;

    // One padded backing per stripe, with the offset of every shard inside it.
    private byte[][] _ringBacking = null!;
    private int[][] _ringOffsets = null!;
    private ReadOnlyMemory<byte>[][] _ringData = null!;
    private Memory<byte>[][] _ringParity = null!;
    private Memory<byte>[][] _ringAll = null!;
    private int _ringIndex;
    private int _data;
    private int _parity;
    private bool[] _present = null!;
    private Current _coder = null!;
    private byte[] _copyBacking = null!;
    private int[] _copyOffsets = null!;

    [Params("10+4", "8+8", "50+20")]
    public string Shape = "10+4";

    // 64 KiB stays in L3 for every shape. 1 MiB leaves it only for 50+20 (a 70 MiB stripe).
    // 16 MiB makes every stripe DRAM-resident on its own.
    [Params(65_536, 1_048_576, 16_777_216)]
    public int Shard_Size;

    [Params(1, 2, 4, 8, 16)]
    public int Workers;

    [GlobalSetup]
    public void Setup()
    {
        (_data, _parity) = Geometry.Parse(Shape);
        int shards = _data + _parity;
        _present = Enumerable.Repeat(true, shards).ToArray();
        for (int i = 0; i < _parity; i++) _present[i] = false;
        _coder = new Current(_data, _parity, new CurrentOptions { MaxDegreeOfParallelism = Workers, ParallelThresholdBytes = 65_536 });

        // Three times a 16 MiB L3 of padded stripes, one per call. Each stripe is encoded once so
        // Reconstruct has valid parity to rebuild from; the data is identical across stripes.
        var source = Geometry.Shards(_data, Shard_Size, 1);
        long stripeBytes = (long)shards * Shard_Size;
        int ring = (int)Math.Max(1, (48L << 20) / stripeBytes);
        _ringBacking = new byte[ring][];
        _ringOffsets = new int[ring][];
        _ringData = new ReadOnlyMemory<byte>[ring][];
        _ringParity = new Memory<byte>[ring][];
        _ringAll = new Memory<byte>[ring][];
        for (int r = 0; r < ring; r++)
        {
            var stripe = Current.AllocateShards(shards, Shard_Size);
            for (int i = 0; i < _data; i++) source[i].CopyTo(stripe[i]);
            (_ringBacking[r], _ringOffsets[r]) = Backing(stripe);
            _ringData[r] = stripe.Take(_data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
            _ringParity[r] = stripe.Skip(_data).ToArray();
            _ringAll[r] = stripe;
            _coder.Encode(_ringData[r], _ringParity[r]);
        }

        (_copyBacking, _copyOffsets) = Backing(Current.AllocateShards(_data, Shard_Size));
    }

    /// <summary>AllocateShards hands out windows over one array; recover it and the offsets so a worker can pin once.</summary>
    private static (byte[] Backing, int[] Offsets) Backing(Memory<byte>[] shards)
    {
        var offsets = new int[shards.Length];
        byte[]? backing = null;
        for (int i = 0; i < shards.Length; i++)
        {
            if (!MemoryMarshal.TryGetArray<byte>(shards[i], out var segment) || segment.Array is null)
                throw new InvalidOperationException("AllocateShards no longer returns array-backed memory; the references need one backing per stripe.");
            backing ??= segment.Array;
            if (!ReferenceEquals(backing, segment.Array))
                throw new InvalidOperationException("AllocateShards returned more than one backing array for a stripe.");
            offsets[i] = segment.Offset;
        }

        return (backing!, offsets);
    }

    private int Next() => _ringIndex = (_ringIndex + 1) % _ringData.Length;

    [Benchmark(Description = "Encode")]
    public byte Encode()
    {
        int r = Next();
        _coder.Encode(_ringData[r], _ringParity[r]);
        return _ringParity[r][0].Span[0];
    }

    [Benchmark(Description = "Reconstruct")]
    public byte Reconstruct()
    {
        int r = Next();
        _coder.Reconstruct(_ringAll[r], _present);
        return _ringAll[r][0].Span[0];
    }

    // Writes the XOR into the stripe's parity, so this stripe no longer verifies; each case is its
    // own process with its own ring, so no coder case sees it.
    [Benchmark(Description = "XOR reference")]
    public byte XorReference()
    {
        int r = Next();
        byte[] backing = _ringBacking[r];
        int[] offsets = _ringOffsets[r];
        int data = _data, parity = _parity;
        Split(Shard_Size, Workers, (start, length) =>
        {
            fixed (byte* p = backing) XorRange(p, offsets, data, parity, start, length);
        });
        return backing[offsets[data]];
    }

    [Benchmark(Description = "copy reference")]
    public byte CopyReference()
    {
        int r = Next();
        byte[] backing = _ringBacking[r];
        int[] offsets = _ringOffsets[r];
        byte[] scratch = _copyBacking;
        int[] scratchOffsets = _copyOffsets;
        int data = _data;
        Split(Shard_Size, Workers, (start, length) =>
        {
            for (int i = 0; i < data; i++)
                backing.AsSpan(offsets[i] + start, length).CopyTo(scratch.AsSpan(scratchOffsets[i] + start, length));
        });
        return scratch[scratchOffsets[0]];
    }

    /// <summary>
    /// Byte-range split over exactly <paramref name="workers"/> workers, chunks a multiple of 64
    /// like the coder's own split. One worker runs inline, as the coder does.
    /// </summary>
    private static void Split(int length, int workers, Action<int, int> body)
    {
        if (workers <= 1)
        {
            body(0, length);
            return;
        }

        int chunk = (((length + workers - 1) / workers) + 63) & ~63;
        Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, w =>
        {
            int start = w * chunk;
            if (start >= length) return;
            body(start, Math.Min(chunk, length - start));
        });
    }

    /// <summary>
    /// The kernel's traversal without the multiply: for each 64 KiB chunk, for each balanced group
    /// of at most twelve inputs, for each block of up to four outputs, stream the group's inputs
    /// once and XOR them into the block's accumulators. Groups after the first reload the block's
    /// partial sums, as the kernel does; without the grouping, 50 inputs would be live at once,
    /// which is the stream-count wall the kernel exists to avoid, and the reference would sit
    /// below the coder instead of above it.
    /// </summary>
    private static void XorRange(byte* backing, int[] offsets, int data, int parity, int start, int length)
    {
        int groups = (data + MaxInputGroup - 1) / MaxInputGroup;
        int groupSize = (data + groups - 1) / groups;
        byte** srcs = stackalloc byte*[data];
        for (int chunkStart = start; chunkStart < start + length; chunkStart += ChunkBytes)
        {
            int chunkLength = Math.Min(ChunkBytes, start + length - chunkStart);
            for (int s = 0; s < data; s++) srcs[s] = backing + offsets[s] + chunkStart;
            for (int first = 0; first < data; first += groupSize)
            {
                int last = Math.Min(first + groupSize, data);
                bool accumulate = first > 0;
                for (int block = 0; block < parity; block += 4)
                {
                    int outputs = Math.Min(4, parity - block);
                    byte* d0 = backing + offsets[data + block] + chunkStart;
                    byte* d1 = backing + offsets[data + Math.Min(block + 1, parity - 1)] + chunkStart;
                    byte* d2 = backing + offsets[data + Math.Min(block + 2, parity - 1)] + chunkStart;
                    byte* d3 = backing + offsets[data + Math.Min(block + 3, parity - 1)] + chunkStart;
                    int i = 0;
                    for (; i + 64 <= chunkLength; i += 64)
                    {
                        Vector512<byte> a0, a1, a2, a3;
                        if (accumulate)
                        {
                            a0 = Vector512.Load(d0 + i);
                            a1 = Vector512.Load(d1 + i);
                            a2 = Vector512.Load(d2 + i);
                            a3 = Vector512.Load(d3 + i);
                        }
                        else
                        {
                            a0 = a1 = a2 = a3 = Vector512<byte>.Zero;
                        }

                        for (int s = first; s < last; s++)
                        {
                            var v = Vector512.Load(srcs[s] + i);
                            a0 ^= v;
                            a1 ^= v;
                            a2 ^= v;
                            a3 ^= v;
                        }

                        Vector512.Store(a0, d0 + i);
                        if (outputs > 1) Vector512.Store(a1, d1 + i);
                        if (outputs > 2) Vector512.Store(a2, d2 + i);
                        if (outputs > 3) Vector512.Store(a3, d3 + i);
                    }

                    for (; i < chunkLength; i++)
                    {
                        byte acc = accumulate ? d0[i] : (byte)0;
                        for (int s = first; s < last; s++) acc ^= srcs[s][i];
                        d0[i] = acc;
                        if (outputs > 1) d1[i] = acc;
                        if (outputs > 2) d2[i] = acc;
                        if (outputs > 3) d3[i] = acc;
                    }
                }
            }
        }
    }
}
