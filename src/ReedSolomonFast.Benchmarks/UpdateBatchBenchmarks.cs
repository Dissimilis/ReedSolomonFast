extern alias Baseline;

using BenchmarkDotNet.Attributes;
using BaselineCoder = Baseline::ReedSolomonFast.ReedSolomon;

namespace ReedSolomonFast.Benchmarks;

/// <summary>Compares Update batching with the frozen implementation on identical caller buffers and includes unchanged Encode controls.</summary>
public class UpdateBatchBenchmarks
{
    [Params("10+4", "50+20")]
    public string Shape = "10+4";

    [Params(65_536, 1_048_576)]
    public int Shard_Size;

    [Params("arrays", "padded")]
    public string Layout = "arrays";

    [Params(1, 2, 3, 4, 5, 8)]
    public int Changes;

    private ReedSolomon _current = null!;
    private BaselineCoder _baseline = null!;
    private ReadOnlyMemory<byte>[] _data = null!;
    private ReadOnlyMemory<byte>[] _old = null!;
    private ReadOnlyMemory<byte>[] _new = null!;
    private Memory<byte>[] _parity = null!;
    private int[] _indices = null!;

    [GlobalSetup]
    public void Setup()
    {
        var (data, parity) = Geometry.Parse(Shape);
        _current = new ReedSolomon(data, parity);
        _baseline = new BaselineCoder(data, parity);
        Memory<byte>[] buffers = Layout == "padded"
            ? ReedSolomon.AllocateShards(data + parity + Changes, Shard_Size)
            : Enumerable.Range(0, data + parity + Changes).Select(_ => new Memory<byte>(new byte[Shard_Size])).ToArray();
        var random = new Random(143);
        foreach (var buffer in buffers) random.NextBytes(buffer.Span);
        _data = buffers.Take(data).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _parity = buffers.Skip(data).Take(parity).ToArray();
        _indices = Enumerable.Range(0, Changes).Select(i => data - 1 - i).ToArray();
        _old = _indices.Select(i => _data[i]).ToArray();
        _new = buffers.Skip(data + parity).Select(m => (ReadOnlyMemory<byte>)m).ToArray();
        _current.Encode(_data, _parity);
        byte[][] expected = _parity.Select(m => m.ToArray()).ToArray();
        _baseline.Update(_indices, _old, _new, _parity);
        _current.Update(_indices, _old, _new, _parity);
        for (int p = 0; p < parity; p++)
            if (!_parity[p].Span.SequenceEqual(expected[p])) throw new InvalidOperationException("Batched Update disagrees with baseline.");
    }

    [Benchmark(Baseline = true)]
    public byte BeforeUpdate() { _baseline.Update(_indices, _old, _new, _parity); return _parity[0].Span[0]; }

    [Benchmark]
    public byte AfterUpdate() { _current.Update(_indices, _old, _new, _parity); return _parity[0].Span[0]; }

    [Benchmark]
    public byte BeforeEncode() { _baseline.Encode(_data, _parity); return _parity[0].Span[0]; }

    [Benchmark]
    public byte AfterEncode() { _current.Encode(_data, _parity); return _parity[0].Span[0]; }
}
