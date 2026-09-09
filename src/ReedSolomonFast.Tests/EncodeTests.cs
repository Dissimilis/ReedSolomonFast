namespace ReedSolomonFast.Tests;

public class EncodeTests
{
    public static IEnumerable<object[]> Geometries() =>
        from g in new[] { (1, 1), (2, 1), (3, 2), (5, 3), (10, 4), (8, 8), (17, 3), (200, 56) }
        from kind in new[] { MatrixKind.Vandermonde, MatrixKind.Cauchy }
        select new object[] { g.Item1, g.Item2, kind };

    [Theory]
    [MemberData(nameof(Geometries))]
    public void Encode_AllOverloads_MatchReference(int data, int parity, MatrixKind kind)
    {
        var rs = new ReedSolomon(data, parity, new ReedSolomonOptions { Matrix = kind });
        foreach (int len in new[] { 1, 63, 64, 65, 1000, 4097 })
        {
            var shards = TestData.RandomShards(data + parity, len, len);
            var expected = TestData.ReferenceParity(rs, shards);

            var arrays = TestData.Clone(shards);
            rs.Encode(arrays);
            for (int p = 0; p < parity; p++) Assert.Equal(expected[p], arrays[data + p]);

            var memories = TestData.Clone(shards);
            rs.Encode(TestData.AsReadOnly(memories, 0, data), TestData.AsMemory(memories, data, parity));
            for (int p = 0; p < parity; p++) Assert.Equal(expected[p], memories[data + p]);

            var stripe = TestData.Stripe(shards);
            rs.Encode(stripe.AsSpan(0, data * len), stripe.AsSpan(data * len), len);
            for (int p = 0; p < parity; p++) Assert.Equal(expected[p], stripe.AsSpan((data + p) * len, len).ToArray());
        }
    }

    [Fact]
    public void Encode_DoesNotTouchData()
    {
        var rs = new ReedSolomon(4, 2);
        var shards = TestData.RandomShards(6, 100, 1);
        var before = TestData.Clone(shards);
        rs.Encode(shards);
        for (int i = 0; i < 4; i++) Assert.Equal(before[i], shards[i]);
    }

    [Fact]
    public void Encode_ZeroLength_IsNoOp()
    {
        var rs = new ReedSolomon(3, 2);
        rs.Encode(new byte[5][] { [], [], [], [], [] });
        rs.Encode(ReadOnlySpan<byte>.Empty, Span<byte>.Empty, 0);
    }

    [Fact]
    public void Encode_UnalignedMemorySlices_Work()
    {
        var rs = new ReedSolomon(3, 2);
        var backing = new byte[5 * 1003 + 7];
        new Random(3).NextBytes(backing);
        var data = new ReadOnlyMemory<byte>[3];
        var parity = new Memory<byte>[2];
        for (int i = 0; i < 3; i++) data[i] = backing.AsMemory(7 + i * 1003, 1003);
        for (int i = 0; i < 2; i++) parity[i] = backing.AsMemory(7 + (3 + i) * 1003, 1003);
        rs.Encode(data, parity);

        var plain = data.Select(m => m.ToArray()).Concat(parity.Select(m => m.ToArray())).ToArray();
        var expected = TestData.ReferenceParity(rs, plain);
        Assert.Equal(expected[0], parity[0].ToArray());
        Assert.Equal(expected[1], parity[1].ToArray());
    }

    [Fact]
    public void Encode_NativeMemory_IsPinnedThroughMemoryManager()
    {
        var rs = new ReedSolomon(2, 1);
        using var a = new NativeMemory(500);
        using var b = new NativeMemory(500);
        using var p = new NativeMemory(500);
        new Random(9).NextBytes(a.Memory.Span);
        new Random(10).NextBytes(b.Memory.Span);
        rs.Encode(new ReadOnlyMemory<byte>[] { a.Memory, b.Memory }, new[] { p.Memory });

        var expected = TestData.ReferenceParity(rs, [a.Memory.ToArray(), b.Memory.ToArray(), new byte[500]]);
        Assert.Equal(expected[0], p.Memory.ToArray());
    }

    [Fact]
    public void EncodeShard_Incrementally_EqualsEncode()
    {
        var rs = new ReedSolomon(5, 3);
        var shards = TestData.RandomShards(8, 777, 5);
        var expected = TestData.ReferenceParity(rs, shards);

        var parity = new Memory<byte>[3];
        for (int p = 0; p < 3; p++) parity[p] = new byte[777];
        foreach (int d in new[] { 3, 0, 4, 1, 2 }) rs.EncodeShard(d, shards[d], parity);
        for (int p = 0; p < 3; p++) Assert.Equal(expected[p], parity[p].ToArray());

        var stripe = new byte[3 * 777];
        foreach (int d in new[] { 4, 3, 2, 1, 0 }) rs.EncodeShard(d, shards[d], stripe, 777);
        for (int p = 0; p < 3; p++) Assert.Equal(expected[p], stripe.AsSpan(p * 777, 777).ToArray());
    }

    [Fact]
    public void EncodeShards_Batches_EqualEncode()
    {
        var rs = new ReedSolomon(10, 4);
        var shards = TestData.RandomShards(14, 4097, 6);
        var expected = TestData.ReferenceParity(rs, shards);
        var data = TestData.AsReadOnly(shards, 0, 10);

        var parity = new Memory<byte>[4];
        for (int p = 0; p < 4; p++) parity[p] = new byte[4097];
        rs.EncodeShards(Enumerable.Range(0, 10).ToArray(), data, parity);
        for (int p = 0; p < 4; p++) Assert.Equal(expected[p], parity[p].ToArray());

        // Two batches in scrambled order, mixed with single-shard calls.
        for (int p = 0; p < 4; p++) parity[p].Span.Clear();
        rs.EncodeShards([7, 2, 9], [data[7], data[2], data[9]], parity);
        rs.EncodeShard(0, data[0].Span, parity);
        rs.EncodeShards([1, 3, 4, 5, 6, 8], [data[1], data[3], data[4], data[5], data[6], data[8]], parity);
        for (int p = 0; p < 4; p++) Assert.Equal(expected[p], parity[p].ToArray());

        Assert.Throws<ArgumentException>(() => rs.EncodeShards([1, 1], [data[1], data[1]], parity));
        Assert.Throws<ArgumentOutOfRangeException>(() => rs.EncodeShards([10], [data[0]], parity));
        Assert.Throws<ArgumentException>(() => rs.EncodeShards([0, 1], [data[0]], parity));
    }

    [Theory]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(65_537)]
    public void EncodeShards_ThirteenSources_AccumulatesAcrossInputGroups(int len)
    {
        // Thirteen sources are two input groups on every tier; the second group must add to what
        // the first wrote, and both must add to the parity that was already there.
        var rs = new ReedSolomon(13, 9);
        var data = TestData.RandomShards(13, len, len);
        var parity = TestData.RandomShards(9, len, len + 1);
        var initial = TestData.Clone(parity);
        var contribution = TestData.ReferenceParity(rs, data.Concat(Enumerable.Range(0, 9).Select(_ => new byte[len])).ToArray());

        int[] indices = [12, 0, 7, 1, 11, 2, 9, 3, 8, 4, 10, 5, 6];
        rs.EncodeShards(indices, indices.Select(i => (ReadOnlyMemory<byte>)data[i]).ToArray(), TestData.AsMemory(parity, 0, 9));
        for (int p = 0; p < 9; p++)
            for (int i = 0; i < len; i++)
                Assert.True((byte)(initial[p][i] ^ contribution[p][i]) == parity[p][i], $"len {len} parity {p} byte {i}");
    }

    [Theory]
    [InlineData(2, 961)]
    [InlineData(3, 65_537)]
    [InlineData(5, 262_144)]
    [InlineData(5, 262_145)]
    public void Parallel_AccumulatePaths_MatchSerial(int workers, int len)
    {
        // 13+10: ten outputs split by output block (4+4+2) up to 256 KiB, by byte range above it;
        // 13 inputs are two groups. Encode, a scrambled EncodeShards batch into nonzero parity,
        // and an Update must all agree with the serial coder byte for byte.
        var serial = new ReedSolomon(13, 10);
        var parallel = new ReedSolomon(13, 10, new ReedSolomonOptions { MaxDegreeOfParallelism = workers, ParallelThresholdBytes = 1 });
        var shards = TestData.RandomShards(23, len, workers * 1000 + 1);
        var a = TestData.Clone(shards);
        var b = TestData.Clone(shards);
        serial.Encode(a);
        parallel.Encode(b);
        TestData.AssertShardsEqual(a, b, "encode");

        var data = TestData.AsReadOnly(shards, 0, 13);
        int[] indices = [12, 0, 7, 1, 11, 2, 9, 3, 8, 4, 10, 5, 6];
        var batch = indices.Select(i => data[i]).ToArray();
        var pa = TestData.RandomShards(10, len, 3);
        var pb = TestData.Clone(pa);
        serial.EncodeShards(indices, batch, TestData.AsMemory(pa, 0, 10));
        parallel.EncodeShards(indices, batch, TestData.AsMemory(pb, 0, 10));
        TestData.AssertShardsEqual(pa, pb, "batch");

        var changed = TestData.RandomShards(2, len, 4);
        serial.Update([0, 7], [data[0], data[7]], [changed[0], changed[1]], TestData.AsMemory(pa, 0, 10));
        parallel.Update([0, 7], [data[0], data[7]], [changed[0], changed[1]], TestData.AsMemory(pb, 0, 10));
        TestData.AssertShardsEqual(pa, pb, "update");
    }

    [Fact]
    public void Update_ChangedShards_GivesFreshParity()
    {
        var rs = new ReedSolomon(6, 3);
        var shards = TestData.RandomShards(9, 300, 8);
        rs.Encode(shards);

        var changed = new byte[9][];
        for (int i = 0; i < 9; i++) changed[i] = shards[i].ToArray();
        new Random(11).NextBytes(changed[1]);
        new Random(12).NextBytes(changed[4]);

        rs.Update(
            [1, 4],
            [shards[1], shards[4]],
            [changed[1], changed[4]],
            TestData.AsMemory(changed, 6, 3));

        var expected = TestData.ReferenceParity(rs, changed);
        for (int p = 0; p < 3; p++) Assert.Equal(expected[p], changed[6 + p]);
    }

    [Fact]
    public void Encode_CustomParityRows_AreUsed()
    {
        byte[][] rows = [[1, 1, 1], [1, 2, 3]];
        var rs = new ReedSolomon(3, 2, new ReedSolomonOptions { CustomParityRows = rows });
        var shards = TestData.RandomShards(5, 50, 4);
        rs.Encode(shards);
        for (int b = 0; b < 50; b++)
        {
            Assert.Equal((byte)(shards[0][b] ^ shards[1][b] ^ shards[2][b]), shards[3][b]);
            Assert.Equal((byte)(shards[0][b] ^ Gf256Reference.Multiply(2, shards[1][b]) ^ Gf256Reference.Multiply(3, shards[2][b])), shards[4][b]);
        }
    }

    [Fact]
    public void Encode_EveryKernelTier_AgreesWithScalar()
    {
        var scalar = new ReedSolomon(10, 4, new ReedSolomonOptions { Kernel = KernelTier.Scalar });
        var shards = TestData.RandomShards(14, 10_007, 13);
        var expected = TestData.Clone(shards);
        scalar.Encode(expected);

        foreach (var tier in Enum.GetValues<KernelTier>())
        {
            if (!Internal.Kernel.IsSupported(tier)) continue;
            var rs = new ReedSolomon(10, 4, new ReedSolomonOptions { Kernel = tier });
            Assert.Equal(tier, rs.Kernel);
            var actual = TestData.Clone(shards);
            rs.Encode(actual);
            for (int p = 0; p < 4; p++) Assert.True(expected[10 + p].AsSpan().SequenceEqual(actual[10 + p]), $"tier {tier} parity {p}");
        }
    }

    [Fact]
    public void Encode_Parallel_MatchesSerial()
    {
        var serial = new ReedSolomon(10, 4);
        var parallel = new ReedSolomon(10, 4, new ReedSolomonOptions { MaxDegreeOfParallelism = 4, ParallelThresholdBytes = 4096 });
        var shards = TestData.RandomShards(14, 1 << 16, 21);
        var a = TestData.Clone(shards);
        var b = TestData.Clone(shards);
        serial.Encode(a);
        parallel.Encode(b);
        for (int p = 0; p < 4; p++) Assert.Equal(a[10 + p], b[10 + p]);

        var present = Enumerable.Repeat(true, 14).ToArray();
        present[0] = present[5] = present[11] = false;
        parallel.Reconstruct(b, present);
        for (int i = 0; i < 14; i++) Assert.Equal(a[i], b[i]);
    }

    [Fact]
    public void Encode_ParallelByOutputs_MatchesSerial()
    {
        // Eight or more outputs at L3-resident sizes are split by output block across workers.
        foreach (var (data, parity) in new[] { (8, 8), (50, 20), (3, 9) })
        {
            var serial = new ReedSolomon(data, parity);
            var parallel = new ReedSolomon(data, parity, new ReedSolomonOptions { MaxDegreeOfParallelism = 5, ParallelThresholdBytes = 1024 });
            foreach (int len in new[] { 1000, 65_537 })
            {
                var shards = TestData.RandomShards(data + parity, len, len + data);
                var a = TestData.Clone(shards);
                var b = TestData.Clone(shards);
                serial.Encode(a);
                parallel.Encode(b);
                for (int p = 0; p < parity; p++) Assert.True(a[data + p].AsSpan().SequenceEqual(b[data + p]), $"{data}+{parity} len {len} parity {p}");

                var present = Enumerable.Repeat(true, data + parity).ToArray();
                for (int i = 0; i < parity; i++) present[i * (data + parity) / parity] = false;
                var c = a.Select((s, i) => present[i] ? s.ToArray() : new byte[len]).ToArray();
                parallel.Reconstruct(c, present);
                for (int i = 0; i < data + parity; i++) Assert.True(a[i].AsSpan().SequenceEqual(c[i]), $"{data}+{parity} len {len} shard {i}");
            }
        }
    }

    [Fact]
    public void Encode_MemoryOverload_DoesNotAllocate()
    {
        var rs = new ReedSolomon(10, 4);
        var shards = TestData.RandomShards(14, 4096, 2);
        var data = TestData.AsReadOnly(shards, 0, 10);
        var parity = TestData.AsMemory(shards, 10, 4);
        rs.Encode(data, parity);
        rs.Encode(shards);

        long before = GC.GetAllocatedBytesForCurrentThread();
        rs.Encode(data, parity);
        rs.Encode(shards);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    /// <summary>A Memory over unmanaged memory, to prove non-array memories are pinned correctly.</summary>
    private sealed unsafe class NativeMemory : System.Buffers.MemoryManager<byte>
    {
        private readonly int _length;
        private byte* _ptr;

        public NativeMemory(int length)
        {
            _length = length;
            _ptr = (byte*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)length);
        }

        public override Span<byte> GetSpan() => new(_ptr, _length);
        public override System.Buffers.MemoryHandle Pin(int elementIndex = 0) => new(_ptr + elementIndex);
        public override void Unpin() { }

        protected override void Dispose(bool disposing)
        {
            if (_ptr is not null) System.Runtime.InteropServices.NativeMemory.Free(_ptr);
            _ptr = null;
        }
    }
}
