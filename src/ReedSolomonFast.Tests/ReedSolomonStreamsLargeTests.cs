using System.Security.Cryptography;
using System.Runtime.InteropServices;

namespace ReedSolomonFast.Tests;

[CollectionDefinition("Large", DisableParallelization = true)]
public class LargeCollectionDefinition
{
}

[Collection("Large")]
[Trait("Category", "Large")]
public class ReedSolomonStreamsLargeTests
{
    [Fact]
    public async Task EncodeAsync_ReconstructAsync_BeyondTwoGiBPerShard_MatchesInMemoryWithBoundedMemory()
    {
        const int window = 64 * 1024;
        // Cross both Int32.MaxValue and 2 GiB, and exercise a final partial window.
        const long shardLength = 2L * 1024 * 1024 * 1024 + 5;
        var rs = new ReedSolomon(2, 1);
        var options = new ReedSolomonStreamingOptions
        {
            WindowSizeBytes = window,
            MaxConcurrentIoOperations = 1
        };

        // An odd period prevents every window from containing identical bytes.
        var file = new byte[2 * (window + 17) - 1];
        new Random(42).NextBytes(file);
        byte[][] pattern = rs.Split(file);
        rs.Encode(pattern);
        Assert.True(rs.Verify(pattern));
        byte[]?[] rebuilt = [null, pattern[1].ToArray(), pattern[2].ToArray()];
        rs.Reconstruct(rebuilt);
        Assert.Equal(pattern[0], rebuilt[0]);

        // Warm kernel caches and async paths before measuring call-owned memory.
        using (var warmData0 = new GeneratedStream(pattern[0], window))
        using (var warmData1 = new GeneratedStream(pattern[1], window))
        using (var warmParity = new ChecksummingNullStream(pattern[2], window))
        {
            await ReedSolomonStreams.EncodeAsync(rs, [warmData0, warmData1], [warmParity], window, options);
            warmParity.AssertComplete();
        }
        using (var warmData1 = new GeneratedStream(pattern[1], window))
        using (var warmParity = new GeneratedStream(pattern[2], window))
        using (var warmOutput = new ChecksummingNullStream(rebuilt[0]!, window))
        {
            await ReedSolomonStreams.ReconstructAsync(rs, [null, warmData1, warmParity],
                [warmOutput, null, null], window, options);
            warmOutput.AssertComplete();
        }

        var memory = new MemoryProbe(rs.TotalShards, window);
        using var data0 = new GeneratedStream(pattern[0], shardLength, memory);
        using var data1 = new GeneratedStream(pattern[1], shardLength, memory);
        using var parityOutput = new ChecksummingNullStream(pattern[2], shardLength, memory);
        using var remainingData = new GeneratedStream(pattern[1], shardLength, memory);
        using var parityInput = new GeneratedStream(pattern[2], shardLength, memory);
        using var reconstructed = new ChecksummingNullStream(rebuilt[0]!, shardLength, memory);

        memory.Start();
        await ReedSolomonStreams.EncodeAsync(rs, [data0, data1], [parityOutput], shardLength, options);
        memory.Sample(forceCollection: true);
        Assert.Equal(shardLength, data0.BytesRead);
        Assert.Equal(shardLength, data1.BytesRead);
        parityOutput.AssertComplete();
        memory.AssertBounded();

        // The encoded parity hash was checked against this source's exact byte sequence.
        memory.Start();
        await ReedSolomonStreams.ReconstructAsync(rs, [null, remainingData, parityInput],
            [reconstructed, null, null], shardLength, options);
        memory.Sample(forceCollection: true);
        Assert.Equal(shardLength, remainingData.BytesRead);
        Assert.Equal(shardLength, parityInput.BytesRead);
        reconstructed.AssertComplete();
        memory.AssertBounded();
    }

    private sealed class MemoryProbe
    {
        private readonly long stripeBound;
        private long baseline;
        private long peak;
        private int samples;
        private byte[]? stripe;

        public MemoryProbe(int shards, int window)
        {
            stripeBound = shards * (((long)window + 63) / 64 * 64 + 256) + 64;
        }

        public void Start()
        {
            stripe = null;
            baseline = GC.GetTotalMemory(forceFullCollection: true);
            peak = baseline;
            samples = 0;
        }

        public void Observe(ReadOnlyMemory<byte> buffer)
        {
            // AllocateShards supplies one backing array, which must survive every window unchanged.
            Assert.True(MemoryMarshal.TryGetArray(buffer, out ArraySegment<byte> segment));
            Assert.NotNull(segment.Array);
            Assert.True(segment.Array.LongLength <= stripeBound, "The stripe exceeds the documented byte bound.");
            if (stripe is null)
            {
                stripe = segment.Array;
            }
            else
            {
                Assert.Same(stripe, segment.Array);
            }
            Sample();
        }

        public void Sample(bool forceCollection = false)
        {
            // Only collected samples count: an uncollected one includes garbage from async
            // bookkeeping and from anything else in the process, which is not retained memory.
            if (!forceCollection && ++samples % 1024 != 0) return;
            peak = Math.Max(peak, GC.GetTotalMemory(forceFullCollection: true));
        }

        public void AssertBounded()
        {
            // The contract does not quantify O(N) bookkeeping or coder/runtime overhead.
            // A fixed allowance tolerates those costs without scaling with the 2 GiB shard length.
            const long bookkeepingAndRuntimeAllowance = 8L * 1024 * 1024;
            long allowed = stripeBound + bookkeepingAndRuntimeAllowance;
            Assert.True(peak - baseline <= allowed,
                $"Peak sampled managed growth {peak - baseline:N0} exceeded {allowed:N0} bytes " +
                $"(stripe {stripeBound:N0}, fixed overhead allowance {bookkeepingAndRuntimeAllowance:N0}).");
        }
    }

    private abstract class NonSeekableStream : Stream
    {
        public override bool CanSeek => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Flush() => throw new InvalidOperationException("Caller streams must not be flushed.");
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class GeneratedStream(byte[] pattern, long length, MemoryProbe? memory = null) : NonSeekableStream
    {
        public long BytesRead { get; private set; }
        public override bool CanRead => true;
        public override bool CanWrite => false;

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            memory?.Observe(buffer);
            int count = (int)Math.Min(buffer.Length, length - BytesRead);
            int copied = 0;
            while (copied < count)
            {
                int offset = (int)((BytesRead + copied) % pattern.Length);
                int take = Math.Min(count - copied, pattern.Length - offset);
                pattern.AsSpan(offset, take).CopyTo(buffer.Span.Slice(copied, take));
                copied += take;
            }
            BytesRead += count;
            return ValueTask.FromResult(count);
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }

    private sealed class ChecksummingNullStream(byte[] expectedPattern, long length, MemoryProbe? memory = null) : NonSeekableStream
    {
        private readonly IncrementalHash actual = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        private readonly IncrementalHash expected = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        private long bytesWritten;
        public override bool CanRead => false;
        public override bool CanWrite => true;

        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            memory?.Observe(buffer);
            Assert.True(buffer.Length <= length - bytesWritten, "The sink received bytes beyond shardLength.");
            actual.AppendData(buffer.Span);
            int consumed = 0;
            while (consumed < buffer.Length)
            {
                int offset = (int)((bytesWritten + consumed) % expectedPattern.Length);
                int take = Math.Min(buffer.Length - consumed, expectedPattern.Length - offset);
                expected.AppendData(expectedPattern.AsSpan(offset, take));
                consumed += take;
            }
            bytesWritten += buffer.Length;
            return ValueTask.CompletedTask;
        }

        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public void AssertComplete()
        {
            Assert.Equal(length, bytesWritten);
            Assert.Equal(expected.GetHashAndReset(), actual.GetHashAndReset());
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                actual.Dispose();
                expected.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
