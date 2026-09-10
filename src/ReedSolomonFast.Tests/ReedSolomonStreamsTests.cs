using System.Collections.Concurrent;

namespace ReedSolomonFast.Tests;

public class ReedSolomonStreamsTests
{
    private const int Window = 4096;
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    private static ReedSolomonStreamingOptions Options(int concurrency = 4) => new()
    {
        WindowSizeBytes = Window,
        MaxConcurrentIoOperations = concurrency
    };

    public static IEnumerable<object[]> EncodeCases() =>
        from length in new[] { 0, 1, 7, Window - 1, Window, Window + 1, 3 * Window + 5 }
        from kind in new[] { MatrixKind.Vandermonde, MatrixKind.Cauchy }
        from padded in new[] { false, true }
        select new object[] { length, kind, padded };

    [Theory]
    [MemberData(nameof(EncodeCases))]
    public async Task EncodeAsync_WindowBoundaries_MatchesSplitAndEncode(int shardLength, MatrixKind kind, bool padded)
    {
        var rs = new ReedSolomon(3, 2, new ReedSolomonOptions { Matrix = kind });
        // Scale file length so the requested cases exercise shard window boundaries too.
        int fileLength = shardLength * rs.DataShards - (padded && shardLength > 0 ? 2 : 0);
        byte[][] expected = Encoded(rs, fileLength);
        using var streams = new StreamSet(expected);
        await ReedSolomonStreams.EncodeAsync(rs, streams.Inputs[..3], streams.Outputs[3..], shardLength, Options());
        for (int i = 3; i < rs.TotalShards; i++)
        {
            Assert.Equal(expected[i], streams.Writers[i].Bytes);
        }
        streams.AssertCallerOwnership();
    }

    public static IEnumerable<object[]> Erasures() =>
        from missing in new[] { new[] { 0, 2 }, new[] { 3, 4 }, new[] { 1, 4 } }
        from subset in new[] { false, true }
        from kind in new[] { MatrixKind.Vandermonde, MatrixKind.Cauchy }
        select new object[] { missing, subset, kind };

    [Theory]
    [MemberData(nameof(Erasures))]
    public async Task ReconstructAsync_ErasuresAndRequestedSubset_MatchesInMemory(int[] missing, bool subset, MatrixKind kind)
    {
        var rs = new ReedSolomon(3, 2, new ReedSolomonOptions { Matrix = kind });
        byte[][] original = Encoded(rs, 3 * (3 * Window + 5) - 2);
        byte[]?[] expected = original.Select((s, i) => missing.Contains(i) ? null : s.ToArray()).ToArray();
        rs.Reconstruct(expected);
        using var streams = new StreamSet(original);
        Stream?[] inputs = streams.Inputs.Select((s, i) => missing.Contains(i) ? null : s).ToArray();
        int[] requested = subset ? missing[..1] : missing;
        Stream?[] outputs = streams.Outputs.Select((s, i) => requested.Contains(i) ? s : null).ToArray();

        await ReedSolomonStreams.ReconstructAsync(rs, inputs, outputs, original[0].Length, Options());

        for (int i = 0; i < rs.TotalShards; i++)
        {
            if (requested.Contains(i))
            {
                Assert.Equal(expected[i], streams.Writers[i].Bytes);
            }
            else
            {
                Assert.Null(outputs[i]);
                Assert.Empty(streams.Writers[i].Calls);
                Assert.Empty(streams.Writers[i].Bytes);
            }
        }
        streams.AssertCallerOwnership();
    }

    [Fact]
    public async Task VerifyAsync_IntactStreams_MatchesInMemory()
    {
        var rs = new ReedSolomon(3, 2);
        byte[][] expected = Encoded(rs, 3 * (3 * Window + 5) - 1);
        using var streams = new StreamSet(expected);
        Assert.Equal(rs.Verify(expected), await ReedSolomonStreams.VerifyAsync(rs, streams.Inputs, expected[0].Length, Options()));
        Assert.All(streams.Readers, s => Assert.Equal(expected[0].Length, s.BytesRead));
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, Window + 3)]
    [InlineData(0, 3 * Window + 4)]
    [InlineData(4, 0)]
    [InlineData(4, Window + 3)]
    [InlineData(4, 3 * Window + 4)]
    public async Task VerifyAsync_Corruption_StopsAfterMismatchingWindow(int shard, int offset)
    {
        var rs = new ReedSolomon(3, 2);
        const int length = 3 * Window + 5;
        byte[][] corrupt = Encoded(rs, 3 * length);
        corrupt[shard][offset] ^= 1;
        using var streams = new StreamSet(corrupt);
        Assert.False(rs.Verify(corrupt));
        Assert.Equal(rs.Verify(corrupt), await ReedSolomonStreams.VerifyAsync(rs, streams.Inputs, length, Options()));
        long consumed = Math.Min(length, (offset / Window + 1) * Window);
        Assert.All(streams.Readers, s => Assert.Equal(consumed, s.Position));
        streams.AssertCallerOwnership();
    }

    [Fact]
    public async Task ReconstructAsync_InvalidPresence_ThrowsBeforeIo()
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        Stream?[] inputs = [null, streams.Inputs[1], null, null, streams.Inputs[4]];
        Stream?[] outputs = [streams.Outputs[0], null, null, null, null];
        var ex = await Assert.ThrowsAsync<InsufficientShardsException>(() =>
            ReedSolomonStreams.ReconstructAsync(rs, inputs, outputs, 10, Options()));
        Assert.Equal(2, ex.Present);
        Assert.Equal(3, ex.Required);
        streams.AssertNoIo();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            ReedSolomonStreams.ReconstructAsync(rs, streams.Inputs, outputs, 10, Options()));
        streams.AssertNoIo();
    }

    [Fact]
    public async Task ReconstructAsync_NoOutputs_ValidatesAndPerformsNoIo()
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        Stream?[] inputs = [streams.Inputs[0], null, null, null, null];
        await ReedSolomonStreams.ReconstructAsync(rs, inputs, new Stream?[5], 10, Options());
        await ReedSolomonStreams.ReconstructAsync(rs, new Stream?[5], new Stream?[5], 10, Options());
        await Assert.ThrowsAsync<ArgumentException>(() =>
            ReedSolomonStreams.ReconstructAsync(rs, inputs, new Stream?[4], 10, Options()));
        streams.AssertNoIo();
    }

    [Fact]
    public async Task ReconstructAsync_NoOutputs_StillRejectsInvalidCapabilitiesAndDuplicates()
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        streams.Readers[0].Readable = false;
        await Assert.ThrowsAsync<ArgumentException>(() =>
            ReedSolomonStreams.ReconstructAsync(rs, streams.Inputs, new Stream?[5], 10, Options()));
        streams.AssertNoIo();
        streams.Readers[0].Readable = true;
        streams.Inputs[1] = streams.Inputs[0];
        await Assert.ThrowsAsync<ArgumentException>(() =>
            ReedSolomonStreams.ReconstructAsync(rs, streams.Inputs, new Stream?[5], 10, Options()));
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_NonSeekableOneByteReads_MatchesInMemory(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        byte[][] expected = Encoded(rs, 3 * (Window + 7) - 2);
        using var streams = new StreamSet(expected, oneByteReads: true);
        await Run(operation, rs, streams, expected[0].Length);
        AssertResult(operation, expected, streams);
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_ShortInput_ThrowsWithShardIndex(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        byte[][] expected = Encoded(rs, 3 * (Window + 7));
        // Shard 1 participates even when shard 0 is being reconstructed.
        expected[1] = expected[1][..^1];
        using var streams = new StreamSet(expected);
        var ex = await Assert.ThrowsAsync<EndOfStreamException>(() => Run(operation, rs, streams, Window + 7));
        Assert.Matches(@"\b1\b", ex.Message);
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode", 0)]
    [InlineData("Reconstruct", 0)]
    [InlineData("Verify", 0)]
    [InlineData("Encode", 11)]
    [InlineData("Reconstruct", 11)]
    [InlineData("Verify", 11)]
    public async Task StreamingAsync_CurrentPositionsAndUnequalSuffixes_OnlyTouchesRequestedRegion(string operation, int prefix)
    {
        var rs = new ReedSolomon(3, 2);
        const int length = Window + 7;
        byte[][] expected = Encoded(rs, 3 * length - 2);
        using var streams = new StreamSet(expected, prefixLength: prefix, suffixLength: 17);
        await Run(operation, rs, streams, length);
        foreach (int i in ParticipatingInputs(operation))
        {
            Assert.Equal(prefix + length, streams.Readers[i].Position);
            Assert.Equal(17 + i, streams.Readers[i].Bytes.Length - streams.Readers[i].Position);
        }
        foreach (int i in RequestedOutputs(operation))
        {
            byte[] bytes = streams.Writers[i].Bytes;
            Assert.Equal(expected[i], bytes.AsSpan(prefix, length).ToArray());
            Assert.All(bytes[..prefix], b => Assert.Equal((byte)0xA5, b));
            Assert.All(bytes[(prefix + length)..], b => Assert.Equal((byte)0xA5, b));
            Assert.Equal(prefix + length, streams.Writers[i].Position);
            Assert.Equal(prefix + length + 17 + i, bytes.Length);
        }
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_ZeroLength_PerformsNoIo(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        await Run(operation, rs, streams, 0);
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_DuplicateOrUnreadableInput_ThrowsBeforeIo(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        Stream saved = streams.Inputs[2];
        streams.Inputs[2] = streams.Inputs[1];
        await Assert.ThrowsAsync<ArgumentException>(() => Run(operation, rs, streams, 10));
        streams.AssertNoIo();
        streams.Inputs[2] = saved;
        streams.Readers[1].Readable = false;
        await Assert.ThrowsAsync<ArgumentException>(() => Run(operation, rs, streams, 10));
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    public async Task StreamingAsync_DuplicateOutputsOrInputOutputAliasOrUnwritableOutput_ThrowsBeforeIo(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        int[] indices = RequestedOutputs(operation);
        Stream saved = streams.Outputs[indices[1]];
        streams.Outputs[indices[1]] = streams.Outputs[indices[0]];
        await Assert.ThrowsAsync<ArgumentException>(() => Run(operation, rs, streams, 10));
        streams.AssertNoIo();
        streams.Outputs[indices[1]] = streams.Inputs[1];
        await Assert.ThrowsAsync<ArgumentException>(() => Run(operation, rs, streams, 10));
        streams.AssertNoIo();
        streams.Outputs[indices[1]] = saved;
        streams.Writers[indices[1]].Writable = false;
        await Assert.ThrowsAsync<ArgumentException>(() => Run(operation, rs, streams, 10));
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_PreCanceledToken_PerformsNoIo(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Run(operation, rs, streams, 10, cancellation.Token));
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_CanceledDuringRead_AwaitsOutstandingIo(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        using var cancellation = new CancellationTokenSource();
        using var blocked = new TokenObservingStream();
        streams.Inputs[1] = blocked;
        Task call = Run(operation, rs, streams, 10, cancellation.Token);
        try
        {
            await blocked.Started.Task.WaitAsync(Timeout);
            cancellation.Cancel();
            await blocked.Canceled.Task.WaitAsync(Timeout);
            Assert.False(call.IsCompleted);
        }
        finally
        {
            cancellation.Cancel();
            blocked.Release.TrySetResult(true);
            var thrown = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call.WaitAsync(Timeout));
            Assert.Equal(cancellation.Token, thrown.CancellationToken);   // the caller's token, not the adapter's linked one
        }
        blocked.MarkReturned();
        streams.MarkReturned();
        Assert.True(blocked.Finished);
        await Task.Yield();
        Assert.Empty(blocked.CallsAfterReturn);
        streams.AssertCallerOwnership();
        blocked.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    public async Task StreamingAsync_CanceledAtFirstWrite_ObservesTokenBeforeAcceptingBytes(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 3 * (Window + 1)));
        using var cancellation = new CancellationTokenSource();
        int firstOutput = RequestedOutputs(operation)[0];
        // The first WriteAsync entry is the public observation point after the inline kernel.
        streams.Writers[firstOutput].BeforeWrite = token =>
        {
            Assert.True(token.CanBeCanceled);
            cancellation.Cancel();
            Assert.True(token.IsCancellationRequested);
            token.ThrowIfCancellationRequested();
        };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            Run(operation, rs, streams, Window + 1, cancellation.Token, concurrency: 1));
        Assert.All(streams.Writers, s => Assert.Equal(0, s.BytesWritten));
        Assert.All(ParticipatingInputs(operation), i => Assert.Equal(Window, streams.Readers[i].BytesRead));
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_InputThrowsMidWindow_CancelsAndAwaitsSiblings(string operation)
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 3 * (Window + 1)));
        using var sibling = new TokenObservingStream();
        var failure = new IOException("The input failed mid-window.");
        using var throwing = new ThrowingStream(sibling.Started.Task, failure);
        streams.Inputs[1] = throwing;
        streams.Inputs[2] = sibling;
        Task call = Run(operation, rs, streams, Window + 1);
        try
        {
            await sibling.Canceled.Task.WaitAsync(Timeout);
            Assert.False(call.IsCompleted);
        }
        finally
        {
            sibling.Release.TrySetResult(true);
            var actual = await Assert.ThrowsAsync<IOException>(() => call.WaitAsync(Timeout));
            Assert.Same(failure, actual);
        }
        streams.MarkReturned();
        sibling.MarkReturned();
        throwing.MarkReturned();
        Assert.True(sibling.Finished);
        Assert.Equal(1, throwing.BytesRead);
        // Yield after releasing controlled work so late continuations can leave evidence.
        await Task.Yield();
        await Task.Yield();
        Assert.Empty(sibling.CallsAfterReturn);
        Assert.Empty(throwing.CallsAfterReturn);
        streams.AssertCallerOwnership();
        sibling.AssertCallerOwnership();
        throwing.AssertCallerOwnership();
    }

    [Theory]
    [InlineData("Encode", 1)]
    [InlineData("Encode", 5)]
    [InlineData("Reconstruct", 1)]
    [InlineData("Reconstruct", 5)]
    [InlineData("Verify", 1)]
    [InlineData("Verify", 5)]
    public async Task StreamingAsync_IoConcurrencyLimit_IsHonoredAndOutputUnchanged(string operation, int limit)
    {
        var rs = new ReedSolomon(3, 2);
        byte[][] expected = Encoded(rs, 3 * (Window + 7));
        using var streams = new StreamSet(expected);
        int firstBatch = Math.Min(limit, ParticipatingInputs(operation).Length);
        var counter = new IoCounter(firstBatch);
        foreach (RecordingStream stream in streams.All)
        {
            stream.Counter = counter;
        }
        Task call = Run(operation, rs, streams, Window + 7, concurrency: limit);
        try
        {
            await counter.FirstBatchStarted.Task.WaitAsync(Timeout);
        }
        finally
        {
            counter.Release.TrySetResult(true);
            await call.WaitAsync(Timeout);
        }
        Assert.InRange(counter.Peak, firstBatch, limit);
        Assert.Equal(0, counter.Active);
        Assert.False(counter.OverlappedReadAndWrite);
        Assert.All(streams.All, s => Assert.InRange(s.PeakOperations, 0, 1));
        AssertResult(operation, expected, streams);
        streams.AssertCallerOwnership();
    }

    private static byte[][] Encoded(ReedSolomon rs, int fileLength)
    {
        var file = new byte[fileLength];
        new Random(fileLength + 17).NextBytes(file);
        byte[][] shards = rs.Split(file);
        rs.Encode(shards);
        return shards;
    }

    private static int[] ParticipatingInputs(string operation) => operation switch
    {
        "Encode" => [0, 1, 2],
        "Reconstruct" => [1, 2, 4],
        _ => [0, 1, 2, 3, 4]
    };

    private static int[] RequestedOutputs(string operation) => operation switch
    {
        "Encode" => [3, 4],
        "Reconstruct" => [0, 3],
        _ => []
    };

    private static async Task Run(string operation, ReedSolomon rs, StreamSet streams, long length,
        CancellationToken token = default, int concurrency = 4)
    {
        switch (operation)
        {
            case "Encode":
                await ReedSolomonStreams.EncodeAsync(rs, streams.Inputs[..3], streams.Outputs[3..], length, Options(concurrency), token);
                break;
            case "Reconstruct":
                Stream?[] inputs = [null, streams.Inputs[1], streams.Inputs[2], null, streams.Inputs[4]];
                Stream?[] outputs = [streams.Outputs[0], null, null, streams.Outputs[3], null];
                await ReedSolomonStreams.ReconstructAsync(rs, inputs, outputs, length, Options(concurrency), token);
                break;
            case "Verify":
                Assert.True(await ReedSolomonStreams.VerifyAsync(rs, streams.Inputs, length, Options(concurrency), token));
                break;
            default:
                throw new ArgumentException("Unknown operation.", nameof(operation));
        }
    }

    private static void AssertResult(string operation, byte[][] expected, StreamSet streams)
    {
        foreach (int index in RequestedOutputs(operation))
        {
            Assert.Equal(expected[index], streams.Writers[index].Bytes);
        }
    }

    [Fact]
    public async Task ReconstructAsync_SurplusInput_IsNeverRead()
    {
        // Four of five shards are present but three suffice; the fourth would fail if touched,
        // and the rebuild must not depend on it.
        var rs = new ReedSolomon(3, 2);
        byte[][] original = Encoded(rs, 3 * (2 * Window + 3));
        using var streams = new StreamSet(original);
        using var poison = new PoisonStream();
        Stream?[] inputs = [streams.Inputs[0], streams.Inputs[1], streams.Inputs[2], poison, null];
        Stream?[] outputs = [null, null, null, null, streams.Outputs[4]];

        await ReedSolomonStreams.ReconstructAsync(rs, inputs, outputs, original[0].Length, Options());

        Assert.Equal(original[4], streams.Writers[4].Bytes);
        Assert.DoesNotContain("ReadAsync", poison.Calls);
        streams.AssertCallerOwnership();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EncodeAsync_WriterThrowsSynchronously_BlockedSiblingIsCancelledAndAwaited(bool callbackThrows)
    {
        // A WriteAsync that throws before returning a task must still cancel the write already in
        // flight and wait for it, and a cancellation callback that throws must not hide the failure.
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 3 * (Window + 1)));
        using var blocked = new BlockingWriteStream(callbackThrows);
        var failure = new IOException("The writer failed synchronously.");
        using var throwing = new SyncThrowingWriteStream(failure);
        Task call = ReedSolomonStreams.EncodeAsync(rs, streams.Inputs[..3], [blocked, throwing], Window + 1, Options());

        await blocked.Canceled.Task.WaitAsync(Timeout);
        Assert.False(call.IsCompleted);
        blocked.Release.TrySetResult(true);
        var actual = await Assert.ThrowsAsync<IOException>(() => call.WaitAsync(Timeout));
        Assert.Same(failure, actual);
        Assert.True(blocked.Finished);
        streams.AssertCallerOwnership();
    }

    [Fact]
    public async Task StreamingAsync_BadArguments_ThrowBeforeAnyIo()
    {
        var rs = new ReedSolomon(3, 2);
        using var streams = new StreamSet(Encoded(rs, 30));
        Stream[] data = streams.Inputs[..3], parity = streams.Outputs[3..];
        Stream?[] inputs = [.. streams.Inputs], outputs = new Stream?[5];
        var zeroWindow = new ReedSolomonStreamingOptions { WindowSizeBytes = 0 };
        var zeroIo = new ReedSolomonStreamingOptions { MaxConcurrentIoOperations = 0 };

        await Assert.ThrowsAsync<ArgumentNullException>(() => ReedSolomonStreams.EncodeAsync(null!, data, parity, 10));
        await Assert.ThrowsAsync<ArgumentNullException>(() => ReedSolomonStreams.EncodeAsync(rs, null!, parity, 10));
        await Assert.ThrowsAsync<ArgumentNullException>(() => ReedSolomonStreams.ReconstructAsync(rs, inputs, null!, 10));
        await Assert.ThrowsAsync<ArgumentNullException>(() => ReedSolomonStreams.VerifyAsync(rs, null!, 10));
        await Assert.ThrowsAsync<ArgumentException>(() => ReedSolomonStreams.EncodeAsync(rs, data[..2], parity, 10));
        await Assert.ThrowsAsync<ArgumentException>(() => ReedSolomonStreams.EncodeAsync(rs, data, parity[..1], 10));
        await Assert.ThrowsAsync<ArgumentException>(() => ReedSolomonStreams.ReconstructAsync(rs, inputs[..4], outputs, 10));
        await Assert.ThrowsAsync<ArgumentException>(() => ReedSolomonStreams.VerifyAsync(rs, streams.Inputs[..4], 10));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => ReedSolomonStreams.EncodeAsync(rs, data, parity, -1));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => ReedSolomonStreams.ReconstructAsync(rs, inputs, outputs, -1));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => ReedSolomonStreams.VerifyAsync(rs, streams.Inputs, -1));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => ReedSolomonStreams.EncodeAsync(rs, data, parity, 10, zeroWindow));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => ReedSolomonStreams.VerifyAsync(rs, streams.Inputs, 10, zeroIo));
        var wrongName = await Assert.ThrowsAsync<ArgumentException>(() => ReedSolomonStreams.EncodeAsync(rs, data, [parity[0], parity[0]], 10));
        Assert.Equal("parity", wrongName.ParamName);   // named after the caller's parameter, not the internal array
        streams.AssertNoIo();
    }

    [Theory]
    [InlineData("Encode")]
    [InlineData("Reconstruct")]
    [InlineData("Verify")]
    public async Task StreamingAsync_ReaderThrowsSynchronouslyInLaterWindow_PropagatesAndStops(string operation)
    {
        // A synchronous throw from ReadAsync in the third window: earlier windows are complete,
        // the exception keeps its type, and nothing is written for the failed window.
        var rs = new ReedSolomon(3, 2);
        byte[][] expected = Encoded(rs, 3 * (4 * Window));
        using var streams = new StreamSet(expected);
        var failure = new IOException("The input failed in window three.");
        using var throwing = new SyncThrowingReadStream(expected[1], failAtCall: 3, failure);
        streams.Inputs[1] = throwing;

        var actual = await Assert.ThrowsAsync<IOException>(() => Run(operation, rs, streams, 4 * Window));

        Assert.Same(failure, actual);
        Assert.Equal(2 * Window, throwing.BytesRead);
        foreach (int index in RequestedOutputs(operation))
        {
            Assert.Equal(2 * Window, streams.Writers[index].BytesWritten);
            Assert.Equal(expected[index][..(2 * Window)], streams.Writers[index].Bytes);
        }
        streams.AssertCallerOwnership();
    }

    private sealed class SyncThrowingReadStream(byte[] bytes, int failAtCall, IOException failure) : RecordingStream(bytes)
    {
        private int calls;

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (Interlocked.Increment(ref calls) == failAtCall) { Record("ReadAsync"); throw failure; }
            return base.ReadAsync(buffer, cancellationToken);
        }
    }

    private sealed class PoisonStream : RecordingStream
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("ReadAsync");
            throw new InvalidOperationException("A surplus input was read.");
        }
    }

    private sealed class BlockingWriteStream(bool callbackThrows) : RecordingStream
    {
        public TaskCompletionSource<bool> Canceled { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource<bool> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Finished { get; private set; }

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("WriteAsync");
            using var registration = cancellationToken.Register(() =>
            {
                Canceled.TrySetResult(true);
                if (callbackThrows) throw new InvalidOperationException("The cancellation callback failed.");
            });
            try
            {
                await Release.Task;
                cancellationToken.ThrowIfCancellationRequested();
            }
            finally
            {
                Finished = true;
            }
        }
    }

    private sealed class SyncThrowingWriteStream(IOException failure) : RecordingStream
    {
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("WriteAsync");
            throw failure;
        }
    }

    private sealed class StreamSet : IDisposable
    {
        public RecordingStream[] Readers { get; }
        public RecordingStream[] Writers { get; }
        public Stream[] Inputs { get; }
        public Stream[] Outputs { get; }
        public IEnumerable<RecordingStream> All => Readers.Concat(Writers);

        public StreamSet(byte[][] shards, bool oneByteReads = false, int prefixLength = 0, int suffixLength = 0)
        {
            Readers = shards.Select((bytes, i) =>
            {
                if (oneByteReads)
                {
                    return (RecordingStream)new OneByteReadStream(bytes);
                }
                int suffix = suffixLength == 0 ? 0 : suffixLength + i;
                byte[] storage = Enumerable.Repeat((byte)0xA5, prefixLength + bytes.Length + suffix).ToArray();
                bytes.CopyTo(storage, prefixLength);
                return new RecordingStream(storage, prefixLength);
            }).ToArray();
            Writers = shards.Select((bytes, i) =>
            {
                if (prefixLength == 0 && suffixLength == 0)
                {
                    return new RecordingStream();
                }
                return new RecordingStream(Enumerable.Repeat((byte)0xA5,
                    prefixLength + bytes.Length + suffixLength + i).ToArray(), prefixLength);
            }).ToArray();
            Inputs = Readers.Cast<Stream>().ToArray();
            Outputs = Writers.Cast<Stream>().ToArray();
        }

        public void AssertNoIo()
        {
            Assert.All(All, s => Assert.Empty(s.Calls));
        }

        public void AssertCallerOwnership()
        {
            foreach (RecordingStream stream in All)
            {
                stream.AssertCallerOwnership();
                Assert.Empty(stream.CallsAfterReturn);
            }
        }

        public void MarkReturned()
        {
            foreach (RecordingStream stream in All)
            {
                stream.MarkReturned();
            }
        }

        public void Dispose()
        {
            foreach (RecordingStream stream in All)
            {
                stream.Dispose();
            }
        }
    }

    private class RecordingStream : Stream
    {
        private readonly MemoryStream inner;
        private int returned;
        private int active;
        public ConcurrentQueue<string> Calls { get; } = new();
        public ConcurrentQueue<string> CallsAfterReturn { get; } = new();
        public bool Readable { get; set; } = true;
        public bool Writable { get; set; } = true;
        public long BytesRead { get; protected set; }
        public long BytesWritten { get; private set; }
        public int PeakOperations { get; private set; }
        public byte[] Bytes => inner.ToArray();
        public IoCounter? Counter { get; set; }
        public Action<CancellationToken>? BeforeWrite { get; set; }

        public RecordingStream(byte[]? bytes = null, int position = 0)
        {
            inner = new MemoryStream();
            if (bytes is not null)
            {
                inner.Write(bytes);
            }
            inner.Position = position;
        }

        protected void Record(string call)
        {
            Calls.Enqueue(call);
            if (Volatile.Read(ref returned) != 0)
            {
                CallsAfterReturn.Enqueue(call);
            }
        }

        public void MarkReturned() => Volatile.Write(ref returned, 1);

        public void AssertCallerOwnership()
        {
            Assert.DoesNotContain("Flush", Calls);
            Assert.DoesNotContain("Close", Calls);
            Assert.DoesNotContain("Dispose", Calls);
            Assert.DoesNotContain("Length", Calls);
            Assert.DoesNotContain("Seek", Calls);
            Assert.DoesNotContain("SetLength", Calls);
        }

        public override bool CanRead => Readable;
        public override bool CanWrite => Writable;
        public override bool CanSeek => true;
        public override long Length
        {
            get
            {
                Record("Length");
                throw new NotSupportedException();
            }
        }
        public override long Position
        {
            get => inner.Position;
            set => throw new NotSupportedException();
        }

        protected virtual int ReadCore(Span<byte> buffer) => inner.Read(buffer);

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("ReadAsync");
            int count = Interlocked.Increment(ref active);
            PeakOperations = Math.Max(PeakOperations, count);
            Counter?.Enter(writing: false);
            try
            {
                if (Counter is not null)
                {
                    await Counter.Release.Task;
                    await Task.Yield();
                }
                cancellationToken.ThrowIfCancellationRequested();
                int read = ReadCore(buffer.Span);
                BytesRead += read;
                return read;
            }
            finally
            {
                Counter?.Exit(writing: false);
                Interlocked.Decrement(ref active);
            }
        }

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("WriteAsync");
            int count = Interlocked.Increment(ref active);
            PeakOperations = Math.Max(PeakOperations, count);
            Counter?.Enter(writing: true);
            try
            {
                if (Counter is not null)
                {
                    await Counter.Release.Task;
                    await Task.Yield();
                }
                BeforeWrite?.Invoke(cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                inner.Write(buffer.Span);
                BytesWritten += buffer.Length;
            }
            finally
            {
                Counter?.Exit(writing: true);
                Interlocked.Decrement(ref active);
            }
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        public override int Read(byte[] buffer, int offset, int count)
        {
            Record("Read");
            throw new InvalidOperationException("Expected asynchronous I/O.");
        }
        public override void Write(byte[] buffer, int offset, int count)
        {
            Record("Write");
            throw new InvalidOperationException("Expected asynchronous I/O.");
        }
        public override long Seek(long offset, SeekOrigin origin)
        {
            Record("Seek");
            throw new NotSupportedException();
        }
        public override void SetLength(long value)
        {
            Record("SetLength");
            throw new NotSupportedException();
        }
        public override void Flush() => Record("Flush");
        public override Task FlushAsync(CancellationToken cancellationToken)
        {
            Record("Flush");
            return Task.CompletedTask;
        }
        public override void Close()
        {
            Record("Close");
            base.Close();
        }
        protected override void Dispose(bool disposing)
        {
            Record("Dispose");
            if (disposing)
            {
                inner.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    private sealed class OneByteReadStream(byte[] bytes) : RecordingStream(bytes)
    {
        public override bool CanSeek => false;
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        protected override int ReadCore(Span<byte> buffer) => base.ReadCore(buffer[..Math.Min(1, buffer.Length)]);
    }

    private sealed class TokenObservingStream : RecordingStream
    {
        public TaskCompletionSource<bool> Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource<bool> Canceled { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource<bool> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Finished { get; private set; }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("ReadAsync");
            Assert.True(cancellationToken.CanBeCanceled);
            using var registration = cancellationToken.Register(() => Canceled.TrySetResult(true));
            Started.TrySetResult(true);
            try
            {
                // Cancellation acknowledgement is separate from completion to expose early returns.
                await Release.Task;
                cancellationToken.ThrowIfCancellationRequested();
                return 0;
            }
            finally
            {
                Finished = true;
                Record("ReadCompleted");
            }
        }
    }

    private sealed class ThrowingStream(Task siblingStarted, IOException failure) : RecordingStream
    {
        private int reads;

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Record("ReadAsync");
            if (Interlocked.Increment(ref reads) == 1)
            {
                buffer.Span[0] = 0;
                BytesRead++;
                return 1;
            }
            await siblingStarted;
            throw failure;
        }
    }

    private sealed class IoCounter(int firstBatch)
    {
        private readonly object sync = new();
        private int reads;
        private int writes;
        public TaskCompletionSource<bool> FirstBatchStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource<bool> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public int Active { get; private set; }
        public int Peak { get; private set; }
        public bool OverlappedReadAndWrite { get; private set; }

        public void Enter(bool writing)
        {
            lock (sync)
            {
                if (writing)
                {
                    writes++;
                }
                else
                {
                    reads++;
                }
                OverlappedReadAndWrite |= reads > 0 && writes > 0;
                Active++;
                Peak = Math.Max(Peak, Active);
                if (Active >= firstBatch)
                {
                    FirstBatchStarted.TrySetResult(true);
                }
            }
        }

        public void Exit(bool writing)
        {
            lock (sync)
            {
                if (writing)
                {
                    writes--;
                }
                else
                {
                    reads--;
                }
                Active--;
            }
        }
    }
}
