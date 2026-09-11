namespace ReedSolomonFast;

/// <summary>
/// Encodes, reconstructs and verifies shards held in <see cref="Stream"/>s, one window at a time,
/// with memory bounded by the window size rather than the shard length. The streams carry the
/// same bytes as the in-memory API: data shard <c>i</c> of a file is bytes
/// <c>[i * shardLength, (i + 1) * shardLength)</c> of the zero-padded file, as <see cref="ReedSolomon.Split(ReadOnlySpan{byte})"/>
/// lays it out, and parity stream <c>j</c> equals parity shard <c>j</c> from <see cref="ReedSolomon.Encode(byte[][])"/>.
/// </summary>
/// <remarks>
/// A method that runs to completion reads exactly <c>shardLength</c> bytes from each participating
/// input's current position and writes exactly that many to each output; <see cref="VerifyAsync"/>
/// stops at the first mismatching window, and a reconstruct that asks for nothing, or reads
/// only the inputs it needs, leaves the rest untouched. Nothing is padded, sought, flushed or
/// disposed: a stream that ends early throws <see cref="EndOfStreamException"/>, a longer one keeps
/// its suffix unread, and the caller supplies any zero padding. The kernel runs on the thread the
/// awaited I/O completes on; the coder's own parallelism settings apply to it as usual.
/// On failure or cancellation, in-flight I/O is cancelled and awaited before the exception
/// surfaces; outputs may then hold a partial window and inputs may have advanced unequally.
/// </remarks>
public static class ReedSolomonStreams
{
    /// <summary>
    /// Reads <paramref name="shardLength"/> bytes from each of <c>DataShards</c> data streams and
    /// writes the <c>ParityShards</c> parity streams. The caller pads the data to a multiple of
    /// <c>DataShards</c> times the shard length beforehand, as <see cref="ReedSolomon.Split(ReadOnlySpan{byte})"/> would.
    /// </summary>
    public static async Task EncodeAsync(
        ReedSolomon coder, Stream[] data, Stream[] parity, long shardLength,
        ReedSolomonStreamingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(coder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(parity);
        if (data.Length != coder.DataShards) throw new ArgumentException($"Expected {coder.DataShards} data streams, got {data.Length}.", nameof(data));
        if (parity.Length != coder.ParityShards) throw new ArgumentException($"Expected {coder.ParityShards} parity streams, got {parity.Length}.", nameof(parity));
        options ??= ReedSolomonStreamingOptions.Default;
        CheckOptions(options);
        ArgumentOutOfRangeException.ThrowIfNegative(shardLength);

        // Copy the references first, so a caller mutating the arrays mid-call cannot change the plan.
        var inputs = new Stream?[coder.TotalShards];
        var outputs = new Stream?[coder.TotalShards];
        for (int i = 0; i < data.Length; i++) inputs[i] = data[i] ?? throw new ArgumentException($"Data stream {i} is null.", nameof(data));
        for (int p = 0; p < parity.Length; p++) outputs[coder.DataShards + p] = parity[p] ?? throw new ArgumentException($"Parity stream {p} is null.", nameof(parity));
        CheckStreams(inputs, outputs, nameof(data), nameof(parity));
        if (shardLength == 0) return;

        var window = new Window(coder, shardLength, options);
        var dataMemories = new ReadOnlyMemory<byte>[coder.DataShards];
        var parityMemories = new Memory<byte>[coder.ParityShards];
        while (window.Next())
        {
            cancellationToken.ThrowIfCancellationRequested();
            await window.ReadAsync(inputs, cancellationToken).ConfigureAwait(false);

            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < coder.DataShards; i++) dataMemories[i] = window.Shards[i];
            for (int p = 0; p < coder.ParityShards; p++) parityMemories[p] = window.Shards[coder.DataShards + p];
            coder.Encode(dataMemories, parityMemories);

            cancellationToken.ThrowIfCancellationRequested();
            await window.WriteAsync(outputs, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Rebuilds missing shards from streams. <paramref name="inputs"/> and <paramref name="outputs"/>
    /// both hold <c>TotalShards</c> entries, data first. A null input is a missing shard; a non-null
    /// output asks for the shard at that index to be rebuilt into it, and is only allowed where the
    /// input is null. Present shards are trusted as they are; use <see cref="VerifyAsync"/> or
    /// checksums of your own to decide what to trust. Each window is a reconstruct call of its own,
    /// so the coder's inversion cache (on by default) is what keeps the matrix from being inverted
    /// once per window; with <see cref="ReedSolomonOptions.InversionCache"/> off, it is.
    /// </summary>
    /// <exception cref="InsufficientShardsException">Fewer than <c>DataShards</c> inputs are present and at least one output was requested.</exception>
    public static async Task ReconstructAsync(
        ReedSolomon coder, Stream?[] inputs, Stream?[] outputs, long shardLength,
        ReedSolomonStreamingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(coder);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        if (inputs.Length != coder.TotalShards) throw new ArgumentException($"Expected {coder.TotalShards} input entries, got {inputs.Length}.", nameof(inputs));
        if (outputs.Length != coder.TotalShards) throw new ArgumentException($"Expected {coder.TotalShards} output entries, got {outputs.Length}.", nameof(outputs));
        options ??= ReedSolomonStreamingOptions.Default;
        CheckOptions(options);
        ArgumentOutOfRangeException.ThrowIfNegative(shardLength);

        inputs = (Stream?[])inputs.Clone();
        outputs = (Stream?[])outputs.Clone();
        var present = new bool[coder.TotalShards];
        var required = new bool[coder.TotalShards];
        int presentCount = 0, requiredCount = 0;
        for (int i = 0; i < coder.TotalShards; i++)
        {
            present[i] = inputs[i] is not null;
            required[i] = outputs[i] is not null;
            if (present[i]) presentCount++;
            if (required[i]) requiredCount++;
            if (present[i] && required[i]) throw new ArgumentException($"Shard {i} is present, so it cannot also be an output.", nameof(outputs));
        }

        CheckStreams(inputs, outputs, nameof(inputs), nameof(outputs));
        if (requiredCount == 0) return;   // nothing asked for; validated, no I/O
        if (presentCount < coder.DataShards) throw new InsufficientShardsException(presentCount, coder.DataShards);
        if (shardLength == 0) return;

        // Only the first DataShards present inputs take part, the same choice the in-memory
        // planner makes, so a surplus input is never read: it cannot fail or stall the rebuild.
        for (int i = 0, chosen = 0; i < coder.TotalShards; i++)
        {
            if (!present[i]) continue;
            if (chosen < coder.DataShards) { chosen++; continue; }
            present[i] = false;
            inputs[i] = null;
        }

        var window = new Window(coder, shardLength, options);
        while (window.Next())
        {
            cancellationToken.ThrowIfCancellationRequested();
            await window.ReadAsync(inputs, cancellationToken).ConfigureAwait(false);

            cancellationToken.ThrowIfCancellationRequested();
            coder.ReconstructSome(window.Shards, present, required);

            cancellationToken.ThrowIfCancellationRequested();
            await window.WriteAsync(outputs, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Returns true when the parity streams match the data streams over <paramref name="shardLength"/>
    /// bytes. Stops at the first window that does not match, leaving every stream positioned after
    /// it. Like <see cref="ReedSolomon.Verify(byte[][])"/>, a false result does not say which shard is wrong.
    /// </summary>
    public static async Task<bool> VerifyAsync(
        ReedSolomon coder, Stream[] shards, long shardLength,
        ReedSolomonStreamingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(coder);
        ArgumentNullException.ThrowIfNull(shards);
        if (shards.Length != coder.TotalShards) throw new ArgumentException($"Expected {coder.TotalShards} shard streams, got {shards.Length}.", nameof(shards));
        options ??= ReedSolomonStreamingOptions.Default;
        CheckOptions(options);
        ArgumentOutOfRangeException.ThrowIfNegative(shardLength);

        var inputs = new Stream?[coder.TotalShards];
        for (int i = 0; i < shards.Length; i++) inputs[i] = shards[i] ?? throw new ArgumentException($"Shard stream {i} is null.", nameof(shards));
        CheckStreams(inputs, new Stream?[coder.TotalShards], nameof(shards), nameof(shards));
        if (shardLength == 0) return true;

        var window = new Window(coder, shardLength, options);
        var memories = new ReadOnlyMemory<byte>[coder.TotalShards];
        long scratchLength = (long)coder.ParityShards * window.Size;   // the scratch overload wants ParityShards times the shard length
        if (scratchLength > Array.MaxLength) throw new ArgumentOutOfRangeException(nameof(options), "WindowSizeBytes times ParityShards exceeds the maximum array length.");
        byte[] scratch = new byte[scratchLength];
        while (window.Next())
        {
            cancellationToken.ThrowIfCancellationRequested();
            await window.ReadAsync(inputs, cancellationToken).ConfigureAwait(false);

            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < coder.TotalShards; i++) memories[i] = window.Shards[i];
            bool matches = coder.Verify(memories, scratch.AsSpan(0, coder.ParityShards * window.Count));
            cancellationToken.ThrowIfCancellationRequested();   // a cancelled call reports cancellation, not a verdict
            if (!matches) return false;
        }

        return true;
    }

    private static void CheckOptions(ReedSolomonStreamingOptions options)
    {
        if (options.WindowSizeBytes <= 0) throw new ArgumentOutOfRangeException(nameof(options), "WindowSizeBytes must be positive.");
        if (options.MaxConcurrentIoOperations <= 0) throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentIoOperations must be positive.");
    }

    /// <summary>Every input readable, every output writable, no stream object used twice. Indices are shard indices, data first.</summary>
    private static void CheckStreams(Stream?[] inputs, Stream?[] outputs, string inputsName, string outputsName)
    {
        var seen = new HashSet<Stream>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < inputs.Length; i++)
        {
            Stream? s = inputs[i];
            if (s is null) continue;
            if (!s.CanRead) throw new ArgumentException($"The stream for shard {i} is not readable.", inputsName);
            if (!seen.Add(s)) throw new ArgumentException($"The stream for shard {i} is the same object as another stream; each shard needs its own stream and position.", inputsName);
        }

        for (int i = 0; i < outputs.Length; i++)
        {
            Stream? s = outputs[i];
            if (s is null) continue;
            if (!s.CanWrite) throw new ArgumentException($"The output stream for shard {i} is not writable.", outputsName);
            if (!seen.Add(s)) throw new ArgumentException($"The output stream for shard {i} is the same object as another stream; each shard needs its own stream and position.", outputsName);
        }
    }

    /// <summary>
    /// One padded stripe of <c>TotalShards</c> windows, allocated once and reused, with the bounded
    /// read and write phases over it. <see cref="Shards"/> is re-sliced to the current window's length.
    /// </summary>
    private sealed class Window
    {
        private readonly Memory<byte>[] _full;
        private readonly long _shardLength;
        private readonly int _maxConcurrent;
        private long _offset = -1;

        public Window(ReedSolomon coder, long shardLength, ReedSolomonStreamingOptions options)
        {
            _shardLength = shardLength;
            _maxConcurrent = options.MaxConcurrentIoOperations;
            Size = (int)Math.Min(options.WindowSizeBytes, shardLength);
            _full = ReedSolomon.AllocateShards(coder.TotalShards, Size);
            Shards = new Memory<byte>[coder.TotalShards];
        }

        /// <summary>Bytes per shard in a full window.</summary>
        public int Size { get; }

        /// <summary>Bytes per shard in the current window; only the last one can be shorter.</summary>
        public int Count { get; private set; }

        /// <summary>The current window of every shard, <see cref="Count"/> bytes each.</summary>
        public Memory<byte>[] Shards { get; }

        public bool Next()
        {
            _offset = _offset < 0 ? 0 : _offset + Count;
            if (_offset >= _shardLength) return false;
            Count = (int)Math.Min(Size, _shardLength - _offset);
            for (int i = 0; i < Shards.Length; i++) Shards[i] = _full[i].Slice(0, Count);
            return true;
        }

        public Task ReadAsync(Stream?[] inputs, CancellationToken cancellationToken)
            => RunBoundedAsync(inputs, ReadOneAsync, cancellationToken);

        public Task WriteAsync(Stream?[] outputs, CancellationToken cancellationToken)
            => RunBoundedAsync(outputs, (s, i, ct) => s.WriteAsync(Shards[i], ct).AsTask(), cancellationToken);

        private async Task ReadOneAsync(Stream stream, int index, CancellationToken cancellationToken)
        {
            try
            {
                await stream.ReadExactlyAsync(Shards[index], cancellationToken).ConfigureAwait(false);
            }
            catch (EndOfStreamException e)
            {
                throw new EndOfStreamException($"Shard stream {index} ended before offset {_offset + Count}; {_shardLength} bytes were required.", e);
            }
        }

        /// <summary>
        /// Runs one operation per non-null stream with at most <see cref="_maxConcurrent"/> in flight.
        /// The first failure cancels the rest; every started operation is awaited before it is rethrown.
        /// </summary>
        private async Task RunBoundedAsync(Stream?[] streams, Func<Stream, int, CancellationToken, Task> operation, CancellationToken cancellationToken)
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var running = new List<Task>(Math.Min(_maxConcurrent, streams.Length));
            Exception? failure = null;
            int next = 0;
            while (true)
            {
                // Launch up to the limit, unless something has failed or the caller has cancelled.
                // A stream's WriteAsync may throw synchronously; that counts as a failure like any other.
                while (failure is null && running.Count < _maxConcurrent && next < streams.Length)
                {
                    if (cancellationToken.IsCancellationRequested) { failure = new OperationCanceledException(cancellationToken); break; }
                    if (streams[next] is { } stream)
                    {
                        try { running.Add(operation(stream, next, linked.Token)); }
                        catch (Exception e) { failure = e; }
                    }

                    next++;
                }

                if (failure is not null) Cancel(linked);
                if (running.Count == 0) break;
                await Task.WhenAny(running).ConfigureAwait(false);

                // Observe every finished task before launching more, so a fault is seen before a
                // replacement starts and no exception goes unobserved. When several fault in the
                // same round, which one is reported is not defined.
                for (int i = running.Count - 1; i >= 0; i--)
                {
                    Task t = running[i];
                    if (!t.IsCompleted) continue;
                    running.RemoveAt(i);
                    try { await t.ConfigureAwait(false); }
                    catch (Exception e) { failure ??= e; }
                }

                if (failure is not null) Cancel(linked);
            }

            if (failure is not null)
            {
                // A cancellation caused by the caller's token carries that token, whether it came
                // through the linked source or was raised by the runner; anything else keeps its type and stack.
                if (failure is OperationCanceledException oce && cancellationToken.IsCancellationRequested && oce.CancellationToken != cancellationToken)
                    throw new OperationCanceledException(oce.Message, oce, cancellationToken);
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Throw(failure);
            }
        }

        /// <summary>Cancels sibling I/O; a stream's own cancellation callback throwing must not hide the first failure.</summary>
        private static void Cancel(CancellationTokenSource linked)
        {
            if (linked.IsCancellationRequested) return;
            try { linked.Cancel(); }
            catch (AggregateException) { }
        }
    }
}
