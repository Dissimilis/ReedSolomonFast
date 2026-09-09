using System.Buffers;
using System.Runtime.InteropServices;

namespace ReedSolomonFast.Internal;

/// <summary>An operation that runs once every shard has a stable address.</summary>
internal unsafe interface IPinnedOperation
{
    void Execute(byte** ptrs);
}

/// <summary>
/// Pins any number of shards without allocating a handle per shard: each shard is pinned by a
/// <c>fixed</c> statement in its own recursion frame, and the operation runs at the bottom of the
/// recursion with every pointer valid. Up to 256 shards means up to 256 small frames, which is
/// cheaper than 256 <see cref="GCHandle"/> allocations on every call. Memory that is not backed by
/// an array (native memory, a custom <see cref="MemoryManager{T}"/>) is pinned through
/// <see cref="ReadOnlyMemory{T}.Pin"/> instead, in the same frame.
/// </summary>
/// <remarks>
/// A null array or empty memory pins as a null pointer; callers only pass those for shards the
/// operation will not touch (absent shards in reconstruction).
/// </remarks>
internal static unsafe class Pinner
{
    /// <summary>Pins <paramref name="shards"/> at a common <paramref name="offset"/> into <paramref name="ptrs"/> and runs <paramref name="op"/>.</summary>
    public static void Run<TOp>(ReadOnlySpan<byte[]?> shards, int offset, byte** ptrs, ref TOp op)
        where TOp : struct, IPinnedOperation
    {
        PinArrays(shards, offset, ptrs, 0, ref op);
    }

    /// <summary>
    /// Pins three groups of memories in order (typically inputs, then more inputs, then outputs)
    /// into one pointer array and runs <paramref name="op"/>. Empty groups are fine.
    /// </summary>
    public static void Run<TOp>(
        ReadOnlySpan<ReadOnlyMemory<byte>> a,
        ReadOnlySpan<ReadOnlyMemory<byte>> b,
        ReadOnlySpan<Memory<byte>> c,
        byte** ptrs,
        ref TOp op)
        where TOp : struct, IPinnedOperation
    {
        PinMemories(a, b, c, ptrs, 0, ref op);
    }

    private static void PinArrays<TOp>(ReadOnlySpan<byte[]?> shards, int offset, byte** ptrs, int index, ref TOp op)
        where TOp : struct, IPinnedOperation
    {
        if (index == shards.Length)
        {
            op.Execute(ptrs);
            return;
        }

        byte[]? array = shards[index];
        if (array is null || array.Length == 0)
        {
            ptrs[index] = null;
            PinArrays(shards, offset, ptrs, index + 1, ref op);
            return;
        }

        fixed (byte* p = array)
        {
            ptrs[index] = p + offset;
            PinArrays(shards, offset, ptrs, index + 1, ref op);
        }
    }

    private static void PinMemories<TOp>(
        ReadOnlySpan<ReadOnlyMemory<byte>> a,
        ReadOnlySpan<ReadOnlyMemory<byte>> b,
        ReadOnlySpan<Memory<byte>> c,
        byte** ptrs,
        int index,
        ref TOp op)
        where TOp : struct, IPinnedOperation
    {
        int total = a.Length + b.Length + c.Length;
        if (index == total)
        {
            op.Execute(ptrs);
            return;
        }

        ReadOnlyMemory<byte> memory = index < a.Length ? a[index]
            : index < a.Length + b.Length ? b[index - a.Length]
            : c[index - a.Length - b.Length];

        if (memory.IsEmpty)
        {
            ptrs[index] = null;
            PinMemories(a, b, c, ptrs, index + 1, ref op);
            return;
        }

        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<byte> segment))
        {
            fixed (byte* p = segment.Array)
            {
                ptrs[index] = p + segment.Offset;
                PinMemories(a, b, c, ptrs, index + 1, ref op);
            }

            return;
        }

        using MemoryHandle handle = memory.Pin();
        ptrs[index] = (byte*)handle.Pointer;
        PinMemories(a, b, c, ptrs, index + 1, ref op);
    }
}
