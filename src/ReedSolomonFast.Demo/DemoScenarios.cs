namespace ReedSolomonFast.Demo;

/// <summary>Three short follow-ups to the file recovery walkthrough, each on its own copy of the encoded shards.</summary>
/// <remarks>
/// Each returns true when the library did what the scenario expects, which for the first two
/// is to refuse or to notice something.
/// </remarks>
internal static class DemoScenarios
{
    /// <summary>Drop three of six shards. Three remain and four are needed, so TryReconstruct says no.</summary>
    public static bool TooManyMissing(ReedSolomon rs, byte[][] encoded)
    {
        byte[]?[] shards = Copy(encoded);
        shards[0] = shards[2] = shards[4] = null;

        if (rs.TryReconstruct(shards)) return Program.Fail("TryReconstruct succeeded with too few shards");
        Console.WriteLine($"Too many missing: 3 of {shards.Length} shards remain and {rs.DataShards} are required, so TryReconstruct returned false.");
        return true;
    }

    /// <summary>Flip one byte in a parity shard: Verify notices, and the shard is rebuilt once it is treated as lost.</summary>
    /// <remarks>Verify cannot say which shard changed. The demo knows, so it drops that shard and rebuilds it.</remarks>
    public static bool CorruptionDetection(ReedSolomon rs, byte[][] encoded)
    {
        byte[]?[] shards = Copy(encoded);
        int victim = rs.DataShards;      // the first parity shard, shard-04.bin in the walkthrough
        byte[] pristine = encoded[victim];
        shards[victim]![0] ^= 0x01;

        if (rs.Verify(shards!)) return Program.Fail("Verify accepted a corrupted parity shard");
        Console.WriteLine($"Corruption: one byte flipped in {Program.ShardFileName(victim)}, the first parity shard. Verify returned false; it cannot say which shard is bad.");

        shards[victim] = null;           // we know which one it was, so treat it as lost
        rs.Reconstruct(shards);
        if (!rs.Verify(shards!)) return Program.Fail("parity does not verify after rebuilding the corrupted shard");
        if (!shards[victim]!.AsSpan().SequenceEqual(pristine)) return Program.Fail("the rebuilt shard differs from the original");
        Console.WriteLine("  Dropped that shard and rebuilt it from the other five: parity verified, bytes match.");
        return true;
    }

    /// <summary>Build parity one data shard at a time, then patch it with Update after a data shard changes.</summary>
    /// <remarks>Both results must equal a full Encode.</remarks>
    public static bool IncrementalParity(ReedSolomon rs, byte[][] encoded)
    {
        byte[][] shards = Copy(encoded);
        ReadOnlyMemory<byte>[] data = shards[..rs.DataShards].Select(s => (ReadOnlyMemory<byte>)s).ToArray();
        Memory<byte>[] parity = ReedSolomon.AllocateShards(rs.ParityShards, shards[0].Length);
        foreach (var p in parity) p.Span.Clear();   // incremental encoding accumulates, so parity starts at zero

        // Feed every data index exactly once, in any order: two in one batch, then two singly.
        rs.EncodeShards([0, 1], data[..2], parity);
        rs.EncodeShard(2, data[2].Span, parity);
        rs.EncodeShard(3, data[3].Span, parity);
        if (!ParityEquals(rs, parity, shards)) return Program.Fail("incremental parity differs from Encode");
        Console.WriteLine("Incremental parity: EncodeShards(0, 1) + EncodeShard(2) + EncodeShard(3) equals a full Encode.");

        // Change data shard 0. Update needs its old and new bytes, not the other data shards.
        byte[] oldShard = shards[0];
        byte[] newShard = oldShard.ToArray();
        for (int i = 0; i < newShard.Length; i += 7) newShard[i] ^= 0xA5;
        rs.Update(changedIndices: [0], oldData: [oldShard], newData: [newShard], parity);

        shards[0] = newShard;
        rs.Encode(shards);
        if (!ParityEquals(rs, parity, shards)) return Program.Fail("updated parity differs from Encode");
        Console.WriteLine("  Update after changing shard-00.bin equals a fresh Encode of the changed data.");
        return true;
    }

    /// <summary>A small nonempty input for the follow-ups when the file given to the demo is empty.</summary>
    public static byte[][] BuiltInBlock(ReedSolomon rs)
    {
        byte[] block = new byte[64];
        for (int i = 0; i < block.Length; i++) block[i] = (byte)(i * 31 + 7);
        byte[][] shards = rs.Split(block);
        rs.Encode(shards);
        return shards;
    }

    private static bool ParityEquals(ReedSolomon rs, Memory<byte>[] parity, byte[][] reference)
    {
        for (int i = 0; i < rs.ParityShards; i++)
            if (!parity[i].Span.SequenceEqual(reference[rs.DataShards + i])) return false;
        return true;
    }

    private static byte[][] Copy(byte[][] shards) => shards.Select(s => s.ToArray()).ToArray();
}
