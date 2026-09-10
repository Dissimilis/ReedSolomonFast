namespace ReedSolomonFast.Demo;

/// <summary>Four short follow-ups to the file recovery walkthrough, each on its own copy of the encoded shards.</summary>
/// <remarks>
/// Each returns true when the library did what the scenario expects, which for the first
/// is to refuse and for the next two starts with noticing something.
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

    /// <summary>Damage two shards without saying which: Verify notices, the stored digests say which, and both are rebuilt.</summary>
    /// <remarks>
    /// Reconstruct only rebuilds shards it is told are missing and trusts the rest, so a
    /// corrupted shard passed in as good would poison the output. The digests taken at encode
    /// time are what turn "something is wrong" into "these two are wrong".
    /// </remarks>
    public static bool ChecksumRecovery(ReedSolomon rs, byte[] original, byte[][] encoded)
    {
        string[] stored = encoded.Select(s => Program.Sha256(s)).ToArray();   // taken at encode time, kept with the shards
        byte[][] shards = Copy(encoded);
        shards[1][shards[1].Length / 2] ^= 0x40;   // a data shard, in the middle
        shards[5][^1] ^= 0x01;                     // a parity shard, at the end

        if (rs.Verify(shards)) return Program.Fail("Verify accepted a stripe with two corrupted shards");
        Console.WriteLine("Checksums: one byte flipped in each of two shards, without telling the decoder which.");
        Console.WriteLine("  Verify returned false, but it cannot say which shards changed; parity only proves the stripe is inconsistent.");

        // Classify every shard by its digest alone; the flip positions above play no part.
        bool[] present = new bool[shards.Length];
        for (int i = 0; i < shards.Length; i++) present[i] = Program.Sha256(shards[i]) == stored[i];
        int[] bad = Enumerable.Range(0, shards.Length).Where(i => !present[i]).ToArray();
        Console.WriteLine($"  SHA-256 against the stored digests: {string.Join(" and ", bad.Select(Program.ShardFileName))} do not match, the other {shards.Length - bad.Length} do.");
        if (bad.Length != 2) return Program.Fail($"expected exactly 2 digest mismatches, found {bad.Length}");

        rs.Reconstruct(shards, present);           // rebuilds the two unmatched shards in place
        for (int i = 0; i < shards.Length; i++)
            if (Program.Sha256(shards[i]) != stored[i]) return Program.Fail($"rebuilt {Program.ShardFileName(i)} does not match its digest");
        if (!rs.Verify(shards)) return Program.Fail("parity does not verify after rebuilding the corrupted shards");
        if (!rs.Join(shards, original.Length).AsSpan().SequenceEqual(original)) return Program.Fail("the joined bytes differ from the original");
        Console.WriteLine("  Marked those two missing and rebuilt them from the other four: all six digests match, parity verified, the joined file matches.");
        Console.WriteLine("  This assumes the digests themselves are intact: an unsigned manifest catches accidental damage, it does not authenticate the data or protect itself.");
        return true;
    }

    /// <summary>Damage three shards in three different places: as whole shards that is one too many, block by block it is not.</summary>
    /// <remarks>
    /// Coding is independent per byte position, so a block of every shard at the same offset is a
    /// stripe of its own and can be rebuilt through the Memory slice overloads. With a digest per
    /// block, a shard with one bad block still lends its other blocks to the repair.
    /// </remarks>
    public static bool BlockRecovery(ReedSolomon rs, byte[] original, byte[][] encoded)
    {
        int shardLength = encoded[0].Length;
        if (shardLength < 3)
        {
            Console.WriteLine($"Blocks: shards of {shardLength} byte{(shardLength == 1 ? "" : "s")} cannot hold three separate blocks, so this follow-up uses a built-in 64-byte block instead.");
            (original, encoded) = BuiltInBlock(rs);
            shardLength = encoded[0].Length;
        }

        // About eight blocks per shard, the last one possibly shorter. Digests are per (shard, block).
        int blockSize = Math.Max(1, (shardLength + 7) / 8);
        int blocks = (shardLength + blockSize - 1) / blockSize;
        string[][] stored = encoded.Select(s => Enumerable.Range(0, blocks).Select(b => Program.Sha256(Block(s, b))).ToArray()).ToArray();
        ReadOnlySpan<byte> Block(byte[] shard, int b) => shard.AsSpan(b * blockSize, Math.Min(blockSize, shardLength - b * blockSize));

        byte[][] shards = Copy(encoded);
        (int shard, int block)[] hits = [(0, 0), (2, blocks / 2), (5, blocks - 1)];   // three shards, three different blocks
        foreach (var (s, b) in hits) shards[s][b * blockSize] ^= 0x80;
        Console.WriteLine($"Blocks: one byte flipped in each of three shards, each in a different block; blocks are up to {blockSize} bytes, {blocks} per shard, with a digest each.");

        // Whole-shard digests condemn three of six shards, and four are needed.
        bool[] wholePresent = shards.Select((s, i) => Program.Sha256(s) == Program.Sha256(encoded[i])).ToArray();
        if (rs.TryReconstruct(shards, wholePresent)) return Program.Fail("TryReconstruct succeeded with three whole shards condemned");
        Console.WriteLine($"  Judged per shard, {wholePresent.Count(p => p)} of {shards.Length} are trustworthy and {rs.DataShards} are needed: TryReconstruct returned false.");

        // Judged per block, each damaged block has five good neighbours at the same offset.
        byte[][] before = Copy(shards);
        var slices = new Memory<byte>[shards.Length];
        bool[] present = new bool[shards.Length];
        int repairedBlocks = 0;
        for (int b = 0; b < blocks; b++)
        {
            int offset = b * blockSize, count = Math.Min(blockSize, shardLength - offset);
            for (int i = 0; i < shards.Length; i++)
            {
                slices[i] = shards[i].AsMemory(offset, count);
                present[i] = Program.Sha256(slices[i].Span) == stored[i][b];
            }

            if (present.All(p => p)) continue;
            if (present.Count(p => p) < rs.DataShards) return Program.Fail($"block {b} has fewer than {rs.DataShards} trustworthy slices");
            rs.Reconstruct(slices, present);   // one block-wide stripe, rebuilt in place inside the shard arrays
            repairedBlocks++;
        }

        // Nothing outside the three repaired blocks moved, and every shard is back to its encoded bytes.
        for (int i = 0; i < shards.Length; i++)
        {
            for (int p = 0; p < shardLength; p++)
                if (!hits.Contains((i, p / blockSize)) && shards[i][p] != before[i][p]) return Program.Fail($"byte {p} of {Program.ShardFileName(i)} changed outside a repaired block");
            if (!shards[i].AsSpan().SequenceEqual(encoded[i])) return Program.Fail($"{Program.ShardFileName(i)} differs from its encoded bytes after block repair");
        }

        if (repairedBlocks != hits.Length) return Program.Fail($"repaired {repairedBlocks} blocks, expected {hits.Length}");
        if (!rs.Join(shards, original.Length).AsSpan().SequenceEqual(original)) return Program.Fail("the joined bytes differ from the original");
        Console.WriteLine($"  Judged per block, {repairedBlocks} of {blocks * shards.Length} blocks are bad, each with 5 intact slices at the same offset. Rebuilt those three; every other byte is unchanged, all shards and the joined file match.");
        Console.WriteLine($"  The rule is per block: at least {rs.DataShards} of {shards.Length} trustworthy slices at the same offset. It does not raise the limit on whole lost shards.");
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

    /// <summary>A small nonempty input for the follow-ups when the file given to the demo is empty or too small.</summary>
    public static (byte[] Original, byte[][] Encoded) BuiltInBlock(ReedSolomon rs)
    {
        byte[] block = new byte[64];
        for (int i = 0; i < block.Length; i++) block[i] = (byte)(i * 31 + 7);
        byte[][] shards = rs.Split(block);
        rs.Encode(shards);
        return (block, shards);
    }

    private static bool ParityEquals(ReedSolomon rs, Memory<byte>[] parity, byte[][] reference)
    {
        for (int i = 0; i < rs.ParityShards; i++)
            if (!parity[i].Span.SequenceEqual(reference[rs.DataShards + i])) return false;
        return true;
    }

    private static byte[][] Copy(byte[][] shards) => shards.Select(s => s.ToArray()).ToArray();
}
