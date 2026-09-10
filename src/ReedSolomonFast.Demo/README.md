# ReedSolomonFast demo

A console app that takes a file through the whole erasure-coding workflow: split it into four
data shards, compute two parity shards, delete one of each, rebuild the file from the four that
remain, and check that its bytes match. Four short follow-ups show the recovery limit, two
kinds of checksum-assisted repair, and incremental parity with `EncodeShards`, `EncodeShard`
and `Update`.

The checksum follow-ups make a point the walkthrough cannot: `Verify` says a stripe is
inconsistent, not which shard is wrong, and `Reconstruct` trusts every shard marked present and
rebuilds only those marked missing. A SHA-256 per shard, taken at encode time and kept in the
manifest, is what turns "something is wrong" into "these two are wrong", so they can be marked
missing and rebuilt. The block follow-up keeps a separate digest per block, in memory, taken
before the damage: with three shards damaged in three different places, whole-shard digests
leave too few shards to rebuild from, but each damaged block still has five intact slices at
the same offset and is rebuilt through the `Memory` slice overloads, touching nothing else.
Nothing from the block follow-up is written to disk. The digests only catch accidental damage;
an unsigned manifest does not authenticate anything.

From the repository root, with the .NET 10 SDK installed:

```text
dotnet run --project src/ReedSolomonFast.Demo -c Release
```

Run it on a file of your own (up to 16 MiB, the demo reads it whole):

```text
dotnet run --project src/ReedSolomonFast.Demo -c Release -- --file "photos/holiday.jpg"
```

`--output "path"` picks the parent directory for results; `--help` lists the options.

## Bundled samples

| File | What it shows |
|---|---|
| `hello.txt` | A short text with a few non-ASCII characters. |
| `readings.csv` | A tiny table of invented sensor readings, easy to open and compare. |
| `pattern.bin` | A 1,027-byte pattern containing every byte value: binary data, and padding since 1,027 is not divisible by four. |

## Output

Each run creates a new directory, by default `artifacts/demo/<timestamp>` under the current
directory, and prints its absolute path. Every input gets its own subdirectory:

| Entry | Contents |
|---|---|
| `encoded/` | All six shards as written by `Split` and `Encode`. |
| `manifest.json` | The original filename and length, geometry, matrix kind and a SHA-256 per shard. `Split` does not store the length, so `Join` needs it from here; the decoder is built from the geometry and matrix; the digests decide which shards read back can be trusted. |
| `damaged/` | The four surviving shards; `shard-01.bin` and `shard-05.bin` are missing. |
| `recovered/` | All six shards after `Reconstruct` from the damaged directory. A shard that is missing, has the wrong length or fails its digest counts as missing. |
| `restored/` | The rebuilt file, byte-identical to the input. |

Recovery reads only the manifest and the damaged directory. The input file and earlier runs are
never modified, and the run directory can be deleted afterwards.

The exit code is 0 only when every check passes: 1 for a failed check, 2 for bad arguments,
3 for a missing or oversized input, 4 for an I/O error.
