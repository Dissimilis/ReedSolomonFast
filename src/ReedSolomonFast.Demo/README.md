# ReedSolomonFast demo

A console app that takes a file through the whole erasure-coding workflow: split it into four
data shards, compute two parity shards, delete one of each, rebuild the file from the four that
remain, and check that its bytes match. Three short follow-ups show the recovery limit,
corruption detection, and incremental parity with `EncodeShards`, `EncodeShard` and `Update`.

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
| `manifest.json` | The original filename and length, geometry and matrix kind. `Split` does not store the length, so `Join` needs it from here, and the decoder is built from the geometry and matrix. |
| `damaged/` | The four surviving shards; `shard-01.bin` and `shard-05.bin` are missing. |
| `recovered/` | All six shards after `Reconstruct` from the damaged directory. |
| `restored/` | The rebuilt file, byte-identical to the input. |

Recovery reads only the manifest and the damaged directory. The input file and earlier runs are
never modified, and the run directory can be deleted afterwards.

The exit code is 0 only when every check passes: 1 for a failed check, 2 for bad arguments,
3 for a missing or oversized input, 4 for an I/O error.
