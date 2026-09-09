using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace ReedSolomonFast.Demo;

internal static class Program
{
    /// <summary>Four data and two parity shards: any four of the six rebuild the file.</summary>
    public const int DataShards = 4;
    public const int ParityShards = 2;

    /// <summary>Inputs are read whole into memory, so the demo caps them. The library has no such limit.</summary>
    public const long MaxInputBytes = 16 * 1024 * 1024;

    private static readonly string[] SampleNames = ["hello.txt", "readings.csv", "pattern.bin"];

    /// <summary>The two shards the walkthrough deletes: one data, one parity.</summary>
    private static readonly int[] LostShards = [1, 5];

    private static int Main(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;   // same number formatting on every machine
        Console.OutputEncoding = Encoding.UTF8;                        // file names may not be ASCII
        try
        {
            return (int)Run(args);
        }
        catch (DemoException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return (int)e.Code;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return (int)ExitCode.Io;
        }
    }

    private static ExitCode Run(string[] args)
    {
        string? file = null;
        string? output = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--help" or "-h":
                    PrintHelp();
                    return ExitCode.Success;
                case "--file":
                    file = OptionValue(args, ref i);
                    break;
                case "--output":
                    output = OptionValue(args, ref i);
                    break;
                default:
                    throw new DemoException(ExitCode.Usage, $"unexpected argument \"{args[i]}\". Run with --help.");
            }
        }

        string[] inputs = file is null ? BundledSamples() : [CheckInput(file)];
        string runDirectory = CreateRunDirectory(output);
        Console.WriteLine($"ReedSolomonFast demo. Results go to {runDirectory}");

        bool allPassed = true;
        byte[][]? firstEncoded = null;
        foreach (string input in inputs)
        {
            Console.WriteLine();
            string sampleDirectory = Path.Combine(runDirectory, Path.GetFileNameWithoutExtension(input));
            Directory.CreateDirectory(sampleDirectory);
            allPassed &= RecoverFile(input, sampleDirectory, out byte[][] encoded);
            firstEncoded ??= encoded;
        }

        // The three follow-ups run once, on copies of the first input's shards.
        Console.WriteLine();
        var rs = new ReedSolomon(DataShards, ParityShards);
        if (firstEncoded![0].Length == 0)
        {
            Console.WriteLine("The input is empty, so the follow-ups use a built-in 64-byte block instead.");
            firstEncoded = DemoScenarios.BuiltInBlock(rs);
        }

        allPassed &= DemoScenarios.TooManyMissing(rs, firstEncoded);
        allPassed &= DemoScenarios.CorruptionDetection(rs, firstEncoded);
        allPassed &= DemoScenarios.IncrementalParity(rs, firstEncoded);

        Console.WriteLine();
        Console.WriteLine(allPassed
            ? "Every check passed. The results directory can be deleted when you are done."
            : "Some checks failed; see the lines marked FAILED above.");
        return allPassed ? ExitCode.Success : ExitCode.CheckFailed;
    }

    /// <summary>Split, encode, lose two shards, recover from disk alone, join and compare.</summary>
    /// <remarks>Returns false when a check fails. <paramref name="encoded"/> gets the six shards either way.</remarks>
    private static bool RecoverFile(string inputPath, string sampleDirectory, out byte[][] encoded)
    {
        string name = Path.GetFileName(inputPath);
        byte[] original = File.ReadAllBytes(inputPath);
        var rs = new ReedSolomon(DataShards, ParityShards);

        Console.WriteLine($"File: {name} ({original.Length:N0} bytes), output in {Path.GetFileName(sampleDirectory)}{Path.DirectorySeparatorChar}");
        Console.WriteLine($"Layout: {DataShards} data + {ParityShards} parity; any {DataShards} shards are enough. Kernel: {rs.Kernel}.");

        // 1. Split into equal shards (the last data shard is zero-padded) and compute parity.
        encoded = rs.Split(original);
        rs.Encode(encoded);
        string encodedDirectory = Path.Combine(sampleDirectory, "encoded");
        WriteShards(encodedDirectory, encoded);
        if (!rs.Verify(encoded)) return Fail("parity does not verify right after Encode");
        int shardLength = rs.GetShardLength(original.Length);
        int padding = rs.GetPaddingLength(original.Length);
        Console.WriteLine($"Encoded {encoded.Length} shards of {shardLength:N0} bytes each; the last data shard ends with {padding} padding byte{(padding == 1 ? "" : "s")}. Parity verified.");
        Console.WriteLine($"  {string.Join(" ", Enumerable.Range(0, encoded.Length).Select(ShardFileName))}");

        // 2. The shards do not record the original length, so Join could not strip the padding
        //    without it. The manifest keeps it, with the geometry and matrix the decoder must match.
        Console.WriteLine("  The shards do not store the file length; manifest.json keeps it for Join.");
        string manifestPath = Path.Combine(sampleDirectory, "manifest.json");
        new Manifest(name, original.Length, DataShards, ParityShards, shardLength, MatrixKind.Vandermonde).Save(manifestPath);

        // 3. Lose one data shard and one parity shard: only the surviving four are copied.
        string damaged = Path.Combine(sampleDirectory, "damaged");
        Directory.CreateDirectory(damaged);
        for (int i = 0; i < encoded.Length; i++)
            if (!LostShards.Contains(i))
                File.Copy(Path.Combine(encodedDirectory, ShardFileName(i)), Path.Combine(damaged, ShardFileName(i)));
        Console.WriteLine($"Simulated loss: {string.Join(", ", LostShards.Select(i => $"{ShardFileName(i)} ({(i < DataShards ? "data" : "parity")})"))}");

        // 4. Recovery starts from disk only: the manifest, the four remaining files, a new coder.
        Manifest stored = Manifest.Load(manifestPath);
        var decoder = new ReedSolomon(stored.DataShards, stored.ParityShards, new ReedSolomonOptions { Matrix = stored.Matrix });
        byte[]?[] loaded = new byte[]?[decoder.TotalShards];
        for (int i = 0; i < loaded.Length; i++)
        {
            string path = Path.Combine(damaged, ShardFileName(i));
            if (!File.Exists(path)) continue;   // a null entry marks a missing shard
            byte[] shard = File.ReadAllBytes(path);
            if (shard.Length != stored.ShardLength)
                return Fail($"{ShardFileName(i)} is {shard.Length} bytes, the manifest says {stored.ShardLength}");
            loaded[i] = shard;
        }

        decoder.Reconstruct(loaded);   // allocates and fills the null entries in place
        byte[][] recovered = loaded!;  // no entry is null after Reconstruct
        if (!decoder.Verify(recovered)) return Fail("parity does not verify after Reconstruct");
        WriteShards(Path.Combine(sampleDirectory, "recovered"), recovered);
        Console.WriteLine("Recovered both missing shards. Parity verified.");

        // 5. Join the data shards into a file of the original length and compare.
        byte[] restored = decoder.Join(recovered, stored.OriginalLength);
        string restoredDirectory = Path.Combine(sampleDirectory, "restored");
        Directory.CreateDirectory(restoredDirectory);
        File.WriteAllBytes(Path.Combine(restoredDirectory, stored.FileName), restored);
        bool match = restored.AsSpan().SequenceEqual(original);
        Console.WriteLine($"Restored file: {(match ? "MATCH" : "FAILED, bytes differ")}");
        Console.WriteLine($"  SHA-256 original: {Sha256(original)}");
        Console.WriteLine($"  SHA-256 restored: {Sha256(restored)}");
        return match;
    }

    public static string ShardFileName(int index) => $"shard-{index:D2}.bin";

    public static bool Fail(string what)
    {
        Console.WriteLine($"FAILED: {what}");
        return false;
    }

    private static void WriteShards(string directory, byte[][] shards)
    {
        Directory.CreateDirectory(directory);
        for (int i = 0; i < shards.Length; i++)
            File.WriteAllBytes(Path.Combine(directory, ShardFileName(i)), shards[i]);
    }

    private static string Sha256(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static string OptionValue(string[] args, ref int i)
    {
        if (i + 1 >= args.Length || string.IsNullOrWhiteSpace(args[i + 1]) || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            throw new DemoException(ExitCode.Usage, $"{args[i]} needs a path. Run with --help.");
        return args[++i];
    }

    private static string[] BundledSamples()
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "Samples");
        var paths = new List<string>();
        foreach (string sample in SampleNames)
        {
            string path = Path.Combine(directory, sample);
            if (!File.Exists(path))
                throw new DemoException(ExitCode.Input, $"bundled sample \"{path}\" is missing; build the project again.");
            paths.Add(path);
        }

        return [.. paths];
    }

    private static string CheckInput(string path)
    {
        if (Directory.Exists(path)) throw new DemoException(ExitCode.Input, $"\"{path}\" is a directory; --file needs a file.");
        var info = new FileInfo(path);
        if (!info.Exists) throw new DemoException(ExitCode.Input, $"input file \"{path}\" does not exist.");
        if (info.Length > MaxInputBytes)
            throw new DemoException(ExitCode.Input,
                $"input file \"{path}\" is {info.Length:N0} bytes; the demo reads files whole and accepts up to {MaxInputBytes:N0}. The library has no such limit.");
        return info.FullName;
    }

    /// <summary>A fresh directory per run under artifacts/demo (or --output), so earlier results are never overwritten.</summary>
    private static string CreateRunDirectory(string? output)
    {
        string root;
        try
        {
            root = Path.GetFullPath(output ?? Path.Combine("artifacts", "demo"));
        }
        catch (ArgumentException e)
        {
            throw new DemoException(ExitCode.Usage, $"--output \"{output}\" is not a valid path: {e.Message}");
        }

        while (true)
        {
            string directory = Path.Combine(root, $"{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..6]}");
            if (Directory.Exists(directory)) continue;
            Directory.CreateDirectory(directory);
            return directory;
        }
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            ReedSolomonFast demo: split a file into shards, lose some, recover the file.

            Usage (from the repository root, with the .NET 10 SDK):
              dotnet run --project src/ReedSolomonFast.Demo -c Release [-- options]

            With no options the walkthrough runs on the three bundled samples. Each run writes
            the shards, the damaged copy and the restored file into a new directory and prints
            its path. Nothing outside that directory is touched.

            Options:
              --file "path"      Run the walkthrough on this file instead (up to 16 MiB).
              --output "path"    Parent directory for results (default: artifacts/demo).
              --help             Show this text.
            """);
    }
}
