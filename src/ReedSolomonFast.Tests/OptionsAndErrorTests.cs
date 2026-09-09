namespace ReedSolomonFast.Tests;

public class OptionsAndErrorTests
{
    [Fact]
    public void Constructor_AcceptsLimits()
    {
        _ = new ReedSolomon(1, 1);
        var rs = new ReedSolomon(255, 1);
        Assert.Equal(256, rs.TotalShards);
        _ = new ReedSolomon(1, 255);
    }

    [Fact]
    public void Constructor_RejectsBadOptions()
    {
        Assert.Throws<ArgumentNullException>(() => new ReedSolomon(2, 2, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { MaxDegreeOfParallelism = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { MaxDegreeOfParallelism = -2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { InversionCacheSize = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { ParallelThresholdBytes = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { Kernel = (KernelTier)99 }));
        Assert.Throws<ArgumentException>(() => new ReedSolomon(3, 2, new ReedSolomonOptions { CustomParityRows = [[1, 2, 3]] }));
        Assert.Throws<ArgumentException>(() => new ReedSolomon(3, 2, new ReedSolomonOptions { CustomParityRows = [[1, 2], [3, 4]] }));
    }

    [Fact]
    public void Constructor_UnsupportedKernel_Throws()
    {
        var unsupported = Enum.GetValues<KernelTier>().Where(t => !Internal.Kernel.IsSupported(t)).ToArray();
        foreach (var tier in unsupported)
            Assert.Throws<PlatformNotSupportedException>(() => new ReedSolomon(2, 2, new ReedSolomonOptions { Kernel = tier }));
    }

    [Fact]
    public void Defaults_AreAsDocumented()
    {
        var rs = new ReedSolomon(4, 2);
        Assert.Same(ReedSolomonOptions.Default, rs.Options);
        Assert.Equal(MatrixKind.Vandermonde, rs.Options.Matrix);
        Assert.True(rs.Options.InversionCache);
        Assert.Equal(1, rs.Options.MaxDegreeOfParallelism);
        // The process default honours REEDSOLOMONFAST_KERNEL (CI runs the suite per tier);
        // BestSupportedKernel reports the hardware regardless.
        Assert.Equal(Internal.Kernel.Default, rs.Kernel);
        Assert.True(Internal.Kernel.IsSupported(rs.Kernel));
        Assert.True(Internal.Kernel.IsSupported(ReedSolomon.BestSupportedKernel));
        if (Environment.GetEnvironmentVariable("REEDSOLOMONFAST_KERNEL") is null) Assert.Equal(ReedSolomon.BestSupportedKernel, rs.Kernel);
    }

    [Fact]
    public void Encode_RejectsBadShapes()
    {
        var rs = new ReedSolomon(2, 2);
        Assert.Throws<ArgumentNullException>(() => rs.Encode((byte[][])null!));
        Assert.Throws<ArgumentException>(() => rs.Encode(new byte[3][] { new byte[4], new byte[4], new byte[4] }));
        Assert.Throws<ArgumentException>(() => rs.Encode(new byte[4][] { new byte[4], new byte[5], new byte[4], new byte[4] }));
        Assert.Throws<ArgumentException>(() => rs.Encode(new byte[4][] { new byte[4], null!, new byte[4], new byte[4] }));
        Assert.Throws<ArgumentException>(() => rs.Encode(new byte[7], new byte[8], 4));
        Assert.Throws<ArgumentException>(() => rs.Encode(new byte[8], new byte[7], 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => rs.Encode(new byte[8], new byte[8], -1));
        Assert.Throws<ArgumentException>(() => rs.Encode(new ReadOnlyMemory<byte>[1], new Memory<byte>[2]));
        Assert.Throws<ArgumentException>(() => rs.Encode(new ReadOnlyMemory<byte>[] { new byte[3], new byte[4] }, new Memory<byte>[] { new byte[3], new byte[3] }));
        Assert.Throws<ArgumentOutOfRangeException>(() => rs.EncodeShard(2, new byte[4], new Memory<byte>[] { new byte[4], new byte[4] }));
        Assert.Throws<ArgumentException>(() => rs.EncodeShard(0, new byte[3], new Memory<byte>[] { new byte[4], new byte[4] }));
        Assert.Throws<ArgumentOutOfRangeException>(() => rs.Update([5], [new byte[4]], [new byte[4]], new Memory<byte>[] { new byte[4], new byte[4] }));
        Assert.Throws<ArgumentException>(() => rs.Update([0], [new byte[4]], [], new Memory<byte>[] { new byte[4], new byte[4] }));
    }

    [Fact]
    public void Kernel_EnvironmentOverride_IsHonoured()
    {
        // The variable is read once per process, so this test only asserts the resolved value is
        // consistent with the environment it was read in.
        string? name = Environment.GetEnvironmentVariable(Internal.Kernel.EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(name))
            Assert.Equal(Internal.Kernel.Best(), Internal.Kernel.Default);
        else
            Assert.Equal(Enum.Parse<KernelTier>(name, ignoreCase: true), Internal.Kernel.Default);
    }
}
