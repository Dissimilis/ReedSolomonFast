using ReedSolomonFast.Internal;

namespace ReedSolomonFast.Tests;

public class Gf256Tests
{
    [Fact]
    public void Multiply_MatchesReference_AllPairs()
    {
        for (int a = 0; a < 256; a++)
            for (int b = 0; b < 256; b++)
            {
                byte expected = Gf256Reference.Multiply((byte)a, (byte)b);
                Assert.Equal(expected, Gf256.Multiply((byte)a, (byte)b));
                Assert.Equal(expected, Gf256.Mul[(a << 8) | b]);
            }
    }

    [Fact]
    public void Inverse_TimesSelf_IsOne()
    {
        for (int a = 1; a < 256; a++)
            Assert.Equal(1, Gf256.Multiply((byte)a, Gf256.Inverse((byte)a)));
    }

    [Fact]
    public void Divide_ByZero_Throws() => Assert.Throws<DivideByZeroException>(() => Gf256.Divide(5, 0));

    [Fact]
    public void Divide_IsInverseOfMultiply()
    {
        for (int a = 0; a < 256; a++)
            for (int b = 1; b < 256; b++)
                Assert.Equal(a, Gf256.Divide(Gf256.Multiply((byte)a, (byte)b), (byte)b));
    }

    [Fact]
    public void Exp_Log_RoundTrip()
    {
        for (int i = 0; i < 255; i++) Assert.Equal(i, Gf256.Log[Gf256.Exp[i]]);
        for (int i = 0; i < 255; i++) Assert.Equal(Gf256.Exp[i], Gf256.Exp[i + 255]);
    }

    [Fact]
    public void Power_MatchesRepeatedMultiply()
    {
        for (int a = 0; a < 256; a++)
        {
            byte acc = 1;
            for (int n = 0; n < 20; n++)
            {
                Assert.Equal(acc, Gf256.Power((byte)a, n));
                acc = Gf256Reference.Multiply(acc, (byte)a);
            }
        }
    }
}
