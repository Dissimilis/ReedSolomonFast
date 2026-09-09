namespace ReedSolomonFast.Tests;

/// <summary>Naive shift-and-add GF(2^8) arithmetic over polynomial 0x11D, used as the oracle for every table and kernel.</summary>
internal static class Gf256Reference
{
    public static byte Multiply(byte a, byte b)
    {
        int result = 0;
        int x = a;
        int y = b;
        while (y != 0)
        {
            if ((y & 1) != 0) result ^= x;
            x <<= 1;
            if ((x & 0x100) != 0) x ^= 0x11D;
            y >>= 1;
        }
        return (byte)result;
    }

    public static byte Inverse(byte a)
    {
        if (a == 0) throw new DivideByZeroException();
        for (int i = 1; i < 256; i++)
            if (Multiply(a, (byte)i) == 1) return (byte)i;
        throw new InvalidOperationException("no inverse");
    }
}
