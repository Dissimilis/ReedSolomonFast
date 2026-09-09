using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ReedSolomonFast.Internal;

/// <summary>GF(2^8) arithmetic over the primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D), generator 2.</summary>
internal static class Gf256
{
    public const int Polynomial = 0x11D;

    /// <summary>Exp[i] = 2^i, doubled to 512 entries so Exp[Log[a] + Log[b]] needs no modulo.</summary>
    public static readonly byte[] Exp = BuildExp();

    /// <summary>Log[2^i] = i for i in [0,255); Log[0] is unused (0).</summary>
    public static readonly byte[] Log = BuildLog(Exp);

    /// <summary>Flat 256x256 product table indexed by (a &lt;&lt; 8) | b. Pinned, so <see cref="MulPointer"/> is stable.</summary>
    public static readonly byte[] Mul = BuildMul(Exp, Log);

    /// <summary>Raw pointer to <see cref="Mul"/> for the scalar kernel and vector tails.</summary>
    public static readonly unsafe byte* MulPointer = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(Mul));

    private static byte[] BuildExp()
    {
        var exp = new byte[512];
        int x = 1;
        for (int i = 0; i < 255; i++)
        {
            exp[i] = (byte)x;
            x <<= 1;
            if ((x & 0x100) != 0) x ^= Polynomial;
        }
        for (int i = 255; i < 512; i++) exp[i] = exp[i - 255];
        return exp;
    }

    private static byte[] BuildLog(byte[] exp)
    {
        var log = new byte[256];
        for (int i = 0; i < 255; i++) log[exp[i]] = (byte)i;
        return log;
    }

    private static byte[] BuildMul(byte[] exp, byte[] log)
    {
        var mul = GC.AllocateArray<byte>(65536, pinned: true);
        for (int a = 1; a < 256; a++)
        {
            int la = log[a];
            int row = a << 8;
            for (int b = 1; b < 256; b++) mul[row | b] = exp[la + log[b]];
        }
        return mul;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Multiply(byte a, byte b) => Mul[(a << 8) | b];

    public static byte Divide(byte a, byte b)
    {
        if (b == 0) throw new DivideByZeroException();
        if (a == 0) return 0;
        return Exp[Log[a] + 255 - Log[b]];
    }

    public static byte Inverse(byte a)
    {
        if (a == 0) throw new DivideByZeroException();
        return Exp[255 - Log[a]];
    }

    public static byte Power(byte a, int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        if (n == 0) return 1;
        if (a == 0) return 0;
        return Exp[(Log[a] * n) % 255];
    }
}
