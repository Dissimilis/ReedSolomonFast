

namespace ReedSolomonFast.Internal;

/// <summary>Dense byte matrix over GF(2^8) with the operations needed for systematic erasure coding.</summary>
internal sealed class CodingMatrix
{
    private readonly byte[] _m;

    public int Rows { get; }
    public int Cols { get; }

    public CodingMatrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        _m = new byte[rows * cols];
    }

    public byte this[int r, int c]
    {
        get => _m[r * Cols + c];
        set => _m[r * Cols + c] = value;
    }

    /// <summary>Row-major view of all entries.</summary>
    public ReadOnlySpan<byte> Data => _m;

    public ReadOnlySpan<byte> RowSpan(int r) => _m.AsSpan(r * Cols, Cols);

    public byte[] Row(int r) => RowSpan(r).ToArray();

    /// <summary>Builds the (data + parity) x data systematic matrix whose top square is the identity.</summary>
    public static CodingMatrix Create(MatrixKind kind, int data, int parity) => kind switch
    {
        MatrixKind.Cauchy => Cauchy(data, parity),
        MatrixKind.Vandermonde => Vandermonde(data, parity),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    /// <summary>Identity on top of caller-supplied parity rows.</summary>
    public static CodingMatrix FromParityRows(byte[][] rows, int data, int parity)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Length != parity) throw new ArgumentException($"Expected {parity} parity rows, got {rows.Length}.", nameof(rows));
        var m = new CodingMatrix(data + parity, data);
        for (int i = 0; i < data; i++) m[i, i] = 1;
        for (int r = 0; r < parity; r++)
        {
            byte[] row = rows[r] ?? throw new ArgumentException($"Parity row {r} is null.", nameof(rows));
            if (row.Length != data) throw new ArgumentException($"Parity row {r} has {row.Length} entries, expected {data}.", nameof(rows));
            row.CopyTo(m._m, (data + r) * data);
        }

        return m;
    }

    private static CodingMatrix Cauchy(int data, int parity)
    {
        var m = new CodingMatrix(data + parity, data);
        for (int i = 0; i < data; i++) m[i, i] = 1;
        for (int r = 0; r < parity; r++)
            for (int c = 0; c < data; c++)
                m[data + r, c] = Gf256.Inverse((byte)((data + r) ^ c));
        return m;
    }

    private static CodingMatrix Vandermonde(int data, int parity)
    {
        int total = data + parity;
        var v = new CodingMatrix(total, data);
        for (int r = 0; r < total; r++)
            for (int c = 0; c < data; c++)
                v[r, c] = Gf256.Power((byte)r, c);
        var top = v.SubMatrix(Enumerable.Range(0, data).ToArray());
        return v.Multiply(top.Invert());
    }

    public CodingMatrix SubMatrix(ReadOnlySpan<int> rows)
    {
        var s = new CodingMatrix(rows.Length, Cols);
        for (int i = 0; i < rows.Length; i++) RowSpan(rows[i]).CopyTo(s._m.AsSpan(i * Cols, Cols));
        return s;
    }

    public CodingMatrix Multiply(CodingMatrix other)
    {
        if (Cols != other.Rows) throw new ArgumentException("Dimension mismatch.", nameof(other));
        var p = new CodingMatrix(Rows, other.Cols);
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < other.Cols; c++)
            {
                byte acc = 0;
                for (int k = 0; k < Cols; k++) acc ^= Gf256.Multiply(this[r, k], other[k, c]);
                p[r, c] = acc;
            }
        return p;
    }

    /// <summary>Gauss-Jordan inversion. Throws <see cref="InvalidOperationException"/> when singular.</summary>
    public CodingMatrix Invert()
    {
        if (Rows != Cols) throw new InvalidOperationException("Only square matrices can be inverted.");
        int n = Rows;
        var work = new CodingMatrix(n, 2 * n);
        for (int r = 0; r < n; r++)
        {
            RowSpan(r).CopyTo(work._m.AsSpan(r * 2 * n, n));
            work[r, n + r] = 1;
        }

        for (int col = 0; col < n; col++)
        {
            int pivot = -1;
            for (int r = col; r < n; r++)
                if (work[r, col] != 0) { pivot = r; break; }
            if (pivot < 0) throw new InvalidOperationException("Matrix is singular.");
            if (pivot != col) work.SwapRows(pivot, col);

            byte inv = Gf256.Inverse(work[col, col]);
            if (inv != 1)
                for (int c = 0; c < 2 * n; c++) work[col, c] = Gf256.Multiply(work[col, c], inv);

            for (int r = 0; r < n; r++)
            {
                if (r == col) continue;
                byte f = work[r, col];
                if (f == 0) continue;
                for (int c = 0; c < 2 * n; c++) work[r, c] ^= Gf256.Multiply(f, work[col, c]);
            }
        }

        var result = new CodingMatrix(n, n);
        for (int r = 0; r < n; r++) work._m.AsSpan(r * 2 * n + n, n).CopyTo(result._m.AsSpan(r * n, n));
        return result;
    }

    private void SwapRows(int a, int b)
    {
        Span<byte> ra = _m.AsSpan(a * Cols, Cols);
        Span<byte> rb = _m.AsSpan(b * Cols, Cols);
        byte[] tmp = ra.ToArray();
        rb.CopyTo(ra);
        tmp.CopyTo(rb);
    }

    public bool IsIdentity()
    {
        if (Rows != Cols) return false;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                if (this[r, c] != (r == c ? 1 : 0)) return false;
        return true;
    }
}
