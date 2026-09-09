namespace ReedSolomonFast;

/// <summary>Construction used for the systematic coding matrix.</summary>
public enum MatrixKind
{
    /// <summary>Vandermonde matrix built the Backblaze way (Vandermonde times the inverse of its top square). Produces parity identical to Backblaze-derived libraries. The default.</summary>
    Vandermonde,

    /// <summary>Cauchy matrix. Every square submatrix is invertible by construction, and there is no inversion at construction time. Not byte-compatible with the Backblaze family.</summary>
    Cauchy,
}
