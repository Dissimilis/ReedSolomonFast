# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in ReedSolomonFast, please report it privately to the maintainer rather than opening a public issue.

**Contacting the maintainer:** Create a private security advisory on GitHub via the repository's Security tab.

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Considerations

This is a **community-maintained, pure managed C#** erasure coding library. It has **not been independently audited**.

Reed-Solomon erasure coding is not a cryptographic primitive: it repairs missing shards, not tampered ones. `Verify` says whether parity matches data; it cannot say which shard is wrong, and whoever can change a shard can change its parity too. Authenticate shards separately (a keyed hash per shard) when integrity against an adversary matters.

The implementation:

- Uses hardware intrinsics and unsafe pointer code over pinned buffers. Buffer lengths and supported overlap rules are checked before coding operations.
- Allocates shard buffers in convenience overloads, rents scratch for verification, and builds decode plans for uncached reconstruction patterns. Callers should bound shard sizes and concurrent work when handling untrusted input.
- Has no runtime package dependencies.
