# Contributing

Bug reports and pull requests are welcome. A few things make them easy to take.

## Building and testing

Requires the .NET 10 SDK (`global.json` pins the version).

```bash
dotnet build ReedSolomonFast.sln -c Release
dotnet test src/ReedSolomonFast.Tests -c Release

# The suite on one kernel tier (CI runs Scalar, Ssse3, Avx2, the best tier, Windows and ARM64)
REEDSOLOMONFAST_KERNEL=Avx2 dotnet test src/ReedSolomonFast.Tests -c Release
```

Every kernel tier must produce exactly what the shift-and-add reference in `Gf256Reference` produces,
and parity must stay byte-identical to the Backblaze ports (`InteropTests`). A change that makes a
test fail on any tier is a bug in the change until proven otherwise.

## Performance changes

Measure, do not reason. The benchmark project compares the working tree against a frozen copy of the
library (`src/ReedSolomonFast.Baseline`) in one process and prints an IMPROVED, REGRESSED or NO RESULT
verdict per case:

```bash
dotnet run --project src/ReedSolomonFast.Benchmarks -c Release
```

A performance change lands on IMPROVED verdicts in two separate runs, with no REGRESSED case and a
flat control case (one that does not run the changed code). Absolute times from different sessions
are not comparable; only ratios inside one run are. Please include the verdict lines in the pull
request, and say which machine and kernel tier produced them.

## Code

- Public API changes need tests for every overload shape (arrays, memories, contiguous stripe).
- Kernels are generic over `IGfVector<T>`; a change to the engine applies to every tier, so run the
  suite with `REEDSOLOMONFAST_KERNEL` set to each tier your machine supports.
- Comments say why, and cite the measurement when the reason is a number.
- No new runtime dependencies. Build-time packages are fine with `PrivateAssets="all"`.
