# ROCK 5B ARM NEON benchmarks, 2026-11-03

Measured pushed commit `8ad7b12` against its parent `680a614`, using the same caller buffers in one process. The historical generated baseline snapshot was not changed. A separate comparison project compiled parent sources under a distinct assembly name. Identical-source calibration used current sources for both assemblies.

RK3588 Cortex-A76, pinned CPU4, one worker, .NET 10.0.12 / SDK10.0.401. AdvSimd selected; tiered compilation and ReadyToRun disabled. Governor remained ondemand. Reused arrays and padded layouts, 64 KiB per shard. These are warm-buffer measurements, not cold-memory or default-tiering claims.

Two construction orders each for identical-source and parent/current comparisons. Thirty ABBA/BAAB paired samples per cell, 99.9% Student-t intervals on log ratios; individual sample ticks retained. Maximum identical-source deviation plus one percentage point gave **6.89%** practical margin. All eight focused cells had upper interval bounds below 1 minus this margin in both processes. This is a benchmark confirmation of the already committed change, not a new four-process experiment promotion.

| Operation | Shape | Layout | Parent us | Current us | Less time across two processes |
|---|---|---|---:|---:|---:|
| Encode | 10+5 | arrays | 342.38 | 292.88 | 13.9-15.0% |
| Reconstruct | 10+5 | arrays | 349.83 | 278.33 | 19.6-21.2% |
| Encode | 10+5 | padded | 297.62 | 254.29 | 11.7-17.2% |
| Reconstruct | 10+5 | padded | 291.34 | 265.31 | 8.9-9.0% |
| Encode | 11+3 | arrays | 242.88 | 207.01 | 14.0-15.6% |
| Reconstruct | 11+3 | arrays | 245.09 | 208.54 | 14.7-15.1% |
| Encode | 11+3 | padded | 207.31 | 189.88 | 7.8-9.0% |
| Reconstruct | 11+3 | padded | 198.09 | 178.64 | 7.6-12.2% |

Times shown are medians of two process means. Percentage ranges span process point estimates, not confidence interval endpoints. Reconstruction uses the suite's worst-loss pattern (first P shards marked missing).

The broad calibration covered 5+2, 10+4, 8+8 and 50+20 at 64 KiB/1 MiB, plus Update subsets. It was stopped during the first calibration order after identical code showed 23.67% variation in 10+4/1 MiB padded reconstruction. The required 24.67% margin exceeded the predeclared 10% ceiling. **No broad parent/current run or broad speedup claim was made.** The separately preselected 10+5 and 11+3 suite passed its own calibration. No retries or enlarged limits were used to turn the broad failure into a pass.

All 601 non-large native tests passed, and every completed benchmark process passed the 17,194,200-check correctness gate. The initial calibration overlapped the end of the test run and was discarded. A second partial calibration was stopped to fix construction-order handling in the temporary Update harness and add raw sample logging; it is also excluded. Both diagnostic attempts remain archived. Library source was not altered.

Focused telemetry peaked at 55.461 C; CPU4 sampled frequency was 2256000 kHz. No services, governor or cooling settings changed. Rock Canvas queue was temporarily paused and restored to unpaused.

Raw source snapshots, protocols, sample logs, binary hashes, test TRX, calibration and telemetry are retained in the local benchmark archive. SHA256 of `rockpi-evidence.tar.gz`: `62575c38a39e80e719090f7298c020c60098f977d0a5ac02279a16f7dc315e7c`.

The temporary remote workspace and transfer archive were removed after verifying the collected archive hash. No benchmark workers remain, the benchmark lock was released, and the Rock Canvas queue was restored to unpaused. No benchmark helper files were created in the checkout.
