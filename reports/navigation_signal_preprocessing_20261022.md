# Exact RGB preprocessing acceleration, staged only

All three implementations were tested on the same fixed 63 TRAIN images and
236,284 candidate coordinates. Every patch byte and FP32/FP16 feature matched
the original. No candidate was dropped and no model was fitted during profiling.

| Implementation | Feature + classifier median | p95 | Probability agreement |
|---|---:|---:|---|
| Shared integer averaging, NumPy reduction | 343.35 ms | 489.84 ms | Exact |
| Shared averaging, compiled sequential CPU reduction | 49.09 ms | 65.80 ms | Exact |
| Shared averaging, sequential CUDA reduction + GPU trees | 22.87 ms | 30.63 ms | Max error 6.66e−16; no acceptance/phase changes |

The CPU compiled profile measured its original control at 972.83 ms median /
1371.20 ms p95. The CUDA profile measured its own original control at 1638.27 ms /
2374.49 ms. These are separate host-condition measurements; do not infer a precise
cross-run speedup or isolated hardware throughput from their ratio. CUDA peak
allocated tensors were 11.09 MB; native-kernel warm-up/compilation took 0.37 s.
CPU compiled warm-up took 0.31 s.

The optimized timings include per-image integer-average preparation and normalized
image transfer where applicable. CUDA head timing includes the FP16 feature boundary,
all trees in batches of 256 and returning probabilities to CPU. They exclude image
decode, neural proposer, candidate generation, three-camera scheduling, IPC, tracking,
the policy and simulator. These results do not qualify a 10 Hz driving pipeline.

Shared 3×3 averages preserve all alignment phases. Since nine integer pixels cannot
average to a half integer, `(sum+4)//9` reproduces the original mean-and-round step.
The original interleaved layout matters to NumPy's accumulation order. Compiled CPU
and CUDA implementations preserve sequential FP32 means and variances; fast math
and fused multiply-add are disabled, with correctly rounded GPU square roots.

The first compiled CPU process passed but a second process exposed an invalid
Numba disk-cache module identity for dynamically loaded staging code. Disk caching
was disabled and tests rerun before real-image profiling. Sources remain staged in
`patches/`; no deployed navigation module imports them. A later explicitly registered
TRAIN audit may use the exact CPU path with fresh independent parity checks.

**Metadata correction:** The CUDA request/assessment inherited
`concurrent_cpu_audit=true`, but the main audit had finished before this profile
started after interruption. There was brief independent-report work and late cache
preparation overlap, not an active main CPU audit. Original receipts are retained;
this report and the journal correct their concurrency annotation. The earlier
NumPy and compiled CPU profiles did run alongside the main audit.

Evidence: `outputs/navigation_signal_context_profile_20261022`,
`outputs/navigation_signal_context_fused_profile_20261022`,
`outputs/navigation_signal_context_cuda_profile_20261022`; their registered protocols
and per-image timing/equality receipts. No component-VAL or driving result changed.
