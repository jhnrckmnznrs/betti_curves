# U16 H2 structural sweep profiling (v23.4 experimental)

## Motivation

Three U16 H2 union-side experiments have now shown diminishing or negative returns on CX09T1:

- root deduplication removed many nominal union calls but increased runtime;
- the two-hop parent shortcut avoided millions of real finds but paid for too many speculative parent reads;
- cached-parent fallback reuse removed only zero-hop finds and did not improve the fastest d16 configuration.

The next optimization should therefore be chosen from measured ownership of the local-sweep cost rather than from another union-find micro-hypothesis.

## Production reference

The structural profiler fixes the current production/reference leaf configuration:

- slab depth: 16;
- native U16 persistence keys: on;
- plateau-zero elision: on;
- local H2 birth storage: compact;
- global H2 birth storage: compact;
- global H2 UF layout: packed;
- hierarchical cross storage: direct;
- hierarchical outside structural pruning: off;
- neighbor root check: `parent-shortcut`;
- native-U16 root deduplication: off.

No new optimization is enabled by v23.4.

## Sampled timing regions

`--sweep-diagnostics` enables deterministic hashed sampling at an expected rate of 1/1024 voxels. The hash is applied to the global linear voxel index so the sample does not line up periodically with rows, slices, slab boundaries, or filtration plateaus.

For sampled voxels, three contiguous regions are timed:

1. **activation/boundary/interface bookkeeping** — activate the new voxel, update the active state, determine slab-face/interface identity, mark global outside state, and emit immediate boundary Outside events;
2. **neighborhood/pruning** — construct the active-neighbor mask, consult/compute the neighborhood-component pruning result, and obtain representative neighbors;
3. **union/persistence/events** — process the representative neighbors through the current root-carrying union kernel, update persistence state, and emit local persistence events.

The profiler reports sampled nanoseconds and per-sampled-voxel averages. These sampled timings are diagnostic shares, not benchmark timings: `Instant` calls and sampling branches intentionally perturb the diagnostic run slightly. Production timing comparisons must continue to use runs with sweep diagnostics disabled.

## Exact counters

The same diagnostic run reports exact totals for:

- total/interior-fast/boundary voxels;
- pruning-mask calls;
- active-state checks and active-neighbor hits;
- representative visits;
- pruning-cache hits and misses;
- component-mask computations;
- union attempts, successful unions, and same-root unions;
- direct-parent checks/hits and fallback-find counts;
- interface representative queries/writes and interface-related unions;
- positive final pairs, attach events, outside events, interface-merge events, and zero-persistence pairs elided.

These exact counters should be used to explain the sampled timing result before selecting the next optimization.

## Decision rule

The first v23.4 run intentionally uses only d16/CX09T1. The next candidate should target the dominant sampled region:

- if neighborhood/pruning dominates, inspect mask construction, cache locality/hit rate, and representative generation;
- if union/persistence/events dominates, separate persistence/action/event storage from core UF work before changing UF mechanics again;
- if activation/boundary bookkeeping is unexpectedly large, inspect active-state layout and slab/global boundary/interface bookkeeping;
- if no region clearly dominates, use the exact counters and a second-stage split rather than adding another speculative shortcut.

The v23.4 diagnostic itself is not a production optimization and should not change persistence output.
