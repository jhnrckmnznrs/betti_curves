# v23.5.2 blocked fast-interface-query confirmation

These files are the reviewed promotion evidence for the v23.6 native-U16 H2 fast interface-query default.

Design: slab depth 16, 5 blocks × 7 measured paired A/B runs, one warmup pair per block, alternating order, 10 s cooldowns, diagnostics off. The candidate won 29/35 wall-time pairs; median paired wall/local-sweep deltas were -1.518526%/-1.750983%; the exact two-sided wall sign-test p-value was 0.000116842. Four of five blocks and both run-order strata favored the candidate.

`perf stat` counters were unavailable because of the host kernel performance-monitoring policy; perf was pre-declared as corroborating rather than required evidence. See `v23_5_2_u16_h2_fast_interface_query_d16_blocked_decision.txt` and `docs/u16-h2-fast-interface-query.md`.
