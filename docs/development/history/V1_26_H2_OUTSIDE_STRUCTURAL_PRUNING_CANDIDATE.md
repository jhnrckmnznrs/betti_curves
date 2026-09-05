# v1.26 H2 outside-dominated structural pruning candidate

This candidate builds on the exact v1.25 direct-cross hierarchical H2 path.

## Promoted from v1.25

Hierarchical H2 now defaults to `--h2-hier-cross-storage direct`. On the full CX09T1 F32 fixture, direct cross preserved the exact 220,093-interval persistence multiset, reduced filesystem outputs at d8/d16/d32, improved median wall time at d8 and d16, and was neutral within run-to-run noise at d32.

## New ablation

`--h2-hier-outside-structural-pruning <off|outside-dominated>`

The default remains `off` until the v1.26 exactness/profile gate passes.

For a merge of two surviving H2 terminal components, the exported summary is canonicalized using the distinguished outside component:

- outside + outside: export no structural edge;
- outside + finite: export one Outside event on the finite terminal;
- finite + outside: export one Outside event on the finite terminal;
- finite + finite: export the original Interface edge.

The working union is still performed exactly as before. The change affects only the relative summary exported to the next fan-in level. In the augmented graph, outside-connected terminals are already mutually connected through the distinguished outside node, so the removed terminal-to-terminal edge carries no additional connectivity information.

The same rule is applied in optimized hierarchical leaves and intermediate fan-ins. Flat H2 and `off` remain reference oracles.

## Invariants

- F32 native32 keys remain 4 bytes end-to-end.
- Local births reuse the F32 input buffer.
- Local and hierarchical elder-dominated attach pruning remain enabled.
- Pairwise hierarchical UF remains packed/native32 at 8 bytes per pair node.
- Pair frontier remains at most four z-faces.
- Root remains terminal-free and is never materialized.
- Equal-value child processing remains outside -> attach -> interface -> cross.
- Direct cross is the hierarchical default.

## Acceptance

Run `validation/check_v1_26_h2_outside_structural_pruning.py`. Promotion requires exact persistence-multiset equality against the in-memory H2 oracle and the `off` hierarchical path at d4/d8/d16.

Then run `scripts/profile_v1_26_h2_outside_structural_pruning.py` on full CX09T1. Promotion should be based on total hierarchical summary bytes, filesystem outputs, wall/system time, and RSS, not on event count alone.
