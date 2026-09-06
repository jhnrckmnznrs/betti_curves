# U16 H2 plateau-zero persistence elision (v22 experimental)

The native-U16 hierarchical H2 persistence profile showed roughly 28.8 million
finite zero-persistence pairs on CX09T1 at slab depth 16. Before v22 those
pairs were materialized as `FinitePair`/`LocalPersistenceAction` values and
then discarded by the local event filter because `birth == death`.

v22 performs that equality test at the union decision instead. When the
finite child has `birth == merge_value`, the union-find state transition is
unchanged but no finite-pair object or final-pair action is constructed.
Positive finite pairs, attach events, interface merges, and Outside events are
unchanged.

The optimization is enabled by default for the native-U16 hierarchical H2
path. Set `BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION=0` to restore the v21
materialize-and-filter behavior in the same binary.

Correctness follows because the old path never emitted `[t,t)` intervals: it
materialized them and immediately filtered them. v22 moves that same filter
earlier without changing the elder-rule union, surviving birth state,
interface representative, or Outside state.
