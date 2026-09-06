# U16 H2 cached-parent fallback find (v23.3 experimental)

## Motivation

The v23.2 diagnostic on CX09T1 showed that the production `parent-shortcut` path performs roughly 36.1–36.6 million neighbor fallback finds per run. Most are one-hop finds. A two-hop precheck removed 2.41–2.95 million finds and 3.64–4.52 million parent traversals, but required 37.5–38.2 million extra second-parent checks and did not improve timing. More-than-two-hop same-root fallbacks were only 33,968–38,462 per run, so deeper unconditional prechecks have negligible headroom.

The v23.3 candidate therefore removes duplicated work instead of adding another check.

## Candidate

`--neighbor-root-check parent-cached-find`

For each root-carrying neighbor:

1. if the neighbor is `current_root`, return immediately;
2. load the neighbor's parent/root word once;
3. if that parent is `current_root`, return immediately;
4. if the neighbor itself is already a root, merge that known root without calling `find`;
5. otherwise continue the ordinary path-halving find from the already-known first parent.

`find_from_known_parent` performs the same parent-edge traversal and the same path-halving writes as `find(neighbor)`. The optimization only removes redundant initial reads/control work.

## Freeze gate

Do not promote the candidate unless all of the following hold:

- exact ordered H2 intervals match `parent-shortcut` at d16/d32/d64;
- the interval multiset is invariant across slab depths;
- the interleaved 11-pair benchmark shows a reproducible local-sweep improvement without material RSS regression;
- root dedup remains disabled in both A/B members.
