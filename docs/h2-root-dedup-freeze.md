# H2 root-dedup freeze checkpoint

This checkpoint freezes the v18 H2 leaf optimization after a controlled same-binary A/B study on the CX09T1 274×274×448 stack at slab depth 16 with one H2 leaf worker.

## Decision

- **Enabled by default:** exact UF-root deduplication (`BETTI_HIER_H2_ROOT_DEDUP`, default on).
- **Disabled by default:** lazy shell-component pruning (`BETTI_HIER_H2_SHELL_PRUNE`, opt in with `=1`).

The root-dedup pass resolves the current roots of active face-neighbor candidates, keeps one candidate per distinct root in a fixed stack buffer, and sends only those unique components to union/action processing. Thus it removes edges that are already known to be connectivity no-ops.

The shell pass is mathematically exact but was not retained as a default performance optimization. Its local connectivity proof required hundreds of millions of additional shell-state queries on this benchmark, which outweighed the UF work it removed. It remains available for research/ablation use.

## Controlled benchmark

Median of three runs, same v18 binary, H2, slab depth 16, one leaf worker:

| Configuration | Compute (s) | Leaf stage (s) | Peak RSS (MiB) | Relative compute |
| --- | ---: | ---: | ---: | ---: |
| Reference: shell off, root dedup off | 5.324 | 4.267 | 180.3 | baseline |
| **Root dedup only** | **4.954** | **3.912** | **176.3** | **-6.95%** |
| Shell only | 7.454 | 6.381 | 176.2 | +40.0% |
| Shell + root dedup | 7.208 | 6.128 | 181.2 | +35.4% |

The audited combined run confirmed that root dedup eliminated already-connected union attempts after pruning: 33,634,020 union attempts and 33,634,020 successful unions. However, the shell stage required about 432 million extra shell-state checks, explaining its unfavorable runtime.

## Reproducible ablations

Default frozen behavior:

```bash
BETTI_HIER_H2_SHELL_PRUNE=0 BETTI_HIER_H2_ROOT_DEDUP=1 ...
```

Exact pre-v18 reference behavior:

```bash
BETTI_HIER_H2_SHELL_PRUNE=0 BETTI_HIER_H2_ROOT_DEDUP=0 ...
```

Experimental shell ablation:

```bash
BETTI_HIER_H2_SHELL_PRUNE=1 BETTI_HIER_H2_ROOT_DEDUP=0 ...
```

The no-environment-variable default is equivalent to shell pruning off and root deduplication on.
