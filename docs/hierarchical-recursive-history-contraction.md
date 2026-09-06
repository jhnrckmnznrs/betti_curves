# Recursive hierarchy-history contraction

The hierarchical branch-tree path can contract finalized zero-persistence history at every node, not only inside leaf slabs.

For a hierarchy node with child histories `L` and `R` and newly generated finalized history `P`, the optimization uses

```text
C(C(L) || C(R) || P)
```

instead of forwarding `C(L)`, `C(R)`, and every event in `P` to the root. Here `C` is the same online parent-repair/zero-persistence contraction used by the validated centralized reducer.

The required invariant is that once a branch is finalized inside a node it is absent from that node's exported boundary state. A parent node therefore cannot create a new future reference to that dying branch; it can only encounter retained references already generated inside the child subtree. Those references can be repaired before the child summary is returned.

H0 redirects retained references whose death occurs strictly after a positive parent's death, and also redirects equality for a zero-persistence parent. H2 uses the reversed filtration inequality. Outside is never indexed as a finite provisional parent.

The root still passes its retained positive history through the centralized contractor before tree replay. This is intentionally retained as a safety net and keeps the final replay representation unchanged.

Set `BETTI_HIER_RECURSIVE_HISTORY=0` to disable recursive internal-node contraction while keeping v11 leaf-local contraction enabled.
