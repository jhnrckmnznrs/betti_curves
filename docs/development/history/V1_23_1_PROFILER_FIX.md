# v1.23.1 attach-pruning profiler correction

The v1.23 pruning candidate can finalize some persistence intervals at an earlier hierarchy level than the reference path. This may change the order in which otherwise identical `(birth, death)` rows are appended to the output CSV.

The original v1.23 performance profiler used an order-sensitive SHA-256 of the emitted rows and could therefore report a false persistence mismatch. The dedicated equivalence validator already compared exact multisets with `collections.Counter` and was not affected.

v1.23.1 keeps the ordered digest for diagnostics and adds a canonical multiset digest formed from sorted `(birth, death, multiplicity)` records. Promotion decisions must use the canonical digest / exact Counter comparison.

There are no Rust changes in v1.23.1.
