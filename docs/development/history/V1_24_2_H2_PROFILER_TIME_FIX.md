# v1.24.2 H2 profiler elapsed-time fix

This harness-only correction fixes GNU `/usr/bin/time -v` parsing in
`scripts/profile_v1_24_h2_hierarchical_stream.py`.

GNU time prints, for example:

```
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:11.92
```

The v1.24/v1.24.1 profiler used a lazy regular expression whose first possible
colon was the colon in the literal `h:mm:ss` label.  It therefore captured
`mm:ss or m:ss): 0:11.92` rather than `0:11.92`.

v1.24.2 anchors the match after the label's closing parenthesis and accepts
plain seconds, `m:ss[.ff]`, and `h:mm:ss[.ff]`.

There are no Rust implementation changes and no change to H2 persistence,
hierarchical composition, pruning, or profiling metrics.  A v1.24.1 release
binary can be reused without rebuilding.
