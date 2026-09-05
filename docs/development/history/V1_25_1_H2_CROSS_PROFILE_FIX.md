# v1.25.1 H2 cross-profile compatibility fix

This is a harness-only correction to v1.25.

The v1.24-derived disk hierarchical H2 combine/final log lines contain
`cross_retained=...` but do not carry an explicit `cross_storage=disk` tag.
The v1.25 direct path does carry `cross_storage=direct`.

The original v1.25 validator accidentally required the tag for both paths,
causing an exact disk/direct/in-memory persistence comparison to pass and then
fail during profile-accounting validation with `no hierarchical cross profile records`.

v1.25.1 treats an untagged hierarchical cross record as the legacy disk path,
while still requiring direct records to be explicitly tagged. The profiler uses
the same interpretation.

There are no Rust changes and no rebuild is required.
