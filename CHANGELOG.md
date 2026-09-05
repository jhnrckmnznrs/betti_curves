# Changelog

All notable user-facing changes are documented here. This project follows [Semantic Versioning](https://semver.org/) and the structure of [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed

- GitHub-readiness gate fixes: native-F32 H2 test widening, targeted Clippy annotations for tuning-heavy kernels, and a named hierarchical H2 combine result type.
- Added `scripts/prepare_for_commit.sh` so local validation formats the tree before enforcing `cargo fmt --check`.

### Planned
- Publish thread-scaling measurements with explicit `RAYON_NUM_THREADS` capture.
- Run the 64/128-slice real-volume H2 prefix benchmark for d8 and d16.
- Publish tagged multi-platform binaries after the GitHub repository migration.

## [0.2.0] - 2026-09-05

### Added
- Disk-backed hierarchical H0 persistence with bounded pairwise interface state.
- Elder-dominated H0 attach finalization and terminal-free root reduction.
- Native 4-byte F32 key path and packed global H0 union-find state.
- Outside-aware hierarchical H2 persistence with compact births and packed state.
- Elder-dominated H2 attach pruning, direct cross-interface consumption, and outside-dominated structural pruning.
- Reproducible benchmark data, scaling plots, deterministic synthetic-data generation, and Python baseline tooling.
- GitHub Actions CI, benchmark automation, and tagged multi-platform release workflow.
- End-to-end CLI integration-test fixtures and reviewer-facing architecture documentation.
- Criterion process-level benchmark harness and Linux flamegraph helper.

### Changed
- Repository/package identity renamed from `betti_curves` to `stream_betti_curves`.
- Public presentation reframed around memory-efficient exact processing of large 3-D image volumes.
- Internal optimization notes moved out of the repository root into `docs/development/`.
- The executable remains named `betti_curves` for compatibility.

### Performance
- H0 real F32 prefix: 3792×3792×128 processed at 1.66 Mvox/s with 2.04 GiB peak RSS and 4.22 GiB peak scratch in the recorded d8 hierarchical run.
- H2 CX09T1 d16: hierarchical direct/outside-pruned path measured at 3.54 s and 20.4 MiB peak RSS versus 5.81 s and 43.2 MiB for the corresponding flat path.

## [0.1.0] - Prior public snapshot

### Added
- Initial public Rust implementation for Betti-0 and Betti-2 curves on TIFF stacks.
- Threshold-wise and event-based slabwise algorithms.

[Unreleased]: https://github.com/jhnrckmnznrs/stream_betti_curves/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jhnrckmnznrs/stream_betti_curves/releases/tag/v0.2.0
[0.1.0]: https://github.com/jhnrckmnznrs/stream_betti_curves/releases/tag/v0.1.0
