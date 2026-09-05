# Releasing

The project follows Semantic Versioning.

## Version policy

- `0.x.y`: API/CLI may still evolve, but compatibility changes are documented.
- patch release: correctness/performance fixes without intended interface changes;
- minor release: new modes, material repository/tooling improvements, or new output capabilities;
- `1.0.0`: reserved for a stable CLI/output contract and a documented public benchmark baseline.

## Release checklist

1. Update `CHANGELOG.md` and `Cargo.toml`.
2. Ensure `Cargo.lock` has the same package version.
3. Run:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   python3 validation/test_oracles.py
   ```

4. Run the production equivalence gate on a representative F32 stack.
5. Run at least one public synthetic benchmark and archive the CSV.
6. Commit and tag:

   ```bash
   git tag -s v0.2.0 -m "stream_betti_curves v0.2.0"
   git push origin v0.2.0
   ```

7. GitHub Actions builds and attaches prebuilt Linux/macOS/Windows binaries to the tagged release.
8. Verify release checksums and download one artifact for a smoke test.
