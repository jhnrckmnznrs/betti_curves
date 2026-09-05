# GitHub repository migration

The public repository currently lives at `jhnrckmnznrs/betti_curves` on the `master` branch. The cleaned repository is prepared for the new name `stream_betti_curves`.

## Recommended migration

From an authenticated GitHub CLI checkout of the current repository:

```bash
gh repo rename stream_betti_curves --yes
git remote set-url origin git@github.com:jhnrckmnznrs/stream_betti_curves.git
```

Or use **Repository Settings → General → Repository name** and enter `stream_betti_curves`, then update the local remote.

The workflows in `.github/workflows/` accept both `master` and `main` for CI pushes. Renaming the default branch is optional and should be done separately from the repository rename so failures are easier to diagnose.

## Publish the cleaned tree

Before replacing the public tree:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets --locked
python3 validation/test_oracles.py
```

Then commit the repository cleanup as one reviewable change, push it, and let CI pass before tagging `v0.2.0`.

## First tagged release

After CI is green:

```bash
git tag -s v0.2.0 -m "stream_betti_curves v0.2.0"
git push origin v0.2.0
```

The release workflow will build and attach Linux, macOS, and Windows binaries plus SHA-256 files.
