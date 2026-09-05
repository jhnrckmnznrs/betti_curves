use anyhow::{Context, Result, bail};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static RUN_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique temporary directory for one disk-backed reduction.
///
/// The directory is removed on every ordinary error path through `Drop`.
/// Successful callers use `close` so a cleanup failure is reported.
#[derive(Debug)]
pub(crate) struct TempRunDirectory {
    path: PathBuf,
    active: bool,
}

impl TempRunDirectory {
    pub(crate) fn create(prefix: &str) -> Result<Self> {
        let base = std::env::var_os("BETTI_TEMP_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        fs::create_dir_all(&base)
            .with_context(|| format!("could not create temporary base directory {base:?}"))?;

        for _ in 0..10_000 {
            let sequence = RUN_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{prefix}_{}_{}", std::process::id(), sequence));

            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, active: true }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("could not create temporary run directory {path:?}")
                    });
                }
            }
        }

        bail!("could not allocate a unique temporary run directory in {base:?}")
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn close(mut self) -> Result<()> {
        let result = fs::remove_dir_all(&self.path)
            .with_context(|| format!("could not remove temporary run directory {:?}", self.path));
        if result.is_ok() {
            self.active = false;
        }
        result
    }
}

impl Drop for TempRunDirectory {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_are_unique_and_removed_on_drop() {
        let first = TempRunDirectory::create("betti_test").unwrap();
        let first_path = first.path().to_path_buf();
        let second = TempRunDirectory::create("betti_test").unwrap();
        let second_path = second.path().to_path_buf();

        assert_ne!(first_path, second_path);
        assert!(first_path.is_dir());
        assert!(second_path.is_dir());

        drop(first);
        drop(second);
        assert!(!first_path.exists());
        assert!(!second_path.exists());
    }
}
