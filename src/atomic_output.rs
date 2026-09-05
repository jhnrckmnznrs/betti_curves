//! Transactional final-output files.
//!
//! A result is first written to a uniquely named file in the destination
//! directory. Calling commit flushes and synchronizes that file and then
//! renames it over the requested path. A failed or interrupted computation
//! therefore leaves the previous complete result in place and never presents
//! a partial CSV under the final name.

use anyhow::{Context, Result, bail};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(crate) struct AtomicOutput {
    writer: Option<BufWriter<File>>,
    temporary_path: PathBuf,
    final_path: PathBuf,
    committed: bool,
}

impl AtomicOutput {
    pub(crate) fn create(final_path: &Path) -> Result<Self> {
        let parent = final_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty());
        if let Some(parent) = parent
            && !parent.is_dir()
        {
            bail!("output directory does not exist: {parent:?}");
        }

        let file_name = final_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("output path has no file name: {final_path:?}"))?;
        let directory = parent.unwrap_or_else(|| Path::new("."));

        for _ in 0..128 {
            let sequence = OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut temporary_name = OsString::from(".");
            temporary_name.push(file_name);
            temporary_name.push(format!(".{}.{}.partial", std::process::id(), sequence));
            let temporary_path = directory.join(temporary_name);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary_path)
            {
                Ok(file) => {
                    return Ok(Self {
                        writer: Some(BufWriter::new(file)),
                        temporary_path,
                        final_path: final_path.to_path_buf(),
                        committed: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("could not create temporary output beside {final_path:?}")
                    });
                }
            }
        }
        bail!("could not choose a unique temporary output name beside {final_path:?}")
    }

    pub(crate) fn commit(mut self) -> Result<()> {
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| anyhow::anyhow!("output writer was already closed"))?;
        writer
            .flush()
            .with_context(|| format!("could not flush output {:?}", self.temporary_path))?;
        writer
            .get_ref()
            .sync_all()
            .with_context(|| format!("could not synchronize output {:?}", self.temporary_path))?;
        drop(writer);

        std::fs::rename(&self.temporary_path, &self.final_path).with_context(|| {
            format!(
                "could not atomically replace {:?} with completed output",
                self.final_path
            )
        })?;

        #[cfg(unix)]
        if let Some(parent) = self
            .final_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .with_context(|| format!("could not synchronize output directory {parent:?}"))?;
        }

        self.committed = true;
        Ok(())
    }
}

impl Write for AtomicOutput {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.writer
            .as_mut()
            .expect("atomic output writer is closed")
            .write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer
            .as_mut()
            .expect("atomic output writer is closed")
            .flush()
    }
}

impl Drop for AtomicOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temporary_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_path(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "betti_curves_atomic_{label}_{}_{}",
            std::process::id(),
            sequence
        ))
    }

    #[test]
    fn commit_replaces_the_final_file() {
        let path = test_path("commit");
        std::fs::write(&path, b"old").unwrap();
        let mut output = AtomicOutput::create(&path).unwrap();
        output.write_all(b"new").unwrap();
        output.commit().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn drop_preserves_the_previous_file() {
        let path = test_path("drop");
        std::fs::write(&path, b"complete").unwrap();
        {
            let mut output = AtomicOutput::create(&path).unwrap();
            output.write_all(b"partial").unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"complete");
        std::fs::remove_file(path).unwrap();
    }
}
