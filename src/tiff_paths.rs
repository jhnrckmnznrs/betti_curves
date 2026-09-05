use anyhow::{Context, Result, bail};
use std::cmp::Ordering;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn is_tiff_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("tif") || extension.eq_ignore_ascii_case("tiff")
        })
}

pub(crate) fn list_tiff_slices(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let entries = fs::read_dir(directory)
        .with_context(|| format!("could not read TIFF directory {directory:?}"))?;

    for entry in entries {
        let entry = entry.with_context(|| format!("could not read an entry in {directory:?}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("could not inspect directory entry {path:?}"))?;

        if (file_type.is_file() || (file_type.is_symlink() && path.is_file()))
            && is_tiff_path(&path)
        {
            files.push(path);
        }
    }

    files.sort_by(|left, right| {
        natural_name_cmp(
            left.file_name().unwrap_or_else(|| left.as_os_str()),
            right.file_name().unwrap_or_else(|| right.as_os_str()),
        )
        .then_with(|| left.cmp(right))
    });

    if files.is_empty() {
        bail!("no .tif or .tiff files found in {directory:?}");
    }

    Ok(files)
}

/// Recursively find every directory containing at least one TIFF slice.
///
/// Symlinked directories are deliberately not followed, which prevents directory
/// cycles and keeps every returned path underneath `root`.
pub(crate) fn find_tiff_stack_directories(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("batch root is not a directory: {root:?}");
    }

    let mut pending = vec![root.to_path_buf()];
    let mut stacks = Vec::new();

    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("could not read batch directory {directory:?}"))?;
        let mut contains_tiff = false;
        let mut children = Vec::new();

        for entry in entries {
            let entry =
                entry.with_context(|| format!("could not read an entry in {directory:?}"))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("could not inspect directory entry {path:?}"))?;

            if file_type.is_dir() {
                children.push(path);
            } else if (file_type.is_file() || (file_type.is_symlink() && path.is_file()))
                && is_tiff_path(&path)
            {
                contains_tiff = true;
            }
        }

        if contains_tiff {
            stacks.push(directory);
        }

        children.sort();
        pending.extend(children.into_iter().rev());
    }

    stacks.sort();
    Ok(stacks)
}

fn natural_name_cmp(left: &OsStr, right: &OsStr) -> Ordering {
    let left_lossy = left.to_string_lossy();
    let right_lossy = right.to_string_lossy();
    let left = left_lossy.as_bytes();
    let right = right_lossy.as_bytes();
    let mut left_index = 0usize;
    let mut right_index = 0usize;

    while left_index < left.len() && right_index < right.len() {
        if left[left_index].is_ascii_digit() && right[right_index].is_ascii_digit() {
            let left_end = digit_run_end(left, left_index);
            let right_end = digit_run_end(right, right_index);
            let left_significant = trim_leading_zeroes(&left[left_index..left_end]);
            let right_significant = trim_leading_zeroes(&right[right_index..right_end]);

            let number_order = left_significant
                .len()
                .cmp(&right_significant.len())
                .then_with(|| left_significant.cmp(right_significant))
                .then_with(|| (left_end - left_index).cmp(&(right_end - right_index)));
            if number_order != Ordering::Equal {
                return number_order;
            }

            left_index = left_end;
            right_index = right_end;
        } else {
            let byte_order = left[left_index].cmp(&right[right_index]);
            if byte_order != Ordering::Equal {
                return byte_order;
            }
            left_index += 1;
            right_index += 1;
        }
    }

    left.len().cmp(&right.len())
}

fn digit_run_end(bytes: &[u8], start: usize) -> usize {
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    end
}

fn trim_leading_zeroes(mut digits: &[u8]) -> &[u8] {
    while digits.len() > 1 && digits[0] == b'0' {
        digits = &digits[1..];
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let sequence = TEMP_DIRECTORY_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        std::env::temp_dir().join(format!(
            "betti_curves_paths_{}_{}",
            std::process::id(),
            sequence
        ))
    }

    #[test]
    fn natural_order_handles_unpadded_slice_numbers() {
        let mut names = [
            OsStr::new("slice10.tif"),
            OsStr::new("slice2.tif"),
            OsStr::new("slice1.tif"),
            OsStr::new("slice01.tif"),
        ];
        names.sort_by(|left, right| natural_name_cmp(left, right));

        assert_eq!(
            names,
            [
                OsStr::new("slice1.tif"),
                OsStr::new("slice01.tif"),
                OsStr::new("slice2.tif"),
                OsStr::new("slice10.tif"),
            ]
        );
    }

    #[test]
    fn recursively_finds_nested_tiff_stacks() {
        let root = temporary_directory();
        let first = root.join("ALN/CX09T1");
        let second = root.join("PBL/CX10C");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        File::create(first.join("slice1.tif")).unwrap();
        File::create(second.join("slice1.tiff")).unwrap();
        File::create(root.join("notes.txt")).unwrap();

        let stacks = find_tiff_stack_directories(&root).unwrap();
        assert_eq!(stacks, vec![first, second]);

        fs::remove_dir_all(root).unwrap();
    }
}
