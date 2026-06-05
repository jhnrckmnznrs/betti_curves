use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn write_sparse_betti0_csv(path: &Path, results: &[(u16, i64)]) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "threshold,betti0")?;

    for &(threshold, beta0) in results {
        writeln!(file, "{},{}", threshold, beta0)?;
    }

    Ok(())
}

pub fn write_sparse_betti2_csv(path: &Path, results: &[(u16, i64)]) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "threshold,betti2")?;

    for &(threshold, beta2) in results {
        writeln!(file, "{},{}", threshold, beta2)?;
    }

    Ok(())
}

pub fn compress_changes(results: &[(u16, i64)]) -> Vec<(u16, i64)> {
    let mut compressed: Vec<(u16, i64)> = Vec::new();
    let mut previous: Option<i64> = None;

    for &(threshold, value) in results {
        if previous != Some(value) {
            compressed.push((threshold, value));
            previous = Some(value);
        }
    }

    compressed
}
