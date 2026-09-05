use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_betti_curves")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn unique_output(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "stream_betti_curves_{name}_{}_{}.csv",
        std::process::id(),
        nonce
    ))
}

fn run(stack: &Path, slab: usize, connectivity: usize, mode: &str, extra: &[&str]) -> Vec<String> {
    let output = unique_output(mode);
    let result = Command::new(binary())
        .arg(stack)
        .arg(slab.to_string())
        .arg(connectivity.to_string())
        .arg(mode)
        .arg(&output)
        .args(extra)
        .output()
        .expect("failed to run betti_curves");
    assert!(
        result.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(&output).expect("missing output csv");
    let _ = fs::remove_file(output);
    text.lines().skip(1).map(str::to_owned).collect()
}

fn canonical(mut rows: Vec<String>) -> Vec<String> {
    rows.sort();
    rows
}

#[test]
fn version_reports_package_version() {
    let result = Command::new(binary())
        .arg("--version")
        .output()
        .expect("failed to run --version");
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn constant_volume_has_one_essential_h0_class() {
    let rows = run(&fixture("constant_u8"), 1, 6, "h0-scalar-stream", &[]);
    assert_eq!(canonical(rows), vec!["0,inf"]);
}

#[test]
fn shell_volume_has_one_h2_interval() {
    let rows = run(&fixture("shell_u8"), 1, 6, "h2-scalar-stream", &[]);
    assert_eq!(canonical(rows), vec!["0,1"]);
}

#[test]
fn hierarchical_f32_h0_matches_flat_stream() {
    let stack = fixture("shell_f32");
    let flat = run(&stack, 1, 6, "h0-scalar-stream", &[]);
    let hierarchical = run(
        &stack,
        1,
        6,
        "h0-scalar-hierarchical-stream",
        &[
            "--f32-key-mode",
            "native32",
            "--h0-birth-buffer",
            "reuse-input",
            "--h0-event-storage",
            "direct",
            "--global-h0-uf-layout",
            "packed",
            "--h0-hier-attach-pruning",
            "elder-dominated",
        ],
    );
    assert_eq!(canonical(flat), canonical(hierarchical));
}

#[test]
fn hierarchical_f32_h2_matches_flat_stream() {
    let stack = fixture("shell_f32");
    let flat = run(&stack, 1, 6, "h2-scalar-stream", &[]);
    let hierarchical = run(
        &stack,
        1,
        6,
        "h2-scalar-hierarchical-stream",
        &[
            "--f32-key-mode",
            "native32",
            "--local-h2-birth-state",
            "compact",
            "--global-h2-birth-state",
            "compact",
            "--global-h2-uf-layout",
            "packed",
            "--h2-hier-cross-storage",
            "direct",
            "--h2-hier-outside-structural-pruning",
            "outside-dominated",
        ],
    );
    assert_eq!(canonical(flat), canonical(hierarchical));
}
