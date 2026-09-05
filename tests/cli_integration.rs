use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_PATH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_betti_curves")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn unique_test_path(prefix: &str, suffix: &str) -> PathBuf {
    let sequence = TEST_PATH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "stream_betti_curves_{prefix}_{}_{}{suffix}",
        std::process::id(),
        sequence
    ))
}

fn unique_output(name: &str) -> PathBuf {
    unique_test_path(name, ".csv")
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

fn run_branch_nodes(
    stack: &Path,
    slab: usize,
    connectivity: usize,
    mode: &str,
    node_file: &str,
) -> Vec<[String; 4]> {
    let workdir = unique_test_path(&format!("branch_{mode}"), "");
    fs::create_dir_all(&workdir).expect("could not create branch-tree test directory");
    let result = Command::new(binary())
        .arg(stack)
        .arg(slab.to_string())
        .arg(connectivity.to_string())
        .arg(mode)
        .current_dir(&workdir)
        .output()
        .expect("failed to run branch-tree mode");
    assert!(
        result.status.success(),
        "branch-tree command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(workdir.join(node_file)).expect("missing branch-tree node csv");
    let _ = fs::remove_dir_all(workdir);
    text.lines()
        .skip(1)
        .map(|line| {
            let mut fields = line.split(',');
            [
                fields.next().unwrap_or_default().to_owned(),
                fields.next().unwrap_or_default().to_owned(),
                fields.next().unwrap_or_default().to_owned(),
                fields.next().unwrap_or_default().to_owned(),
            ]
        })
        .collect()
}

fn canonicalize_branch_rows(mut rows: Vec<[String; 4]>, h2: bool) -> Vec<[String; 4]> {
    let original = rows.clone();
    for row in &mut rows {
        if row[1].is_empty() {
            continue;
        }
        let threshold_index = if h2 { 2 } else { 3 };
        let threshold = row[threshold_index].clone();
        let mut parent: usize = row[1].parse().expect("invalid branch parent");
        let mut steps = 0usize;
        while !original[parent][1].is_empty() && original[parent][threshold_index] == threshold {
            parent = original[parent][1]
                .parse()
                .expect("invalid ancestor parent");
            steps += 1;
            assert!(steps <= original.len(), "branch-tree parent cycle");
        }
        row[1] = parent.to_string();
    }
    rows
}

#[test]
fn hierarchical_h0_branch_tree_matches_plateau_canonical_flat_tree() {
    let stack = fixture("shell_u8");
    let flat = run_branch_nodes(&stack, 1, 6, "branch-tree-h0", "h0_merge_tree_nodes.csv");
    let hierarchical = run_branch_nodes(
        &stack,
        1,
        6,
        "branch-tree-h0-hierarchical",
        "h0_branch_tree_hierarchical_nodes.csv",
    );
    assert_eq!(canonicalize_branch_rows(flat, false), hierarchical);
}

#[test]
fn hierarchical_h2_branch_tree_matches_plateau_canonical_flat_tree() {
    let stack = fixture("shell_u8");
    let flat = run_branch_nodes(&stack, 1, 6, "branch-tree-h2", "h2_merge_tree_nodes.csv");
    let hierarchical = run_branch_nodes(
        &stack,
        1,
        6,
        "branch-tree-h2-hierarchical",
        "h2_branch_tree_hierarchical_nodes.csv",
    );
    assert_eq!(canonicalize_branch_rows(flat, true), hierarchical);
}
