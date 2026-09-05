use criterion::{criterion_group, criterion_main, Criterion};
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("criterion crate must live under benchmarks/criterion")
        .to_path_buf()
}

fn binary() -> PathBuf {
    std::env::var_os("BETTI_BENCH_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/release/betti_curves"))
}

fn fixture() -> PathBuf {
    std::env::var_os("BETTI_BENCH_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("tests/fixtures/shell_u8"))
}

fn run(mode: &str, output: &Path) {
    let status = Command::new(binary())
        .arg(fixture())
        .arg("1")
        .arg("6")
        .arg(mode)
        .arg(output)
        .status()
        .expect("failed to launch betti_curves");
    assert!(status.success());
}

fn cli_benchmarks(c: &mut Criterion) {
    let temp = std::env::temp_dir();
    c.bench_function("h0_scalar_stream_tiny", |b| {
        let output = temp.join("stream_betti_criterion_h0.csv");
        b.iter(|| run("h0-scalar-stream", &output));
    });
    c.bench_function("h2_scalar_stream_tiny", |b| {
        let output = temp.join("stream_betti_criterion_h2.csv");
        b.iter(|| run("h2-scalar-stream", &output));
    });
}

criterion_group!(benches, cli_benchmarks);
criterion_main!(benches);
