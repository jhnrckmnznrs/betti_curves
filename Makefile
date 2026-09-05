.PHONY: format check test build bench-smoke reproduce prepare

format:
	cargo fmt --all

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	cargo test --all-targets --locked

test:
	cargo test --all-targets --locked
	python3 validation/test_oracles.py

build:
	cargo build --release --locked

bench-smoke: build
	python3 benchmarks/generate_synthetic_stack.py --output /tmp/stream-betti-smoke --shape 32 32 32 --dtype u8 --seed 42 --overwrite
	python3 benchmarks/run_suite.py /tmp/stream-betti-smoke --binary target/release/betti_curves --modes h0-scalar-stream h2-scalar-stream --slab-depths 4 8 --threads 1 2 --repeats 1 --output /tmp/stream-betti-smoke.csv

reproduce:
	scripts/reproduce_benchmarks.sh

prepare:
	scripts/prepare_for_commit.sh
