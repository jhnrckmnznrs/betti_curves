#![deny(unsafe_code)]
#![allow(clippy::uninlined_format_args)]

#[allow(unsafe_code)]
mod allocator_trim;
mod atomic_output;
mod betti0;
mod betti2;
mod binary_io;
mod connectivity;
mod csv;
mod event_betti0;
mod event_betti0_scalar_stream;
mod event_betti0_stream;
mod event_betti2;
mod event_betti2_scalar_stream;
mod event_betti2_stream;
mod event_scalar_common;
mod interface_sparsify;
mod interface_sparsify_scalar;
mod io;
mod io_scalar;
mod local_pruning;
mod local_uf_state;
mod memory_audit;
mod merge_tree_common;
mod merge_tree_h0;
mod merge_tree_h0_stream;
mod merge_tree_h2;
mod merge_tree_h2_stream;
mod merge_tree_hierarchical;
mod persistence_h0;
mod persistence_h0_scalar;
mod persistence_h0_scalar_stream;
mod persistence_h0_stream;
mod persistence_h2;
mod persistence_h2_scalar;
mod persistence_h2_scalar_stream;
mod persistence_h2_stream;
mod scalar;
mod scalar_order;
mod scalar_stream_tuning;
mod size_plan;
mod slab_interface;
mod temp_runs;
mod tiff_paths;
mod union_find;

use anyhow::{Context, Result};
use betti0::compute_sparse_global_betti0_parallel;
use betti2::compute_sparse_betti2_curve_parallel;
use connectivity::{Mode, dual_connectivity, parse_connectivity, parse_mode};
use csv::{compress_changes, write_sparse_betti0_csv, write_sparse_betti2_csv};
use event_betti0::{compute_event_based_betti0_zslabs, write_event_betti0_csv};
use event_betti0_scalar_stream::compute_event_based_betti0_scalar_stream_zslabs;
use event_betti0_stream::{
    compute_event_based_betti0_stream_zslabs, write_event_betti0_stream_csv,
};
use event_betti2::{compute_event_based_betti2_zslabs, write_event_betti2_csv};
use event_betti2_scalar_stream::compute_event_based_betti2_scalar_stream_zslabs;
use event_betti2_stream::{
    compute_event_based_betti2_stream_zslabs, write_event_betti2_stream_csv,
};
use io::{TiffStackReader, collect_unique_values_by_slabs, print_volume_info};
use io_scalar::{ScalarTiffStackReader, print_scalar_volume_info};
use merge_tree_common::{write_merge_tree_csv, write_merge_tree_nodes_csv};
use merge_tree_h0::compute_h0_merge_tree_zslabs;
use merge_tree_h0_stream::compute_h0_merge_tree_stream_zslabs;
use merge_tree_h2::compute_h2_merge_tree_zslabs;
use merge_tree_h2_stream::compute_h2_merge_tree_stream_zslabs;
use merge_tree_hierarchical::{
    compute_h0_merge_tree_hierarchical_zslabs, compute_h2_merge_tree_hierarchical_zslabs,
};
use persistence_h0::{
    betti0_curve_from_h0_intervals, compute_h0_persistence_zslabs, write_h0_persistence_csv,
};
use persistence_h0_scalar::{
    betti0_curve_from_scalar_h0_intervals, compute_h0_persistence_scalar_hierarchical_zslabs,
    compute_h0_persistence_scalar_zslabs, compute_h0_scalar_batch, write_h0_scalar_persistence_csv,
    write_scalar_betti0_curve_csv,
};
use persistence_h0_scalar_stream::{
    compute_h0_persistence_scalar_hierarchical_stream_zslabs,
    compute_h0_persistence_scalar_stream_zslabs, compute_h0_scalar_stream_batch,
};
use persistence_h0_stream::compute_h0_persistence_stream_zslabs;
use persistence_h2::{
    betti2_curve_from_h2_intervals, compute_h2_persistence_zslabs, write_h2_persistence_csv,
};
use persistence_h2_scalar::{
    betti2_curve_from_scalar_h2_intervals, compute_h2_persistence_scalar_zslabs,
    compute_h2_scalar_batch, write_h2_scalar_persistence_csv, write_scalar_betti2_curve_csv,
};
use persistence_h2_scalar_stream::{
    compute_h2_persistence_scalar_hierarchical_stream_zslabs,
    compute_h2_persistence_scalar_stream_zslabs, compute_h2_scalar_stream_batch,
};
use persistence_h2_stream::compute_h2_persistence_stream_zslabs;
use scalar_stream_tuning::{
    ActiveStateStrategy, EventOrderStrategy, F32KeyMode, GlobalH0UnionFindLayoutStrategy,
    GlobalH2BirthStateStrategy, GlobalH2UnionFindLayoutStrategy, H0BirthBufferStrategy,
    H0EventStorageStrategy, H0HierAttachPruningStrategy, H0PruningCacheStrategy,
    H2HierCrossStorageStrategy, H2HierOutsideStructuralPruningStrategy, InterfaceOrderStrategy,
    InterfaceStateStrategy, LocalH2BirthStateStrategy, MergeStrategy, NeighborKernelStrategy,
    NeighborRootCheckStrategy, PhaseTrimStrategy, RepresentativeActiveCheckStrategy,
    ScalarStreamTuning, UnionFindLayoutStrategy, UnionKernelStrategy,
};
use size_plan::SizePlan;
use std::env;
use std::path::Path;
use std::time::Instant;

fn print_usage() {
    eprintln!("Usage:");
    eprintln!(
        "  cargo run --release -- <directory_of_tiff_slices> \
         [slab_depth] [foreground_connectivity] [mode] [output_path] [--verbose] [--dry-run-plan]"
    );
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  betti0               threshold-wise Betti-0");
    eprintln!("  betti2               threshold-wise Betti-2");
    eprintln!("  both                 threshold-wise Betti-0 and Betti-2");
    eprintln!("  event-betti0         in-memory event-based Betti-0");
    eprintln!("  event-betti0-stream  disk-backed event-based Betti-0");
    eprintln!("  event-betti0-scalar-stream exact scalar disk-backed Betti-0 curve");
    eprintln!("  event-betti2         in-memory event-based Betti-2");
    eprintln!("  event-betti2-stream  disk-backed event-based Betti-2");
    eprintln!("  event-betti2-scalar-stream exact scalar disk-backed Betti-2 curve");
    eprintln!("  h0                   in-memory slabwise H0 persistence");
    eprintln!("  h0-stream            disk-backed slabwise H0 persistence");
    eprintln!("  h0-scalar            exact U8/U16/F32/F64 H0 persistence");
    eprintln!("  h0-scalar-hierarchical experimental exact hierarchical H0 persistence");
    eprintln!(
        "  h0-scalar-hierarchical-stream experimental optimized disk-backed hierarchical scalar H0 persistence (U8/U16/F32/F64)"
    );
    eprintln!("  h0-scalar-stream     disk-backed exact scalar H0 persistence");
    eprintln!("  h0-scalar-batch      in-memory recursive batch processing");
    eprintln!("  h0-scalar-batch-stream disk-backed recursive batch processing");
    eprintln!("  h2                   in-memory slabwise H2 persistence");
    eprintln!("  h2-stream            disk-backed slabwise H2 persistence");
    eprintln!("  h2-scalar            exact U8/U16/F32/F64 H2 persistence");
    eprintln!("  h2-scalar-batch      in-memory recursive batch processing");
    eprintln!("  h2-scalar-stream     disk-backed exact scalar H2 persistence");
    eprintln!(
        "  h2-scalar-hierarchical-stream experimental optimized outside-aware hierarchical scalar H2 persistence (U8/U16/F32/F64)"
    );
    eprintln!("  h2-scalar-batch-stream disk-backed recursive batch processing");
    eprintln!("  branch-tree-h0        in-memory H0 elder-rule branch tree");
    eprintln!("  branch-tree-h0-hierarchical bounded-interface H0 plateau-canonical branch tree");
    eprintln!("  branch-tree-h0-stream disk-backed H0 elder-rule branch tree");
    eprintln!("  branch-tree-h2        in-memory H2 elder-rule branch tree");
    eprintln!("  branch-tree-h2-hierarchical bounded-interface H2 plateau-canonical branch tree");
    eprintln!("  branch-tree-h2-stream disk-backed H2 elder-rule branch tree");
    eprintln!("  (legacy merge-tree-* spellings remain accepted)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --verbose, -v       show threshold-scan progress");
    eprintln!(
        "  --dry-run-plan      validate dimensions and print the size plan without computing"
    );
    eprintln!("  Scalar persistence stream tuning/ablation options:");
    eprintln!(
        "  --merge-strategy <scan|heap|auto>         default: scan (auto selects by reader fan-in)"
    );
    eprintln!("  --interface-order <comparison|radix>      default: radix");
    eprintln!("  --event-order <resort|verify>              default: verify");
    eprintln!("  --f32-key-mode <legacy64|native32>         default: native32");
    eprintln!("  --neighbor-kernel <generic|interior-fast>   default: interior-fast");
    eprintln!("  --representative-active-check <recheck|trust-pruner> default: recheck");
    eprintln!("  --union-kernel <conventional|root-carrying> default: root-carrying");
    eprintln!(
        "  --neighbor-root-check <find|parent-shortcut|parent-two-hop|parent-cached-find> default: parent-shortcut (H2 root-carrying only)"
    );
    eprintln!("  --h0-pruning-cache <off|4k|16k|64k|256k>  default: 64k (H0 26-connectivity only)");
    eprintln!(
        "  --h0-birth-buffer <copy|reuse-input>        H0 F32 local birth storage ablation; default: copy"
    );
    eprintln!(
        "  --h0-event-storage <buffered|direct>         H0 F32 local event storage ablation; default: buffered"
    );
    eprintln!(
        "  --h0-hier-attach-pruning <off|elder-dominated> hierarchical H0 attach propagation ablation; default: off"
    );
    eprintln!(
        "  --active-state <separate|parent-sentinel>   defaults: H0=separate, H2=parent-sentinel"
    );
    eprintln!("  --interface-state <vector|root-invariant>   default: root-invariant");
    eprintln!("  --uf-layout <parent-rank|packed>            default: packed");
    eprintln!(
        "  --phase-trim <off|before-reduce>            H2 default: before-reduce on Linux/glibc, otherwise off"
    );
    eprintln!(
        "  --local-h2-birth-state <tagged|compact>     H2 local birth storage; default: compact (tagged=reference)"
    );
    eprintln!(
        "  --global-h0-uf-layout <parent-rank|packed>    H0 global UF layout; default: packed (parent-rank=reference)"
    );
    eprintln!(
        "  --global-h2-birth-state <tagged|compact>    H2 global birth storage; default: compact"
    );
    eprintln!(
        "  --global-h2-uf-layout <parent-rank|packed>    H2 global UF ablation; default: parent-rank"
    );
    eprintln!(
        "  --h2-hier-cross-storage <disk|direct>          hierarchical H2 cross-interface storage ablation; default: direct"
    );
    eprintln!(
        "  --h2-hier-outside-structural-pruning <off|outside-dominated> hierarchical H2 structural summary pruning; default: off"
    );
    eprintln!("  --sweep-diagnostics                         emit local-sweep operation counters");
    eprintln!(
        "  --h2-memory-audit                           emit H2 phase RSS/HWM and vector-capacity audit"
    );
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  cargo run --release -- /data/volume 32 26 event-betti0");
    eprintln!("  cargo run --release -- /data/volume 32 26 event-betti0-stream");
    eprintln!(
        "  cargo run --release -- /data/float_volume 32 26 event-betti0-scalar-stream output.csv"
    );
    eprintln!("  cargo run --release -- /data/volume 32 26 event-betti2");
    eprintln!("  cargo run --release -- /data/volume 32 26 event-betti2-stream");
    eprintln!(
        "  cargo run --release -- /data/float_volume 32 26 event-betti2-scalar-stream output.csv"
    );
    eprintln!("  cargo run --release -- /data/volume 32 26 h0");
    eprintln!("  cargo run --release -- /data/volume 32 26 h0-stream");
    eprintln!("  cargo run --release -- /data/float_volume 32 26 h0-scalar");
    eprintln!("  cargo run --release -- /data/float_volume 32 26 h0-scalar-stream output.csv");
    eprintln!("  cargo run --release -- /data/datasets 32 26 h0-scalar-batch results");
    eprintln!("  cargo run --release -- /data/datasets 32 26 h0-scalar-batch-stream results");
    eprintln!("  cargo run --release -- /data/volume 32 26 h2");
    eprintln!("  cargo run --release -- /data/volume 32 26 h2-stream");
    eprintln!("  cargo run --release -- /data/float_volume 32 26 h2-scalar");
    eprintln!("  cargo run --release -- /data/datasets 32 26 h2-scalar-batch results");
    eprintln!("  cargo run --release -- /data/float_volume 32 26 h2-scalar-stream output.csv");
    eprintln!("  cargo run --release -- /data/datasets 32 26 h2-scalar-batch-stream results");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h0");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h0-hierarchical");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h0-stream");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h2");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h2-hierarchical");
    eprintln!("  cargo run --release -- /data/volume 32 26 branch-tree-h2-stream");
}

fn collect_threshold_values(
    volume: &TiffStackReader,
    slab_depth: usize,
    verbose: bool,
) -> Result<Vec<u16>> {
    println!("Collecting unique intensity values...");
    let scan_start = Instant::now();

    let values = collect_unique_values_by_slabs(volume, slab_depth, verbose)?;

    println!(
        "Unique-value scan took {:.3} seconds",
        scan_start.elapsed().as_secs_f64()
    );
    println!("Found {} unique intensity values", values.len());

    Ok(values)
}

#[derive(Debug, Clone, Copy)]
struct RunOptions {
    verbose: bool,
    dry_run_plan: bool,
    scalar_tuning: ScalarStreamTuning,
    scalar_tuning_overridden: bool,
    active_state_overridden: bool,
    phase_trim_overridden: bool,
}

fn split_run_arguments(args: &[String]) -> Result<(Vec<&String>, RunOptions)> {
    let mut positional = Vec::new();
    let mut options = RunOptions {
        verbose: false,
        dry_run_plan: false,
        scalar_tuning: ScalarStreamTuning::default(),
        scalar_tuning_overridden: false,
        active_state_overridden: false,
        phase_trim_overridden: false,
    };
    let mut options_ended = false;
    let mut index = 1usize;

    while index < args.len() {
        let argument = &args[index];
        if !options_ended {
            match argument.as_str() {
                "--" => {
                    options_ended = true;
                    index += 1;
                    continue;
                }
                "--verbose" | "-v" => {
                    options.verbose = true;
                    index += 1;
                    continue;
                }
                "--dry-run-plan" => {
                    options.dry_run_plan = true;
                    index += 1;
                    continue;
                }
                "--merge-strategy" => {
                    index += 1;
                    let value = args
                        .get(index)
                        .ok_or_else(|| anyhow::anyhow!("--merge-strategy requires scan or heap"))?;
                    options.scalar_tuning.merge_strategy = MergeStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--interface-order" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--interface-order requires comparison or radix")
                    })?;
                    options.scalar_tuning.interface_order = InterfaceOrderStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--event-order" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--event-order requires resort or verify")
                    })?;
                    options.scalar_tuning.event_order = EventOrderStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--f32-key-mode" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--f32-key-mode requires legacy64 or native32")
                    })?;
                    options.scalar_tuning.f32_key_mode = F32KeyMode::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--neighbor-kernel" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--neighbor-kernel requires generic or interior-fast")
                    })?;
                    options.scalar_tuning.neighbor_kernel = NeighborKernelStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--representative-active-check" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!(
                            "--representative-active-check requires recheck or trust-pruner"
                        )
                    })?;
                    options.scalar_tuning.representative_active_check =
                        RepresentativeActiveCheckStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--union-kernel" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--union-kernel requires conventional or root-carrying")
                    })?;
                    options.scalar_tuning.union_kernel = UnionKernelStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--neighbor-root-check" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!(
                            "--neighbor-root-check requires find, parent-shortcut, parent-two-hop, or parent-cached-find"
                        )
                    })?;
                    options.scalar_tuning.neighbor_root_check =
                        NeighborRootCheckStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h0-pruning-cache" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--h0-pruning-cache requires off, 4k, 16k, 64k, or 256k")
                    })?;
                    options.scalar_tuning.h0_pruning_cache = H0PruningCacheStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--active-state" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--active-state requires separate or parent-sentinel")
                    })?;
                    options.scalar_tuning.active_state = ActiveStateStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    options.active_state_overridden = true;
                    index += 1;
                    continue;
                }
                "--interface-state" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--interface-state requires vector or root-invariant")
                    })?;
                    options.scalar_tuning.interface_state = InterfaceStateStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--uf-layout" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--uf-layout requires parent-rank or packed")
                    })?;
                    options.scalar_tuning.uf_layout = UnionFindLayoutStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--phase-trim" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--phase-trim requires off or before-reduce")
                    })?;
                    options.scalar_tuning.phase_trim = PhaseTrimStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    options.phase_trim_overridden = true;
                    index += 1;
                    continue;
                }
                "--local-h2-birth-state" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--local-h2-birth-state requires tagged or compact")
                    })?;
                    options.scalar_tuning.local_h2_birth_state =
                        LocalH2BirthStateStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h0-birth-buffer" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--h0-birth-buffer requires copy or reuse-input")
                    })?;
                    options.scalar_tuning.h0_birth_buffer = H0BirthBufferStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h0-event-storage" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--h0-event-storage requires buffered or direct")
                    })?;
                    options.scalar_tuning.h0_event_storage = H0EventStorageStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h0-hier-attach-pruning" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--h0-hier-attach-pruning requires off or elder-dominated")
                    })?;
                    options.scalar_tuning.h0_hier_attach_pruning =
                        H0HierAttachPruningStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--global-h0-uf-layout" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--global-h0-uf-layout requires parent-rank or packed")
                    })?;
                    options.scalar_tuning.global_h0_uf_layout =
                        GlobalH0UnionFindLayoutStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h2-hier-cross-storage" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--h2-hier-cross-storage requires disk or direct")
                    })?;
                    options.scalar_tuning.h2_hier_cross_storage =
                        H2HierCrossStorageStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h2-hier-outside-structural-pruning" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!(
                            "--h2-hier-outside-structural-pruning requires off or outside-dominated"
                        )
                    })?;
                    options.scalar_tuning.h2_hier_outside_structural_pruning =
                        H2HierOutsideStructuralPruningStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--global-h2-birth-state" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--global-h2-birth-state requires tagged or compact")
                    })?;
                    options.scalar_tuning.global_h2_birth_state =
                        GlobalH2BirthStateStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--global-h2-uf-layout" => {
                    index += 1;
                    let value = args.get(index).ok_or_else(|| {
                        anyhow::anyhow!("--global-h2-uf-layout requires parent-rank or packed")
                    })?;
                    options.scalar_tuning.global_h2_uf_layout =
                        GlobalH2UnionFindLayoutStrategy::parse(value)?;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--sweep-diagnostics" => {
                    options.scalar_tuning.sweep_diagnostics = true;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                "--h2-memory-audit" => {
                    options.scalar_tuning.h2_memory_audit = true;
                    options.scalar_tuning_overridden = true;
                    index += 1;
                    continue;
                }
                _ => {}
            }

            if let Some(value) = argument.strip_prefix("--merge-strategy=") {
                options.scalar_tuning.merge_strategy = MergeStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--interface-order=") {
                options.scalar_tuning.interface_order = InterfaceOrderStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--event-order=") {
                options.scalar_tuning.event_order = EventOrderStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--f32-key-mode=") {
                options.scalar_tuning.f32_key_mode = F32KeyMode::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--neighbor-kernel=") {
                options.scalar_tuning.neighbor_kernel = NeighborKernelStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--representative-active-check=") {
                options.scalar_tuning.representative_active_check =
                    RepresentativeActiveCheckStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--union-kernel=") {
                options.scalar_tuning.union_kernel = UnionKernelStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--neighbor-root-check=") {
                options.scalar_tuning.neighbor_root_check =
                    NeighborRootCheckStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h0-pruning-cache=") {
                options.scalar_tuning.h0_pruning_cache = H0PruningCacheStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--active-state=") {
                options.scalar_tuning.active_state = ActiveStateStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                options.active_state_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--interface-state=") {
                options.scalar_tuning.interface_state = InterfaceStateStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--uf-layout=") {
                options.scalar_tuning.uf_layout = UnionFindLayoutStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--phase-trim=") {
                options.scalar_tuning.phase_trim = PhaseTrimStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                options.phase_trim_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--local-h2-birth-state=") {
                options.scalar_tuning.local_h2_birth_state =
                    LocalH2BirthStateStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h0-birth-buffer=") {
                options.scalar_tuning.h0_birth_buffer = H0BirthBufferStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h0-event-storage=") {
                options.scalar_tuning.h0_event_storage = H0EventStorageStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h0-hier-attach-pruning=") {
                options.scalar_tuning.h0_hier_attach_pruning =
                    H0HierAttachPruningStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--global-h0-uf-layout=") {
                options.scalar_tuning.global_h0_uf_layout =
                    GlobalH0UnionFindLayoutStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h2-hier-cross-storage=") {
                options.scalar_tuning.h2_hier_cross_storage =
                    H2HierCrossStorageStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--h2-hier-outside-structural-pruning=") {
                options.scalar_tuning.h2_hier_outside_structural_pruning =
                    H2HierOutsideStructuralPruningStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--global-h2-birth-state=") {
                options.scalar_tuning.global_h2_birth_state =
                    GlobalH2BirthStateStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--global-h2-uf-layout=") {
                options.scalar_tuning.global_h2_uf_layout =
                    GlobalH2UnionFindLayoutStrategy::parse(value)?;
                options.scalar_tuning_overridden = true;
                index += 1;
                continue;
            }
            if argument.starts_with('-') {
                anyhow::bail!("unknown option {argument:?}; use --help to see supported options");
            }
        }
        positional.push(argument);
        index += 1;
    }

    Ok((positional, options))
}

fn effective_scalar_tuning(mode: Mode, options: &RunOptions) -> ScalarStreamTuning {
    let mut tuning = options.scalar_tuning;

    if matches!(
        mode,
        Mode::PersistenceH2ScalarStream
            | Mode::PersistenceH2ScalarHierarchicalStream
            | Mode::PersistenceH2ScalarBatchStream
    ) {
        if !options.active_state_overridden {
            tuning.active_state = ActiveStateStrategy::ParentSentinel;
        }
        if !options.phase_trim_overridden {
            tuning.phase_trim = if cfg!(all(target_os = "linux", target_env = "gnu")) {
                PhaseTrimStrategy::BeforeReduce
            } else {
                PhaseTrimStrategy::Off
            };
        }
    }

    tuning
}

fn has_option_before_terminator(args: &[String], names: &[&str]) -> bool {
    for argument in args.iter().skip(1) {
        if argument == "--" {
            return false;
        }
        if names.contains(&argument.as_str()) {
            return true;
        }
    }
    false
}

fn planning_connectivity(
    mode: Mode,
    foreground: connectivity::Connectivity,
    background: connectivity::Connectivity,
) -> connectivity::Connectivity {
    match mode {
        Mode::Betti2
        | Mode::EventBetti2
        | Mode::EventBetti2Stream
        | Mode::EventBetti2ScalarStream
        | Mode::PersistenceH2
        | Mode::PersistenceH2Stream
        | Mode::PersistenceH2Scalar
        | Mode::PersistenceH2ScalarBatch
        | Mode::PersistenceH2ScalarStream
        | Mode::PersistenceH2ScalarHierarchicalStream
        | Mode::PersistenceH2ScalarBatchStream
        | Mode::MergeTreeH2
        | Mode::MergeTreeH2Hierarchical
        | Mode::MergeTreeH2Stream => background,
        Mode::Both => connectivity::Connectivity::TwentySix,
        _ => foreground,
    }
}

fn preflight(
    shape: [usize; 3],
    slab_depth: usize,
    connectivity: connectivity::Connectivity,
) -> Result<()> {
    let plan = SizePlan::new(shape[0], shape[1], shape[2], slab_depth, connectivity)?;
    plan.print();
    Ok(())
}

fn preflight_hierarchical(
    shape: [usize; 3],
    slab_depth: usize,
    connectivity: connectivity::Connectivity,
) -> Result<()> {
    let plan = SizePlan::new_hierarchical(shape[0], shape[1], shape[2], slab_depth, connectivity)?;
    plan.print_hierarchical();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if has_option_before_terminator(&args, &["--help", "-h"]) {
        print_usage();
        return Ok(());
    }

    if has_option_before_terminator(&args, &["--version", "-V"]) {
        println!("betti_curves {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let (positional, run_options) = split_run_arguments(&args)?;
    let verbose = run_options.verbose;
    let dry_run_plan = run_options.dry_run_plan;

    if positional.is_empty() {
        print_usage();
        std::process::exit(1);
    }
    if positional.len() > 5 {
        anyhow::bail!("too many positional arguments; use --help to see the command syntax");
    }

    let dir = Path::new(positional[0]);

    let slab_depth: usize = if positional.len() >= 2 {
        positional[1]
            .parse()
            .context("Could not parse slab_depth")?
    } else {
        4
    };

    if slab_depth == 0 {
        anyhow::bail!("slab_depth must be positive");
    }

    let foreground_connectivity = if positional.len() >= 3 {
        parse_connectivity(positional[2])?
    } else {
        connectivity::Connectivity::Six
    };

    let mode = if positional.len() >= 4 {
        parse_mode(positional[3])?
    } else {
        Mode::Both
    };

    let scalar_tuning = effective_scalar_tuning(mode, &run_options);

    if run_options.scalar_tuning_overridden
        && !matches!(
            mode,
            Mode::PersistenceH0ScalarStream
                | Mode::PersistenceH0ScalarHierarchicalStream
                | Mode::PersistenceH0ScalarBatchStream
                | Mode::PersistenceH2ScalarStream
                | Mode::PersistenceH2ScalarHierarchicalStream
                | Mode::PersistenceH2ScalarBatchStream
        )
    {
        anyhow::bail!(
            "scalar persistence stream tuning options are supported only by h0-scalar-stream, \
h0-scalar-hierarchical-stream, h0-scalar-batch-stream, h2-scalar-stream, h2-scalar-hierarchical-stream, and h2-scalar-batch-stream"
        );
    }

    if positional.len() == 5
        && !matches!(
            mode,
            Mode::EventBetti0ScalarStream
                | Mode::EventBetti2ScalarStream
                | Mode::PersistenceH0Scalar
                | Mode::PersistenceH0ScalarHierarchical
                | Mode::PersistenceH0ScalarHierarchicalStream
                | Mode::PersistenceH0ScalarBatch
                | Mode::PersistenceH0ScalarStream
                | Mode::PersistenceH0ScalarBatchStream
                | Mode::PersistenceH2Scalar
                | Mode::PersistenceH2ScalarBatch
                | Mode::PersistenceH2ScalarStream
                | Mode::PersistenceH2ScalarHierarchicalStream
                | Mode::PersistenceH2ScalarBatchStream
        )
    {
        anyhow::bail!("mode {mode:?} does not accept an output-path positional argument");
    }

    let background_connectivity = dual_connectivity(foreground_connectivity);
    let plan_connectivity =
        planning_connectivity(mode, foreground_connectivity, background_connectivity);
    let total_start = Instant::now();

    if dry_run_plan
        && matches!(
            mode,
            Mode::PersistenceH0ScalarBatch
                | Mode::PersistenceH0ScalarBatchStream
                | Mode::PersistenceH2ScalarBatch
                | Mode::PersistenceH2ScalarBatchStream
        )
    {
        anyhow::bail!(
            "--dry-run-plan currently accepts one TIFF stack, not a recursive batch root"
        );
    }

    // Scalar modes use an order-preserving 64-bit key and do not open the
    // integer-only reader used by the legacy modes.
    match mode {
        Mode::EventBetti0ScalarStream => {
            let output = positional.get(4).map(Path::new).unwrap_or_else(|| {
                Path::new("event_global_betti0_scalar_stream_curve_changes.csv")
            });

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_event_based_betti0_scalar_stream_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
                output,
            )?;
            println!("Wrote streaming scalar Betti-0 curve to {:?}", output);
            println!("Sparse curve rows: {}", stats.rows);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::EventBetti2ScalarStream => {
            let output = positional.get(4).map(Path::new).unwrap_or_else(|| {
                Path::new("event_global_betti2_scalar_stream_curve_changes.csv")
            });

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Background connectivity = {:?}", background_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_event_based_betti2_scalar_stream_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
                output,
            )?;
            println!("Wrote streaming scalar Betti-2 curve to {:?}", output);
            println!("Sparse curve rows: {}", stats.rows);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0Scalar => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_persistence_scalar.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let intervals =
                compute_h0_persistence_scalar_zslabs(&volume, slab_depth, foreground_connectivity)?;
            write_h0_scalar_persistence_csv(output, &intervals)?;
            println!("Wrote scalar H0 persistence intervals to {:?}", output);

            let curve = betti0_curve_from_scalar_h0_intervals(&intervals);
            let curve_output = Path::new("h0_reconstructed_betti0_scalar_curve.csv");
            write_scalar_betti0_curve_csv(curve_output, &curve)?;
            println!(
                "Wrote scalar Betti-0 curve reconstructed from H0 intervals to {:?}",
                curve_output
            );

            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0ScalarHierarchical => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_persistence_scalar_hierarchical.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);
            println!("Hierarchical H0 summary fan-in = pairwise");

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight_hierarchical(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let intervals = compute_h0_persistence_scalar_hierarchical_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
            )?;
            write_h0_scalar_persistence_csv(output, &intervals)?;
            println!(
                "Wrote hierarchical scalar H0 persistence intervals to {:?}",
                output
            );
            println!("Hierarchical intervals: {}", intervals.len());
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0ScalarHierarchicalStream => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_persistence_scalar_hierarchical_stream.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);
            println!("Hierarchical H0 streaming fan-in = online pairwise");

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight_hierarchical(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_h0_persistence_scalar_hierarchical_stream_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
                output,
                scalar_tuning,
            )?;
            println!(
                "Wrote optimized hierarchical scalar H0 intervals to {:?}",
                output
            );
            println!("Finite intervals: {}", stats.finite_intervals);
            println!("Essential intervals: {}", stats.essential_intervals);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0ScalarBatch => {
            let output_root = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_scalar_batch_results"));

            println!("Batch root = {:?}", dir);
            println!("Output root = {:?}", output_root);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);

            let count =
                compute_h0_scalar_batch(dir, output_root, slab_depth, foreground_connectivity)?;
            println!("Completed {} scalar TIFF stacks", count);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0ScalarStream => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_persistence_scalar_stream.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Foreground connectivity = {:?}", foreground_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_h0_persistence_scalar_stream_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
                output,
                scalar_tuning,
            )?;
            println!("Wrote streaming scalar H0 intervals to {:?}", output);
            println!("Finite intervals: {}", stats.finite_intervals);
            println!("Essential intervals: {}", stats.essential_intervals);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH0ScalarBatchStream => {
            let output_root = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h0_scalar_batch_stream_results"));

            let count = compute_h0_scalar_stream_batch(
                dir,
                output_root,
                slab_depth,
                foreground_connectivity,
                scalar_tuning,
            )?;
            println!("Completed {} streaming scalar TIFF stacks", count);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH2Scalar => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h2_persistence_scalar.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Background connectivity = {:?}", background_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let intervals =
                compute_h2_persistence_scalar_zslabs(&volume, slab_depth, background_connectivity)?;
            write_h2_scalar_persistence_csv(output, &intervals)?;
            println!("Wrote scalar H2 persistence intervals to {:?}", output);

            let curve = betti2_curve_from_scalar_h2_intervals(&intervals);
            let curve_output = Path::new("h2_reconstructed_betti2_scalar_curve.csv");
            write_scalar_betti2_curve_csv(curve_output, &curve)?;
            println!(
                "Wrote scalar Betti-2 curve reconstructed from H2 intervals to {:?}",
                curve_output
            );

            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH2ScalarBatch => {
            let output_root = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h2_scalar_batch_results"));

            println!("Batch root = {:?}", dir);
            println!("Output root = {:?}", output_root);
            println!("Using slab_depth = {}", slab_depth);
            println!("Background connectivity = {:?}", background_connectivity);

            let count =
                compute_h2_scalar_batch(dir, output_root, slab_depth, background_connectivity)?;
            println!("Completed {} scalar TIFF stacks", count);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH2ScalarStream => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h2_persistence_scalar_stream.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Background connectivity = {:?}", background_connectivity);

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_h2_persistence_scalar_stream_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
                output,
                scalar_tuning,
            )?;
            println!("Wrote streaming scalar H2 intervals to {:?}", output);
            println!("Finite intervals: {}", stats.finite_intervals);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH2ScalarHierarchicalStream => {
            let output = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h2_persistence_scalar_hierarchical_stream.csv"));

            println!("Opening scalar TIFF stack from {:?}", dir);
            println!("Using slab_depth = {}", slab_depth);
            println!("Background connectivity = {:?}", background_connectivity);
            println!(
                "Hierarchical H2 streaming fan-in = online pairwise with distinguished outside"
            );

            let volume = ScalarTiffStackReader::open(dir)?;
            print_scalar_volume_info(&volume);
            preflight_hierarchical(volume.shape(), slab_depth, plan_connectivity)?;
            if dry_run_plan {
                return Ok(());
            }
            let stats = compute_h2_persistence_scalar_hierarchical_stream_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
                output,
                scalar_tuning,
            )?;
            println!(
                "Wrote optimized hierarchical scalar H2 intervals to {:?}",
                output
            );
            println!("Finite intervals: {}", stats.finite_intervals);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        Mode::PersistenceH2ScalarBatchStream => {
            let output_root = positional
                .get(4)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("h2_scalar_batch_stream_results"));

            let count = compute_h2_scalar_stream_batch(
                dir,
                output_root,
                slab_depth,
                background_connectivity,
                scalar_tuning,
            )?;
            println!("Completed {} streaming scalar TIFF stacks", count);
            println!(
                "Total runtime: {:.3} seconds",
                total_start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        _ => {}
    }

    println!("Opening TIFF stack from {:?}", dir);
    println!("Using slab_depth = {}", slab_depth);
    println!("Mode = {:?}", mode);
    println!("Foreground connectivity = {:?}", foreground_connectivity);
    println!("Background connectivity = {:?}", background_connectivity);

    let volume = TiffStackReader::open(dir)?;
    print_volume_info(&volume);
    if matches!(
        mode,
        Mode::MergeTreeH0Hierarchical | Mode::MergeTreeH2Hierarchical
    ) {
        preflight_hierarchical(volume.shape(), slab_depth, plan_connectivity)?;
    } else {
        preflight(volume.shape(), slab_depth, plan_connectivity)?;
    }
    if dry_run_plan {
        return Ok(());
    }

    match mode {
        Mode::Betti0 => {
            let unique_values = collect_threshold_values(&volume, slab_depth, verbose)?;

            let results = compute_sparse_global_betti0_parallel(
                &volume,
                slab_depth,
                foreground_connectivity,
                &unique_values,
            )?;

            let changed_only = compress_changes(&results);
            let output = Path::new("global_betti0_curve_changes.csv");
            write_sparse_betti0_csv(output, &changed_only)?;
            println!("Wrote sparse Betti-0 changes to {:?}", output);
        }

        Mode::Betti2 => {
            let unique_values = collect_threshold_values(&volume, slab_depth, verbose)?;

            let results = compute_sparse_betti2_curve_parallel(
                &volume,
                slab_depth,
                &unique_values,
                background_connectivity,
            )?;

            let changed_only = compress_changes(&results);
            let output = Path::new("global_betti2_curve_changes.csv");
            write_sparse_betti2_csv(output, &changed_only)?;
            println!("Wrote sparse Betti-2 changes to {:?}", output);
        }

        Mode::Both => {
            let unique_values = collect_threshold_values(&volume, slab_depth, verbose)?;

            let betti0_results = compute_sparse_global_betti0_parallel(
                &volume,
                slab_depth,
                foreground_connectivity,
                &unique_values,
            )?;
            let betti0_changed = compress_changes(&betti0_results);
            let betti0_output = Path::new("global_betti0_curve_changes.csv");
            write_sparse_betti0_csv(betti0_output, &betti0_changed)?;
            println!("Wrote sparse Betti-0 changes to {:?}", betti0_output);

            let betti2_results = compute_sparse_betti2_curve_parallel(
                &volume,
                slab_depth,
                &unique_values,
                background_connectivity,
            )?;
            let betti2_changed = compress_changes(&betti2_results);
            let betti2_output = Path::new("global_betti2_curve_changes.csv");
            write_sparse_betti2_csv(betti2_output, &betti2_changed)?;
            println!("Wrote sparse Betti-2 changes to {:?}", betti2_output);
        }

        Mode::EventBetti0 => {
            println!("Reducer = in-memory");

            let curve =
                compute_event_based_betti0_zslabs(&volume, slab_depth, foreground_connectivity)?;

            let output = Path::new("event_global_betti0_curve_changes.csv");
            write_event_betti0_csv(output, &curve)?;
            println!("Wrote event-based Betti-0 changes to {:?}", output);
        }

        Mode::EventBetti0Stream => {
            println!("Reducer = disk-backed streaming");

            let curve = compute_event_based_betti0_stream_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
            )?;

            let output = Path::new("event_global_betti0_stream_curve_changes.csv");
            write_event_betti0_stream_csv(output, &curve)?;
            println!(
                "Wrote streaming event-based Betti-0 changes to {:?}",
                output
            );
        }

        Mode::EventBetti2 => {
            println!("Reducer = in-memory");

            let curve =
                compute_event_based_betti2_zslabs(&volume, slab_depth, background_connectivity)?;

            let output = Path::new("event_global_betti2_curve_changes.csv");
            write_event_betti2_csv(output, &curve)?;
            println!("Wrote event-based Betti-2 changes to {:?}", output);
        }

        Mode::EventBetti2Stream => {
            println!("Reducer = disk-backed streaming");

            let curve = compute_event_based_betti2_stream_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
            )?;

            let output = Path::new("event_global_betti2_stream_curve_changes.csv");
            write_event_betti2_stream_csv(output, &curve)?;
            println!(
                "Wrote streaming event-based Betti-2 changes to {:?}",
                output
            );
        }

        Mode::PersistenceH0 => {
            println!("Computing in-memory slabwise H0 persistence...");

            let intervals =
                compute_h0_persistence_zslabs(&volume, slab_depth, foreground_connectivity)?;

            let output = Path::new("h0_persistence.csv");
            write_h0_persistence_csv(output, &intervals)?;
            println!("Wrote H0 persistence intervals to {:?}", output);

            let reconstructed = betti0_curve_from_h0_intervals(&intervals);
            let curve_output = Path::new("h0_reconstructed_betti0_curve.csv");
            write_event_betti0_csv(curve_output, &reconstructed)?;
            println!(
                "Wrote Betti-0 curve reconstructed from H0 intervals to {:?}",
                curve_output
            );
        }

        Mode::PersistenceH0Stream => {
            println!("Computing disk-backed slabwise H0 persistence...");

            let output = Path::new("h0_persistence_stream.csv");
            let stats = compute_h0_persistence_stream_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
                output,
            )?;

            println!("Wrote streaming H0 persistence intervals to {:?}", output);
            println!("Finite intervals: {}", stats.finite_intervals);
            println!("Essential intervals: {}", stats.essential_intervals);
        }

        Mode::PersistenceH2 => {
            println!("Computing in-memory slabwise H2 persistence...");

            let intervals =
                compute_h2_persistence_zslabs(&volume, slab_depth, background_connectivity)?;

            let output = Path::new("h2_persistence.csv");
            write_h2_persistence_csv(output, &intervals)?;
            println!("Wrote H2 persistence intervals to {:?}", output);

            let reconstructed = betti2_curve_from_h2_intervals(&intervals);
            let curve_output = Path::new("h2_reconstructed_betti2_curve.csv");
            write_event_betti2_csv(curve_output, &reconstructed)?;
            println!(
                "Wrote Betti-2 curve reconstructed from H2 intervals to {:?}",
                curve_output
            );
        }

        Mode::PersistenceH2Stream => {
            println!("Computing disk-backed slabwise H2 persistence...");

            let output = Path::new("h2_persistence_stream.csv");
            let stats = compute_h2_persistence_stream_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
                output,
            )?;

            println!("Wrote streaming H2 persistence intervals to {:?}", output);
            println!("Finite intervals: {}", stats.finite_intervals);
        }

        Mode::MergeTreeH0 => {
            let tree = compute_h0_merge_tree_zslabs(&volume, slab_depth, foreground_connectivity)?;
            let output = Path::new("h0_merge_tree.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote H0 merge tree to {:?}", output);
            let nodes_output = Path::new("h0_merge_tree_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote H0 elder-rule branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::MergeTreeH0Hierarchical => {
            let tree = compute_h0_merge_tree_hierarchical_zslabs(
                &volume,
                slab_depth,
                foreground_connectivity,
            )?;
            let output = Path::new("h0_branch_tree_hierarchical.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote hierarchical H0 branch tree to {:?}", output);
            let nodes_output = Path::new("h0_branch_tree_hierarchical_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote plateau-canonical hierarchical H0 branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::MergeTreeH0Stream => {
            let tree =
                compute_h0_merge_tree_stream_zslabs(&volume, slab_depth, foreground_connectivity)?;
            let output = Path::new("h0_merge_tree_stream.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote streaming H0 merge tree to {:?}", output);
            let nodes_output = Path::new("h0_merge_tree_stream_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote streaming H0 elder-rule branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::MergeTreeH2 => {
            let tree = compute_h2_merge_tree_zslabs(&volume, slab_depth, background_connectivity)?;
            let output = Path::new("h2_merge_tree.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote H2 merge tree to {:?}", output);
            let nodes_output = Path::new("h2_merge_tree_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote H2 elder-rule branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::MergeTreeH2Hierarchical => {
            let tree = compute_h2_merge_tree_hierarchical_zslabs(
                &volume,
                slab_depth,
                background_connectivity,
            )?;
            let output = Path::new("h2_branch_tree_hierarchical.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote hierarchical H2 branch tree to {:?}", output);
            let nodes_output = Path::new("h2_branch_tree_hierarchical_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote plateau-canonical hierarchical H2 branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::MergeTreeH2Stream => {
            let tree =
                compute_h2_merge_tree_stream_zslabs(&volume, slab_depth, background_connectivity)?;
            let output = Path::new("h2_merge_tree_stream.csv");
            write_merge_tree_csv(output, &tree)?;
            println!("Wrote streaming H2 merge tree to {:?}", output);
            let nodes_output = Path::new("h2_merge_tree_stream_nodes.csv");
            write_merge_tree_nodes_csv(nodes_output, &tree)?;
            println!(
                "Wrote streaming H2 elder-rule branch-tree nodes to {:?}",
                nodes_output
            );
            println!("Retained branch-tree nodes: {}", tree.node_count);
        }

        Mode::EventBetti0ScalarStream
        | Mode::EventBetti2ScalarStream
        | Mode::PersistenceH0Scalar
        | Mode::PersistenceH0ScalarHierarchical
        | Mode::PersistenceH0ScalarHierarchicalStream
        | Mode::PersistenceH0ScalarBatch
        | Mode::PersistenceH0ScalarStream
        | Mode::PersistenceH0ScalarBatchStream
        | Mode::PersistenceH2Scalar
        | Mode::PersistenceH2ScalarBatch
        | Mode::PersistenceH2ScalarStream
        | Mode::PersistenceH2ScalarHierarchicalStream
        | Mode::PersistenceH2ScalarBatchStream => {
            unreachable!("scalar modes return before opening the integer TIFF reader");
        }
    }

    println!(
        "Total runtime: {:.3} seconds",
        total_start.elapsed().as_secs_f64()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn common_options_are_removed_without_reordering_positionals() {
        let args = strings(&["betti_curves", "stack", "--verbose", "4", "--dry-run-plan"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack", "4"]
        );
        assert!(options.verbose);
        assert!(options.dry_run_plan);
    }

    #[test]
    fn unknown_options_are_not_misread_as_paths() {
        let args = strings(&["betti_curves", "--memory-buget", "stack"]);
        let error = split_run_arguments(&args).unwrap_err();
        assert!(error.to_string().contains("unknown option"));
    }

    #[test]
    fn option_terminator_allows_reserved_names_as_literal_paths() {
        let args = strings(&["betti_curves", "--", "--help"]);
        assert!(!has_option_before_terminator(&args, &["--help", "-h"]));

        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["--help"]
        );
        assert!(!options.verbose);
        assert!(!options.dry_run_plan);
    }

    #[test]
    fn neighbor_kernel_option_is_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--neighbor-kernel",
            "interior-fast",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.neighbor_kernel,
            NeighborKernelStrategy::InteriorFast
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn representative_active_check_and_diagnostics_are_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--representative-active-check",
            "trust-pruner",
            "--sweep-diagnostics",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.representative_active_check,
            RepresentativeActiveCheckStrategy::TrustPruner
        );
        assert!(options.scalar_tuning.sweep_diagnostics);
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn union_kernel_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--union-kernel", "root-carrying"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.union_kernel,
            UnionKernelStrategy::RootCarrying
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn neighbor_root_check_option_is_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--neighbor-root-check=parent-shortcut",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.neighbor_root_check,
            NeighborRootCheckStrategy::ParentShortcut
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn neighbor_root_two_hop_option_is_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--neighbor-root-check=parent-two-hop",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.neighbor_root_check,
            NeighborRootCheckStrategy::ParentTwoHop
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn neighbor_root_cached_find_option_is_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--neighbor-root-check=parent-cached-find",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.neighbor_root_check,
            NeighborRootCheckStrategy::ParentCachedFind
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn h0_pruning_cache_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--h0-pruning-cache=256k"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.h0_pruning_cache,
            H0PruningCacheStrategy::Entries256K
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn active_state_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--active-state=parent-sentinel"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.active_state,
            ActiveStateStrategy::ParentSentinel
        );
        assert!(options.scalar_tuning_overridden);
        assert!(options.active_state_overridden);
    }
    #[test]
    fn interface_state_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--interface-state=root-invariant"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.interface_state,
            InterfaceStateStrategy::RootInvariant
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn uf_layout_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--uf-layout=packed"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.uf_layout,
            UnionFindLayoutStrategy::Packed
        );
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn phase_trim_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--phase-trim=before-reduce"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.phase_trim,
            PhaseTrimStrategy::BeforeReduce
        );
        assert!(options.scalar_tuning_overridden);
        assert!(options.phase_trim_overridden);
    }

    #[test]
    fn h2_memory_audit_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--h2-memory-audit"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert!(options.scalar_tuning.h2_memory_audit);
        assert!(options.scalar_tuning_overridden);
    }

    #[test]
    fn local_h2_birth_state_option_is_parsed() {
        let args = strings(&["betti_curves", "stack", "--local-h2-birth-state", "compact"]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.local_h2_birth_state,
            LocalH2BirthStateStrategy::Compact
        );
    }

    #[test]
    fn global_h2_birth_state_option_is_parsed() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--global-h2-birth-state",
            "compact",
        ]);
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.global_h2_birth_state,
            GlobalH2BirthStateStrategy::Compact
        );
    }

    #[test]
    fn h0_birth_buffer_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--h0-birth-buffer=reuse-input".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.h0_birth_buffer,
            H0BirthBufferStrategy::ReuseInput
        );
    }

    #[test]
    fn h0_event_storage_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--h0-event-storage=direct".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.h0_event_storage,
            H0EventStorageStrategy::Direct
        );
    }

    #[test]
    fn h0_hier_attach_pruning_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--h0-hier-attach-pruning=elder-dominated".to_owned(),
        ];
        let (_positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            options.scalar_tuning.h0_hier_attach_pruning,
            H0HierAttachPruningStrategy::ElderDominated
        );
    }

    #[test]
    fn global_h0_uf_layout_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--global-h0-uf-layout=packed".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.global_h0_uf_layout,
            GlobalH0UnionFindLayoutStrategy::Packed
        );
    }

    #[test]
    fn h2_hier_cross_storage_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--h2-hier-cross-storage=direct".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.h2_hier_cross_storage,
            H2HierCrossStorageStrategy::Direct
        );
    }

    #[test]
    fn h2_hier_outside_structural_pruning_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--h2-hier-outside-structural-pruning=outside-dominated".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.h2_hier_outside_structural_pruning,
            H2HierOutsideStructuralPruningStrategy::OutsideDominated
        );
    }

    #[test]
    fn global_h2_uf_layout_option_is_parsed() {
        let args = vec![
            "betti_curves".to_owned(),
            "stack".to_owned(),
            "--global-h2-uf-layout=packed".to_owned(),
        ];
        let (positionals, options) = split_run_arguments(&args).unwrap();
        assert_eq!(
            positionals
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            ["stack"]
        );
        assert_eq!(
            options.scalar_tuning.global_h2_uf_layout,
            GlobalH2UnionFindLayoutStrategy::Packed
        );
    }

    #[test]
    fn production_defaults_are_dimension_specific() {
        let args = strings(&["betti_curves", "stack"]);
        let (_, options) = split_run_arguments(&args).unwrap();

        let h0 = effective_scalar_tuning(Mode::PersistenceH0ScalarStream, &options);
        assert_eq!(h0.active_state, ActiveStateStrategy::Separate);
        assert_eq!(h0.uf_layout, UnionFindLayoutStrategy::Packed);
        assert_eq!(h0.phase_trim, PhaseTrimStrategy::Off);

        let h2 = effective_scalar_tuning(Mode::PersistenceH2ScalarStream, &options);
        assert_eq!(h2.active_state, ActiveStateStrategy::ParentSentinel);
        assert_eq!(h2.uf_layout, UnionFindLayoutStrategy::Packed);
        let expected_trim = if cfg!(all(target_os = "linux", target_env = "gnu")) {
            PhaseTrimStrategy::BeforeReduce
        } else {
            PhaseTrimStrategy::Off
        };
        assert_eq!(h2.phase_trim, expected_trim);
    }

    #[test]
    fn unrelated_override_does_not_disable_h2_production_defaults() {
        let args = strings(&["betti_curves", "stack", "--merge-strategy=heap"]);
        let (_, options) = split_run_arguments(&args).unwrap();
        let h2 = effective_scalar_tuning(Mode::PersistenceH2ScalarStream, &options);
        assert_eq!(h2.merge_strategy, MergeStrategy::Heap);
        assert_eq!(h2.active_state, ActiveStateStrategy::ParentSentinel);
        assert_eq!(h2.uf_layout, UnionFindLayoutStrategy::Packed);
    }

    #[test]
    fn explicit_dimension_specific_overrides_win() {
        let args = strings(&[
            "betti_curves",
            "stack",
            "--active-state=separate",
            "--phase-trim=off",
        ]);
        let (_, options) = split_run_arguments(&args).unwrap();
        let h2 = effective_scalar_tuning(Mode::PersistenceH2ScalarStream, &options);
        assert_eq!(h2.active_state, ActiveStateStrategy::Separate);
        assert_eq!(h2.phase_trim, PhaseTrimStrategy::Off);
    }
}
