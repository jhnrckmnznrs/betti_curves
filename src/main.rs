mod betti0;
mod betti2;
mod connectivity;
mod csv;
mod event_betti0;
mod event_betti2;
mod io;
mod union_find;

use crate::event_betti0::{compute_event_based_betti0_zslabs, write_event_betti0_csv};
use crate::event_betti2::{compute_event_based_betti2_zslabs, write_event_betti2_csv};
use anyhow::{Context, Result};
use betti0::compute_sparse_global_betti0_parallel;
use betti2::compute_sparse_betti2_curve_parallel;
use connectivity::{Mode, dual_connectivity, parse_connectivity, parse_mode};
use csv::{compress_changes, write_sparse_betti0_csv, write_sparse_betti2_csv};
use io::{TiffStackReader, collect_unique_values_by_slabs, print_volume_info};
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!(
            "  cargo run --release -- <directory_of_tiff_slices> [slab_depth] [foreground_connectivity] [mode]"
        );
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  cargo run --release -- /data/volume 4 6 betti0");
        eprintln!("  cargo run --release -- /data/volume 4 6 betti2");
        eprintln!("  cargo run --release -- /data/volume 4 6 both");
        std::process::exit(1);
    }

    let dir = Path::new(&args[1]);

    let slab_depth: usize = if args.len() >= 3 {
        args[2].parse().context("Could not parse slab_depth")?
    } else {
        4
    };

    let foreground_connectivity = if args.len() >= 4 {
        parse_connectivity(&args[3])?
    } else {
        connectivity::Connectivity::Six
    };

    let mode = if args.len() >= 5 {
        parse_mode(&args[4])?
    } else {
        Mode::Both
    };

    let background_connectivity = dual_connectivity(foreground_connectivity);
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");

    let total_start = Instant::now();

    println!("Opening TIFF stack from {:?}", dir);
    println!("Using slab_depth = {}", slab_depth);
    println!("Mode = {:?}", mode);
    println!("Foreground connectivity = {:?}", foreground_connectivity);
    println!("Background connectivity = {:?}", background_connectivity);

    let volume = TiffStackReader::open(dir)?;
    print_volume_info(&volume);

    println!("Collecting unique intensity values...");
    let scan_start = Instant::now();

    let unique_values = collect_unique_values_by_slabs(&volume, slab_depth, verbose)?;

    println!(
        "Unique-value scan took {:.3} seconds",
        scan_start.elapsed().as_secs_f64()
    );

    println!("Found {} unique intensity values", unique_values.len());

    if unique_values.is_empty() {
        println!("No values found.");
        return Ok(());
    }

    println!("First few unique values:");
    for v in unique_values.iter().take(20) {
        println!("  {}", v);
    }

    println!("Last few unique values:");
    for v in unique_values.iter().rev().take(20).rev() {
        println!("  {}", v);
    }

    println!();

    match mode {
        Mode::Betti0 | Mode::Both => {
            println!("Computing sparse global Betti-0 curve in parallel...");

            let betti0_results = compute_sparse_global_betti0_parallel(
                &volume,
                slab_depth,
                foreground_connectivity,
                &unique_values,
            )?;

            let betti0_changed_only = compress_changes(&betti0_results);

            let betti0_out_path = Path::new("global_betti0_curve_changes.csv");
            write_sparse_betti0_csv(betti0_out_path, &betti0_changed_only)?;

            println!("Wrote sparse Betti-0 changes to {:?}", betti0_out_path);
            println!();
        }
        _ => {}
    }

    if let Mode::EventBetti0 = mode {
        println!("Computing event-based global Betti-0 curve...");

        let event_curve =
            compute_event_based_betti0_zslabs(&volume, slab_depth, foreground_connectivity)?;

        let out_path = Path::new("event_global_betti0_curve_changes.csv");
        write_event_betti0_csv(out_path, &event_curve)?;

        println!("Wrote event-based Betti-0 changes to {:?}", out_path);
        println!("=== Event-based sparse Betti-0 changes ===");
        println!();
    }

    match mode {
        Mode::Betti2 | Mode::Both => {
            println!("Computing sparse global Betti-2 curve in parallel...");

            let betti2_results = compute_sparse_betti2_curve_parallel(
                &volume,
                slab_depth,
                &unique_values,
                background_connectivity,
            )?;

            let betti2_changed_only = compress_changes(&betti2_results);

            let betti2_out_path = Path::new("global_betti2_curve_changes.csv");
            write_sparse_betti2_csv(betti2_out_path, &betti2_changed_only)?;

            println!("Wrote sparse Betti-2 changes to {:?}", betti2_out_path);
            println!();
        }
        _ => {}
    }

    if let Mode::EventBetti2 = mode {
        println!("Computing event-based slabwise Betti-2 curve...");

        let event_betti2_curve =
            compute_event_based_betti2_zslabs(&volume, slab_depth, background_connectivity)?;

        let out_path = Path::new("event_global_betti2_curve_changes.csv");
        write_event_betti2_csv(out_path, &event_betti2_curve)?;

        println!("Wrote event-based Betti-2 changes to {:?}", out_path);
        println!("=== Event-based sparse Betti-2 changes ===");
        println!();
    }

    println!(
        "Total runtime: {:.3} seconds",
        total_start.elapsed().as_secs_f64()
    );

    Ok(())
}
