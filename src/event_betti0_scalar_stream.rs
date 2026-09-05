use anyhow::{Context, Result, bail};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::event_betti0::GlobalInterfaceUnionFind;
use crate::event_scalar_common::{
    DeltaEvent, DeltaRunReader, PairEvent, PairRunReader, ScalarEventCurveStats, SlabDescriptor,
    create_buffered_file, local_boundary_node_id, lower_face_global_id, make_slab_descriptors,
    report_scalar_event_memory_estimate, sorted_scalar_indices, upper_face_global_id,
    write_delta_event, write_pair_event,
};
use crate::interface_sparsify_scalar::sparsify_scalar_sublevel_interface;
use crate::io_scalar::{ScalarBlock, ScalarTiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::scalar::ScalarKey;
use crate::temp_runs::TempRunDirectory;

const NO_INTERFACE_REP: u32 = u32::MAX;
const EVENT_BUFFER_LIMIT: usize = 1 << 20;

#[derive(Debug)]
struct LocalEventUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    interface_rep: Vec<u32>,
}

impl LocalEventUnionFind {
    fn new(voxel_count: usize) -> Self {
        assert!(
            voxel_count <= u32::MAX as usize,
            "scalar slab has too many voxels for u32 union-find indices"
        );
        Self {
            parent: (0..voxel_count as u32).collect(),
            rank: vec![0u8; voxel_count],
            interface_rep: vec![NO_INTERFACE_REP; voxel_count],
        }
    }

    fn find(&mut self, mut node: u32) -> u32 {
        while self.parent[node as usize] != node {
            let parent = self.parent[node as usize];
            let grandparent = self.parent[parent as usize];
            self.parent[node as usize] = grandparent;
            node = parent;
        }
        node
    }

    fn set_interface_rep(&mut self, node: u32, interface_rep: u32) {
        let root = self.find(node);
        self.interface_rep[root as usize] = interface_rep;
    }

    /// Returns `None` for a redundant edge. A successful union returns an
    /// optional pair of interface representatives that became equivalent.
    fn union_with_interface_event(&mut self, a: u32, b: u32) -> Option<Option<(u32, u32)>> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let rep_a = self.interface_rep[root_a as usize];
        let rep_b = self.interface_rep[root_b as usize];
        let interface_event =
            if rep_a != NO_INTERFACE_REP && rep_b != NO_INTERFACE_REP && rep_a != rep_b {
                Some((rep_a, rep_b))
            } else {
                None
            };

        let rank_a = self.rank[root_a as usize];
        let rank_b = self.rank[root_b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.parent[root_b as usize] = root_a;
        if rank_a == rank_b {
            self.rank[root_a as usize] += 1;
        }
        self.interface_rep[root_a as usize] = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };

        Some(interface_event)
    }
}

#[derive(Debug)]
struct ScalarBoundaryFaces {
    lower_values: Vec<ScalarKey>,
    upper_values: Vec<ScalarKey>,
}

#[derive(Debug)]
struct StoredUpperFace {
    descriptor: SlabDescriptor,
    values: Vec<ScalarKey>,
}

fn flush_pair_events(writer: &mut BufWriter<File>, pending: &mut Vec<PairEvent>) -> Result<()> {
    for event in pending.drain(..) {
        write_pair_event(writer, event)?;
    }
    Ok(())
}

fn process_scalar_h0_slab(
    desc: &SlabDescriptor,
    block: &ScalarBlock,
    connectivity: Connectivity,
    delta_writer: &mut BufWriter<File>,
    interface_writer: &mut BufWriter<File>,
) -> Result<ScalarBoundaryFaces> {
    let voxel_count = block.voxel_count();
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let face_size = width
        .checked_mul(height)
        .expect("scalar H0 face size overflow");

    let mut union_find = LocalEventUnionFind::new(voxel_count);
    let mut active = vec![0u8; voxel_count];
    let order = sorted_scalar_indices(&block.values, block.pixel_type);
    let mut local_pruner = NeighborhoodComponentPruner::new(connectivity);
    let mut pending_interface_events = Vec::new();

    let mut begin = 0usize;
    while begin < order.len() {
        let value = block.values[order[begin] as usize];
        let mut end = begin + 1;
        while end < order.len() && block.values[order[end] as usize] == value {
            end += 1;
        }

        let mut delta = 0i64;
        for &index_u32 in &order[begin..end] {
            let index = index_u32 as usize;
            active[index] = 1;
            delta += 1;

            let x = index % width;
            let y = (index / width) % height;
            let z = index / face_size;
            let face_index = y * width + x;
            if let Some(local_interface_node) =
                local_boundary_node_id(z, depth, face_index, face_size)
            {
                union_find.set_interface_rep(index_u32, local_interface_node);
            }

            local_pruner.for_each_representative_neighbor(
                &active,
                x,
                y,
                z,
                width,
                height,
                depth,
                |neighbor| {
                    if active[neighbor] == 0 {
                        return;
                    }
                    if let Some(interface_event) =
                        union_find.union_with_interface_event(index_u32, neighbor as u32)
                    {
                        delta -= 1;
                        if let Some((a, b)) = interface_event {
                            pending_interface_events.push(PairEvent {
                                value,
                                a: desc.interface_base + a,
                                b: desc.interface_base + b,
                            });
                        }
                    }
                },
            );

            if pending_interface_events.len() >= EVENT_BUFFER_LIMIT {
                flush_pair_events(interface_writer, &mut pending_interface_events)?;
            }
        }

        if delta != 0 {
            write_delta_event(delta_writer, DeltaEvent { value, delta })?;
        }
        begin = end;
    }

    flush_pair_events(interface_writer, &mut pending_interface_events)?;

    let lower_values = block.values[..face_size].to_vec();
    let upper_start = (depth - 1) * face_size;
    let upper_values = block.values[upper_start..upper_start + face_size].to_vec();
    Ok(ScalarBoundaryFaces {
        lower_values,
        upper_values,
    })
}

fn write_cross_run(
    directory: &Path,
    pair_id: usize,
    left: &StoredUpperFace,
    right_desc: &SlabDescriptor,
    right_lower_values: &[ScalarKey],
    face_shape: [usize; 2],
    connectivity: Connectivity,
) -> Result<PathBuf> {
    let [width, height] = face_shape;
    let face_size = width.checked_mul(height).expect("face size overflow");
    let path = directory.join(format!("betti0_scalar_cross_{pair_id:05}.bin"));
    let mut writer = create_buffered_file(&path)?;

    let stats = sparsify_scalar_sublevel_interface(
        &left.values,
        right_lower_values,
        width,
        height,
        connectivity,
        |edge| {
            write_pair_event(
                &mut writer,
                PairEvent {
                    value: edge.value,
                    a: upper_face_global_id(
                        &left.descriptor,
                        face_size,
                        edge.left_face_index as usize,
                    ),
                    b: lower_face_global_id(right_desc, edge.right_face_index as usize),
                },
            )
        },
    )?;
    writer.flush()?;

    println!(
        "Scalar Betti-0 interface {pair_id}: retained {} of {} cross edges",
        stats.retained_edges, stats.candidate_edges
    );
    Ok(path)
}

struct PreparedScalarBetti0Runs {
    total_interface_nodes: usize,
    delta_run_paths: Vec<PathBuf>,
    interface_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
}

fn prepare_scalar_betti0_runs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    directory: &Path,
) -> Result<PreparedScalarBetti0Runs> {
    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth)?;
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let mut delta_run_paths = Vec::with_capacity(descriptors.len());
    let mut interface_run_paths = Vec::with_capacity(descriptors.len());
    let mut cross_run_paths = Vec::with_capacity(descriptors.len().saturating_sub(1));
    let mut previous_upper_face: Option<StoredUpperFace> = None;

    for desc in &descriptors {
        println!(
            "Preparing scalar Betti-0 slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );

        let delta_path = directory.join(format!("betti0_scalar_delta_{:05}.bin", desc.slab_id));
        let interface_path =
            directory.join(format!("betti0_scalar_interface_{:05}.bin", desc.slab_id));
        let mut delta_writer = create_buffered_file(&delta_path)?;
        let mut interface_writer = create_buffered_file(&interface_path)?;

        let block = volume.read_z_slab(desc.z0, desc.z1)?;
        let faces = process_scalar_h0_slab(
            desc,
            &block,
            connectivity,
            &mut delta_writer,
            &mut interface_writer,
        )?;
        delta_writer.flush()?;
        interface_writer.flush()?;

        assert_eq!(
            desc.interface_count as usize,
            if desc.local_depth == 1 {
                volume.width * volume.height
            } else {
                2 * volume.width * volume.height
            }
        );
        drop(block);

        if let Some(previous) = previous_upper_face.take() {
            cross_run_paths.push(write_cross_run(
                directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &faces.lower_values,
                [volume.width, volume.height],
                connectivity,
            )?);
        }

        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: faces.upper_values,
        });
        delta_run_paths.push(delta_path);
        interface_run_paths.push(interface_path);
    }

    Ok(PreparedScalarBetti0Runs {
        total_interface_nodes,
        delta_run_paths,
        interface_run_paths,
        cross_run_paths,
    })
}

type MinHeap = BinaryHeap<Reverse<(ScalarKey, usize)>>;

fn open_delta_runs(paths: &[PathBuf]) -> Result<(Vec<DeltaRunReader>, MinHeap)> {
    let mut readers = paths
        .iter()
        .map(|path| DeltaRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter_mut().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push(Reverse((value, index)));
        }
    }
    Ok((readers, heap))
}

fn open_pair_runs(paths: &[PathBuf]) -> Result<(Vec<PairRunReader>, MinHeap)> {
    let mut readers = paths
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter_mut().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push(Reverse((value, index)));
        }
    }
    Ok((readers, heap))
}

fn min_heap_value(heap: &MinHeap) -> Option<ScalarKey> {
    heap.peek().map(|Reverse((value, _))| *value)
}

fn next_value(delta: &MinHeap, interface: &MinHeap, cross: &MinHeap) -> Option<ScalarKey> {
    [
        min_heap_value(delta),
        min_heap_value(interface),
        min_heap_value(cross),
    ]
    .into_iter()
    .flatten()
    .min()
}

fn drain_delta_value(
    value: ScalarKey,
    readers: &mut [DeltaRunReader],
    heap: &mut MinHeap,
) -> Result<i64> {
    let mut total = 0i64;
    while min_heap_value(heap) == Some(value) {
        let Reverse((_, reader_index)) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-0 delta run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-0 delta heap is out of order");
        }
        total += event.delta;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push(Reverse((next, reader_index)));
        }
    }
    Ok(total)
}

fn drain_pair_value(
    value: ScalarKey,
    readers: &mut [PairRunReader],
    heap: &mut MinHeap,
    mut apply: impl FnMut(u32, u32),
) -> Result<()> {
    while min_heap_value(heap) == Some(value) {
        let Reverse((_, reader_index)) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-0 pair run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-0 pair heap is out of order");
        }
        apply(event.a, event.b);
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push(Reverse((next, reader_index)));
        }
    }
    Ok(())
}

fn reduce_scalar_betti0_runs(
    prepared: &PreparedScalarBetti0Runs,
    output_path: &Path,
) -> Result<ScalarEventCurveStats> {
    println!(
        "Allocating global scalar Betti-0 interface union-find for {} nodes",
        prepared.total_interface_nodes
    );
    let mut global_union_find = GlobalInterfaceUnionFind::new(prepared.total_interface_nodes);
    let (mut delta_readers, mut delta_heap) = open_delta_runs(&prepared.delta_run_paths)?;
    let (mut interface_readers, mut interface_heap) =
        open_pair_runs(&prepared.interface_run_paths)?;
    let (mut cross_readers, mut cross_heap) = open_pair_runs(&prepared.cross_run_paths)?;

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar Betti-0 output {output_path:?}"))?;
    writeln!(output, "threshold,betti0")?;

    let mut beta0 = 0i64;
    let mut previous = None;
    let mut rows = 0u64;

    while let Some(value) = next_value(&delta_heap, &interface_heap, &cross_heap) {
        beta0 += drain_delta_value(value, &mut delta_readers, &mut delta_heap)?;

        let beta0_ref = &mut beta0;
        drain_pair_value(
            value,
            &mut interface_readers,
            &mut interface_heap,
            |a, b| {
                if !global_union_find.union(a, b) {
                    *beta0_ref += 1;
                }
            },
        )?;
        drain_pair_value(value, &mut cross_readers, &mut cross_heap, |a, b| {
            if global_union_find.union(a, b) {
                *beta0_ref -= 1;
            }
        })?;

        if beta0 < 0 {
            bail!("scalar Betti-0 became negative at threshold {value}: {beta0}");
        }
        if previous != Some(beta0) {
            writeln!(output, "{value},{beta0}")?;
            previous = Some(beta0);
            rows += 1;
        }
    }

    output.commit()?;
    Ok(ScalarEventCurveStats { rows })
}

pub fn compute_event_based_betti0_scalar_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    output_path: &Path,
) -> Result<ScalarEventCurveStats> {
    let start = Instant::now();
    report_scalar_event_memory_estimate(
        volume.width,
        volume.height,
        volume.depth,
        slab_depth,
        connectivity,
        false,
    )?;
    let temp_directory = TempRunDirectory::create("betti0_scalar_stream_runs")?;
    println!(
        "Scalar Betti-0 streaming temporary directory: {:?}",
        temp_directory.path()
    );

    let prepared =
        prepare_scalar_betti0_runs(volume, slab_depth, connectivity, temp_directory.path())?;
    println!("Scalar Betti-0 slab preparation completed.");
    println!("Starting disk-backed scalar Betti-0 curve reduction.");

    let result = reduce_scalar_betti0_runs(&prepared, output_path);
    let stats = result?;
    temp_directory.close()?;
    println!(
        "Streaming scalar Betti-0 curve computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(stats)
}
