use anyhow::{Context, Result, bail};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::event_betti2::GlobalOutsideUnionFind;
use crate::event_scalar_common::{
    DeltaEvent, DeltaRunReader, PairEvent, PairRunReader, ScalarEventCurveStats, SlabDescriptor,
    create_buffered_file, local_boundary_node_id, lower_face_global_id, make_slab_descriptors,
    read_scalar_key, report_scalar_event_memory_estimate, sorted_scalar_indices,
    upper_face_global_id, write_delta_event, write_pair_event, write_scalar_key,
};
use crate::interface_sparsify_scalar::sparsify_scalar_superlevel_interface;
use crate::io_scalar::{ScalarBlock, ScalarTiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::scalar::ScalarKey;
use crate::temp_runs::TempRunDirectory;

const NO_INTERFACE_REP: u32 = u32::MAX;
const EVENT_BUFFER_LIMIT: usize = 1 << 20;
const CURVE_RECORD_BYTES: u64 = 16;
const REVERSE_BLOCK_RECORDS: u64 = 65_536;

#[derive(Debug)]
struct LocalBackgroundUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    interface_rep: Vec<u32>,
    touches_outside: Vec<bool>,
}

#[derive(Debug, Clone, Copy)]
struct LocalUnionOutcome {
    local_delta: i64,
    interface_merge: Option<(u32, u32)>,
    outside_event: Option<u32>,
}

impl LocalBackgroundUnionFind {
    fn new(voxel_count: usize) -> Self {
        assert!(
            voxel_count <= u32::MAX as usize,
            "scalar slab has too many voxels for u32 union-find indices"
        );
        Self {
            parent: (0..voxel_count as u32).collect(),
            rank: vec![0u8; voxel_count],
            interface_rep: vec![NO_INTERFACE_REP; voxel_count],
            touches_outside: vec![false; voxel_count],
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

    fn mark_outside(&mut self, node: u32) {
        let root = self.find(node);
        self.touches_outside[root as usize] = true;
    }

    fn union_with_events(&mut self, a: u32, b: u32) -> Option<LocalUnionOutcome> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let rep_a = self.interface_rep[root_a as usize];
        let rep_b = self.interface_rep[root_b as usize];
        let outside_a = self.touches_outside[root_a as usize];
        let outside_b = self.touches_outside[root_b as usize];
        let local_delta = if outside_a && outside_b { 0 } else { -1 };

        let interface_merge =
            if rep_a != NO_INTERFACE_REP && rep_b != NO_INTERFACE_REP && rep_a != rep_b {
                Some((rep_a, rep_b))
            } else {
                None
            };
        let outside_event = if rep_a != NO_INTERFACE_REP
            && rep_b == NO_INTERFACE_REP
            && !outside_a
            && outside_b
        {
            Some(rep_a)
        } else if rep_a == NO_INTERFACE_REP && rep_b != NO_INTERFACE_REP && outside_a && !outside_b
        {
            Some(rep_b)
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
        self.touches_outside[root_a as usize] = outside_a || outside_b;

        Some(LocalUnionOutcome {
            local_delta,
            interface_merge,
            outside_event,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MergeEvent {
    value: ScalarKey,
    a: u32,
    b: u32,
    local_delta: i64,
}

fn write_merge_event(writer: &mut impl Write, event: MergeEvent) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    writer.write_all(&event.local_delta.to_le_bytes())?;
    Ok(())
}

fn read_merge_event(reader: &mut impl Read) -> Result<Option<MergeEvent>> {
    let Some(value) = read_scalar_key(reader)? else {
        return Ok(None);
    };
    let mut a_bytes = [0u8; 4];
    let mut b_bytes = [0u8; 4];
    let mut delta_bytes = [0u8; 8];
    reader.read_exact(&mut a_bytes)?;
    reader.read_exact(&mut b_bytes)?;
    reader.read_exact(&mut delta_bytes)?;
    Ok(Some(MergeEvent {
        value,
        a: u32::from_le_bytes(a_bytes),
        b: u32::from_le_bytes(b_bytes),
        local_delta: i64::from_le_bytes(delta_bytes),
    }))
}

struct MergeRunReader {
    reader: BufReader<File>,
    next: Option<MergeEvent>,
}

impl MergeRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_merge_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn peek_value(&self) -> Option<ScalarKey> {
        self.next.map(|event| event.value)
    }

    fn pop(&mut self) -> Result<Option<MergeEvent>> {
        let current = self.next.take();
        if current.is_some() {
            self.next = read_merge_event(&mut self.reader)?;
        }
        Ok(current)
    }
}

#[derive(Debug, Clone, Copy)]
struct OutsideEvent {
    value: ScalarKey,
    node: u32,
    local_delta: i64,
}

fn write_outside_event(writer: &mut impl Write, event: OutsideEvent) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.node.to_le_bytes())?;
    writer.write_all(&event.local_delta.to_le_bytes())?;
    Ok(())
}

fn read_outside_event(reader: &mut impl Read) -> Result<Option<OutsideEvent>> {
    let Some(value) = read_scalar_key(reader)? else {
        return Ok(None);
    };
    let mut node_bytes = [0u8; 4];
    let mut delta_bytes = [0u8; 8];
    reader.read_exact(&mut node_bytes)?;
    reader.read_exact(&mut delta_bytes)?;
    Ok(Some(OutsideEvent {
        value,
        node: u32::from_le_bytes(node_bytes),
        local_delta: i64::from_le_bytes(delta_bytes),
    }))
}

struct OutsideRunReader {
    reader: BufReader<File>,
    next: Option<OutsideEvent>,
}

impl OutsideRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_outside_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn peek_value(&self) -> Option<ScalarKey> {
        self.next.map(|event| event.value)
    }

    fn pop(&mut self) -> Result<Option<OutsideEvent>> {
        let current = self.next.take();
        if current.is_some() {
            self.next = read_outside_event(&mut self.reader)?;
        }
        Ok(current)
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

fn voxel_touches_global_boundary(
    block: &ScalarBlock,
    x: usize,
    y: usize,
    z_local: usize,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
) -> bool {
    let z_global = block.z0 + z_local;
    x == 0
        || x + 1 == global_width
        || y == 0
        || y + 1 == global_height
        || z_global == 0
        || z_global + 1 == global_depth
}

fn flush_merge_events(writer: &mut BufWriter<File>, pending: &mut Vec<MergeEvent>) -> Result<()> {
    for event in pending.drain(..) {
        write_merge_event(writer, event)?;
    }
    Ok(())
}

fn flush_outside_events(
    writer: &mut BufWriter<File>,
    pending: &mut Vec<OutsideEvent>,
) -> Result<()> {
    for event in pending.drain(..) {
        write_outside_event(writer, event)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_scalar_h2_slab(
    desc: &SlabDescriptor,
    block: &ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    delta_writer: &mut BufWriter<File>,
    merge_writer: &mut BufWriter<File>,
    outside_writer: &mut BufWriter<File>,
) -> Result<ScalarBoundaryFaces> {
    let voxel_count = block.voxel_count();
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let face_size = width
        .checked_mul(height)
        .expect("scalar H2 face size overflow");

    let mut union_find = LocalBackgroundUnionFind::new(voxel_count);
    let mut active = vec![0u8; voxel_count];
    let order = sorted_scalar_indices(&block.values, block.pixel_type);
    let mut local_pruner = NeighborhoodComponentPruner::new(background_connectivity);
    let mut pending_merge_events = Vec::new();
    let mut pending_outside_events = Vec::new();

    let mut group_end = order.len();
    while group_end > 0 {
        let value = block.values[order[group_end - 1] as usize];
        let mut group_start = group_end - 1;
        while group_start > 0 && block.values[order[group_start - 1] as usize] == value {
            group_start -= 1;
        }

        let mut delta = 0i64;
        for &index_u32 in &order[group_start..group_end] {
            let index = index_u32 as usize;
            active[index] = 1;
            let x = index % width;
            let y = (index / width) % height;
            let z = index / face_size;
            let face_index = y * width + x;
            let local_interface_node = local_boundary_node_id(z, depth, face_index, face_size);

            if let Some(node) = local_interface_node {
                union_find.set_interface_rep(index_u32, node);
            }

            if voxel_touches_global_boundary(
                block,
                x,
                y,
                z,
                global_width,
                global_height,
                global_depth,
            ) {
                union_find.mark_outside(index_u32);
                if let Some(node) = local_interface_node {
                    pending_outside_events.push(OutsideEvent {
                        value,
                        node: desc.interface_base + node,
                        local_delta: 0,
                    });
                }
            } else {
                delta += 1;
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
                    if let Some(outcome) = union_find.union_with_events(index_u32, neighbor as u32)
                    {
                        delta += outcome.local_delta;
                        if let Some((a, b)) = outcome.interface_merge {
                            pending_merge_events.push(MergeEvent {
                                value,
                                a: desc.interface_base + a,
                                b: desc.interface_base + b,
                                local_delta: outcome.local_delta,
                            });
                        }
                        if let Some(node) = outcome.outside_event {
                            pending_outside_events.push(OutsideEvent {
                                value,
                                node: desc.interface_base + node,
                                local_delta: outcome.local_delta,
                            });
                        }
                    }
                },
            );

            if pending_merge_events.len() >= EVENT_BUFFER_LIMIT {
                flush_merge_events(merge_writer, &mut pending_merge_events)?;
            }
            if pending_outside_events.len() >= EVENT_BUFFER_LIMIT {
                flush_outside_events(outside_writer, &mut pending_outside_events)?;
            }
        }

        if delta != 0 {
            write_delta_event(delta_writer, DeltaEvent { value, delta })?;
        }
        group_end = group_start;
    }

    flush_merge_events(merge_writer, &mut pending_merge_events)?;
    flush_outside_events(outside_writer, &mut pending_outside_events)?;

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
    background_connectivity: Connectivity,
) -> Result<PathBuf> {
    let [width, height] = face_shape;
    let face_size = width.checked_mul(height).expect("face size overflow");
    let path = directory.join(format!("betti2_scalar_cross_{pair_id:05}.bin"));
    let mut writer = create_buffered_file(&path)?;
    let stats = sparsify_scalar_superlevel_interface(
        &left.values,
        right_lower_values,
        width,
        height,
        background_connectivity,
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
        "Scalar Betti-2 interface {pair_id}: retained {} of {} cross edges",
        stats.retained_edges, stats.candidate_edges
    );
    Ok(path)
}

struct PreparedScalarBetti2Runs {
    total_interface_nodes: usize,
    minimum: ScalarKey,
    maximum: ScalarKey,
    delta_run_paths: Vec<PathBuf>,
    merge_run_paths: Vec<PathBuf>,
    outside_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
}

fn prepare_scalar_betti2_runs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    directory: &Path,
) -> Result<PreparedScalarBetti2Runs> {
    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth)?;
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);
    let [global_width, global_height, global_depth] = volume.shape();

    let mut minimum: Option<ScalarKey> = None;
    let mut maximum: Option<ScalarKey> = None;
    let mut delta_run_paths = Vec::with_capacity(descriptors.len());
    let mut merge_run_paths = Vec::with_capacity(descriptors.len());
    let mut outside_run_paths = Vec::with_capacity(descriptors.len());
    let mut cross_run_paths = Vec::with_capacity(descriptors.len().saturating_sub(1));
    let mut previous_upper_face: Option<StoredUpperFace> = None;

    for desc in &descriptors {
        println!(
            "Preparing scalar Betti-2 slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );
        let delta_path = directory.join(format!("betti2_scalar_delta_{:05}.bin", desc.slab_id));
        let merge_path = directory.join(format!("betti2_scalar_merge_{:05}.bin", desc.slab_id));
        let outside_path = directory.join(format!("betti2_scalar_outside_{:05}.bin", desc.slab_id));
        let mut delta_writer = create_buffered_file(&delta_path)?;
        let mut merge_writer = create_buffered_file(&merge_path)?;
        let mut outside_writer = create_buffered_file(&outside_path)?;

        let block = volume.read_z_slab(desc.z0, desc.z1)?;
        if let Some(block_minimum) = block.values.iter().copied().min() {
            minimum = Some(minimum.map_or(block_minimum, |value| value.min(block_minimum)));
        }
        if let Some(block_maximum) = block.values.iter().copied().max() {
            maximum = Some(maximum.map_or(block_maximum, |value| value.max(block_maximum)));
        }

        let faces = process_scalar_h2_slab(
            desc,
            &block,
            global_width,
            global_height,
            global_depth,
            background_connectivity,
            &mut delta_writer,
            &mut merge_writer,
            &mut outside_writer,
        )?;
        delta_writer.flush()?;
        merge_writer.flush()?;
        outside_writer.flush()?;
        drop(block);

        if let Some(previous) = previous_upper_face.take() {
            cross_run_paths.push(write_cross_run(
                directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &faces.lower_values,
                [volume.width, volume.height],
                background_connectivity,
            )?);
        }
        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: faces.upper_values,
        });

        delta_run_paths.push(delta_path);
        merge_run_paths.push(merge_path);
        outside_run_paths.push(outside_path);
    }

    Ok(PreparedScalarBetti2Runs {
        total_interface_nodes,
        minimum: minimum.ok_or_else(|| anyhow::anyhow!("scalar volume is empty"))?,
        maximum: maximum.ok_or_else(|| anyhow::anyhow!("scalar volume is empty"))?,
        delta_run_paths,
        merge_run_paths,
        outside_run_paths,
        cross_run_paths,
    })
}

type MaxHeap = BinaryHeap<(ScalarKey, usize)>;

fn open_delta_runs(paths: &[PathBuf]) -> Result<(Vec<DeltaRunReader>, MaxHeap)> {
    let readers = paths
        .iter()
        .map(|path| DeltaRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push((value, index));
        }
    }
    Ok((readers, heap))
}

fn open_pair_runs(paths: &[PathBuf]) -> Result<(Vec<PairRunReader>, MaxHeap)> {
    let readers = paths
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push((value, index));
        }
    }
    Ok((readers, heap))
}

fn open_merge_runs(paths: &[PathBuf]) -> Result<(Vec<MergeRunReader>, MaxHeap)> {
    let readers = paths
        .iter()
        .map(|path| MergeRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push((value, index));
        }
    }
    Ok((readers, heap))
}

fn open_outside_runs(paths: &[PathBuf]) -> Result<(Vec<OutsideRunReader>, MaxHeap)> {
    let readers = paths
        .iter()
        .map(|path| OutsideRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push((value, index));
        }
    }
    Ok((readers, heap))
}

fn max_heap_value(heap: &MaxHeap) -> Option<ScalarKey> {
    heap.peek().map(|(value, _)| *value)
}

fn next_value(heaps: [&MaxHeap; 4]) -> Option<ScalarKey> {
    heaps.into_iter().filter_map(max_heap_value).max()
}

fn drain_delta_value(
    value: ScalarKey,
    readers: &mut [DeltaRunReader],
    heap: &mut MaxHeap,
) -> Result<i64> {
    let mut total = 0i64;
    while max_heap_value(heap) == Some(value) {
        let (_, reader_index) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-2 delta run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-2 delta heap is out of order");
        }
        total += event.delta;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push((next, reader_index));
        }
    }
    Ok(total)
}

fn drain_pair_value(
    value: ScalarKey,
    readers: &mut [PairRunReader],
    heap: &mut MaxHeap,
    mut apply: impl FnMut(PairEvent) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, reader_index) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-2 cross run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-2 cross heap is out of order");
        }
        apply(event)?;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push((next, reader_index));
        }
    }
    Ok(())
}

fn drain_merge_value(
    value: ScalarKey,
    readers: &mut [MergeRunReader],
    heap: &mut MaxHeap,
    mut apply: impl FnMut(MergeEvent) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, reader_index) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-2 merge run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-2 merge heap is out of order");
        }
        apply(event)?;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push((next, reader_index));
        }
    }
    Ok(())
}

fn drain_outside_value(
    value: ScalarKey,
    readers: &mut [OutsideRunReader],
    heap: &mut MaxHeap,
    mut apply: impl FnMut(OutsideEvent) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, reader_index) = heap.pop().expect("heap head disappeared");
        let event = readers[reader_index]
            .pop()?
            .ok_or_else(|| anyhow::anyhow!("scalar Betti-2 outside run ended unexpectedly"))?;
        if event.value != value {
            bail!("scalar Betti-2 outside heap is out of order");
        }
        apply(event)?;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push((next, reader_index));
        }
    }
    Ok(())
}

fn write_curve_record(writer: &mut impl Write, threshold: ScalarKey, beta2: i64) -> Result<()> {
    write_scalar_key(writer, threshold)?;
    writer.write_all(&beta2.to_le_bytes())?;
    Ok(())
}

struct DescendingCurveCompressor<'a> {
    writer: &'a mut BufWriter<File>,
    plateau_beta2: i64,
    plateau_lowest_threshold: ScalarKey,
    rows: u64,
}

impl<'a> DescendingCurveCompressor<'a> {
    fn new(writer: &'a mut BufWriter<File>, maximum: ScalarKey) -> Self {
        Self {
            writer,
            plateau_beta2: 0,
            plateau_lowest_threshold: maximum,
            rows: 0,
        }
    }

    fn observe(&mut self, threshold: ScalarKey, beta2: i64) -> Result<()> {
        if threshold > self.plateau_lowest_threshold {
            bail!("scalar Betti-2 thresholds were not observed in descending order");
        }
        if beta2 == self.plateau_beta2 {
            self.plateau_lowest_threshold = threshold;
        } else {
            write_curve_record(
                self.writer,
                self.plateau_lowest_threshold,
                self.plateau_beta2,
            )?;
            self.rows += 1;
            self.plateau_beta2 = beta2;
            self.plateau_lowest_threshold = threshold;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        write_curve_record(
            self.writer,
            self.plateau_lowest_threshold,
            self.plateau_beta2,
        )?;
        self.rows += 1;
        Ok(self.rows)
    }
}

fn reverse_curve_records_to_csv(
    reverse_path: &Path,
    output_path: &Path,
    expected_rows: u64,
) -> Result<()> {
    let mut input = File::open(reverse_path)?;
    let byte_length = input.metadata()?.len();
    if byte_length % CURVE_RECORD_BYTES != 0 {
        bail!("scalar Betti-2 reverse curve file is truncated");
    }
    let record_count = byte_length / CURVE_RECORD_BYTES;
    if record_count != expected_rows {
        bail!(
            "scalar Betti-2 reverse curve row mismatch: expected {expected_rows}, found {record_count}"
        );
    }

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar Betti-2 output {output_path:?}"))?;
    writeln!(output, "threshold,betti2")?;

    let mut end_record = record_count;
    while end_record > 0 {
        let start_record = end_record.saturating_sub(REVERSE_BLOCK_RECORDS);
        let records_in_block = end_record - start_record;
        let bytes_in_block = usize::try_from(records_in_block * CURVE_RECORD_BYTES)
            .expect("curve reverse block is too large");
        let mut bytes = vec![0u8; bytes_in_block];
        input.seek(SeekFrom::Start(start_record * CURVE_RECORD_BYTES))?;
        input.read_exact(&mut bytes)?;

        for record in bytes.chunks_exact(CURVE_RECORD_BYTES as usize).rev() {
            let threshold = ScalarKey::from_raw(u64::from_le_bytes(
                record[..8].try_into().expect("invalid threshold field"),
            ));
            let beta2 =
                i64::from_le_bytes(record[8..16].try_into().expect("invalid Betti-2 field"));
            writeln!(output, "{threshold},{beta2}")?;
        }
        end_record = start_record;
    }
    output.commit()
}

fn reduce_scalar_betti2_runs(
    prepared: &PreparedScalarBetti2Runs,
    reverse_curve_path: &Path,
    output_path: &Path,
) -> Result<ScalarEventCurveStats> {
    println!(
        "Allocating global scalar Betti-2 interface union-find for {} nodes",
        prepared.total_interface_nodes
    );
    let mut global_union_find = GlobalOutsideUnionFind::new(prepared.total_interface_nodes);
    let (mut delta_readers, mut delta_heap) = open_delta_runs(&prepared.delta_run_paths)?;
    let (mut outside_readers, mut outside_heap) = open_outside_runs(&prepared.outside_run_paths)?;
    let (mut merge_readers, mut merge_heap) = open_merge_runs(&prepared.merge_run_paths)?;
    let (mut cross_readers, mut cross_heap) = open_pair_runs(&prepared.cross_run_paths)?;

    let mut reverse_writer = create_buffered_file(reverse_curve_path)?;
    let mut compressor = DescendingCurveCompressor::new(&mut reverse_writer, prepared.maximum);
    let mut last_observed = prepared.maximum;
    let mut beta2 = 0i64;

    while let Some(value) = next_value([&delta_heap, &outside_heap, &merge_heap, &cross_heap]) {
        if value != last_observed {
            compressor.observe(value, beta2)?;
            last_observed = value;
        }
        if value == prepared.minimum {
            break;
        }

        beta2 += drain_delta_value(value, &mut delta_readers, &mut delta_heap)?;
        drain_outside_value(value, &mut outside_readers, &mut outside_heap, |event| {
            match event.local_delta {
                0 => global_union_find.mark_outside(event.node),
                -1 => {
                    if global_union_find.is_outside(event.node) {
                        beta2 += 1;
                    } else {
                        global_union_find.mark_outside(event.node);
                    }
                }
                other => bail!("unexpected scalar outside-event delta: {other}"),
            }
            Ok(())
        })?;
        drain_merge_value(value, &mut merge_readers, &mut merge_heap, |event| {
            let actual_delta = match global_union_find.union(event.a, event.b) {
                Some((outside_a, outside_b)) if outside_a && outside_b => 0,
                Some(_) => -1,
                None => 0,
            };
            beta2 += actual_delta - event.local_delta;
            Ok(())
        })?;
        drain_pair_value(value, &mut cross_readers, &mut cross_heap, |event| {
            if let Some((outside_a, outside_b)) = global_union_find.union(event.a, event.b)
                && !(outside_a && outside_b)
            {
                beta2 -= 1;
            }
            Ok(())
        })?;

        if beta2 < 0 {
            bail!("scalar Betti-2 became negative below threshold {value}: {beta2}");
        }
    }

    if last_observed != prepared.minimum {
        compressor.observe(prepared.minimum, beta2)?;
    }
    let rows = compressor.finish()?;
    reverse_writer.flush()?;
    drop(reverse_writer);
    reverse_curve_records_to_csv(reverse_curve_path, output_path, rows)?;
    Ok(ScalarEventCurveStats { rows })
}

pub fn compute_event_based_betti2_scalar_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    output_path: &Path,
) -> Result<ScalarEventCurveStats> {
    let start = Instant::now();
    report_scalar_event_memory_estimate(
        volume.width,
        volume.height,
        volume.depth,
        slab_depth,
        background_connectivity,
        true,
    )?;
    let temp_directory = TempRunDirectory::create("betti2_scalar_stream_runs")?;
    println!(
        "Scalar Betti-2 streaming temporary directory: {:?}",
        temp_directory.path()
    );

    let prepared = prepare_scalar_betti2_runs(
        volume,
        slab_depth,
        background_connectivity,
        temp_directory.path(),
    )?;
    println!("Scalar Betti-2 slab preparation completed.");
    println!("Starting disk-backed scalar Betti-2 curve reduction.");

    let reverse_curve_path = temp_directory
        .path()
        .join("betti2_scalar_curve_reverse.bin");
    let result = reduce_scalar_betti2_runs(&prepared, &reverse_curve_path, output_path);
    let stats = result?;
    temp_directory.close()?;
    println!(
        "Streaming scalar Betti-2 curve computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(stats)
}
