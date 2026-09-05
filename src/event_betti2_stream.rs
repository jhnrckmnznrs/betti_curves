use anyhow::{Result, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::event_betti2::{
    GlobalOutsideUnionFind, InterfaceMergeEvent, InterfaceOutsideEvent,
    process_slab_event_based_betti2,
};
use crate::interface_sparsify::{InterfaceFiltration, sparsify_cross_interface};
use crate::io::TiffStackReader;
use crate::temp_runs::TempRunDirectory;

const NUM_U16_VALUES: usize = 65_536;

#[derive(Debug, Clone)]
struct SlabDescriptor {
    slab_id: usize,
    z0: usize,
    z1: usize,
    local_depth: usize,
    interface_base: u32,
    interface_count: u32,
}

fn make_slab_descriptors(
    width: usize,
    height: usize,
    depth: usize,
    slab_depth: usize,
) -> Vec<SlabDescriptor> {
    assert!(slab_depth > 0, "slab_depth must be positive");

    let face_size = width.checked_mul(height).expect("face size overflow");
    let mut descriptors = Vec::new();
    let mut next_interface_id = 0u64;
    let mut slab_id = 0usize;
    let mut z0 = 0usize;

    while z0 < depth {
        let z1 = z0.saturating_add(slab_depth).min(depth);
        let local_depth = z1 - z0;

        let interface_count = if local_depth == 1 {
            face_size
        } else {
            2 * face_size
        };

        let interface_base =
            u32::try_from(next_interface_id).expect("too many global interface nodes");
        let interface_count =
            u32::try_from(interface_count).expect("too many interface nodes in one slab");

        descriptors.push(SlabDescriptor {
            slab_id,
            z0,
            z1,
            local_depth,
            interface_base,
            interface_count,
        });

        next_interface_id = next_interface_id
            .checked_add(u64::from(interface_count))
            .expect("interface node count overflow");

        assert!(
            next_interface_id <= u32::MAX as u64,
            "too many global interface nodes for u32 IDs"
        );

        slab_id += 1;
        z0 = z1;
    }

    descriptors
}

fn lower_face_global_id(desc: &SlabDescriptor, face_index: usize) -> u32 {
    desc.interface_base + u32::try_from(face_index).expect("face index overflow")
}

fn upper_face_global_id(desc: &SlabDescriptor, face_size: usize, face_index: usize) -> u32 {
    let local_id = if desc.local_depth == 1 {
        face_index
    } else {
        face_size + face_index
    };

    desc.interface_base + u32::try_from(local_id).expect("face index overflow")
}

struct StoredUpperFace {
    descriptor: SlabDescriptor,
    values: Vec<u16>,
}

#[derive(Debug, Clone, Copy)]
struct PairEvent {
    value: u16,
    a: u32,
    b: u32,
}

#[derive(Debug, Clone, Copy)]
struct MergeEvent {
    value: u16,
    a: u32,
    b: u32,
    local_delta: i64,
}

#[derive(Debug, Clone, Copy)]
struct OutsideEvent {
    value: u16,
    node: u32,
    local_delta: i64,
}

fn read_first_field(reader: &mut BufReader<File>) -> Result<Option<[u8; 2]>> {
    let mut value_bytes = [0u8; 2];

    if read_exact_or_eof(reader, &mut value_bytes)? {
        Ok(Some(value_bytes))
    } else {
        Ok(None)
    }
}

fn write_pair_event(writer: &mut BufWriter<File>, event: PairEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair_event(reader: &mut BufReader<File>) -> Result<Option<PairEvent>> {
    let Some(value_bytes) = read_first_field(reader)? else {
        return Ok(None);
    };

    let mut a_bytes = [0u8; 4];
    let mut b_bytes = [0u8; 4];
    reader.read_exact(&mut a_bytes)?;
    reader.read_exact(&mut b_bytes)?;

    Ok(Some(PairEvent {
        value: u16::from_le_bytes(value_bytes),
        a: u32::from_le_bytes(a_bytes),
        b: u32::from_le_bytes(b_bytes),
    }))
}

fn write_merge_event(writer: &mut BufWriter<File>, event: MergeEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    writer.write_all(&event.local_delta.to_le_bytes())?;
    Ok(())
}

fn read_merge_event(reader: &mut BufReader<File>) -> Result<Option<MergeEvent>> {
    let Some(value_bytes) = read_first_field(reader)? else {
        return Ok(None);
    };

    let mut a_bytes = [0u8; 4];
    let mut b_bytes = [0u8; 4];
    let mut delta_bytes = [0u8; 8];

    reader.read_exact(&mut a_bytes)?;
    reader.read_exact(&mut b_bytes)?;
    reader.read_exact(&mut delta_bytes)?;

    Ok(Some(MergeEvent {
        value: u16::from_le_bytes(value_bytes),
        a: u32::from_le_bytes(a_bytes),
        b: u32::from_le_bytes(b_bytes),
        local_delta: i64::from_le_bytes(delta_bytes),
    }))
}

fn write_outside_event(writer: &mut BufWriter<File>, event: OutsideEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.node.to_le_bytes())?;
    writer.write_all(&event.local_delta.to_le_bytes())?;
    Ok(())
}

fn read_outside_event(reader: &mut BufReader<File>) -> Result<Option<OutsideEvent>> {
    let Some(value_bytes) = read_first_field(reader)? else {
        return Ok(None);
    };

    let mut node_bytes = [0u8; 4];
    let mut delta_bytes = [0u8; 8];

    reader.read_exact(&mut node_bytes)?;
    reader.read_exact(&mut delta_bytes)?;

    Ok(Some(OutsideEvent {
        value: u16::from_le_bytes(value_bytes),
        node: u32::from_le_bytes(node_bytes),
        local_delta: i64::from_le_bytes(delta_bytes),
    }))
}

struct PairRunReader {
    reader: BufReader<File>,
    next: Option<PairEvent>,
}

impl PairRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_pair_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_at_value(&mut self, value: u16) -> Result<Option<(u32, u32)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_pair_event(&mut self.reader)?;
                Ok(Some((event.a, event.b)))
            }
            _ => Ok(None),
        }
    }
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

    fn pop_at_value(&mut self, value: u16) -> Result<Option<(u32, u32, i64)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_merge_event(&mut self.reader)?;
                Ok(Some((event.a, event.b, event.local_delta)))
            }
            _ => Ok(None),
        }
    }
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

    fn pop_at_value(&mut self, value: u16) -> Result<Option<(u32, i64)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_outside_event(&mut self.reader)?;
                Ok(Some((event.node, event.local_delta)))
            }
            _ => Ok(None),
        }
    }
}

fn write_interface_merge_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[InterfaceMergeEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("betti2_interface_{:05}.bin", desc.slab_id));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    for event in events {
        write_merge_event(
            &mut writer,
            MergeEvent {
                value: event.value,
                a: desc.interface_base + event.a,
                b: desc.interface_base + event.b,
                local_delta: event.local_delta,
            },
        )?;
    }

    writer.flush()?;
    Ok(path)
}

fn write_outside_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[InterfaceOutsideEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("betti2_outside_{:05}.bin", desc.slab_id));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    for event in events {
        write_outside_event(
            &mut writer,
            OutsideEvent {
                value: event.value,
                node: desc.interface_base + event.node,
                local_delta: event.local_delta,
            },
        )?;
    }

    writer.flush()?;
    Ok(path)
}

fn write_betti2_cross_run(
    directory: &Path,
    pair_id: usize,
    left: &StoredUpperFace,
    right_desc: &SlabDescriptor,
    right_lower_values: &[u16],
    face_shape: [usize; 2],
    connectivity: Connectivity,
) -> Result<PathBuf> {
    let [width, height] = face_shape;
    let face_size = width.checked_mul(height).expect("face size overflow");

    assert_eq!(left.values.len(), face_size);
    assert_eq!(right_lower_values.len(), face_size);

    let path = directory.join(format!("betti2_cross_{pair_id:05}.bin"));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    let stats = sparsify_cross_interface(
        &left.values,
        right_lower_values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SuperlevelMin,
        |edge| {
            let left_index = edge.left_face_index as usize;
            let right_index = edge.right_face_index as usize;

            write_pair_event(
                &mut writer,
                PairEvent {
                    value: edge.value,
                    a: upper_face_global_id(&left.descriptor, face_size, left_index),
                    b: lower_face_global_id(right_desc, right_index),
                },
            )
        },
    )?;

    writer.flush()?;

    println!(
        "Betti-2 interface {pair_id}: retained {} of {} cross edges",
        stats.retained_edges, stats.candidate_edges
    );

    Ok(path)
}

struct PreparedBetti2Runs {
    local_delta_by_value: Vec<i64>,
    total_interface_nodes: usize,
    interface_run_paths: Vec<PathBuf>,
    outside_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
    max_value: u16,
}

fn prepare_event_betti2_runs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    temp_directory: &Path,
) -> Result<PreparedBetti2Runs> {
    std::fs::create_dir_all(temp_directory)?;

    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let [global_width, global_height, global_depth] = volume.shape();

    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let mut local_delta_by_value = vec![0i64; NUM_U16_VALUES];
    let mut interface_run_paths = Vec::new();
    let mut outside_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace> = None;
    let mut max_value = 0u16;

    for desc in &descriptors {
        println!(
            "Preparing Betti-2 slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );

        let block = volume.read_z_slab(desc.z0, desc.z1)?;

        if let Some(block_max) = block.values.iter().copied().max() {
            max_value = max_value.max(block_max);
        }

        let summary = process_slab_event_based_betti2(
            desc.slab_id,
            &block,
            global_width,
            global_height,
            global_depth,
            background_connectivity,
        );

        assert_eq!(summary.interface_node_count, desc.interface_count);
        assert_eq!(summary.z_min_face.width, volume.width);
        assert_eq!(summary.z_min_face.height, volume.height);
        assert_eq!(summary.z_max_face.width, volume.width);
        assert_eq!(summary.z_max_face.height, volume.height);

        for event in &summary.local_delta_events {
            local_delta_by_value[event.value as usize] += event.delta;
        }

        interface_run_paths.push(write_interface_merge_run(
            temp_directory,
            desc,
            &summary.interface_merge_events,
        )?);

        outside_run_paths.push(write_outside_run(
            temp_directory,
            desc,
            &summary.interface_outside_events,
        )?);

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;

        if let Some(previous) = previous_upper_face.take() {
            cross_run_paths.push(write_betti2_cross_run(
                temp_directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &lower_values,
                [volume.width, volume.height],
                background_connectivity,
            )?);
        }

        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: upper_values,
        });
    }

    Ok(PreparedBetti2Runs {
        local_delta_by_value,
        total_interface_nodes,
        interface_run_paths,
        outside_run_paths,
        cross_run_paths,
        max_value,
    })
}

fn reduce_event_betti2_runs(prepared: &PreparedBetti2Runs) -> Result<Vec<(u16, i64)>> {
    println!(
        "Allocating global Betti-2 interface union-find for {} nodes",
        prepared.total_interface_nodes
    );

    let mut global_uf = GlobalOutsideUnionFind::new(prepared.total_interface_nodes);

    let mut outside_readers = prepared
        .outside_run_paths
        .iter()
        .map(|path| OutsideRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    let mut interface_readers = prepared
        .interface_run_paths
        .iter()
        .map(|path| MergeRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    let mut cross_readers = prepared
        .cross_run_paths
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    let max_value = prepared.max_value as usize;
    let mut dense_betti2 = vec![0i64; max_value + 1];
    let mut beta2 = 0i64;

    for s in (1..=max_value).rev() {
        beta2 += prepared.local_delta_by_value[s];
        let value = s as u16;

        for reader in &mut outside_readers {
            while let Some((node, local_delta)) = reader.pop_at_value(value)? {
                match local_delta {
                    0 => global_uf.mark_outside(node),
                    -1 => {
                        if global_uf.is_outside(node) {
                            beta2 += 1;
                        } else {
                            global_uf.mark_outside(node);
                        }
                    }
                    _ => bail!("unexpected outside event local_delta: {local_delta}"),
                }
            }
        }

        for reader in &mut interface_readers {
            while let Some((a, b, local_delta)) = reader.pop_at_value(value)? {
                let actual_delta = match global_uf.union(a, b) {
                    Some((a_outside, b_outside)) if a_outside && b_outside => 0,
                    Some(_) => -1,
                    None => 0,
                };

                beta2 += actual_delta - local_delta;
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(value)? {
                if let Some((a_outside, b_outside)) = global_uf.union(a, b)
                    && !(a_outside && b_outside)
                {
                    beta2 -= 1;
                }
            }
        }

        dense_betti2[s - 1] = beta2;
    }

    dense_betti2[max_value] = 0;

    let mut sparse_curve = Vec::new();
    let mut previous: Option<i64> = None;

    for (threshold, &value) in dense_betti2.iter().enumerate() {
        if previous != Some(value) {
            sparse_curve.push((threshold as u16, value));
            previous = Some(value);
        }
    }

    Ok(sparse_curve)
}

pub fn compute_event_based_betti2_stream_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let start = Instant::now();

    let temp_directory = TempRunDirectory::create("betti2_stream_runs")?;

    println!(
        "Streaming reducer temporary directory: {:?}",
        temp_directory.path()
    );

    let prepared = prepare_event_betti2_runs(
        volume,
        slab_depth,
        background_connectivity,
        temp_directory.path(),
    )?;

    println!("Betti-2 slab preparation completed.");
    println!("Starting disk-backed Betti-2 global reduction.");

    let result = reduce_event_betti2_runs(&prepared);
    let curve = result?;
    temp_directory.close()?;

    println!(
        "Streaming event-based Betti-2 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(curve)
}

pub fn write_event_betti2_stream_csv(path: &Path, curve: &[(u16, i64)]) -> Result<()> {
    let mut file = AtomicOutput::create(path)?;
    writeln!(file, "threshold,betti2")?;

    for &(threshold, beta2) in curve {
        writeln!(file, "{threshold},{beta2}")?;
    }

    file.commit()
}
