use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::event_betti0::{
    GlobalInterfaceUnionFind, InterfaceMergeEvent, process_slab_event_based_betti0,
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

fn write_pair_event(writer: &mut BufWriter<File>, event: PairEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair_event(reader: &mut BufReader<File>) -> Result<Option<PairEvent>> {
    let mut value_bytes = [0u8; 2];

    if !read_exact_or_eof(reader, &mut value_bytes)? {
        return Ok(None);
    }

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

fn write_interface_merge_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[InterfaceMergeEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("betti0_interface_{:05}.bin", desc.slab_id));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    for event in events {
        write_pair_event(
            &mut writer,
            PairEvent {
                value: event.value,
                a: desc.interface_base + event.a,
                b: desc.interface_base + event.b,
            },
        )?;
    }

    writer.flush()?;
    Ok(path)
}

fn write_betti0_cross_run(
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

    let path = directory.join(format!("betti0_cross_{pair_id:05}.bin"));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    let stats = sparsify_cross_interface(
        &left.values,
        right_lower_values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SublevelMax,
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
        "Betti-0 interface {pair_id}: retained {} of {} cross edges",
        stats.retained_edges, stats.candidate_edges
    );

    Ok(path)
}

struct PreparedBetti0Runs {
    local_delta_by_value: Vec<i64>,
    total_interface_nodes: usize,
    interface_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
}

fn prepare_event_betti0_runs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    temp_directory: &Path,
) -> Result<PreparedBetti0Runs> {
    std::fs::create_dir_all(temp_directory)?;

    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);

    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let mut local_delta_by_value = vec![0i64; NUM_U16_VALUES];
    let mut interface_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace> = None;

    for desc in &descriptors {
        println!(
            "Preparing Betti-0 slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );

        let block = volume.read_z_slab(desc.z0, desc.z1)?;
        let summary = process_slab_event_based_betti0(desc.slab_id, &block, connectivity);

        assert_eq!(summary.interface_node_count, desc.interface_count);
        assert_eq!(summary.z_min_face.width, volume.width);
        assert_eq!(summary.z_min_face.height, volume.height);
        assert_eq!(summary.z_max_face.width, volume.width);
        assert_eq!(summary.z_max_face.height, volume.height);

        for event in &summary.local_betti0_events {
            local_delta_by_value[event.value as usize] += event.delta;
        }

        interface_run_paths.push(write_interface_merge_run(
            temp_directory,
            desc,
            &summary.interface_merge_events,
        )?);

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;

        if let Some(previous) = previous_upper_face.take() {
            cross_run_paths.push(write_betti0_cross_run(
                temp_directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &lower_values,
                [volume.width, volume.height],
                connectivity,
            )?);
        }

        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: upper_values,
        });
    }

    Ok(PreparedBetti0Runs {
        local_delta_by_value,
        total_interface_nodes,
        interface_run_paths,
        cross_run_paths,
    })
}

fn reduce_event_betti0_runs(prepared: &PreparedBetti0Runs) -> Result<Vec<(u16, i64)>> {
    println!(
        "Allocating global Betti-0 interface union-find for {} nodes",
        prepared.total_interface_nodes
    );

    let mut global_uf = GlobalInterfaceUnionFind::new(prepared.total_interface_nodes);

    let mut interface_readers = prepared
        .interface_run_paths
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    let mut cross_readers = prepared
        .cross_run_paths
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    let mut beta0 = 0i64;
    let mut sparse_curve = Vec::new();
    let mut previous: Option<i64> = None;

    for value in 0..NUM_U16_VALUES {
        beta0 += prepared.local_delta_by_value[value];
        let value_u16 = value as u16;

        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(value_u16)? {
                if !global_uf.union(a, b) {
                    beta0 += 1;
                }
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(value_u16)? {
                if global_uf.union(a, b) {
                    beta0 -= 1;
                }
            }
        }

        debug_assert!(
            beta0 >= 0,
            "Betti-0 became negative at threshold {}: beta0 = {}",
            value,
            beta0
        );

        if previous != Some(beta0) {
            sparse_curve.push((value_u16, beta0));
            previous = Some(beta0);
        }
    }

    Ok(sparse_curve)
}

pub fn compute_event_based_betti0_stream_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let start = Instant::now();

    let temp_directory = TempRunDirectory::create("betti0_stream_runs")?;

    println!(
        "Streaming reducer temporary directory: {:?}",
        temp_directory.path()
    );

    let prepared =
        prepare_event_betti0_runs(volume, slab_depth, connectivity, temp_directory.path())?;

    println!("Betti-0 slab preparation completed.");
    println!("Starting disk-backed Betti-0 global reduction.");

    let result = reduce_event_betti0_runs(&prepared);
    let curve = result?;
    temp_directory.close()?;

    println!(
        "Streaming event-based Betti-0 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(curve)
}

pub fn write_event_betti0_stream_csv(path: &Path, curve: &[(u16, i64)]) -> Result<()> {
    let mut file = AtomicOutput::create(path)?;
    writeln!(file, "threshold,betti0")?;

    for &(threshold, beta0) in curve {
        writeln!(file, "{threshold},{beta0}")?;
    }

    file.commit()
}
