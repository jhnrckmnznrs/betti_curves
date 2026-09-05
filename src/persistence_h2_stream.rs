use anyhow::{Result, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::interface_sparsify::{InterfaceFiltration, sparsify_cross_interface};
use crate::io::TiffStackReader;
use crate::persistence_h2::{
    AttachEvent, FinitePair, GlobalBackgroundPersistenceUnionFind, InterfaceMergeEvent,
    NUM_U16_VALUES, OutsideEvent, process_slab_h2_persistence,
};
use crate::temp_runs::TempRunDirectory;

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
struct AttachDiskEvent {
    value: u16,
    node: u32,
    branch_birth: u16,
}

#[derive(Debug, Clone, Copy)]
struct OutsideDiskEvent {
    value: u16,
    node: u32,
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

fn write_attach_event(writer: &mut BufWriter<File>, event: AttachDiskEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.node.to_le_bytes())?;
    writer.write_all(&event.branch_birth.to_le_bytes())?;
    Ok(())
}

fn read_attach_event(reader: &mut BufReader<File>) -> Result<Option<AttachDiskEvent>> {
    let Some(value_bytes) = read_first_field(reader)? else {
        return Ok(None);
    };

    let mut node_bytes = [0u8; 4];
    let mut birth_bytes = [0u8; 2];
    reader.read_exact(&mut node_bytes)?;
    reader.read_exact(&mut birth_bytes)?;

    Ok(Some(AttachDiskEvent {
        value: u16::from_le_bytes(value_bytes),
        node: u32::from_le_bytes(node_bytes),
        branch_birth: u16::from_le_bytes(birth_bytes),
    }))
}

fn write_outside_event(writer: &mut BufWriter<File>, event: OutsideDiskEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.node.to_le_bytes())?;
    Ok(())
}

fn read_outside_event(reader: &mut BufReader<File>) -> Result<Option<OutsideDiskEvent>> {
    let Some(value_bytes) = read_first_field(reader)? else {
        return Ok(None);
    };

    let mut node_bytes = [0u8; 4];
    reader.read_exact(&mut node_bytes)?;

    Ok(Some(OutsideDiskEvent {
        value: u16::from_le_bytes(value_bytes),
        node: u32::from_le_bytes(node_bytes),
    }))
}

fn write_finite_pair(writer: &mut BufWriter<File>, pair: FinitePair) -> Result<()> {
    writer.write_all(&pair.birth.to_le_bytes())?;
    writer.write_all(&pair.death.to_le_bytes())?;
    Ok(())
}

fn read_finite_pair(reader: &mut BufReader<File>) -> Result<Option<FinitePair>> {
    let mut birth_bytes = [0u8; 2];
    if !read_exact_or_eof(reader, &mut birth_bytes)? {
        return Ok(None);
    }

    let mut death_bytes = [0u8; 2];
    reader.read_exact(&mut death_bytes)?;

    Ok(Some(FinitePair {
        birth: u16::from_le_bytes(birth_bytes),
        death: u16::from_le_bytes(death_bytes),
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

struct AttachRunReader {
    reader: BufReader<File>,
    next: Option<AttachDiskEvent>,
}

impl AttachRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_attach_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_at_value(&mut self, value: u16) -> Result<Option<(u32, u16)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_attach_event(&mut self.reader)?;
                Ok(Some((event.node, event.branch_birth)))
            }
            _ => Ok(None),
        }
    }
}

struct OutsideRunReader {
    reader: BufReader<File>,
    next: Option<OutsideDiskEvent>,
}

impl OutsideRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_outside_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_at_value(&mut self, value: u16) -> Result<Option<u32>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_outside_event(&mut self.reader)?;
                Ok(Some(event.node))
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
    let path = directory.join(format!("h2_interface_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

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

fn write_attach_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[AttachEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_attach_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

    for event in events {
        write_attach_event(
            &mut writer,
            AttachDiskEvent {
                value: event.value,
                node: desc.interface_base + event.interface_node,
                branch_birth: event.branch_birth,
            },
        )?;
    }

    writer.flush()?;
    Ok(path)
}

fn write_outside_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[OutsideEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_outside_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

    for event in events {
        write_outside_event(
            &mut writer,
            OutsideDiskEvent {
                value: event.value,
                node: desc.interface_base + event.interface_node,
            },
        )?;
    }

    writer.flush()?;
    Ok(path)
}

fn write_cross_run(
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

    let path = directory.join(format!("h2_cross_{pair_id:05}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);

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
        "H2 persistence interface {pair_id}: retained {} of {} cross edges",
        stats.retained_edges, stats.candidate_edges
    );
    Ok(path)
}

struct PreparedH2Runs {
    total_interface_nodes: usize,
    births_path: PathBuf,
    local_pairs_path: PathBuf,
    attach_run_paths: Vec<PathBuf>,
    outside_run_paths: Vec<PathBuf>,
    interface_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
}

fn prepare_h2_runs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    temp_directory: &Path,
) -> Result<PreparedH2Runs> {
    std::fs::create_dir_all(temp_directory)?;

    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let [global_width, global_height, global_depth] = volume.shape();
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let births_path = temp_directory.join("h2_interface_births.bin");
    let local_pairs_path = temp_directory.join("h2_local_pairs.bin");
    let mut birth_writer = BufWriter::new(File::create(&births_path)?);
    let mut local_pair_writer = BufWriter::new(File::create(&local_pairs_path)?);

    let mut attach_run_paths = Vec::new();
    let mut outside_run_paths = Vec::new();
    let mut interface_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace> = None;

    for desc in &descriptors {
        println!(
            "Preparing H2-persistence slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );

        let block = volume.read_z_slab(desc.z0, desc.z1)?;
        let summary = process_slab_h2_persistence(
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

        for &pair in &summary.finalized_pairs {
            write_finite_pair(&mut local_pair_writer, pair)?;
        }

        for &value in &summary.z_min_face.values {
            birth_writer.write_all(&value.to_le_bytes())?;
        }
        if desc.local_depth > 1 {
            for &value in &summary.z_max_face.values {
                birth_writer.write_all(&value.to_le_bytes())?;
            }
        }

        attach_run_paths.push(write_attach_run(
            temp_directory,
            desc,
            &summary.attach_events,
        )?);
        outside_run_paths.push(write_outside_run(
            temp_directory,
            desc,
            &summary.outside_events,
        )?);
        interface_run_paths.push(write_interface_merge_run(
            temp_directory,
            desc,
            &summary.interface_merge_events,
        )?);

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;

        if let Some(previous) = previous_upper_face.take() {
            cross_run_paths.push(write_cross_run(
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

    birth_writer.flush()?;
    local_pair_writer.flush()?;

    Ok(PreparedH2Runs {
        total_interface_nodes,
        births_path,
        local_pairs_path,
        attach_run_paths,
        outside_run_paths,
        interface_run_paths,
        cross_run_paths,
    })
}

fn read_interface_births(path: &Path, count: usize) -> Result<Vec<u16>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut births = Vec::with_capacity(count);

    for _ in 0..count {
        let mut bytes = [0u8; 2];
        reader.read_exact(&mut bytes)?;
        births.push(u16::from_le_bytes(bytes));
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("interface birth file contains trailing data");
    }

    Ok(births)
}

fn write_pair_csv(writer: &mut impl Write, pair: FinitePair) -> Result<bool> {
    if pair.birth >= pair.death {
        return Ok(false);
    }

    writeln!(writer, "{},{}", pair.birth, pair.death)?;
    Ok(true)
}

#[derive(Debug, Clone, Copy)]
pub struct H2PersistenceStats {
    pub finite_intervals: u64,
}

fn reduce_h2_runs(prepared: &PreparedH2Runs, output_path: &Path) -> Result<H2PersistenceStats> {
    let births = read_interface_births(&prepared.births_path, prepared.total_interface_nodes)?;
    let mut global_uf = GlobalBackgroundPersistenceUnionFind::new(births);

    let mut outside_readers = prepared
        .outside_run_paths
        .iter()
        .map(|path| OutsideRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut attach_readers = prepared
        .attach_run_paths
        .iter()
        .map(|path| AttachRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
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

    let mut output = AtomicOutput::create(output_path)?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    for value in (0..NUM_U16_VALUES).rev() {
        let merge_value = value as u16;

        for reader in &mut outside_readers {
            while let Some(node) = reader.pop_at_value(merge_value)? {
                if let Some(pair) = global_uf.connect_outside(node, merge_value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut attach_readers {
            while let Some((node, branch_birth)) = reader.pop_at_value(merge_value)? {
                if let Some(pair) = global_uf.attach_branch(node, branch_birth, merge_value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(merge_value)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, merge_value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(merge_value)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, merge_value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }
    }

    let remaining = global_uf.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "H2 persistence reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    output.commit()?;
    Ok(H2PersistenceStats { finite_intervals })
}

pub fn compute_h2_persistence_stream_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    output_path: &Path,
) -> Result<H2PersistenceStats> {
    let start = Instant::now();

    let temp_directory = TempRunDirectory::create("h2_stream_runs")?;

    println!(
        "H2 streaming temporary directory: {:?}",
        temp_directory.path()
    );

    let prepared = prepare_h2_runs(
        volume,
        slab_depth,
        background_connectivity,
        temp_directory.path(),
    )?;
    println!("H2 slab preparation completed.");
    println!("Starting disk-backed H2 persistence reduction.");

    let result = reduce_h2_runs(&prepared, output_path);
    let stats = result?;
    temp_directory.close()?;

    println!(
        "Streaming H2 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(stats)
}
