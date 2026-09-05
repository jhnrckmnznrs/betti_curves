use anyhow::{Result, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::io::TiffStackReader;
use crate::merge_tree_common::{H2Branch, H2BranchMerge, H2TreeRecorder, MergeTree};
use crate::merge_tree_h2::{
    AttachEvent, GlobalH2MergeTreeUnionFind, InterfaceMergeEvent, NUM_U16_VALUES,
    process_slab_h2_merge_tree,
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
    let mut next_interface = 0u64;
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
            u32::try_from(next_interface).expect("too many global interface nodes");
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

        next_interface = next_interface
            .checked_add(u64::from(interface_count))
            .expect("interface node count overflow");
        assert!(
            next_interface <= u32::MAX as u64,
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
    branch: H2Branch,
}

fn write_branch(writer: &mut BufWriter<File>, branch: H2Branch) -> Result<()> {
    match branch {
        H2Branch::Outside => {
            writer.write_all(&[0u8])?;
            writer.write_all(&0u64.to_le_bytes())?;
            writer.write_all(&0u16.to_le_bytes())?;
        }
        H2Branch::Finite { id, birth } => {
            writer.write_all(&[1u8])?;
            writer.write_all(&id.to_le_bytes())?;
            writer.write_all(&birth.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_branch(reader: &mut BufReader<File>) -> Result<H2Branch> {
    let mut tag = [0u8; 1];
    let mut id = [0u8; 8];
    let mut birth = [0u8; 2];
    reader.read_exact(&mut tag)?;
    reader.read_exact(&mut id)?;
    reader.read_exact(&mut birth)?;

    match tag[0] {
        0 => Ok(H2Branch::Outside),
        1 => Ok(H2Branch::Finite {
            id: u64::from_le_bytes(id),
            birth: u16::from_le_bytes(birth),
        }),
        other => bail!("invalid H2 branch tag {other}"),
    }
}

fn write_merge(writer: &mut BufWriter<File>, merge: H2BranchMerge) -> Result<()> {
    writer.write_all(&merge.value.to_le_bytes())?;
    writer.write_all(&merge.child_id.to_le_bytes())?;
    writer.write_all(&merge.child_birth.to_le_bytes())?;
    write_branch(writer, merge.parent)?;
    Ok(())
}

fn read_merge(reader: &mut BufReader<File>) -> Result<Option<H2BranchMerge>> {
    let mut value = [0u8; 2];
    if !read_exact_or_eof(reader, &mut value)? {
        return Ok(None);
    }
    let mut child_id = [0u8; 8];
    let mut child_birth = [0u8; 2];
    reader.read_exact(&mut child_id)?;
    reader.read_exact(&mut child_birth)?;
    Ok(Some(H2BranchMerge {
        value: u16::from_le_bytes(value),
        child_id: u64::from_le_bytes(child_id),
        child_birth: u16::from_le_bytes(child_birth),
        parent: read_branch(reader)?,
    }))
}

fn write_pair(writer: &mut BufWriter<File>, event: PairEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair(reader: &mut BufReader<File>) -> Result<Option<PairEvent>> {
    let mut value = [0u8; 2];
    if !read_exact_or_eof(reader, &mut value)? {
        return Ok(None);
    }
    let mut a = [0u8; 4];
    let mut b = [0u8; 4];
    reader.read_exact(&mut a)?;
    reader.read_exact(&mut b)?;
    Ok(Some(PairEvent {
        value: u16::from_le_bytes(value),
        a: u32::from_le_bytes(a),
        b: u32::from_le_bytes(b),
    }))
}

fn write_attach(writer: &mut BufWriter<File>, event: AttachDiskEvent) -> Result<()> {
    writer.write_all(&event.value.to_le_bytes())?;
    writer.write_all(&event.node.to_le_bytes())?;
    write_branch(writer, event.branch)?;
    Ok(())
}

fn read_attach(reader: &mut BufReader<File>) -> Result<Option<AttachDiskEvent>> {
    let mut value = [0u8; 2];
    if !read_exact_or_eof(reader, &mut value)? {
        return Ok(None);
    }
    let mut node = [0u8; 4];
    reader.read_exact(&mut node)?;
    Ok(Some(AttachDiskEvent {
        value: u16::from_le_bytes(value),
        node: u32::from_le_bytes(node),
        branch: read_branch(reader)?,
    }))
}

struct MergeRunReader {
    reader: BufReader<File>,
    next: Option<H2BranchMerge>,
}

impl MergeRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_merge(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_at_value(&mut self, value: u16) -> Result<Option<H2BranchMerge>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_merge(&mut self.reader)?;
                Ok(Some(event))
            }
            _ => Ok(None),
        }
    }
}

struct PairRunReader {
    reader: BufReader<File>,
    next: Option<PairEvent>,
}

impl PairRunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_pair(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_at_value(&mut self, value: u16) -> Result<Option<(u32, u32)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_pair(&mut self.reader)?;
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
        let next = read_attach(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn pop_all_at_value(&mut self, value: u16) -> Result<Vec<(u32, H2Branch)>> {
        let mut events = Vec::new();
        while let Some(event) = self.next {
            if event.value != value {
                break;
            }
            events.push((event.node, event.branch));
            self.next = read_attach(&mut self.reader)?;
        }
        Ok(events)
    }
}

fn write_local_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[H2BranchMerge],
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_tree_local_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);
    for &event in events {
        write_merge(&mut writer, event)?;
    }
    writer.flush()?;
    Ok(path)
}

fn write_attach_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[AttachEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_tree_attach_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);
    for event in events {
        write_attach(
            &mut writer,
            AttachDiskEvent {
                value: event.value,
                node: desc.interface_base + event.interface_node,
                branch: event.branch,
            },
        )?;
    }
    writer.flush()?;
    Ok(path)
}

fn write_interface_run(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[InterfaceMergeEvent],
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_tree_interface_{:05}.bin", desc.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);
    for event in events {
        write_pair(
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

fn write_cross_run(
    directory: &Path,
    pair_id: usize,
    left: &StoredUpperFace,
    right: &SlabDescriptor,
    right_values: &[u16],
    shape: [usize; 2],
    connectivity: Connectivity,
) -> Result<PathBuf> {
    let [width, height] = shape;
    let face_size = width * height;
    let mut buckets: Vec<Vec<(u32, u32)>> = (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();

    for y in 0..height {
        for x in 0..width {
            let left_idx = y * width + x;
            let a = upper_face_global_id(&left.descriptor, face_size, left_idx);
            let left_value = left.values[left_idx];

            match connectivity {
                Connectivity::Six => {
                    let b = lower_face_global_id(right, left_idx);
                    let value = left_value.min(right_values[left_idx]);
                    buckets[value as usize].push((a, b));
                }
                Connectivity::TwentySix => {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                                continue;
                            }
                            let right_idx = ny as usize * width + nx as usize;
                            let b = lower_face_global_id(right, right_idx);
                            let value = left_value.min(right_values[right_idx]);
                            buckets[value as usize].push((a, b));
                        }
                    }
                }
            }
        }
    }

    let path = directory.join(format!("h2_tree_cross_{pair_id:05}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);
    for value in (0..NUM_U16_VALUES).rev() {
        for &(a, b) in &buckets[value] {
            write_pair(
                &mut writer,
                PairEvent {
                    value: value as u16,
                    a,
                    b,
                },
            )?;
        }
    }
    writer.flush()?;
    Ok(path)
}

struct PreparedRuns {
    total_interface_nodes: usize,
    branches_path: PathBuf,
    local_runs: Vec<PathBuf>,
    attach_runs: Vec<PathBuf>,
    interface_runs: Vec<PathBuf>,
    cross_runs: Vec<PathBuf>,
}

fn prepare_runs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    directory: &Path,
) -> Result<PreparedRuns> {
    std::fs::create_dir_all(directory)?;
    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let branches_path = directory.join("h2_tree_interface_branches.bin");
    let mut branch_writer = BufWriter::new(File::create(&branches_path)?);
    let mut local_runs = Vec::new();
    let mut attach_runs = Vec::new();
    let mut interface_runs = Vec::new();
    let mut cross_runs = Vec::new();
    let mut previous_upper: Option<StoredUpperFace> = None;

    for desc in &descriptors {
        println!(
            "Preparing H2 merge-tree slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );
        let block = volume.read_z_slab(desc.z0, desc.z1)?;
        let summary =
            process_slab_h2_merge_tree(desc.slab_id, &block, volume.shape(), connectivity);
        assert_eq!(summary.interface_node_count, desc.interface_count);

        local_runs.push(write_local_run(
            directory,
            desc,
            &summary.local_merge_events,
        )?);
        attach_runs.push(write_attach_run(directory, desc, &summary.attach_events)?);
        interface_runs.push(write_interface_run(
            directory,
            desc,
            &summary.interface_merge_events,
        )?);

        for (&birth, &id) in summary
            .z_min_face
            .values
            .iter()
            .zip(summary.z_min_face.branch_ids.iter())
        {
            write_branch(&mut branch_writer, H2Branch::Finite { id, birth })?;
        }
        if desc.local_depth > 1 {
            for (&birth, &id) in summary
                .z_max_face
                .values
                .iter()
                .zip(summary.z_max_face.branch_ids.iter())
            {
                write_branch(&mut branch_writer, H2Branch::Finite { id, birth })?;
            }
        }

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;
        if let Some(previous) = previous_upper.take() {
            cross_runs.push(write_cross_run(
                directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &lower_values,
                [volume.width, volume.height],
                connectivity,
            )?);
        }
        previous_upper = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: upper_values,
        });
    }

    branch_writer.flush()?;
    Ok(PreparedRuns {
        total_interface_nodes,
        branches_path,
        local_runs,
        attach_runs,
        interface_runs,
        cross_runs,
    })
}

fn read_interface_branches(path: &Path, count: usize) -> Result<Vec<H2Branch>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut branches = Vec::with_capacity(count);
    for _ in 0..count {
        branches.push(read_branch(&mut reader)?);
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("interface branch file contains trailing data");
    }
    Ok(branches)
}

fn reduce_runs(prepared: &PreparedRuns) -> Result<MergeTree> {
    let branches =
        read_interface_branches(&prepared.branches_path, prepared.total_interface_nodes)?;
    let mut global_uf = GlobalH2MergeTreeUnionFind::new(branches);
    let mut recorder = H2TreeRecorder::default();

    let mut local_readers = prepared
        .local_runs
        .iter()
        .map(|path| MergeRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut attach_readers = prepared
        .attach_runs
        .iter()
        .map(|path| AttachRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut interface_readers = prepared
        .interface_runs
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut cross_readers = prepared
        .cross_runs
        .iter()
        .map(|path| PairRunReader::open(path))
        .collect::<Result<Vec<_>>>()?;

    for value in (0..NUM_U16_VALUES).rev() {
        let value_u16 = value as u16;
        let mut outside_events = Vec::new();
        let mut finite_events = Vec::new();

        for reader in &mut attach_readers {
            for event in reader.pop_all_at_value(value_u16)? {
                if event.1 == H2Branch::Outside {
                    outside_events.push(event);
                } else {
                    finite_events.push(event);
                }
            }
        }

        for (node, branch) in outside_events {
            if let Some(merge) = global_uf.attach(node, branch, value_u16) {
                recorder.record_merge(merge)?;
            }
        }
        for reader in &mut local_readers {
            while let Some(merge) = reader.pop_at_value(value_u16)? {
                recorder.record_merge(merge)?;
            }
        }
        for (node, branch) in finite_events {
            if let Some(merge) = global_uf.attach(node, branch, value_u16) {
                recorder.record_merge(merge)?;
            }
        }
        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(value_u16)? {
                if let Some(merge) = global_uf.union(a, b, value_u16) {
                    recorder.record_merge(merge)?;
                }
            }
        }
        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(value_u16)? {
                if let Some(merge) = global_uf.union(a, b, value_u16) {
                    recorder.record_merge(merge)?;
                }
            }
        }

        recorder.finish_threshold()?;
    }

    let remaining = global_uf.remaining_finite_roots();
    if !remaining.is_empty() {
        bail!(
            "H2 merge-tree reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    recorder.add_outside_root();
    recorder.into_tree()
}

pub fn compute_h2_merge_tree_stream_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<MergeTree> {
    let start = Instant::now();
    let directory = TempRunDirectory::create("h2_merge_tree_runs")?;
    println!("H2 merge-tree temporary directory: {:?}", directory.path());

    let prepared = prepare_runs(
        volume,
        slab_depth,
        background_connectivity,
        directory.path(),
    )?;
    let result = reduce_runs(&prepared);
    let tree = result?;
    directory.close()?;

    println!(
        "Streaming H2 merge-tree computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(tree)
}
