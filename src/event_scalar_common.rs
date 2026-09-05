use anyhow::{Result, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::scalar::ScalarKey;
pub(crate) use crate::scalar_order::sorted_scalar_indices;
pub(crate) use crate::slab_interface::local_boundary_node_id;

#[derive(Debug, Clone)]
pub(crate) struct SlabDescriptor {
    pub(crate) slab_id: usize,
    pub(crate) z0: usize,
    pub(crate) z1: usize,
    pub(crate) local_depth: usize,
    pub(crate) interface_base: u32,
    pub(crate) interface_count: u32,
}

pub(crate) fn make_slab_descriptors(
    width: usize,
    height: usize,
    depth: usize,
    slab_depth: usize,
) -> Result<Vec<SlabDescriptor>> {
    if slab_depth == 0 {
        bail!("slab_depth must be positive");
    }

    let face_size = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("face size overflow"))?;
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
            face_size
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("slab interface count overflow"))?
        };

        let interface_base = u32::try_from(next_interface_id)
            .map_err(|_| anyhow::anyhow!("too many global interface nodes"))?;
        let interface_count = u32::try_from(interface_count)
            .map_err(|_| anyhow::anyhow!("too many interface nodes in one slab"))?;

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
            .ok_or_else(|| anyhow::anyhow!("interface node count overflow"))?;
        if next_interface_id > u32::MAX as u64 {
            bail!("too many global interface nodes for u32 IDs");
        }

        slab_id += 1;
        z0 = z1;
    }

    Ok(descriptors)
}

pub(crate) fn report_scalar_event_memory_estimate(
    width: usize,
    height: usize,
    depth: usize,
    slab_depth: usize,
    cross_connectivity: Connectivity,
    includes_outside_flags: bool,
) -> Result<()> {
    let descriptors = make_slab_descriptors(width, height, depth, slab_depth)?;
    let face_size = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("face size overflow"))?;
    let largest_local_depth = slab_depth.min(depth);
    let local_voxels = face_size
        .checked_mul(largest_local_depth)
        .ok_or_else(|| anyhow::anyhow!("scalar slab voxel count overflow"))?;
    if local_voxels > u32::MAX as usize {
        bail!(
            "a {width} x {height} x {largest_local_depth} scalar slab has {local_voxels} voxels, \
             exceeding the u32 event-index limit; lower slab_depth"
        );
    }

    let interface_nodes = descriptors
        .last()
        .map(|descriptor| descriptor.interface_base as u64 + u64::from(descriptor.interface_count))
        .unwrap_or(0);
    let candidate_edges = match cross_connectivity {
        Connectivity::Six => face_size as u128,
        Connectivity::TwentySix => {
            let horizontal = if width == 1 { 1 } else { 3 * width as u128 - 2 };
            let vertical = if height == 1 {
                1
            } else {
                3 * height as u128 - 2
            };
            horizontal * vertical
        }
    };

    // Local processing holds an eight-byte ScalarKey volume, a four-byte
    // sorted index, and compact union-find/active metadata. Boundary and event
    // buffers are included as a conservative fixed allowance.
    let local_bytes_per_voxel = if includes_outside_flags { 22.125 } else { 22.0 };
    let local_bytes = local_voxels as f64 * local_bytes_per_voxel
        + face_size as f64 * 8.0
        + 64.0 * 1024.0 * 1024.0;

    // The face sparsifier now orders only the 2A face vertices. It never
    // materializes the nearly 9A candidate edges of a 26-connected face.
    // This estimate includes two input faces, the vertex order, and (for
    // 26-connectivity) the compact union-find.
    let cross_bytes = if descriptors.len() > 1 {
        let union_find_bytes = if matches!(cross_connectivity, Connectivity::TwentySix) {
            face_size as f64 * 8.0
        } else {
            0.0
        };
        face_size as f64 * 24.0 + union_find_bytes
    } else {
        0.0
    };
    let global_bytes_per_node = if includes_outside_flags { 5.125 } else { 5.0 };
    let global_bytes = interface_nodes as f64 * global_bytes_per_node;
    let gib = 1024.0_f64.powi(3);

    println!("=== Scalar event-stream memory estimate ===");
    println!(
        "largest local slab: {local_voxels} voxels, approximately {:.2} GiB",
        local_bytes / gib
    );
    if descriptors.len() > 1 {
        println!(
            "cross-face sparsification: checks {candidate_edges} edges without storing them; approximately {:.2} GiB",
            cross_bytes / gib
        );
    }
    println!(
        "global interface union-find: {interface_nodes} nodes, approximately {:.2} GiB",
        global_bytes / gib
    );
    println!(
        "estimated peak before allocator/TIFF overhead: approximately {:.2} GiB",
        local_bytes.max(cross_bytes).max(global_bytes) / gib
    );
    println!();
    Ok(())
}

pub(crate) fn lower_face_global_id(desc: &SlabDescriptor, face_index: usize) -> u32 {
    desc.interface_base + u32::try_from(face_index).expect("face index overflow")
}

pub(crate) fn upper_face_global_id(
    desc: &SlabDescriptor,
    face_size: usize,
    face_index: usize,
) -> u32 {
    let local_id = if desc.local_depth == 1 {
        face_index
    } else {
        face_size + face_index
    };
    desc.interface_base + u32::try_from(local_id).expect("face index overflow")
}

pub(crate) fn write_scalar_key(writer: &mut impl Write, value: ScalarKey) -> Result<()> {
    writer.write_all(&value.raw().to_le_bytes())?;
    Ok(())
}

pub(crate) fn read_scalar_key(reader: &mut impl Read) -> Result<Option<ScalarKey>> {
    let mut bytes = [0u8; 8];
    if read_exact_or_eof(reader, &mut bytes)? {
        Ok(Some(ScalarKey::from_raw(u64::from_le_bytes(bytes))))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeltaEvent {
    pub(crate) value: ScalarKey,
    pub(crate) delta: i64,
}

pub(crate) fn write_delta_event(writer: &mut impl Write, event: DeltaEvent) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.delta.to_le_bytes())?;
    Ok(())
}

fn read_delta_event(reader: &mut impl Read) -> Result<Option<DeltaEvent>> {
    let Some(value) = read_scalar_key(reader)? else {
        return Ok(None);
    };
    let mut delta_bytes = [0u8; 8];
    reader.read_exact(&mut delta_bytes)?;
    Ok(Some(DeltaEvent {
        value,
        delta: i64::from_le_bytes(delta_bytes),
    }))
}

pub(crate) struct DeltaRunReader {
    reader: BufReader<File>,
    next: Option<DeltaEvent>,
}

impl DeltaRunReader {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_delta_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    pub(crate) fn peek_value(&self) -> Option<ScalarKey> {
        self.next.map(|event| event.value)
    }

    pub(crate) fn pop(&mut self) -> Result<Option<DeltaEvent>> {
        let current = self.next.take();
        if current.is_some() {
            self.next = read_delta_event(&mut self.reader)?;
        }
        Ok(current)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PairEvent {
    pub(crate) value: ScalarKey,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

pub(crate) fn write_pair_event(writer: &mut impl Write, event: PairEvent) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair_event(reader: &mut impl Read) -> Result<Option<PairEvent>> {
    let Some(value) = read_scalar_key(reader)? else {
        return Ok(None);
    };
    let mut a_bytes = [0u8; 4];
    let mut b_bytes = [0u8; 4];
    reader.read_exact(&mut a_bytes)?;
    reader.read_exact(&mut b_bytes)?;
    Ok(Some(PairEvent {
        value,
        a: u32::from_le_bytes(a_bytes),
        b: u32::from_le_bytes(b_bytes),
    }))
}

pub(crate) struct PairRunReader {
    reader: BufReader<File>,
    next: Option<PairEvent>,
}

impl PairRunReader {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_pair_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    pub(crate) fn peek_value(&self) -> Option<ScalarKey> {
        self.next.map(|event| event.value)
    }

    pub(crate) fn pop(&mut self) -> Result<Option<PairEvent>> {
        let current = self.next.take();
        if current.is_some() {
            self.next = read_pair_event(&mut self.reader)?;
        }
        Ok(current)
    }
}

pub(crate) fn create_buffered_file(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(File::create(path)?))
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarEventCurveStats {
    pub rows: u64,
}
