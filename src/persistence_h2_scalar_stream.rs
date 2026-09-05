use anyhow::{Context, Result, bail};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::allocator_trim::trim_process_heap;
use crate::atomic_output::AtomicOutput;
use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::interface_sparsify_scalar::sparsify_scalar_superlevel_interface_with_order;
use crate::io_scalar::{
    ScalarPixelType, ScalarReadProfile, ScalarTiffStackReader, print_scalar_volume_info,
};
use crate::memory_audit::{emit as emit_memory_snapshot, vec_capacity_bytes};
use crate::persistence_h2_scalar::{
    AttachEvent, CompactGlobalBackgroundPersistenceUnionFind, FinitePair, H2LocalStreamEvent,
    H2SlabPreparationProfile, InterfaceMergeEvent, OutsideEvent, SlabH2Summary,
    StreamingGlobalBackgroundPersistenceUnionFind,
    process_slab_h2_persistence_f32_native_direct_profiled,
    process_slab_h2_persistence_f32_native_profiled_with_memory_audit,
    process_slab_h2_persistence_profiled_with_memory_audit,
};
use crate::scalar::{F32Key, LocalScalarKey, ScalarKey};
use crate::scalar_order::RadixScalarKey;
use crate::scalar_stream_tuning::{
    EventOrderStrategy, F32KeyMode, GlobalH2BirthStateStrategy, H2HierCrossStorageStrategy,
    H2HierOutsideStructuralPruningStrategy, InterfaceOrderStrategy, MergeStrategy,
    PhaseTrimStrategy, ScalarStreamTuning,
};
use crate::temp_runs::TempRunDirectory;
use crate::tiff_paths::find_tiff_stack_directories;

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
    assert!(slab_depth > 0);
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
        assert!(next_interface_id <= u32::MAX as u64);

        slab_id += 1;
        z0 = z1;
    }

    descriptors
}

fn lower_face_global_id(descriptor: &SlabDescriptor, face_index: usize) -> u32 {
    descriptor.interface_base + u32::try_from(face_index).expect("face index overflow")
}

fn upper_face_global_id(descriptor: &SlabDescriptor, face_size: usize, face_index: usize) -> u32 {
    let local_id = if descriptor.local_depth == 1 {
        face_index
    } else {
        face_size + face_index
    };
    descriptor.interface_base + u32::try_from(local_id).expect("face index overflow")
}

struct StoredUpperFace<K> {
    descriptor: SlabDescriptor,
    values: Vec<K>,
}

#[derive(Debug, Clone, Copy)]
struct PairEvent<K> {
    value: K,
    a: u32,
    b: u32,
}

#[derive(Debug, Clone, Copy)]
struct AttachDiskEvent<K> {
    value: K,
    node: u32,
    branch_birth: K,
}

#[derive(Debug, Clone, Copy)]
struct OutsideDiskEvent<K> {
    value: K,
    node: u32,
}

trait DiskScalarKey: LocalScalarKey + RadixScalarKey {
    const DISK_BYTES: usize;

    fn write_disk(self, writer: &mut BufWriter<File>) -> Result<()>;
    fn read_disk(reader: &mut BufReader<File>) -> Result<Option<Self>>;
}

impl DiskScalarKey for ScalarKey {
    const DISK_BYTES: usize = 8;

    #[inline]
    fn write_disk(self, writer: &mut BufWriter<File>) -> Result<()> {
        writer.write_all(&self.raw().to_le_bytes())?;
        Ok(())
    }

    #[inline]
    fn read_disk(reader: &mut BufReader<File>) -> Result<Option<Self>> {
        let mut bytes = [0u8; 8];
        if read_exact_or_eof(reader, &mut bytes)? {
            Ok(Some(ScalarKey::from_raw(u64::from_le_bytes(bytes))))
        } else {
            Ok(None)
        }
    }
}

impl DiskScalarKey for F32Key {
    const DISK_BYTES: usize = 4;

    #[inline]
    fn write_disk(self, writer: &mut BufWriter<File>) -> Result<()> {
        writer.write_all(&self.raw().to_le_bytes())?;
        Ok(())
    }

    #[inline]
    fn read_disk(reader: &mut BufReader<File>) -> Result<Option<Self>> {
        let mut bytes = [0u8; 4];
        if read_exact_or_eof(reader, &mut bytes)? {
            Ok(Some(F32Key::from_raw(u32::from_le_bytes(bytes))))
        } else {
            Ok(None)
        }
    }
}

fn write_key<K: DiskScalarKey>(writer: &mut BufWriter<File>, key: K) -> Result<()> {
    key.write_disk(writer)
}

fn read_optional_key<K: DiskScalarKey>(reader: &mut BufReader<File>) -> Result<Option<K>> {
    K::read_disk(reader)
}

fn read_required_key<K: DiskScalarKey>(reader: &mut BufReader<File>) -> Result<K> {
    read_optional_key::<K>(reader)?.ok_or_else(|| anyhow::anyhow!("truncated scalar event file"))
}

fn write_pair_event<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    event: PairEvent<K>,
) -> Result<()> {
    write_key(writer, event.value)?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair_event<K: DiskScalarKey>(reader: &mut BufReader<File>) -> Result<Option<PairEvent<K>>> {
    let Some(value) = read_optional_key::<K>(reader)? else {
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

fn write_attach_event<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    event: AttachDiskEvent<K>,
) -> Result<()> {
    write_key(writer, event.value)?;
    writer.write_all(&event.node.to_le_bytes())?;
    write_key(writer, event.branch_birth)?;
    Ok(())
}

fn read_attach_event<K: DiskScalarKey>(
    reader: &mut BufReader<File>,
) -> Result<Option<AttachDiskEvent<K>>> {
    let Some(value) = read_optional_key::<K>(reader)? else {
        return Ok(None);
    };
    let mut node_bytes = [0u8; 4];
    reader.read_exact(&mut node_bytes)?;
    let branch_birth = read_required_key::<K>(reader)?;
    Ok(Some(AttachDiskEvent {
        value,
        node: u32::from_le_bytes(node_bytes),
        branch_birth,
    }))
}

fn write_outside_event<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    event: OutsideDiskEvent<K>,
) -> Result<()> {
    write_key(writer, event.value)?;
    writer.write_all(&event.node.to_le_bytes())?;
    Ok(())
}

fn read_outside_event<K: DiskScalarKey>(
    reader: &mut BufReader<File>,
) -> Result<Option<OutsideDiskEvent<K>>> {
    let Some(value) = read_optional_key::<K>(reader)? else {
        return Ok(None);
    };
    let mut node_bytes = [0u8; 4];
    reader.read_exact(&mut node_bytes)?;
    Ok(Some(OutsideDiskEvent {
        value,
        node: u32::from_le_bytes(node_bytes),
    }))
}

fn write_finite_pair<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    pair: FinitePair<K>,
) -> Result<()> {
    write_key(writer, pair.birth)?;
    write_key(writer, pair.death)?;
    Ok(())
}

fn read_finite_pair<K: DiskScalarKey>(
    reader: &mut BufReader<File>,
) -> Result<Option<FinitePair<K>>> {
    let Some(birth) = read_optional_key::<K>(reader)? else {
        return Ok(None);
    };
    let death = read_required_key::<K>(reader)?;
    Ok(Some(FinitePair { birth, death }))
}

struct PairRunReader<K> {
    reader: BufReader<File>,
    next: Option<PairEvent<K>>,
}

impl<K: DiskScalarKey> PairRunReader<K> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_pair_event::<K>(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn next_value(&self) -> Option<K> {
        self.next.map(|event| event.value)
    }

    fn pop_at_value(&mut self, value: K) -> Result<Option<(u32, u32)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_pair_event::<K>(&mut self.reader)?;
                Ok(Some((event.a, event.b)))
            }
            _ => Ok(None),
        }
    }
}

struct AttachRunReader<K> {
    reader: BufReader<File>,
    next: Option<AttachDiskEvent<K>>,
}

impl<K: DiskScalarKey> AttachRunReader<K> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_attach_event::<K>(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn next_value(&self) -> Option<K> {
        self.next.map(|event| event.value)
    }

    fn pop_at_value(&mut self, value: K) -> Result<Option<(u32, K)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_attach_event::<K>(&mut self.reader)?;
                Ok(Some((event.node, event.branch_birth)))
            }
            _ => Ok(None),
        }
    }
}

struct OutsideRunReader<K> {
    reader: BufReader<File>,
    next: Option<OutsideDiskEvent<K>>,
}

impl<K: DiskScalarKey> OutsideRunReader<K> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_outside_event::<K>(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn next_value(&self) -> Option<K> {
        self.next.map(|event| event.value)
    }

    fn pop_at_value(&mut self, value: K) -> Result<Option<u32>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_outside_event::<K>(&mut self.reader)?;
                Ok(Some(event.node))
            }
            _ => Ok(None),
        }
    }
}

fn write_interface_merge_run<K: DiskScalarKey>(
    directory: &Path,
    descriptor: &SlabDescriptor,
    events: &mut [InterfaceMergeEvent<K>],
    event_order: EventOrderStrategy,
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_scalar_interface_{:05}.bin", descriptor.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

    match event_order {
        EventOrderStrategy::Verify => {
            let mut previous = None;
            for &event in events.iter() {
                if previous.is_some_and(|value| event.value > value) {
                    bail!(
                        "scalar H2 interface events are not monotone within slab {}",
                        descriptor.slab_id
                    );
                }
                previous = Some(event.value);
                write_pair_event(
                    &mut writer,
                    PairEvent {
                        value: event.value,
                        a: descriptor.interface_base + event.a,
                        b: descriptor.interface_base + event.b,
                    },
                )?;
            }
        }
        EventOrderStrategy::Resort => {
            events.sort_unstable_by_key(|event| Reverse(event.value));
            for &event in events.iter() {
                write_pair_event(
                    &mut writer,
                    PairEvent {
                        value: event.value,
                        a: descriptor.interface_base + event.a,
                        b: descriptor.interface_base + event.b,
                    },
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(path)
}

fn write_attach_run<K: DiskScalarKey>(
    directory: &Path,
    descriptor: &SlabDescriptor,
    events: &mut [AttachEvent<K>],
    event_order: EventOrderStrategy,
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_scalar_attach_{:05}.bin", descriptor.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

    match event_order {
        EventOrderStrategy::Verify => {
            let mut previous = None;
            for &event in events.iter() {
                if previous.is_some_and(|value| event.value > value) {
                    bail!(
                        "scalar H2 attach events are not monotone within slab {}",
                        descriptor.slab_id
                    );
                }
                previous = Some(event.value);
                write_attach_event(
                    &mut writer,
                    AttachDiskEvent {
                        value: event.value,
                        node: descriptor.interface_base + event.interface_node,
                        branch_birth: event.branch_birth,
                    },
                )?;
            }
        }
        EventOrderStrategy::Resort => {
            events.sort_unstable_by_key(|event| Reverse(event.value));
            for &event in events.iter() {
                write_attach_event(
                    &mut writer,
                    AttachDiskEvent {
                        value: event.value,
                        node: descriptor.interface_base + event.interface_node,
                        branch_birth: event.branch_birth,
                    },
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(path)
}

fn write_outside_run<K: DiskScalarKey>(
    directory: &Path,
    descriptor: &SlabDescriptor,
    events: &mut [OutsideEvent<K>],
    event_order: EventOrderStrategy,
) -> Result<PathBuf> {
    let path = directory.join(format!("h2_scalar_outside_{:05}.bin", descriptor.slab_id));
    let mut writer = BufWriter::new(File::create(&path)?);

    match event_order {
        EventOrderStrategy::Verify => {
            let mut previous = None;
            for &event in events.iter() {
                if previous.is_some_and(|value| event.value > value) {
                    bail!(
                        "scalar H2 outside events are not monotone within slab {}",
                        descriptor.slab_id
                    );
                }
                previous = Some(event.value);
                write_outside_event(
                    &mut writer,
                    OutsideDiskEvent {
                        value: event.value,
                        node: descriptor.interface_base + event.interface_node,
                    },
                )?;
            }
        }
        EventOrderStrategy::Resort => {
            events.sort_unstable_by_key(|event| Reverse(event.value));
            for &event in events.iter() {
                write_outside_event(
                    &mut writer,
                    OutsideDiskEvent {
                        value: event.value,
                        node: descriptor.interface_base + event.interface_node,
                    },
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(path)
}

#[derive(Debug, Clone, Copy)]
struct CrossRunConfig {
    width: usize,
    height: usize,
    connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
}

fn write_cross_run<K: DiskScalarKey>(
    directory: &Path,
    pair_id: usize,
    left: &StoredUpperFace<K>,
    right_descriptor: &SlabDescriptor,
    right_lower_values: &[K],
    config: CrossRunConfig,
) -> Result<PathBuf> {
    let width = config.width;
    let height = config.height;
    let connectivity = config.connectivity;

    let face_size = width * height;
    let path = directory.join(format!("h2_scalar_cross_{pair_id:05}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);

    let stats = sparsify_scalar_superlevel_interface_with_order(
        &left.values,
        right_lower_values,
        width,
        height,
        connectivity,
        config.interface_order,
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
                    b: lower_face_global_id(right_descriptor, edge.right_face_index as usize),
                },
            )
        },
    )?;

    writer.flush()?;
    println!(
        "Scalar interface {pair_id}: retained {} of {} candidate edges",
        stats.retained_edges, stats.candidate_edges
    );
    Ok(path)
}

#[derive(Debug, Default, Clone, Copy)]
struct DetailedPreparationProfile {
    decode_seconds: f64,
    key_conversion_seconds: f64,
    slab_copy_seconds: f64,
    scalar_order_seconds: f64,
    local_sweep_seconds: f64,
    local_run_write_seconds: f64,
    cross_interface_seconds: f64,
    total_voxels: u64,
    interior_fast_voxels: u64,
    pruning_mask_calls: u64,
    active_state_checks: u64,
    active_neighbor_hits: u64,
    representative_visits: u64,
    pruning_cache_hits: u64,
    pruning_cache_misses: u64,
    component_mask_computations: u64,
    union_attempts: u64,
    successful_unions: u64,
    same_root_unions: u64,
    find_calls: u64,
    find_parent_steps: u64,
    active_rechecks: u64,
    active_recheck_failures: u64,
    root_carry_attempts: u64,
    avoided_find_calls: u64,
    direct_parent_checks: u64,
    direct_parent_hits: u64,
    avoided_neighbor_find_calls: u64,
    interface_rep_queries: u64,
    interface_rep_writes: u64,
    interface_forced_root_unions: u64,
    interface_interface_unions: u64,
    interface_state_bytes: u64,
    max_rank_observed: u64,
    uf_parent_state_bytes: u64,
    uf_rank_state_bytes: u64,
}

struct PreparedH2ScalarRuns {
    total_interface_nodes: usize,
    births_path: PathBuf,
    local_pairs_path: PathBuf,
    attach_run_paths: Vec<PathBuf>,
    outside_run_paths: Vec<PathBuf>,
    interface_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
    disk_key_bytes: usize,
    interface_birth_bytes: u64,
    local_pair_bytes: u64,
    attach_bytes: u64,
    outside_bytes: u64,
    interface_bytes: u64,
    cross_bytes: u64,
    total_run_bytes: u64,
    preparation_profile: DetailedPreparationProfile,
}

fn path_bytes(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}

fn paths_bytes(paths: &[PathBuf]) -> Result<u64> {
    paths
        .iter()
        .try_fold(0u64, |sum, path| Ok(sum.saturating_add(path_bytes(path)?)))
}

fn prepare_h2_scalar_runs_generic<K, F>(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
    mut process_slab: F,
) -> Result<PreparedH2ScalarRuns>
where
    K: DiskScalarKey,
    F: FnMut(
        &SlabDescriptor,
    ) -> Result<(
        SlabH2Summary<K>,
        H2SlabPreparationProfile,
        ScalarReadProfile,
    )>,
{
    fs::create_dir_all(temp_directory)?;
    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let total_interface_nodes = descriptors
        .last()
        .map(|descriptor| descriptor.interface_base as usize + descriptor.interface_count as usize)
        .unwrap_or(0);

    let births_path = temp_directory.join("h2_scalar_interface_births.bin");
    let local_pairs_path = temp_directory.join("h2_scalar_local_pairs.bin");
    let mut birth_writer = BufWriter::new(File::create(&births_path)?);
    let mut local_pair_writer = BufWriter::new(File::create(&local_pairs_path)?);

    let mut attach_run_paths = Vec::new();
    let mut outside_run_paths = Vec::new();
    let mut interface_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace<K>> = None;
    let mut preparation_profile = DetailedPreparationProfile::default();

    let cross_run_config = CrossRunConfig {
        width: volume.width,
        height: volume.height,
        connectivity: background_connectivity,
        interface_order: tuning.interface_order,
    };

    for descriptor in &descriptors {
        println!(
            "Preparing scalar H2 slab {}: z={}..{}",
            descriptor.slab_id, descriptor.z0, descriptor.z1
        );

        if tuning.h2_memory_audit {
            emit_memory_snapshot(
                "scalar_h2_stream",
                "before_slab_read",
                Some(descriptor.slab_id),
            );
        }

        let (mut summary, slab_profile, read_profile) = process_slab(descriptor)?;
        preparation_profile.decode_seconds += read_profile.decode_seconds;
        preparation_profile.key_conversion_seconds += read_profile.key_conversion_seconds;
        preparation_profile.slab_copy_seconds += read_profile.slab_copy_seconds;

        if tuning.h2_memory_audit {
            emit_memory_snapshot(
                "scalar_h2_stream",
                "after_block_drop",
                Some(descriptor.slab_id),
            );
            println!(
                "PROFILE_MEMVEC scalar_h2_stream stage=summary_live slab={} key_bytes={} finite_pairs_len={} finite_pairs_capacity={} finite_pairs_capacity_bytes={} attach_len={} attach_capacity={} attach_capacity_bytes={} outside_len={} outside_capacity={} outside_capacity_bytes={} interface_merge_len={} interface_merge_capacity={} interface_merge_capacity_bytes={} z_min_node_capacity_bytes={} z_min_value_capacity_bytes={} z_max_node_capacity_bytes={} z_max_value_capacity_bytes={}",
                descriptor.slab_id,
                K::DISK_BYTES,
                summary.finalized_pairs.len(),
                summary.finalized_pairs.capacity(),
                vec_capacity_bytes(&summary.finalized_pairs),
                summary.attach_events.len(),
                summary.attach_events.capacity(),
                vec_capacity_bytes(&summary.attach_events),
                summary.outside_events.len(),
                summary.outside_events.capacity(),
                vec_capacity_bytes(&summary.outside_events),
                summary.interface_merge_events.len(),
                summary.interface_merge_events.capacity(),
                vec_capacity_bytes(&summary.interface_merge_events),
                vec_capacity_bytes(&summary.z_min_face.node_ids),
                vec_capacity_bytes(&summary.z_min_face.values),
                vec_capacity_bytes(&summary.z_max_face.node_ids),
                vec_capacity_bytes(&summary.z_max_face.values),
            );
        }

        preparation_profile.scalar_order_seconds += slab_profile.scalar_order_seconds;
        preparation_profile.local_sweep_seconds += slab_profile.local_sweep_seconds;
        preparation_profile.total_voxels += slab_profile.total_voxels;
        preparation_profile.interior_fast_voxels += slab_profile.interior_fast_voxels;
        preparation_profile.pruning_mask_calls += slab_profile.pruning_mask_calls;
        preparation_profile.active_state_checks += slab_profile.active_state_checks;
        preparation_profile.active_neighbor_hits += slab_profile.active_neighbor_hits;
        preparation_profile.representative_visits += slab_profile.representative_visits;
        preparation_profile.pruning_cache_hits += slab_profile.pruning_cache_hits;
        preparation_profile.pruning_cache_misses += slab_profile.pruning_cache_misses;
        preparation_profile.component_mask_computations += slab_profile.component_mask_computations;
        preparation_profile.union_attempts += slab_profile.union_attempts;
        preparation_profile.successful_unions += slab_profile.successful_unions;
        preparation_profile.same_root_unions += slab_profile.same_root_unions;
        preparation_profile.find_calls += slab_profile.find_calls;
        preparation_profile.find_parent_steps += slab_profile.find_parent_steps;
        preparation_profile.active_rechecks += slab_profile.active_rechecks;
        preparation_profile.active_recheck_failures += slab_profile.active_recheck_failures;
        preparation_profile.root_carry_attempts += slab_profile.root_carry_attempts;
        preparation_profile.avoided_find_calls += slab_profile.avoided_find_calls;
        preparation_profile.direct_parent_checks += slab_profile.direct_parent_checks;
        preparation_profile.direct_parent_hits += slab_profile.direct_parent_hits;
        preparation_profile.avoided_neighbor_find_calls += slab_profile.avoided_neighbor_find_calls;
        preparation_profile.interface_rep_queries += slab_profile.interface_rep_queries;
        preparation_profile.interface_rep_writes += slab_profile.interface_rep_writes;
        preparation_profile.interface_forced_root_unions +=
            slab_profile.interface_forced_root_unions;
        preparation_profile.interface_interface_unions += slab_profile.interface_interface_unions;
        preparation_profile.interface_state_bytes = preparation_profile
            .interface_state_bytes
            .max(slab_profile.interface_state_bytes);
        preparation_profile.max_rank_observed = preparation_profile
            .max_rank_observed
            .max(slab_profile.max_rank_observed);
        preparation_profile.uf_parent_state_bytes = preparation_profile
            .uf_parent_state_bytes
            .max(slab_profile.uf_parent_state_bytes);
        preparation_profile.uf_rank_state_bytes = preparation_profile
            .uf_rank_state_bytes
            .max(slab_profile.uf_rank_state_bytes);

        assert_eq!(summary.interface_node_count, descriptor.interface_count);

        let local_write_start = Instant::now();
        for pair in summary.finalized_pairs {
            write_finite_pair::<K>(&mut local_pair_writer, pair)?;
        }

        for &value in &summary.z_min_face.values {
            write_key::<K>(&mut birth_writer, value)?;
        }
        if descriptor.local_depth > 1 {
            for &value in &summary.z_max_face.values {
                write_key::<K>(&mut birth_writer, value)?;
            }
        }

        attach_run_paths.push(write_attach_run::<K>(
            temp_directory,
            descriptor,
            &mut summary.attach_events,
            tuning.event_order,
        )?);
        outside_run_paths.push(write_outside_run::<K>(
            temp_directory,
            descriptor,
            &mut summary.outside_events,
            tuning.event_order,
        )?);
        interface_run_paths.push(write_interface_merge_run::<K>(
            temp_directory,
            descriptor,
            &mut summary.interface_merge_events,
            tuning.event_order,
        )?);
        preparation_profile.local_run_write_seconds += local_write_start.elapsed().as_secs_f64();
        if tuning.h2_memory_audit {
            emit_memory_snapshot(
                "scalar_h2_stream",
                "after_local_run_write",
                Some(descriptor.slab_id),
            );
        }

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;

        if let Some(previous) = previous_upper_face.take() {
            let cross_start = Instant::now();
            cross_run_paths.push(write_cross_run::<K>(
                temp_directory,
                descriptor.slab_id - 1,
                &previous,
                descriptor,
                &lower_values,
                cross_run_config,
            )?);
            preparation_profile.cross_interface_seconds += cross_start.elapsed().as_secs_f64();
        }

        previous_upper_face = Some(StoredUpperFace {
            descriptor: descriptor.clone(),
            values: upper_values,
        });
        if tuning.h2_memory_audit {
            emit_memory_snapshot(
                "scalar_h2_stream",
                "after_slab_complete",
                Some(descriptor.slab_id),
            );
            if let Some(face) = previous_upper_face.as_ref() {
                println!(
                    "PROFILE_MEMVEC scalar_h2_stream stage=after_slab_complete slab={} retained_upper_face_len={} retained_upper_face_capacity={} retained_upper_face_capacity_bytes={} attach_run_count={} outside_run_count={} interface_run_count={} cross_run_count={}",
                    descriptor.slab_id,
                    face.values.len(),
                    face.values.capacity(),
                    vec_capacity_bytes(&face.values),
                    attach_run_paths.len(),
                    outside_run_paths.len(),
                    interface_run_paths.len(),
                    cross_run_paths.len(),
                );
            }
        }
    }

    let final_flush_start = Instant::now();
    birth_writer.flush()?;
    local_pair_writer.flush()?;
    preparation_profile.local_run_write_seconds += final_flush_start.elapsed().as_secs_f64();
    if tuning.h2_memory_audit {
        emit_memory_snapshot("scalar_h2_stream", "after_prepare_flush", None);
    }

    let interface_birth_bytes = path_bytes(&births_path)?;
    let local_pair_bytes = path_bytes(&local_pairs_path)?;
    let attach_bytes = paths_bytes(&attach_run_paths)?;
    let outside_bytes = paths_bytes(&outside_run_paths)?;
    let interface_bytes = paths_bytes(&interface_run_paths)?;
    let cross_bytes = paths_bytes(&cross_run_paths)?;
    let total_run_bytes = interface_birth_bytes
        .saturating_add(local_pair_bytes)
        .saturating_add(attach_bytes)
        .saturating_add(outside_bytes)
        .saturating_add(interface_bytes)
        .saturating_add(cross_bytes);

    Ok(PreparedH2ScalarRuns {
        total_interface_nodes,
        births_path,
        local_pairs_path,
        attach_run_paths,
        outside_run_paths,
        interface_run_paths,
        cross_run_paths,
        disk_key_bytes: K::DISK_BYTES,
        interface_birth_bytes,
        local_pair_bytes,
        attach_bytes,
        outside_bytes,
        interface_bytes,
        cross_bytes,
        total_run_bytes,
        preparation_profile,
    })
}

fn prepare_h2_scalar_runs_f32_native(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
) -> Result<PreparedH2ScalarRuns> {
    let [global_width, global_height, global_depth] = volume.shape();
    prepare_h2_scalar_runs_generic::<F32Key, _>(
        volume,
        slab_depth,
        background_connectivity,
        temp_directory,
        tuning,
        |descriptor| {
            let (block, read_profile) =
                volume.read_z_slab_f32_native_profiled(descriptor.z0, descriptor.z1)?;
            if tuning.h2_memory_audit {
                emit_memory_snapshot(
                    "scalar_h2_stream",
                    "after_slab_read",
                    Some(descriptor.slab_id),
                );
            }
            let (summary, slab_profile) =
                process_slab_h2_persistence_f32_native_profiled_with_memory_audit(
                    descriptor.slab_id,
                    &block,
                    global_width,
                    global_height,
                    global_depth,
                    background_connectivity,
                    tuning.neighbor_kernel,
                    tuning.representative_active_check,
                    tuning.union_kernel,
                    tuning.neighbor_root_check,
                    tuning.active_state,
                    tuning.interface_state,
                    tuning.uf_layout,
                    tuning.local_h2_birth_state,
                    tuning.sweep_diagnostics,
                    tuning.h2_memory_audit,
                );
            if tuning.h2_memory_audit {
                emit_memory_snapshot(
                    "scalar_h2_stream",
                    "after_local_process_before_block_drop",
                    Some(descriptor.slab_id),
                );
            }
            Ok((summary, slab_profile, read_profile))
        },
    )
}

fn prepare_h2_scalar_runs_scalar(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
) -> Result<PreparedH2ScalarRuns> {
    let [global_width, global_height, global_depth] = volume.shape();
    prepare_h2_scalar_runs_generic::<ScalarKey, _>(
        volume,
        slab_depth,
        background_connectivity,
        temp_directory,
        tuning,
        |descriptor| {
            let (block, read_profile) =
                volume.read_z_slab_profiled(descriptor.z0, descriptor.z1)?;
            if tuning.h2_memory_audit {
                emit_memory_snapshot(
                    "scalar_h2_stream",
                    "after_slab_read",
                    Some(descriptor.slab_id),
                );
            }
            let (summary, slab_profile) = process_slab_h2_persistence_profiled_with_memory_audit(
                descriptor.slab_id,
                &block,
                global_width,
                global_height,
                global_depth,
                background_connectivity,
                tuning.neighbor_kernel,
                tuning.representative_active_check,
                tuning.union_kernel,
                tuning.neighbor_root_check,
                tuning.active_state,
                tuning.interface_state,
                tuning.uf_layout,
                tuning.local_h2_birth_state,
                tuning.sweep_diagnostics,
                tuning.h2_memory_audit,
            );
            if tuning.h2_memory_audit {
                emit_memory_snapshot(
                    "scalar_h2_stream",
                    "after_local_process_before_block_drop",
                    Some(descriptor.slab_id),
                );
            }
            Ok((summary, slab_profile, read_profile))
        },
    )
}

fn read_interface_births<K: DiskScalarKey>(path: &Path, count: usize) -> Result<Vec<K>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut births = Vec::with_capacity(count);
    for _ in 0..count {
        births.push(read_required_key::<K>(&mut reader)?);
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("scalar interface birth file contains trailing data");
    }
    Ok(births)
}

fn write_pair_csv<K: LocalScalarKey>(writer: &mut impl Write, pair: FinitePair<K>) -> Result<bool> {
    if pair.birth >= pair.death {
        return Ok(false);
    }
    let birth = pair.birth.widen();
    let death = pair.death.widen();
    writeln!(writer, "{birth},{death}")?;
    Ok(true)
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarH2PersistenceStats {
    pub finite_intervals: u64,
}

type MaxHeap<K> = BinaryHeap<(K, Reverse<usize>)>;

fn open_outside_runs<K: DiskScalarKey>(
    paths: &[PathBuf],
) -> Result<(Vec<OutsideRunReader<K>>, MaxHeap<K>)> {
    let readers = paths
        .iter()
        .map(|path| OutsideRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.next_value() {
            heap.push((value, Reverse(index)));
        }
    }
    Ok((readers, heap))
}

fn open_attach_runs<K: DiskScalarKey>(
    paths: &[PathBuf],
) -> Result<(Vec<AttachRunReader<K>>, MaxHeap<K>)> {
    let readers = paths
        .iter()
        .map(|path| AttachRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.next_value() {
            heap.push((value, Reverse(index)));
        }
    }
    Ok((readers, heap))
}

fn open_pair_runs<K: DiskScalarKey>(
    paths: &[PathBuf],
) -> Result<(Vec<PairRunReader<K>>, MaxHeap<K>)> {
    let readers = paths
        .iter()
        .map(|path| PairRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.next_value() {
            heap.push((value, Reverse(index)));
        }
    }
    Ok((readers, heap))
}

fn max_heap_value<K: DiskScalarKey>(heap: &MaxHeap<K>) -> Option<K> {
    heap.peek().map(|(value, _)| *value)
}

fn max_next_value<K: DiskScalarKey>(heaps: [&MaxHeap<K>; 4]) -> Option<K> {
    heaps.into_iter().filter_map(max_heap_value).max()
}

fn drain_outside_value<K: DiskScalarKey>(
    value: K,
    readers: &mut [OutsideRunReader<K>],
    heap: &mut MaxHeap<K>,
    mut apply: impl FnMut(u32) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, Reverse(reader_index)) = heap.pop().expect("heap head disappeared");
        let node = readers[reader_index]
            .pop_at_value(value)?
            .ok_or_else(|| anyhow::anyhow!("scalar H2 outside heap is out of order"))?;
        apply(node)?;
        if let Some(next) = readers[reader_index].next_value() {
            heap.push((next, Reverse(reader_index)));
        }
    }
    Ok(())
}

fn drain_attach_value<K: DiskScalarKey>(
    value: K,
    readers: &mut [AttachRunReader<K>],
    heap: &mut MaxHeap<K>,
    mut apply: impl FnMut(u32, K) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, Reverse(reader_index)) = heap.pop().expect("heap head disappeared");
        let (node, branch_birth) = readers[reader_index]
            .pop_at_value(value)?
            .ok_or_else(|| anyhow::anyhow!("scalar H2 attach heap is out of order"))?;
        apply(node, branch_birth)?;
        if let Some(next) = readers[reader_index].next_value() {
            heap.push((next, Reverse(reader_index)));
        }
    }
    Ok(())
}

fn drain_pair_value<K: DiskScalarKey>(
    value: K,
    readers: &mut [PairRunReader<K>],
    heap: &mut MaxHeap<K>,
    mut apply: impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    while max_heap_value(heap) == Some(value) {
        let (_, Reverse(reader_index)) = heap.pop().expect("heap head disappeared");
        let (a, b) = readers[reader_index]
            .pop_at_value(value)?
            .ok_or_else(|| anyhow::anyhow!("scalar H2 pair heap is out of order"))?;
        apply(a, b)?;
        if let Some(next) = readers[reader_index].next_value() {
            heap.push((next, Reverse(reader_index)));
        }
    }
    Ok(())
}

fn max_next_value_scan<K: DiskScalarKey>(
    outside_readers: &[OutsideRunReader<K>],
    attach_readers: &[AttachRunReader<K>],
    interface_readers: &[PairRunReader<K>],
    cross_readers: &[PairRunReader<K>],
) -> Option<K> {
    outside_readers
        .iter()
        .filter_map(OutsideRunReader::next_value)
        .chain(
            attach_readers
                .iter()
                .filter_map(AttachRunReader::next_value),
        )
        .chain(
            interface_readers
                .iter()
                .filter_map(PairRunReader::next_value),
        )
        .chain(cross_readers.iter().filter_map(PairRunReader::next_value))
        .max()
}

fn reduce_h2_scalar_runs_scan(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    memory_audit: bool,
    global_birth_state: GlobalH2BirthStateStrategy,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "before_births_read", None);
    }
    let births =
        read_interface_births::<ScalarKey>(&prepared.births_path, prepared.total_interface_nodes)?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_births_read", None);
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_births_read births_len={} births_capacity={} births_capacity_bytes={}",
            births.len(),
            births.capacity(),
            vec_capacity_bytes(&births)
        );
    }
    let mut global_union_find = StreamingGlobalBackgroundPersistenceUnionFind::new(
        births,
        global_birth_state,
        global_uf_layout,
    );
    let (global_parent_bytes, global_rank_bytes, global_birth_bytes) =
        global_union_find.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H2_STATE strategy={} uf_layout={} nodes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_birth_state.as_str(),
        global_uf_layout.as_str(),
        prepared.total_interface_nodes.saturating_add(1),
        global_parent_bytes,
        global_rank_bytes,
        global_birth_bytes,
        global_parent_bytes + global_rank_bytes + global_birth_bytes,
    );
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_global_uf_alloc", None);
        let (parent_bytes, rank_bytes, birth_bytes) = global_union_find.capacity_bytes();
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_global_uf_alloc global_parent_capacity_bytes={} global_rank_capacity_bytes={} global_birth_capacity_bytes={}",
            parent_bytes, rank_bytes, birth_bytes
        );
    }

    let mut outside_readers = prepared
        .outside_run_paths
        .iter()
        .map(|path| OutsideRunReader::<ScalarKey>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut attach_readers = prepared
        .attach_run_paths
        .iter()
        .map(|path| AttachRunReader::<ScalarKey>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut interface_readers = prepared
        .interface_run_paths
        .iter()
        .map(|path| PairRunReader::<ScalarKey>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut cross_readers = prepared
        .cross_run_paths
        .iter()
        .map(|path| PairRunReader::<ScalarKey>::open(path))
        .collect::<Result<Vec<_>>>()?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_run_readers_open", None);
        let reader_buffer_bytes: usize = outside_readers
            .iter()
            .map(|r| r.reader.capacity())
            .sum::<usize>()
            + attach_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>()
            + interface_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>()
            + cross_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>();
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_run_readers_open outside_readers={} attach_readers={} interface_readers={} cross_readers={} reader_buffer_capacity_bytes={}",
            outside_readers.len(),
            attach_readers.len(),
            interface_readers.len(),
            cross_readers.len(),
            reader_buffer_bytes
        );
    }

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar H2 output {output_path:?}"))?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<ScalarKey>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_local_pairs_copy", None);
    }

    while let Some(value) = max_next_value_scan(
        &outside_readers,
        &attach_readers,
        &interface_readers,
        &cross_readers,
    ) {
        for reader in &mut outside_readers {
            while let Some(node) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.connect_outside(node, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut attach_readers {
            while let Some((node, branch_birth)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.attach_branch(node, branch_birth, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }
    }

    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_event_reduction", None);
    }
    let remaining = global_union_find.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "scalar H2 streaming reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    output.commit()?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_output_commit", None);
    }
    Ok(ScalarH2PersistenceStats { finite_intervals })
}

fn reduce_h2_scalar_runs_heap(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    memory_audit: bool,
    global_birth_state: GlobalH2BirthStateStrategy,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "before_births_read", None);
    }
    let births =
        read_interface_births::<ScalarKey>(&prepared.births_path, prepared.total_interface_nodes)?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_births_read", None);
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_births_read births_len={} births_capacity={} births_capacity_bytes={}",
            births.len(),
            births.capacity(),
            vec_capacity_bytes(&births)
        );
    }
    let mut global_union_find = StreamingGlobalBackgroundPersistenceUnionFind::new(
        births,
        global_birth_state,
        global_uf_layout,
    );
    let (global_parent_bytes, global_rank_bytes, global_birth_bytes) =
        global_union_find.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H2_STATE strategy={} uf_layout={} nodes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_birth_state.as_str(),
        global_uf_layout.as_str(),
        prepared.total_interface_nodes.saturating_add(1),
        global_parent_bytes,
        global_rank_bytes,
        global_birth_bytes,
        global_parent_bytes + global_rank_bytes + global_birth_bytes,
    );
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_global_uf_alloc", None);
        let (parent_bytes, rank_bytes, birth_bytes) = global_union_find.capacity_bytes();
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_global_uf_alloc global_parent_capacity_bytes={} global_rank_capacity_bytes={} global_birth_capacity_bytes={}",
            parent_bytes, rank_bytes, birth_bytes
        );
    }

    let (mut outside_readers, mut outside_heap) =
        open_outside_runs::<ScalarKey>(&prepared.outside_run_paths)?;
    let (mut attach_readers, mut attach_heap) =
        open_attach_runs::<ScalarKey>(&prepared.attach_run_paths)?;
    let (mut interface_readers, mut interface_heap) =
        open_pair_runs::<ScalarKey>(&prepared.interface_run_paths)?;
    let (mut cross_readers, mut cross_heap) =
        open_pair_runs::<ScalarKey>(&prepared.cross_run_paths)?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_run_readers_open", None);
        let reader_buffer_bytes: usize = outside_readers
            .iter()
            .map(|r| r.reader.capacity())
            .sum::<usize>()
            + attach_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>()
            + interface_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>()
            + cross_readers
                .iter()
                .map(|r| r.reader.capacity())
                .sum::<usize>();
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_run_readers_open outside_readers={} attach_readers={} interface_readers={} cross_readers={} reader_buffer_capacity_bytes={} outside_heap_capacity={} attach_heap_capacity={} interface_heap_capacity={} cross_heap_capacity={}",
            outside_readers.len(),
            attach_readers.len(),
            interface_readers.len(),
            cross_readers.len(),
            reader_buffer_bytes,
            outside_heap.capacity(),
            attach_heap.capacity(),
            interface_heap.capacity(),
            cross_heap.capacity()
        );
    }

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar H2 output {output_path:?}"))?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<ScalarKey>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_local_pairs_copy", None);
    }

    while let Some(value) =
        max_next_value([&outside_heap, &attach_heap, &interface_heap, &cross_heap])
    {
        drain_outside_value(value, &mut outside_readers, &mut outside_heap, |node| {
            if let Some(pair) = global_union_find.connect_outside(node, value)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
            Ok(())
        })?;

        drain_attach_value(
            value,
            &mut attach_readers,
            &mut attach_heap,
            |node, branch_birth| {
                if let Some(pair) = global_union_find.attach_branch(node, branch_birth, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(
            value,
            &mut interface_readers,
            &mut interface_heap,
            |a, b| {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(value, &mut cross_readers, &mut cross_heap, |a, b| {
            if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
            Ok(())
        })?;
    }

    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_event_reduction", None);
    }
    let remaining = global_union_find.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "scalar H2 streaming reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    output.commit()?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_output_commit", None);
    }
    Ok(ScalarH2PersistenceStats { finite_intervals })
}

fn reduce_h2_compact_runs_scan<K: DiskScalarKey>(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    memory_audit: bool,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "before_births_read", None);
    }
    let births = read_interface_births::<K>(&prepared.births_path, prepared.total_interface_nodes)?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_births_read", None);
        println!(
            "PROFILE_MEMVEC scalar_h2_reduce stage=after_births_read births_len={} births_capacity={} births_capacity_bytes={}",
            births.len(),
            births.capacity(),
            vec_capacity_bytes(&births)
        );
    }
    let mut global_union_find =
        CompactGlobalBackgroundPersistenceUnionFind::<K>::new(births, global_uf_layout);
    let (global_parent_bytes, global_rank_bytes, global_birth_bytes) =
        global_union_find.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H2_STATE strategy=compact uf_layout={} nodes={} key_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_uf_layout.as_str(),
        prepared.total_interface_nodes.saturating_add(1),
        K::DISK_BYTES,
        global_parent_bytes,
        global_rank_bytes,
        global_birth_bytes,
        global_parent_bytes + global_rank_bytes + global_birth_bytes,
    );
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_global_uf_alloc", None);
    }

    let mut outside_readers = prepared
        .outside_run_paths
        .iter()
        .map(|path| OutsideRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut attach_readers = prepared
        .attach_run_paths
        .iter()
        .map(|path| AttachRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut interface_readers = prepared
        .interface_run_paths
        .iter()
        .map(|path| PairRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut cross_readers = prepared
        .cross_run_paths
        .iter()
        .map(|path| PairRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar H2 output {output_path:?}"))?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<K>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    while let Some(value) = max_next_value_scan(
        &outside_readers,
        &attach_readers,
        &interface_readers,
        &cross_readers,
    ) {
        for reader in &mut outside_readers {
            while let Some(node) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.connect_outside(node, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut attach_readers {
            while let Some((node, branch_birth)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.attach_branch(node, branch_birth, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(value)? {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }
    }

    let remaining = global_union_find.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "scalar H2 streaming reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    output.commit()?;
    Ok(ScalarH2PersistenceStats { finite_intervals })
}

fn reduce_h2_compact_runs_heap<K: DiskScalarKey>(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    memory_audit: bool,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "before_births_read", None);
    }
    let births = read_interface_births::<K>(&prepared.births_path, prepared.total_interface_nodes)?;
    if memory_audit {
        emit_memory_snapshot("scalar_h2_reduce", "after_births_read", None);
    }
    let mut global_union_find =
        CompactGlobalBackgroundPersistenceUnionFind::<K>::new(births, global_uf_layout);
    let (global_parent_bytes, global_rank_bytes, global_birth_bytes) =
        global_union_find.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H2_STATE strategy=compact uf_layout={} nodes={} key_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_uf_layout.as_str(),
        prepared.total_interface_nodes.saturating_add(1),
        K::DISK_BYTES,
        global_parent_bytes,
        global_rank_bytes,
        global_birth_bytes,
        global_parent_bytes + global_rank_bytes + global_birth_bytes,
    );

    let (mut outside_readers, mut outside_heap) =
        open_outside_runs::<K>(&prepared.outside_run_paths)?;
    let (mut attach_readers, mut attach_heap) = open_attach_runs::<K>(&prepared.attach_run_paths)?;
    let (mut interface_readers, mut interface_heap) =
        open_pair_runs::<K>(&prepared.interface_run_paths)?;
    let (mut cross_readers, mut cross_heap) = open_pair_runs::<K>(&prepared.cross_run_paths)?;

    let mut output = AtomicOutput::create(output_path)
        .with_context(|| format!("could not create scalar H2 output {output_path:?}"))?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<K>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    while let Some(value) =
        max_next_value([&outside_heap, &attach_heap, &interface_heap, &cross_heap])
    {
        drain_outside_value(value, &mut outside_readers, &mut outside_heap, |node| {
            if let Some(pair) = global_union_find.connect_outside(node, value)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
            Ok(())
        })?;

        drain_attach_value(
            value,
            &mut attach_readers,
            &mut attach_heap,
            |node, branch_birth| {
                if let Some(pair) = global_union_find.attach_branch(node, branch_birth, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(
            value,
            &mut interface_readers,
            &mut interface_heap,
            |a, b| {
                if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(value, &mut cross_readers, &mut cross_heap, |a, b| {
            if let Some(pair) = global_union_find.union_with_persistence(a, b, value)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
            Ok(())
        })?;
    }

    let remaining = global_union_find.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "scalar H2 streaming reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    output.commit()?;
    Ok(ScalarH2PersistenceStats { finite_intervals })
}

fn effective_h2_merge_strategy(
    prepared: &PreparedH2ScalarRuns,
    requested: MergeStrategy,
) -> MergeStrategy {
    let reader_count = prepared.attach_run_paths.len()
        + prepared.outside_run_paths.len()
        + prepared.interface_run_paths.len()
        + prepared.cross_run_paths.len();
    let effective = match requested {
        MergeStrategy::Auto if reader_count > 32 => MergeStrategy::Heap,
        MergeStrategy::Auto => MergeStrategy::Scan,
        other => other,
    };
    println!(
        "PROFILE_MERGE_SELECT scalar_h2_stream requested={} effective={} readers={} threshold=32",
        requested.as_str(),
        effective.as_str(),
        reader_count,
    );
    effective
}

fn reduce_h2_compact_runs<K: DiskScalarKey>(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    merge_strategy: MergeStrategy,
    memory_audit: bool,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    match effective_h2_merge_strategy(prepared, merge_strategy) {
        MergeStrategy::Scan => {
            reduce_h2_compact_runs_scan::<K>(prepared, output_path, memory_audit, global_uf_layout)
        }
        MergeStrategy::Heap => {
            reduce_h2_compact_runs_heap::<K>(prepared, output_path, memory_audit, global_uf_layout)
        }
        MergeStrategy::Auto => {
            unreachable!("auto merge strategy must be resolved before reduction")
        }
    }
}

fn reduce_h2_scalar_runs(
    prepared: &PreparedH2ScalarRuns,
    output_path: &Path,
    merge_strategy: MergeStrategy,
    memory_audit: bool,
    global_birth_state: GlobalH2BirthStateStrategy,
    global_uf_layout: crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy,
) -> Result<ScalarH2PersistenceStats> {
    match effective_h2_merge_strategy(prepared, merge_strategy) {
        MergeStrategy::Scan => reduce_h2_scalar_runs_scan(
            prepared,
            output_path,
            memory_audit,
            global_birth_state,
            global_uf_layout,
        ),
        MergeStrategy::Heap => reduce_h2_scalar_runs_heap(
            prepared,
            output_path,
            memory_audit,
            global_birth_state,
            global_uf_layout,
        ),
        MergeStrategy::Auto => {
            unreachable!("auto merge strategy must be resolved before reduction")
        }
    }
}

pub fn compute_h2_persistence_scalar_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    output_path: &Path,
    tuning: ScalarStreamTuning,
) -> Result<ScalarH2PersistenceStats> {
    if matches!(
        tuning.global_h2_birth_state,
        GlobalH2BirthStateStrategy::Tagged
    ) && matches!(
        tuning.global_h2_uf_layout,
        crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy::Packed
    ) {
        bail!("--global-h2-uf-layout packed currently requires --global-h2-birth-state compact");
    }
    let start = Instant::now();
    let temp_directory = TempRunDirectory::create("h2_scalar_stream_runs")?;

    println!(
        "Scalar H2 streaming temporary directory: {:?}",
        temp_directory.path()
    );
    if tuning.h2_memory_audit {
        emit_memory_snapshot("scalar_h2_stream", "stream_start", None);
    }
    let native_f32_pipeline =
        volume.pixel_type == ScalarPixelType::F32 && tuning.f32_key_mode == F32KeyMode::Native32;
    if native_f32_pipeline
        && !matches!(
            tuning.global_h2_birth_state,
            GlobalH2BirthStateStrategy::Compact
        )
    {
        bail!(
            "H2 native32 end-to-end storage requires --global-h2-birth-state compact; use --f32-key-mode legacy64 for the tagged reference"
        );
    }
    println!(
        "Scalar H2 key storage: pipeline={} disk_key_bytes={} global_h2_uf_layout={}",
        if native_f32_pipeline {
            "native32-end-to-end"
        } else {
            "wide64"
        },
        if native_f32_pipeline {
            <F32Key as DiskScalarKey>::DISK_BYTES
        } else {
            <ScalarKey as DiskScalarKey>::DISK_BYTES
        },
        tuning.global_h2_uf_layout.as_str(),
    );

    let prepare_start = Instant::now();
    println!(
        "Scalar H2 tuning: merge={} interface_order={} event_order={} f32_key_mode={} neighbor_kernel={} representative_active_check={} union_kernel={} neighbor_root_check={} active_state={} interface_state={} uf_layout={} local_h2_birth_state={} phase_trim={} global_h2_birth_state={} global_h2_uf_layout={} sweep_diagnostics={} h2_memory_audit={}",
        tuning.merge_strategy.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.f32_key_mode.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.neighbor_root_check.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.local_h2_birth_state.as_str(),
        tuning.phase_trim.as_str(),
        tuning.global_h2_birth_state.as_str(),
        tuning.global_h2_uf_layout.as_str(),
        tuning.sweep_diagnostics,
        tuning.h2_memory_audit
    );
    let prepared = if native_f32_pipeline {
        prepare_h2_scalar_runs_f32_native(
            volume,
            slab_depth,
            background_connectivity,
            temp_directory.path(),
            tuning,
        )?
    } else {
        prepare_h2_scalar_runs_scalar(
            volume,
            slab_depth,
            background_connectivity,
            temp_directory.path(),
            tuning,
        )?
    };
    let prepare_seconds = prepare_start.elapsed().as_secs_f64();
    if tuning.h2_memory_audit {
        emit_memory_snapshot("scalar_h2_stream", "after_prepare", None);
        println!(
            "PROFILE_MEMVEC scalar_h2_stream stage=after_prepare total_interface_nodes={} attach_runs={} outside_runs={} interface_runs={} cross_runs={}",
            prepared.total_interface_nodes,
            prepared.attach_run_paths.len(),
            prepared.outside_run_paths.len(),
            prepared.interface_run_paths.len(),
            prepared.cross_run_paths.len(),
        );
    }
    println!("Scalar H2 slab preparation completed in {prepare_seconds:.3} seconds.");
    println!(
        "PROFILE_H2_STORAGE scalar_h2_stream pipeline={} disk_key_bytes={} interface_nodes={} interface_birth_bytes={} local_pair_bytes={} attach_bytes={} outside_bytes={} interface_bytes={} cross_bytes={} total_run_bytes={} global_h2_birth_state={} global_h2_uf_layout={}",
        if native_f32_pipeline {
            "native32-end-to-end"
        } else {
            "wide64"
        },
        prepared.disk_key_bytes,
        prepared.total_interface_nodes,
        prepared.interface_birth_bytes,
        prepared.local_pair_bytes,
        prepared.attach_bytes,
        prepared.outside_bytes,
        prepared.interface_bytes,
        prepared.cross_bytes,
        prepared.total_run_bytes,
        tuning.global_h2_birth_state.as_str(),
        tuning.global_h2_uf_layout.as_str(),
    );

    let trim_start = Instant::now();
    let (trim_attempted, trim_released) = match tuning.phase_trim {
        PhaseTrimStrategy::Off => (false, false),
        PhaseTrimStrategy::BeforeReduce => {
            if tuning.h2_memory_audit {
                emit_memory_snapshot("scalar_h2_stream", "before_phase_trim", None);
            }
            let outcome = trim_process_heap()?;
            if tuning.h2_memory_audit {
                emit_memory_snapshot("scalar_h2_stream", "after_phase_trim", None);
            }
            (true, outcome.released)
        }
    };
    let trim_seconds = trim_start.elapsed().as_secs_f64();
    println!(
        "PROFILE_TRIM scalar_h2_stream strategy={} attempted={} released={} seconds={trim_seconds:.6}",
        tuning.phase_trim.as_str(),
        trim_attempted,
        trim_released,
    );

    println!("Starting disk-backed scalar H2 reduction.");

    let reduce_start = Instant::now();
    let result = if native_f32_pipeline {
        reduce_h2_compact_runs::<F32Key>(
            &prepared,
            output_path,
            tuning.merge_strategy,
            tuning.h2_memory_audit,
            tuning.global_h2_uf_layout,
        )
    } else {
        reduce_h2_scalar_runs(
            &prepared,
            output_path,
            tuning.merge_strategy,
            tuning.h2_memory_audit,
            tuning.global_h2_birth_state,
            tuning.global_h2_uf_layout,
        )
    };
    let stats = result?;
    let reduce_seconds = reduce_start.elapsed().as_secs_f64();
    if tuning.h2_memory_audit {
        emit_memory_snapshot("scalar_h2_stream", "after_reduce", None);
    }

    let cleanup_start = Instant::now();
    temp_directory.close()?;
    let cleanup_seconds = cleanup_start.elapsed().as_secs_f64();
    if tuning.h2_memory_audit {
        emit_memory_snapshot("scalar_h2_stream", "after_cleanup", None);
    }
    let total_seconds = start.elapsed().as_secs_f64();

    println!("Streaming scalar H2 persistence computation took {total_seconds:.3} seconds");
    println!(
        "PROFILE scalar_h2_stream prepare_seconds={prepare_seconds:.6} \
reduce_seconds={reduce_seconds:.6} cleanup_seconds={cleanup_seconds:.6} \
total_seconds={total_seconds:.6}"
    );
    let detailed = prepared.preparation_profile;
    let detailed_sum = detailed.decode_seconds
        + detailed.key_conversion_seconds
        + detailed.slab_copy_seconds
        + detailed.scalar_order_seconds
        + detailed.local_sweep_seconds
        + detailed.local_run_write_seconds
        + detailed.cross_interface_seconds;
    let unaccounted_seconds = (prepare_seconds - detailed_sum).max(0.0);
    println!(
        "PROFILE_PREP scalar_h2_stream decode_seconds={:.6} key_conversion_seconds={:.6} \
slab_copy_seconds={:.6} scalar_order_seconds={:.6} local_sweep_seconds={:.6} \
local_run_write_seconds={:.6} cross_interface_seconds={:.6} unaccounted_seconds={:.6} \
total_voxels={} interior_fast_voxels={}",
        detailed.decode_seconds,
        detailed.key_conversion_seconds,
        detailed.slab_copy_seconds,
        detailed.scalar_order_seconds,
        detailed.local_sweep_seconds,
        detailed.local_run_write_seconds,
        detailed.cross_interface_seconds,
        unaccounted_seconds,
        detailed.total_voxels,
        detailed.interior_fast_voxels
    );

    if tuning.sweep_diagnostics {
        println!(
            "PROFILE_SWEEP scalar_h2_stream pruning_mask_calls={} active_state_checks={} active_neighbor_hits={} \
representative_visits={} pruning_cache_hits={} pruning_cache_misses={} \
component_mask_computations={} union_attempts={} successful_unions={} \
same_root_unions={} find_calls={} find_parent_steps={} active_rechecks={} \
active_recheck_failures={} root_carry_attempts={} avoided_find_calls={} \
direct_parent_checks={} direct_parent_hits={} avoided_neighbor_find_calls={} \
interface_rep_queries={} interface_rep_writes={} interface_forced_root_unions={} \
interface_interface_unions={} interface_state_bytes={} max_rank_observed={} uf_parent_state_bytes={} uf_rank_state_bytes={}",
            detailed.pruning_mask_calls,
            detailed.active_state_checks,
            detailed.active_neighbor_hits,
            detailed.representative_visits,
            detailed.pruning_cache_hits,
            detailed.pruning_cache_misses,
            detailed.component_mask_computations,
            detailed.union_attempts,
            detailed.successful_unions,
            detailed.same_root_unions,
            detailed.find_calls,
            detailed.find_parent_steps,
            detailed.active_rechecks,
            detailed.active_recheck_failures,
            detailed.root_carry_attempts,
            detailed.avoided_find_calls,
            detailed.direct_parent_checks,
            detailed.direct_parent_hits,
            detailed.avoided_neighbor_find_calls,
            detailed.interface_rep_queries,
            detailed.interface_rep_writes,
            detailed.interface_forced_root_unions,
            detailed.interface_interface_unions,
            detailed.interface_state_bytes,
            detailed.max_rank_observed,
            detailed.uf_parent_state_bytes,
            detailed.uf_rank_state_bytes
        );
        if matches!(
            tuning.representative_active_check,
            crate::scalar_stream_tuning::RepresentativeActiveCheckStrategy::Recheck
        ) && detailed.active_recheck_failures != 0
        {
            bail!(
                "local pruner returned {} inactive representative neighbors",
                detailed.active_recheck_failures
            );
        }
    }

    println!(
        "PROFILE_CONFIG scalar_h2_stream merge_strategy={} interface_order={} event_order={} f32_key_mode={} neighbor_kernel={} representative_active_check={} union_kernel={} neighbor_root_check={} active_state={} interface_state={} uf_layout={} local_h2_birth_state={} phase_trim={} global_h2_birth_state={} global_h2_uf_layout={} sweep_diagnostics={} h2_memory_audit={}",
        tuning.merge_strategy.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.f32_key_mode.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.neighbor_root_check.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.local_h2_birth_state.as_str(),
        tuning.phase_trim.as_str(),
        tuning.global_h2_birth_state.as_str(),
        tuning.global_h2_uf_layout.as_str(),
        tuning.sweep_diagnostics,
        tuning.h2_memory_audit
    );
    Ok(stats)
}

const H2_HIER_PACKED_ROOT_BASE: u32 = u32::MAX - u8::MAX as u32;

struct DiskHierarchicalH2Summary<K: DiskScalarKey> {
    z0: usize,
    z1: usize,
    width: usize,
    height: usize,
    interface_node_count: u32,
    lower_values: Vec<K>,
    upper_values: Vec<K>,
    attach_path: PathBuf,
    outside_path: PathBuf,
    interface_path: PathBuf,
}

impl<K: DiskScalarKey> DiskHierarchicalH2Summary<K> {
    fn depth(&self) -> usize {
        self.z1 - self.z0
    }

    fn face_size(&self) -> usize {
        self.width * self.height
    }

    fn upper_local_id(&self, face_index: usize) -> u32 {
        let local = if self.depth() == 1 {
            face_index
        } else {
            self.face_size() + face_index
        };
        u32::try_from(local).expect("hierarchical H2 face index exceeds u32")
    }
}

#[derive(Debug, Clone, Copy)]
enum DiskHierarchicalH2Action<K> {
    None,
    FinalPair(FinitePair<K>),
    Attach(AttachDiskEvent<K>),
    Outside(OutsideDiskEvent<K>),
    Interface(PairEvent<K>),
}

struct DiskHierarchicalH2UnionFind<K: DiskScalarKey> {
    parent: Vec<u32>,
    birth: Vec<K>,
    left_count: u32,
    right_offset: u32,
    right_upper_start: u32,
    face_size: u32,
    attach_finalized_early: u64,
    attach_propagated: u64,
    outside_propagated: u64,
    prune_outside_dominated_structural: bool,
    outside_structural_elided: u64,
}

impl<K: DiskScalarKey> DiskHierarchicalH2UnionFind<K> {
    fn new(
        birth: Vec<K>,
        left_count: u32,
        right_offset: u32,
        right_upper_start: u32,
        face_size: u32,
        prune_outside_dominated_structural: bool,
    ) -> Result<Self> {
        if birth.len() >= H2_HIER_PACKED_ROOT_BASE as usize {
            bail!(
                "hierarchical packed H2 pair needs {} nodes, but the packed root encoding supports fewer than {}",
                birth.len(),
                H2_HIER_PACKED_ROOT_BASE
            );
        }
        debug_assert!(birth.iter().all(|value| !value.is_outside_marker()));
        Ok(Self {
            parent: vec![Self::root_word(0); birth.len()],
            birth,
            left_count,
            right_offset,
            right_upper_start,
            face_size,
            attach_finalized_early: 0,
            attach_propagated: 0,
            outside_propagated: 0,
            prune_outside_dominated_structural,
            outside_structural_elided: 0,
        })
    }

    #[inline]
    fn root_word(rank: u8) -> u32 {
        u32::MAX - u32::from(rank)
    }

    #[inline]
    fn root_rank(word: u32) -> Option<u8> {
        (word >= H2_HIER_PACKED_ROOT_BASE).then(|| (u32::MAX - word) as u8)
    }

    fn find(&mut self, mut node: u32) -> u32 {
        loop {
            let word = self.parent[node as usize];
            if Self::root_rank(word).is_some() {
                return node;
            }
            let parent = word;
            let parent_word = self.parent[parent as usize];
            if Self::root_rank(parent_word).is_none() {
                self.parent[node as usize] = parent_word;
            }
            node = parent;
        }
    }

    fn terminal_parent_id(&self, root: u32) -> Option<u32> {
        if root < self.face_size && root < self.left_count {
            return Some(root);
        }
        let start = self.right_offset + self.right_upper_start;
        let end = start + self.face_size;
        if root >= start && root < end {
            return Some(self.face_size + (root - start));
        }
        None
    }

    fn rank(&self, root: u32) -> u8 {
        Self::root_rank(self.parent[root as usize]).expect("hierarchical H2 root marker missing")
    }

    fn set_rank(&mut self, root: u32, rank: u8) {
        self.parent[root as usize] = Self::root_word(rank);
    }

    fn force_link(&mut self, survivor: u32, absorbed: u32) -> u32 {
        let survivor_rank = self.rank(survivor);
        let absorbed_rank = self.rank(absorbed);
        self.parent[absorbed as usize] = survivor;
        if absorbed_rank >= survivor_rank {
            self.set_rank(survivor, absorbed_rank.saturating_add(1));
        }
        survivor
    }

    fn rank_link(&mut self, mut a: u32, mut b: u32) -> u32 {
        let rank_a = self.rank(a);
        let rank_b = self.rank(b);
        if rank_a < rank_b {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        if rank_a == rank_b {
            self.set_rank(a, rank_a.saturating_add(1));
        }
        a
    }

    fn link_preserving_terminal(&mut self, a: u32, b: u32) -> u32 {
        match (
            self.terminal_parent_id(a).is_some(),
            self.terminal_parent_id(b).is_some(),
        ) {
            (true, false) => self.force_link(a, b),
            (false, true) => self.force_link(b, a),
            (true, true) => self.force_link(a, b),
            (false, false) => self.rank_link(a, b),
        }
    }

    #[inline]
    fn is_outside_birth(value: K) -> bool {
        value.is_outside_marker()
    }

    #[inline]
    fn older_birth(a: K, b: K) -> K {
        if Self::is_outside_birth(a) || Self::is_outside_birth(b) {
            K::outside_marker()
        } else {
            a.max(b)
        }
    }

    fn pair_for_merge(a: K, b: K, merge_value: K) -> Option<FinitePair<K>> {
        match (Self::is_outside_birth(a), Self::is_outside_birth(b)) {
            (true, true) => None,
            (true, false) => Some(FinitePair {
                birth: merge_value,
                death: b,
            }),
            (false, true) => Some(FinitePair {
                birth: merge_value,
                death: a,
            }),
            (false, false) => Some(FinitePair {
                birth: merge_value,
                death: a.min(b),
            }),
        }
    }

    fn union_with_summary(
        &mut self,
        a: u32,
        b: u32,
        merge_value: K,
    ) -> Option<DiskHierarchicalH2Action<K>> {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return None;
        }
        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let terminal_a = self.terminal_parent_id(root_a);
        let terminal_b = self.terminal_parent_id(root_b);

        let action = match (terminal_a, terminal_b) {
            (None, None) => Self::pair_for_merge(birth_a, birth_b, merge_value)
                .map(DiskHierarchicalH2Action::FinalPair)
                .unwrap_or(DiskHierarchicalH2Action::None),
            (Some(node), None) => {
                if Self::is_outside_birth(birth_b) {
                    if Self::is_outside_birth(birth_a) {
                        DiskHierarchicalH2Action::None
                    } else {
                        self.outside_propagated += 1;
                        DiskHierarchicalH2Action::Outside(OutsideDiskEvent {
                            value: merge_value,
                            node,
                        })
                    }
                } else if Self::is_outside_birth(birth_a) || birth_a >= birth_b {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH2Action::FinalPair(FinitePair {
                        birth: merge_value,
                        death: birth_b,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH2Action::Attach(AttachDiskEvent {
                        value: merge_value,
                        node,
                        branch_birth: birth_b,
                    })
                }
            }
            (None, Some(node)) => {
                if Self::is_outside_birth(birth_a) {
                    if Self::is_outside_birth(birth_b) {
                        DiskHierarchicalH2Action::None
                    } else {
                        self.outside_propagated += 1;
                        DiskHierarchicalH2Action::Outside(OutsideDiskEvent {
                            value: merge_value,
                            node,
                        })
                    }
                } else if Self::is_outside_birth(birth_b) || birth_b >= birth_a {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH2Action::FinalPair(FinitePair {
                        birth: merge_value,
                        death: birth_a,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH2Action::Attach(AttachDiskEvent {
                        value: merge_value,
                        node,
                        branch_birth: birth_a,
                    })
                }
            }
            (Some(a), Some(b)) => {
                if self.prune_outside_dominated_structural {
                    match (
                        Self::is_outside_birth(birth_a),
                        Self::is_outside_birth(birth_b),
                    ) {
                        (true, true) => {
                            self.outside_structural_elided += 1;
                            DiskHierarchicalH2Action::None
                        }
                        (true, false) => {
                            self.outside_structural_elided += 1;
                            self.outside_propagated += 1;
                            DiskHierarchicalH2Action::Outside(OutsideDiskEvent {
                                value: merge_value,
                                node: b,
                            })
                        }
                        (false, true) => {
                            self.outside_structural_elided += 1;
                            self.outside_propagated += 1;
                            DiskHierarchicalH2Action::Outside(OutsideDiskEvent {
                                value: merge_value,
                                node: a,
                            })
                        }
                        (false, false) => DiskHierarchicalH2Action::Interface(PairEvent {
                            value: merge_value,
                            a,
                            b,
                        }),
                    }
                } else {
                    DiskHierarchicalH2Action::Interface(PairEvent {
                        value: merge_value,
                        a,
                        b,
                    })
                }
            }
        };

        let merged_birth = Self::older_birth(birth_a, birth_b);
        let new_root = self.link_preserving_terminal(root_a, root_b);
        self.birth[new_root as usize] = merged_birth;
        Some(action)
    }

    fn attach_branch(
        &mut self,
        node: u32,
        branch_birth: K,
        merge_value: K,
    ) -> DiskHierarchicalH2Action<K> {
        let root = self.find(node);
        let root_birth = self.birth[root as usize];
        let terminal = self.terminal_parent_id(root);
        let merged = Self::older_birth(root_birth, branch_birth);
        self.birth[root as usize] = merged;
        match terminal {
            Some(node) => {
                if Self::is_outside_birth(root_birth) || root_birth >= branch_birth {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH2Action::FinalPair(FinitePair {
                        birth: merge_value,
                        death: branch_birth,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH2Action::Attach(AttachDiskEvent {
                        value: merge_value,
                        node,
                        branch_birth,
                    })
                }
            }
            None => Self::pair_for_merge(root_birth, branch_birth, merge_value)
                .map(DiskHierarchicalH2Action::FinalPair)
                .unwrap_or(DiskHierarchicalH2Action::None),
        }
    }

    fn connect_outside(&mut self, node: u32, merge_value: K) -> DiskHierarchicalH2Action<K> {
        let root = self.find(node);
        let root_birth = self.birth[root as usize];
        if Self::is_outside_birth(root_birth) {
            return DiskHierarchicalH2Action::None;
        }
        self.birth[root as usize] = K::outside_marker();
        match self.terminal_parent_id(root) {
            Some(node) => {
                self.outside_propagated += 1;
                DiskHierarchicalH2Action::Outside(OutsideDiskEvent {
                    value: merge_value,
                    node,
                })
            }
            None => DiskHierarchicalH2Action::FinalPair(FinitePair {
                birth: merge_value,
                death: root_birth,
            }),
        }
    }

    fn counts(&self) -> (u64, u64, u64, u64) {
        (
            self.attach_finalized_early,
            self.attach_propagated,
            self.outside_propagated,
            self.outside_structural_elided,
        )
    }

    fn count_internal_finite_roots(&mut self) -> usize {
        let mut count = 0usize;
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32
                && self.terminal_parent_id(node_u32).is_none()
                && !Self::is_outside_birth(self.birth[node])
            {
                count += 1;
            }
        }
        count
    }
}

fn write_hierarchical_h2_action<K: DiskScalarKey>(
    action: DiskHierarchicalH2Action<K>,
    pair_writer: &mut BufWriter<File>,
    attach_writer: &mut BufWriter<File>,
    outside_writer: &mut BufWriter<File>,
    interface_writer: &mut BufWriter<File>,
) -> Result<()> {
    match action {
        DiskHierarchicalH2Action::None => {}
        DiskHierarchicalH2Action::FinalPair(pair) => {
            if pair.birth < pair.death {
                write_finite_pair(pair_writer, pair)?;
            }
        }
        DiskHierarchicalH2Action::Attach(event) => write_attach_event(attach_writer, event)?,
        DiskHierarchicalH2Action::Outside(event) => write_outside_event(outside_writer, event)?,
        DiskHierarchicalH2Action::Interface(event) => write_pair_event(interface_writer, event)?,
    }
    Ok(())
}

fn append_hierarchical_h2_births<K: DiskScalarKey>(
    summary: &DiskHierarchicalH2Summary<K>,
    births: &mut Vec<K>,
) {
    births.extend_from_slice(&summary.lower_values);
    if summary.depth() > 1 {
        births.extend_from_slice(&summary.upper_values);
    }
}

fn next_hierarchical_h2_value<K: DiskScalarKey>(
    left_outside: &OutsideRunReader<K>,
    right_outside: &OutsideRunReader<K>,
    left_attach: &AttachRunReader<K>,
    right_attach: &AttachRunReader<K>,
    left_interface: &PairRunReader<K>,
    right_interface: &PairRunReader<K>,
    cross: &PairRunReader<K>,
) -> Option<K> {
    [
        left_outside.next_value(),
        right_outside.next_value(),
        left_attach.next_value(),
        right_attach.next_value(),
        left_interface.next_value(),
        right_interface.next_value(),
        cross.next_value(),
    ]
    .into_iter()
    .flatten()
    .max()
}

fn next_hierarchical_h2_child_value<K: DiskScalarKey>(
    left_outside: &OutsideRunReader<K>,
    right_outside: &OutsideRunReader<K>,
    left_attach: &AttachRunReader<K>,
    right_attach: &AttachRunReader<K>,
    left_interface: &PairRunReader<K>,
    right_interface: &PairRunReader<K>,
) -> Option<K> {
    [
        left_outside.next_value(),
        right_outside.next_value(),
        left_attach.next_value(),
        right_attach.next_value(),
        left_interface.next_value(),
        right_interface.next_value(),
    ]
    .into_iter()
    .flatten()
    .max()
}

#[allow(clippy::too_many_arguments)]
fn drain_hierarchical_h2_child_value<K: DiskScalarKey>(
    value: K,
    left_outside: &mut OutsideRunReader<K>,
    right_outside: &mut OutsideRunReader<K>,
    left_attach: &mut AttachRunReader<K>,
    right_attach: &mut AttachRunReader<K>,
    left_interface: &mut PairRunReader<K>,
    right_interface: &mut PairRunReader<K>,
    right_offset: u32,
    uf: &mut DiskHierarchicalH2UnionFind<K>,
    pair_writer: &mut BufWriter<File>,
    parent_attach_writer: &mut BufWriter<File>,
    parent_outside_writer: &mut BufWriter<File>,
    parent_interface_writer: &mut BufWriter<File>,
) -> Result<()> {
    while let Some(node) = left_outside.pop_at_value(value)? {
        let action = uf.connect_outside(node, value);
        write_hierarchical_h2_action(
            action,
            pair_writer,
            parent_attach_writer,
            parent_outside_writer,
            parent_interface_writer,
        )?;
    }
    while let Some(node) = right_outside.pop_at_value(value)? {
        let action = uf.connect_outside(right_offset + node, value);
        write_hierarchical_h2_action(
            action,
            pair_writer,
            parent_attach_writer,
            parent_outside_writer,
            parent_interface_writer,
        )?;
    }
    while let Some((node, branch_birth)) = left_attach.pop_at_value(value)? {
        let action = uf.attach_branch(node, branch_birth, value);
        write_hierarchical_h2_action(
            action,
            pair_writer,
            parent_attach_writer,
            parent_outside_writer,
            parent_interface_writer,
        )?;
    }
    while let Some((node, branch_birth)) = right_attach.pop_at_value(value)? {
        let action = uf.attach_branch(right_offset + node, branch_birth, value);
        write_hierarchical_h2_action(
            action,
            pair_writer,
            parent_attach_writer,
            parent_outside_writer,
            parent_interface_writer,
        )?;
    }
    while let Some((a, b)) = left_interface.pop_at_value(value)? {
        if let Some(action) = uf.union_with_summary(a, b, value) {
            write_hierarchical_h2_action(
                action,
                pair_writer,
                parent_attach_writer,
                parent_outside_writer,
                parent_interface_writer,
            )?;
        }
    }
    while let Some((a, b)) = right_interface.pop_at_value(value)? {
        if let Some(action) = uf.union_with_summary(right_offset + a, right_offset + b, value) {
            write_hierarchical_h2_action(
                action,
                pair_writer,
                parent_attach_writer,
                parent_outside_writer,
                parent_interface_writer,
            )?;
        }
    }
    Ok(())
}

type DiskH2CombineResult<K> = (DiskHierarchicalH2Summary<K>, usize, u64, u64, u64, u64, u64);

#[allow(clippy::too_many_arguments)]
fn combine_disk_h2_summaries_direct_cross<K: DiskScalarKey>(
    left: DiskHierarchicalH2Summary<K>,
    right: DiskHierarchicalH2Summary<K>,
    background_connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
    prune_outside_dominated_structural: bool,
) -> Result<DiskH2CombineResult<K>> {
    if left.z1 != right.z0 {
        bail!(
            "disk-backed hierarchical H2 summaries are not adjacent: left z={}..{}, right z={}..{}",
            left.z0,
            left.z1,
            right.z0,
            right.z1
        );
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H2 summary face dimensions differ");
    }
    let face_size = left.face_size();
    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if total_nodes >= H2_HIER_PACKED_ROOT_BASE as usize {
        bail!("disk-backed hierarchical H2 pair is too large for packed u32 roots");
    }
    if left.lower_values.is_empty() || right.lower_values.is_empty() {
        bail!("empty hierarchical H2 boundary face");
    }
    let mut births = Vec::with_capacity(total_nodes);
    append_hierarchical_h2_births(&left, &mut births);
    let right_offset = births.len();
    append_hierarchical_h2_births(&right, &mut births);
    debug_assert_eq!(births.len(), total_nodes);
    let right_upper_start = if right.depth() == 1 { 0 } else { face_size };
    let mut uf = DiskHierarchicalH2UnionFind::new(
        births,
        left.interface_node_count,
        u32::try_from(right_offset).expect("hierarchical H2 right offset exceeds u32"),
        u32::try_from(right_upper_start).expect("hierarchical H2 right upper offset exceeds u32"),
        u32::try_from(face_size).expect("hierarchical H2 face size exceeds u32"),
        prune_outside_dominated_structural,
    )?;

    let parent_attach_path = directory.join(format!("h2_hier_attach_{summary_id:06}.bin"));
    let parent_outside_path = directory.join(format!("h2_hier_outside_{summary_id:06}.bin"));
    let parent_interface_path = directory.join(format!("h2_hier_interface_{summary_id:06}.bin"));
    let mut parent_attach_writer = BufWriter::new(File::create(&parent_attach_path)?);
    let mut parent_outside_writer = BufWriter::new(File::create(&parent_outside_path)?);
    let mut parent_interface_writer = BufWriter::new(File::create(&parent_interface_path)?);

    let mut left_outside = OutsideRunReader::<K>::open(&left.outside_path)?;
    let mut right_outside = OutsideRunReader::<K>::open(&right.outside_path)?;
    let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
    let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
    let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
    let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
    let offset = u32::try_from(right_offset).expect("hierarchical H2 right offset exceeds u32");
    let mut previous_cross_value: Option<K> = None;

    let cross_stats = sparsify_scalar_superlevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        background_connectivity,
        interface_order,
        |edge| {
            if previous_cross_value.is_some_and(|previous| edge.value > previous) {
                bail!("direct hierarchical H2 cross edges are not monotone");
            }
            previous_cross_value = Some(edge.value);

            loop {
                let Some(child_value) = next_hierarchical_h2_child_value(
                    &left_outside,
                    &right_outside,
                    &left_attach,
                    &right_attach,
                    &left_interface,
                    &right_interface,
                ) else {
                    break;
                };
                if child_value < edge.value {
                    break;
                }
                drain_hierarchical_h2_child_value(
                    child_value,
                    &mut left_outside,
                    &mut right_outside,
                    &mut left_attach,
                    &mut right_attach,
                    &mut left_interface,
                    &mut right_interface,
                    offset,
                    &mut uf,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
                if child_value == edge.value {
                    break;
                }
            }

            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = offset
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H2 face index exceeds u32");
            if let Some(action) = uf.union_with_summary(left_node, right_node, edge.value) {
                write_hierarchical_h2_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
            }
            Ok(())
        },
    )?;

    while let Some(value) = next_hierarchical_h2_child_value(
        &left_outside,
        &right_outside,
        &left_attach,
        &right_attach,
        &left_interface,
        &right_interface,
    ) {
        drain_hierarchical_h2_child_value(
            value,
            &mut left_outside,
            &mut right_outside,
            &mut left_attach,
            &mut right_attach,
            &mut left_interface,
            &mut right_interface,
            offset,
            &mut uf,
            pair_writer,
            &mut parent_attach_writer,
            &mut parent_outside_writer,
            &mut parent_interface_writer,
        )?;
    }

    let internal_finite_roots = uf.count_internal_finite_roots();
    if internal_finite_roots != 0 {
        bail!(
            "direct-cross hierarchical H2 composition left {internal_finite_roots} finite components without an outer-face representative"
        );
    }

    parent_attach_writer.flush()?;
    parent_outside_writer.flush()?;
    parent_interface_writer.flush()?;
    drop(parent_attach_writer);
    drop(parent_outside_writer);
    drop(parent_interface_writer);

    for path in [
        &left.attach_path,
        &left.outside_path,
        &left.interface_path,
        &right.attach_path,
        &right.outside_path,
        &right.interface_path,
    ] {
        fs::remove_file(path)
            .with_context(|| format!("could not delete consumed hierarchical H2 run {path:?}"))?;
    }

    let parent_depth = right.z1 - left.z0;
    let parent_interface_nodes = if parent_depth == 1 {
        face_size
    } else {
        2 * face_size
    };
    let state_bytes = u64::try_from(total_nodes)
        .expect("hierarchical H2 pair node count exceeds u64")
        .saturating_mul((4 + K::DISK_BYTES) as u64);
    let parent_attach_bytes = path_bytes(&parent_attach_path)?;
    let parent_outside_bytes = path_bytes(&parent_outside_path)?;
    let parent_interface_bytes = path_bytes(&parent_interface_path)?;
    let cross_run_bytes_avoided = cross_stats
        .retained_edges
        .saturating_mul((K::DISK_BYTES + 8) as u64);
    let (attach_finalized_early, attach_propagated, outside_propagated, outside_structural_elided) =
        uf.counts();
    println!(
        "PROFILE_H2_HIER_STREAM_COMBINE z0={} z1={} pair_nodes={} pair_state_bytes={} parent_interface_nodes={} cross_retained={} cross_candidates={} parent_attach_bytes={} parent_outside_bytes={} parent_interface_bytes={} attach_finalized_early={} attach_propagated={} outside_propagated={} outside_structural_elided={} cross_storage=direct cross_run_bytes=0 cross_run_bytes_avoided={}",
        left.z0,
        right.z1,
        total_nodes,
        state_bytes,
        parent_interface_nodes,
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
        parent_attach_bytes,
        parent_outside_bytes,
        parent_interface_bytes,
        attach_finalized_early,
        attach_propagated,
        outside_propagated,
        outside_structural_elided,
        cross_run_bytes_avoided,
    );

    Ok((
        DiskHierarchicalH2Summary {
            z0: left.z0,
            z1: right.z1,
            width: left.width,
            height: left.height,
            interface_node_count: u32::try_from(parent_interface_nodes)
                .expect("hierarchical H2 parent interface count exceeds u32"),
            lower_values: left.lower_values,
            upper_values: right.upper_values,
            attach_path: parent_attach_path,
            outside_path: parent_outside_path,
            interface_path: parent_interface_path,
        },
        total_nodes,
        state_bytes,
        attach_finalized_early,
        attach_propagated,
        outside_propagated,
        outside_structural_elided,
    ))
}

#[allow(clippy::too_many_arguments)]
fn combine_disk_h2_summaries<K: DiskScalarKey>(
    left: DiskHierarchicalH2Summary<K>,
    right: DiskHierarchicalH2Summary<K>,
    background_connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
    prune_outside_dominated_structural: bool,
) -> Result<DiskH2CombineResult<K>> {
    if left.z1 != right.z0 {
        bail!(
            "disk-backed hierarchical H2 summaries are not adjacent: left z={}..{}, right z={}..{}",
            left.z0,
            left.z1,
            right.z0,
            right.z1
        );
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H2 summary face dimensions differ");
    }
    let face_size = left.face_size();
    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if total_nodes >= H2_HIER_PACKED_ROOT_BASE as usize {
        bail!("disk-backed hierarchical H2 pair is too large for packed u32 roots");
    }
    if left.lower_values.is_empty() || right.lower_values.is_empty() {
        bail!("empty hierarchical H2 boundary face");
    }
    let mut births = Vec::with_capacity(total_nodes);
    append_hierarchical_h2_births(&left, &mut births);
    let right_offset = births.len();
    append_hierarchical_h2_births(&right, &mut births);
    debug_assert_eq!(births.len(), total_nodes);
    let right_upper_start = if right.depth() == 1 { 0 } else { face_size };
    let mut uf = DiskHierarchicalH2UnionFind::new(
        births,
        left.interface_node_count,
        u32::try_from(right_offset).expect("hierarchical H2 right offset exceeds u32"),
        u32::try_from(right_upper_start).expect("hierarchical H2 right upper offset exceeds u32"),
        u32::try_from(face_size).expect("hierarchical H2 face size exceeds u32"),
        prune_outside_dominated_structural,
    )?;

    let cross_path = directory.join(format!("h2_hier_cross_{summary_id:06}.bin"));
    let mut cross_writer = BufWriter::new(File::create(&cross_path)?);
    let cross_stats = sparsify_scalar_superlevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        background_connectivity,
        interface_order,
        |edge| {
            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = u32::try_from(right_offset)
                .expect("hierarchical H2 right offset exceeds u32")
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H2 face index exceeds u32");
            write_pair_event(
                &mut cross_writer,
                PairEvent {
                    value: edge.value,
                    a: left_node,
                    b: right_node,
                },
            )
        },
    )?;
    cross_writer.flush()?;
    drop(cross_writer);

    let parent_attach_path = directory.join(format!("h2_hier_attach_{summary_id:06}.bin"));
    let parent_outside_path = directory.join(format!("h2_hier_outside_{summary_id:06}.bin"));
    let parent_interface_path = directory.join(format!("h2_hier_interface_{summary_id:06}.bin"));
    let mut parent_attach_writer = BufWriter::new(File::create(&parent_attach_path)?);
    let mut parent_outside_writer = BufWriter::new(File::create(&parent_outside_path)?);
    let mut parent_interface_writer = BufWriter::new(File::create(&parent_interface_path)?);

    {
        let mut left_outside = OutsideRunReader::<K>::open(&left.outside_path)?;
        let mut right_outside = OutsideRunReader::<K>::open(&right.outside_path)?;
        let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
        let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
        let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
        let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
        let mut cross = PairRunReader::<K>::open(&cross_path)?;
        let offset = u32::try_from(right_offset).expect("hierarchical H2 right offset exceeds u32");

        while let Some(value) = next_hierarchical_h2_value(
            &left_outside,
            &right_outside,
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
        ) {
            // Preserve the flat H2 deterministic priority at equal superlevel
            // value: outside, attach, interface, then cross-interface edges.
            while let Some(node) = left_outside.pop_at_value(value)? {
                let action = uf.connect_outside(node, value);
                write_hierarchical_h2_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some(node) = right_outside.pop_at_value(value)? {
                let action = uf.connect_outside(offset + node, value);
                write_hierarchical_h2_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some((node, branch_birth)) = left_attach.pop_at_value(value)? {
                let action = uf.attach_branch(node, branch_birth, value);
                write_hierarchical_h2_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some((node, branch_birth)) = right_attach.pop_at_value(value)? {
                let action = uf.attach_branch(offset + node, branch_birth, value);
                write_hierarchical_h2_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_outside_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some((a, b)) = left_interface.pop_at_value(value)? {
                if let Some(action) = uf.union_with_summary(a, b, value) {
                    write_hierarchical_h2_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_outside_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
            while let Some((a, b)) = right_interface.pop_at_value(value)? {
                if let Some(action) = uf.union_with_summary(offset + a, offset + b, value) {
                    write_hierarchical_h2_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_outside_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
            while let Some((a, b)) = cross.pop_at_value(value)? {
                if let Some(action) = uf.union_with_summary(a, b, value) {
                    write_hierarchical_h2_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_outside_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
        }
    }

    let internal_finite_roots = uf.count_internal_finite_roots();
    if internal_finite_roots != 0 {
        bail!(
            "disk-backed hierarchical H2 composition left {internal_finite_roots} finite components without an outer-face representative"
        );
    }

    parent_attach_writer.flush()?;
    parent_outside_writer.flush()?;
    parent_interface_writer.flush()?;
    drop(parent_attach_writer);
    drop(parent_outside_writer);
    drop(parent_interface_writer);

    for path in [
        &left.attach_path,
        &left.outside_path,
        &left.interface_path,
        &right.attach_path,
        &right.outside_path,
        &right.interface_path,
        &cross_path,
    ] {
        fs::remove_file(path)
            .with_context(|| format!("could not delete consumed hierarchical H2 run {path:?}"))?;
    }

    let parent_depth = right.z1 - left.z0;
    let parent_interface_nodes = if parent_depth == 1 {
        face_size
    } else {
        2 * face_size
    };
    let state_bytes = u64::try_from(total_nodes)
        .expect("hierarchical H2 pair node count exceeds u64")
        .saturating_mul((4 + K::DISK_BYTES) as u64);
    let parent_attach_bytes = path_bytes(&parent_attach_path)?;
    let parent_outside_bytes = path_bytes(&parent_outside_path)?;
    let parent_interface_bytes = path_bytes(&parent_interface_path)?;
    let (attach_finalized_early, attach_propagated, outside_propagated, outside_structural_elided) =
        uf.counts();
    println!(
        "PROFILE_H2_HIER_STREAM_COMBINE z0={} z1={} pair_nodes={} pair_state_bytes={} parent_interface_nodes={} cross_retained={} cross_candidates={} parent_attach_bytes={} parent_outside_bytes={} parent_interface_bytes={} attach_finalized_early={} attach_propagated={} outside_propagated={} outside_structural_elided={}",
        left.z0,
        right.z1,
        total_nodes,
        state_bytes,
        parent_interface_nodes,
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
        parent_attach_bytes,
        parent_outside_bytes,
        parent_interface_bytes,
        attach_finalized_early,
        attach_propagated,
        outside_propagated,
        outside_structural_elided,
    );

    Ok((
        DiskHierarchicalH2Summary {
            z0: left.z0,
            z1: right.z1,
            width: left.width,
            height: left.height,
            interface_node_count: u32::try_from(parent_interface_nodes)
                .expect("hierarchical H2 parent interface count exceeds u32"),
            lower_values: left.lower_values,
            upper_values: right.upper_values,
            attach_path: parent_attach_path,
            outside_path: parent_outside_path,
            interface_path: parent_interface_path,
        },
        total_nodes,
        state_bytes,
        attach_finalized_early,
        attach_propagated,
        outside_propagated,
        outside_structural_elided,
    ))
}

#[allow(clippy::too_many_arguments)]
fn drain_hierarchical_h2_final_child_value<K: DiskScalarKey>(
    value: K,
    left_outside: &mut OutsideRunReader<K>,
    right_outside: &mut OutsideRunReader<K>,
    left_attach: &mut AttachRunReader<K>,
    right_attach: &mut AttachRunReader<K>,
    left_interface: &mut PairRunReader<K>,
    right_interface: &mut PairRunReader<K>,
    right_offset: u32,
    global_uf: &mut CompactGlobalBackgroundPersistenceUnionFind<K>,
    pair_writer: &mut BufWriter<File>,
) -> Result<()> {
    while let Some(node) = left_outside.pop_at_value(value)? {
        if let Some(pair) = global_uf.connect_outside(node, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    while let Some(node) = right_outside.pop_at_value(value)? {
        if let Some(pair) = global_uf.connect_outside(right_offset + node, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    while let Some((node, branch_birth)) = left_attach.pop_at_value(value)? {
        if let Some(pair) = global_uf.attach_branch(node, branch_birth, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    while let Some((node, branch_birth)) = right_attach.pop_at_value(value)? {
        if let Some(pair) = global_uf.attach_branch(right_offset + node, branch_birth, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    while let Some((a, b)) = left_interface.pop_at_value(value)? {
        if let Some(pair) = global_uf.union_with_persistence(a, b, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    while let Some((a, b)) = right_interface.pop_at_value(value)? {
        if let Some(pair) =
            global_uf.union_with_persistence(right_offset + a, right_offset + b, value)
            && pair.birth < pair.death
        {
            write_finite_pair(pair_writer, pair)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finalize_disk_h2_pair_direct_cross<K: DiskScalarKey>(
    left: DiskHierarchicalH2Summary<K>,
    right: DiskHierarchicalH2Summary<K>,
    background_connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    pair_writer: &mut BufWriter<File>,
) -> Result<(usize, u64)> {
    if left.z1 != right.z0 {
        bail!("disk-backed hierarchical H2 final summaries are not adjacent");
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H2 final summary face dimensions differ");
    }
    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if left.lower_values.is_empty() || right.lower_values.is_empty() {
        bail!("empty hierarchical H2 final boundary face");
    }
    let mut births = Vec::with_capacity(total_nodes);
    append_hierarchical_h2_births(&left, &mut births);
    let right_offset = births.len();
    append_hierarchical_h2_births(&right, &mut births);
    debug_assert_eq!(births.len(), total_nodes);
    let mut global_uf = CompactGlobalBackgroundPersistenceUnionFind::<K>::new(
        births,
        crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy::Packed,
    );
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    let state_bytes = parent_bytes + rank_bytes + birth_bytes;

    let mut left_outside = OutsideRunReader::<K>::open(&left.outside_path)?;
    let mut right_outside = OutsideRunReader::<K>::open(&right.outside_path)?;
    let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
    let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
    let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
    let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
    let offset =
        u32::try_from(right_offset).expect("hierarchical H2 final right offset exceeds u32");
    let mut previous_cross_value: Option<K> = None;

    let cross_stats = sparsify_scalar_superlevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        background_connectivity,
        interface_order,
        |edge| {
            if previous_cross_value.is_some_and(|previous| edge.value > previous) {
                bail!("direct final hierarchical H2 cross edges are not monotone");
            }
            previous_cross_value = Some(edge.value);

            loop {
                let Some(child_value) = next_hierarchical_h2_child_value(
                    &left_outside,
                    &right_outside,
                    &left_attach,
                    &right_attach,
                    &left_interface,
                    &right_interface,
                ) else {
                    break;
                };
                if child_value < edge.value {
                    break;
                }
                drain_hierarchical_h2_final_child_value(
                    child_value,
                    &mut left_outside,
                    &mut right_outside,
                    &mut left_attach,
                    &mut right_attach,
                    &mut left_interface,
                    &mut right_interface,
                    offset,
                    &mut global_uf,
                    pair_writer,
                )?;
                if child_value == edge.value {
                    break;
                }
            }

            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = offset
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H2 final face index exceeds u32");
            if let Some(pair) = global_uf.union_with_persistence(left_node, right_node, edge.value)
                && pair.birth < pair.death
            {
                write_finite_pair(pair_writer, pair)?;
            }
            Ok(())
        },
    )?;

    while let Some(value) = next_hierarchical_h2_child_value(
        &left_outside,
        &right_outside,
        &left_attach,
        &right_attach,
        &left_interface,
        &right_interface,
    ) {
        drain_hierarchical_h2_final_child_value(
            value,
            &mut left_outside,
            &mut right_outside,
            &mut left_attach,
            &mut right_attach,
            &mut left_interface,
            &mut right_interface,
            offset,
            &mut global_uf,
            pair_writer,
        )?;
    }

    let remaining = global_uf.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "direct-cross hierarchical H2 terminal-free final reduction ended with {} finite background components not connected to outside",
            remaining.len()
        );
    }
    for path in [
        &left.attach_path,
        &left.outside_path,
        &left.interface_path,
        &right.attach_path,
        &right.outside_path,
        &right.interface_path,
    ] {
        fs::remove_file(path).with_context(|| {
            format!("could not delete consumed final hierarchical H2 run {path:?}")
        })?;
    }
    let cross_run_bytes_avoided = cross_stats
        .retained_edges
        .saturating_mul((K::DISK_BYTES + 8) as u64);
    println!(
        "PROFILE_H2_HIER_STREAM_FINAL z0={} z1={} pair_nodes={} pair_state_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} cross_retained={} cross_candidates={} root_materialized=false cross_storage=direct cross_run_bytes=0 cross_run_bytes_avoided={}",
        left.z0,
        right.z1,
        total_nodes,
        state_bytes,
        parent_bytes,
        rank_bytes,
        birth_bytes,
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
        cross_run_bytes_avoided,
    );
    Ok((total_nodes, state_bytes))
}

#[allow(clippy::too_many_arguments)]
fn finalize_disk_h2_pair<K: DiskScalarKey>(
    left: DiskHierarchicalH2Summary<K>,
    right: DiskHierarchicalH2Summary<K>,
    background_connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
) -> Result<(usize, u64)> {
    if left.z1 != right.z0 {
        bail!("disk-backed hierarchical H2 final summaries are not adjacent");
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H2 final summary face dimensions differ");
    }
    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if left.lower_values.is_empty() || right.lower_values.is_empty() {
        bail!("empty hierarchical H2 final boundary face");
    }
    let mut births = Vec::with_capacity(total_nodes);
    append_hierarchical_h2_births(&left, &mut births);
    let right_offset = births.len();
    append_hierarchical_h2_births(&right, &mut births);
    debug_assert_eq!(births.len(), total_nodes);
    let mut global_uf = CompactGlobalBackgroundPersistenceUnionFind::<K>::new(
        births,
        crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy::Packed,
    );
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    let state_bytes = parent_bytes + rank_bytes + birth_bytes;

    let cross_path = directory.join(format!("h2_hier_final_cross_{summary_id:06}.bin"));
    let mut cross_writer = BufWriter::new(File::create(&cross_path)?);
    let cross_stats = sparsify_scalar_superlevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        background_connectivity,
        interface_order,
        |edge| {
            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = u32::try_from(right_offset)
                .expect("hierarchical H2 final right offset exceeds u32")
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H2 final face index exceeds u32");
            write_pair_event(
                &mut cross_writer,
                PairEvent {
                    value: edge.value,
                    a: left_node,
                    b: right_node,
                },
            )
        },
    )?;
    cross_writer.flush()?;
    drop(cross_writer);

    {
        let mut left_outside = OutsideRunReader::<K>::open(&left.outside_path)?;
        let mut right_outside = OutsideRunReader::<K>::open(&right.outside_path)?;
        let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
        let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
        let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
        let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
        let mut cross = PairRunReader::<K>::open(&cross_path)?;
        let offset =
            u32::try_from(right_offset).expect("hierarchical H2 final right offset exceeds u32");

        while let Some(value) = next_hierarchical_h2_value(
            &left_outside,
            &right_outside,
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
        ) {
            while let Some(node) = left_outside.pop_at_value(value)? {
                if let Some(pair) = global_uf.connect_outside(node, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some(node) = right_outside.pop_at_value(value)? {
                if let Some(pair) = global_uf.connect_outside(offset + node, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((node, branch_birth)) = left_attach.pop_at_value(value)? {
                if let Some(pair) = global_uf.attach_branch(node, branch_birth, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((node, branch_birth)) = right_attach.pop_at_value(value)? {
                if let Some(pair) = global_uf.attach_branch(offset + node, branch_birth, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = left_interface.pop_at_value(value)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = right_interface.pop_at_value(value)? {
                if let Some(pair) = global_uf.union_with_persistence(offset + a, offset + b, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = cross.pop_at_value(value)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, value)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
        }
    }

    let remaining = global_uf.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "hierarchical H2 terminal-free final reduction ended with {} finite background components not connected to outside",
            remaining.len()
        );
    }
    for path in [
        &left.attach_path,
        &left.outside_path,
        &left.interface_path,
        &right.attach_path,
        &right.outside_path,
        &right.interface_path,
        &cross_path,
    ] {
        fs::remove_file(path).with_context(|| {
            format!("could not delete consumed final hierarchical H2 run {path:?}")
        })?;
    }
    println!(
        "PROFILE_H2_HIER_STREAM_FINAL z0={} z1={} pair_nodes={} pair_state_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} cross_retained={} cross_candidates={} root_materialized=false",
        left.z0,
        right.z1,
        total_nodes,
        state_bytes,
        parent_bytes,
        rank_bytes,
        birth_bytes,
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
    );
    Ok((total_nodes, state_bytes))
}

#[allow(clippy::too_many_arguments)]
fn prepare_disk_h2_leaf_f32(
    volume: &ScalarTiffStackReader,
    slab_id: usize,
    z0: usize,
    z1: usize,
    background_connectivity: Connectivity,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
    tuning: ScalarStreamTuning,
) -> Result<(
    DiskHierarchicalH2Summary<F32Key>,
    H2SlabPreparationProfile,
    u64,
    u64,
    u64,
)> {
    let attach_path = directory.join(format!("h2_hier_attach_{summary_id:06}.bin"));
    let outside_path = directory.join(format!("h2_hier_outside_{summary_id:06}.bin"));
    let interface_path = directory.join(format!("h2_hier_interface_{summary_id:06}.bin"));
    let mut attach_writer = BufWriter::new(File::create(&attach_path)?);
    let mut outside_writer = BufWriter::new(File::create(&outside_path)?);
    let mut interface_writer = BufWriter::new(File::create(&interface_path)?);
    let mut previous_attach_value: Option<F32Key> = None;
    let mut previous_outside_value: Option<F32Key> = None;
    let mut previous_interface_value: Option<F32Key> = None;
    let mut final_pairs = 0u64;
    let mut attach_events = 0u64;
    let mut outside_events = 0u64;

    let (block, _read_profile) = volume.read_z_slab_f32_native_profiled(z0, z1)?;
    let (summary, slab_profile) = process_slab_h2_persistence_f32_native_direct_profiled(
        slab_id,
        z0,
        block,
        volume.width,
        volume.height,
        volume.depth,
        background_connectivity,
        tuning.neighbor_kernel,
        tuning.representative_active_check,
        tuning.union_kernel,
        tuning.neighbor_root_check,
        tuning.active_state,
        tuning.interface_state,
        tuning.uf_layout,
        tuning.local_h2_birth_state,
        tuning.sweep_diagnostics,
        true,
        matches!(
            tuning.h2_hier_outside_structural_pruning,
            H2HierOutsideStructuralPruningStrategy::OutsideDominated
        ),
        |event| {
            match event {
                H2LocalStreamEvent::FinalPair(pair) => {
                    if pair.birth < pair.death {
                        final_pairs += 1;
                        write_finite_pair(pair_writer, pair)?;
                    }
                }
                H2LocalStreamEvent::Attach(event) => {
                    if previous_attach_value.is_some_and(|value| event.value > value) {
                        bail!(
                            "hierarchical direct H2 attach events are not monotone within slab {slab_id}"
                        );
                    }
                    previous_attach_value = Some(event.value);
                    attach_events += 1;
                    write_attach_event(
                        &mut attach_writer,
                        AttachDiskEvent {
                            value: event.value,
                            node: event.interface_node,
                            branch_birth: event.branch_birth,
                        },
                    )?;
                }
                H2LocalStreamEvent::Outside(event) => {
                    if previous_outside_value.is_some_and(|value| event.value > value) {
                        bail!(
                            "hierarchical direct H2 outside events are not monotone within slab {slab_id}"
                        );
                    }
                    previous_outside_value = Some(event.value);
                    outside_events += 1;
                    write_outside_event(
                        &mut outside_writer,
                        OutsideDiskEvent {
                            value: event.value,
                            node: event.interface_node,
                        },
                    )?;
                }
                H2LocalStreamEvent::InterfaceMerge(event) => {
                    if previous_interface_value.is_some_and(|value| event.value > value) {
                        bail!(
                            "hierarchical direct H2 interface events are not monotone within slab {slab_id}"
                        );
                    }
                    previous_interface_value = Some(event.value);
                    write_pair_event(
                        &mut interface_writer,
                        PairEvent {
                            value: event.value,
                            a: event.a,
                            b: event.b,
                        },
                    )?;
                }
            }
            Ok(())
        },
    )?;
    attach_writer.flush()?;
    outside_writer.flush()?;
    interface_writer.flush()?;

    println!(
        "PROFILE_H2_HIER_STREAM_LEAF slab={} z0={} z1={} final_pairs={} attach_events={} outside_events={} attach_bytes={} outside_bytes={} interface_bytes={} local_birth_storage=reuse-input local_event_storage=direct local_attach_pruning=elder-dominated local_outside_structural_pruning={}",
        slab_id,
        z0,
        z1,
        final_pairs,
        attach_events,
        outside_events,
        path_bytes(&attach_path)?,
        path_bytes(&outside_path)?,
        path_bytes(&interface_path)?,
        tuning.h2_hier_outside_structural_pruning.as_str(),
    );

    Ok((
        DiskHierarchicalH2Summary {
            z0,
            z1,
            width: volume.width,
            height: volume.height,
            interface_node_count: summary.interface_node_count,
            lower_values: summary.z_min_face.values,
            upper_values: summary.z_max_face.values,
            attach_path,
            outside_path,
            interface_path,
        },
        slab_profile,
        final_pairs,
        attach_events,
        outside_events,
    ))
}

fn finalize_single_disk_h2_root<K: DiskScalarKey>(
    root: DiskHierarchicalH2Summary<K>,
    pair_writer: &mut BufWriter<File>,
) -> Result<(usize, u64)> {
    if root.lower_values.is_empty() {
        bail!("empty single-slab hierarchical H2 root");
    }
    let mut births = Vec::with_capacity(root.interface_node_count as usize);
    append_hierarchical_h2_births(&root, &mut births);
    debug_assert_eq!(births.len(), root.interface_node_count as usize);
    let mut global_uf = CompactGlobalBackgroundPersistenceUnionFind::<K>::new(
        births,
        crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy::Packed,
    );
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    let state_bytes = parent_bytes + rank_bytes + birth_bytes;
    let mut outside = OutsideRunReader::<K>::open(&root.outside_path)?;
    let mut attach = AttachRunReader::<K>::open(&root.attach_path)?;
    let mut interface = PairRunReader::<K>::open(&root.interface_path)?;
    loop {
        let Some(value) = [
            outside.next_value(),
            attach.next_value(),
            interface.next_value(),
        ]
        .into_iter()
        .flatten()
        .max() else {
            break;
        };
        while let Some(node) = outside.pop_at_value(value)? {
            if let Some(pair) = global_uf.connect_outside(node, value)
                && pair.birth < pair.death
            {
                write_finite_pair(pair_writer, pair)?;
            }
        }
        while let Some((node, branch_birth)) = attach.pop_at_value(value)? {
            if let Some(pair) = global_uf.attach_branch(node, branch_birth, value)
                && pair.birth < pair.death
            {
                write_finite_pair(pair_writer, pair)?;
            }
        }
        while let Some((a, b)) = interface.pop_at_value(value)? {
            if let Some(pair) = global_uf.union_with_persistence(a, b, value)
                && pair.birth < pair.death
            {
                write_finite_pair(pair_writer, pair)?;
            }
        }
    }
    let remaining = global_uf.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "single-slab hierarchical H2 root ended with {} finite background components not connected to outside",
            remaining.len()
        );
    }
    for path in [&root.attach_path, &root.outside_path, &root.interface_path] {
        fs::remove_file(path)?;
    }
    Ok((root.interface_node_count as usize, state_bytes))
}

pub fn compute_h2_persistence_scalar_hierarchical_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
    output_path: &Path,
    tuning: ScalarStreamTuning,
) -> Result<ScalarH2PersistenceStats> {
    let start = Instant::now();
    if volume.pixel_type != ScalarPixelType::F32 || tuning.f32_key_mode != F32KeyMode::Native32 {
        bail!(
            "h2-scalar-hierarchical-stream currently requires an F32 TIFF stack with --f32-key-mode native32"
        );
    }
    if !matches!(
        tuning.local_h2_birth_state,
        crate::scalar_stream_tuning::LocalH2BirthStateStrategy::Compact
    ) {
        bail!("h2-scalar-hierarchical-stream requires --local-h2-birth-state compact");
    }
    if !matches!(
        tuning.global_h2_birth_state,
        GlobalH2BirthStateStrategy::Compact
    ) {
        bail!("h2-scalar-hierarchical-stream requires --global-h2-birth-state compact");
    }
    if !matches!(
        tuning.global_h2_uf_layout,
        crate::scalar_stream_tuning::GlobalH2UnionFindLayoutStrategy::Packed
    ) {
        bail!("h2-scalar-hierarchical-stream requires --global-h2-uf-layout packed");
    }

    let temp_directory = TempRunDirectory::create("h2_scalar_hierarchical_stream_runs")?;
    println!(
        "Scalar hierarchical H2 streaming temporary directory: {:?}",
        temp_directory.path()
    );
    let pairs_path = temp_directory.path().join("h2_hier_finalized_pairs.bin");
    let mut pair_writer = BufWriter::new(File::create(&pairs_path)?);

    let mut ranges = Vec::new();
    let mut z0 = 0usize;
    let mut slab_id = 0usize;
    while z0 < volume.depth {
        let z1 = z0.saturating_add(slab_depth).min(volume.depth);
        ranges.push((slab_id, z0, z1));
        slab_id += 1;
        z0 = z1;
    }
    let face_size = volume
        .width
        .checked_mul(volume.height)
        .expect("hierarchical H2 face size overflow");
    let planned_pair_nodes = if ranges.len() > 1 {
        face_size.saturating_mul(4)
    } else if volume.depth == 1 {
        face_size
    } else {
        face_size.saturating_mul(2)
    };
    let planned_pair_state_bytes = u64::try_from(planned_pair_nodes)
        .expect("hierarchical H2 planned pair nodes exceed u64")
        .saturating_mul(8);
    println!(
        "Scalar hierarchical H2 storage: pipeline=native32-direct-{}-fanin-outside disk_key_bytes=4 pair_state_bytes_per_node=8 planned_max_pair_nodes={} planned_max_pair_state_bytes={}",
        tuning.h2_hier_cross_storage.as_str(),
        planned_pair_nodes,
        planned_pair_state_bytes,
    );
    println!(
        "PROFILE_CONFIG scalar_h2_hierarchical_stream f32_key_mode={} interface_order={} event_order={} neighbor_kernel={} representative_active_check={} union_kernel={} neighbor_root_check={} active_state={} interface_state={} uf_layout={} local_h2_birth_state={} global_h2_birth_state={} global_h2_uf_layout={} local_birth_storage=reuse-input local_event_storage=direct local_attach_pruning=elder-dominated hierarchy_attach_pruning=elder-dominated outside_state=distinguished h2_hier_cross_storage={} h2_hier_outside_structural_pruning={}",
        tuning.f32_key_mode.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.neighbor_root_check.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.local_h2_birth_state.as_str(),
        tuning.global_h2_birth_state.as_str(),
        tuning.global_h2_uf_layout.as_str(),
        tuning.h2_hier_cross_storage.as_str(),
        tuning.h2_hier_outside_structural_pruning.as_str(),
    );

    let mut slots: Vec<Option<DiskHierarchicalH2Summary<F32Key>>> = Vec::new();
    let mut next_summary_id = ranges.len();
    let mut combines = 0usize;
    let mut max_live_summaries = 0usize;
    let mut max_pair_nodes = 0usize;
    let mut max_pair_state_bytes = 0u64;
    let mut attach_finalized_early_total = 0u64;
    let mut attach_propagated_total = 0u64;
    let mut outside_propagated_total = 0u64;
    let mut outside_structural_elided_total = 0u64;
    let mut leaf_attach_events_total = 0u64;
    let mut leaf_outside_events_total = 0u64;
    let mut final_children: Option<(
        DiskHierarchicalH2Summary<F32Key>,
        DiskHierarchicalH2Summary<F32Key>,
    )> = None;

    'leaf_loop: for &(leaf_id, leaf_z0, leaf_z1) in &ranges {
        println!("Preparing optimized hierarchical H2 leaf {leaf_id}: z={leaf_z0}..{leaf_z1}");
        let (mut current, _profile, _leaf_pairs, leaf_attach, leaf_outside) =
            prepare_disk_h2_leaf_f32(
                volume,
                leaf_id,
                leaf_z0,
                leaf_z1,
                background_connectivity,
                temp_directory.path(),
                leaf_id,
                &mut pair_writer,
                tuning,
            )?;
        leaf_attach_events_total += leaf_attach;
        leaf_outside_events_total += leaf_outside;
        let mut level = 0usize;
        loop {
            if level == slots.len() {
                slots.push(Some(current));
                break;
            }
            if let Some(left) = slots[level].take() {
                let covers_entire_volume = ranges.len() > 1
                    && leaf_id + 1 == ranges.len()
                    && left.z0 == 0
                    && current.z1 == volume.depth;
                if covers_entire_volume {
                    final_children = Some((left, current));
                    break 'leaf_loop;
                }
                println!(
                    "Optimized hierarchical H2 fan-in level {level}: z={}..{} + z={}..{}",
                    left.z0, left.z1, current.z0, current.z1,
                );
                let (
                    parent,
                    pair_nodes,
                    pair_state_bytes,
                    finalized,
                    propagated,
                    outside_propagated,
                    outside_structural_elided,
                ) = match tuning.h2_hier_cross_storage {
                    H2HierCrossStorageStrategy::Disk => combine_disk_h2_summaries(
                        left,
                        current,
                        background_connectivity,
                        tuning.interface_order,
                        temp_directory.path(),
                        next_summary_id,
                        &mut pair_writer,
                        matches!(
                            tuning.h2_hier_outside_structural_pruning,
                            H2HierOutsideStructuralPruningStrategy::OutsideDominated
                        ),
                    )?,
                    H2HierCrossStorageStrategy::Direct => combine_disk_h2_summaries_direct_cross(
                        left,
                        current,
                        background_connectivity,
                        tuning.interface_order,
                        temp_directory.path(),
                        next_summary_id,
                        &mut pair_writer,
                        matches!(
                            tuning.h2_hier_outside_structural_pruning,
                            H2HierOutsideStructuralPruningStrategy::OutsideDominated
                        ),
                    )?,
                };
                next_summary_id += 1;
                combines += 1;
                max_pair_nodes = max_pair_nodes.max(pair_nodes);
                max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
                attach_finalized_early_total += finalized;
                attach_propagated_total += propagated;
                outside_propagated_total += outside_propagated;
                outside_structural_elided_total += outside_structural_elided;
                current = parent;
                level += 1;
            } else {
                slots[level] = Some(current);
                break;
            }
        }
        max_live_summaries =
            max_live_summaries.max(slots.iter().filter(|slot| slot.is_some()).count());
    }

    let mut root_fallback: Option<DiskHierarchicalH2Summary<F32Key>> = None;
    if final_children.is_none() {
        let mut remaining: Vec<_> = slots.into_iter().flatten().collect();
        if remaining.is_empty() {
            bail!("hierarchical H2 stream received an empty volume");
        }
        remaining.sort_by_key(|summary| summary.z0);
        if remaining.len() == 1 {
            root_fallback = remaining.pop();
        } else {
            let mut left = remaining.remove(0);
            while remaining.len() > 1 {
                let right = remaining.remove(0);
                let (
                    parent,
                    pair_nodes,
                    pair_state_bytes,
                    finalized,
                    propagated,
                    outside_propagated,
                    outside_structural_elided,
                ) = match tuning.h2_hier_cross_storage {
                    H2HierCrossStorageStrategy::Disk => combine_disk_h2_summaries(
                        left,
                        right,
                        background_connectivity,
                        tuning.interface_order,
                        temp_directory.path(),
                        next_summary_id,
                        &mut pair_writer,
                        matches!(
                            tuning.h2_hier_outside_structural_pruning,
                            H2HierOutsideStructuralPruningStrategy::OutsideDominated
                        ),
                    )?,
                    H2HierCrossStorageStrategy::Direct => combine_disk_h2_summaries_direct_cross(
                        left,
                        right,
                        background_connectivity,
                        tuning.interface_order,
                        temp_directory.path(),
                        next_summary_id,
                        &mut pair_writer,
                        matches!(
                            tuning.h2_hier_outside_structural_pruning,
                            H2HierOutsideStructuralPruningStrategy::OutsideDominated
                        ),
                    )?,
                };
                next_summary_id += 1;
                combines += 1;
                max_pair_nodes = max_pair_nodes.max(pair_nodes);
                max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
                attach_finalized_early_total += finalized;
                attach_propagated_total += propagated;
                outside_propagated_total += outside_propagated;
                outside_structural_elided_total += outside_structural_elided;
                left = parent;
            }
            let right = remaining
                .pop()
                .expect("hierarchical H2 final child unexpectedly missing");
            final_children = Some((left, right));
        }
    }

    if let Some((left, right)) = final_children {
        println!(
            "Optimized hierarchical H2 terminal-free final fan-in: z={}..{} + z={}..{}",
            left.z0, left.z1, right.z0, right.z1,
        );
        let (pair_nodes, pair_state_bytes) = match tuning.h2_hier_cross_storage {
            H2HierCrossStorageStrategy::Disk => finalize_disk_h2_pair(
                left,
                right,
                background_connectivity,
                tuning.interface_order,
                temp_directory.path(),
                next_summary_id,
                &mut pair_writer,
            )?,
            H2HierCrossStorageStrategy::Direct => finalize_disk_h2_pair_direct_cross(
                left,
                right,
                background_connectivity,
                tuning.interface_order,
                &mut pair_writer,
            )?,
        };
        combines += 1;
        max_pair_nodes = max_pair_nodes.max(pair_nodes);
        max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
    } else if let Some(root) = root_fallback {
        let (pair_nodes, pair_state_bytes) = finalize_single_disk_h2_root(root, &mut pair_writer)?;
        max_pair_nodes = max_pair_nodes.max(pair_nodes);
        max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
    }

    pair_writer.flush()?;
    drop(pair_writer);
    let finalized_pair_bytes = path_bytes(&pairs_path)?;

    let mut output = AtomicOutput::create(output_path)?;
    writeln!(output, "birth,death")?;
    let mut finite_intervals = 0u64;
    let mut reader = BufReader::new(File::open(&pairs_path)?);
    while let Some(pair) = read_finite_pair::<F32Key>(&mut reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }
    output.commit()?;

    println!(
        "PROFILE_H2_HIER_STREAM leaf_slabs={} combines={} max_live_summaries={} max_pair_nodes={} max_pair_state_bytes={} final_interface_nodes=0 finalized_pair_bytes={} root_attach_bytes=0 root_outside_bytes=0 root_interface_bytes=0 root_materialized=false disk_key_bytes=4 local_birth_storage=reuse-input local_event_storage=direct global_h2_birth_state=compact global_h2_uf_layout=packed leaf_attach_events={} leaf_outside_events={} attach_finalized_early={} attach_propagated={} outside_propagated={} outside_structural_elided={} attach_pruning=elder-dominated h2_hier_cross_storage={} h2_hier_outside_structural_pruning={}",
        ranges.len(),
        combines,
        max_live_summaries,
        max_pair_nodes,
        max_pair_state_bytes,
        finalized_pair_bytes,
        leaf_attach_events_total,
        leaf_outside_events_total,
        attach_finalized_early_total,
        attach_propagated_total,
        outside_propagated_total,
        outside_structural_elided_total,
        tuning.h2_hier_cross_storage.as_str(),
        tuning.h2_hier_outside_structural_pruning.as_str(),
    );

    temp_directory.close()?;
    println!(
        "Hierarchical scalar H2 streaming computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(ScalarH2PersistenceStats { finite_intervals })
}

pub fn compute_h2_scalar_stream_batch(
    root: &Path,
    output_root: &Path,
    slab_depth: usize,
    background_connectivity: Connectivity,
    tuning: ScalarStreamTuning,
) -> Result<usize> {
    let directories = find_tiff_stack_directories(root)?;

    if directories.is_empty() {
        bail!("no TIFF-stack directories found at or below {root:?}");
    }

    fs::create_dir_all(output_root)?;

    for (index, directory) in directories.iter().enumerate() {
        let relative = directory.strip_prefix(root).with_context(|| {
            format!("dataset directory {directory:?} is not underneath batch root {root:?}")
        })?;
        let dataset_output = output_root.join(relative);
        fs::create_dir_all(&dataset_output)?;

        println!();
        println!(
            "=== Streaming scalar H2 dataset {} of {}: {} ===",
            index + 1,
            directories.len(),
            directory.display()
        );
        let volume = ScalarTiffStackReader::open(directory)?;
        print_scalar_volume_info(&volume);
        let output = dataset_output.join("h2_persistence_scalar_stream.csv");
        compute_h2_persistence_scalar_stream_zslabs(
            &volume,
            slab_depth,
            background_connectivity,
            &output,
            tuning,
        )?;
    }

    Ok(directories.len())
}
