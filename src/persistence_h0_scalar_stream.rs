use anyhow::{Context, Result, bail};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::binary_io::read_exact_or_eof;
use crate::connectivity::Connectivity;
use crate::interface_sparsify_scalar::sparsify_scalar_sublevel_interface_with_order;
use crate::io_scalar::{ScalarPixelType, ScalarTiffStackReader, print_scalar_volume_info};
use crate::persistence_h0_scalar::{
    AttachEvent, FinitePair, GlobalScalarPersistenceUnionFind, H0LocalStreamEvent,
    H0SlabPreparationProfile, InterfaceMergeEvent, SlabH0Summary,
    process_slab_h0_persistence_f32_native_direct_profiled,
    process_slab_h0_persistence_f32_native_owned_profiled,
    process_slab_h0_persistence_f32_native_profiled, process_slab_h0_persistence_scalar_profiled,
};
use crate::scalar::{F32Key, LocalScalarKey, ScalarKey};
use crate::scalar_order::RadixScalarKey;
use crate::scalar_stream_tuning::{
    EventOrderStrategy, F32KeyMode, GlobalH0UnionFindLayoutStrategy, H0BirthBufferStrategy,
    H0EventStorageStrategy, H0HierAttachPruningStrategy, InterfaceOrderStrategy, MergeStrategy,
    ScalarStreamTuning,
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

#[derive(Debug, Clone)]
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

fn write_scalar_key<K: DiskScalarKey>(writer: &mut BufWriter<File>, value: K) -> Result<()> {
    value.write_disk(writer)
}

fn read_scalar_key<K: DiskScalarKey>(reader: &mut BufReader<File>) -> Result<Option<K>> {
    K::read_disk(reader)
}

fn write_pair_event<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    event: PairEvent<K>,
) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.a.to_le_bytes())?;
    writer.write_all(&event.b.to_le_bytes())?;
    Ok(())
}

fn read_pair_event<K: DiskScalarKey>(reader: &mut BufReader<File>) -> Result<Option<PairEvent<K>>> {
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

fn write_attach_event<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    event: AttachDiskEvent<K>,
) -> Result<()> {
    write_scalar_key(writer, event.value)?;
    writer.write_all(&event.node.to_le_bytes())?;
    write_scalar_key(writer, event.branch_birth)?;
    Ok(())
}

fn read_attach_event<K: DiskScalarKey>(
    reader: &mut BufReader<File>,
) -> Result<Option<AttachDiskEvent<K>>> {
    let Some(value) = read_scalar_key(reader)? else {
        return Ok(None);
    };

    let mut node_bytes = [0u8; 4];
    reader.read_exact(&mut node_bytes)?;
    let branch_birth = read_scalar_key(reader)?
        .ok_or_else(|| anyhow::anyhow!("truncated scalar H0 attach event after node field"))?;

    Ok(Some(AttachDiskEvent {
        value,
        node: u32::from_le_bytes(node_bytes),
        branch_birth,
    }))
}

fn write_finite_pair<K: DiskScalarKey>(
    writer: &mut BufWriter<File>,
    pair: FinitePair<K>,
) -> Result<()> {
    write_scalar_key(writer, pair.birth)?;
    write_scalar_key(writer, pair.death)?;
    Ok(())
}

fn read_finite_pair<K: DiskScalarKey>(
    reader: &mut BufReader<File>,
) -> Result<Option<FinitePair<K>>> {
    let Some(birth) = read_scalar_key(reader)? else {
        return Ok(None);
    };
    let death = read_scalar_key(reader)?
        .ok_or_else(|| anyhow::anyhow!("truncated scalar H0 finite pair after birth field"))?;

    Ok(Some(FinitePair { birth, death }))
}

struct PairRunReader<K: DiskScalarKey> {
    reader: BufReader<File>,
    next: Option<PairEvent<K>>,
}

impl<K: DiskScalarKey> PairRunReader<K> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_pair_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn peek_value(&self) -> Option<K> {
        self.next.map(|event| event.value)
    }

    fn pop_at_value(&mut self, value: K) -> Result<Option<(u32, u32)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_pair_event(&mut self.reader)?;
                Ok(Some((event.a, event.b)))
            }
            _ => Ok(None),
        }
    }
}

struct AttachRunReader<K: DiskScalarKey> {
    reader: BufReader<File>,
    next: Option<AttachDiskEvent<K>>,
}

impl<K: DiskScalarKey> AttachRunReader<K> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let next = read_attach_event(&mut reader)?;
        Ok(Self { reader, next })
    }

    fn peek_value(&self) -> Option<K> {
        self.next.map(|event| event.value)
    }

    fn pop_at_value(&mut self, value: K) -> Result<Option<(u32, K)>> {
        match self.next {
            Some(event) if event.value == value => {
                self.next = read_attach_event(&mut self.reader)?;
                Ok(Some((event.node, event.branch_birth)))
            }
            _ => Ok(None),
        }
    }
}

fn write_interface_merge_run<K: DiskScalarKey>(
    directory: &Path,
    desc: &SlabDescriptor,
    events: &[InterfaceMergeEvent<K>],
    event_order: EventOrderStrategy,
) -> Result<PathBuf> {
    let path = directory.join(format!("h0_scalar_interface_{:05}.bin", desc.slab_id));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    match event_order {
        EventOrderStrategy::Verify => {
            let mut previous = None;
            for &event in events {
                if previous.is_some_and(|value| event.value < value) {
                    bail!(
                        "scalar H0 interface events are not monotone within slab {}",
                        desc.slab_id
                    );
                }
                previous = Some(event.value);
                write_pair_event(
                    &mut writer,
                    PairEvent {
                        value: event.value,
                        a: desc.interface_base + event.a,
                        b: desc.interface_base + event.b,
                    },
                )?;
            }
        }
        EventOrderStrategy::Resort => {
            let mut sorted_events = events.to_vec();
            sorted_events.sort_unstable_by_key(|event| event.value);
            for event in sorted_events {
                write_pair_event(
                    &mut writer,
                    PairEvent {
                        value: event.value,
                        a: desc.interface_base + event.a,
                        b: desc.interface_base + event.b,
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
    desc: &SlabDescriptor,
    events: &[AttachEvent<K>],
    event_order: EventOrderStrategy,
) -> Result<PathBuf> {
    let path = directory.join(format!("h0_scalar_attach_{:05}.bin", desc.slab_id));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    match event_order {
        EventOrderStrategy::Verify => {
            let mut previous = None;
            for &event in events {
                if previous.is_some_and(|value| event.value < value) {
                    bail!(
                        "scalar H0 attach events are not monotone within slab {}",
                        desc.slab_id
                    );
                }
                previous = Some(event.value);
                write_attach_event(
                    &mut writer,
                    AttachDiskEvent {
                        value: event.value,
                        node: desc.interface_base + event.interface_node,
                        branch_birth: event.branch_birth,
                    },
                )?;
            }
        }
        EventOrderStrategy::Resort => {
            let mut sorted_events = events.to_vec();
            sorted_events.sort_unstable_by_key(|event| event.value);
            for event in sorted_events {
                write_attach_event(
                    &mut writer,
                    AttachDiskEvent {
                        value: event.value,
                        node: desc.interface_base + event.interface_node,
                        branch_birth: event.branch_birth,
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
    right_desc: &SlabDescriptor,
    right_lower_values: &[K],
    config: CrossRunConfig,
) -> Result<PathBuf> {
    let face_size = config
        .width
        .checked_mul(config.height)
        .expect("face size overflow");
    assert_eq!(left.values.len(), face_size);
    assert_eq!(right_lower_values.len(), face_size);

    let path = directory.join(format!("h0_scalar_cross_{pair_id:05}.bin"));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    sparsify_scalar_sublevel_interface_with_order(
        &left.values,
        right_lower_values,
        config.width,
        config.height,
        config.connectivity,
        config.interface_order,
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
    pruning_cache_lookups: u64,
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
    interface_rep_queries: u64,
    interface_rep_writes: u64,
    interface_forced_root_unions: u64,
    interface_interface_unions: u64,
    interface_state_bytes: u64,
    max_rank_observed: u64,
    uf_parent_state_bytes: u64,
    uf_rank_state_bytes: u64,
}

struct PreparedH0Runs {
    total_interface_nodes: usize,
    births_path: PathBuf,
    local_pairs_path: PathBuf,
    attach_run_paths: Vec<PathBuf>,
    interface_run_paths: Vec<PathBuf>,
    cross_run_paths: Vec<PathBuf>,
    preparation_profile: DetailedPreparationProfile,
    disk_key_bytes: usize,
    interface_birth_bytes: u64,
    local_pair_bytes: u64,
    attach_bytes: u64,
    interface_bytes: u64,
    cross_bytes: u64,
    total_run_bytes: u64,
}

fn path_len(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}

fn sum_path_lengths(paths: &[PathBuf]) -> Result<u64> {
    paths.iter().try_fold(0u64, |total, path| {
        total
            .checked_add(path_len(path)?)
            .ok_or_else(|| anyhow::anyhow!("temporary H0 run byte count overflow"))
    })
}

type PreparedH0Slab<K> = (SlabH0Summary<K>, H0SlabPreparationProfile, f64, f64, f64);

#[allow(clippy::too_many_arguments)]
fn prepare_h0_runs_with<K, F>(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
    mut prepare_slab: F,
) -> Result<PreparedH0Runs>
where
    K: DiskScalarKey,
    F: FnMut(&SlabDescriptor) -> Result<PreparedH0Slab<K>>,
{
    std::fs::create_dir_all(temp_directory)?;

    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);

    let births_path = temp_directory.join("h0_scalar_interface_births.bin");
    let local_pairs_path = temp_directory.join("h0_scalar_local_pairs.bin");
    let mut birth_writer = BufWriter::new(File::create(&births_path)?);
    let mut local_pair_writer = BufWriter::new(File::create(&local_pairs_path)?);

    let mut attach_run_paths = Vec::new();
    let mut interface_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace<K>> = None;
    let mut preparation_profile = DetailedPreparationProfile::default();
    let cross_run_config = CrossRunConfig {
        width: volume.width,
        height: volume.height,
        connectivity,
        interface_order: tuning.interface_order,
    };

    for desc in &descriptors {
        println!(
            "Preparing scalar H0-persistence slab {}: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );

        let (summary, slab_profile, decode_seconds, key_conversion_seconds, slab_copy_seconds) =
            prepare_slab(desc)?;
        preparation_profile.decode_seconds += decode_seconds;
        preparation_profile.key_conversion_seconds += key_conversion_seconds;
        preparation_profile.slab_copy_seconds += slab_copy_seconds;
        preparation_profile.scalar_order_seconds += slab_profile.scalar_order_seconds;
        preparation_profile.local_sweep_seconds += slab_profile.local_sweep_seconds;
        preparation_profile.total_voxels += slab_profile.total_voxels;
        preparation_profile.interior_fast_voxels += slab_profile.interior_fast_voxels;
        preparation_profile.pruning_mask_calls += slab_profile.pruning_mask_calls;
        preparation_profile.active_state_checks += slab_profile.active_state_checks;
        preparation_profile.active_neighbor_hits += slab_profile.active_neighbor_hits;
        preparation_profile.representative_visits += slab_profile.representative_visits;
        preparation_profile.pruning_cache_lookups += slab_profile.pruning_cache_lookups;
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

        assert_eq!(summary.interface_node_count, desc.interface_count);
        assert_eq!(summary.z_min_face.width, volume.width);
        assert_eq!(summary.z_min_face.height, volume.height);
        assert_eq!(summary.z_max_face.width, volume.width);
        assert_eq!(summary.z_max_face.height, volume.height);

        let local_write_start = Instant::now();
        for &pair in &summary.finalized_pairs {
            write_finite_pair(&mut local_pair_writer, pair)?;
        }

        for &value in &summary.z_min_face.values {
            write_scalar_key(&mut birth_writer, value)?;
        }
        if desc.local_depth > 1 {
            for &value in &summary.z_max_face.values {
                write_scalar_key(&mut birth_writer, value)?;
            }
        }

        attach_run_paths.push(write_attach_run(
            temp_directory,
            desc,
            &summary.attach_events,
            tuning.event_order,
        )?);
        interface_run_paths.push(write_interface_merge_run(
            temp_directory,
            desc,
            &summary.interface_merge_events,
            tuning.event_order,
        )?);
        preparation_profile.local_run_write_seconds += local_write_start.elapsed().as_secs_f64();

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;

        if let Some(previous) = previous_upper_face.take() {
            let cross_start = Instant::now();
            cross_run_paths.push(write_cross_run(
                temp_directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &lower_values,
                cross_run_config,
            )?);
            preparation_profile.cross_interface_seconds += cross_start.elapsed().as_secs_f64();
        }

        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: upper_values,
        });
    }

    let final_flush_start = Instant::now();
    birth_writer.flush()?;
    local_pair_writer.flush()?;
    preparation_profile.local_run_write_seconds += final_flush_start.elapsed().as_secs_f64();

    let interface_birth_bytes = path_len(&births_path)?;
    let local_pair_bytes = path_len(&local_pairs_path)?;
    let attach_bytes = sum_path_lengths(&attach_run_paths)?;
    let interface_bytes = sum_path_lengths(&interface_run_paths)?;
    let cross_bytes = sum_path_lengths(&cross_run_paths)?;
    let total_run_bytes = interface_birth_bytes
        .checked_add(local_pair_bytes)
        .and_then(|v| v.checked_add(attach_bytes))
        .and_then(|v| v.checked_add(interface_bytes))
        .and_then(|v| v.checked_add(cross_bytes))
        .ok_or_else(|| anyhow::anyhow!("temporary H0 run byte count overflow"))?;

    Ok(PreparedH0Runs {
        total_interface_nodes,
        births_path,
        local_pairs_path,
        attach_run_paths,
        interface_run_paths,
        cross_run_paths,
        preparation_profile,
        disk_key_bytes: K::DISK_BYTES,
        interface_birth_bytes,
        local_pair_bytes,
        attach_bytes,
        interface_bytes,
        cross_bytes,
        total_run_bytes,
    })
}

fn prepare_h0_runs_scalar(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
) -> Result<PreparedH0Runs> {
    prepare_h0_runs_with::<ScalarKey, _>(
        volume,
        slab_depth,
        connectivity,
        temp_directory,
        tuning,
        |desc| {
            let (block, read_profile) = volume.read_z_slab_profiled(desc.z0, desc.z1)?;
            let (summary, slab_profile) = process_slab_h0_persistence_scalar_profiled(
                desc.slab_id,
                &block,
                connectivity,
                tuning.neighbor_kernel,
                tuning.representative_active_check,
                tuning.union_kernel,
                tuning.h0_pruning_cache,
                tuning.active_state,
                tuning.interface_state,
                tuning.uf_layout,
                tuning.sweep_diagnostics,
            );
            Ok((
                summary,
                slab_profile,
                read_profile.decode_seconds,
                read_profile.key_conversion_seconds,
                read_profile.slab_copy_seconds,
            ))
        },
    )
}

fn prepare_h0_runs_f32_native(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
) -> Result<PreparedH0Runs> {
    prepare_h0_runs_with::<F32Key, _>(
        volume,
        slab_depth,
        connectivity,
        temp_directory,
        tuning,
        |desc| {
            let (block, read_profile) = volume.read_z_slab_f32_native_profiled(desc.z0, desc.z1)?;
            let (summary, slab_profile) = match tuning.h0_birth_buffer {
                H0BirthBufferStrategy::Copy => process_slab_h0_persistence_f32_native_profiled(
                    desc.slab_id,
                    &block,
                    connectivity,
                    tuning.neighbor_kernel,
                    tuning.representative_active_check,
                    tuning.union_kernel,
                    tuning.h0_pruning_cache,
                    tuning.active_state,
                    tuning.interface_state,
                    tuning.uf_layout,
                    tuning.sweep_diagnostics,
                ),
                H0BirthBufferStrategy::ReuseInput => {
                    process_slab_h0_persistence_f32_native_owned_profiled(
                        desc.slab_id,
                        block,
                        connectivity,
                        tuning.neighbor_kernel,
                        tuning.representative_active_check,
                        tuning.union_kernel,
                        tuning.h0_pruning_cache,
                        tuning.active_state,
                        tuning.interface_state,
                        tuning.uf_layout,
                        tuning.sweep_diagnostics,
                    )
                }
            };
            Ok((
                summary,
                slab_profile,
                read_profile.decode_seconds,
                read_profile.key_conversion_seconds,
                read_profile.slab_copy_seconds,
            ))
        },
    )
}

fn prepare_h0_runs_f32_native_direct(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    temp_directory: &Path,
    tuning: ScalarStreamTuning,
) -> Result<PreparedH0Runs> {
    if !matches!(tuning.event_order, EventOrderStrategy::Verify) {
        bail!("--h0-event-storage direct requires --event-order verify");
    }
    if !matches!(tuning.h0_birth_buffer, H0BirthBufferStrategy::ReuseInput) {
        bail!("--h0-event-storage direct currently requires --h0-birth-buffer reuse-input");
    }

    fs::create_dir_all(temp_directory)?;
    let descriptors = make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let total_interface_nodes = descriptors
        .last()
        .map(|desc| desc.interface_base as usize + desc.interface_count as usize)
        .unwrap_or(0);
    let births_path = temp_directory.join("h0_scalar_interface_births.bin");
    let local_pairs_path = temp_directory.join("h0_scalar_local_pairs.bin");
    let mut birth_writer = BufWriter::new(File::create(&births_path)?);
    let mut local_pair_writer = BufWriter::new(File::create(&local_pairs_path)?);
    let mut attach_run_paths = Vec::new();
    let mut interface_run_paths = Vec::new();
    let mut cross_run_paths = Vec::new();
    let mut previous_upper_face: Option<StoredUpperFace<F32Key>> = None;
    let mut preparation_profile = DetailedPreparationProfile::default();
    let cross_run_config = CrossRunConfig {
        width: volume.width,
        height: volume.height,
        connectivity,
        interface_order: tuning.interface_order,
    };

    for desc in &descriptors {
        println!(
            "Preparing scalar H0-persistence slab {} with direct event sinks: z={}..{}",
            desc.slab_id, desc.z0, desc.z1
        );
        let attach_path = temp_directory.join(format!("h0_scalar_attach_{:05}.bin", desc.slab_id));
        let interface_path =
            temp_directory.join(format!("h0_scalar_interface_{:05}.bin", desc.slab_id));
        let mut attach_writer = BufWriter::new(File::create(&attach_path)?);
        let mut interface_writer = BufWriter::new(File::create(&interface_path)?);
        let mut previous_attach_value: Option<F32Key> = None;
        let mut previous_interface_value: Option<F32Key> = None;

        let (block, read_profile) = volume.read_z_slab_f32_native_profiled(desc.z0, desc.z1)?;
        let (summary, slab_profile) = process_slab_h0_persistence_f32_native_direct_profiled(
            desc.slab_id,
            block,
            connectivity,
            tuning.neighbor_kernel,
            tuning.representative_active_check,
            tuning.union_kernel,
            tuning.h0_pruning_cache,
            tuning.active_state,
            tuning.interface_state,
            tuning.uf_layout,
            tuning.sweep_diagnostics,
            |event| {
                match event {
                    H0LocalStreamEvent::FinalPair(pair) => {
                        write_finite_pair(&mut local_pair_writer, pair)?;
                    }
                    H0LocalStreamEvent::Attach(event) => {
                        if previous_attach_value.is_some_and(|value| event.value < value) {
                            bail!(
                                "direct scalar H0 attach events are not monotone within slab {}",
                                desc.slab_id
                            );
                        }
                        previous_attach_value = Some(event.value);
                        write_attach_event(
                            &mut attach_writer,
                            AttachDiskEvent {
                                value: event.value,
                                node: desc.interface_base + event.interface_node,
                                branch_birth: event.branch_birth,
                            },
                        )?;
                    }
                    H0LocalStreamEvent::InterfaceMerge(event) => {
                        if previous_interface_value.is_some_and(|value| event.value < value) {
                            bail!(
                                "direct scalar H0 interface events are not monotone within slab {}",
                                desc.slab_id
                            );
                        }
                        previous_interface_value = Some(event.value);
                        write_pair_event(
                            &mut interface_writer,
                            PairEvent {
                                value: event.value,
                                a: desc.interface_base + event.a,
                                b: desc.interface_base + event.b,
                            },
                        )?;
                    }
                }
                Ok(())
            },
        )?;

        preparation_profile.decode_seconds += read_profile.decode_seconds;
        preparation_profile.key_conversion_seconds += read_profile.key_conversion_seconds;
        preparation_profile.slab_copy_seconds += read_profile.slab_copy_seconds;
        preparation_profile.scalar_order_seconds += slab_profile.scalar_order_seconds;
        preparation_profile.local_sweep_seconds += slab_profile.local_sweep_seconds;
        preparation_profile.total_voxels += slab_profile.total_voxels;
        preparation_profile.interior_fast_voxels += slab_profile.interior_fast_voxels;
        preparation_profile.pruning_mask_calls += slab_profile.pruning_mask_calls;
        preparation_profile.active_state_checks += slab_profile.active_state_checks;
        preparation_profile.active_neighbor_hits += slab_profile.active_neighbor_hits;
        preparation_profile.representative_visits += slab_profile.representative_visits;
        preparation_profile.pruning_cache_lookups += slab_profile.pruning_cache_lookups;
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

        assert_eq!(summary.interface_node_count, desc.interface_count);
        let write_start = Instant::now();
        for &value in &summary.z_min_face.values {
            write_scalar_key(&mut birth_writer, value)?;
        }
        if desc.local_depth > 1 {
            for &value in &summary.z_max_face.values {
                write_scalar_key(&mut birth_writer, value)?;
            }
        }
        attach_writer.flush()?;
        interface_writer.flush()?;
        preparation_profile.local_run_write_seconds += write_start.elapsed().as_secs_f64();
        attach_run_paths.push(attach_path);
        interface_run_paths.push(interface_path);

        let lower_values = summary.z_min_face.values;
        let upper_values = summary.z_max_face.values;
        if let Some(previous) = previous_upper_face.take() {
            let cross_start = Instant::now();
            cross_run_paths.push(write_cross_run(
                temp_directory,
                desc.slab_id - 1,
                &previous,
                desc,
                &lower_values,
                cross_run_config,
            )?);
            preparation_profile.cross_interface_seconds += cross_start.elapsed().as_secs_f64();
        }
        previous_upper_face = Some(StoredUpperFace {
            descriptor: desc.clone(),
            values: upper_values,
        });
    }

    let flush_start = Instant::now();
    birth_writer.flush()?;
    local_pair_writer.flush()?;
    preparation_profile.local_run_write_seconds += flush_start.elapsed().as_secs_f64();

    let interface_birth_bytes = path_len(&births_path)?;
    let local_pair_bytes = path_len(&local_pairs_path)?;
    let attach_bytes = sum_path_lengths(&attach_run_paths)?;
    let interface_bytes = sum_path_lengths(&interface_run_paths)?;
    let cross_bytes = sum_path_lengths(&cross_run_paths)?;
    let total_run_bytes = interface_birth_bytes
        .checked_add(local_pair_bytes)
        .and_then(|v| v.checked_add(attach_bytes))
        .and_then(|v| v.checked_add(interface_bytes))
        .and_then(|v| v.checked_add(cross_bytes))
        .ok_or_else(|| anyhow::anyhow!("temporary H0 run byte count overflow"))?;

    Ok(PreparedH0Runs {
        total_interface_nodes,
        births_path,
        local_pairs_path,
        attach_run_paths,
        interface_run_paths,
        cross_run_paths,
        preparation_profile,
        disk_key_bytes: <F32Key as DiskScalarKey>::DISK_BYTES,
        interface_birth_bytes,
        local_pair_bytes,
        attach_bytes,
        interface_bytes,
        cross_bytes,
        total_run_bytes,
    })
}

fn read_interface_births<K: DiskScalarKey>(path: &Path, count: usize) -> Result<Vec<K>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut births = Vec::with_capacity(count);

    for _ in 0..count {
        let value = read_scalar_key(&mut reader)?.ok_or_else(|| {
            anyhow::anyhow!("interface birth file ended before {count} entries were read")
        })?;
        births.push(value);
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        anyhow::bail!("scalar interface birth file contains trailing data");
    }

    Ok(births)
}

fn write_pair_csv<K: DiskScalarKey>(writer: &mut impl Write, pair: FinitePair<K>) -> Result<bool> {
    if pair.birth >= pair.death {
        return Ok(false);
    }

    let birth = pair.birth.widen();
    let death = pair.death.widen();
    writeln!(writer, "{birth},{death}")?;
    Ok(true)
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarH0PersistenceStats {
    pub finite_intervals: u64,
    pub essential_intervals: u64,
}

type MinHeap<K> = BinaryHeap<Reverse<(K, usize)>>;

fn open_attach_runs<K: DiskScalarKey>(
    paths: &[PathBuf],
) -> Result<(Vec<AttachRunReader<K>>, MinHeap<K>)> {
    let readers = paths
        .iter()
        .map(|path| AttachRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push(Reverse((value, index)));
        }
    }
    Ok((readers, heap))
}

fn open_pair_runs<K: DiskScalarKey>(
    paths: &[PathBuf],
) -> Result<(Vec<PairRunReader<K>>, MinHeap<K>)> {
    let readers = paths
        .iter()
        .map(|path| PairRunReader::<K>::open(path))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (index, reader) in readers.iter().enumerate() {
        if let Some(value) = reader.peek_value() {
            heap.push(Reverse((value, index)));
        }
    }
    Ok((readers, heap))
}

fn min_heap_value<K: DiskScalarKey>(heap: &MinHeap<K>) -> Option<K> {
    heap.peek().map(|Reverse((value, _))| *value)
}

fn next_scalar_value<K: DiskScalarKey>(
    attach: &MinHeap<K>,
    interface: &MinHeap<K>,
    cross: &MinHeap<K>,
) -> Option<K> {
    [
        min_heap_value(attach),
        min_heap_value(interface),
        min_heap_value(cross),
    ]
    .into_iter()
    .flatten()
    .min()
}

fn drain_attach_value<K: DiskScalarKey>(
    value: K,
    readers: &mut [AttachRunReader<K>],
    heap: &mut MinHeap<K>,
    mut apply: impl FnMut(u32, K) -> Result<()>,
) -> Result<()> {
    while min_heap_value(heap) == Some(value) {
        let Reverse((_, reader_index)) = heap.pop().expect("heap head disappeared");
        let (node, branch_birth) = readers[reader_index]
            .pop_at_value(value)?
            .ok_or_else(|| anyhow::anyhow!("scalar H0 attach heap is out of order"))?;
        apply(node, branch_birth)?;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push(Reverse((next, reader_index)));
        }
    }
    Ok(())
}

fn drain_pair_value<K: DiskScalarKey>(
    value: K,
    readers: &mut [PairRunReader<K>],
    heap: &mut MinHeap<K>,
    mut apply: impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    while min_heap_value(heap) == Some(value) {
        let Reverse((_, reader_index)) = heap.pop().expect("heap head disappeared");
        let (a, b) = readers[reader_index]
            .pop_at_value(value)?
            .ok_or_else(|| anyhow::anyhow!("scalar H0 pair heap is out of order"))?;
        apply(a, b)?;
        if let Some(next) = readers[reader_index].peek_value() {
            heap.push(Reverse((next, reader_index)));
        }
    }
    Ok(())
}

fn next_scalar_value_scan<K: DiskScalarKey>(
    attach_readers: &[AttachRunReader<K>],
    interface_readers: &[PairRunReader<K>],
    cross_readers: &[PairRunReader<K>],
) -> Option<K> {
    attach_readers
        .iter()
        .filter_map(|reader| reader.peek_value())
        .chain(
            interface_readers
                .iter()
                .filter_map(|reader| reader.peek_value()),
        )
        .chain(
            cross_readers
                .iter()
                .filter_map(|reader| reader.peek_value()),
        )
        .min()
}

fn reduce_h0_runs_scan<K: DiskScalarKey>(
    prepared: &PreparedH0Runs,
    output_path: &Path,
    global_h0_uf_layout: GlobalH0UnionFindLayoutStrategy,
) -> Result<ScalarH0PersistenceStats> {
    let births = read_interface_births::<K>(&prepared.births_path, prepared.total_interface_nodes)?;
    let mut global_uf = GlobalScalarPersistenceUnionFind::<K>::new(births, global_h0_uf_layout);
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H0_STATE scalar_h0_stream layout={} nodes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_h0_uf_layout.as_str(),
        prepared.total_interface_nodes,
        parent_bytes,
        rank_bytes,
        birth_bytes,
        parent_bytes + rank_bytes + birth_bytes
    );

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

    let mut output = AtomicOutput::create(output_path)?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<K>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    while let Some(death) =
        next_scalar_value_scan(&attach_readers, &interface_readers, &cross_readers)
    {
        for reader in &mut attach_readers {
            while let Some((node, branch_birth)) = reader.pop_at_value(death)? {
                let pair = global_uf.attach_branch(node, branch_birth, death);
                if write_pair_csv(&mut output, pair)? {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut interface_readers {
            while let Some((a, b)) = reader.pop_at_value(death)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }

        for reader in &mut cross_readers {
            while let Some((a, b)) = reader.pop_at_value(death)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
            }
        }
    }

    let essential_births = global_uf.essential_births();
    for &birth in &essential_births {
        let birth = birth.widen();
        writeln!(output, "{birth},inf")?;
    }
    output.commit()?;

    Ok(ScalarH0PersistenceStats {
        finite_intervals,
        essential_intervals: essential_births.len() as u64,
    })
}

fn reduce_h0_runs_heap<K: DiskScalarKey>(
    prepared: &PreparedH0Runs,
    output_path: &Path,
    global_h0_uf_layout: GlobalH0UnionFindLayoutStrategy,
) -> Result<ScalarH0PersistenceStats> {
    let births = read_interface_births::<K>(&prepared.births_path, prepared.total_interface_nodes)?;
    let mut global_uf = GlobalScalarPersistenceUnionFind::<K>::new(births, global_h0_uf_layout);
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H0_STATE scalar_h0_stream layout={} nodes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        global_h0_uf_layout.as_str(),
        prepared.total_interface_nodes,
        parent_bytes,
        rank_bytes,
        birth_bytes,
        parent_bytes + rank_bytes + birth_bytes
    );

    let (mut attach_readers, mut attach_heap) = open_attach_runs::<K>(&prepared.attach_run_paths)?;
    let (mut interface_readers, mut interface_heap) =
        open_pair_runs::<K>(&prepared.interface_run_paths)?;
    let (mut cross_readers, mut cross_heap) = open_pair_runs::<K>(&prepared.cross_run_paths)?;

    let mut output = AtomicOutput::create(output_path)?;
    writeln!(output, "birth,death")?;

    let mut finite_intervals = 0u64;
    let mut local_reader = BufReader::new(File::open(&prepared.local_pairs_path)?);
    while let Some(pair) = read_finite_pair::<K>(&mut local_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    while let Some(death) = next_scalar_value(&attach_heap, &interface_heap, &cross_heap) {
        drain_attach_value(
            death,
            &mut attach_readers,
            &mut attach_heap,
            |node, branch_birth| {
                let pair = global_uf.attach_branch(node, branch_birth, death);
                if write_pair_csv(&mut output, pair)? {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(
            death,
            &mut interface_readers,
            &mut interface_heap,
            |a, b| {
                if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                    && write_pair_csv(&mut output, pair)?
                {
                    finite_intervals += 1;
                }
                Ok(())
            },
        )?;

        drain_pair_value(death, &mut cross_readers, &mut cross_heap, |a, b| {
            if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
            Ok(())
        })?;
    }

    let essential_births = global_uf.essential_births();
    for &birth in &essential_births {
        let birth = birth.widen();
        writeln!(output, "{birth},inf")?;
    }
    output.commit()?;

    Ok(ScalarH0PersistenceStats {
        finite_intervals,
        essential_intervals: essential_births.len() as u64,
    })
}

fn reduce_h0_runs<K: DiskScalarKey>(
    prepared: &PreparedH0Runs,
    output_path: &Path,
    merge_strategy: MergeStrategy,
    global_h0_uf_layout: GlobalH0UnionFindLayoutStrategy,
) -> Result<ScalarH0PersistenceStats> {
    let reader_count = prepared.attach_run_paths.len()
        + prepared.interface_run_paths.len()
        + prepared.cross_run_paths.len();
    let effective = match merge_strategy {
        MergeStrategy::Auto if reader_count > 32 => MergeStrategy::Heap,
        MergeStrategy::Auto => MergeStrategy::Scan,
        other => other,
    };
    println!(
        "PROFILE_MERGE_SELECT scalar_h0_stream requested={} effective={} readers={} threshold=32",
        merge_strategy.as_str(),
        effective.as_str(),
        reader_count,
    );
    match effective {
        MergeStrategy::Scan => reduce_h0_runs_scan::<K>(prepared, output_path, global_h0_uf_layout),
        MergeStrategy::Heap => reduce_h0_runs_heap::<K>(prepared, output_path, global_h0_uf_layout),
        MergeStrategy::Auto => {
            unreachable!("auto merge strategy must be resolved before reduction")
        }
    }
}

// -----------------------------------------------------------------------------
// Experimental disk-backed hierarchical H0 stream (v1.21 candidate)
// -----------------------------------------------------------------------------

const H0_HIER_PACKED_ROOT_BASE: u32 = u32::MAX - u8::MAX as u32;

#[derive(Debug)]
struct DiskHierarchicalH0Summary<K: DiskScalarKey> {
    z0: usize,
    z1: usize,
    width: usize,
    height: usize,
    interface_node_count: u32,
    lower_values: Vec<K>,
    upper_values: Vec<K>,
    attach_path: PathBuf,
    interface_path: PathBuf,
}

impl<K: DiskScalarKey> DiskHierarchicalH0Summary<K> {
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
        u32::try_from(local).expect("hierarchical H0 face index exceeds u32")
    }
}

#[derive(Debug, Clone, Copy)]
enum DiskHierarchicalH0Action<K> {
    FinalPair(FinitePair<K>),
    Attach(AttachDiskEvent<K>),
    Interface(PairEvent<K>),
}

struct DiskHierarchicalH0UnionFind<K: DiskScalarKey> {
    parent: Vec<u32>,
    birth: Vec<K>,
    left_count: u32,
    right_offset: u32,
    right_upper_start: u32,
    face_size: u32,
    attach_pruning: H0HierAttachPruningStrategy,
    attach_finalized_early: u64,
    attach_propagated: u64,
}

impl<K: DiskScalarKey> DiskHierarchicalH0UnionFind<K> {
    fn new(
        birth: Vec<K>,
        left_count: u32,
        right_offset: u32,
        right_upper_start: u32,
        face_size: u32,
        attach_pruning: H0HierAttachPruningStrategy,
    ) -> Result<Self> {
        if birth.len() >= H0_HIER_PACKED_ROOT_BASE as usize {
            bail!(
                "hierarchical packed H0 pair needs {} nodes, but the packed root encoding supports fewer than {}",
                birth.len(),
                H0_HIER_PACKED_ROOT_BASE
            );
        }
        Ok(Self {
            parent: vec![Self::root_word(0); birth.len()],
            birth,
            left_count,
            right_offset,
            right_upper_start,
            face_size,
            attach_pruning,
            attach_finalized_early: 0,
            attach_propagated: 0,
        })
    }

    #[inline]
    fn root_word(rank: u8) -> u32 {
        u32::MAX - u32::from(rank)
    }

    #[inline]
    fn root_rank(word: u32) -> Option<u8> {
        (word >= H0_HIER_PACKED_ROOT_BASE).then(|| (u32::MAX - word) as u8)
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
        // The left child's lower face is always local IDs 0..A.
        if root < self.face_size && root < self.left_count {
            return Some(root);
        }
        // The right child's upper face occupies either 0..A for a depth-1
        // child or A..2A otherwise, translated by right_offset.
        let start = self.right_offset + self.right_upper_start;
        let end = start + self.face_size;
        if root >= start && root < end {
            return Some(self.face_size + (root - start));
        }
        None
    }

    fn rank(&self, root: u32) -> u8 {
        Self::root_rank(self.parent[root as usize]).expect("hierarchical H0 root marker missing")
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
            // The reference summary algebra carries rep_a when both components
            // already touch the parent boundary. Force the first terminal root
            // to survive so the disk-backed summary uses the same representative
            // convention while still avoiding a terminal-representative vector.
            (true, true) => self.force_link(a, b),
            (false, false) => self.rank_link(a, b),
        }
    }

    fn union_with_summary(
        &mut self,
        a: u32,
        b: u32,
        death: K,
    ) -> Option<DiskHierarchicalH0Action<K>> {
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
            (None, None) => DiskHierarchicalH0Action::FinalPair(FinitePair {
                birth: birth_a.max(birth_b),
                death,
            }),
            (Some(node), None) => {
                if matches!(
                    self.attach_pruning,
                    H0HierAttachPruningStrategy::ElderDominated
                ) && birth_a <= birth_b
                {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH0Action::FinalPair(FinitePair {
                        birth: birth_b,
                        death,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH0Action::Attach(AttachDiskEvent {
                        value: death,
                        node,
                        branch_birth: birth_b,
                    })
                }
            }
            (None, Some(node)) => {
                if matches!(
                    self.attach_pruning,
                    H0HierAttachPruningStrategy::ElderDominated
                ) && birth_b <= birth_a
                {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH0Action::FinalPair(FinitePair {
                        birth: birth_a,
                        death,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH0Action::Attach(AttachDiskEvent {
                        value: death,
                        node,
                        branch_birth: birth_a,
                    })
                }
            }
            (Some(a), Some(b)) => {
                DiskHierarchicalH0Action::Interface(PairEvent { value: death, a, b })
            }
        };

        let new_root = self.link_preserving_terminal(root_a, root_b);
        self.birth[new_root as usize] = birth_a.min(birth_b);
        Some(action)
    }

    fn attach_branch(
        &mut self,
        node: u32,
        branch_birth: K,
        death: K,
    ) -> DiskHierarchicalH0Action<K> {
        let root = self.find(node);
        let root_birth = self.birth[root as usize];
        self.birth[root as usize] = root_birth.min(branch_birth);
        match self.terminal_parent_id(root) {
            Some(node) => {
                if matches!(
                    self.attach_pruning,
                    H0HierAttachPruningStrategy::ElderDominated
                ) && root_birth <= branch_birth
                {
                    self.attach_finalized_early += 1;
                    DiskHierarchicalH0Action::FinalPair(FinitePair {
                        birth: branch_birth,
                        death,
                    })
                } else {
                    self.attach_propagated += 1;
                    DiskHierarchicalH0Action::Attach(AttachDiskEvent {
                        value: death,
                        node,
                        branch_birth,
                    })
                }
            }
            None => DiskHierarchicalH0Action::FinalPair(FinitePair {
                birth: root_birth.max(branch_birth),
                death,
            }),
        }
    }

    fn attach_pruning_counts(&self) -> (u64, u64) {
        (self.attach_finalized_early, self.attach_propagated)
    }

    fn count_internal_roots(&mut self) -> usize {
        let mut count = 0usize;
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 && self.terminal_parent_id(node_u32).is_none() {
                count += 1;
            }
        }
        count
    }
}

fn write_hierarchical_action<K: DiskScalarKey>(
    action: DiskHierarchicalH0Action<K>,
    pair_writer: &mut BufWriter<File>,
    attach_writer: &mut BufWriter<File>,
    interface_writer: &mut BufWriter<File>,
) -> Result<()> {
    match action {
        DiskHierarchicalH0Action::FinalPair(pair) => {
            if pair.birth < pair.death {
                write_finite_pair(pair_writer, pair)?;
            }
        }
        DiskHierarchicalH0Action::Attach(event) => write_attach_event(attach_writer, event)?,
        DiskHierarchicalH0Action::Interface(event) => write_pair_event(interface_writer, event)?,
    }
    Ok(())
}

fn assign_hierarchical_births<K: DiskScalarKey>(
    summary: &DiskHierarchicalH0Summary<K>,
    offset: usize,
    births: &mut [K],
) {
    let face_size = summary.face_size();
    for (index, &value) in summary.lower_values.iter().enumerate() {
        births[offset + index] = value;
    }
    if summary.depth() > 1 {
        for (index, &value) in summary.upper_values.iter().enumerate() {
            births[offset + face_size + index] = value;
        }
    }
}

fn next_hierarchical_value<K: DiskScalarKey>(
    left_attach: &AttachRunReader<K>,
    right_attach: &AttachRunReader<K>,
    left_interface: &PairRunReader<K>,
    right_interface: &PairRunReader<K>,
    cross: &PairRunReader<K>,
) -> Option<K> {
    [
        left_attach.peek_value(),
        right_attach.peek_value(),
        left_interface.peek_value(),
        right_interface.peek_value(),
        cross.peek_value(),
    ]
    .into_iter()
    .flatten()
    .min()
}

#[allow(clippy::too_many_arguments)]
fn combine_disk_h0_summaries<K: DiskScalarKey>(
    left: DiskHierarchicalH0Summary<K>,
    right: DiskHierarchicalH0Summary<K>,
    connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
    attach_pruning: H0HierAttachPruningStrategy,
) -> Result<(DiskHierarchicalH0Summary<K>, usize, u64, u64, u64)> {
    if left.z1 != right.z0 {
        bail!(
            "disk-backed hierarchical H0 summaries are not adjacent: left z={}..{}, right z={}..{}",
            left.z0,
            left.z1,
            right.z0,
            right.z1
        );
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H0 summary face dimensions differ");
    }
    let face_size = left.face_size();
    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if total_nodes >= H0_HIER_PACKED_ROOT_BASE as usize {
        bail!("disk-backed hierarchical H0 pair is too large for packed u32 roots");
    }
    let seed = *left
        .lower_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty hierarchical H0 lower face"))?;
    let mut births = vec![seed; total_nodes];
    assign_hierarchical_births(&left, 0, &mut births);
    let right_offset = left.interface_node_count as usize;
    assign_hierarchical_births(&right, right_offset, &mut births);

    let right_upper_start = if right.depth() == 1 { 0 } else { face_size };
    let mut uf = DiskHierarchicalH0UnionFind::new(
        births,
        left.interface_node_count,
        u32::try_from(right_offset).expect("hierarchical H0 right offset exceeds u32"),
        u32::try_from(right_upper_start).expect("hierarchical H0 right upper offset exceeds u32"),
        u32::try_from(face_size).expect("hierarchical H0 face size exceeds u32"),
        attach_pruning,
    )?;

    let cross_path = directory.join(format!("h0_hier_cross_{summary_id:06}.bin"));
    let mut cross_writer = BufWriter::new(File::create(&cross_path)?);
    let cross_stats = sparsify_scalar_sublevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        connectivity,
        interface_order,
        |edge| {
            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = u32::try_from(right_offset)
                .expect("hierarchical H0 right offset exceeds u32")
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H0 face index exceeds u32");
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

    let parent_attach_path = directory.join(format!("h0_hier_attach_{summary_id:06}.bin"));
    let parent_interface_path = directory.join(format!("h0_hier_interface_{summary_id:06}.bin"));
    let mut parent_attach_writer = BufWriter::new(File::create(&parent_attach_path)?);
    let mut parent_interface_writer = BufWriter::new(File::create(&parent_interface_path)?);

    {
        let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
        let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
        let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
        let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
        let mut cross = PairRunReader::<K>::open(&cross_path)?;

        while let Some(death) = next_hierarchical_value(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
        ) {
            // Preserve the summary algebra's deterministic priority:
            // attach before interface before cross at equal filtration value.
            while let Some((node, branch_birth)) = left_attach.pop_at_value(death)? {
                let action = uf.attach_branch(node, branch_birth, death);
                write_hierarchical_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some((node, branch_birth)) = right_attach.pop_at_value(death)? {
                let shifted = u32::try_from(right_offset)
                    .expect("hierarchical H0 right offset exceeds u32")
                    + node;
                let action = uf.attach_branch(shifted, branch_birth, death);
                write_hierarchical_action(
                    action,
                    pair_writer,
                    &mut parent_attach_writer,
                    &mut parent_interface_writer,
                )?;
            }
            while let Some((a, b)) = left_interface.pop_at_value(death)? {
                if let Some(action) = uf.union_with_summary(a, b, death) {
                    write_hierarchical_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
            while let Some((a, b)) = right_interface.pop_at_value(death)? {
                let offset =
                    u32::try_from(right_offset).expect("hierarchical H0 right offset exceeds u32");
                if let Some(action) = uf.union_with_summary(offset + a, offset + b, death) {
                    write_hierarchical_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
            while let Some((a, b)) = cross.pop_at_value(death)? {
                if let Some(action) = uf.union_with_summary(a, b, death) {
                    write_hierarchical_action(
                        action,
                        pair_writer,
                        &mut parent_attach_writer,
                        &mut parent_interface_writer,
                    )?;
                }
            }
        }
    }

    let internal_roots = uf.count_internal_roots();
    if internal_roots != 0 {
        bail!(
            "disk-backed hierarchical H0 composition left {internal_roots} components without an outer-face representative"
        );
    }

    parent_attach_writer.flush()?;
    parent_interface_writer.flush()?;
    drop(parent_attach_writer);
    drop(parent_interface_writer);

    // Once the parent summary is durable, both child event summaries and the
    // transient shared-face cross run can be discarded immediately.
    for path in [
        &left.attach_path,
        &left.interface_path,
        &right.attach_path,
        &right.interface_path,
        &cross_path,
    ] {
        fs::remove_file(path)
            .with_context(|| format!("could not delete consumed hierarchical run {path:?}"))?;
    }

    let parent_depth = right.z1 - left.z0;
    let parent_interface_nodes = if parent_depth == 1 {
        face_size
    } else {
        2 * face_size
    };
    let state_bytes = u64::try_from(total_nodes)
        .expect("hierarchical H0 pair node count exceeds u64")
        .saturating_mul((4 + K::DISK_BYTES) as u64);

    let parent_attach_bytes = path_len(&parent_attach_path)?;
    let parent_interface_bytes = path_len(&parent_interface_path)?;
    let (attach_finalized_early, attach_propagated) = uf.attach_pruning_counts();
    println!(
        "PROFILE_H0_HIER_STREAM_COMBINE z0={} z1={} pair_nodes={} pair_state_bytes={} parent_interface_nodes={} cross_retained={} cross_candidates={} parent_attach_bytes={} parent_interface_bytes={} attach_finalized_early={} attach_propagated={} attach_pruning={}",
        left.z0,
        right.z1,
        total_nodes,
        state_bytes,
        parent_interface_nodes,
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
        parent_attach_bytes,
        parent_interface_bytes,
        attach_finalized_early,
        attach_propagated,
        attach_pruning.as_str(),
    );

    Ok((
        DiskHierarchicalH0Summary {
            z0: left.z0,
            z1: right.z1,
            width: left.width,
            height: left.height,
            interface_node_count: u32::try_from(parent_interface_nodes)
                .expect("hierarchical H0 parent interface count exceeds u32"),
            lower_values: left.lower_values,
            upper_values: right.upper_values,
            attach_path: parent_attach_path,
            interface_path: parent_interface_path,
        },
        total_nodes,
        state_bytes,
        attach_finalized_early,
        attach_propagated,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finalize_disk_h0_pair<K: DiskScalarKey>(
    left: DiskHierarchicalH0Summary<K>,
    right: DiskHierarchicalH0Summary<K>,
    connectivity: Connectivity,
    interface_order: InterfaceOrderStrategy,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
) -> Result<(Vec<K>, usize, u64)> {
    if left.z1 != right.z0 {
        bail!(
            "disk-backed hierarchical H0 final summaries are not adjacent: left z={}..{}, right z={}..{}",
            left.z0,
            left.z1,
            right.z0,
            right.z1
        );
    }
    if left.width != right.width || left.height != right.height {
        bail!("disk-backed hierarchical H0 final summary face dimensions differ");
    }

    let total_nodes = left.interface_node_count as usize + right.interface_node_count as usize;
    if total_nodes >= H0_HIER_PACKED_ROOT_BASE as usize {
        bail!("disk-backed hierarchical H0 final pair is too large for packed u32 roots");
    }
    let seed = *left
        .lower_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty hierarchical H0 final lower face"))?;
    let mut births = vec![seed; total_nodes];
    assign_hierarchical_births(&left, 0, &mut births);
    let right_offset = left.interface_node_count as usize;
    assign_hierarchical_births(&right, right_offset, &mut births);

    // The final fan-in has no surviving spatial terminals: it is the entire
    // image domain. Therefore it can use the ordinary persistence UF directly
    // and emit all remaining finite pairs instead of materializing another
    // relative (root) summary on disk.
    let mut global_uf =
        GlobalScalarPersistenceUnionFind::<K>::new(births, GlobalH0UnionFindLayoutStrategy::Packed);
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    let state_bytes = parent_bytes + rank_bytes + birth_bytes;

    let cross_path = directory.join(format!("h0_hier_final_cross_{summary_id:06}.bin"));
    let mut cross_writer = BufWriter::new(File::create(&cross_path)?);
    let cross_stats = sparsify_scalar_sublevel_interface_with_order(
        &left.upper_values,
        &right.lower_values,
        left.width,
        left.height,
        connectivity,
        interface_order,
        |edge| {
            let left_node = left.upper_local_id(edge.left_face_index as usize);
            let right_node = u32::try_from(right_offset)
                .expect("hierarchical H0 final right offset exceeds u32")
                + u32::try_from(edge.right_face_index as usize)
                    .expect("hierarchical H0 final face index exceeds u32");
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
        let mut left_attach = AttachRunReader::<K>::open(&left.attach_path)?;
        let mut right_attach = AttachRunReader::<K>::open(&right.attach_path)?;
        let mut left_interface = PairRunReader::<K>::open(&left.interface_path)?;
        let mut right_interface = PairRunReader::<K>::open(&right.interface_path)?;
        let mut cross = PairRunReader::<K>::open(&cross_path)?;

        while let Some(death) = next_hierarchical_value(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
        ) {
            // Match the deterministic event priority used by the relative
            // summary combiner: attach before interface before cross.
            while let Some((node, branch_birth)) = left_attach.pop_at_value(death)? {
                let pair = global_uf.attach_branch(node, branch_birth, death);
                if pair.birth < pair.death {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((node, branch_birth)) = right_attach.pop_at_value(death)? {
                let shifted = u32::try_from(right_offset)
                    .expect("hierarchical H0 final right offset exceeds u32")
                    + node;
                let pair = global_uf.attach_branch(shifted, branch_birth, death);
                if pair.birth < pair.death {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = left_interface.pop_at_value(death)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = right_interface.pop_at_value(death)? {
                let offset = u32::try_from(right_offset)
                    .expect("hierarchical H0 final right offset exceeds u32");
                if let Some(pair) = global_uf.union_with_persistence(offset + a, offset + b, death)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
            while let Some((a, b)) = cross.pop_at_value(death)? {
                if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                    && pair.birth < pair.death
                {
                    write_finite_pair(pair_writer, pair)?;
                }
            }
        }
    }

    let essential_births = global_uf.essential_births();
    for path in [
        &left.attach_path,
        &left.interface_path,
        &right.attach_path,
        &right.interface_path,
        &cross_path,
    ] {
        fs::remove_file(path).with_context(|| {
            format!("could not delete consumed final hierarchical run {path:?}")
        })?;
    }

    println!(
        "PROFILE_H0_HIER_STREAM_FINAL z0={} z1={} pair_nodes={} pair_state_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} cross_retained={} cross_candidates={} root_materialized=false",
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

    Ok((essential_births, total_nodes, state_bytes))
}

#[allow(clippy::too_many_arguments)]
fn prepare_disk_h0_leaf_f32(
    volume: &ScalarTiffStackReader,
    slab_id: usize,
    z0: usize,
    z1: usize,
    connectivity: Connectivity,
    directory: &Path,
    summary_id: usize,
    pair_writer: &mut BufWriter<File>,
    tuning: ScalarStreamTuning,
) -> Result<(DiskHierarchicalH0Summary<F32Key>, H0SlabPreparationProfile)> {
    let attach_path = directory.join(format!("h0_hier_attach_{summary_id:06}.bin"));
    let interface_path = directory.join(format!("h0_hier_interface_{summary_id:06}.bin"));
    let mut attach_writer = BufWriter::new(File::create(&attach_path)?);
    let mut interface_writer = BufWriter::new(File::create(&interface_path)?);
    let mut previous_attach_value: Option<F32Key> = None;
    let mut previous_interface_value: Option<F32Key> = None;

    let (block, _read_profile) = volume.read_z_slab_f32_native_profiled(z0, z1)?;
    let (summary, slab_profile) = process_slab_h0_persistence_f32_native_direct_profiled(
        slab_id,
        block,
        connectivity,
        tuning.neighbor_kernel,
        tuning.representative_active_check,
        tuning.union_kernel,
        tuning.h0_pruning_cache,
        tuning.active_state,
        tuning.interface_state,
        tuning.uf_layout,
        tuning.sweep_diagnostics,
        |event| {
            match event {
                H0LocalStreamEvent::FinalPair(pair) => write_finite_pair(pair_writer, pair)?,
                H0LocalStreamEvent::Attach(event) => {
                    if previous_attach_value.is_some_and(|value| event.value < value) {
                        bail!(
                            "hierarchical direct H0 attach events are not monotone within slab {slab_id}"
                        );
                    }
                    previous_attach_value = Some(event.value);
                    write_attach_event(
                        &mut attach_writer,
                        AttachDiskEvent {
                            value: event.value,
                            node: event.interface_node,
                            branch_birth: event.branch_birth,
                        },
                    )?;
                }
                H0LocalStreamEvent::InterfaceMerge(event) => {
                    if previous_interface_value.is_some_and(|value| event.value < value) {
                        bail!(
                            "hierarchical direct H0 interface events are not monotone within slab {slab_id}"
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
    interface_writer.flush()?;

    Ok((
        DiskHierarchicalH0Summary {
            z0,
            z1,
            width: volume.width,
            height: volume.height,
            interface_node_count: summary.interface_node_count,
            lower_values: summary.z_min_face.values,
            upper_values: summary.z_max_face.values,
            attach_path,
            interface_path,
        },
        slab_profile,
    ))
}

fn reduce_disk_h0_root<K: DiskScalarKey>(
    root: DiskHierarchicalH0Summary<K>,
    pairs_path: &Path,
    output_path: &Path,
) -> Result<ScalarH0PersistenceStats> {
    let face_size = root.face_size();
    let mut births = Vec::with_capacity(root.interface_node_count as usize);
    births.extend_from_slice(&root.lower_values);
    if root.depth() > 1 {
        births.extend_from_slice(&root.upper_values);
    }
    if births.len() != root.interface_node_count as usize {
        bail!("hierarchical H0 root birth count does not match interface node count");
    }

    let mut global_uf =
        GlobalScalarPersistenceUnionFind::<K>::new(births, GlobalH0UnionFindLayoutStrategy::Packed);
    let (parent_bytes, rank_bytes, birth_bytes) = global_uf.capacity_bytes();
    println!(
        "PROFILE_GLOBAL_H0_STATE scalar_h0_hierarchical_stream layout=packed nodes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
        root.interface_node_count,
        parent_bytes,
        rank_bytes,
        birth_bytes,
        parent_bytes + rank_bytes + birth_bytes,
    );

    let mut output = AtomicOutput::create(output_path)?;
    writeln!(output, "birth,death")?;
    let mut finite_intervals = 0u64;
    let mut pair_reader = BufReader::new(File::open(pairs_path)?);
    while let Some(pair) = read_finite_pair::<K>(&mut pair_reader)? {
        if write_pair_csv(&mut output, pair)? {
            finite_intervals += 1;
        }
    }

    let mut attach = AttachRunReader::<K>::open(&root.attach_path)?;
    let mut interface = PairRunReader::<K>::open(&root.interface_path)?;
    loop {
        let death = [attach.peek_value(), interface.peek_value()]
            .into_iter()
            .flatten()
            .min();
        let Some(death) = death else { break };
        while let Some((node, branch_birth)) = attach.pop_at_value(death)? {
            let pair = global_uf.attach_branch(node, branch_birth, death);
            if write_pair_csv(&mut output, pair)? {
                finite_intervals += 1;
            }
        }
        while let Some((a, b)) = interface.pop_at_value(death)? {
            if let Some(pair) = global_uf.union_with_persistence(a, b, death)
                && write_pair_csv(&mut output, pair)?
            {
                finite_intervals += 1;
            }
        }
    }

    let essential_births = global_uf.essential_births();
    for birth in &essential_births {
        writeln!(output, "{},inf", birth.widen())?;
    }
    output.commit()?;

    // face_size is intentionally reported to make the 2A final-state invariant auditable.
    println!(
        "PROFILE_H0_HIER_STREAM_ROOT face_size={} final_interface_nodes={}",
        face_size, root.interface_node_count,
    );
    Ok(ScalarH0PersistenceStats {
        finite_intervals,
        essential_intervals: essential_births.len() as u64,
    })
}

pub fn compute_h0_persistence_scalar_hierarchical_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    output_path: &Path,
    tuning: ScalarStreamTuning,
) -> Result<ScalarH0PersistenceStats> {
    let start = Instant::now();
    if volume.pixel_type != ScalarPixelType::F32 || tuning.f32_key_mode != F32KeyMode::Native32 {
        bail!(
            "h0-scalar-hierarchical-stream currently requires an F32 TIFF stack with --f32-key-mode native32"
        );
    }
    if !matches!(tuning.h0_birth_buffer, H0BirthBufferStrategy::ReuseInput) {
        bail!("h0-scalar-hierarchical-stream requires --h0-birth-buffer reuse-input");
    }
    if !matches!(tuning.h0_event_storage, H0EventStorageStrategy::Direct) {
        bail!("h0-scalar-hierarchical-stream requires --h0-event-storage direct");
    }

    let temp_directory = TempRunDirectory::create("h0_scalar_hierarchical_stream_runs")?;
    println!(
        "Scalar hierarchical H0 streaming temporary directory: {:?}",
        temp_directory.path()
    );
    let pairs_path = temp_directory.path().join("h0_hier_finalized_pairs.bin");
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
        .expect("hierarchical H0 face size overflow");
    let planned_pair_nodes = if ranges.len() > 1 {
        face_size.saturating_mul(4)
    } else if volume.depth == 1 {
        face_size
    } else {
        face_size.saturating_mul(2)
    };
    let planned_pair_state_bytes = u64::try_from(planned_pair_nodes)
        .expect("hierarchical H0 planned pair nodes exceed u64")
        .saturating_mul(8);
    println!(
        "Scalar hierarchical H0 storage: pipeline=native32-direct-disk-fanin disk_key_bytes=4 pair_state_bytes_per_node=8 planned_max_pair_nodes={} planned_max_pair_state_bytes={} final_interface_bound={}",
        planned_pair_nodes,
        planned_pair_state_bytes,
        if volume.depth == 1 {
            face_size
        } else {
            face_size.saturating_mul(2)
        },
    );
    println!(
        "PROFILE_CONFIG scalar_h0_hierarchical_stream f32_key_mode={} interface_order={} event_order={} neighbor_kernel={} representative_active_check={} union_kernel={} h0_pruning_cache={} h0_birth_buffer={} active_state={} interface_state={} uf_layout={} global_h0_uf_layout=packed h0_event_storage={} h0_hier_attach_pruning={}",
        tuning.f32_key_mode.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.h0_pruning_cache.as_str(),
        tuning.h0_birth_buffer.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.h0_event_storage.as_str(),
        tuning.h0_hier_attach_pruning.as_str(),
    );

    let mut slots: Vec<Option<DiskHierarchicalH0Summary<F32Key>>> = Vec::new();
    let mut next_summary_id = ranges.len();
    let mut combines = 0usize;
    let mut max_live_summaries = 0usize;
    let mut max_pair_nodes = 0usize;
    let mut max_pair_state_bytes = 0u64;
    let mut attach_finalized_early_total = 0u64;
    let mut attach_propagated_total = 0u64;
    let mut final_children: Option<(
        DiskHierarchicalH0Summary<F32Key>,
        DiskHierarchicalH0Summary<F32Key>,
    )> = None;

    'leaf_loop: for &(leaf_id, leaf_z0, leaf_z1) in &ranges {
        println!("Preparing optimized hierarchical H0 leaf {leaf_id}: z={leaf_z0}..{leaf_z1}");
        let (mut current, _profile) = prepare_disk_h0_leaf_f32(
            volume,
            leaf_id,
            leaf_z0,
            leaf_z1,
            connectivity,
            temp_directory.path(),
            leaf_id,
            &mut pair_writer,
            tuning,
        )?;
        let mut level = 0usize;
        loop {
            if level == slots.len() {
                slots.push(Some(current));
                break;
            }
            if let Some(left) = slots[level].take() {
                // If these two summaries cover the entire image, do not create
                // another relative root summary. Keep them as the final two
                // children and perform a terminal-free persistence reduction
                // directly after leaf preparation finishes.
                let covers_entire_volume = ranges.len() > 1
                    && leaf_id + 1 == ranges.len()
                    && left.z0 == 0
                    && current.z1 == volume.depth;
                if covers_entire_volume {
                    final_children = Some((left, current));
                    break 'leaf_loop;
                }

                println!(
                    "Optimized hierarchical H0 fan-in level {level}: z={}..{} + z={}..{}",
                    left.z0, left.z1, current.z0, current.z1,
                );
                let (
                    parent,
                    pair_nodes,
                    pair_state_bytes,
                    attach_finalized_early,
                    attach_propagated,
                ) = combine_disk_h0_summaries(
                    left,
                    current,
                    connectivity,
                    tuning.interface_order,
                    temp_directory.path(),
                    next_summary_id,
                    &mut pair_writer,
                    tuning.h0_hier_attach_pruning,
                )?;
                attach_finalized_early_total += attach_finalized_early;
                attach_propagated_total += attach_propagated;
                next_summary_id += 1;
                combines += 1;
                max_pair_nodes = max_pair_nodes.max(pair_nodes);
                max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
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

    let mut root_fallback: Option<DiskHierarchicalH0Summary<F32Key>> = None;
    if final_children.is_none() {
        let mut remaining: Vec<_> = slots.into_iter().flatten().collect();
        if remaining.is_empty() {
            bail!("hierarchical H0 stream received an empty volume");
        }
        remaining.sort_by_key(|summary| summary.z0);

        if remaining.len() == 1 {
            // Single-slab volumes have no pairwise fan-in to intercept. Keep
            // the existing root reducer as the exact fallback for this case.
            root_fallback = remaining.pop();
        } else {
            let mut left = remaining.remove(0);
            while remaining.len() > 1 {
                let right = remaining.remove(0);
                let (
                    parent,
                    pair_nodes,
                    pair_state_bytes,
                    attach_finalized_early,
                    attach_propagated,
                ) = combine_disk_h0_summaries(
                    left,
                    right,
                    connectivity,
                    tuning.interface_order,
                    temp_directory.path(),
                    next_summary_id,
                    &mut pair_writer,
                    tuning.h0_hier_attach_pruning,
                )?;
                attach_finalized_early_total += attach_finalized_early;
                attach_propagated_total += attach_propagated;
                next_summary_id += 1;
                combines += 1;
                max_pair_nodes = max_pair_nodes.max(pair_nodes);
                max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
                left = parent;
            }
            let right = remaining
                .pop()
                .expect("hierarchical H0 final child unexpectedly missing");
            final_children = Some((left, right));
        }
    }

    let mut terminal_free_result: Option<(Vec<F32Key>, usize, u64)> = None;
    if let Some((left, right)) = final_children {
        println!(
            "Optimized hierarchical H0 terminal-free final fan-in: z={}..{} + z={}..{}",
            left.z0, left.z1, right.z0, right.z1,
        );
        let (essential_births, pair_nodes, pair_state_bytes) = finalize_disk_h0_pair(
            left,
            right,
            connectivity,
            tuning.interface_order,
            temp_directory.path(),
            next_summary_id,
            &mut pair_writer,
        )?;
        combines += 1;
        max_pair_nodes = max_pair_nodes.max(pair_nodes);
        max_pair_state_bytes = max_pair_state_bytes.max(pair_state_bytes);
        terminal_free_result = Some((essential_births, pair_nodes, pair_state_bytes));
    }

    pair_writer.flush()?;
    drop(pair_writer);
    let finalized_pair_bytes = path_len(&pairs_path)?;

    let stats = if let Some((essential_births, final_pair_nodes, final_pair_state_bytes)) =
        terminal_free_result
    {
        println!(
            "PROFILE_H0_HIER_STREAM leaf_slabs={} combines={} max_live_summaries={} max_pair_nodes={} max_pair_state_bytes={} final_interface_nodes=0 finalized_pair_bytes={} root_attach_bytes=0 root_interface_bytes=0 final_pair_nodes={} final_pair_state_bytes={} root_materialized=false disk_key_bytes=4 h0_birth_buffer=reuse-input h0_event_storage=direct global_h0_uf_layout=packed attach_finalized_early={} attach_propagated={} h0_hier_attach_pruning={}",
            ranges.len(),
            combines,
            max_live_summaries,
            max_pair_nodes,
            max_pair_state_bytes,
            finalized_pair_bytes,
            final_pair_nodes,
            final_pair_state_bytes,
            attach_finalized_early_total,
            attach_propagated_total,
            tuning.h0_hier_attach_pruning.as_str(),
        );

        let mut output = AtomicOutput::create(output_path)?;
        writeln!(output, "birth,death")?;
        let mut finite_intervals = 0u64;
        let mut pair_reader = BufReader::new(File::open(&pairs_path)?);
        while let Some(pair) = read_finite_pair::<F32Key>(&mut pair_reader)? {
            if write_pair_csv(&mut output, pair)? {
                finite_intervals += 1;
            }
        }
        for birth in &essential_births {
            writeln!(output, "{},inf", birth.widen())?;
        }
        output.commit()?;
        ScalarH0PersistenceStats {
            finite_intervals,
            essential_intervals: essential_births.len() as u64,
        }
    } else {
        let root = root_fallback
            .ok_or_else(|| anyhow::anyhow!("hierarchical H0 final root unexpectedly missing"))?;
        let root_attach_bytes = path_len(&root.attach_path)?;
        let root_interface_bytes = path_len(&root.interface_path)?;
        println!(
            "PROFILE_H0_HIER_STREAM leaf_slabs={} combines={} max_live_summaries={} max_pair_nodes={} max_pair_state_bytes={} final_interface_nodes={} finalized_pair_bytes={} root_attach_bytes={} root_interface_bytes={} root_materialized=true disk_key_bytes=4 h0_birth_buffer=reuse-input h0_event_storage=direct global_h0_uf_layout=packed attach_finalized_early={} attach_propagated={} h0_hier_attach_pruning={}",
            ranges.len(),
            combines,
            max_live_summaries,
            max_pair_nodes,
            max_pair_state_bytes,
            root.interface_node_count,
            finalized_pair_bytes,
            root_attach_bytes,
            root_interface_bytes,
            attach_finalized_early_total,
            attach_propagated_total,
            tuning.h0_hier_attach_pruning.as_str(),
        );
        reduce_disk_h0_root(root, &pairs_path, output_path)?
    };

    let total_seconds = start.elapsed().as_secs_f64();
    println!("PROFILE scalar_h0_hierarchical_stream total_seconds={total_seconds:.6}");
    temp_directory.close()?;
    Ok(stats)
}

pub fn compute_h0_persistence_scalar_stream_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    output_path: &Path,
    tuning: ScalarStreamTuning,
) -> Result<ScalarH0PersistenceStats> {
    let start = Instant::now();

    let temp_directory = TempRunDirectory::create("h0_scalar_stream_runs")?;

    println!(
        "Scalar H0 streaming temporary directory: {:?}",
        temp_directory.path()
    );

    let native_f32_pipeline =
        volume.pixel_type == ScalarPixelType::F32 && tuning.f32_key_mode == F32KeyMode::Native32;
    if matches!(tuning.h0_birth_buffer, H0BirthBufferStrategy::ReuseInput) && !native_f32_pipeline {
        bail!(
            "--h0-birth-buffer reuse-input currently requires an F32 stack with --f32-key-mode native32"
        );
    }
    let planned_descriptors =
        make_slab_descriptors(volume.width, volume.height, volume.depth, slab_depth);
    let planned_interface_nodes = planned_descriptors
        .last()
        .map(|desc| u64::from(desc.interface_base) + u64::from(desc.interface_count))
        .unwrap_or(0);
    let planned_disk_key_bytes = if native_f32_pipeline { 4u64 } else { 8u64 };
    let planned_parent_bytes = planned_interface_nodes.saturating_mul(4);
    let planned_rank_bytes = match tuning.global_h0_uf_layout {
        GlobalH0UnionFindLayoutStrategy::ParentRank => planned_interface_nodes,
        GlobalH0UnionFindLayoutStrategy::Packed => 0,
    };
    let planned_birth_bytes = planned_interface_nodes.saturating_mul(planned_disk_key_bytes);
    let planned_global_bytes = planned_parent_bytes
        .saturating_add(planned_rank_bytes)
        .saturating_add(planned_birth_bytes);
    println!(
        "Scalar H0 key storage: pipeline={} disk_key_bytes={} global_h0_uf_layout={} interface_nodes={} estimated_global_parent_bytes={} estimated_global_rank_bytes={} estimated_global_birth_bytes={} estimated_global_total_bytes={}",
        if native_f32_pipeline {
            "native32-end-to-end"
        } else {
            "wide64"
        },
        planned_disk_key_bytes,
        tuning.global_h0_uf_layout.as_str(),
        planned_interface_nodes,
        planned_parent_bytes,
        planned_rank_bytes,
        planned_birth_bytes,
        planned_global_bytes
    );

    let prepare_start = Instant::now();
    println!(
        "Scalar H0 tuning: merge={} interface_order={} event_order={} f32_key_mode={} neighbor_kernel={} representative_active_check={} union_kernel={} h0_pruning_cache={} h0_birth_buffer={} active_state={} interface_state={} uf_layout={} global_h0_uf_layout={} h0_event_storage={} sweep_diagnostics={}",
        tuning.merge_strategy.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.f32_key_mode.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.h0_pruning_cache.as_str(),
        tuning.h0_birth_buffer.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.global_h0_uf_layout.as_str(),
        tuning.h0_event_storage.as_str(),
        tuning.sweep_diagnostics
    );
    let prepared = if native_f32_pipeline {
        match tuning.h0_event_storage {
            H0EventStorageStrategy::Buffered => prepare_h0_runs_f32_native(
                volume,
                slab_depth,
                connectivity,
                temp_directory.path(),
                tuning,
            )?,
            H0EventStorageStrategy::Direct => prepare_h0_runs_f32_native_direct(
                volume,
                slab_depth,
                connectivity,
                temp_directory.path(),
                tuning,
            )?,
        }
    } else {
        if matches!(tuning.h0_event_storage, H0EventStorageStrategy::Direct) {
            bail!(
                "--h0-event-storage direct currently requires an F32 stack with --f32-key-mode native32"
            );
        }
        prepare_h0_runs_scalar(
            volume,
            slab_depth,
            connectivity,
            temp_directory.path(),
            tuning,
        )?
    };
    let prepare_seconds = prepare_start.elapsed().as_secs_f64();
    println!("Scalar H0 slab preparation completed in {prepare_seconds:.3} seconds.");
    println!(
        "PROFILE_H0_STORAGE scalar_h0_stream pipeline={} disk_key_bytes={} interface_nodes={} interface_birth_bytes={} local_pair_bytes={} attach_bytes={} interface_bytes={} cross_bytes={} total_run_bytes={} estimated_global_parent_bytes={} estimated_global_rank_bytes={} estimated_global_birth_bytes={} estimated_global_total_bytes={} global_h0_uf_layout={} h0_birth_buffer={} h0_event_storage={}",
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
        prepared.interface_bytes,
        prepared.cross_bytes,
        prepared.total_run_bytes,
        planned_parent_bytes,
        planned_rank_bytes,
        planned_birth_bytes,
        planned_global_bytes,
        tuning.global_h0_uf_layout.as_str(),
        tuning.h0_birth_buffer.as_str(),
        tuning.h0_event_storage.as_str()
    );
    println!("Starting disk-backed scalar H0 persistence reduction.");

    let reduce_start = Instant::now();
    let stats = if native_f32_pipeline {
        reduce_h0_runs::<F32Key>(
            &prepared,
            output_path,
            tuning.merge_strategy,
            tuning.global_h0_uf_layout,
        )?
    } else {
        reduce_h0_runs::<ScalarKey>(
            &prepared,
            output_path,
            tuning.merge_strategy,
            tuning.global_h0_uf_layout,
        )?
    };
    let reduce_seconds = reduce_start.elapsed().as_secs_f64();

    let cleanup_start = Instant::now();
    temp_directory.close()?;
    let cleanup_seconds = cleanup_start.elapsed().as_secs_f64();
    let total_seconds = start.elapsed().as_secs_f64();

    println!("Streaming scalar H0 persistence computation took {total_seconds:.3} seconds");
    println!(
        "PROFILE scalar_h0_stream prepare_seconds={prepare_seconds:.6} \
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
        "PROFILE_PREP scalar_h0_stream decode_seconds={:.6} key_conversion_seconds={:.6} \
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
            "PROFILE_CACHE scalar_h0_stream mode={} entries={} storage_bytes={}",
            tuning.h0_pruning_cache.as_str(),
            tuning.h0_pruning_cache.entries().unwrap_or(0),
            tuning.h0_pruning_cache.storage_bytes(),
        );
        println!(
            "PROFILE_SWEEP scalar_h0_stream pruning_mask_calls={} active_state_checks={} active_neighbor_hits={} \
representative_visits={} pruning_cache_hits={} pruning_cache_misses={} \
component_mask_computations={} union_attempts={} successful_unions={} \
same_root_unions={} find_calls={} find_parent_steps={} active_rechecks={} \
active_recheck_failures={} root_carry_attempts={} avoided_find_calls={} pruning_cache_lookups={} \
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
            detailed.pruning_cache_lookups,
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
        "PROFILE_CONFIG scalar_h0_stream merge_strategy={} interface_order={} event_order={} f32_key_mode={} neighbor_kernel={} representative_active_check={} union_kernel={} h0_pruning_cache={} h0_birth_buffer={} active_state={} interface_state={} uf_layout={} global_h0_uf_layout={} h0_event_storage={} sweep_diagnostics={}",
        tuning.merge_strategy.as_str(),
        tuning.interface_order.as_str(),
        tuning.event_order.as_str(),
        tuning.f32_key_mode.as_str(),
        tuning.neighbor_kernel.as_str(),
        tuning.representative_active_check.as_str(),
        tuning.union_kernel.as_str(),
        tuning.h0_pruning_cache.as_str(),
        tuning.h0_birth_buffer.as_str(),
        tuning.active_state.as_str(),
        tuning.interface_state.as_str(),
        tuning.uf_layout.as_str(),
        tuning.global_h0_uf_layout.as_str(),
        tuning.h0_event_storage.as_str(),
        tuning.sweep_diagnostics
    );

    Ok(stats)
}

pub fn compute_h0_scalar_stream_batch(
    root: &Path,
    output_root: &Path,
    slab_depth: usize,
    connectivity: Connectivity,
    tuning: ScalarStreamTuning,
) -> Result<usize> {
    let directories = find_tiff_stack_directories(root)?;

    if directories.is_empty() {
        bail!("no TIFF-stack directories found at or below {root:?}");
    }

    fs::create_dir_all(output_root)
        .with_context(|| format!("could not create output directory {output_root:?}"))?;

    for (index, directory) in directories.iter().enumerate() {
        let relative = directory.strip_prefix(root).with_context(|| {
            format!("dataset directory {directory:?} is not underneath batch root {root:?}")
        })?;
        let dataset_output = output_root.join(relative);
        fs::create_dir_all(&dataset_output)?;

        println!();
        println!(
            "=== Streaming scalar H0 dataset {} of {}: {} ===",
            index + 1,
            directories.len(),
            directory.display()
        );
        let volume = ScalarTiffStackReader::open(directory)?;
        print_scalar_volume_info(&volume);
        let output = dataset_output.join("h0_persistence_scalar_stream.csv");
        compute_h0_persistence_scalar_stream_zslabs(
            &volume,
            slab_depth,
            connectivity,
            &output,
            tuning,
        )?;
    }

    Ok(directories.len())
}
