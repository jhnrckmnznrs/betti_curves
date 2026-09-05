use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::interface_sparsify_scalar::sparsify_scalar_sublevel_interface;
use crate::io_scalar::{
    F32ScalarBlock, ScalarBlock, ScalarTiffStackReader, print_scalar_volume_info,
};
use crate::local_pruning::{NeighborhoodComponentPruner, NeighborhoodPruningStats};
use crate::local_uf_state::LocalUnionFindState;
use crate::scalar::{F32Key, LocalScalarKey, ScalarKey};
use crate::scalar_order::{sorted_f32_indices, sorted_scalar_indices};
use crate::scalar_stream_tuning::{
    ActiveStateStrategy, GlobalH0UnionFindLayoutStrategy, H0PruningCacheStrategy,
    InterfaceStateStrategy, NeighborKernelStrategy, RepresentativeActiveCheckStrategy,
    UnionFindLayoutStrategy, UnionKernelStrategy,
};
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};
use crate::tiff_paths::find_tiff_stack_directories;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarH0Interval {
    pub birth: ScalarKey,
    pub death: Option<ScalarKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FinitePair<K = ScalarKey> {
    pub(crate) birth: K,
    pub(crate) death: K,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttachEvent<K = ScalarKey> {
    pub(crate) value: K,
    pub(crate) interface_node: u32,
    pub(crate) branch_birth: K,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InterfaceMergeEvent<K = ScalarKey> {
    pub(crate) value: K,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

#[derive(Debug)]
pub(crate) struct BoundaryFaceNodes<K = ScalarKey> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) node_ids: Vec<u32>,
    pub(crate) values: Vec<K>,
}

#[derive(Debug)]
pub(crate) struct SlabH0Summary<K = ScalarKey> {
    pub(crate) slab_id: usize,
    pub(crate) finalized_pairs: Vec<FinitePair<K>>,
    pub(crate) attach_events: Vec<AttachEvent<K>>,
    pub(crate) interface_merge_events: Vec<InterfaceMergeEvent<K>>,
    pub(crate) interface_node_count: u32,
    pub(crate) z_min_face: BoundaryFaceNodes<K>,
    pub(crate) z_max_face: BoundaryFaceNodes<K>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct H0SlabPreparationProfile {
    pub(crate) scalar_order_seconds: f64,
    pub(crate) local_sweep_seconds: f64,
    pub(crate) total_voxels: u64,
    pub(crate) interior_fast_voxels: u64,
    pub(crate) pruning_mask_calls: u64,
    pub(crate) active_state_checks: u64,
    pub(crate) active_neighbor_hits: u64,
    pub(crate) representative_visits: u64,
    pub(crate) pruning_cache_lookups: u64,
    pub(crate) pruning_cache_hits: u64,
    pub(crate) pruning_cache_misses: u64,
    pub(crate) component_mask_computations: u64,
    pub(crate) union_attempts: u64,
    pub(crate) successful_unions: u64,
    pub(crate) same_root_unions: u64,
    pub(crate) find_calls: u64,
    pub(crate) find_parent_steps: u64,
    pub(crate) active_rechecks: u64,
    pub(crate) active_recheck_failures: u64,
    pub(crate) root_carry_attempts: u64,
    pub(crate) avoided_find_calls: u64,
    pub(crate) interface_rep_queries: u64,
    pub(crate) interface_rep_writes: u64,
    pub(crate) interface_forced_root_unions: u64,
    pub(crate) interface_interface_unions: u64,
    pub(crate) interface_state_bytes: u64,
    pub(crate) max_rank_observed: u64,
    pub(crate) uf_parent_state_bytes: u64,
    pub(crate) uf_rank_state_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct LocalUnionFindStats {
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
    max_rank_observed: u64,
    uf_parent_state_bytes: u64,
    uf_rank_state_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
enum LocalPersistenceAction<K> {
    FinalPair(FinitePair<K>),
    Attach(AttachEvent<K>),
    InterfaceMerge(InterfaceMergeEvent<K>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum H0LocalStreamEvent<K = ScalarKey> {
    FinalPair(FinitePair<K>),
    Attach(AttachEvent<K>),
    InterfaceMerge(InterfaceMergeEvent<K>),
}

#[derive(Debug)]
struct LocalPersistenceUnionFind<K> {
    uf_state: LocalUnionFindState,
    birth: Vec<K>,
    interface_state: InterfaceStateStrategy,
    interface_rep: Option<Vec<u32>>,
    slice_size: usize,
    depth: usize,
    diagnostics: Option<LocalUnionFindStats>,
}

impl<K: LocalScalarKey> LocalPersistenceUnionFind<K> {
    fn new(
        values: &[K],
        shape: [usize; 3],
        diagnostics: bool,
        active_state: ActiveStateStrategy,
        interface_state: InterfaceStateStrategy,
        uf_layout: UnionFindLayoutStrategy,
    ) -> Self {
        Self::new_owned(
            values.to_vec(),
            shape,
            diagnostics,
            active_state,
            interface_state,
            uf_layout,
        )
    }

    fn new_owned(
        birth: Vec<K>,
        shape: [usize; 3],
        diagnostics: bool,
        active_state: ActiveStateStrategy,
        interface_state: InterfaceStateStrategy,
        uf_layout: UnionFindLayoutStrategy,
    ) -> Self {
        let uf_state = LocalUnionFindState::new(birth.len(), active_state, uf_layout);

        let interface_rep = matches!(interface_state, InterfaceStateStrategy::Vector)
            .then(|| vec![NO_INTERFACE_REP; birth.len()]);

        Self {
            uf_state,
            birth,
            interface_state,
            interface_rep,
            slice_size: shape[0] * shape[1],
            depth: shape[2],
            diagnostics: diagnostics.then(LocalUnionFindStats::default),
        }
    }

    #[inline]
    fn scalar_key_at(&self, index: usize) -> K {
        self.birth[index]
    }

    fn activate(&mut self, x: u32) {
        self.uf_state.activate(x);
    }

    fn parent_state_is_active(&self, x: usize) -> bool {
        self.uf_state.is_active(x)
    }

    fn find(&mut self, x: u32) -> u32 {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.find_calls += 1;
        }
        let (root, steps) = self.uf_state.find(x);
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.find_parent_steps += steps;
        }
        root
    }

    fn record_active_recheck(&mut self, failed: bool) {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.active_rechecks += 1;
            if failed {
                stats.active_recheck_failures += 1;
            }
        }
    }

    fn diagnostic_stats(&self) -> LocalUnionFindStats {
        let mut stats = self.diagnostics.unwrap_or_default();
        stats.uf_parent_state_bytes = self.uf_state.parent_state_bytes();
        stats.uf_rank_state_bytes = self.uf_state.rank_state_bytes();
        stats
    }

    fn interface_node_for_root(&mut self, root: u32) -> u32 {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.interface_rep_queries += 1;
        }
        match self.interface_state {
            InterfaceStateStrategy::Vector => self
                .interface_rep
                .as_ref()
                .expect("vector interface state requires interface_rep")[root as usize],
            InterfaceStateStrategy::RootInvariant => {
                let index = root as usize;
                let z = index / self.slice_size;
                let face_index = index % self.slice_size;
                local_boundary_node_id(z, self.depth, face_index, self.slice_size)
                    .unwrap_or(NO_INTERFACE_REP)
            }
        }
    }

    fn set_interface_rep(&mut self, x: u32, interface_node: u32) {
        let root = self.find(x);
        match self.interface_state {
            InterfaceStateStrategy::Vector => {
                self.interface_rep
                    .as_mut()
                    .expect("vector interface state requires interface_rep")[root as usize] =
                    interface_node;
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.interface_rep_writes += 1;
                }
            }
            InterfaceStateStrategy::RootInvariant => {
                let derived = self.interface_node_for_root(root);
                debug_assert_eq!(derived, interface_node);
            }
        }
    }

    fn link_roots(&mut self, mut root_a: u32, mut root_b: u32, rep_a: u32, rep_b: u32) -> u32 {
        let rank_a = self.uf_state.root_rank(root_a);
        let rank_b = self.uf_state.root_rank(root_b);

        if matches!(self.interface_state, InterfaceStateStrategy::RootInvariant) {
            let a_interface = rep_a != NO_INTERFACE_REP;
            let b_interface = rep_b != NO_INTERFACE_REP;
            if a_interface && b_interface {
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.interface_interface_unions += 1;
                }
            }
            if a_interface != b_interface {
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.interface_forced_root_unions += 1;
                }
                if !a_interface {
                    std::mem::swap(&mut root_a, &mut root_b);
                }
                let survivor_rank = self.uf_state.root_rank(root_a);
                let child_rank = self.uf_state.root_rank(root_b);
                self.uf_state.link_root_under(root_b, root_a);
                let new_rank = survivor_rank.max(child_rank.saturating_add(1));
                self.uf_state.set_root_rank(root_a, new_rank);
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.max_rank_observed = stats.max_rank_observed.max(u64::from(new_rank));
                }
                return root_a;
            }
        }

        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.uf_state.link_root_under(root_b, root_a);
        let new_rank = if rank_a == rank_b {
            self.uf_state.root_rank(root_a).saturating_add(1)
        } else {
            self.uf_state.root_rank(root_a)
        };
        self.uf_state.set_root_rank(root_a, new_rank);
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.max_rank_observed = stats.max_rank_observed.max(u64::from(new_rank));
        }
        root_a
    }

    fn merge_known_roots_with_persistence(
        &mut self,
        root_a: u32,
        root_b: u32,
        death: K,
    ) -> (u32, Option<LocalPersistenceAction<K>>) {
        debug_assert!(self.uf_state.is_root(root_a));
        debug_assert!(self.uf_state.is_root(root_b));

        if root_a == root_b {
            if let Some(stats) = self.diagnostics.as_mut() {
                stats.same_root_unions += 1;
            }
            return (root_a, None);
        }
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.successful_unions += 1;
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let rep_a = self.interface_node_for_root(root_a);
        let rep_b = self.interface_node_for_root(root_b);
        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => LocalPersistenceAction::FinalPair(FinitePair {
                birth: birth_a.max(birth_b),
                death,
            }),
            (true, false) => LocalPersistenceAction::Attach(AttachEvent {
                value: death,
                interface_node: rep_a,
                branch_birth: birth_b,
            }),
            (false, true) => LocalPersistenceAction::Attach(AttachEvent {
                value: death,
                interface_node: rep_b,
                branch_birth: birth_a,
            }),
            (true, true) => LocalPersistenceAction::InterfaceMerge(InterfaceMergeEvent {
                value: death,
                a: rep_a,
                b: rep_b,
            }),
        };

        let merged_birth = birth_a.min(birth_b);
        let new_root = self.link_roots(root_a, root_b, rep_a, rep_b);
        self.birth[new_root as usize] = merged_birth;
        if matches!(self.interface_state, InterfaceStateStrategy::Vector) {
            self.interface_rep
                .as_mut()
                .expect("vector interface state requires interface_rep")[new_root as usize] =
                if rep_a != NO_INTERFACE_REP {
                    rep_a
                } else {
                    rep_b
                };
            if let Some(stats) = self.diagnostics.as_mut() {
                stats.interface_rep_writes += 1;
            }
        }

        (new_root, Some(action))
    }

    fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        death: K,
    ) -> Option<LocalPersistenceAction<K>> {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.union_attempts += 1;
        }
        let root_a = self.find(a);
        let root_b = self.find(b);
        self.merge_known_roots_with_persistence(root_a, root_b, death)
            .1
    }

    fn union_from_current_root_with_persistence(
        &mut self,
        current_root: u32,
        neighbor: u32,
        death: K,
    ) -> (u32, Option<LocalPersistenceAction<K>>) {
        debug_assert!(self.uf_state.is_root(current_root));
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.union_attempts += 1;
            stats.root_carry_attempts += 1;
            stats.avoided_find_calls += 1;
        }
        let neighbor_root = self.find(neighbor);
        self.merge_known_roots_with_persistence(current_root, neighbor_root, death)
    }
}

fn handle_local_action<K: LocalScalarKey>(
    action: LocalPersistenceAction<K>,
    finalized_pairs: &mut Vec<FinitePair<K>>,
    attach_events: &mut Vec<AttachEvent<K>>,
    interface_merge_events: &mut Vec<InterfaceMergeEvent<K>>,
) {
    match action {
        LocalPersistenceAction::FinalPair(pair) => {
            if pair.birth < pair.death {
                finalized_pairs.push(pair);
            }
        }
        LocalPersistenceAction::Attach(event) => attach_events.push(event),
        LocalPersistenceAction::InterfaceMerge(event) => interface_merge_events.push(event),
    }
}

struct LocalPersistenceBuffers<'a, K> {
    finalized_pairs: &'a mut Vec<FinitePair<K>>,
    attach_events: &'a mut Vec<AttachEvent<K>>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent<K>>,
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
fn union_active_neighbor_h0<K: LocalScalarKey>(
    uf: &mut LocalPersistenceUnionFind<K>,
    active: Option<&[u8]>,
    active_state: ActiveStateStrategy,
    current: u32,
    current_root: &mut u32,
    neighbor: usize,
    value: K,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    buffers: &mut LocalPersistenceBuffers<'_, K>,
) {
    if matches!(
        representative_active_check,
        RepresentativeActiveCheckStrategy::Recheck
    ) {
        let failed = match active_state {
            ActiveStateStrategy::Separate => {
                active.expect("separate active state requires byte array")[neighbor] == 0
            }
            ActiveStateStrategy::ParentSentinel => !uf.parent_state_is_active(neighbor),
        };
        uf.record_active_recheck(failed);
        if failed {
            return;
        }
    }

    let action = match union_kernel {
        UnionKernelStrategy::Conventional => {
            uf.union_with_persistence(current, neighbor as u32, value)
        }
        UnionKernelStrategy::RootCarrying => {
            let (new_root, action) =
                uf.union_from_current_root_with_persistence(*current_root, neighbor as u32, value);
            *current_root = new_root;
            action
        }
    };

    if let Some(action) = action {
        handle_local_action(
            action,
            buffers.finalized_pairs,
            buffers.attach_events,
            buffers.interface_merge_events,
        );
    }
}

#[allow(clippy::too_many_arguments)] // Explicit tuning knobs are intentionally passed separately.
fn union_active_neighbor_h0_sink<K, F>(
    uf: &mut LocalPersistenceUnionFind<K>,
    active: Option<&[u8]>,
    active_state: ActiveStateStrategy,
    current: u32,
    current_root: &mut u32,
    neighbor: usize,
    value: K,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    emit: &mut F,
) -> Result<()>
where
    K: LocalScalarKey,
    F: FnMut(H0LocalStreamEvent<K>) -> Result<()>,
{
    if matches!(
        representative_active_check,
        RepresentativeActiveCheckStrategy::Recheck
    ) {
        let failed = match active_state {
            ActiveStateStrategy::Separate => {
                active.expect("separate active state requires byte array")[neighbor] == 0
            }
            ActiveStateStrategy::ParentSentinel => !uf.parent_state_is_active(neighbor),
        };
        uf.record_active_recheck(failed);
        if failed {
            return Ok(());
        }
    }

    let action = match union_kernel {
        UnionKernelStrategy::Conventional => {
            uf.union_with_persistence(current, neighbor as u32, value)
        }
        UnionKernelStrategy::RootCarrying => {
            let (new_root, action) =
                uf.union_from_current_root_with_persistence(*current_root, neighbor as u32, value);
            *current_root = new_root;
            action
        }
    };

    if let Some(action) = action {
        match action {
            LocalPersistenceAction::FinalPair(pair) => {
                if pair.birth < pair.death {
                    emit(H0LocalStreamEvent::FinalPair(pair))?;
                }
            }
            LocalPersistenceAction::Attach(event) => emit(H0LocalStreamEvent::Attach(event))?,
            LocalPersistenceAction::InterfaceMerge(event) => {
                emit(H0LocalStreamEvent::InterfaceMerge(event))?
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_slab_h0_persistence_values_owned_sink<K, F>(
    slab_id: usize,
    shape: [usize; 3],
    values: Vec<K>,
    order: Vec<u32>,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
    mut emit: F,
) -> Result<(
    SlabH0Summary<K>,
    NeighborhoodPruningStats,
    LocalUnionFindStats,
)>
where
    K: LocalScalarKey,
    F: FnMut(H0LocalStreamEvent<K>) -> Result<()>,
{
    let width = shape[0];
    let height = shape[1];
    let depth = shape[2];
    let voxel_count = width * height * depth;
    let slice_size = width * height;
    assert_eq!(values.len(), voxel_count);

    // Boundary values must be copied before ownership of the input scalar
    // vector is transferred to union-find birth storage.
    let z_min_face = extract_boundary_face_nodes_values(shape, &values, 0);
    let z_max_face = extract_boundary_face_nodes_values(shape, &values, depth - 1);
    let mut uf = LocalPersistenceUnionFind::new_owned(
        values,
        shape,
        sweep_diagnostics,
        active_state,
        interface_state,
        uf_layout,
    );
    let mut active =
        matches!(active_state, ActiveStateStrategy::Separate).then(|| vec![0u8; voxel_count]);
    let mut local_pruner = NeighborhoodComponentPruner::with_diagnostics_and_cache_entries(
        connectivity,
        sweep_diagnostics,
        pruning_cache.entries(),
    );
    let linear_offsets = matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
        .then(|| local_pruner.linear_offsets(width, height));
    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut begin = 0usize;
    while begin < order.len() {
        let value = uf.scalar_key_at(order[begin] as usize);
        let mut end = begin + 1;
        while end < order.len() && uf.scalar_key_at(order[end] as usize) == value {
            end += 1;
        }

        for &idx_u32 in &order[begin..end] {
            let idx = idx_u32 as usize;
            uf.activate(idx_u32);
            if let Some(active) = active.as_mut() {
                active[idx] = 1;
            }
            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                uf.set_interface_rep(idx_u32, interface_node);
            }
            let x = idx % width;
            let y = (idx / width) % height;
            let mut current_root = idx_u32;
            let use_interior_fast_path =
                matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
                    && x > 0
                    && x + 1 < width
                    && y > 0
                    && y + 1 < height
                    && z > 0
                    && z + 1 < depth;

            let representatives = if use_interior_fast_path {
                let offsets = linear_offsets
                    .as_ref()
                    .expect("interior-fast kernel requires linear offsets");
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_interior_by(
                            idx,
                            offsets,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_interior_by(idx, offsets, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            } else {
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_by(
                            x,
                            y,
                            z,
                            width,
                            height,
                            depth,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_by(x, y, z, width, height, depth, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            };

            for neighbor in representatives.iter() {
                union_active_neighbor_h0_sink(
                    &mut uf,
                    active.as_deref(),
                    active_state,
                    idx_u32,
                    &mut current_root,
                    neighbor,
                    value,
                    representative_active_check,
                    union_kernel,
                    &mut emit,
                )?;
            }
        }
        begin = end;
    }

    let pruning_stats = local_pruner.diagnostic_stats();
    let union_find_stats = uf.diagnostic_stats();
    Ok((
        SlabH0Summary {
            slab_id,
            finalized_pairs: Vec::new(),
            attach_events: Vec::new(),
            interface_merge_events: Vec::new(),
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        pruning_stats,
        union_find_stats,
    ))
}

#[allow(clippy::too_many_arguments)] // Hot-path kernel keeps tuning knobs explicit to avoid per-voxel config indirection.
fn process_slab_h0_persistence_values<K: LocalScalarKey>(
    slab_id: usize,
    shape: [usize; 3],
    values: &[K],
    order: Vec<u32>,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
) -> (
    SlabH0Summary<K>,
    NeighborhoodPruningStats,
    LocalUnionFindStats,
) {
    let width = shape[0];
    let height = shape[1];
    let depth = shape[2];
    let voxel_count = width * height * depth;
    let slice_size = width * height;
    assert_eq!(values.len(), voxel_count);

    let mut uf = LocalPersistenceUnionFind::new(
        values,
        shape,
        sweep_diagnostics,
        active_state,
        interface_state,
        uf_layout,
    );
    let mut active =
        matches!(active_state, ActiveStateStrategy::Separate).then(|| vec![0u8; voxel_count]);
    let mut local_pruner = NeighborhoodComponentPruner::with_diagnostics_and_cache_entries(
        connectivity,
        sweep_diagnostics,
        pruning_cache.entries(),
    );
    let linear_offsets = matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
        .then(|| local_pruner.linear_offsets(width, height));

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut finalized_pairs = Vec::new();
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();

    let mut begin = 0usize;
    while begin < order.len() {
        let value = values[order[begin] as usize];
        let mut end = begin + 1;
        while end < order.len() && values[order[end] as usize] == value {
            end += 1;
        }

        for &idx_u32 in &order[begin..end] {
            let idx = idx_u32 as usize;
            uf.activate(idx_u32);
            if let Some(active) = active.as_mut() {
                active[idx] = 1;
            }
            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let x = idx % width;
            let y = (idx / width) % height;

            let mut buffers = LocalPersistenceBuffers {
                finalized_pairs: &mut finalized_pairs,
                attach_events: &mut attach_events,
                interface_merge_events: &mut interface_merge_events,
            };

            let mut current_root = idx_u32;

            let use_interior_fast_path =
                matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
                    && x > 0
                    && x + 1 < width
                    && y > 0
                    && y + 1 < height
                    && z > 0
                    && z + 1 < depth;

            let representatives = if use_interior_fast_path {
                let offsets = linear_offsets
                    .as_ref()
                    .expect("interior-fast kernel requires linear offsets");
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_interior_by(
                            idx,
                            offsets,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_interior_by(idx, offsets, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            } else {
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_by(
                            x,
                            y,
                            z,
                            width,
                            height,
                            depth,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_by(x, y, z, width, height, depth, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            };

            for neighbor in representatives.iter() {
                union_active_neighbor_h0(
                    &mut uf,
                    active.as_deref(),
                    active_state,
                    idx_u32,
                    &mut current_root,
                    neighbor,
                    value,
                    representative_active_check,
                    union_kernel,
                    &mut buffers,
                );
            }
        }

        begin = end;
    }

    let z_min_face = extract_boundary_face_nodes_values(shape, values, 0);
    let z_max_face = extract_boundary_face_nodes_values(shape, values, depth - 1);

    let pruning_stats = local_pruner.diagnostic_stats();
    let union_find_stats = uf.diagnostic_stats();
    (
        SlabH0Summary {
            slab_id,
            finalized_pairs,
            attach_events,
            interface_merge_events,
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        pruning_stats,
        union_find_stats,
    )
}

#[allow(clippy::too_many_arguments)] // Owned-buffer variant mirrors the hot-path kernel configuration.
fn process_slab_h0_persistence_values_owned<K: LocalScalarKey>(
    slab_id: usize,
    shape: [usize; 3],
    values: Vec<K>,
    order: Vec<u32>,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
) -> (
    SlabH0Summary<K>,
    NeighborhoodPruningStats,
    LocalUnionFindStats,
) {
    let width = shape[0];
    let height = shape[1];
    let depth = shape[2];
    let voxel_count = width * height * depth;
    let slice_size = width * height;
    assert_eq!(values.len(), voxel_count);

    let z_min_face = extract_boundary_face_nodes_values(shape, &values, 0);
    let z_max_face = extract_boundary_face_nodes_values(shape, &values, depth - 1);
    let mut uf = LocalPersistenceUnionFind::new_owned(
        values,
        shape,
        sweep_diagnostics,
        active_state,
        interface_state,
        uf_layout,
    );
    let mut active =
        matches!(active_state, ActiveStateStrategy::Separate).then(|| vec![0u8; voxel_count]);
    let mut local_pruner = NeighborhoodComponentPruner::with_diagnostics_and_cache_entries(
        connectivity,
        sweep_diagnostics,
        pruning_cache.entries(),
    );
    let linear_offsets = matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
        .then(|| local_pruner.linear_offsets(width, height));

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut finalized_pairs = Vec::new();
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();

    let mut begin = 0usize;
    while begin < order.len() {
        let value = uf.scalar_key_at(order[begin] as usize);
        let mut end = begin + 1;
        while end < order.len() && uf.scalar_key_at(order[end] as usize) == value {
            end += 1;
        }

        for &idx_u32 in &order[begin..end] {
            let idx = idx_u32 as usize;
            uf.activate(idx_u32);
            if let Some(active) = active.as_mut() {
                active[idx] = 1;
            }
            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let x = idx % width;
            let y = (idx / width) % height;

            let mut buffers = LocalPersistenceBuffers {
                finalized_pairs: &mut finalized_pairs,
                attach_events: &mut attach_events,
                interface_merge_events: &mut interface_merge_events,
            };

            let mut current_root = idx_u32;

            let use_interior_fast_path =
                matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
                    && x > 0
                    && x + 1 < width
                    && y > 0
                    && y + 1 < height
                    && z > 0
                    && z + 1 < depth;

            let representatives = if use_interior_fast_path {
                let offsets = linear_offsets
                    .as_ref()
                    .expect("interior-fast kernel requires linear offsets");
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_interior_by(
                            idx,
                            offsets,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_interior_by(idx, offsets, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            } else {
                match active_state {
                    ActiveStateStrategy::Separate => {
                        let active_ref = active
                            .as_deref()
                            .expect("separate active state requires byte array");
                        local_pruner.representative_neighbors_by(
                            x,
                            y,
                            z,
                            width,
                            height,
                            depth,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_by(x, y, z, width, height, depth, |neighbor| {
                            uf.parent_state_is_active(neighbor)
                        }),
                }
            };

            for neighbor in representatives.iter() {
                union_active_neighbor_h0(
                    &mut uf,
                    active.as_deref(),
                    active_state,
                    idx_u32,
                    &mut current_root,
                    neighbor,
                    value,
                    representative_active_check,
                    union_kernel,
                    &mut buffers,
                );
            }
        }

        begin = end;
    }

    let pruning_stats = local_pruner.diagnostic_stats();
    let union_find_stats = uf.diagnostic_stats();
    (
        SlabH0Summary {
            slab_id,
            finalized_pairs,
            attach_events,
            interface_merge_events,
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        pruning_stats,
        union_find_stats,
    )
}

pub(crate) fn process_slab_h0_persistence_scalar(
    slab_id: usize,
    block: &ScalarBlock,
    connectivity: Connectivity,
) -> SlabH0Summary {
    process_slab_h0_persistence_scalar_profiled(
        slab_id,
        block,
        connectivity,
        NeighborKernelStrategy::Generic,
        RepresentativeActiveCheckStrategy::Recheck,
        UnionKernelStrategy::Conventional,
        H0PruningCacheStrategy::Entries64K,
        ActiveStateStrategy::Separate,
        InterfaceStateStrategy::Vector,
        UnionFindLayoutStrategy::ParentRank,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h0_persistence_scalar_profiled(
    slab_id: usize,
    block: &ScalarBlock,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
) -> (SlabH0Summary, H0SlabPreparationProfile) {
    let order_start = Instant::now();
    let order = sorted_scalar_indices(&block.values, block.pixel_type);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h0_persistence_values(
        slab_id,
        block.shape,
        &block.values,
        order,
        connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        pruning_cache,
        active_state,
        interface_state,
        uf_layout,
        sweep_diagnostics,
    );
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    (
        summary,
        H0SlabPreparationProfile {
            scalar_order_seconds,
            local_sweep_seconds,
            total_voxels: u64::try_from(block.values.len()).expect("slab voxel count exceeds u64"),
            interior_fast_voxels: if matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
            {
                u64::try_from(block.shape[0].saturating_sub(2)).expect("width exceeds u64")
                    * u64::try_from(block.shape[1].saturating_sub(2)).expect("height exceeds u64")
                    * u64::try_from(block.shape[2].saturating_sub(2)).expect("depth exceeds u64")
            } else {
                0
            },
            pruning_mask_calls: pruning_stats.mask_calls,
            active_state_checks: pruning_stats.active_state_checks,
            active_neighbor_hits: pruning_stats.active_neighbor_hits,
            representative_visits: pruning_stats.representative_visits,
            pruning_cache_lookups: pruning_stats.cache_lookups,
            pruning_cache_hits: pruning_stats.cache_hits,
            pruning_cache_misses: pruning_stats.cache_misses,
            component_mask_computations: pruning_stats.component_mask_computations,
            union_attempts: union_find_stats.union_attempts,
            successful_unions: union_find_stats.successful_unions,
            same_root_unions: union_find_stats.same_root_unions,
            find_calls: union_find_stats.find_calls,
            find_parent_steps: union_find_stats.find_parent_steps,
            active_rechecks: union_find_stats.active_rechecks,
            active_recheck_failures: union_find_stats.active_recheck_failures,
            root_carry_attempts: union_find_stats.root_carry_attempts,
            avoided_find_calls: union_find_stats.avoided_find_calls,
            interface_rep_queries: union_find_stats.interface_rep_queries,
            interface_rep_writes: union_find_stats.interface_rep_writes,
            interface_forced_root_unions: union_find_stats.interface_forced_root_unions,
            interface_interface_unions: union_find_stats.interface_interface_unions,
            interface_state_bytes: if matches!(interface_state, InterfaceStateStrategy::Vector) {
                u64::try_from(block.values.len()).expect("slab voxel count exceeds u64") * 4
            } else {
                0
            },
            max_rank_observed: union_find_stats.max_rank_observed,
            uf_parent_state_bytes: union_find_stats.uf_parent_state_bytes,
            uf_rank_state_bytes: union_find_stats.uf_rank_state_bytes,
        },
    )
}

#[cfg(test)]
pub(crate) fn process_slab_h0_persistence_f32_native(
    slab_id: usize,
    block: &F32ScalarBlock,
    connectivity: Connectivity,
) -> SlabH0Summary<F32Key> {
    process_slab_h0_persistence_f32_native_profiled(
        slab_id,
        block,
        connectivity,
        NeighborKernelStrategy::Generic,
        RepresentativeActiveCheckStrategy::Recheck,
        UnionKernelStrategy::Conventional,
        H0PruningCacheStrategy::Entries64K,
        ActiveStateStrategy::Separate,
        InterfaceStateStrategy::Vector,
        UnionFindLayoutStrategy::ParentRank,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h0_persistence_f32_native_profiled(
    slab_id: usize,
    block: &F32ScalarBlock,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
) -> (SlabH0Summary<F32Key>, H0SlabPreparationProfile) {
    let order_start = Instant::now();
    let order = sorted_f32_indices(&block.values);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h0_persistence_values(
        slab_id,
        block.shape,
        &block.values,
        order,
        connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        pruning_cache,
        active_state,
        interface_state,
        uf_layout,
        sweep_diagnostics,
    );
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    (
        summary,
        H0SlabPreparationProfile {
            scalar_order_seconds,
            local_sweep_seconds,
            total_voxels: u64::try_from(block.values.len()).expect("slab voxel count exceeds u64"),
            interior_fast_voxels: if matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
            {
                u64::try_from(block.shape[0].saturating_sub(2)).expect("width exceeds u64")
                    * u64::try_from(block.shape[1].saturating_sub(2)).expect("height exceeds u64")
                    * u64::try_from(block.shape[2].saturating_sub(2)).expect("depth exceeds u64")
            } else {
                0
            },
            pruning_mask_calls: pruning_stats.mask_calls,
            active_state_checks: pruning_stats.active_state_checks,
            active_neighbor_hits: pruning_stats.active_neighbor_hits,
            representative_visits: pruning_stats.representative_visits,
            pruning_cache_lookups: pruning_stats.cache_lookups,
            pruning_cache_hits: pruning_stats.cache_hits,
            pruning_cache_misses: pruning_stats.cache_misses,
            component_mask_computations: pruning_stats.component_mask_computations,
            union_attempts: union_find_stats.union_attempts,
            successful_unions: union_find_stats.successful_unions,
            same_root_unions: union_find_stats.same_root_unions,
            find_calls: union_find_stats.find_calls,
            find_parent_steps: union_find_stats.find_parent_steps,
            active_rechecks: union_find_stats.active_rechecks,
            active_recheck_failures: union_find_stats.active_recheck_failures,
            root_carry_attempts: union_find_stats.root_carry_attempts,
            avoided_find_calls: union_find_stats.avoided_find_calls,
            interface_rep_queries: union_find_stats.interface_rep_queries,
            interface_rep_writes: union_find_stats.interface_rep_writes,
            interface_forced_root_unions: union_find_stats.interface_forced_root_unions,
            interface_interface_unions: union_find_stats.interface_interface_unions,
            interface_state_bytes: if matches!(interface_state, InterfaceStateStrategy::Vector) {
                u64::try_from(block.values.len()).expect("slab voxel count exceeds u64") * 4
            } else {
                0
            },
            max_rank_observed: union_find_stats.max_rank_observed,
            uf_parent_state_bytes: union_find_stats.uf_parent_state_bytes,
            uf_rank_state_bytes: union_find_stats.uf_rank_state_bytes,
        },
    )
}

/// Streaming-oriented native-F32 slab processor that transfers ownership of
/// the decoded scalar buffer into the local union-find birth state.  The
/// ordered scalar values of unactivated voxels remain unchanged until their
/// filtration group is processed, so the input vector can safely double as
/// mutable component-birth storage after the radix order and boundary-face
/// snapshots have been formed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_slab_h0_persistence_f32_native_owned_profiled(
    slab_id: usize,
    block: F32ScalarBlock,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
) -> (SlabH0Summary<F32Key>, H0SlabPreparationProfile) {
    let shape = block.shape;
    let voxel_count = block.values.len();
    let order_start = Instant::now();
    let order = sorted_f32_indices(&block.values);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h0_persistence_values_owned(
        slab_id,
        shape,
        block.values,
        order,
        connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        pruning_cache,
        active_state,
        interface_state,
        uf_layout,
        sweep_diagnostics,
    );
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    (
        summary,
        H0SlabPreparationProfile {
            scalar_order_seconds,
            local_sweep_seconds,
            total_voxels: u64::try_from(voxel_count).expect("slab voxel count exceeds u64"),
            interior_fast_voxels: if matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
            {
                u64::try_from(shape[0].saturating_sub(2)).expect("width exceeds u64")
                    * u64::try_from(shape[1].saturating_sub(2)).expect("height exceeds u64")
                    * u64::try_from(shape[2].saturating_sub(2)).expect("depth exceeds u64")
            } else {
                0
            },
            pruning_mask_calls: pruning_stats.mask_calls,
            active_state_checks: pruning_stats.active_state_checks,
            active_neighbor_hits: pruning_stats.active_neighbor_hits,
            representative_visits: pruning_stats.representative_visits,
            pruning_cache_lookups: pruning_stats.cache_lookups,
            pruning_cache_hits: pruning_stats.cache_hits,
            pruning_cache_misses: pruning_stats.cache_misses,
            component_mask_computations: pruning_stats.component_mask_computations,
            union_attempts: union_find_stats.union_attempts,
            successful_unions: union_find_stats.successful_unions,
            same_root_unions: union_find_stats.same_root_unions,
            find_calls: union_find_stats.find_calls,
            find_parent_steps: union_find_stats.find_parent_steps,
            active_rechecks: union_find_stats.active_rechecks,
            active_recheck_failures: union_find_stats.active_recheck_failures,
            root_carry_attempts: union_find_stats.root_carry_attempts,
            avoided_find_calls: union_find_stats.avoided_find_calls,
            interface_rep_queries: union_find_stats.interface_rep_queries,
            interface_rep_writes: union_find_stats.interface_rep_writes,
            interface_forced_root_unions: union_find_stats.interface_forced_root_unions,
            interface_interface_unions: union_find_stats.interface_interface_unions,
            interface_state_bytes: if matches!(interface_state, InterfaceStateStrategy::Vector) {
                u64::try_from(voxel_count).expect("slab voxel count exceeds u64") * 4
            } else {
                0
            },
            max_rank_observed: union_find_stats.max_rank_observed,
            uf_parent_state_bytes: union_find_stats.uf_parent_state_bytes,
            uf_rank_state_bytes: union_find_stats.uf_rank_state_bytes,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_slab_h0_persistence_f32_native_direct_profiled<F>(
    slab_id: usize,
    block: F32ScalarBlock,
    connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    pruning_cache: H0PruningCacheStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    sweep_diagnostics: bool,
    emit: F,
) -> Result<(SlabH0Summary<F32Key>, H0SlabPreparationProfile)>
where
    F: FnMut(H0LocalStreamEvent<F32Key>) -> Result<()>,
{
    let shape = block.shape;
    let voxel_count = block.values.len();
    let order_start = Instant::now();
    let order = sorted_f32_indices(&block.values);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h0_persistence_values_owned_sink(
        slab_id,
        shape,
        block.values,
        order,
        connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        pruning_cache,
        active_state,
        interface_state,
        uf_layout,
        sweep_diagnostics,
        emit,
    )?;
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    Ok((
        summary,
        H0SlabPreparationProfile {
            scalar_order_seconds,
            local_sweep_seconds,
            total_voxels: u64::try_from(voxel_count).expect("slab voxel count exceeds u64"),
            interior_fast_voxels: if matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
            {
                u64::try_from(shape[0].saturating_sub(2)).expect("width exceeds u64")
                    * u64::try_from(shape[1].saturating_sub(2)).expect("height exceeds u64")
                    * u64::try_from(shape[2].saturating_sub(2)).expect("depth exceeds u64")
            } else {
                0
            },
            pruning_mask_calls: pruning_stats.mask_calls,
            active_state_checks: pruning_stats.active_state_checks,
            active_neighbor_hits: pruning_stats.active_neighbor_hits,
            representative_visits: pruning_stats.representative_visits,
            pruning_cache_lookups: pruning_stats.cache_lookups,
            pruning_cache_hits: pruning_stats.cache_hits,
            pruning_cache_misses: pruning_stats.cache_misses,
            component_mask_computations: pruning_stats.component_mask_computations,
            union_attempts: union_find_stats.union_attempts,
            successful_unions: union_find_stats.successful_unions,
            same_root_unions: union_find_stats.same_root_unions,
            find_calls: union_find_stats.find_calls,
            find_parent_steps: union_find_stats.find_parent_steps,
            active_rechecks: union_find_stats.active_rechecks,
            active_recheck_failures: union_find_stats.active_recheck_failures,
            root_carry_attempts: union_find_stats.root_carry_attempts,
            avoided_find_calls: union_find_stats.avoided_find_calls,
            interface_rep_queries: union_find_stats.interface_rep_queries,
            interface_rep_writes: union_find_stats.interface_rep_writes,
            interface_forced_root_unions: union_find_stats.interface_forced_root_unions,
            interface_interface_unions: union_find_stats.interface_interface_unions,
            interface_state_bytes: if matches!(interface_state, InterfaceStateStrategy::Vector) {
                u64::try_from(voxel_count).expect("slab voxel count exceeds u64") * 4
            } else {
                0
            },
            max_rank_observed: union_find_stats.max_rank_observed,
            uf_parent_state_bytes: union_find_stats.uf_parent_state_bytes,
            uf_rank_state_bytes: union_find_stats.uf_rank_state_bytes,
        },
    ))
}

fn extract_boundary_face_nodes_values<K: LocalScalarKey>(
    shape: [usize; 3],
    values_in: &[K],
    z_local: usize,
) -> BoundaryFaceNodes<K> {
    let width = shape[0];
    let height = shape[1];
    let slice_size = width * height;
    let mut node_ids = vec![0u32; slice_size];
    let mut values = vec![values_in[0]; slice_size];

    for y in 0..height {
        for x in 0..width {
            let face_idx = y * width + x;
            let idx = z_local * slice_size + face_idx;
            node_ids[face_idx] = face_node_id(z_local, shape[2], face_idx, slice_size);
            values[face_idx] = values_in[idx];
        }
    }

    BoundaryFaceNodes {
        width,
        height,
        node_ids,
        values,
    }
}

fn make_slab_ranges(depth: usize, slab_depth: usize) -> Vec<(usize, usize, usize)> {
    let mut ranges = Vec::new();
    let mut slab_id = 0usize;
    let mut z0 = 0usize;

    while z0 < depth {
        let z1 = z0.saturating_add(slab_depth).min(depth);
        ranges.push((slab_id, z0, z1));
        slab_id += 1;
        z0 = z1;
    }

    ranges
}

fn process_all_slabs_h0(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<SlabH0Summary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);

    println!(
        "Processing {} scalar H0-persistence slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabH0Summary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_h0_persistence_scalar(
                slab_id,
                &block,
                connectivity,
            ))
        })
        .collect();

    let mut summaries = parallel_results
        .into_iter()
        .collect::<Result<Vec<SlabH0Summary>>>()?;
    summaries.sort_by_key(|summary| summary.slab_id);
    Ok(summaries)
}

fn build_interface_offsets(summaries: &[SlabH0Summary]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(summaries.len() + 1);
    let mut current = 0u32;
    offsets.push(current);

    for summary in summaries {
        current = current
            .checked_add(summary.interface_node_count)
            .expect("too many scalar interface nodes for u32 global IDs");
        offsets.push(current);
    }

    offsets
}

fn global_node_id(offsets: &[u32], slab_id: usize, local_node: u32) -> u32 {
    offsets[slab_id] + local_node
}

#[derive(Debug, Clone, Copy)]
struct GlobalAttachEvent {
    node: u32,
    branch_birth: ScalarKey,
}

#[derive(Debug, Clone, Copy)]
enum GlobalEventKind {
    Attach(GlobalAttachEvent),
    Interface { a: u32, b: u32 },
    Cross { a: u32, b: u32 },
}

#[derive(Debug, Clone, Copy)]
struct GlobalEvent {
    value: ScalarKey,
    kind: GlobalEventKind,
}

impl GlobalEvent {
    fn priority(self) -> u8 {
        match self.kind {
            GlobalEventKind::Attach(_) => 0,
            GlobalEventKind::Interface { .. } => 1,
            GlobalEventKind::Cross { .. } => 2,
        }
    }
}

const GLOBAL_H0_PACKED_ROOT_BASE: u32 = u32::MAX - u8::MAX as u32;

#[derive(Debug)]
pub(crate) struct GlobalScalarPersistenceUnionFind<K = ScalarKey> {
    parent: Vec<u32>,
    rank: Option<Vec<u8>>,
    birth: Vec<K>,
    layout: GlobalH0UnionFindLayoutStrategy,
}

impl<K: LocalScalarKey> GlobalScalarPersistenceUnionFind<K> {
    pub(crate) fn new(birth: Vec<K>, layout: GlobalH0UnionFindLayoutStrategy) -> Self {
        match layout {
            GlobalH0UnionFindLayoutStrategy::ParentRank => {
                assert!(
                    birth.len() <= u32::MAX as usize,
                    "GlobalScalarPersistenceUnionFind uses u32 indices; too many interface nodes"
                );
                Self {
                    parent: (0..birth.len() as u32).collect(),
                    rank: Some(vec![0u8; birth.len()]),
                    birth,
                    layout,
                }
            }
            GlobalH0UnionFindLayoutStrategy::Packed => {
                assert!(
                    birth.len() <= GLOBAL_H0_PACKED_ROOT_BASE as usize,
                    "packed global H0 union-find reserves the top 256 u32 words for root ranks"
                );
                let len = birth.len();
                Self {
                    parent: vec![u32::MAX; len],
                    rank: None,
                    birth,
                    layout,
                }
            }
        }
    }

    #[inline]
    fn packed_root_word(rank: u8) -> u32 {
        u32::MAX - rank as u32
    }

    #[inline]
    fn is_root(&self, node: u32) -> bool {
        let word = self.parent[node as usize];
        match self.layout {
            GlobalH0UnionFindLayoutStrategy::ParentRank => word == node,
            GlobalH0UnionFindLayoutStrategy::Packed => word >= GLOBAL_H0_PACKED_ROOT_BASE,
        }
    }

    #[inline]
    pub(crate) fn find(&mut self, mut node: u32) -> u32 {
        while !self.is_root(node) {
            let parent = self.parent[node as usize];
            debug_assert!(
                parent < GLOBAL_H0_PACKED_ROOT_BASE
                    || matches!(self.layout, GlobalH0UnionFindLayoutStrategy::ParentRank)
            );
            if !self.is_root(parent) {
                self.parent[node as usize] = self.parent[parent as usize];
            }
            node = parent;
        }
        node
    }

    #[inline]
    fn root_rank(&self, root: u32) -> u8 {
        debug_assert!(self.is_root(root));
        match self.layout {
            GlobalH0UnionFindLayoutStrategy::ParentRank => self
                .rank
                .as_ref()
                .expect("parent-rank global H0 layout requires rank vector")[root as usize],
            GlobalH0UnionFindLayoutStrategy::Packed => {
                (u32::MAX - self.parent[root as usize]) as u8
            }
        }
    }

    #[inline]
    fn set_root_rank(&mut self, root: u32, rank: u8) {
        debug_assert!(self.is_root(root));
        match self.layout {
            GlobalH0UnionFindLayoutStrategy::ParentRank => {
                self.rank
                    .as_mut()
                    .expect("parent-rank global H0 layout requires rank vector")
                    [root as usize] = rank;
            }
            GlobalH0UnionFindLayoutStrategy::Packed => {
                self.parent[root as usize] = Self::packed_root_word(rank);
            }
        }
    }

    #[inline]
    fn link_root_under(&mut self, child_root: u32, parent_root: u32) {
        debug_assert!(self.is_root(child_root));
        debug_assert!(self.is_root(parent_root));
        debug_assert!(
            parent_root < GLOBAL_H0_PACKED_ROOT_BASE
                || matches!(self.layout, GlobalH0UnionFindLayoutStrategy::ParentRank)
        );
        self.parent[child_root as usize] = parent_root;
    }

    pub(crate) fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        death: K,
    ) -> Option<FinitePair<K>> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);

        if root_a == root_b {
            return None;
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let surviving_birth = birth_a.min(birth_b);
        let pair = FinitePair {
            birth: birth_a.max(birth_b),
            death,
        };

        let rank_a = self.root_rank(root_a);
        let rank_b = self.root_rank(root_b);
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.link_root_under(root_b, root_a);
        if rank_a == rank_b {
            self.set_root_rank(
                root_a,
                rank_a
                    .checked_add(1)
                    .expect("global H0 union-find rank exceeded u8"),
            );
        }
        self.birth[root_a as usize] = surviving_birth;

        Some(pair)
    }

    pub(crate) fn attach_branch(
        &mut self,
        interface_node: u32,
        branch_birth: K,
        death: K,
    ) -> FinitePair<K> {
        let root = self.find(interface_node);
        let root_birth = self.birth[root as usize];
        self.birth[root as usize] = root_birth.min(branch_birth);

        FinitePair {
            birth: root_birth.max(branch_birth),
            death,
        }
    }

    pub(crate) fn essential_births(&mut self) -> Vec<K> {
        let mut births = Vec::new();
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 {
                births.push(self.birth[node]);
            }
        }
        births
    }

    pub(crate) fn capacity_bytes(&self) -> (u64, u64, u64) {
        let parent = self.parent.capacity() as u64 * core::mem::size_of::<u32>() as u64;
        let rank = self
            .rank
            .as_ref()
            .map(|rank| rank.capacity() as u64 * core::mem::size_of::<u8>() as u64)
            .unwrap_or(0);
        let birth = self.birth.capacity() as u64 * core::mem::size_of::<K>() as u64;
        (parent, rank, birth)
    }
}

fn build_global_interface_births(summaries: &[SlabH0Summary], offsets: &[u32]) -> Vec<ScalarKey> {
    let total = *offsets.last().unwrap_or(&0) as usize;
    let mut births = vec![ScalarKey::ZERO; total];

    for summary in summaries {
        for (local_node, &value) in summary
            .z_min_face
            .node_ids
            .iter()
            .zip(summary.z_min_face.values.iter())
        {
            let global = global_node_id(offsets, summary.slab_id, *local_node) as usize;
            births[global] = value;
        }

        for (local_node, &value) in summary
            .z_max_face
            .node_ids
            .iter()
            .zip(summary.z_max_face.values.iter())
        {
            let global = global_node_id(offsets, summary.slab_id, *local_node) as usize;
            births[global] = value;
        }
    }

    births
}

fn push_finite_interval(intervals: &mut Vec<ScalarH0Interval>, pair: FinitePair) {
    if pair.birth < pair.death {
        intervals.push(ScalarH0Interval {
            birth: pair.birth,
            death: Some(pair.death),
        });
    }
}

fn reduce_h0_persistence(
    summaries: &[SlabH0Summary],
    connectivity: Connectivity,
) -> Result<Vec<ScalarH0Interval>> {
    let offsets = build_interface_offsets(summaries);
    let births = build_global_interface_births(summaries, &offsets);
    let mut global_union_find =
        GlobalScalarPersistenceUnionFind::new(births, GlobalH0UnionFindLayoutStrategy::ParentRank);
    let mut intervals = Vec::new();
    let mut events = Vec::new();

    for summary in summaries {
        for &pair in &summary.finalized_pairs {
            push_finite_interval(&mut intervals, pair);
        }

        for event in &summary.attach_events {
            events.push(GlobalEvent {
                value: event.value,
                kind: GlobalEventKind::Attach(GlobalAttachEvent {
                    node: global_node_id(&offsets, summary.slab_id, event.interface_node),
                    branch_birth: event.branch_birth,
                }),
            });
        }

        for event in &summary.interface_merge_events {
            events.push(GlobalEvent {
                value: event.value,
                kind: GlobalEventKind::Interface {
                    a: global_node_id(&offsets, summary.slab_id, event.a),
                    b: global_node_id(&offsets, summary.slab_id, event.b),
                },
            });
        }
    }

    let mut candidate_cross_edges = 0u64;
    let mut retained_cross_edges = 0u64;

    for slab_pair in summaries.windows(2) {
        let left = &slab_pair[0];
        let right = &slab_pair[1];
        let left_face = &left.z_max_face;
        let right_face = &right.z_min_face;
        assert_eq!(left_face.width, right_face.width);
        assert_eq!(left_face.height, right_face.height);

        let stats = sparsify_scalar_sublevel_interface(
            &left_face.values,
            &right_face.values,
            left_face.width,
            left_face.height,
            connectivity,
            |edge| {
                let left_index = edge.left_face_index as usize;
                let right_index = edge.right_face_index as usize;
                events.push(GlobalEvent {
                    value: edge.value,
                    kind: GlobalEventKind::Cross {
                        a: global_node_id(&offsets, left.slab_id, left_face.node_ids[left_index]),
                        b: global_node_id(
                            &offsets,
                            right.slab_id,
                            right_face.node_ids[right_index],
                        ),
                    },
                });
                Ok(())
            },
        )?;

        candidate_cross_edges += stats.candidate_edges;
        retained_cross_edges += stats.retained_edges;
    }

    println!(
        "Scalar H0 interface sparsification retained {retained_cross_edges} of {candidate_cross_edges} candidate edges"
    );

    events.sort_by(|a, b| {
        let value_order = a.value.cmp(&b.value);
        if value_order == Ordering::Equal {
            a.priority().cmp(&b.priority())
        } else {
            value_order
        }
    });

    for event in events {
        let pair = match event.kind {
            GlobalEventKind::Attach(attach) => {
                Some(global_union_find.attach_branch(attach.node, attach.branch_birth, event.value))
            }
            GlobalEventKind::Interface { a, b } | GlobalEventKind::Cross { a, b } => {
                global_union_find.union_with_persistence(a, b, event.value)
            }
        };

        if let Some(pair) = pair {
            push_finite_interval(&mut intervals, pair);
        }
    }

    for birth in global_union_find.essential_births() {
        intervals.push(ScalarH0Interval { birth, death: None });
    }

    intervals.sort_by_key(|interval| (interval.birth, interval.death));
    Ok(intervals)
}

#[derive(Debug)]
struct HierarchicalH0Summary<K = ScalarKey> {
    z0: usize,
    z1: usize,
    summary: SlabH0Summary<K>,
}

#[derive(Debug, Clone, Copy)]
enum HierarchicalH0EventKind<K> {
    Attach { node: u32, branch_birth: K },
    Interface { a: u32, b: u32 },
    Cross { a: u32, b: u32 },
}

#[derive(Debug, Clone, Copy)]
struct HierarchicalH0Event<K> {
    value: K,
    kind: HierarchicalH0EventKind<K>,
}

impl<K> HierarchicalH0Event<K> {
    fn priority(self) -> u8 {
        match self.kind {
            HierarchicalH0EventKind::Attach { .. } => 0,
            HierarchicalH0EventKind::Interface { .. } => 1,
            HierarchicalH0EventKind::Cross { .. } => 2,
        }
    }
}

#[derive(Debug)]
struct HierarchicalH0UnionFind<K> {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<K>,
    terminal_rep: Vec<u32>,
}

impl<K: LocalScalarKey> HierarchicalH0UnionFind<K> {
    fn new(birth: Vec<K>, terminal_rep: Vec<u32>) -> Self {
        assert_eq!(birth.len(), terminal_rep.len());
        assert!(birth.len() <= u32::MAX as usize);
        Self {
            parent: (0..birth.len() as u32).collect(),
            rank: vec![0; birth.len()],
            birth,
            terminal_rep,
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let parent = self.parent[x as usize];
            let grandparent = self.parent[parent as usize];
            self.parent[x as usize] = grandparent;
            x = parent;
        }
        x
    }

    fn link_roots(&mut self, mut root_a: u32, mut root_b: u32) -> u32 {
        let rank_a = self.rank[root_a as usize];
        let rank_b = self.rank[root_b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b as usize] = root_a;
        if rank_a == rank_b {
            self.rank[root_a as usize] = rank_a.saturating_add(1);
        }
        root_a
    }

    fn union_with_summary(
        &mut self,
        a: u32,
        b: u32,
        death: K,
    ) -> Option<LocalPersistenceAction<K>> {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let rep_a = self.terminal_rep[root_a as usize];
        let rep_b = self.terminal_rep[root_b as usize];

        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => LocalPersistenceAction::FinalPair(FinitePair {
                birth: birth_a.max(birth_b),
                death,
            }),
            (true, false) => LocalPersistenceAction::Attach(AttachEvent {
                value: death,
                interface_node: rep_a,
                branch_birth: birth_b,
            }),
            (false, true) => LocalPersistenceAction::Attach(AttachEvent {
                value: death,
                interface_node: rep_b,
                branch_birth: birth_a,
            }),
            (true, true) => LocalPersistenceAction::InterfaceMerge(InterfaceMergeEvent {
                value: death,
                a: rep_a,
                b: rep_b,
            }),
        };

        let surviving_birth = birth_a.min(birth_b);
        let surviving_rep = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };
        let new_root = self.link_roots(root_a, root_b);
        self.birth[new_root as usize] = surviving_birth;
        self.terminal_rep[new_root as usize] = surviving_rep;
        Some(action)
    }

    fn attach_branch_with_summary(
        &mut self,
        node: u32,
        branch_birth: K,
        death: K,
    ) -> LocalPersistenceAction<K> {
        let root = self.find(node);
        let root_birth = self.birth[root as usize];
        let rep = self.terminal_rep[root as usize];
        self.birth[root as usize] = root_birth.min(branch_birth);
        if rep == NO_INTERFACE_REP {
            LocalPersistenceAction::FinalPair(FinitePair {
                birth: root_birth.max(branch_birth),
                death,
            })
        } else {
            LocalPersistenceAction::Attach(AttachEvent {
                value: death,
                interface_node: rep,
                branch_birth,
            })
        }
    }

    fn count_internal_roots(&mut self) -> usize {
        let mut count = 0usize;
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 && self.terminal_rep[node] == NO_INTERFACE_REP {
                count += 1;
            }
        }
        count
    }
}

fn push_summary_action<K: LocalScalarKey>(
    action: LocalPersistenceAction<K>,
    finalized_pairs: &mut Vec<FinitePair<K>>,
    attach_events: &mut Vec<AttachEvent<K>>,
    interface_merge_events: &mut Vec<InterfaceMergeEvent<K>>,
) {
    match action {
        LocalPersistenceAction::FinalPair(pair) => {
            if pair.birth < pair.death {
                finalized_pairs.push(pair);
            }
        }
        LocalPersistenceAction::Attach(event) => attach_events.push(event),
        LocalPersistenceAction::InterfaceMerge(event) => interface_merge_events.push(event),
    }
}

fn combine_h0_summaries<K>(
    mut left: HierarchicalH0Summary<K>,
    mut right: HierarchicalH0Summary<K>,
    connectivity: Connectivity,
) -> Result<HierarchicalH0Summary<K>>
where
    K: LocalScalarKey + crate::scalar_order::RadixScalarKey,
{
    if left.z1 != right.z0 {
        bail!(
            "hierarchical H0 summaries are not adjacent: left z={}..{}, right z={}..{}",
            left.z0,
            left.z1,
            right.z0,
            right.z1
        );
    }
    let width = left.summary.z_min_face.width;
    let height = left.summary.z_min_face.height;
    if right.summary.z_min_face.width != width || right.summary.z_min_face.height != height {
        bail!("hierarchical H0 summary face dimensions differ");
    }
    let face_size = width.checked_mul(height).expect("face size overflow");
    let left_count = left.summary.interface_node_count as usize;
    let right_count = right.summary.interface_node_count as usize;
    let total_nodes = left_count
        .checked_add(right_count)
        .ok_or_else(|| anyhow::anyhow!("hierarchical H0 pair node count overflow"))?;
    if total_nodes > u32::MAX as usize {
        bail!("hierarchical H0 pair exceeds u32 local node capacity");
    }

    let seed = *left
        .summary
        .z_min_face
        .values
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty hierarchical H0 boundary face"))?;
    let mut births = vec![seed; total_nodes];
    let mut terminals = vec![NO_INTERFACE_REP; total_nodes];

    let left_offset = 0u32;
    let right_offset = u32::try_from(left_count).expect("left interface node count exceeds u32");

    let mut assign_face = |summary: &SlabH0Summary<K>, offset: u32| {
        for (&node, &value) in summary
            .z_min_face
            .node_ids
            .iter()
            .zip(summary.z_min_face.values.iter())
        {
            births[(offset + node) as usize] = value;
        }
        for (&node, &value) in summary
            .z_max_face
            .node_ids
            .iter()
            .zip(summary.z_max_face.values.iter())
        {
            births[(offset + node) as usize] = value;
        }
    };
    assign_face(&left.summary, left_offset);
    assign_face(&right.summary, right_offset);

    let parent_depth = right.z1 - left.z0;
    for face_index in 0..face_size {
        let child_node = left.summary.z_min_face.node_ids[face_index];
        let parent_node = face_node_id(0, parent_depth, face_index, face_size);
        terminals[(left_offset + child_node) as usize] = parent_node;
    }
    for face_index in 0..face_size {
        let child_node = right.summary.z_max_face.node_ids[face_index];
        let parent_node = face_node_id(parent_depth - 1, parent_depth, face_index, face_size);
        terminals[(right_offset + child_node) as usize] = parent_node;
    }

    let mut events = Vec::new();
    for event in &left.summary.attach_events {
        events.push(HierarchicalH0Event {
            value: event.value,
            kind: HierarchicalH0EventKind::Attach {
                node: left_offset + event.interface_node,
                branch_birth: event.branch_birth,
            },
        });
    }
    for event in &right.summary.attach_events {
        events.push(HierarchicalH0Event {
            value: event.value,
            kind: HierarchicalH0EventKind::Attach {
                node: right_offset + event.interface_node,
                branch_birth: event.branch_birth,
            },
        });
    }
    for event in &left.summary.interface_merge_events {
        events.push(HierarchicalH0Event {
            value: event.value,
            kind: HierarchicalH0EventKind::Interface {
                a: left_offset + event.a,
                b: left_offset + event.b,
            },
        });
    }
    for event in &right.summary.interface_merge_events {
        events.push(HierarchicalH0Event {
            value: event.value,
            kind: HierarchicalH0EventKind::Interface {
                a: right_offset + event.a,
                b: right_offset + event.b,
            },
        });
    }

    let cross_stats = sparsify_scalar_sublevel_interface(
        &left.summary.z_max_face.values,
        &right.summary.z_min_face.values,
        width,
        height,
        connectivity,
        |edge| {
            let li = edge.left_face_index as usize;
            let ri = edge.right_face_index as usize;
            events.push(HierarchicalH0Event {
                value: edge.value,
                kind: HierarchicalH0EventKind::Cross {
                    a: left_offset + left.summary.z_max_face.node_ids[li],
                    b: right_offset + right.summary.z_min_face.node_ids[ri],
                },
            });
            Ok(())
        },
    )?;

    events.sort_by(|a, b| {
        let value_order = a.value.cmp(&b.value);
        if value_order == Ordering::Equal {
            a.priority().cmp(&b.priority())
        } else {
            value_order
        }
    });

    let mut uf = HierarchicalH0UnionFind::new(births, terminals);
    let mut finalized_pairs = Vec::new();
    finalized_pairs.append(&mut left.summary.finalized_pairs);
    finalized_pairs.append(&mut right.summary.finalized_pairs);
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();

    for event in events {
        let action = match event.kind {
            HierarchicalH0EventKind::Attach { node, branch_birth } => {
                Some(uf.attach_branch_with_summary(node, branch_birth, event.value))
            }
            HierarchicalH0EventKind::Interface { a, b }
            | HierarchicalH0EventKind::Cross { a, b } => uf.union_with_summary(a, b, event.value),
        };
        if let Some(action) = action {
            push_summary_action(
                action,
                &mut finalized_pairs,
                &mut attach_events,
                &mut interface_merge_events,
            );
        }
    }

    let internal_roots = uf.count_internal_roots();
    if internal_roots != 0 {
        bail!(
            "hierarchical H0 composition left {internal_roots} components without an outer-face representative"
        );
    }

    let lower_values = left.summary.z_min_face.values;
    let upper_values = right.summary.z_max_face.values;
    let mut lower_node_ids = Vec::with_capacity(face_size);
    let mut upper_node_ids = Vec::with_capacity(face_size);
    for face_index in 0..face_size {
        lower_node_ids.push(face_node_id(0, parent_depth, face_index, face_size));
        upper_node_ids.push(face_node_id(
            parent_depth - 1,
            parent_depth,
            face_index,
            face_size,
        ));
    }
    let interface_node_count = u32::try_from(slab_interface_node_count(face_size, parent_depth))
        .expect("hierarchical parent interface node count exceeds u32");

    println!(
        "PROFILE_H0_HIER_COMBINE z0={} z1={} pair_nodes={} parent_interface_nodes={} events={} cross_retained={} cross_candidates={}",
        left.z0,
        right.z1,
        total_nodes,
        interface_node_count,
        attach_events.len() + interface_merge_events.len() + finalized_pairs.len(),
        cross_stats.retained_edges,
        cross_stats.candidate_edges,
    );

    Ok(HierarchicalH0Summary {
        z0: left.z0,
        z1: right.z1,
        summary: SlabH0Summary {
            slab_id: 0,
            finalized_pairs,
            attach_events,
            interface_merge_events,
            interface_node_count,
            z_min_face: BoundaryFaceNodes {
                width,
                height,
                node_ids: lower_node_ids,
                values: lower_values,
            },
            z_max_face: BoundaryFaceNodes {
                width,
                height,
                node_ids: upper_node_ids,
                values: upper_values,
            },
        },
    })
}

pub fn compute_h0_persistence_scalar_hierarchical_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<ScalarH0Interval>> {
    let start = Instant::now();
    let ranges = make_slab_ranges(volume.depth, slab_depth);
    println!(
        "Computing online hierarchical scalar H0 persistence from {} leaf slabs...",
        ranges.len()
    );

    // Binary-counter fan-in: slot L contains at most one completed block made
    // from 2^L consecutive leaf slabs.  When another block reaches that level,
    // compose them immediately and carry the parent to L+1.  Thus the number
    // of retained summaries is O(log(number_of_slabs)), not O(number_of_slabs).
    let mut slots: Vec<Option<HierarchicalH0Summary<ScalarKey>>> = Vec::new();
    let mut combine_count = 0usize;
    let mut max_pair_nodes = 0usize;
    let mut max_live_summaries = 0usize;
    for (slab_id, z0, z1) in ranges {
        let block = volume.read_z_slab(z0, z1)?;
        let mut summary = process_slab_h0_persistence_scalar(slab_id, &block, connectivity);
        summary.slab_id = 0;
        let mut current = HierarchicalH0Summary { z0, z1, summary };
        let mut level = 0usize;

        loop {
            if level == slots.len() {
                slots.push(Some(current));
                break;
            }
            if let Some(left) = slots[level].take() {
                max_pair_nodes = max_pair_nodes.max(
                    left.summary.interface_node_count as usize
                        + current.summary.interface_node_count as usize,
                );
                println!(
                    "Hierarchical H0 online fan-in level {level}: z={}..{} + z={}..{}",
                    left.z0, left.z1, current.z0, current.z1
                );
                current = combine_h0_summaries(left, current, connectivity)?;
                combine_count += 1;
                level += 1;
            } else {
                slots[level] = Some(current);
                break;
            }
        }
        max_live_summaries =
            max_live_summaries.max(slots.iter().filter(|slot| slot.is_some()).count());
    }

    let mut remaining: Vec<_> = slots.into_iter().flatten().collect();
    if remaining.is_empty() {
        bail!("hierarchical H0 received an empty volume");
    }
    remaining.sort_by_key(|summary| summary.z0);
    let mut final_summary = remaining.remove(0);
    for right in remaining {
        max_pair_nodes = max_pair_nodes.max(
            final_summary.summary.interface_node_count as usize
                + right.summary.interface_node_count as usize,
        );
        final_summary = combine_h0_summaries(final_summary, right, connectivity)?;
        combine_count += 1;
    }

    let intervals = reduce_h0_persistence(&[final_summary.summary], connectivity)?;
    println!(
        "PROFILE_H0_HIERARCHY leaf_slabs={} combines={} max_live_summaries={} max_pair_nodes={} final_interface_nodes={}",
        make_slab_ranges(volume.depth, slab_depth).len(),
        combine_count,
        max_live_summaries,
        max_pair_nodes,
        slab_interface_node_count(volume.width * volume.height, volume.depth),
    );
    println!(
        "Hierarchical scalar H0 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(intervals)
}

pub fn compute_h0_persistence_scalar_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<ScalarH0Interval>> {
    let start = Instant::now();
    println!("Computing in-memory slabwise scalar H0 persistence...");
    let summaries = process_all_slabs_h0(volume, slab_depth, connectivity)?;
    println!("Processed {} scalar H0 slab summaries", summaries.len());
    let intervals = reduce_h0_persistence(&summaries, connectivity)?;
    println!(
        "Scalar H0 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(intervals)
}

pub fn write_h0_scalar_persistence_csv(path: &Path, intervals: &[ScalarH0Interval]) -> Result<()> {
    let mut file = AtomicOutput::create(path)
        .with_context(|| format!("could not create scalar H0 output {path:?}"))?;
    writeln!(file, "birth,death")?;
    for interval in intervals {
        match interval.death {
            Some(death) => writeln!(file, "{},{}", interval.birth, death)?,
            None => writeln!(file, "{},inf", interval.birth)?,
        }
    }
    file.commit()
}

pub fn betti0_curve_from_scalar_h0_intervals(
    intervals: &[ScalarH0Interval],
) -> Vec<(ScalarKey, i64)> {
    let mut delta = BTreeMap::<ScalarKey, i64>::new();
    for interval in intervals {
        *delta.entry(interval.birth).or_default() += 1;
        if let Some(death) = interval.death {
            *delta.entry(death).or_default() -= 1;
        }
    }

    let mut beta0 = 0i64;
    let mut sparse = Vec::new();
    let mut previous: Option<i64> = None;
    for (value, change) in delta {
        beta0 += change;
        if previous != Some(beta0) {
            sparse.push((value, beta0));
            previous = Some(beta0);
        }
    }
    sparse
}

pub fn write_scalar_betti0_curve_csv(path: &Path, curve: &[(ScalarKey, i64)]) -> Result<()> {
    let mut file = AtomicOutput::create(path)
        .with_context(|| format!("could not create scalar Betti-0 output {path:?}"))?;
    writeln!(file, "threshold,betti0")?;
    for &(threshold, beta0) in curve {
        writeln!(file, "{threshold},{beta0}")?;
    }
    file.commit()
}

pub fn compute_h0_scalar_batch(
    root: &Path,
    output_root: &Path,
    slab_depth: usize,
    connectivity: Connectivity,
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
            "=== Scalar H0 dataset {} of {}: {} ===",
            index + 1,
            directories.len(),
            directory.display()
        );

        let volume = ScalarTiffStackReader::open(directory)?;
        print_scalar_volume_info(&volume);
        let intervals = compute_h0_persistence_scalar_zslabs(&volume, slab_depth, connectivity)?;

        write_h0_scalar_persistence_csv(
            &dataset_output.join("h0_persistence_scalar.csv"),
            &intervals,
        )?;
        let curve = betti0_curve_from_scalar_h0_intervals(&intervals);
        write_scalar_betti0_curve_csv(
            &dataset_output.join("h0_reconstructed_betti0_scalar_curve.csv"),
            &curve,
        )?;
    }

    Ok(directories.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widen_f32_summary(summary: SlabH0Summary<F32Key>) -> SlabH0Summary {
        let widen_face = |face: BoundaryFaceNodes<F32Key>| BoundaryFaceNodes {
            width: face.width,
            height: face.height,
            node_ids: face.node_ids,
            values: face.values.into_iter().map(|value| value.widen()).collect(),
        };
        SlabH0Summary {
            slab_id: summary.slab_id,
            finalized_pairs: summary
                .finalized_pairs
                .into_iter()
                .map(|pair| FinitePair {
                    birth: pair.birth.widen(),
                    death: pair.death.widen(),
                })
                .collect(),
            attach_events: summary
                .attach_events
                .into_iter()
                .map(|event| AttachEvent {
                    value: event.value.widen(),
                    interface_node: event.interface_node,
                    branch_birth: event.branch_birth.widen(),
                })
                .collect(),
            interface_merge_events: summary
                .interface_merge_events
                .into_iter()
                .map(|event| InterfaceMergeEvent {
                    value: event.value.widen(),
                    a: event.a,
                    b: event.b,
                })
                .collect(),
            interface_node_count: summary.interface_node_count,
            z_min_face: widen_face(summary.z_min_face),
            z_max_face: widen_face(summary.z_max_face),
        }
    }

    #[test]
    fn packed_global_h0_union_find_matches_parent_rank() {
        let births = vec![
            ScalarKey::from_u16(1),
            ScalarKey::from_u16(4),
            ScalarKey::from_u16(2),
            ScalarKey::from_u16(6),
        ];
        let mut reference = GlobalScalarPersistenceUnionFind::new(
            births.clone(),
            GlobalH0UnionFindLayoutStrategy::ParentRank,
        );
        let mut packed =
            GlobalScalarPersistenceUnionFind::new(births, GlobalH0UnionFindLayoutStrategy::Packed);

        let mut reference_pairs = Vec::new();
        let mut packed_pairs = Vec::new();
        for (a, b, death) in [
            (0, 1, ScalarKey::from_u16(5)),
            (2, 3, ScalarKey::from_u16(7)),
            (0, 2, ScalarKey::from_u16(8)),
        ] {
            reference_pairs.push(reference.union_with_persistence(a, b, death));
            packed_pairs.push(packed.union_with_persistence(a, b, death));
        }
        reference_pairs.push(Some(reference.attach_branch(
            3,
            ScalarKey::from_u16(0),
            ScalarKey::from_u16(9),
        )));
        packed_pairs.push(Some(packed.attach_branch(
            3,
            ScalarKey::from_u16(0),
            ScalarKey::from_u16(9),
        )));

        assert_eq!(packed_pairs, reference_pairs);
        assert_eq!(packed.essential_births(), reference.essential_births());
        let (_, reference_rank, _) = reference.capacity_bytes();
        let (_, packed_rank, _) = packed.capacity_bytes();
        assert!(reference_rank > 0);
        assert_eq!(packed_rank, 0);
    }

    #[test]
    fn native_f32_local_h0_matches_legacy_wide_path() {
        let source = [
            0.0f32, 2.0, 1.0, 4.0, 3.0, -1.0, 1.0, 5.0, 2.0, 2.0, -1.0, 6.0, 9.0, 3.0, 3.0, 0.5,
        ];
        let wide = ScalarBlock {
            z0: 0,
            shape: [4, 2, 2],
            values: source
                .iter()
                .copied()
                .map(|value| ScalarKey::from_f32(value).unwrap())
                .collect(),
            pixel_type: crate::io_scalar::ScalarPixelType::F32,
        };
        let native = F32ScalarBlock {
            z0: 0,
            shape: [4, 2, 2],
            values: source
                .iter()
                .copied()
                .map(|value| crate::scalar::F32Key::from_f32(value).unwrap())
                .collect(),
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let wide_summary = process_slab_h0_persistence_scalar(0, &wide, connectivity);
            let native_summary = process_slab_h0_persistence_f32_native(0, &native, connectivity);
            let wide_intervals = reduce_h0_persistence(&[wide_summary], connectivity).unwrap();
            let native_intervals =
                reduce_h0_persistence(&[widen_f32_summary(native_summary)], connectivity).unwrap();
            assert_eq!(native_intervals, wide_intervals);
        }
    }

    #[test]
    fn equal_value_pruning_does_not_create_a_spurious_essential_component() {
        let mut values = vec![ScalarKey::from_u16(5); 4 * 4 * 4];
        let index = |x: usize, y: usize, z: usize| z * 16 + y * 4 + x;
        values[index(0, 1, 1)] = ScalarKey::from_u16(1);
        values[index(1, 1, 1)] = ScalarKey::from_u16(3);
        values[index(2, 1, 1)] = ScalarKey::from_u16(2);
        values[index(3, 1, 1)] = ScalarKey::from_u16(1);

        let block = ScalarBlock {
            z0: 0,
            shape: [4, 4, 4],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        let summary = process_slab_h0_persistence_scalar(0, &block, Connectivity::TwentySix);
        let intervals = reduce_h0_persistence(&[summary], Connectivity::TwentySix).unwrap();

        assert_eq!(
            intervals,
            vec![
                ScalarH0Interval {
                    birth: ScalarKey::from_u16(1),
                    death: None,
                },
                ScalarH0Interval {
                    birth: ScalarKey::from_u16(1),
                    death: Some(ScalarKey::from_u16(3)),
                },
            ]
        );
    }
    #[test]
    fn root_carrying_h0_matches_conventional_and_uses_fewer_finds() {
        let values = (0..64)
            .map(|index| ScalarKey::from_u16(((index * 17 + index / 3) % 11) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [4, 4, 4],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let (conventional, conventional_profile) = process_slab_h0_persistence_scalar_profiled(
                0,
                &block,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::Conventional,
                H0PruningCacheStrategy::Entries64K,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                true,
            );
            let (root_carrying, root_profile) = process_slab_h0_persistence_scalar_profiled(
                0,
                &block,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::RootCarrying,
                H0PruningCacheStrategy::Entries64K,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                true,
            );
            let conventional_intervals =
                reduce_h0_persistence(&[conventional], connectivity).unwrap();
            let root_intervals = reduce_h0_persistence(&[root_carrying], connectivity).unwrap();
            assert_eq!(root_intervals, conventional_intervals);
            assert_eq!(
                root_profile.root_carry_attempts,
                root_profile.union_attempts
            );
            assert_eq!(root_profile.avoided_find_calls, root_profile.union_attempts);
            assert!(root_profile.find_calls < conventional_profile.find_calls);
        }
    }

    #[test]
    fn h0_pruning_cache_modes_preserve_local_persistence() {
        let values = (0..125)
            .map(|index| ScalarKey::from_u16(((index * 29 + index / 5) % 17) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [5, 5, 5],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        let cache_modes = [
            H0PruningCacheStrategy::Off,
            H0PruningCacheStrategy::Entries4K,
            H0PruningCacheStrategy::Entries16K,
            H0PruningCacheStrategy::Entries64K,
            H0PruningCacheStrategy::Entries256K,
        ];
        let mut reference = None;
        for cache in cache_modes {
            let (summary, profile) = process_slab_h0_persistence_scalar_profiled(
                0,
                &block,
                Connectivity::TwentySix,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::RootCarrying,
                cache,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                true,
            );
            let intervals = reduce_h0_persistence(&[summary], Connectivity::TwentySix).unwrap();
            if let Some(expected) = reference.as_ref() {
                assert_eq!(&intervals, expected);
            } else {
                reference = Some(intervals);
            }
            if matches!(cache, H0PruningCacheStrategy::Off) {
                assert_eq!(profile.pruning_cache_lookups, 0);
                assert_eq!(profile.pruning_cache_hits, 0);
                assert_eq!(profile.pruning_cache_misses, 0);
            } else {
                assert_eq!(
                    profile.pruning_cache_lookups,
                    profile.pruning_cache_hits + profile.pruning_cache_misses
                );
                assert_eq!(
                    profile.component_mask_computations,
                    profile.pruning_cache_misses
                );
            }
        }
    }

    #[test]
    fn parent_sentinel_h0_matches_separate_active_state() {
        let values = (0..125)
            .map(|index| ScalarKey::from_u16(((index * 31 + index / 7) % 19) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [5, 5, 5],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let mut reference = None;
            let mut reference_checks = None;
            for active_state in [
                ActiveStateStrategy::Separate,
                ActiveStateStrategy::ParentSentinel,
            ] {
                let (summary, profile) = process_slab_h0_persistence_scalar_profiled(
                    0,
                    &block,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    H0PruningCacheStrategy::Entries64K,
                    active_state,
                    InterfaceStateStrategy::Vector,
                    UnionFindLayoutStrategy::ParentRank,
                    true,
                );
                let intervals = reduce_h0_persistence(&[summary], connectivity).unwrap();
                if let Some(expected) = reference.as_ref() {
                    assert_eq!(&intervals, expected);
                    assert_eq!(Some(profile.active_state_checks), reference_checks);
                } else {
                    reference_checks = Some(profile.active_state_checks);
                    reference = Some(intervals);
                }
                assert_eq!(profile.active_recheck_failures, 0);
            }
        }
    }

    #[test]
    fn root_invariant_h0_matches_interface_vector() {
        let values = (0..216)
            .map(|index| ScalarKey::from_u16(((index * 41 + index / 11) % 29) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [6, 6, 6],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let mut reference = None;
            for interface_state in [
                InterfaceStateStrategy::Vector,
                InterfaceStateStrategy::RootInvariant,
            ] {
                let (summary, profile) = process_slab_h0_persistence_scalar_profiled(
                    0,
                    &block,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    H0PruningCacheStrategy::Entries64K,
                    ActiveStateStrategy::Separate,
                    interface_state,
                    UnionFindLayoutStrategy::ParentRank,
                    true,
                );
                let intervals = reduce_h0_persistence(&[summary], connectivity).unwrap();
                if let Some(expected) = reference.as_ref() {
                    assert_eq!(&intervals, expected);
                } else {
                    reference = Some(intervals);
                }
                match interface_state {
                    InterfaceStateStrategy::Vector => {
                        assert_eq!(profile.interface_state_bytes, 216 * 4);
                        assert!(profile.interface_rep_writes > 0);
                    }
                    InterfaceStateStrategy::RootInvariant => {
                        assert_eq!(profile.interface_state_bytes, 0);
                        assert_eq!(profile.interface_rep_writes, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn hierarchical_h0_pair_composition_matches_unsplit_block() {
        let width = 4usize;
        let height = 4usize;
        let depth = 4usize;
        let slice_size = width * height;
        let values: Vec<ScalarKey> = (0..width * height * depth)
            .map(|index| {
                ScalarKey::from_u16(((index * 53 + index / 3 + (index % 7) * 11) % 41) as u16)
            })
            .collect();
        let whole = ScalarBlock {
            z0: 0,
            shape: [width, height, depth],
            values: values.clone(),
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        let left_block = ScalarBlock {
            z0: 0,
            shape: [width, height, 2],
            values: values[..2 * slice_size].to_vec(),
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        let right_block = ScalarBlock {
            z0: 2,
            shape: [width, height, 2],
            values: values[2 * slice_size..].to_vec(),
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let whole_summary = process_slab_h0_persistence_scalar(0, &whole, connectivity);
            let expected = reduce_h0_persistence(&[whole_summary], connectivity).unwrap();

            let mut left = process_slab_h0_persistence_scalar(0, &left_block, connectivity);
            let mut right = process_slab_h0_persistence_scalar(0, &right_block, connectivity);
            left.slab_id = 0;
            right.slab_id = 0;
            let combined = combine_h0_summaries(
                HierarchicalH0Summary {
                    z0: 0,
                    z1: 2,
                    summary: left,
                },
                HierarchicalH0Summary {
                    z0: 2,
                    z1: 4,
                    summary: right,
                },
                connectivity,
            )
            .unwrap();
            let observed = reduce_h0_persistence(&[combined.summary], connectivity).unwrap();
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn hierarchical_h0_multilevel_and_odd_tail_match_unsplit_block() {
        let width = 3usize;
        let height = 3usize;
        let depth = 5usize;
        let slice_size = width * height;
        let values: Vec<ScalarKey> = (0..width * height * depth)
            .map(|index| {
                ScalarKey::from_u16(((index * 67 + index / 2 + (index % 5) * 13) % 31) as u16)
            })
            .collect();
        let whole = ScalarBlock {
            z0: 0,
            shape: [width, height, depth],
            values: values.clone(),
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let expected = reduce_h0_persistence(
                &[process_slab_h0_persistence_scalar(0, &whole, connectivity)],
                connectivity,
            )
            .unwrap();

            let mut leaves = Vec::new();
            for z in 0..depth {
                let block = ScalarBlock {
                    z0: z,
                    shape: [width, height, 1],
                    values: values[z * slice_size..(z + 1) * slice_size].to_vec(),
                    pixel_type: crate::io_scalar::ScalarPixelType::U16,
                };
                let mut summary = process_slab_h0_persistence_scalar(0, &block, connectivity);
                summary.slab_id = 0;
                leaves.push(HierarchicalH0Summary {
                    z0: z,
                    z1: z + 1,
                    summary,
                });
            }

            let p01 =
                combine_h0_summaries(leaves.remove(0), leaves.remove(0), connectivity).unwrap();
            let p23 =
                combine_h0_summaries(leaves.remove(0), leaves.remove(0), connectivity).unwrap();
            let p03 = combine_h0_summaries(p01, p23, connectivity).unwrap();
            let p04 = combine_h0_summaries(p03, leaves.remove(0), connectivity).unwrap();
            let observed = reduce_h0_persistence(&[p04.summary], connectivity).unwrap();
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn packed_h0_matches_parent_rank_layout() {
        let values = (0..216)
            .map(|index| ScalarKey::from_u16(((index * 47 + index / 7) % 37) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [6, 6, 6],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };
        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let mut reference = None;
            for uf_layout in [
                UnionFindLayoutStrategy::ParentRank,
                UnionFindLayoutStrategy::Packed,
            ] {
                let (summary, profile) = process_slab_h0_persistence_scalar_profiled(
                    0,
                    &block,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    H0PruningCacheStrategy::Entries64K,
                    ActiveStateStrategy::Separate,
                    InterfaceStateStrategy::RootInvariant,
                    uf_layout,
                    true,
                );
                let intervals = reduce_h0_persistence(&[summary], connectivity).unwrap();
                if let Some(expected) = reference.as_ref() {
                    assert_eq!(&intervals, expected);
                } else {
                    reference = Some(intervals);
                }
                match uf_layout {
                    UnionFindLayoutStrategy::ParentRank => {
                        assert_eq!(profile.uf_rank_state_bytes, 216)
                    }
                    UnionFindLayoutStrategy::Packed => assert_eq!(profile.uf_rank_state_bytes, 0),
                }
                assert_eq!(profile.uf_parent_state_bytes, 216 * 4);
            }
        }
    }
}
