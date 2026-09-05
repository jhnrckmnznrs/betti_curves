use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::interface_sparsify_scalar::sparsify_scalar_superlevel_interface;
use crate::io_scalar::{
    F32ScalarBlock, ScalarBlock, ScalarTiffStackReader, print_scalar_volume_info,
};
use crate::local_pruning::{NeighborhoodComponentPruner, NeighborhoodPruningStats};
use crate::local_uf_state::LocalUnionFindState;
use crate::memory_audit::{emit as emit_memory_snapshot, vec_capacity_bytes};
use crate::scalar::{F32Key, LocalScalarKey, ScalarKey};
use crate::scalar_order::{sorted_f32_indices, sorted_scalar_indices};
use crate::scalar_stream_tuning::{
    ActiveStateStrategy, GlobalH2BirthStateStrategy, GlobalH2UnionFindLayoutStrategy,
    InterfaceStateStrategy, LocalH2BirthStateStrategy, NeighborKernelStrategy,
    NeighborRootCheckStrategy, RepresentativeActiveCheckStrategy, UnionFindLayoutStrategy,
    UnionKernelStrategy,
};
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};
use crate::tiff_paths::find_tiff_stack_directories;

/// A foreground H2 persistence interval `[birth, death)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarH2Interval {
    pub birth: ScalarKey,
    pub death: ScalarKey,
}

#[derive(Debug, Clone, Copy)]
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
pub(crate) struct OutsideEvent<K = ScalarKey> {
    pub(crate) value: K,
    pub(crate) interface_node: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InterfaceMergeEvent<K = ScalarKey> {
    pub(crate) value: K,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum H2LocalStreamEvent<K = ScalarKey> {
    FinalPair(FinitePair<K>),
    Attach(AttachEvent<K>),
    Outside(OutsideEvent<K>),
    InterfaceMerge(InterfaceMergeEvent<K>),
}

#[derive(Debug)]
pub(crate) struct BoundaryFaceNodes<K = ScalarKey> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) node_ids: Vec<u32>,
    pub(crate) values: Vec<K>,
}

#[derive(Debug)]
pub(crate) struct SlabH2Summary<K = ScalarKey> {
    pub(crate) slab_id: usize,
    pub(crate) finalized_pairs: Vec<FinitePair<K>>,
    pub(crate) attach_events: Vec<AttachEvent<K>>,
    pub(crate) outside_events: Vec<OutsideEvent<K>>,
    pub(crate) interface_merge_events: Vec<InterfaceMergeEvent<K>>,
    pub(crate) interface_node_count: u32,
    pub(crate) z_min_face: BoundaryFaceNodes<K>,
    pub(crate) z_max_face: BoundaryFaceNodes<K>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct H2SlabPreparationProfile {
    pub(crate) scalar_order_seconds: f64,
    pub(crate) local_sweep_seconds: f64,
    pub(crate) total_voxels: u64,
    pub(crate) interior_fast_voxels: u64,
    pub(crate) pruning_mask_calls: u64,
    pub(crate) active_state_checks: u64,
    pub(crate) active_neighbor_hits: u64,
    pub(crate) representative_visits: u64,
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
    pub(crate) direct_parent_checks: u64,
    pub(crate) direct_parent_hits: u64,
    pub(crate) avoided_neighbor_find_calls: u64,
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
    direct_parent_checks: u64,
    direct_parent_hits: u64,
    avoided_neighbor_find_calls: u64,
    interface_rep_queries: u64,
    interface_rep_writes: u64,
    interface_forced_root_unions: u64,
    interface_interface_unions: u64,
    max_rank_observed: u64,
    uf_parent_state_bytes: u64,
    uf_rank_state_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundBirth {
    Outside,
    Finite(ScalarKey),
}

fn older_background_birth(a: BackgroundBirth, b: BackgroundBirth) -> BackgroundBirth {
    match (a, b) {
        (BackgroundBirth::Outside, _) | (_, BackgroundBirth::Outside) => BackgroundBirth::Outside,
        (BackgroundBirth::Finite(a_value), BackgroundBirth::Finite(b_value)) => {
            BackgroundBirth::Finite(a_value.max(b_value))
        }
    }
}

fn pair_from_background_merge(
    a: BackgroundBirth,
    b: BackgroundBirth,
    merge_value: ScalarKey,
) -> Option<FinitePair> {
    let younger_birth = match (a, b) {
        (BackgroundBirth::Outside, BackgroundBirth::Outside) => return None,
        (BackgroundBirth::Outside, BackgroundBirth::Finite(value))
        | (BackgroundBirth::Finite(value), BackgroundBirth::Outside) => value,
        (BackgroundBirth::Finite(a_value), BackgroundBirth::Finite(b_value)) => {
            a_value.min(b_value)
        }
    };

    Some(FinitePair {
        birth: merge_value,
        death: younger_birth,
    })
}

#[derive(Debug, Clone, Copy)]
enum LocalPersistenceAction<K> {
    None,
    FinalPair(FinitePair<K>),
    Attach(AttachEvent<K>),
    Outside(OutsideEvent<K>),
    InterfaceMerge(InterfaceMergeEvent<K>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalBackgroundBirth<K> {
    Outside,
    Finite(K),
}

#[derive(Debug)]
enum LocalH2BirthStorage<K: LocalScalarKey> {
    Tagged(Vec<LocalBackgroundBirth<K>>),
    Compact(Vec<K>),
}

impl<K: LocalScalarKey> LocalH2BirthStorage<K> {
    fn new(values: &[K], strategy: LocalH2BirthStateStrategy) -> Self {
        match strategy {
            LocalH2BirthStateStrategy::Tagged => Self::Tagged(
                values
                    .iter()
                    .copied()
                    .map(LocalBackgroundBirth::Finite)
                    .collect(),
            ),
            LocalH2BirthStateStrategy::Compact => {
                debug_assert!(values.iter().all(|value| !(*value).is_outside_marker()));
                Self::Compact(values.to_vec())
            }
        }
    }

    fn new_owned(values: Vec<K>, strategy: LocalH2BirthStateStrategy) -> Self {
        match strategy {
            LocalH2BirthStateStrategy::Tagged => Self::Tagged(
                values
                    .into_iter()
                    .map(LocalBackgroundBirth::Finite)
                    .collect(),
            ),
            LocalH2BirthStateStrategy::Compact => {
                debug_assert!(values.iter().all(|value| !(*value).is_outside_marker()));
                Self::Compact(values)
            }
        }
    }

    #[inline]
    fn finite_value_at(&self, index: usize) -> K {
        match self.get(index) {
            LocalBackgroundBirth::Finite(value) => value,
            LocalBackgroundBirth::Outside => {
                panic!("attempted to read scalar value from an already-outside H2 birth entry")
            }
        }
    }

    #[inline]
    fn get(&self, index: usize) -> LocalBackgroundBirth<K> {
        match self {
            Self::Tagged(values) => values[index],
            Self::Compact(values) => {
                let value = values[index];
                if value.is_outside_marker() {
                    LocalBackgroundBirth::Outside
                } else {
                    LocalBackgroundBirth::Finite(value)
                }
            }
        }
    }

    #[inline]
    fn set(&mut self, index: usize, birth: LocalBackgroundBirth<K>) {
        match self {
            Self::Tagged(values) => values[index] = birth,
            Self::Compact(values) => {
                values[index] = match birth {
                    LocalBackgroundBirth::Outside => K::outside_marker(),
                    LocalBackgroundBirth::Finite(value) => {
                        debug_assert!(!value.is_outside_marker());
                        value
                    }
                };
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Tagged(values) => values.len(),
            Self::Compact(values) => values.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Tagged(values) => values.capacity(),
            Self::Compact(values) => values.capacity(),
        }
    }

    fn capacity_bytes(&self) -> u64 {
        match self {
            Self::Tagged(values) => {
                values.capacity() as u64 * core::mem::size_of::<LocalBackgroundBirth<K>>() as u64
            }
            Self::Compact(values) => values.capacity() as u64 * core::mem::size_of::<K>() as u64,
        }
    }

    fn strategy(&self) -> LocalH2BirthStateStrategy {
        match self {
            Self::Tagged(_) => LocalH2BirthStateStrategy::Tagged,
            Self::Compact(_) => LocalH2BirthStateStrategy::Compact,
        }
    }
}

fn older_local_background_birth<K: LocalScalarKey>(
    a: LocalBackgroundBirth<K>,
    b: LocalBackgroundBirth<K>,
) -> LocalBackgroundBirth<K> {
    match (a, b) {
        (LocalBackgroundBirth::Outside, _) | (_, LocalBackgroundBirth::Outside) => {
            LocalBackgroundBirth::Outside
        }
        (LocalBackgroundBirth::Finite(a_value), LocalBackgroundBirth::Finite(b_value)) => {
            LocalBackgroundBirth::Finite(a_value.max(b_value))
        }
    }
}

fn pair_from_local_background_merge<K: LocalScalarKey>(
    a: LocalBackgroundBirth<K>,
    b: LocalBackgroundBirth<K>,
    merge_value: K,
) -> Option<FinitePair<K>> {
    let younger_birth = match (a, b) {
        (LocalBackgroundBirth::Outside, LocalBackgroundBirth::Outside) => return None,
        (LocalBackgroundBirth::Outside, LocalBackgroundBirth::Finite(value))
        | (LocalBackgroundBirth::Finite(value), LocalBackgroundBirth::Outside) => value,
        (LocalBackgroundBirth::Finite(a_value), LocalBackgroundBirth::Finite(b_value)) => {
            a_value.min(b_value)
        }
    };

    Some(FinitePair {
        birth: merge_value,
        death: younger_birth,
    })
}

#[derive(Debug)]
struct LocalBackgroundPersistenceUnionFind<K: LocalScalarKey> {
    uf_state: LocalUnionFindState,
    birth: LocalH2BirthStorage<K>,
    interface_state: InterfaceStateStrategy,
    interface_rep: Option<Vec<u32>>,
    slice_size: usize,
    depth: usize,
    prune_elder_dominated_attaches: bool,
    prune_outside_dominated_structural: bool,
    diagnostics: Option<LocalUnionFindStats>,
}

impl<K: LocalScalarKey> LocalBackgroundPersistenceUnionFind<K> {
    fn new(
        values: &[K],
        shape: [usize; 3],
        diagnostics: bool,
        active_state: ActiveStateStrategy,
        interface_state: InterfaceStateStrategy,
        uf_layout: UnionFindLayoutStrategy,
        birth_state: LocalH2BirthStateStrategy,
    ) -> Self {
        let uf_state = LocalUnionFindState::new(values.len(), active_state, uf_layout);

        let interface_rep = matches!(interface_state, InterfaceStateStrategy::Vector)
            .then(|| vec![NO_INTERFACE_REP; values.len()]);

        Self {
            uf_state,
            birth: LocalH2BirthStorage::new(values, birth_state),
            interface_state,
            interface_rep,
            slice_size: shape[0] * shape[1],
            depth: shape[2],
            prune_elder_dominated_attaches: false,
            prune_outside_dominated_structural: false,
            diagnostics: diagnostics.then(LocalUnionFindStats::default),
        }
    }

    #[allow(clippy::too_many_arguments)] // Construction mirrors the explicit local H2 tuning surface.
    fn new_owned(
        values: Vec<K>,
        shape: [usize; 3],
        diagnostics: bool,
        active_state: ActiveStateStrategy,
        interface_state: InterfaceStateStrategy,
        uf_layout: UnionFindLayoutStrategy,
        birth_state: LocalH2BirthStateStrategy,
        prune_elder_dominated_attaches: bool,
        prune_outside_dominated_structural: bool,
    ) -> Self {
        let value_count = values.len();
        let uf_state = LocalUnionFindState::new(value_count, active_state, uf_layout);
        let interface_rep = matches!(interface_state, InterfaceStateStrategy::Vector)
            .then(|| vec![NO_INTERFACE_REP; value_count]);
        Self {
            uf_state,
            birth: LocalH2BirthStorage::new_owned(values, birth_state),
            interface_state,
            interface_rep,
            slice_size: shape[0] * shape[1],
            depth: shape[2],
            prune_elder_dominated_attaches,
            prune_outside_dominated_structural,
            diagnostics: diagnostics.then(LocalUnionFindStats::default),
        }
    }

    #[inline]
    fn scalar_key_at(&self, index: usize) -> K {
        self.birth.finite_value_at(index)
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

    fn mark_outside(&mut self, x: u32) {
        let root = self.find(x);
        self.birth.set(root as usize, LocalBackgroundBirth::Outside);
    }

    fn merge_known_roots_with_persistence(
        &mut self,
        root_a: u32,
        root_b: u32,
        merge_value: K,
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

        let birth_a = self.birth.get(root_a as usize);
        let birth_b = self.birth.get(root_b as usize);
        let rep_a = self.interface_node_for_root(root_a);
        let rep_b = self.interface_node_for_root(root_b);
        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => pair_from_local_background_merge(birth_a, birth_b, merge_value)
                .map(LocalPersistenceAction::FinalPair)
                .unwrap_or(LocalPersistenceAction::None),

            (true, false) => match birth_b {
                LocalBackgroundBirth::Finite(branch_birth) => {
                    let dominated = self.prune_elder_dominated_attaches
                        && match birth_a {
                            LocalBackgroundBirth::Outside => true,
                            LocalBackgroundBirth::Finite(root_birth) => root_birth >= branch_birth,
                        };
                    if dominated {
                        LocalPersistenceAction::FinalPair(FinitePair {
                            birth: merge_value,
                            death: branch_birth,
                        })
                    } else {
                        LocalPersistenceAction::Attach(AttachEvent {
                            value: merge_value,
                            interface_node: rep_a,
                            branch_birth,
                        })
                    }
                }
                LocalBackgroundBirth::Outside => {
                    if birth_a == LocalBackgroundBirth::Outside {
                        LocalPersistenceAction::None
                    } else {
                        LocalPersistenceAction::Outside(OutsideEvent {
                            value: merge_value,
                            interface_node: rep_a,
                        })
                    }
                }
            },

            (false, true) => match birth_a {
                LocalBackgroundBirth::Finite(branch_birth) => {
                    let dominated = self.prune_elder_dominated_attaches
                        && match birth_b {
                            LocalBackgroundBirth::Outside => true,
                            LocalBackgroundBirth::Finite(root_birth) => root_birth >= branch_birth,
                        };
                    if dominated {
                        LocalPersistenceAction::FinalPair(FinitePair {
                            birth: merge_value,
                            death: branch_birth,
                        })
                    } else {
                        LocalPersistenceAction::Attach(AttachEvent {
                            value: merge_value,
                            interface_node: rep_b,
                            branch_birth,
                        })
                    }
                }
                LocalBackgroundBirth::Outside => {
                    if birth_b == LocalBackgroundBirth::Outside {
                        LocalPersistenceAction::None
                    } else {
                        LocalPersistenceAction::Outside(OutsideEvent {
                            value: merge_value,
                            interface_node: rep_b,
                        })
                    }
                }
            },

            (true, true) => {
                if self.prune_outside_dominated_structural {
                    match (birth_a, birth_b) {
                        (LocalBackgroundBirth::Outside, LocalBackgroundBirth::Outside) => {
                            LocalPersistenceAction::None
                        }
                        (LocalBackgroundBirth::Outside, LocalBackgroundBirth::Finite(_)) => {
                            LocalPersistenceAction::Outside(OutsideEvent {
                                value: merge_value,
                                interface_node: rep_b,
                            })
                        }
                        (LocalBackgroundBirth::Finite(_), LocalBackgroundBirth::Outside) => {
                            LocalPersistenceAction::Outside(OutsideEvent {
                                value: merge_value,
                                interface_node: rep_a,
                            })
                        }
                        (LocalBackgroundBirth::Finite(_), LocalBackgroundBirth::Finite(_)) => {
                            LocalPersistenceAction::InterfaceMerge(InterfaceMergeEvent {
                                value: merge_value,
                                a: rep_a,
                                b: rep_b,
                            })
                        }
                    }
                } else {
                    LocalPersistenceAction::InterfaceMerge(InterfaceMergeEvent {
                        value: merge_value,
                        a: rep_a,
                        b: rep_b,
                    })
                }
            }
        };

        let merged_birth = older_local_background_birth(birth_a, birth_b);
        let new_root = self.link_roots(root_a, root_b, rep_a, rep_b);
        self.birth.set(new_root as usize, merged_birth);
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
        merge_value: K,
    ) -> Option<LocalPersistenceAction<K>> {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.union_attempts += 1;
        }
        let root_a = self.find(a);
        let root_b = self.find(b);
        self.merge_known_roots_with_persistence(root_a, root_b, merge_value)
            .1
    }

    fn union_from_current_root_with_persistence(
        &mut self,
        current_root: u32,
        neighbor: u32,
        merge_value: K,
        neighbor_root_check: NeighborRootCheckStrategy,
    ) -> (u32, Option<LocalPersistenceAction<K>>) {
        debug_assert!(self.uf_state.is_root(current_root));
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.union_attempts += 1;
            stats.root_carry_attempts += 1;
            stats.avoided_find_calls += 1;
        }

        if matches!(
            neighbor_root_check,
            NeighborRootCheckStrategy::ParentShortcut
        ) {
            if let Some(stats) = self.diagnostics.as_mut() {
                stats.direct_parent_checks += 1;
            }
            if neighbor == current_root || self.uf_state.direct_parent_is(neighbor, current_root) {
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.direct_parent_hits += 1;
                    stats.avoided_neighbor_find_calls += 1;
                    stats.same_root_unions += 1;
                }
                return (current_root, None);
            }
        }

        let neighbor_root = self.find(neighbor);
        self.merge_known_roots_with_persistence(current_root, neighbor_root, merge_value)
    }
}

struct LocalPersistenceBuffers<'a, K> {
    finalized_pairs: &'a mut Vec<FinitePair<K>>,
    attach_events: &'a mut Vec<AttachEvent<K>>,
    outside_events: &'a mut Vec<OutsideEvent<K>>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent<K>>,
}

fn handle_local_action<K: LocalScalarKey>(
    action: LocalPersistenceAction<K>,
    buffers: &mut LocalPersistenceBuffers<'_, K>,
) {
    match action {
        LocalPersistenceAction::None => {}
        LocalPersistenceAction::FinalPair(pair) => {
            if pair.birth < pair.death {
                buffers.finalized_pairs.push(pair);
            }
        }
        LocalPersistenceAction::Attach(event) => buffers.attach_events.push(event),
        LocalPersistenceAction::Outside(event) => buffers.outside_events.push(event),
        LocalPersistenceAction::InterfaceMerge(event) => {
            buffers.interface_merge_events.push(event);
        }
    }
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
fn union_active_neighbor<K: LocalScalarKey>(
    union_find: &mut LocalBackgroundPersistenceUnionFind<K>,
    active: Option<&[u8]>,
    active_state: ActiveStateStrategy,
    current: u32,
    current_root: &mut u32,
    neighbor: usize,
    value: K,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
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
            ActiveStateStrategy::ParentSentinel => !union_find.parent_state_is_active(neighbor),
        };
        union_find.record_active_recheck(failed);
        if failed {
            return;
        }
    }

    let action = match union_kernel {
        UnionKernelStrategy::Conventional => {
            union_find.union_with_persistence(current, neighbor as u32, value)
        }
        UnionKernelStrategy::RootCarrying => {
            let (new_root, action) = union_find.union_from_current_root_with_persistence(
                *current_root,
                neighbor as u32,
                value,
                neighbor_root_check,
            );
            *current_root = new_root;
            action
        }
    };

    if let Some(action) = action {
        handle_local_action(action, buffers);
    }
}

fn voxel_touches_global_boundary(
    z0: usize,
    x: usize,
    y: usize,
    z_local: usize,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
) -> bool {
    let z_global = z0 + z_local;
    x == 0
        || x + 1 == global_width
        || y == 0
        || y + 1 == global_height
        || z_global == 0
        || z_global + 1 == global_depth
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
fn process_slab_h2_persistence_values<K: LocalScalarKey>(
    slab_id: usize,
    z0: usize,
    shape: [usize; 3],
    values: &[K],
    order: Vec<u32>,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
    memory_audit: bool,
) -> (
    SlabH2Summary<K>,
    NeighborhoodPruningStats,
    LocalUnionFindStats,
) {
    let width = shape[0];
    let height = shape[1];
    let depth = shape[2];
    let voxel_count = width * height * depth;
    let slice_size = width * height;
    assert_eq!(values.len(), voxel_count);

    let mut union_find = LocalBackgroundPersistenceUnionFind::new(
        values,
        shape,
        sweep_diagnostics,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
    );
    if slab_id == 0 {
        let parent_bytes = union_find.uf_state.parent_capacity_bytes();
        let rank_bytes = union_find.uf_state.rank_capacity_bytes();
        let birth_bytes = union_find.birth.capacity_bytes();
        println!(
            "PROFILE_LOCAL_H2_STATE strategy={} nodes={} key_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
            union_find.birth.strategy().as_str(),
            voxel_count,
            core::mem::size_of::<K>(),
            parent_bytes,
            rank_bytes,
            birth_bytes,
            parent_bytes
                .saturating_add(rank_bytes)
                .saturating_add(birth_bytes),
        );
    }

    let mut active =
        matches!(active_state, ActiveStateStrategy::Separate).then(|| vec![0u8; voxel_count]);
    let mut local_pruner =
        NeighborhoodComponentPruner::with_diagnostics(background_connectivity, sweep_diagnostics);
    let linear_offsets = matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
        .then(|| local_pruner.linear_offsets(width, height));

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut finalized_pairs = Vec::new();
    let mut attach_events = Vec::new();
    let mut outside_events = Vec::new();
    let mut interface_merge_events = Vec::new();

    if memory_audit {
        emit_memory_snapshot("scalar_h2_local", "after_local_state_alloc", Some(slab_id));
        println!(
            "PROFILE_MEMVEC scalar_h2_local stage=after_local_state_alloc slab={} \
input_values_len={} input_values_bytes={} order_len={} order_capacity={} order_capacity_bytes={} \
uf_parent_capacity_bytes={} uf_rank_capacity_bytes={} birth_len={} birth_capacity={} birth_capacity_bytes={} \
active_capacity_bytes={} interface_rep_capacity_bytes={} pruning_cache_capacity_bytes={} \
finite_pairs_capacity_bytes=0 attach_capacity_bytes=0 outside_capacity_bytes=0 interface_merge_capacity_bytes=0",
            slab_id,
            values.len(),
            (values.len() as u64).saturating_mul(core::mem::size_of::<K>() as u64),
            order.len(),
            order.capacity(),
            vec_capacity_bytes(&order),
            union_find.uf_state.parent_capacity_bytes(),
            union_find.uf_state.rank_capacity_bytes(),
            union_find.birth.len(),
            union_find.birth.capacity(),
            union_find.birth.capacity_bytes(),
            active.as_ref().map(vec_capacity_bytes).unwrap_or(0),
            union_find
                .interface_rep
                .as_ref()
                .map(vec_capacity_bytes)
                .unwrap_or(0),
            local_pruner.cache_capacity_bytes(),
        );
    }

    let mut group_end = order.len();
    while group_end > 0 {
        let value = values[order[group_end - 1] as usize];
        let mut group_start = group_end - 1;
        while group_start > 0 && values[order[group_start - 1] as usize] == value {
            group_start -= 1;
        }

        for &index_u32 in &order[group_start..group_end] {
            let index = index_u32 as usize;
            union_find.activate(index_u32);
            if let Some(active) = active.as_mut() {
                active[index] = 1;
            }

            let face_index = index % slice_size;
            let z = index / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                union_find.set_interface_rep(index_u32, interface_node);
            }

            let x = index % width;
            let y = (index / width) % height;

            if voxel_touches_global_boundary(z0, x, y, z, global_width, global_height, global_depth)
            {
                union_find.mark_outside(index_u32);
                if let Some(interface_node) =
                    local_boundary_node_id(z, depth, face_index, slice_size)
                {
                    outside_events.push(OutsideEvent {
                        value,
                        interface_node,
                    });
                }
            }

            let mut buffers = LocalPersistenceBuffers {
                finalized_pairs: &mut finalized_pairs,
                attach_events: &mut attach_events,
                outside_events: &mut outside_events,
                interface_merge_events: &mut interface_merge_events,
            };

            let mut current_root = index_u32;

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
                            index,
                            offsets,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_interior_by(index, offsets, |neighbor| {
                            union_find.parent_state_is_active(neighbor)
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
                            union_find.parent_state_is_active(neighbor)
                        }),
                }
            };

            for neighbor in representatives.iter() {
                union_active_neighbor(
                    &mut union_find,
                    active.as_deref(),
                    active_state,
                    index_u32,
                    &mut current_root,
                    neighbor,
                    value,
                    representative_active_check,
                    union_kernel,
                    neighbor_root_check,
                    &mut buffers,
                );
            }
        }

        group_end = group_start;
    }

    if memory_audit {
        emit_memory_snapshot("scalar_h2_local", "after_local_sweep", Some(slab_id));
        println!(
            "PROFILE_MEMVEC scalar_h2_local stage=after_local_sweep slab={} \
finite_pairs_len={} finite_pairs_capacity={} finite_pairs_capacity_bytes={} \
attach_len={} attach_capacity={} attach_capacity_bytes={} outside_len={} outside_capacity={} outside_capacity_bytes={} \
interface_merge_len={} interface_merge_capacity={} interface_merge_capacity_bytes={} \
order_capacity_bytes={} uf_parent_capacity_bytes={} uf_rank_capacity_bytes={} birth_capacity_bytes={} active_capacity_bytes={} interface_rep_capacity_bytes={} pruning_cache_capacity_bytes={}",
            slab_id,
            finalized_pairs.len(),
            finalized_pairs.capacity(),
            vec_capacity_bytes(&finalized_pairs),
            attach_events.len(),
            attach_events.capacity(),
            vec_capacity_bytes(&attach_events),
            outside_events.len(),
            outside_events.capacity(),
            vec_capacity_bytes(&outside_events),
            interface_merge_events.len(),
            interface_merge_events.capacity(),
            vec_capacity_bytes(&interface_merge_events),
            vec_capacity_bytes(&order),
            union_find.uf_state.parent_capacity_bytes(),
            union_find.uf_state.rank_capacity_bytes(),
            union_find.birth.capacity_bytes(),
            active.as_ref().map(vec_capacity_bytes).unwrap_or(0),
            union_find
                .interface_rep
                .as_ref()
                .map(vec_capacity_bytes)
                .unwrap_or(0),
            local_pruner.cache_capacity_bytes(),
        );
    }

    let z_min_face = extract_boundary_face_nodes_values(shape, values, 0);
    let z_max_face = extract_boundary_face_nodes_values(shape, values, depth - 1);

    if memory_audit {
        emit_memory_snapshot("scalar_h2_local", "after_boundary_faces", Some(slab_id));
        println!(
            "PROFILE_MEMVEC scalar_h2_local stage=after_boundary_faces slab={} \
z_min_node_capacity_bytes={} z_min_value_capacity_bytes={} z_max_node_capacity_bytes={} z_max_value_capacity_bytes={} \
summary_event_capacity_bytes={}",
            slab_id,
            vec_capacity_bytes(&z_min_face.node_ids),
            vec_capacity_bytes(&z_min_face.values),
            vec_capacity_bytes(&z_max_face.node_ids),
            vec_capacity_bytes(&z_max_face.values),
            vec_capacity_bytes(&finalized_pairs)
                .saturating_add(vec_capacity_bytes(&attach_events))
                .saturating_add(vec_capacity_bytes(&outside_events))
                .saturating_add(vec_capacity_bytes(&interface_merge_events)),
        );
    }

    let pruning_stats = local_pruner.diagnostic_stats();
    let union_find_stats = union_find.diagnostic_stats();
    (
        SlabH2Summary {
            slab_id,
            finalized_pairs,
            attach_events,
            outside_events,
            interface_merge_events,
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        pruning_stats,
        union_find_stats,
    )
}

fn emit_h2_local_action<K, F>(action: LocalPersistenceAction<K>, emit: &mut F) -> Result<()>
where
    K: LocalScalarKey,
    F: FnMut(H2LocalStreamEvent<K>) -> Result<()>,
{
    match action {
        LocalPersistenceAction::None => {}
        LocalPersistenceAction::FinalPair(pair) => {
            if pair.birth < pair.death {
                emit(H2LocalStreamEvent::FinalPair(pair))?;
            }
        }
        LocalPersistenceAction::Attach(event) => emit(H2LocalStreamEvent::Attach(event))?,
        LocalPersistenceAction::Outside(event) => emit(H2LocalStreamEvent::Outside(event))?,
        LocalPersistenceAction::InterfaceMerge(event) => {
            emit(H2LocalStreamEvent::InterfaceMerge(event))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn union_active_neighbor_h2_sink<K, F>(
    union_find: &mut LocalBackgroundPersistenceUnionFind<K>,
    active: Option<&[u8]>,
    active_state: ActiveStateStrategy,
    current: u32,
    current_root: &mut u32,
    neighbor: usize,
    value: K,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    emit: &mut F,
) -> Result<()>
where
    K: LocalScalarKey,
    F: FnMut(H2LocalStreamEvent<K>) -> Result<()>,
{
    if matches!(
        representative_active_check,
        RepresentativeActiveCheckStrategy::Recheck
    ) {
        let failed = match active_state {
            ActiveStateStrategy::Separate => {
                active.expect("separate active state requires byte array")[neighbor] == 0
            }
            ActiveStateStrategy::ParentSentinel => !union_find.parent_state_is_active(neighbor),
        };
        union_find.record_active_recheck(failed);
        if failed {
            return Ok(());
        }
    }

    let action = match union_kernel {
        UnionKernelStrategy::Conventional => {
            union_find.union_with_persistence(current, neighbor as u32, value)
        }
        UnionKernelStrategy::RootCarrying => {
            let (new_root, action) = union_find.union_from_current_root_with_persistence(
                *current_root,
                neighbor as u32,
                value,
                neighbor_root_check,
            );
            *current_root = new_root;
            action
        }
    };
    if let Some(action) = action {
        emit_h2_local_action(action, emit)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_slab_h2_persistence_values_owned_sink<K, F>(
    slab_id: usize,
    z0: usize,
    shape: [usize; 3],
    values: Vec<K>,
    order: Vec<u32>,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
    prune_elder_dominated_attaches: bool,
    prune_outside_dominated_structural: bool,
    mut emit: F,
) -> Result<(
    SlabH2Summary<K>,
    NeighborhoodPruningStats,
    LocalUnionFindStats,
)>
where
    K: LocalScalarKey,
    F: FnMut(H2LocalStreamEvent<K>) -> Result<()>,
{
    let width = shape[0];
    let height = shape[1];
    let depth = shape[2];
    let voxel_count = width * height * depth;
    let slice_size = width * height;
    assert_eq!(values.len(), voxel_count);

    // Preserve the original scalar values on the two exposed slab faces before
    // transferring ownership of the input buffer to compact H2 birth storage.
    let z_min_face = extract_boundary_face_values_only(shape, &values, 0);
    let z_max_face = extract_boundary_face_values_only(shape, &values, depth - 1);
    let mut union_find = LocalBackgroundPersistenceUnionFind::new_owned(
        values,
        shape,
        sweep_diagnostics,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        prune_elder_dominated_attaches,
        prune_outside_dominated_structural,
    );
    if slab_id == 0 {
        let parent_bytes = union_find.uf_state.parent_capacity_bytes();
        let rank_bytes = union_find.uf_state.rank_capacity_bytes();
        let birth_bytes = union_find.birth.capacity_bytes();
        println!(
            "PROFILE_LOCAL_H2_STATE strategy={} storage=reuse-input attach_pruning={} nodes={} key_bytes={} parent_bytes={} rank_bytes={} birth_bytes={} total_bytes={}",
            union_find.birth.strategy().as_str(),
            if prune_elder_dominated_attaches {
                "elder-dominated"
            } else {
                "off"
            },
            voxel_count,
            core::mem::size_of::<K>(),
            parent_bytes,
            rank_bytes,
            birth_bytes,
            parent_bytes
                .saturating_add(rank_bytes)
                .saturating_add(birth_bytes),
        );
    }

    let mut active =
        matches!(active_state, ActiveStateStrategy::Separate).then(|| vec![0u8; voxel_count]);
    let mut local_pruner =
        NeighborhoodComponentPruner::with_diagnostics(background_connectivity, sweep_diagnostics);
    let linear_offsets = matches!(neighbor_kernel, NeighborKernelStrategy::InteriorFast)
        .then(|| local_pruner.linear_offsets(width, height));
    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut group_end = order.len();
    while group_end > 0 {
        let value = union_find.scalar_key_at(order[group_end - 1] as usize);
        let mut group_start = group_end - 1;
        while group_start > 0 && union_find.scalar_key_at(order[group_start - 1] as usize) == value
        {
            group_start -= 1;
        }

        for &index_u32 in &order[group_start..group_end] {
            let index = index_u32 as usize;
            union_find.activate(index_u32);
            if let Some(active) = active.as_mut() {
                active[index] = 1;
            }

            let face_index = index % slice_size;
            let z = index / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                union_find.set_interface_rep(index_u32, interface_node);
            }

            let x = index % width;
            let y = (index / width) % height;
            if voxel_touches_global_boundary(z0, x, y, z, global_width, global_height, global_depth)
            {
                union_find.mark_outside(index_u32);
                if let Some(interface_node) =
                    local_boundary_node_id(z, depth, face_index, slice_size)
                {
                    emit(H2LocalStreamEvent::Outside(OutsideEvent {
                        value,
                        interface_node,
                    }))?;
                }
            }

            let mut current_root = index_u32;
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
                            index,
                            offsets,
                            |neighbor| active_ref[neighbor] != 0,
                        )
                    }
                    ActiveStateStrategy::ParentSentinel => local_pruner
                        .representative_neighbors_interior_by(index, offsets, |neighbor| {
                            union_find.parent_state_is_active(neighbor)
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
                            union_find.parent_state_is_active(neighbor)
                        }),
                }
            };

            for neighbor in representatives.iter() {
                union_active_neighbor_h2_sink(
                    &mut union_find,
                    active.as_deref(),
                    active_state,
                    index_u32,
                    &mut current_root,
                    neighbor,
                    value,
                    representative_active_check,
                    union_kernel,
                    neighbor_root_check,
                    &mut emit,
                )?;
            }
        }
        group_end = group_start;
    }

    let pruning_stats = local_pruner.diagnostic_stats();
    let union_find_stats = union_find.diagnostic_stats();
    Ok((
        SlabH2Summary {
            slab_id,
            finalized_pairs: Vec::new(),
            attach_events: Vec::new(),
            outside_events: Vec::new(),
            interface_merge_events: Vec::new(),
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        pruning_stats,
        union_find_stats,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_slab_h2_persistence_f32_native_direct_profiled<F>(
    slab_id: usize,
    z0: usize,
    block: F32ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
    prune_elder_dominated_attaches: bool,
    prune_outside_dominated_structural: bool,
    emit: F,
) -> Result<(SlabH2Summary<F32Key>, H2SlabPreparationProfile)>
where
    F: FnMut(H2LocalStreamEvent<F32Key>) -> Result<()>,
{
    if !matches!(local_h2_birth_state, LocalH2BirthStateStrategy::Compact) {
        bail!("H2 reuse-input/direct leaf processing requires compact local H2 birth storage");
    }
    let shape = block.shape;
    let voxel_count = block.values.len();
    let order_start = Instant::now();
    let order = sorted_f32_indices(&block.values);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h2_persistence_values_owned_sink(
        slab_id,
        z0,
        shape,
        block.values,
        order,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        neighbor_root_check,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        sweep_diagnostics,
        prune_elder_dominated_attaches,
        prune_outside_dominated_structural,
        emit,
    )?;
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    Ok((
        summary,
        H2SlabPreparationProfile {
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
            direct_parent_checks: union_find_stats.direct_parent_checks,
            direct_parent_hits: union_find_stats.direct_parent_hits,
            avoided_neighbor_find_calls: union_find_stats.avoided_neighbor_find_calls,
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

pub(crate) fn process_slab_h2_persistence(
    slab_id: usize,
    block: &ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
) -> SlabH2Summary {
    process_slab_h2_persistence_profiled(
        slab_id,
        block,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        NeighborKernelStrategy::Generic,
        RepresentativeActiveCheckStrategy::Recheck,
        UnionKernelStrategy::Conventional,
        NeighborRootCheckStrategy::Find,
        ActiveStateStrategy::Separate,
        InterfaceStateStrategy::Vector,
        UnionFindLayoutStrategy::ParentRank,
        LocalH2BirthStateStrategy::Tagged,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h2_persistence_profiled(
    slab_id: usize,
    block: &ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
) -> (SlabH2Summary, H2SlabPreparationProfile) {
    process_slab_h2_persistence_profiled_with_memory_audit(
        slab_id,
        block,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        neighbor_root_check,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        sweep_diagnostics,
        false,
    )
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h2_persistence_profiled_with_memory_audit(
    slab_id: usize,
    block: &ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
    memory_audit: bool,
) -> (SlabH2Summary, H2SlabPreparationProfile) {
    let order_start = Instant::now();
    let order = sorted_scalar_indices(&block.values, block.pixel_type);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    if memory_audit {
        emit_memory_snapshot("scalar_h2_local", "after_scalar_order", Some(slab_id));
        println!(
            "PROFILE_MEMVEC scalar_h2_local stage=after_scalar_order slab={} input_values_len={} input_values_capacity={} input_values_capacity_bytes={} order_len={} order_capacity={} order_capacity_bytes={}",
            slab_id,
            block.values.len(),
            block.values.capacity(),
            vec_capacity_bytes(&block.values),
            order.len(),
            order.capacity(),
            vec_capacity_bytes(&order),
        );
    }
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h2_persistence_values(
        slab_id,
        block.z0,
        block.shape,
        &block.values,
        order,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        neighbor_root_check,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        sweep_diagnostics,
        memory_audit,
    );
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    (
        summary,
        H2SlabPreparationProfile {
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
            direct_parent_checks: union_find_stats.direct_parent_checks,
            direct_parent_hits: union_find_stats.direct_parent_hits,
            avoided_neighbor_find_calls: union_find_stats.avoided_neighbor_find_calls,
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
pub(crate) fn process_slab_h2_persistence_f32_native(
    slab_id: usize,
    block: &F32ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
) -> SlabH2Summary<F32Key> {
    process_slab_h2_persistence_f32_native_profiled(
        slab_id,
        block,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        NeighborKernelStrategy::Generic,
        RepresentativeActiveCheckStrategy::Recheck,
        UnionKernelStrategy::Conventional,
        NeighborRootCheckStrategy::Find,
        ActiveStateStrategy::Separate,
        InterfaceStateStrategy::Vector,
        UnionFindLayoutStrategy::ParentRank,
        LocalH2BirthStateStrategy::Tagged,
        false,
    )
    .0
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h2_persistence_f32_native_profiled(
    slab_id: usize,
    block: &F32ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
) -> (SlabH2Summary<F32Key>, H2SlabPreparationProfile) {
    process_slab_h2_persistence_f32_native_profiled_with_memory_audit(
        slab_id,
        block,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        neighbor_root_check,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        sweep_diagnostics,
        false,
    )
}

#[allow(clippy::too_many_arguments)] // Explicit ablation knobs are intentionally passed separately.
pub(crate) fn process_slab_h2_persistence_f32_native_profiled_with_memory_audit(
    slab_id: usize,
    block: &F32ScalarBlock,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
    neighbor_kernel: NeighborKernelStrategy,
    representative_active_check: RepresentativeActiveCheckStrategy,
    union_kernel: UnionKernelStrategy,
    neighbor_root_check: NeighborRootCheckStrategy,
    active_state: ActiveStateStrategy,
    interface_state: InterfaceStateStrategy,
    uf_layout: UnionFindLayoutStrategy,
    local_h2_birth_state: LocalH2BirthStateStrategy,
    sweep_diagnostics: bool,
    memory_audit: bool,
) -> (SlabH2Summary<F32Key>, H2SlabPreparationProfile) {
    let order_start = Instant::now();
    let order = sorted_f32_indices(&block.values);
    let scalar_order_seconds = order_start.elapsed().as_secs_f64();
    if memory_audit {
        emit_memory_snapshot("scalar_h2_local", "after_scalar_order", Some(slab_id));
        println!(
            "PROFILE_MEMVEC scalar_h2_local stage=after_scalar_order slab={} input_values_len={} input_values_capacity={} input_values_capacity_bytes={} order_len={} order_capacity={} order_capacity_bytes={}",
            slab_id,
            block.values.len(),
            block.values.capacity(),
            vec_capacity_bytes(&block.values),
            order.len(),
            order.capacity(),
            vec_capacity_bytes(&order),
        );
    }
    let sweep_start = Instant::now();
    let (summary, pruning_stats, union_find_stats) = process_slab_h2_persistence_values(
        slab_id,
        block.z0,
        block.shape,
        &block.values,
        order,
        global_width,
        global_height,
        global_depth,
        background_connectivity,
        neighbor_kernel,
        representative_active_check,
        union_kernel,
        neighbor_root_check,
        active_state,
        interface_state,
        uf_layout,
        local_h2_birth_state,
        sweep_diagnostics,
        memory_audit,
    );
    let local_sweep_seconds = sweep_start.elapsed().as_secs_f64();
    (
        summary,
        H2SlabPreparationProfile {
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
            direct_parent_checks: union_find_stats.direct_parent_checks,
            direct_parent_hits: union_find_stats.direct_parent_hits,
            avoided_neighbor_find_calls: union_find_stats.avoided_neighbor_find_calls,
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

fn extract_boundary_face_values_only<K: LocalScalarKey>(
    shape: [usize; 3],
    values_in: &[K],
    z_local: usize,
) -> BoundaryFaceNodes<K> {
    let width = shape[0];
    let height = shape[1];
    let slice_size = width * height;
    let start = z_local * slice_size;
    let end = start + slice_size;
    BoundaryFaceNodes {
        width,
        height,
        // The direct hierarchical H2 path never consumes explicit boundary
        // node IDs: face IDs are deterministic from face index and slab depth.
        // Avoid two u32 arrays per leaf and keep only the scalar face values.
        node_ids: Vec::new(),
        values: values_in[start..end].to_vec(),
    }
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
            let face_index = y * width + x;
            let index = z_local * slice_size + face_index;
            node_ids[face_index] = face_node_id(z_local, shape[2], face_index, slice_size);
            values[face_index] = values_in[index];
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

fn process_all_slabs_h2(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<SlabH2Summary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);
    let [global_width, global_height, global_depth] = volume.shape();

    println!(
        "Processing {} scalar H2-persistence slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabH2Summary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_h2_persistence(
                slab_id,
                &block,
                global_width,
                global_height,
                global_depth,
                background_connectivity,
            ))
        })
        .collect();

    let mut summaries = parallel_results
        .into_iter()
        .collect::<Result<Vec<SlabH2Summary>>>()?;
    summaries.sort_by_key(|summary| summary.slab_id);
    Ok(summaries)
}

fn build_interface_offsets(summaries: &[SlabH2Summary]) -> Vec<u32> {
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
    Outside { node: u32 },
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
            GlobalEventKind::Outside { .. } => 0,
            GlobalEventKind::Attach(_) => 1,
            GlobalEventKind::Interface { .. } => 2,
            GlobalEventKind::Cross { .. } => 3,
        }
    }
}

#[derive(Debug)]
pub(crate) struct GlobalBackgroundPersistenceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<BackgroundBirth>,
    outside_node: u32,
}

impl GlobalBackgroundPersistenceUnionFind {
    pub(crate) fn new(interface_births: Vec<ScalarKey>) -> Self {
        assert!(interface_births.len() < u32::MAX as usize);
        let outside_node = interface_births.len() as u32;
        let mut birth: Vec<BackgroundBirth> = interface_births
            .into_iter()
            .map(BackgroundBirth::Finite)
            .collect();
        birth.push(BackgroundBirth::Outside);

        Self {
            parent: (0..birth.len() as u32).collect(),
            rank: vec![0; birth.len()],
            birth,
            outside_node,
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

    pub(crate) fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let surviving_birth = older_background_birth(birth_a, birth_b);
        let pair = pair_from_background_merge(birth_a, birth_b, merge_value);

        let rank_a = self.rank[root_a as usize];
        let rank_b = self.rank[root_b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.parent[root_b as usize] = root_a;
        if rank_a == rank_b {
            self.rank[root_a as usize] += 1;
        }
        self.birth[root_a as usize] = surviving_birth;
        pair
    }

    pub(crate) fn attach_branch(
        &mut self,
        interface_node: u32,
        branch_birth: ScalarKey,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        let root = self.find(interface_node);
        let root_birth = self.birth[root as usize];
        let branch = BackgroundBirth::Finite(branch_birth);
        self.birth[root as usize] = older_background_birth(root_birth, branch);
        pair_from_background_merge(root_birth, branch, merge_value)
    }

    pub(crate) fn connect_outside(
        &mut self,
        interface_node: u32,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        self.union_with_persistence(interface_node, self.outside_node, merge_value)
    }

    pub(crate) fn capacity_bytes(&self) -> (u64, u64, u64) {
        (
            self.parent.capacity() as u64 * core::mem::size_of::<u32>() as u64,
            self.rank.capacity() as u64 * core::mem::size_of::<u8>() as u64,
            self.birth.capacity() as u64 * core::mem::size_of::<BackgroundBirth>() as u64,
        )
    }

    pub(crate) fn remaining_finite_root_births(&mut self) -> Vec<ScalarKey> {
        let mut births = Vec::new();
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32
                && let BackgroundBirth::Finite(value) = self.birth[node]
            {
                births.push(value);
            }
        }
        births
    }
}

/// Streaming-only compact global H2 union-find.
///
/// The historical global reducer stores `BackgroundBirth`, which is a tagged
/// enum around an eight-byte `ScalarKey`. On the supported 64-bit target the
/// enum itself occupies sixteen bytes; the historical constructor can allocate
/// roughly twice the needed capacity when it appends the outside node. The
/// compact representation stores only the finite birth key and keeps the unique
/// outside component identifiable by
/// forcing `outside_node` to remain its union-find root.  Consequently no
/// per-node enum tag is required and the input `Vec<ScalarKey>` can be moved
/// directly into the union-find without a second birth allocation.
const GLOBAL_PACKED_ROOT_BASE: u32 = u32::MAX - u8::MAX as u32;

#[derive(Debug)]
struct GlobalUnionFindState {
    parent: Vec<u32>,
    rank: Option<Vec<u8>>,
    layout: GlobalH2UnionFindLayoutStrategy,
}

impl GlobalUnionFindState {
    fn new(len: usize, layout: GlobalH2UnionFindLayoutStrategy) -> Self {
        match layout {
            GlobalH2UnionFindLayoutStrategy::ParentRank => {
                assert!(len < u32::MAX as usize);
                Self {
                    parent: (0..len as u32).collect(),
                    rank: Some(vec![0; len]),
                    layout,
                }
            }
            GlobalH2UnionFindLayoutStrategy::Packed => {
                assert!(
                    len <= GLOBAL_PACKED_ROOT_BASE as usize,
                    "packed global H2 union-find reserves the top 256 u32 words for root ranks"
                );
                Self {
                    parent: vec![u32::MAX; len],
                    rank: None,
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
            GlobalH2UnionFindLayoutStrategy::ParentRank => word == node,
            GlobalH2UnionFindLayoutStrategy::Packed => word >= GLOBAL_PACKED_ROOT_BASE,
        }
    }

    #[inline]
    fn find(&mut self, mut node: u32) -> u32 {
        while !self.is_root(node) {
            let parent = self.parent[node as usize];
            debug_assert!(
                parent < GLOBAL_PACKED_ROOT_BASE
                    || matches!(self.layout, GlobalH2UnionFindLayoutStrategy::ParentRank)
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
            GlobalH2UnionFindLayoutStrategy::ParentRank => self
                .rank
                .as_ref()
                .expect("parent-rank global layout requires rank vector")[root as usize],
            GlobalH2UnionFindLayoutStrategy::Packed => {
                (u32::MAX - self.parent[root as usize]) as u8
            }
        }
    }

    #[inline]
    fn set_root_rank(&mut self, root: u32, rank: u8) {
        debug_assert!(self.is_root(root));
        match self.layout {
            GlobalH2UnionFindLayoutStrategy::ParentRank => {
                self.rank
                    .as_mut()
                    .expect("parent-rank global layout requires rank vector")[root as usize] = rank;
            }
            GlobalH2UnionFindLayoutStrategy::Packed => {
                self.parent[root as usize] = Self::packed_root_word(rank);
            }
        }
    }

    #[inline]
    fn link_root_under(&mut self, child_root: u32, parent_root: u32) {
        debug_assert!(self.is_root(child_root));
        debug_assert!(self.is_root(parent_root));
        debug_assert!(
            parent_root < GLOBAL_PACKED_ROOT_BASE
                || matches!(self.layout, GlobalH2UnionFindLayoutStrategy::ParentRank)
        );
        self.parent[child_root as usize] = parent_root;
    }

    fn capacity_bytes(&self) -> (u64, u64) {
        (
            self.parent.capacity() as u64 * core::mem::size_of::<u32>() as u64,
            self.rank
                .as_ref()
                .map(|rank| rank.capacity() as u64 * core::mem::size_of::<u8>() as u64)
                .unwrap_or(0),
        )
    }
}

#[derive(Debug)]
pub(crate) struct CompactGlobalBackgroundPersistenceUnionFind<K = ScalarKey> {
    uf_state: GlobalUnionFindState,
    birth: Vec<K>,
    outside_node: u32,
}

impl<K: LocalScalarKey> CompactGlobalBackgroundPersistenceUnionFind<K> {
    pub(crate) fn new(interface_births: Vec<K>, layout: GlobalH2UnionFindLayoutStrategy) -> Self {
        assert!(interface_births.len() < u32::MAX as usize);
        let outside_node = interface_births.len() as u32;
        let node_count = interface_births.len() + 1;
        Self {
            uf_state: GlobalUnionFindState::new(node_count, layout),
            birth: interface_births,
            outside_node,
        }
    }

    #[inline]
    fn find(&mut self, x: u32) -> u32 {
        self.uf_state.find(x)
    }

    #[inline]
    fn force_under_outside(&mut self, finite_root: u32) {
        debug_assert_ne!(finite_root, self.outside_node);
        debug_assert!(self.uf_state.is_root(finite_root));
        debug_assert!(self.uf_state.is_root(self.outside_node));
        let child_rank = self.uf_state.root_rank(finite_root);
        self.uf_state
            .link_root_under(finite_root, self.outside_node);
        let needed_rank = child_rank.saturating_add(1);
        if self.uf_state.root_rank(self.outside_node) < needed_rank {
            self.uf_state.set_root_rank(self.outside_node, needed_rank);
        }
    }

    pub(crate) fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        merge_value: K,
    ) -> Option<FinitePair<K>> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        // The outside component is topologically older than every finite
        // background component.  Preserve that fact structurally by keeping
        // the distinguished outside node as its representative.
        if root_a == self.outside_node || root_b == self.outside_node {
            let finite_root = if root_a == self.outside_node {
                root_b
            } else {
                root_a
            };
            let younger_birth = self.birth[finite_root as usize];
            self.force_under_outside(finite_root);
            return Some(FinitePair {
                birth: merge_value,
                death: younger_birth,
            });
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let surviving_birth = birth_a.max(birth_b);
        let pair = Some(FinitePair {
            birth: merge_value,
            death: birth_a.min(birth_b),
        });

        let rank_a = self.uf_state.root_rank(root_a);
        let rank_b = self.uf_state.root_rank(root_b);
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.uf_state.link_root_under(root_b, root_a);
        if rank_a == rank_b {
            self.uf_state
                .set_root_rank(root_a, rank_a.saturating_add(1));
        }
        self.birth[root_a as usize] = surviving_birth;
        pair
    }

    pub(crate) fn attach_branch(
        &mut self,
        interface_node: u32,
        branch_birth: K,
        merge_value: K,
    ) -> Option<FinitePair<K>> {
        let root = self.find(interface_node);
        if root == self.outside_node {
            return Some(FinitePair {
                birth: merge_value,
                death: branch_birth,
            });
        }

        let root_birth = self.birth[root as usize];
        self.birth[root as usize] = root_birth.max(branch_birth);
        Some(FinitePair {
            birth: merge_value,
            death: root_birth.min(branch_birth),
        })
    }

    pub(crate) fn connect_outside(
        &mut self,
        interface_node: u32,
        merge_value: K,
    ) -> Option<FinitePair<K>> {
        self.union_with_persistence(interface_node, self.outside_node, merge_value)
    }

    pub(crate) fn capacity_bytes(&self) -> (u64, u64, u64) {
        let (parent_bytes, rank_bytes) = self.uf_state.capacity_bytes();
        (
            parent_bytes,
            rank_bytes,
            self.birth.capacity() as u64 * core::mem::size_of::<K>() as u64,
        )
    }

    pub(crate) fn remaining_finite_root_births(&mut self) -> Vec<K> {
        let mut births = Vec::new();
        for node in 0..self.birth.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 {
                births.push(self.birth[node]);
            }
        }
        births
    }
}

#[derive(Debug)]
pub(crate) enum StreamingGlobalBackgroundPersistenceUnionFind {
    Tagged(GlobalBackgroundPersistenceUnionFind),
    Compact(CompactGlobalBackgroundPersistenceUnionFind),
}

impl StreamingGlobalBackgroundPersistenceUnionFind {
    pub(crate) fn new(
        interface_births: Vec<ScalarKey>,
        strategy: GlobalH2BirthStateStrategy,
        global_uf_layout: GlobalH2UnionFindLayoutStrategy,
    ) -> Self {
        match strategy {
            GlobalH2BirthStateStrategy::Tagged => {
                assert!(
                    matches!(
                        global_uf_layout,
                        GlobalH2UnionFindLayoutStrategy::ParentRank
                    ),
                    "tagged global H2 reference supports only parent-rank layout"
                );
                Self::Tagged(GlobalBackgroundPersistenceUnionFind::new(interface_births))
            }
            GlobalH2BirthStateStrategy::Compact => {
                Self::Compact(CompactGlobalBackgroundPersistenceUnionFind::new(
                    interface_births,
                    global_uf_layout,
                ))
            }
        }
    }

    pub(crate) fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        match self {
            Self::Tagged(uf) => uf.union_with_persistence(a, b, merge_value),
            Self::Compact(uf) => uf.union_with_persistence(a, b, merge_value),
        }
    }

    pub(crate) fn attach_branch(
        &mut self,
        interface_node: u32,
        branch_birth: ScalarKey,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        match self {
            Self::Tagged(uf) => uf.attach_branch(interface_node, branch_birth, merge_value),
            Self::Compact(uf) => uf.attach_branch(interface_node, branch_birth, merge_value),
        }
    }

    pub(crate) fn connect_outside(
        &mut self,
        interface_node: u32,
        merge_value: ScalarKey,
    ) -> Option<FinitePair> {
        match self {
            Self::Tagged(uf) => uf.connect_outside(interface_node, merge_value),
            Self::Compact(uf) => uf.connect_outside(interface_node, merge_value),
        }
    }

    pub(crate) fn capacity_bytes(&self) -> (u64, u64, u64) {
        match self {
            Self::Tagged(uf) => uf.capacity_bytes(),
            Self::Compact(uf) => uf.capacity_bytes(),
        }
    }

    pub(crate) fn remaining_finite_root_births(&mut self) -> Vec<ScalarKey> {
        match self {
            Self::Tagged(uf) => uf.remaining_finite_root_births(),
            Self::Compact(uf) => uf.remaining_finite_root_births(),
        }
    }
}

fn build_global_interface_births(summaries: &[SlabH2Summary], offsets: &[u32]) -> Vec<ScalarKey> {
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

fn push_finite_interval(intervals: &mut Vec<ScalarH2Interval>, pair: FinitePair) {
    if pair.birth < pair.death {
        intervals.push(ScalarH2Interval {
            birth: pair.birth,
            death: pair.death,
        });
    }
}

fn reduce_h2_persistence(
    summaries: &[SlabH2Summary],
    connectivity: Connectivity,
) -> Result<Vec<ScalarH2Interval>> {
    let offsets = build_interface_offsets(summaries);
    let births = build_global_interface_births(summaries, &offsets);
    let mut global_union_find = GlobalBackgroundPersistenceUnionFind::new(births);
    let mut intervals = Vec::new();
    let mut events = Vec::new();

    for summary in summaries {
        for &pair in &summary.finalized_pairs {
            push_finite_interval(&mut intervals, pair);
        }

        for event in &summary.outside_events {
            events.push(GlobalEvent {
                value: event.value,
                kind: GlobalEventKind::Outside {
                    node: global_node_id(&offsets, summary.slab_id, event.interface_node),
                },
            });
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

        let stats = sparsify_scalar_superlevel_interface(
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
        "Scalar H2 interface sparsification retained {retained_cross_edges} of {candidate_cross_edges} candidate edges"
    );

    // Stable sorting preserves deterministic slab/event order within a category.
    events.sort_by_key(|event| (Reverse(event.value), event.priority()));

    for event in events {
        let pair = match event.kind {
            GlobalEventKind::Outside { node } => {
                global_union_find.connect_outside(node, event.value)
            }
            GlobalEventKind::Attach(attach) => {
                global_union_find.attach_branch(attach.node, attach.branch_birth, event.value)
            }
            GlobalEventKind::Interface { a, b } | GlobalEventKind::Cross { a, b } => {
                global_union_find.union_with_persistence(a, b, event.value)
            }
        };

        if let Some(pair) = pair {
            push_finite_interval(&mut intervals, pair);
        }
    }

    let remaining = global_union_find.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "scalar H2 reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    intervals.sort_unstable_by_key(|interval| (interval.birth, interval.death));
    Ok(intervals)
}

pub fn compute_h2_persistence_scalar_zslabs(
    volume: &ScalarTiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<ScalarH2Interval>> {
    let start = Instant::now();
    println!("Computing in-memory slabwise scalar H2 persistence...");
    let summaries = process_all_slabs_h2(volume, slab_depth, background_connectivity)?;
    println!("Processed {} scalar H2 slab summaries", summaries.len());
    let intervals = reduce_h2_persistence(&summaries, background_connectivity)?;
    println!(
        "Scalar H2 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(intervals)
}

pub fn write_h2_scalar_persistence_csv(path: &Path, intervals: &[ScalarH2Interval]) -> Result<()> {
    let mut file = AtomicOutput::create(path)
        .with_context(|| format!("could not create scalar H2 output {path:?}"))?;
    writeln!(file, "birth,death")?;
    for interval in intervals {
        writeln!(file, "{},{}", interval.birth, interval.death)?;
    }
    file.commit()
}

pub fn betti2_curve_from_scalar_h2_intervals(
    intervals: &[ScalarH2Interval],
) -> Vec<(ScalarKey, i64)> {
    let mut delta = BTreeMap::<ScalarKey, i64>::new();
    for interval in intervals {
        *delta.entry(interval.birth).or_default() += 1;
        *delta.entry(interval.death).or_default() -= 1;
    }

    let mut beta2 = 0i64;
    let mut sparse = Vec::new();
    let mut previous = None;
    for (value, change) in delta {
        beta2 += change;
        if previous != Some(beta2) {
            sparse.push((value, beta2));
            previous = Some(beta2);
        }
    }
    sparse
}

pub fn write_scalar_betti2_curve_csv(path: &Path, curve: &[(ScalarKey, i64)]) -> Result<()> {
    let mut file = AtomicOutput::create(path)
        .with_context(|| format!("could not create scalar Betti-2 output {path:?}"))?;
    writeln!(file, "threshold,betti2")?;
    for &(threshold, beta2) in curve {
        writeln!(file, "{threshold},{beta2}")?;
    }
    file.commit()
}

pub fn compute_h2_scalar_batch(
    root: &Path,
    output_root: &Path,
    slab_depth: usize,
    background_connectivity: Connectivity,
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
            "=== Scalar H2 dataset {} of {}: {} ===",
            index + 1,
            directories.len(),
            directory.display()
        );

        let volume = ScalarTiffStackReader::open(directory)?;
        print_scalar_volume_info(&volume);
        let intervals =
            compute_h2_persistence_scalar_zslabs(&volume, slab_depth, background_connectivity)?;

        write_h2_scalar_persistence_csv(
            &dataset_output.join("h2_persistence_scalar.csv"),
            &intervals,
        )?;
        let curve = betti2_curve_from_scalar_h2_intervals(&intervals);
        write_scalar_betti2_curve_csv(
            &dataset_output.join("h2_reconstructed_betti2_scalar_curve.csv"),
            &curve,
        )?;
    }

    Ok(directories.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widen_f32_summary(summary: SlabH2Summary<F32Key>) -> SlabH2Summary {
        SlabH2Summary {
            slab_id: summary.slab_id,
            finalized_pairs: summary
                .finalized_pairs
                .into_iter()
                .map(|pair| FinitePair {
                    birth: pair.birth.to_scalar_key(),
                    death: pair.death.to_scalar_key(),
                })
                .collect(),
            attach_events: summary
                .attach_events
                .into_iter()
                .map(|event| AttachEvent {
                    value: event.value.to_scalar_key(),
                    interface_node: event.interface_node,
                    branch_birth: event.branch_birth.to_scalar_key(),
                })
                .collect(),
            outside_events: summary
                .outside_events
                .into_iter()
                .map(|event| OutsideEvent {
                    value: event.value.to_scalar_key(),
                    interface_node: event.interface_node,
                })
                .collect(),
            interface_merge_events: summary
                .interface_merge_events
                .into_iter()
                .map(|event| InterfaceMergeEvent {
                    value: event.value.to_scalar_key(),
                    a: event.a,
                    b: event.b,
                })
                .collect(),
            interface_node_count: summary.interface_node_count,
            z_min_face: BoundaryFaceNodes {
                width: summary.z_min_face.width,
                height: summary.z_min_face.height,
                node_ids: summary.z_min_face.node_ids,
                values: summary
                    .z_min_face
                    .values
                    .into_iter()
                    .map(F32Key::to_scalar_key)
                    .collect(),
            },
            z_max_face: BoundaryFaceNodes {
                width: summary.z_max_face.width,
                height: summary.z_max_face.height,
                node_ids: summary.z_max_face.node_ids,
                values: summary
                    .z_max_face
                    .values
                    .into_iter()
                    .map(F32Key::to_scalar_key)
                    .collect(),
            },
        }
    }

    #[test]
    fn native_f32_local_h2_matches_legacy_wide_path() {
        let mut source = [1.0f32; 27];
        source[13] = 4.0;
        let wide = ScalarBlock {
            z0: 0,
            shape: [3, 3, 3],
            values: source
                .iter()
                .copied()
                .map(|value| ScalarKey::from_f32(value).unwrap())
                .collect(),
            pixel_type: crate::io_scalar::ScalarPixelType::F32,
        };
        let native = F32ScalarBlock {
            z0: 0,
            shape: [3, 3, 3],
            values: source
                .iter()
                .copied()
                .map(|value| crate::scalar::F32Key::from_f32(value).unwrap())
                .collect(),
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let wide_summary = process_slab_h2_persistence(0, &wide, 3, 3, 3, connectivity);
            let native_summary =
                process_slab_h2_persistence_f32_native(0, &native, 3, 3, 3, connectivity);
            let wide_intervals = reduce_h2_persistence(&[wide_summary], connectivity).unwrap();
            let native_intervals =
                reduce_h2_persistence(&[widen_f32_summary(native_summary)], connectivity).unwrap();
            assert_eq!(native_intervals, wide_intervals);
        }
    }

    #[test]
    fn scalar_shell_has_the_expected_half_open_h2_interval() {
        let mut values = vec![ScalarKey::from_u16(1); 27];
        values[13] = ScalarKey::from_u16(4);
        let block = ScalarBlock {
            z0: 0,
            shape: [3, 3, 3],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let summary = process_slab_h2_persistence(0, &block, 3, 3, 3, connectivity);
            let intervals = reduce_h2_persistence(&[summary], connectivity).unwrap();

            assert_eq!(
                intervals,
                vec![ScalarH2Interval {
                    birth: ScalarKey::from_u16(1),
                    death: ScalarKey::from_u16(4),
                }]
            );
        }
    }
    #[test]
    fn root_carrying_h2_matches_conventional_and_uses_fewer_finds() {
        let values = (0..64)
            .map(|index| ScalarKey::from_u16(((index * 19 + index / 5) % 13) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [4, 4, 4],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let (conventional, conventional_profile) = process_slab_h2_persistence_profiled(
                0,
                &block,
                4,
                4,
                4,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::Conventional,
                NeighborRootCheckStrategy::Find,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                LocalH2BirthStateStrategy::Tagged,
                true,
            );
            let (root_carrying, root_profile) = process_slab_h2_persistence_profiled(
                0,
                &block,
                4,
                4,
                4,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::RootCarrying,
                NeighborRootCheckStrategy::Find,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                LocalH2BirthStateStrategy::Tagged,
                true,
            );
            let conventional_intervals =
                reduce_h2_persistence(&[conventional], connectivity).unwrap();
            let root_intervals = reduce_h2_persistence(&[root_carrying], connectivity).unwrap();
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
    fn parent_shortcut_h2_matches_find_root_carrying_and_avoids_neighbor_finds() {
        let values = (0..125)
            .map(|index| ScalarKey::from_u16(((index * 23 + index / 7) % 17) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [5, 5, 5],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let (find_summary, find_profile) = process_slab_h2_persistence_profiled(
                0,
                &block,
                5,
                5,
                5,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::RootCarrying,
                NeighborRootCheckStrategy::Find,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                LocalH2BirthStateStrategy::Tagged,
                true,
            );
            let (shortcut_summary, shortcut_profile) = process_slab_h2_persistence_profiled(
                0,
                &block,
                5,
                5,
                5,
                connectivity,
                NeighborKernelStrategy::InteriorFast,
                RepresentativeActiveCheckStrategy::Recheck,
                UnionKernelStrategy::RootCarrying,
                NeighborRootCheckStrategy::ParentShortcut,
                ActiveStateStrategy::Separate,
                InterfaceStateStrategy::Vector,
                UnionFindLayoutStrategy::ParentRank,
                LocalH2BirthStateStrategy::Tagged,
                true,
            );
            let find_intervals = reduce_h2_persistence(&[find_summary], connectivity).unwrap();
            let shortcut_intervals =
                reduce_h2_persistence(&[shortcut_summary], connectivity).unwrap();
            assert_eq!(shortcut_intervals, find_intervals);
            assert_eq!(find_profile.direct_parent_checks, 0);
            assert_eq!(find_profile.direct_parent_hits, 0);
            assert_eq!(find_profile.avoided_neighbor_find_calls, 0);
            assert_eq!(
                shortcut_profile.direct_parent_checks,
                shortcut_profile.root_carry_attempts
            );
            assert_eq!(
                shortcut_profile.direct_parent_hits,
                shortcut_profile.avoided_neighbor_find_calls
            );
            assert!(shortcut_profile.find_calls <= find_profile.find_calls);
        }
    }

    #[test]
    fn parent_sentinel_h2_matches_separate_active_state() {
        let values = (0..125)
            .map(|index| ScalarKey::from_u16(((index * 37 + index / 9) % 23) as u16))
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
                let (summary, profile) = process_slab_h2_persistence_profiled(
                    0,
                    &block,
                    5,
                    5,
                    5,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    NeighborRootCheckStrategy::ParentShortcut,
                    active_state,
                    InterfaceStateStrategy::Vector,
                    UnionFindLayoutStrategy::ParentRank,
                    LocalH2BirthStateStrategy::Tagged,
                    true,
                );
                let intervals = reduce_h2_persistence(&[summary], connectivity).unwrap();
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
    fn root_invariant_h2_matches_interface_vector() {
        let values = (0..216)
            .map(|index| ScalarKey::from_u16(((index * 43 + index / 13) % 31) as u16))
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
                let (summary, profile) = process_slab_h2_persistence_profiled(
                    0,
                    &block,
                    6,
                    6,
                    6,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    NeighborRootCheckStrategy::ParentShortcut,
                    ActiveStateStrategy::Separate,
                    interface_state,
                    UnionFindLayoutStrategy::ParentRank,
                    LocalH2BirthStateStrategy::Tagged,
                    true,
                );
                let intervals = reduce_h2_persistence(&[summary], connectivity).unwrap();
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
    fn compact_local_h2_birth_storage_round_trips_tagged_state() {
        let values: Vec<_> = [0.0f32, 1.5, -2.0, 8.0]
            .into_iter()
            .map(|value| crate::scalar::F32Key::from_f32(value).unwrap())
            .collect();
        let mut tagged = LocalH2BirthStorage::new(&values, LocalH2BirthStateStrategy::Tagged);
        let mut compact = LocalH2BirthStorage::new(&values, LocalH2BirthStateStrategy::Compact);

        for index in 0..values.len() {
            assert_eq!(tagged.get(index), compact.get(index));
        }
        tagged.set(1, LocalBackgroundBirth::Outside);
        compact.set(1, LocalBackgroundBirth::Outside);
        assert_eq!(tagged.get(1), compact.get(1));

        tagged.set(2, LocalBackgroundBirth::Finite(values[3]));
        compact.set(2, LocalBackgroundBirth::Finite(values[3]));
        assert_eq!(tagged.get(2), compact.get(2));
        assert!(compact.capacity_bytes() < tagged.capacity_bytes());
    }

    #[test]
    fn compact_local_h2_birth_state_matches_tagged_persistence() {
        let values = (0..216)
            .map(|index| ScalarKey::from_u16(((index * 61 + index / 11) % 47) as u16))
            .collect();
        let block = ScalarBlock {
            z0: 0,
            shape: [6, 6, 6],
            values,
            pixel_type: crate::io_scalar::ScalarPixelType::U16,
        };

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let mut reference = None;
            for birth_state in [
                LocalH2BirthStateStrategy::Tagged,
                LocalH2BirthStateStrategy::Compact,
            ] {
                let (summary, _) = process_slab_h2_persistence_profiled(
                    0,
                    &block,
                    6,
                    6,
                    6,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    NeighborRootCheckStrategy::ParentShortcut,
                    ActiveStateStrategy::ParentSentinel,
                    InterfaceStateStrategy::RootInvariant,
                    UnionFindLayoutStrategy::Packed,
                    birth_state,
                    false,
                );
                let intervals = reduce_h2_persistence(&[summary], connectivity).unwrap();
                if let Some(expected) = reference.as_ref() {
                    assert_eq!(&intervals, expected);
                } else {
                    reference = Some(intervals);
                }
            }
        }
    }

    #[test]
    fn packed_h2_matches_parent_rank_layout() {
        let values = (0..216)
            .map(|index| ScalarKey::from_u16(((index * 53 + index / 9) % 41) as u16))
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
                let (summary, profile) = process_slab_h2_persistence_profiled(
                    0,
                    &block,
                    6,
                    6,
                    6,
                    connectivity,
                    NeighborKernelStrategy::InteriorFast,
                    RepresentativeActiveCheckStrategy::Recheck,
                    UnionKernelStrategy::RootCarrying,
                    NeighborRootCheckStrategy::ParentShortcut,
                    ActiveStateStrategy::ParentSentinel,
                    InterfaceStateStrategy::RootInvariant,
                    uf_layout,
                    LocalH2BirthStateStrategy::Tagged,
                    true,
                );
                let intervals = reduce_h2_persistence(&[summary], connectivity).unwrap();
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

    #[test]
    fn compact_global_h2_birth_state_matches_tagged_reference() {
        let births = vec![
            ScalarKey::from_u16(10),
            ScalarKey::from_u16(20),
            ScalarKey::from_u16(30),
            ScalarKey::from_u16(40),
            ScalarKey::from_u16(50),
        ];
        let mut tagged = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births.clone(),
            GlobalH2BirthStateStrategy::Tagged,
            GlobalH2UnionFindLayoutStrategy::ParentRank,
        );
        let mut compact = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births,
            GlobalH2BirthStateStrategy::Compact,
            GlobalH2UnionFindLayoutStrategy::ParentRank,
        );

        let mut tagged_pairs = Vec::new();
        let mut compact_pairs = Vec::new();
        let mut compare = |a: Option<FinitePair>, b: Option<FinitePair>| {
            assert_eq!(a.map(|p| (p.birth, p.death)), b.map(|p| (p.birth, p.death)));
            if let Some(pair) = a {
                tagged_pairs.push((pair.birth, pair.death));
            }
            if let Some(pair) = b {
                compact_pairs.push((pair.birth, pair.death));
            }
        };

        compare(
            tagged.union_with_persistence(0, 1, ScalarKey::from_u16(8)),
            compact.union_with_persistence(0, 1, ScalarKey::from_u16(8)),
        );
        compare(
            tagged.attach_branch(0, ScalarKey::from_u16(25), ScalarKey::from_u16(7)),
            compact.attach_branch(0, ScalarKey::from_u16(25), ScalarKey::from_u16(7)),
        );
        compare(
            tagged.connect_outside(1, ScalarKey::from_u16(6)),
            compact.connect_outside(1, ScalarKey::from_u16(6)),
        );
        compare(
            tagged.union_with_persistence(2, 3, ScalarKey::from_u16(5)),
            compact.union_with_persistence(2, 3, ScalarKey::from_u16(5)),
        );
        compare(
            tagged.union_with_persistence(3, 4, ScalarKey::from_u16(4)),
            compact.union_with_persistence(3, 4, ScalarKey::from_u16(4)),
        );
        compare(
            tagged.connect_outside(4, ScalarKey::from_u16(3)),
            compact.connect_outside(4, ScalarKey::from_u16(3)),
        );

        assert_eq!(tagged_pairs, compact_pairs);
        assert_eq!(
            tagged.remaining_finite_root_births(),
            compact.remaining_finite_root_births()
        );
        let (_, _, tagged_birth_bytes) = tagged.capacity_bytes();
        let (_, _, compact_birth_bytes) = compact.capacity_bytes();
        assert!(compact_birth_bytes < tagged_birth_bytes);
    }

    #[test]
    fn compact_global_h2_birth_state_matches_tagged_over_many_operations() {
        fn pair_tuple(pair: Option<FinitePair>) -> Option<(ScalarKey, ScalarKey)> {
            pair.map(|p| (p.birth, p.death))
        }

        let births = (0..64u16)
            .map(|i| ScalarKey::from_u16((i * 37 + 11) % 251))
            .collect::<Vec<_>>();
        let mut tagged = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births.clone(),
            GlobalH2BirthStateStrategy::Tagged,
            GlobalH2UnionFindLayoutStrategy::ParentRank,
        );
        let mut compact = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births,
            GlobalH2BirthStateStrategy::Compact,
            GlobalH2UnionFindLayoutStrategy::ParentRank,
        );

        let mut state = 0x9e37_79b9_u32;
        for step in 0..2000u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = state % 64;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = state % 64;
            let merge_value = ScalarKey::from_u16((step % 257) as u16);
            let branch_birth = ScalarKey::from_u16(((step * 17 + 29) % 263) as u16);

            let (tagged_pair, compact_pair) = match state % 3 {
                0 => (
                    tagged.union_with_persistence(a, b, merge_value),
                    compact.union_with_persistence(a, b, merge_value),
                ),
                1 => (
                    tagged.attach_branch(a, branch_birth, merge_value),
                    compact.attach_branch(a, branch_birth, merge_value),
                ),
                _ => (
                    tagged.connect_outside(a, merge_value),
                    compact.connect_outside(a, merge_value),
                ),
            };
            assert_eq!(pair_tuple(tagged_pair), pair_tuple(compact_pair));
        }

        assert_eq!(
            tagged.remaining_finite_root_births(),
            compact.remaining_finite_root_births()
        );
    }

    #[test]
    fn compact_global_h2_packed_layout_matches_parent_rank() {
        fn pair_tuple(pair: Option<FinitePair>) -> Option<(ScalarKey, ScalarKey)> {
            pair.map(|p| (p.birth, p.death))
        }

        let births = (0..96u16)
            .map(|i| ScalarKey::from_u16((i * 53 + 7) % 509))
            .collect::<Vec<_>>();
        let mut parent_rank = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births.clone(),
            GlobalH2BirthStateStrategy::Compact,
            GlobalH2UnionFindLayoutStrategy::ParentRank,
        );
        let mut packed = StreamingGlobalBackgroundPersistenceUnionFind::new(
            births,
            GlobalH2BirthStateStrategy::Compact,
            GlobalH2UnionFindLayoutStrategy::Packed,
        );

        let mut state = 0x243f_6a88_u32;
        for step in 0..4000u32 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let a = state % 96;
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let b = state % 96;
            let merge_value = ScalarKey::from_u16((step % 521) as u16);
            let branch_birth = ScalarKey::from_u16(((step * 31 + 17) % 523) as u16);

            let (reference_pair, packed_pair) = match state % 3 {
                0 => (
                    parent_rank.union_with_persistence(a, b, merge_value),
                    packed.union_with_persistence(a, b, merge_value),
                ),
                1 => (
                    parent_rank.attach_branch(a, branch_birth, merge_value),
                    packed.attach_branch(a, branch_birth, merge_value),
                ),
                _ => (
                    parent_rank.connect_outside(a, merge_value),
                    packed.connect_outside(a, merge_value),
                ),
            };
            assert_eq!(pair_tuple(reference_pair), pair_tuple(packed_pair));
        }

        assert_eq!(
            parent_rank.remaining_finite_root_births(),
            packed.remaining_finite_root_births()
        );
        let (parent_bytes, rank_bytes, birth_bytes) = parent_rank.capacity_bytes();
        let (packed_parent_bytes, packed_rank_bytes, packed_birth_bytes) = packed.capacity_bytes();
        assert_eq!(parent_bytes, packed_parent_bytes);
        assert!(rank_bytes > 0);
        assert_eq!(packed_rank_bytes, 0);
        assert_eq!(birth_bytes, packed_birth_bytes);
    }
}
