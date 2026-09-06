use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::local_pruning::{H2ShellFacePruner, NeighborhoodComponentPruner};
use crate::local_uf_state::LocalUnionFindState;
use crate::merge_tree_common::{
    CompactH2BranchMerge, H2Branch, H2BranchMerge, H2PackedTreeRecorder, H2TreeRecorder, MergeTree,
    OUTSIDE_BRANCH_ID, PackedDeferredIdResolver,
};
use crate::scalar_stream_tuning::{ActiveStateStrategy, UnionFindLayoutStrategy};
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};

pub(crate) const NUM_U16_VALUES: usize = 65_536;

fn leaf_history_watch_filter_enabled() -> bool {
    !std::env::var("BETTI_HIER_LEAF_HISTORY_WATCH_FILTER")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

fn leaf_kernel_audit_enabled() -> bool {
    std::env::var("BETTI_HIER_LEAF_KERNEL_AUDIT")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
}

fn leaf_kernel_audit_sample_stride() -> u64 {
    std::env::var("BETTI_HIER_LEAF_AUDIT_SAMPLE_STRIDE")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|&stride| stride > 0)
        .unwrap_or(4096)
}

fn plateau_native_leaf_enabled() -> bool {
    !std::env::var("BETTI_HIER_PLATEAU_NATIVE_LEAF")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

fn h2_shell_pruning_enabled() -> bool {
    std::env::var("BETTI_HIER_H2_SHELL_PRUNE")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
}

fn h2_root_dedup_enabled() -> bool {
    !std::env::var("BETTI_HIER_H2_ROOT_DEDUP")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttachEvent {
    pub(crate) value: u16,
    pub(crate) interface_node: u32,
    pub(crate) branch: H2Branch,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InterfaceMergeEvent {
    pub(crate) value: u16,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

#[derive(Debug)]
pub(crate) struct BoundaryFaceNodes {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) node_ids: Vec<u32>,
    pub(crate) values: Vec<u16>,
    pub(crate) branch_ids: Vec<u64>,
}

#[derive(Debug)]
pub(crate) struct SlabH2MergeTreeSummary {
    pub(crate) slab_id: usize,
    pub(crate) local_merge_events: Vec<H2BranchMerge>,
    pub(crate) attach_events: Vec<AttachEvent>,
    pub(crate) interface_merge_events: Vec<InterfaceMergeEvent>,
    pub(crate) interface_node_count: u32,
    pub(crate) z_min_face: BoundaryFaceNodes,
    pub(crate) z_max_face: BoundaryFaceNodes,
}

#[derive(Debug)]
struct VoxelBuckets {
    offsets: Vec<usize>,
    indices: Vec<u32>,
}

fn build_voxel_buckets_u16(values: &[u16]) -> VoxelBuckets {
    let mut counts = vec![0usize; NUM_U16_VALUES];
    for &value in values {
        counts[value as usize] += 1;
    }

    let mut offsets = vec![0usize; NUM_U16_VALUES + 1];
    for value in 0..NUM_U16_VALUES {
        offsets[value + 1] = offsets[value] + counts[value];
    }

    let mut cursor = offsets.clone();
    let mut indices = vec![0u32; values.len()];
    for (idx, &value) in values.iter().enumerate() {
        let position = cursor[value as usize];
        indices[position] = idx as u32;
        cursor[value as usize] += 1;
    }

    VoxelBuckets { offsets, indices }
}

#[derive(Debug, Clone, Copy)]
struct LocalH2Merge {
    value: u16,
    child_local: u32,
    child_birth: u16,
    parent_local: u32,
}

#[derive(Debug, Clone, Copy)]
enum LocalAction {
    None,
    LocalMerge(LocalH2Merge),
    Attach(AttachEvent),
    InterfaceMerge(InterfaceMergeEvent),
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct H2LeafAttachStats {
    pub(crate) one_boundary_internal: u64,
    pub(crate) finalized_early: u64,
    pub(crate) propagated_attach: u64,
    pub(crate) local_history_input_events: u64,
    pub(crate) local_history_materialized_events: u64,
    pub(crate) local_history_retained_events: u64,
    pub(crate) local_history_contracted_zero_events: u64,
    pub(crate) local_history_repaired_parent_refs: u64,
    pub(crate) local_history_parent_watch_checks: u64,
    pub(crate) local_history_parent_lookup_skips: u64,
    pub(crate) local_history_parent_hash_lookups: u64,
    pub(crate) local_history_zero_fast_drops: u64,
    pub(crate) plateau_zero_local_merges_elided: u64,
    pub(crate) kernel_audit_activations: u64,
    pub(crate) kernel_audit_interior_voxels: u64,
    pub(crate) kernel_audit_boundary_voxels: u64,
    pub(crate) kernel_audit_interface_activations: u64,
    pub(crate) kernel_audit_global_boundary_activations: u64,
    pub(crate) kernel_audit_active_state_checks: u64,
    pub(crate) kernel_audit_active_neighbor_hits: u64,
    pub(crate) kernel_audit_representative_neighbors: u64,
    pub(crate) kernel_audit_pruner_cache_lookups: u64,
    pub(crate) kernel_audit_pruner_cache_hits: u64,
    pub(crate) kernel_audit_pruner_cache_misses: u64,
    pub(crate) kernel_audit_pruner_mask_computations: u64,
    pub(crate) kernel_audit_union_attempts: u64,
    pub(crate) kernel_audit_union_successes: u64,
    pub(crate) kernel_audit_union_already_connected: u64,
    pub(crate) kernel_audit_union_local_local: u64,
    pub(crate) kernel_audit_union_boundary_internal: u64,
    pub(crate) kernel_audit_union_interface_interface: u64,
    pub(crate) kernel_audit_union_outside_involved: u64,
    pub(crate) kernel_audit_current_root_reused: u64,
    pub(crate) kernel_audit_current_root_refinds: u64,
    pub(crate) kernel_audit_find_calls: u64,
    pub(crate) kernel_audit_find_parent_hops: u64,
    pub(crate) kernel_audit_find_max_hops: u64,
    pub(crate) kernel_audit_shell_face_checks: u64,
    pub(crate) kernel_audit_shell_active_face_hits: u64,
    pub(crate) kernel_audit_shell_extra_state_checks: u64,
    pub(crate) kernel_audit_shell_extra_active_hits: u64,
    pub(crate) kernel_audit_shell_representatives: u64,
    pub(crate) kernel_audit_root_dedup_inputs: u64,
    pub(crate) kernel_audit_root_dedup_unique: u64,
    pub(crate) kernel_audit_root_dedup_skipped: u64,
    pub(crate) kernel_audit_timing_samples: u64,
    pub(crate) kernel_audit_activation_bookkeeping_sample_ns: u64,
    pub(crate) kernel_audit_neighbor_pruning_sample_ns: u64,
    pub(crate) kernel_audit_union_action_sample_ns: u64,
    pub(crate) kernel_audit_bucket_build_ns: u64,
}

/// H2 counterpart of the compact leaf-local history contractor. Finite branch
/// IDs stay local until retained events are exported; `u32::MAX` represents
/// Outside. The watch bitset excludes Outside because no finite retained event
/// is indexed under the distinguished root.
#[derive(Debug)]
struct LocalMergeHistory {
    events: Vec<LocalH2Merge>,
    by_parent: HashMap<u32, Vec<usize>>,
    watched_parent_bits: Vec<u64>,
    inline_contract: bool,
    watch_filter: bool,
    input_events: u64,
    contracted_zero_events: u64,
    repaired_parent_refs: u64,
    parent_watch_checks: u64,
    parent_lookup_skips: u64,
    parent_hash_lookups: u64,
    zero_fast_drops: u64,
}

impl LocalMergeHistory {
    const OUTSIDE_LOCAL: u32 = u32::MAX;

    fn new(inline_contract: bool, watch_filter: bool, voxel_count: usize) -> Self {
        Self {
            events: Vec::new(),
            by_parent: HashMap::new(),
            watched_parent_bits: if inline_contract {
                vec![0u64; voxel_count.div_ceil(64)]
            } else {
                Vec::new()
            },
            inline_contract,
            watch_filter,
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
            parent_watch_checks: 0,
            parent_lookup_skips: 0,
            parent_hash_lookups: 0,
            zero_fast_drops: 0,
        }
    }

    #[inline]
    fn is_watched(&self, local: u32) -> bool {
        if local == Self::OUTSIDE_LOCAL {
            return false;
        }
        let index = local as usize;
        let word = index >> 6;
        let mask = 1u64 << (index & 63);
        self.watched_parent_bits
            .get(word)
            .is_some_and(|bits| bits & mask != 0)
    }

    #[inline]
    fn mark_watched(&mut self, local: u32) {
        if local == Self::OUTSIDE_LOCAL {
            return;
        }
        let index = local as usize;
        self.watched_parent_bits[index >> 6] |= 1u64 << (index & 63);
    }

    #[inline]
    fn clear_watched(&mut self, local: u32) {
        if local == Self::OUTSIDE_LOCAL {
            return;
        }
        let index = local as usize;
        self.watched_parent_bits[index >> 6] &= !(1u64 << (index & 63));
    }

    fn push(&mut self, merge: LocalH2Merge) {
        self.input_events += 1;
        if !self.inline_contract {
            self.events.push(merge);
            return;
        }

        let zero = merge.child_birth == merge.value;
        let had_parent_refs =
            self.redirect_existing(merge.child_local, merge.parent_local, merge.value, zero);
        if zero {
            self.contracted_zero_events += 1;
            if self.watch_filter && !had_parent_refs {
                self.zero_fast_drops += 1;
            }
            return;
        }

        let index = self.events.len();
        self.events.push(merge);
        if merge.parent_local != Self::OUTSIDE_LOCAL {
            self.mark_watched(merge.parent_local);
            self.by_parent
                .entry(merge.parent_local)
                .or_default()
                .push(index);
        }
    }

    /// Returns true when retained events actually referenced `child_local`.
    fn redirect_existing(
        &mut self,
        child_local: u32,
        parent_local: u32,
        death: u16,
        zero: bool,
    ) -> bool {
        self.parent_watch_checks += 1;
        if self.watch_filter && !self.is_watched(child_local) {
            self.parent_lookup_skips += 1;
            return false;
        }

        self.parent_hash_lookups += 1;
        let Some(indices) = self.by_parent.remove(&child_local) else {
            debug_assert!(!self.watch_filter || !self.is_watched(child_local));
            self.clear_watched(child_local);
            return false;
        };
        self.clear_watched(child_local);

        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for index in indices {
            let event = &mut self.events[index];
            if zero || event.value < death {
                debug_assert_eq!(event.parent_local, child_local);
                event.parent_local = parent_local;
                self.repaired_parent_refs += 1;
                moved.push(index);
            } else {
                keep.push(index);
            }
        }
        if !keep.is_empty() {
            self.mark_watched(child_local);
            self.by_parent.insert(child_local, keep);
        }
        if parent_local != Self::OUTSIDE_LOCAL && !moved.is_empty() {
            self.mark_watched(parent_local);
            self.by_parent
                .entry(parent_local)
                .or_default()
                .extend(moved);
        }
        true
    }

    fn retained_events(&self) -> u64 {
        self.events.len() as u64
    }

    fn into_events(self, values: &[u16], global_base: u64) -> Vec<H2BranchMerge> {
        self.events
            .into_iter()
            .map(|event| H2BranchMerge {
                value: event.value,
                child_id: global_base + event.child_local as u64,
                child_birth: event.child_birth,
                parent: if event.parent_local == Self::OUTSIDE_LOCAL {
                    H2Branch::Outside
                } else {
                    H2Branch::Finite {
                        id: global_base + event.parent_local as u64,
                        birth: values[event.parent_local as usize],
                    }
                },
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LocalUfAudit {
    enabled: bool,
    union_attempts: u64,
    union_successes: u64,
    union_already_connected: u64,
    union_local_local: u64,
    union_boundary_internal: u64,
    union_interface_interface: u64,
    union_outside_involved: u64,
    current_root_reused: u64,
    current_root_refinds: u64,
    find_calls: u64,
    find_parent_hops: u64,
    find_max_hops: u64,
}

#[derive(Debug)]
struct LocalUnionFind<'a> {
    uf_state: LocalUnionFindState,
    /// Local voxel ID of the elder finite branch, or `u32::MAX` for Outside.
    /// Finite birth/global ID are derived on demand from the slab values.
    elder_local: Vec<u32>,
    interface_rep: Vec<u32>,
    values: &'a [u16],
    global_base: u64,
    audit: LocalUfAudit,
}

impl<'a> LocalUnionFind<'a> {
    const OUTSIDE_LOCAL: u32 = u32::MAX;

    fn new(block: &'a Block) -> Self {
        let voxel_count = block.voxel_count();
        debug_assert_eq!(voxel_count, block.values.len());
        assert!(
            voxel_count < (1usize << 31),
            "compact LocalUnionFind supports fewer than 2^31 voxels per slab"
        );

        let slice_size = block.shape[0] * block.shape[1];
        let global_base = (block.z0 as u64)
            .checked_mul(slice_size as u64)
            .expect("global voxel ID overflow");

        Self {
            uf_state: LocalUnionFindState::new(
                voxel_count,
                ActiveStateStrategy::ParentSentinel,
                UnionFindLayoutStrategy::Packed,
            ),
            elder_local: vec![0u32; voxel_count],
            interface_rep: vec![NO_INTERFACE_REP; voxel_count],
            values: &block.values,
            global_base,
            audit: LocalUfAudit {
                enabled: leaf_kernel_audit_enabled(),
                ..LocalUfAudit::default()
            },
        }
    }

    #[inline]
    fn activate(&mut self, node: u32) {
        self.uf_state.activate(node);
        self.elder_local[node as usize] = node;
    }

    #[inline]
    fn is_active(&self, node: usize) -> bool {
        self.uf_state.is_active(node)
    }

    #[inline]
    fn branch_from_local(&self, local: u32) -> H2Branch {
        if local == Self::OUTSIDE_LOCAL {
            H2Branch::Outside
        } else {
            H2Branch::Finite {
                id: self.global_base + local as u64,
                birth: self.values[local as usize],
            }
        }
    }

    #[inline]
    fn older_local(&self, a: u32, b: u32) -> u32 {
        if a == Self::OUTSIDE_LOCAL || b == Self::OUTSIDE_LOCAL {
            return Self::OUTSIDE_LOCAL;
        }
        let birth_a = self.values[a as usize];
        let birth_b = self.values[b as usize];
        if birth_a > birth_b || (birth_a == birth_b && a <= b) {
            a
        } else {
            b
        }
    }

    fn find(&mut self, node: u32) -> u32 {
        let (root, steps) = self.uf_state.find(node);
        if self.audit.enabled {
            self.audit.find_calls += 1;
            self.audit.find_parent_hops += steps;
            self.audit.find_max_hops = self.audit.find_max_hops.max(steps);
        }
        root
    }

    fn set_interface_rep(&mut self, node: u32, interface_node: u32) {
        let root = self.find(node);
        self.interface_rep[root as usize] = interface_node;
    }

    fn mark_outside(&mut self, node: u32) {
        let root = self.find(node);
        self.elder_local[root as usize] = Self::OUTSIDE_LOCAL;
    }

    #[inline]
    fn local_merge_action(
        &self,
        value: u16,
        child_local: u32,
        parent_local: u32,
        elide_zero_local_merge: bool,
        attach_stats: &mut H2LeafAttachStats,
    ) -> LocalAction {
        debug_assert_ne!(child_local, Self::OUTSIDE_LOCAL);
        if elide_zero_local_merge && self.values[child_local as usize] == value {
            attach_stats.plateau_zero_local_merges_elided += 1;
            LocalAction::None
        } else {
            LocalAction::LocalMerge(LocalH2Merge {
                value,
                child_local,
                child_birth: self.values[child_local as usize],
                parent_local,
            })
        }
    }

    fn union_roots_from_current_root(
        &mut self,
        current_root: u32,
        neighbor_root: u32,
        value: u16,
        finalize_nonpromoting_attach: bool,
        elide_zero_local_merge: bool,
        attach_stats: &mut H2LeafAttachStats,
    ) -> (u32, Option<LocalAction>) {
        if self.audit.enabled {
            self.audit.union_attempts += 1;
        }
        debug_assert!(self.uf_state.is_root(current_root));
        debug_assert!(self.uf_state.is_root(neighbor_root));
        let mut root_a = current_root;
        let mut root_b = neighbor_root;
        if root_a == root_b {
            if self.audit.enabled {
                self.audit.union_already_connected += 1;
            }
            return (root_a, None);
        }

        let elder_a = self.elder_local[root_a as usize];
        let elder_b = self.elder_local[root_b as usize];
        if self.audit.enabled && (elder_a == Self::OUTSIDE_LOCAL || elder_b == Self::OUTSIDE_LOCAL)
        {
            self.audit.union_outside_involved += 1;
        }
        let older_local = self.older_local(elder_a, elder_b);
        let younger_local = if elder_a == elder_b {
            None
        } else if older_local == elder_a {
            Some(elder_b)
        } else {
            Some(elder_a)
        };
        let rep_a = self.interface_rep[root_a as usize];
        let rep_b = self.interface_rep[root_b as usize];
        if self.audit.enabled {
            self.audit.union_successes += 1;
            match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
                (false, false) => self.audit.union_local_local += 1,
                (true, true) => self.audit.union_interface_interface += 1,
                _ => self.audit.union_boundary_internal += 1,
            }
        }

        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => match younger_local {
                Some(child_local) => self.local_merge_action(
                    value,
                    child_local,
                    older_local,
                    elide_zero_local_merge,
                    attach_stats,
                ),
                None => LocalAction::None,
            },
            (true, false) => {
                attach_stats.one_boundary_internal += 1;
                if finalize_nonpromoting_attach && older_local == elder_a {
                    attach_stats.finalized_early += 1;
                    match younger_local {
                        Some(child_local) => self.local_merge_action(
                            value,
                            child_local,
                            elder_a,
                            elide_zero_local_merge,
                            attach_stats,
                        ),
                        None => LocalAction::None,
                    }
                } else {
                    attach_stats.propagated_attach += 1;
                    LocalAction::Attach(AttachEvent {
                        value,
                        interface_node: rep_a,
                        branch: self.branch_from_local(elder_b),
                    })
                }
            }
            (false, true) => {
                attach_stats.one_boundary_internal += 1;
                if finalize_nonpromoting_attach && older_local == elder_b {
                    attach_stats.finalized_early += 1;
                    match younger_local {
                        Some(child_local) => self.local_merge_action(
                            value,
                            child_local,
                            elder_b,
                            elide_zero_local_merge,
                            attach_stats,
                        ),
                        None => LocalAction::None,
                    }
                } else {
                    attach_stats.propagated_attach += 1;
                    LocalAction::Attach(AttachEvent {
                        value,
                        interface_node: rep_b,
                        branch: self.branch_from_local(elder_a),
                    })
                }
            }
            (true, true) => LocalAction::InterfaceMerge(InterfaceMergeEvent {
                value,
                a: rep_a,
                b: rep_b,
            }),
        };

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
        self.elder_local[root_a as usize] = older_local;
        self.interface_rep[root_a as usize] = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };

        (root_a, Some(action))
    }

    fn union_from_current_root(
        &mut self,
        current_root: u32,
        neighbor: u32,
        value: u16,
        finalize_nonpromoting_attach: bool,
        elide_zero_local_merge: bool,
        attach_stats: &mut H2LeafAttachStats,
    ) -> (u32, Option<LocalAction>) {
        let current_is_root = self.uf_state.is_root(current_root);
        if self.audit.enabled {
            if current_is_root {
                self.audit.current_root_reused += 1;
            } else {
                self.audit.current_root_refinds += 1;
            }
        }
        let root_a = if current_is_root {
            current_root
        } else {
            self.find(current_root)
        };
        let root_b = self.find(neighbor);
        self.union_roots_from_current_root(
            root_a,
            root_b,
            value,
            finalize_nonpromoting_attach,
            elide_zero_local_merge,
            attach_stats,
        )
    }
}

struct LocalBuffers<'a> {
    local_merge_history: &'a mut LocalMergeHistory,
    attach_events: &'a mut Vec<AttachEvent>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent>,
    finalize_nonpromoting_attach: bool,
    attach_stats: &'a mut H2LeafAttachStats,
}

fn handle_action(action: LocalAction, buffers: &mut LocalBuffers<'_>) {
    match action {
        LocalAction::None => {}
        LocalAction::LocalMerge(event) => buffers.local_merge_history.push(event),
        LocalAction::Attach(event) => buffers.attach_events.push(event),
        LocalAction::InterfaceMerge(event) => buffers.interface_merge_events.push(event),
    }
}

fn voxel_touches_global_boundary(
    block: &Block,
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

pub(crate) fn process_slab_h2_merge_tree(
    slab_id: usize,
    block: &Block,
    global_shape: [usize; 3],
    background_connectivity: Connectivity,
) -> SlabH2MergeTreeSummary {
    process_slab_h2_merge_tree_impl(
        slab_id,
        block,
        global_shape,
        background_connectivity,
        false,
        false,
    )
    .0
}

pub(crate) fn process_slab_h2_merge_tree_hierarchical(
    slab_id: usize,
    block: &Block,
    global_shape: [usize; 3],
    background_connectivity: Connectivity,
    finalize_nonpromoting_attach: bool,
    inline_contract_local_history: bool,
) -> (SlabH2MergeTreeSummary, H2LeafAttachStats) {
    process_slab_h2_merge_tree_impl(
        slab_id,
        block,
        global_shape,
        background_connectivity,
        finalize_nonpromoting_attach,
        inline_contract_local_history,
    )
}

fn process_slab_h2_merge_tree_impl(
    slab_id: usize,
    block: &Block,
    global_shape: [usize; 3],
    background_connectivity: Connectivity,
    finalize_nonpromoting_attach: bool,
    inline_contract_local_history: bool,
) -> (SlabH2MergeTreeSummary, H2LeafAttachStats) {
    let [global_width, global_height, global_depth] = global_shape;
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalUnionFind::new(block);
    if slab_id == 0 {
        let parent_bytes = uf.uf_state.parent_capacity_bytes();
        let rank_bytes = uf.uf_state.rank_capacity_bytes();
        let elder_bytes =
            (uf.elder_local.capacity() as u64).saturating_mul(core::mem::size_of::<u32>() as u64);
        let interface_rep_bytes =
            (uf.interface_rep.capacity() as u64).saturating_mul(core::mem::size_of::<u32>() as u64);
        let total_bytes = parent_bytes
            .saturating_add(rank_bytes)
            .saturating_add(elder_bytes)
            .saturating_add(interface_rep_bytes);
        println!(
            "PROFILE_BRANCH_H2_LEAF_STATE layout=packed-local-id neighbor_kernel=interior-fast nodes={} parent_bytes={} rank_bytes={} elder_bytes={} interface_rep_bytes={} active_bytes=0 total_bytes={}",
            block.voxel_count(),
            parent_bytes,
            rank_bytes,
            elder_bytes,
            interface_rep_bytes,
            total_bytes,
        );
    }
    let audit_enabled = leaf_kernel_audit_enabled();
    let audit_sample_stride = leaf_kernel_audit_sample_stride();
    let mut attach_stats = H2LeafAttachStats::default();
    let bucket_start = audit_enabled.then(Instant::now);
    let buckets = build_voxel_buckets_u16(&block.values);
    if let Some(start) = bucket_start {
        attach_stats.kernel_audit_bucket_build_ns = start.elapsed().as_nanos() as u64;
    }
    let mut local_pruner =
        NeighborhoodComponentPruner::with_diagnostics(background_connectivity, audit_enabled);
    let linear_offsets = local_pruner.linear_offsets(width, height);
    let shell_pruning =
        matches!(background_connectivity, Connectivity::Six) && h2_shell_pruning_enabled();
    let root_dedup = h2_root_dedup_enabled();
    let mut shell_pruner = H2ShellFacePruner::new(audit_enabled);
    let shell_linear_offsets = shell_pruner.linear_offsets(width, height);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut local_merge_history = LocalMergeHistory::new(
        inline_contract_local_history,
        leaf_history_watch_filter_enabled(),
        block.voxel_count(),
    );
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();
    // Preserve the one-at-a-time activation invariant required by local
    // neighborhood pruning, while suppressing finite [t,t) deaths before they
    // become history records. Outside/interface transitions are still emitted.
    let plateau_native_leaf = inline_contract_local_history && plateau_native_leaf_enabled();

    for value in (0..NUM_U16_VALUES).rev() {
        let start = buckets.offsets[value];
        let end = buckets.offsets[value + 1];
        if start == end {
            continue;
        }
        let value_u16 = value as u16;

        for position in start..end {
            let idx_u32 = buckets.indices[position];
            let idx = idx_u32 as usize;
            if audit_enabled {
                attach_stats.kernel_audit_activations += 1;
            }
            let sample =
                audit_enabled && attach_stats.kernel_audit_activations % audit_sample_stride == 0;
            let bookkeeping_start = sample.then(Instant::now);
            uf.activate(idx_u32);

            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                if audit_enabled {
                    attach_stats.kernel_audit_interface_activations += 1;
                }
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let x = idx % width;
            let y = (idx / width) % height;

            if voxel_touches_global_boundary(
                block,
                x,
                y,
                z,
                global_width,
                global_height,
                global_depth,
            ) {
                if audit_enabled {
                    attach_stats.kernel_audit_global_boundary_activations += 1;
                }
                uf.mark_outside(idx_u32);
                if let Some(interface_node) =
                    local_boundary_node_id(z, depth, face_index, slice_size)
                {
                    attach_events.push(AttachEvent {
                        value: value_u16,
                        interface_node,
                        branch: H2Branch::Outside,
                    });
                }
            }

            let interior =
                x > 0 && x + 1 < width && y > 0 && y + 1 < height && z > 0 && z + 1 < depth;
            if audit_enabled {
                if interior {
                    attach_stats.kernel_audit_interior_voxels += 1;
                } else {
                    attach_stats.kernel_audit_boundary_voxels += 1;
                }
            }
            if let Some(start) = bookkeeping_start {
                attach_stats.kernel_audit_activation_bookkeeping_sample_ns +=
                    start.elapsed().as_nanos() as u64;
            }

            let neighbor_start = sample.then(Instant::now);
            let representatives = if interior && shell_pruning {
                shell_pruner.representative_face_neighbors_interior_by(
                    idx,
                    &shell_linear_offsets,
                    |neighbor| uf.is_active(neighbor),
                )
            } else if interior {
                local_pruner.representative_neighbors_interior_by(
                    idx,
                    &linear_offsets,
                    |neighbor| uf.is_active(neighbor),
                )
            } else {
                local_pruner.representative_neighbors_by(
                    x,
                    y,
                    z,
                    width,
                    height,
                    depth,
                    |neighbor| uf.is_active(neighbor),
                )
            };
            if let Some(start) = neighbor_start {
                attach_stats.kernel_audit_neighbor_pruning_sample_ns +=
                    start.elapsed().as_nanos() as u64;
            }

            let union_start = sample.then(Instant::now);
            {
                let mut buffers = LocalBuffers {
                    local_merge_history: &mut local_merge_history,
                    attach_events: &mut attach_events,
                    interface_merge_events: &mut interface_merge_events,
                    finalize_nonpromoting_attach,
                    attach_stats: &mut attach_stats,
                };
                let mut current_root = idx_u32;
                if root_dedup {
                    let mut distinct_roots = [0u32; 26];
                    let mut distinct_len = 0usize;
                    for neighbor in representatives.iter() {
                        let root = uf.find(neighbor as u32);
                        if audit_enabled {
                            buffers.attach_stats.kernel_audit_root_dedup_inputs += 1;
                        }
                        if distinct_roots[..distinct_len].contains(&root) {
                            if audit_enabled {
                                buffers.attach_stats.kernel_audit_root_dedup_skipped += 1;
                            }
                            continue;
                        }
                        distinct_roots[distinct_len] = root;
                        distinct_len += 1;
                        if audit_enabled {
                            buffers.attach_stats.kernel_audit_root_dedup_unique += 1;
                        }
                    }

                    for neighbor_root in distinct_roots[..distinct_len].iter().copied() {
                        if audit_enabled {
                            uf.audit.current_root_reused += 1;
                        }
                        let (new_root, action) = uf.union_roots_from_current_root(
                            current_root,
                            neighbor_root,
                            value_u16,
                            buffers.finalize_nonpromoting_attach,
                            plateau_native_leaf,
                            &mut *buffers.attach_stats,
                        );
                        current_root = new_root;
                        if let Some(action) = action {
                            handle_action(action, &mut buffers);
                        }
                    }
                } else {
                    for neighbor in representatives.iter() {
                        let (new_root, action) = uf.union_from_current_root(
                            current_root,
                            neighbor as u32,
                            value_u16,
                            buffers.finalize_nonpromoting_attach,
                            plateau_native_leaf,
                            &mut *buffers.attach_stats,
                        );
                        current_root = new_root;
                        if let Some(action) = action {
                            handle_action(action, &mut buffers);
                        }
                    }
                }
            }
            if let Some(start) = union_start {
                attach_stats.kernel_audit_union_action_sample_ns +=
                    start.elapsed().as_nanos() as u64;
                attach_stats.kernel_audit_timing_samples += 1;
            }
        }
    }

    if audit_enabled {
        let pruning = local_pruner.diagnostic_stats();
        let shell = shell_pruner.diagnostic_stats();
        attach_stats.kernel_audit_active_state_checks = pruning
            .active_state_checks
            .saturating_add(shell.face_state_checks)
            .saturating_add(shell.extra_shell_state_checks);
        attach_stats.kernel_audit_active_neighbor_hits = pruning
            .active_neighbor_hits
            .saturating_add(shell.active_face_hits);
        attach_stats.kernel_audit_representative_neighbors = pruning
            .representative_visits
            .saturating_add(shell.representative_visits);
        attach_stats.kernel_audit_shell_face_checks = shell.face_state_checks;
        attach_stats.kernel_audit_shell_active_face_hits = shell.active_face_hits;
        attach_stats.kernel_audit_shell_extra_state_checks = shell.extra_shell_state_checks;
        attach_stats.kernel_audit_shell_extra_active_hits = shell.extra_shell_active_hits;
        attach_stats.kernel_audit_shell_representatives = shell.representative_visits;
        attach_stats.kernel_audit_pruner_cache_lookups = pruning.cache_lookups;
        attach_stats.kernel_audit_pruner_cache_hits = pruning.cache_hits;
        attach_stats.kernel_audit_pruner_cache_misses = pruning.cache_misses;
        attach_stats.kernel_audit_pruner_mask_computations = pruning.component_mask_computations;
        attach_stats.kernel_audit_union_attempts = uf.audit.union_attempts;
        attach_stats.kernel_audit_union_successes = uf.audit.union_successes;
        attach_stats.kernel_audit_union_already_connected = uf.audit.union_already_connected;
        attach_stats.kernel_audit_union_local_local = uf.audit.union_local_local;
        attach_stats.kernel_audit_union_boundary_internal = uf.audit.union_boundary_internal;
        attach_stats.kernel_audit_union_interface_interface = uf.audit.union_interface_interface;
        attach_stats.kernel_audit_union_outside_involved = uf.audit.union_outside_involved;
        attach_stats.kernel_audit_current_root_reused = uf.audit.current_root_reused;
        attach_stats.kernel_audit_current_root_refinds = uf.audit.current_root_refinds;
        attach_stats.kernel_audit_find_calls = uf.audit.find_calls;
        attach_stats.kernel_audit_find_parent_hops = uf.audit.find_parent_hops;
        attach_stats.kernel_audit_find_max_hops = uf.audit.find_max_hops;
    }

    attach_stats.local_history_materialized_events = local_merge_history.input_events;
    attach_stats.local_history_input_events = local_merge_history
        .input_events
        .saturating_add(attach_stats.plateau_zero_local_merges_elided);
    attach_stats.local_history_retained_events = local_merge_history.retained_events();
    attach_stats.local_history_contracted_zero_events = local_merge_history
        .contracted_zero_events
        .saturating_add(attach_stats.plateau_zero_local_merges_elided);
    attach_stats.local_history_repaired_parent_refs = local_merge_history.repaired_parent_refs;
    attach_stats.local_history_parent_watch_checks = local_merge_history.parent_watch_checks;
    attach_stats.local_history_parent_lookup_skips = local_merge_history.parent_lookup_skips;
    attach_stats.local_history_parent_hash_lookups = local_merge_history.parent_hash_lookups;
    attach_stats.local_history_zero_fast_drops = local_merge_history
        .zero_fast_drops
        .saturating_add(attach_stats.plateau_zero_local_merges_elided);
    let local_merge_events = local_merge_history.into_events(&block.values, uf.global_base);

    let z_min_face = extract_boundary_face_nodes(block, 0);
    let z_max_face = extract_boundary_face_nodes(block, depth - 1);

    (
        SlabH2MergeTreeSummary {
            slab_id,
            local_merge_events,
            attach_events,
            interface_merge_events,
            interface_node_count,
            z_min_face,
            z_max_face,
        },
        attach_stats,
    )
}

fn extract_boundary_face_nodes(block: &Block, z_local: usize) -> BoundaryFaceNodes {
    let width = block.shape[0];
    let height = block.shape[1];
    let slice_size = width * height;
    let mut node_ids = vec![0u32; slice_size];
    let mut values = vec![0u16; slice_size];
    let mut branch_ids = vec![0u64; slice_size];
    let global_z = block.z0 + z_local;
    let global_base = (global_z as u64)
        .checked_mul(slice_size as u64)
        .expect("global voxel ID overflow");

    for face_idx in 0..slice_size {
        let idx = z_local * slice_size + face_idx;
        node_ids[face_idx] = face_node_id(z_local, block.shape[2], face_idx, slice_size);
        values[face_idx] = block.values[idx];
        branch_ids[face_idx] = global_base + face_idx as u64;
    }

    BoundaryFaceNodes {
        width,
        height,
        node_ids,
        values,
        branch_ids,
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

fn process_all_slabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<SlabH2MergeTreeSummary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);
    let global_shape = volume.shape();
    println!(
        "Processing {} H2 merge-tree slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabH2MergeTreeSummary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_h2_merge_tree(
                slab_id,
                &block,
                global_shape,
                connectivity,
            ))
        })
        .collect();

    let mut summaries = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;
    summaries.sort_by_key(|summary| summary.slab_id);
    Ok(summaries)
}

fn build_interface_offsets(summaries: &[SlabH2MergeTreeSummary]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(summaries.len() + 1);
    let mut current = 0u32;
    offsets.push(current);
    for summary in summaries {
        current = current
            .checked_add(summary.interface_node_count)
            .expect("too many interface nodes for u32 global IDs");
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
    branch: H2Branch,
}

#[derive(Debug, Clone, Copy)]
struct GlobalPairEvent {
    value: u16,
    a: u32,
    b: u32,
}

#[derive(Debug, Default)]
struct H2DeferredParentResolver {
    watched: HashSet<u64>,
    redirect: HashMap<u64, H2Branch>,
}

impl H2DeferredParentResolver {
    fn from_local_events(local_by_value: &[Vec<H2BranchMerge>]) -> Self {
        let watched = local_by_value
            .iter()
            .flatten()
            .filter_map(|event| event.parent.finite_id())
            .collect();
        Self {
            watched,
            redirect: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn from_compact_local_events(local_by_value: &[Vec<CompactH2BranchMerge>]) -> Self {
        let watched = local_by_value
            .iter()
            .flatten()
            .filter_map(|event| {
                (event.parent_id != crate::merge_tree_common::OUTSIDE_BRANCH_ID)
                    .then_some(event.parent_id)
            })
            .collect();
        Self {
            watched,
            redirect: HashMap::new(),
        }
    }

    fn resolve(&self, parent: H2Branch) -> H2Branch {
        let mut current = parent;
        while let Some(id) = current.finite_id() {
            let Some(&next) = self.redirect.get(&id) else {
                break;
            };
            current = next;
        }
        current
    }

    fn observe_parts(&mut self, child_id: u64, parent: H2Branch) {
        if !self.watched.contains(&child_id) {
            return;
        }
        self.redirect.insert(child_id, parent);
        if let Some(parent_id) = parent.finite_id() {
            self.watched.insert(parent_id);
        }
    }

    fn observe(&mut self, merge: H2BranchMerge) {
        self.observe_parts(merge.child_id, merge.parent);
    }
}

fn repair_h2_local_event(
    event: H2BranchMerge,
    resolver: Option<&H2DeferredParentResolver>,
) -> H2BranchMerge {
    if let Some(resolver) = resolver {
        H2BranchMerge {
            parent: resolver.resolve(event.parent),
            ..event
        }
    } else {
        event
    }
}

#[derive(Debug)]
pub(crate) struct GlobalH2MergeTreeUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    branch: Vec<H2Branch>,
    outside_node: u32,
}

impl GlobalH2MergeTreeUnionFind {
    pub(crate) fn new(mut branch: Vec<H2Branch>) -> Self {
        assert!(
            branch.len() < u32::MAX as usize,
            "too many interface nodes for an outside node"
        );
        let outside_node = branch.len() as u32;
        branch.push(H2Branch::Outside);
        Self {
            parent: (0..branch.len() as u32).collect(),
            rank: vec![0u8; branch.len()],
            branch,
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

    pub(crate) fn union(&mut self, a: u32, b: u32, value: u16) -> Option<H2BranchMerge> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let branch_a = self.branch[root_a as usize];
        let branch_b = self.branch[root_b as usize];
        let same_branch = branch_a == branch_b;
        let older = branch_a.older(branch_b);
        let younger = branch_a.younger(branch_b);

        let rank_a = self.rank[root_a as usize];
        let rank_b = self.rank[root_b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b as usize] = root_a;
        if rank_a == rank_b {
            self.rank[root_a as usize] += 1;
        }
        self.branch[root_a as usize] = older;

        if same_branch {
            None
        } else {
            younger.map(|(child_id, child_birth)| H2BranchMerge {
                value,
                child_id,
                child_birth,
                parent: older,
            })
        }
    }

    pub(crate) fn attach(
        &mut self,
        interface_node: u32,
        branch: H2Branch,
        value: u16,
    ) -> Option<H2BranchMerge> {
        if branch == H2Branch::Outside {
            return self.union(interface_node, self.outside_node, value);
        }

        let root = self.find(interface_node);
        let root_branch = self.branch[root as usize];
        if root_branch == branch {
            return None;
        }
        let older = root_branch.older(branch);
        let younger = root_branch.younger(branch);
        self.branch[root as usize] = older;

        younger.map(|(child_id, child_birth)| H2BranchMerge {
            value,
            child_id,
            child_birth,
            parent: older,
        })
    }

    pub(crate) fn remaining_finite_roots(&mut self) -> Vec<H2Branch> {
        let mut branches = Vec::new();
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32
                && let branch @ H2Branch::Finite { .. } = self.branch[node]
            {
                branches.push(branch);
            }
        }
        branches
    }
}

fn build_global_interface_branches(
    summaries: &[SlabH2MergeTreeSummary],
    offsets: &[u32],
) -> Vec<H2Branch> {
    let total = *offsets.last().unwrap_or(&0) as usize;
    let mut branches = vec![H2Branch::Finite { id: 0, birth: 0 }; total];

    for summary in summaries {
        for face in [&summary.z_min_face, &summary.z_max_face] {
            for ((&local_node, &birth), &id) in face
                .node_ids
                .iter()
                .zip(face.values.iter())
                .zip(face.branch_ids.iter())
            {
                let global = global_node_id(offsets, summary.slab_id, local_node) as usize;
                branches[global] = H2Branch::Finite { id, birth };
            }
        }
    }
    branches
}

fn generate_cross_slab_events(
    summaries: &[SlabH2MergeTreeSummary],
    offsets: &[u32],
    connectivity: Connectivity,
) -> Vec<GlobalPairEvent> {
    let mut events = Vec::new();
    for slab_pair in summaries.windows(2) {
        let left = &slab_pair[0];
        let right = &slab_pair[1];
        let left_face = &left.z_max_face;
        let right_face = &right.z_min_face;
        assert_eq!(left_face.width, right_face.width);
        assert_eq!(left_face.height, right_face.height);

        let width = left_face.width;
        let height = left_face.height;
        for y in 0..height {
            for x in 0..width {
                let left_idx = y * width + x;
                let a = global_node_id(offsets, left.slab_id, left_face.node_ids[left_idx]);
                let left_value = left_face.values[left_idx];

                match connectivity {
                    Connectivity::Six => {
                        let right_idx = left_idx;
                        let b =
                            global_node_id(offsets, right.slab_id, right_face.node_ids[right_idx]);
                        events.push(GlobalPairEvent {
                            value: left_value.min(right_face.values[right_idx]),
                            a,
                            b,
                        });
                    }
                    Connectivity::TwentySix => {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize
                                {
                                    continue;
                                }
                                let right_idx = ny as usize * width + nx as usize;
                                let b = global_node_id(
                                    offsets,
                                    right.slab_id,
                                    right_face.node_ids[right_idx],
                                );
                                events.push(GlobalPairEvent {
                                    value: left_value.min(right_face.values[right_idx]),
                                    a,
                                    b,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    events
}

pub(crate) fn reduce_merge_tree(
    summaries: &[SlabH2MergeTreeSummary],
    connectivity: Connectivity,
) -> Result<MergeTree> {
    reduce_merge_tree_impl(summaries, connectivity, false, None)
}

pub(crate) fn reduce_merge_tree_with_deferred_parent_repair_prebucketed(
    summary: &SlabH2MergeTreeSummary,
    compact_local_by_value: Vec<Vec<CompactH2BranchMerge>>,
    connectivity: Connectivity,
) -> Result<MergeTree> {
    reduce_merge_tree_impl(
        std::slice::from_ref(summary),
        connectivity,
        true,
        Some(compact_local_by_value),
    )
}

fn reduce_merge_tree_impl(
    summaries: &[SlabH2MergeTreeSummary],
    connectivity: Connectivity,
    repair_deferred_parents: bool,
    prebucketed_local: Option<Vec<Vec<CompactH2BranchMerge>>>,
) -> Result<MergeTree> {
    let offsets = build_interface_offsets(summaries);
    let branches = build_global_interface_branches(summaries, &offsets);
    let mut global_uf = GlobalH2MergeTreeUnionFind::new(branches);

    let using_prebucketed_local = prebucketed_local.is_some();
    if let Some(local_by_value) = prebucketed_local.as_ref()
        && local_by_value.len() != NUM_U16_VALUES
    {
        anyhow::bail!(
            "H2 compact prebucketed local event table has {} buckets; expected {}",
            local_by_value.len(),
            NUM_U16_VALUES
        );
    }
    let mut local_by_value: Option<Vec<Vec<H2BranchMerge>>> =
        (!using_prebucketed_local).then(|| (0..NUM_U16_VALUES).map(|_| Vec::new()).collect());
    let mut attach_by_value: Vec<Vec<GlobalAttachEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut interface_by_value: Vec<Vec<GlobalPairEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut cross_by_value: Vec<Vec<GlobalPairEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();

    for summary in summaries {
        if let Some(local_by_value) = local_by_value.as_mut() {
            for &event in &summary.local_merge_events {
                local_by_value[event.value as usize].push(event);
            }
        } else if !summary.local_merge_events.is_empty() {
            anyhow::bail!(
                "H2 compact prebucketed reduction requires summary.local_merge_events to be empty"
            );
        }
        for event in &summary.attach_events {
            let node = global_node_id(&offsets, summary.slab_id, event.interface_node);
            attach_by_value[event.value as usize].push(GlobalAttachEvent {
                node,
                branch: event.branch,
            });
        }
        for event in &summary.interface_merge_events {
            let a = global_node_id(&offsets, summary.slab_id, event.a);
            let b = global_node_id(&offsets, summary.slab_id, event.b);
            interface_by_value[event.value as usize].push(GlobalPairEvent {
                value: event.value,
                a,
                b,
            });
        }
    }

    for event in generate_cross_slab_events(summaries, &offsets, connectivity) {
        cross_by_value[event.value as usize].push(event);
    }

    if let Some(compact) = prebucketed_local.as_ref() {
        // Hierarchical packed fast path. Deferred-parent resolution and
        // same-threshold plateau contraction use integer-only open-addressed
        // state; the flat reducer below remains the independent reference.
        let mut resolver = if repair_deferred_parents {
            let mut resolver = PackedDeferredIdResolver::default();
            for event in compact.iter().flatten() {
                resolver.watch(event.parent_id);
            }
            Some(resolver)
        } else {
            None
        };
        let mut recorder = H2PackedTreeRecorder::default();
        let mut outside_attach_merges: Vec<H2BranchMerge> = Vec::new();
        let mut finite_attach_merges: Vec<H2BranchMerge> = Vec::new();
        let mut duplicate_targets: HashSet<(u64, u64)> = HashSet::new();
        let mut matched_duplicates: HashSet<(u64, u64)> = HashSet::new();
        let mut resolved_local_parent_ids: Vec<u64> = Vec::new();
        let mut observed_prefix: Vec<(u64, u64)> = Vec::new();
        let mut observed_suffix: Vec<(u64, u64)> = Vec::new();

        for value in (0..NUM_U16_VALUES).rev() {
            let value_u16 = value as u16;
            outside_attach_merges.clear();
            finite_attach_merges.clear();
            duplicate_targets.clear();
            matched_duplicates.clear();
            resolved_local_parent_ids.clear();
            observed_prefix.clear();
            observed_suffix.clear();

            // Preserve the exact global-UF mutation order from v5.
            outside_attach_merges.reserve(attach_by_value[value].len());
            for event in &attach_by_value[value] {
                if event.branch == H2Branch::Outside
                    && let Some(merge) = global_uf.attach(event.node, event.branch, value_u16)
                {
                    outside_attach_merges.push(merge);
                }
            }
            finite_attach_merges.reserve(attach_by_value[value].len());
            for event in &attach_by_value[value] {
                if event.branch != H2Branch::Outside
                    && let Some(merge) = global_uf.attach(event.node, event.branch, value_u16)
                {
                    finite_attach_merges.push(merge);
                }
            }

            if repair_deferred_parents {
                duplicate_targets.extend(
                    outside_attach_merges
                        .iter()
                        .chain(finite_attach_merges.iter())
                        .map(|merge| {
                            let parent_id = match merge.parent {
                                H2Branch::Outside => OUTSIDE_BRANCH_ID,
                                H2Branch::Finite { id, .. } => id,
                            };
                            (merge.child_id, parent_id)
                        }),
                );
                matched_duplicates.reserve(duplicate_targets.len());
            }

            resolved_local_parent_ids.reserve(compact[value].len());
            for &event in &compact[value] {
                let repaired_parent_id = if let Some(resolver) = resolver.as_ref() {
                    resolver.resolve_id(event.parent_id)?
                } else {
                    event.parent_id
                };
                if !duplicate_targets.is_empty()
                    && duplicate_targets.contains(&(event.child_id, repaired_parent_id))
                {
                    matched_duplicates.insert((event.child_id, repaired_parent_id));
                }
                resolved_local_parent_ids.push(repaired_parent_id);
            }

            for &merge in &outside_attach_merges {
                let parent_id = match merge.parent {
                    H2Branch::Outside => OUTSIDE_BRANCH_ID,
                    H2Branch::Finite { id, .. } => id,
                };
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && matched_duplicates.contains(&(merge.child_id, parent_id));
                if !duplicate_hierarchical_transition {
                    recorder.record_merge(merge)?;
                    observed_prefix.push((merge.child_id, parent_id));
                }
            }

            for (&event, &repaired_parent_id) in
                compact[value].iter().zip(resolved_local_parent_ids.iter())
            {
                recorder.record_compact_merge_parent_id(value_u16, event, repaired_parent_id)?;
            }

            for &merge in &finite_attach_merges {
                let parent_id = match merge.parent {
                    H2Branch::Outside => OUTSIDE_BRANCH_ID,
                    H2Branch::Finite { id, .. } => id,
                };
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && matched_duplicates.contains(&(merge.child_id, parent_id));
                if !duplicate_hierarchical_transition {
                    recorder.record_merge(merge)?;
                    observed_suffix.push((merge.child_id, parent_id));
                }
            }
            for event in &interface_by_value[value] {
                if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                    let parent_id = match merge.parent {
                        H2Branch::Outside => OUTSIDE_BRANCH_ID,
                        H2Branch::Finite { id, .. } => id,
                    };
                    recorder.record_merge(merge)?;
                    observed_suffix.push((merge.child_id, parent_id));
                }
            }
            for event in &cross_by_value[value] {
                if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                    let parent_id = match merge.parent {
                        H2Branch::Outside => OUTSIDE_BRANCH_ID,
                        H2Branch::Finite { id, .. } => id,
                    };
                    recorder.record_merge(merge)?;
                    observed_suffix.push((merge.child_id, parent_id));
                }
            }

            recorder.finish_threshold()?;
            if let Some(resolver) = resolver.as_mut() {
                // Preserve exact observation order: outside attaches, local
                // events, finite attaches, interface events, cross events.
                for &(child_id, parent_id) in &observed_prefix {
                    resolver.observe(child_id, parent_id);
                }
                for (&event, &repaired_parent_id) in
                    compact[value].iter().zip(resolved_local_parent_ids.iter())
                {
                    resolver.observe(event.child_id, repaired_parent_id);
                }
                for &(child_id, parent_id) in &observed_suffix {
                    resolver.observe(child_id, parent_id);
                }
            }
        }

        let remaining = global_uf.remaining_finite_roots();
        if !remaining.is_empty() {
            bail!(
                "H2 merge-tree reduction ended with {} background components not connected to outside",
                remaining.len()
            );
        }
        recorder.add_outside_root();
        return recorder.into_tree();
    }

    // General flat/in-memory reference reducer retained unchanged.
    let local = local_by_value
        .as_ref()
        .expect("non-prebucketed H2 reducer must own local event buckets");
    let mut resolver = if repair_deferred_parents {
        Some(H2DeferredParentResolver::from_local_events(local))
    } else {
        None
    };
    let mut recorder = H2TreeRecorder::default();
    for value in (0..NUM_U16_VALUES).rev() {
        let value_u16 = value as u16;
        let mut observed_merges = Vec::new();

        let mut recorded_local_merges: HashSet<(u64, Option<u64>)> = HashSet::new();
        for &event in &local[value] {
            let repaired = repair_h2_local_event(event, resolver.as_ref());
            recorded_local_merges.insert((repaired.child_id, repaired.parent.finite_id()));
        }

        for event in &attach_by_value[value] {
            if event.branch == H2Branch::Outside
                && let Some(merge) = global_uf.attach(event.node, event.branch, value_u16)
            {
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && recorded_local_merges.contains(&(merge.child_id, merge.parent.finite_id()));
                if !duplicate_hierarchical_transition {
                    recorder.record_merge(merge)?;
                    observed_merges.push(merge);
                }
            }
        }

        for &event in &local[value] {
            let repaired = repair_h2_local_event(event, resolver.as_ref());
            recorder.record_merge(repaired)?;
            observed_merges.push(repaired);
        }
        for event in &attach_by_value[value] {
            if event.branch != H2Branch::Outside
                && let Some(merge) = global_uf.attach(event.node, event.branch, value_u16)
            {
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && recorded_local_merges.contains(&(merge.child_id, merge.parent.finite_id()));
                if !duplicate_hierarchical_transition {
                    recorder.record_merge(merge)?;
                    observed_merges.push(merge);
                }
            }
        }
        for event in &interface_by_value[value] {
            if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                recorder.record_merge(merge)?;
                observed_merges.push(merge);
            }
        }
        for event in &cross_by_value[value] {
            if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                recorder.record_merge(merge)?;
                observed_merges.push(merge);
            }
        }

        recorder.finish_threshold()?;
        if let Some(resolver) = resolver.as_mut() {
            for merge in observed_merges {
                resolver.observe(merge);
            }
        }
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

pub fn compute_h2_merge_tree_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<MergeTree> {
    let start = Instant::now();
    println!("Computing in-memory slabwise H2 merge tree...");

    let summaries = process_all_slabs(volume, slab_depth, background_connectivity)?;
    println!("Processed {} H2 merge-tree slab summaries", summaries.len());
    let tree = reduce_merge_tree(&summaries, background_connectivity)?;

    println!(
        "In-memory H2 merge-tree computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(tree)
}

#[cfg(test)]
mod deferred_parent_tests {
    use super::*;

    #[test]
    fn global_h2_same_branch_replay_does_not_emit_a_second_death() {
        let branch = H2Branch::Finite { id: 10, birth: 12 };
        let mut uf = GlobalH2MergeTreeUnionFind::new(vec![branch, branch]);

        assert!(uf.attach(0, branch, 8).is_none());
        assert!(uf.union(0, 1, 8).is_none());
        assert_eq!(uf.remaining_finite_roots(), vec![branch]);
    }

    #[test]
    fn deferred_h2_parent_follows_an_earlier_superlevel_death() {
        let provisional_parent = H2Branch::Finite { id: 20, birth: 12 };
        let survivor = H2Branch::Finite { id: 30, birth: 14 };
        let local = [vec![H2BranchMerge {
            value: 4,
            child_id: 10,
            child_birth: 9,
            parent: provisional_parent,
        }]];
        let mut resolver = H2DeferredParentResolver::from_local_events(&local);

        resolver.observe(H2BranchMerge {
            value: 8,
            child_id: 20,
            child_birth: 12,
            parent: survivor,
        });

        assert_eq!(resolver.resolve(provisional_parent), survivor);
    }

    #[test]
    fn deferred_h2_parent_compact_bucket_matches_full_bucket() {
        let provisional_parent = H2Branch::Finite { id: 20, birth: 12 };
        let survivor = H2Branch::Finite { id: 30, birth: 14 };
        let compact = [vec![CompactH2BranchMerge::from_merge(H2BranchMerge {
            value: 4,
            child_id: 10,
            child_birth: 9,
            parent: provisional_parent,
        })]];
        let mut resolver = H2DeferredParentResolver::from_compact_local_events(&compact);

        resolver.observe(H2BranchMerge {
            value: 8,
            child_id: 20,
            child_birth: 12,
            parent: survivor,
        });

        assert_eq!(resolver.resolve(provisional_parent), survivor);
    }

    #[test]
    fn deferred_h2_parent_can_resolve_to_outside() {
        let provisional_parent = H2Branch::Finite { id: 20, birth: 12 };
        let local = [vec![H2BranchMerge {
            value: 3,
            child_id: 10,
            child_birth: 9,
            parent: provisional_parent,
        }]];
        let mut resolver = H2DeferredParentResolver::from_local_events(&local);

        resolver.observe(H2BranchMerge {
            value: 7,
            child_id: 20,
            child_birth: 12,
            parent: H2Branch::Outside,
        });

        assert_eq!(resolver.resolve(provisional_parent), H2Branch::Outside);
    }
}

#[cfg(test)]
mod hierarchical_leaf_attach_pruning_tests {
    use super::*;

    fn two_node_uf(values: &[u16], boundary: H2Branch, internal: H2Branch) -> LocalUnionFind<'_> {
        let mut uf = LocalUnionFind {
            uf_state: LocalUnionFindState::new(
                2,
                ActiveStateStrategy::ParentSentinel,
                UnionFindLayoutStrategy::Packed,
            ),
            elder_local: vec![
                if boundary == H2Branch::Outside {
                    LocalUnionFind::OUTSIDE_LOCAL
                } else {
                    0
                },
                if internal == H2Branch::Outside {
                    LocalUnionFind::OUTSIDE_LOCAL
                } else {
                    1
                },
            ],
            interface_rep: vec![0, NO_INTERFACE_REP],
            values,
            global_base: 1,
            audit: LocalUfAudit::default(),
        };
        uf.uf_state.activate(0);
        uf.uf_state.activate(1);
        uf
    }

    #[test]
    fn hierarchical_h2_finalizes_nonpromoting_internal_branch_early() {
        let boundary = H2Branch::Finite { id: 1, birth: 12 };
        let internal = H2Branch::Finite { id: 2, birth: 9 };
        let values = [
            match boundary {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
            match internal {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
        ];
        let mut uf = two_node_uf(&values, boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, false, &mut stats);
        let action = action.unwrap();
        match action {
            LocalAction::LocalMerge(merge) => {
                assert_eq!(merge.value, 5);
                assert_eq!(merge.child_local, 1);
                assert_eq!(merge.child_birth, 9);
                assert_eq!(merge.parent_local, 0);
            }
            other => panic!("expected early local finalization, got {other:?}"),
        }
        assert_eq!(stats.one_boundary_internal, 1);
        assert_eq!(stats.finalized_early, 1);
        assert_eq!(stats.propagated_attach, 0);
    }

    #[test]
    fn hierarchical_h2_keeps_promoting_internal_branch_as_attach() {
        let boundary = H2Branch::Finite { id: 1, birth: 9 };
        let internal = H2Branch::Finite { id: 2, birth: 12 };
        let values = [
            match boundary {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
            match internal {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
        ];
        let mut uf = two_node_uf(&values, boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, false, &mut stats);
        let action = action.unwrap();
        match action {
            LocalAction::Attach(event) => {
                assert_eq!(event.value, 5);
                assert_eq!(event.interface_node, 0);
                assert_eq!(event.branch, internal);
            }
            other => panic!("expected propagated attach, got {other:?}"),
        }
        assert_eq!(stats.finalized_early, 0);
        assert_eq!(stats.propagated_attach, 1);
    }

    #[test]
    fn hierarchical_h2_finalizes_finite_branch_against_outside() {
        let boundary = H2Branch::Outside;
        let internal = H2Branch::Finite { id: 2, birth: 12 };
        let values = [
            match boundary {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
            match internal {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
        ];
        let mut uf = two_node_uf(&values, boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, false, &mut stats);
        let action = action.unwrap();
        match action {
            LocalAction::LocalMerge(merge) => {
                assert_eq!(merge.child_local, 1);
                assert_eq!(merge.parent_local, LocalUnionFind::OUTSIDE_LOCAL);
            }
            other => panic!("expected finite-to-outside finalization, got {other:?}"),
        }
    }

    #[test]
    fn h2_plateau_native_elides_zero_local_merge_before_history() {
        let boundary = H2Branch::Finite { id: 1, birth: 5 };
        let internal = H2Branch::Finite { id: 2, birth: 5 };
        let values = [5u16, 5u16];
        let mut uf = two_node_uf(&values, boundary, internal);
        uf.interface_rep[0] = NO_INTERFACE_REP;
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, true, &mut stats);
        assert!(matches!(action, Some(LocalAction::None)));
        assert_eq!(stats.plateau_zero_local_merges_elided, 1);
    }

    #[test]
    fn h2_plateau_native_keeps_positive_local_merge() {
        let boundary = H2Branch::Finite { id: 1, birth: 7 };
        let internal = H2Branch::Finite { id: 2, birth: 9 };
        let values = [7u16, 9u16];
        let mut uf = two_node_uf(&values, boundary, internal);
        uf.interface_rep[0] = NO_INTERFACE_REP;
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, true, &mut stats);
        assert!(matches!(action, Some(LocalAction::LocalMerge(_))));
        assert_eq!(stats.plateau_zero_local_merges_elided, 0);
    }

    #[test]
    fn h2_reference_leaf_policy_preserves_original_attach() {
        let boundary = H2Branch::Finite { id: 1, birth: 12 };
        let internal = H2Branch::Finite { id: 2, birth: 9 };
        let values = [
            match boundary {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
            match internal {
                H2Branch::Finite { birth, .. } => birth,
                H2Branch::Outside => 0,
            },
        ];
        let mut uf = two_node_uf(&values, boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, false, false, &mut stats);
        assert!(matches!(action, Some(LocalAction::Attach(_))));
        assert_eq!(stats.finalized_early, 0);
        assert_eq!(stats.propagated_attach, 1);
    }

    #[test]
    fn h2_inline_leaf_history_contracts_diagonal_parent() {
        let mut history = LocalMergeHistory::new(true, true, 64);
        history.push(LocalH2Merge {
            value: 10,
            child_local: 10,
            child_birth: 12,
            parent_local: 50,
        });
        history.push(LocalH2Merge {
            value: 15,
            child_local: 50,
            child_birth: 15,
            parent_local: 60,
        });

        assert_eq!(history.input_events, 2);
        assert_eq!(history.contracted_zero_events, 1);
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(history.retained_events(), 1);
        assert_eq!(history.events[0].parent_local, 60);
        assert_eq!(history.parent_hash_lookups, 1);
        assert_eq!(history.parent_lookup_skips, 1);
    }

    #[test]
    fn h2_inline_leaf_history_keeps_equal_threshold_positive_parent() {
        let mut history = LocalMergeHistory::new(true, true, 64);
        history.push(LocalH2Merge {
            value: 15,
            child_local: 10,
            child_birth: 16,
            parent_local: 50,
        });
        history.push(LocalH2Merge {
            value: 15,
            child_local: 50,
            child_birth: 20,
            parent_local: 60,
        });
        assert_eq!(history.repaired_parent_refs, 0);
        assert_eq!(history.events[0].parent_local, 50);
        assert_eq!(history.retained_events(), 2);
    }

    #[test]
    fn h2_inline_leaf_history_repairs_to_outside() {
        let mut history = LocalMergeHistory::new(true, true, 64);
        history.push(LocalH2Merge {
            value: 10,
            child_local: 10,
            child_birth: 12,
            parent_local: 50,
        });
        history.push(LocalH2Merge {
            value: 15,
            child_local: 50,
            child_birth: 15,
            parent_local: LocalMergeHistory::OUTSIDE_LOCAL,
        });
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(
            history.events[0].parent_local,
            LocalMergeHistory::OUTSIDE_LOCAL
        );
    }
}
