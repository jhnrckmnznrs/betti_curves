use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::local_uf_state::LocalUnionFindState;
use crate::merge_tree_common::{
    CompactH0BranchMerge, H0Branch, H0BranchMerge, H0PackedTreeRecorder, H0TreeRecorder, MergeTree,
    PackedDeferredIdResolver,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttachEvent {
    pub(crate) value: u16,
    pub(crate) interface_node: u32,
    pub(crate) branch: H0Branch,
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
pub(crate) struct SlabH0MergeTreeSummary {
    pub(crate) slab_id: usize,
    pub(crate) local_merge_events: Vec<H0BranchMerge>,
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
struct LocalH0Merge {
    value: u16,
    child_local: u32,
    child_birth: u16,
    parent_local: u32,
}

#[derive(Debug, Clone, Copy)]
enum LocalAction {
    None,
    LocalMerge(LocalH0Merge),
    Attach(AttachEvent),
    InterfaceMerge(InterfaceMergeEvent),
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct H0LeafAttachStats {
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
    pub(crate) kernel_audit_timing_samples: u64,
    pub(crate) kernel_audit_activation_bookkeeping_sample_ns: u64,
    pub(crate) kernel_audit_neighbor_pruning_sample_ns: u64,
    pub(crate) kernel_audit_union_action_sample_ns: u64,
    pub(crate) kernel_audit_bucket_build_ns: u64,
}

/// Leaf-local finalized-history contractor using slab-local branch IDs.
///
/// Positive events are rare, so only their provisional parents are tracked in
/// `by_parent`. A dense one-bit watch filter lets the overwhelmingly common
/// zero-persistence death prove in O(1) that no retained event can reference
/// its child before touching the hash table. Full global branch objects are
/// reconstructed only for retained events at the end of the slab.
#[derive(Debug)]
struct LocalMergeHistory {
    events: Vec<LocalH0Merge>,
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
        let index = local as usize;
        let word = index >> 6;
        let mask = 1u64 << (index & 63);
        self.watched_parent_bits
            .get(word)
            .is_some_and(|bits| bits & mask != 0)
    }

    #[inline]
    fn mark_watched(&mut self, local: u32) {
        let index = local as usize;
        self.watched_parent_bits[index >> 6] |= 1u64 << (index & 63);
    }

    #[inline]
    fn clear_watched(&mut self, local: u32) {
        let index = local as usize;
        self.watched_parent_bits[index >> 6] &= !(1u64 << (index & 63));
    }

    fn push(&mut self, merge: LocalH0Merge) {
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
        self.mark_watched(merge.parent_local);
        self.by_parent
            .entry(merge.parent_local)
            .or_default()
            .push(index);
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
            // In unfiltered reference mode this is the normal no-reference
            // path. With filtering enabled it can only occur if a stale bit
            // escaped a bookkeeping bug.
            debug_assert!(!self.watch_filter || !self.is_watched(child_local));
            self.clear_watched(child_local);
            return false;
        };
        self.clear_watched(child_local);

        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for index in indices {
            let event = &mut self.events[index];
            if zero || event.value > death {
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
        if !moved.is_empty() {
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

    fn into_events(self, values: &[u16], global_base: u64) -> Vec<H0BranchMerge> {
        self.events
            .into_iter()
            .map(|event| H0BranchMerge {
                value: event.value,
                child: H0Branch {
                    id: global_base + event.child_local as u64,
                    birth: event.child_birth,
                },
                parent: H0Branch {
                    id: global_base + event.parent_local as u64,
                    birth: values[event.parent_local as usize],
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
    /// Local voxel ID of the elder branch carried by each current root.
    /// Birth and global branch ID are derived from `values` and `global_base`,
    /// avoiding a 16-byte `H0Branch` entry for every voxel.
    elder_local: Vec<u32>,
    interface_rep: Vec<u32>,
    values: &'a [u16],
    global_base: u64,
    audit: LocalUfAudit,
}

impl<'a> LocalUnionFind<'a> {
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
    fn branch_from_local(&self, local: u32) -> H0Branch {
        H0Branch {
            id: self.global_base + local as u64,
            birth: self.values[local as usize],
        }
    }

    #[inline]
    fn older_local(&self, a: u32, b: u32) -> u32 {
        let key_a = (self.values[a as usize], a);
        let key_b = (self.values[b as usize], b);
        if key_a <= key_b { a } else { b }
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

    #[inline]
    fn local_merge_action(
        &self,
        value: u16,
        child_local: u32,
        parent_local: u32,
        elide_zero_local_merge: bool,
        attach_stats: &mut H0LeafAttachStats,
    ) -> LocalAction {
        if elide_zero_local_merge && self.values[child_local as usize] == value {
            attach_stats.plateau_zero_local_merges_elided += 1;
            LocalAction::None
        } else {
            LocalAction::LocalMerge(LocalH0Merge {
                value,
                child_local,
                child_birth: self.values[child_local as usize],
                parent_local,
            })
        }
    }

    fn union_from_current_root(
        &mut self,
        current_root: u32,
        neighbor: u32,
        value: u16,
        finalize_nonpromoting_attach: bool,
        elide_zero_local_merge: bool,
        attach_stats: &mut H0LeafAttachStats,
    ) -> (u32, Option<LocalAction>) {
        if self.audit.enabled {
            self.audit.union_attempts += 1;
        }
        let current_is_root = self.uf_state.is_root(current_root);
        if self.audit.enabled {
            if current_is_root {
                self.audit.current_root_reused += 1;
            } else {
                self.audit.current_root_refinds += 1;
            }
        }
        let mut root_a = if current_is_root {
            current_root
        } else {
            self.find(current_root)
        };
        let mut root_b = self.find(neighbor);
        if root_a == root_b {
            if self.audit.enabled {
                self.audit.union_already_connected += 1;
            }
            return (root_a, None);
        }

        let elder_a = self.elder_local[root_a as usize];
        let elder_b = self.elder_local[root_b as usize];
        let older_local = self.older_local(elder_a, elder_b);
        let younger_local = if older_local == elder_a {
            elder_b
        } else {
            elder_a
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
            (false, false) => {
                if elder_a == elder_b {
                    LocalAction::None
                } else {
                    self.local_merge_action(
                        value,
                        younger_local,
                        older_local,
                        elide_zero_local_merge,
                        attach_stats,
                    )
                }
            }
            (true, false) => {
                attach_stats.one_boundary_internal += 1;
                if finalize_nonpromoting_attach && older_local == elder_a {
                    attach_stats.finalized_early += 1;
                    if elder_a == elder_b {
                        LocalAction::None
                    } else {
                        self.local_merge_action(
                            value,
                            elder_b,
                            elder_a,
                            elide_zero_local_merge,
                            attach_stats,
                        )
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
                    if elder_a == elder_b {
                        LocalAction::None
                    } else {
                        self.local_merge_action(
                            value,
                            elder_a,
                            elder_b,
                            elide_zero_local_merge,
                            attach_stats,
                        )
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
}

struct LocalBuffers<'a> {
    local_merge_history: &'a mut LocalMergeHistory,
    attach_events: &'a mut Vec<AttachEvent>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent>,
    finalize_nonpromoting_attach: bool,
    attach_stats: &'a mut H0LeafAttachStats,
}

fn handle_action(action: LocalAction, buffers: &mut LocalBuffers<'_>) {
    match action {
        LocalAction::None => {}
        LocalAction::LocalMerge(event) => buffers.local_merge_history.push(event),
        LocalAction::Attach(event) => buffers.attach_events.push(event),
        LocalAction::InterfaceMerge(event) => buffers.interface_merge_events.push(event),
    }
}

pub(crate) fn process_slab_h0_merge_tree(
    slab_id: usize,
    block: &Block,
    connectivity: Connectivity,
) -> SlabH0MergeTreeSummary {
    process_slab_h0_merge_tree_impl(slab_id, block, connectivity, false, false).0
}

pub(crate) fn process_slab_h0_merge_tree_hierarchical(
    slab_id: usize,
    block: &Block,
    connectivity: Connectivity,
    finalize_nonpromoting_attach: bool,
    inline_contract_local_history: bool,
) -> (SlabH0MergeTreeSummary, H0LeafAttachStats) {
    process_slab_h0_merge_tree_impl(
        slab_id,
        block,
        connectivity,
        finalize_nonpromoting_attach,
        inline_contract_local_history,
    )
}

fn process_slab_h0_merge_tree_impl(
    slab_id: usize,
    block: &Block,
    connectivity: Connectivity,
    finalize_nonpromoting_attach: bool,
    inline_contract_local_history: bool,
) -> (SlabH0MergeTreeSummary, H0LeafAttachStats) {
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
            "PROFILE_BRANCH_H0_LEAF_STATE layout=packed-local-id neighbor_kernel=interior-fast nodes={} parent_bytes={} rank_bytes={} elder_bytes={} interface_rep_bytes={} active_bytes=0 total_bytes={}",
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
    let mut attach_stats = H0LeafAttachStats::default();
    let bucket_start = audit_enabled.then(Instant::now);
    let buckets = build_voxel_buckets_u16(&block.values);
    if let Some(start) = bucket_start {
        attach_stats.kernel_audit_bucket_build_ns = start.elapsed().as_nanos() as u64;
    }
    let mut local_pruner =
        NeighborhoodComponentPruner::with_diagnostics(connectivity, audit_enabled);
    let linear_offsets = local_pruner.linear_offsets(width, height);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut local_merge_history = LocalMergeHistory::new(
        inline_contract_local_history,
        leaf_history_watch_filter_enabled(),
        block.voxel_count(),
    );
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();
    // The exact neighborhood pruner requires one-at-a-time activation.  We keep
    // that invariant, but contract same-threshold local deaths at the union
    // decision itself.  This is plateau-native after zero-length contraction:
    // no [t,t) local branch ever enters the leaf history contractor.
    let plateau_native_leaf = inline_contract_local_history && plateau_native_leaf_enabled();

    for value in 0..NUM_U16_VALUES {
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
            let representatives = if interior {
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
            if let Some(start) = union_start {
                attach_stats.kernel_audit_union_action_sample_ns +=
                    start.elapsed().as_nanos() as u64;
                attach_stats.kernel_audit_timing_samples += 1;
            }
        }
    }

    if audit_enabled {
        let pruning = local_pruner.diagnostic_stats();
        attach_stats.kernel_audit_active_state_checks = pruning.active_state_checks;
        attach_stats.kernel_audit_active_neighbor_hits = pruning.active_neighbor_hits;
        attach_stats.kernel_audit_representative_neighbors = pruning.representative_visits;
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
        SlabH0MergeTreeSummary {
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
) -> Result<Vec<SlabH0MergeTreeSummary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);
    println!(
        "Processing {} H0 merge-tree slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabH0MergeTreeSummary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_h0_merge_tree(slab_id, &block, connectivity))
        })
        .collect();

    let mut summaries = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;
    summaries.sort_by_key(|summary| summary.slab_id);
    Ok(summaries)
}

fn build_interface_offsets(summaries: &[SlabH0MergeTreeSummary]) -> Vec<u32> {
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
    branch: H0Branch,
}

#[derive(Debug, Clone, Copy)]
struct GlobalPairEvent {
    value: u16,
    a: u32,
    b: u32,
}

#[derive(Debug, Default)]
struct H0DeferredParentResolver {
    watched: HashSet<u64>,
    redirect: HashMap<u64, H0Branch>,
}

impl H0DeferredParentResolver {
    fn from_local_events(local_by_value: &[Vec<H0BranchMerge>]) -> Self {
        let watched = local_by_value
            .iter()
            .flatten()
            .map(|event| event.parent.id)
            .collect();
        Self {
            watched,
            redirect: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn from_compact_local_events(local_by_value: &[Vec<CompactH0BranchMerge>]) -> Self {
        let watched = local_by_value
            .iter()
            .flatten()
            .map(|event| event.parent_id)
            .collect();
        Self {
            watched,
            redirect: HashMap::new(),
        }
    }

    fn resolve(&self, parent: H0Branch) -> H0Branch {
        let mut current = parent;
        while let Some(&next) = self.redirect.get(&current.id) {
            current = next;
        }
        current
    }

    fn observe_parts(&mut self, child_id: u64, parent: H0Branch) {
        if !self.watched.contains(&child_id) {
            return;
        }
        self.redirect.insert(child_id, parent);
        self.watched.insert(parent.id);
    }

    fn observe(&mut self, merge: H0BranchMerge) {
        self.observe_parts(merge.child.id, merge.parent);
    }
}

#[derive(Debug)]
pub(crate) struct GlobalH0MergeTreeUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    branch: Vec<H0Branch>,
}

impl GlobalH0MergeTreeUnionFind {
    pub(crate) fn new(branch: Vec<H0Branch>) -> Self {
        assert!(
            branch.len() <= u32::MAX as usize,
            "too many interface nodes for u32 IDs"
        );
        Self {
            parent: (0..branch.len() as u32).collect(),
            rank: vec![0u8; branch.len()],
            branch,
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

    pub(crate) fn union(&mut self, a: u32, b: u32, value: u16) -> Option<H0BranchMerge> {
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
            Some(H0BranchMerge {
                value,
                child: younger,
                parent: older,
            })
        }
    }

    pub(crate) fn attach(
        &mut self,
        interface_node: u32,
        branch: H0Branch,
        value: u16,
    ) -> Option<H0BranchMerge> {
        let root = self.find(interface_node);
        let root_branch = self.branch[root as usize];
        if root_branch == branch {
            return None;
        }
        let older = root_branch.older(branch);
        let younger = root_branch.younger(branch);
        self.branch[root as usize] = older;

        Some(H0BranchMerge {
            value,
            child: younger,
            parent: older,
        })
    }

    pub(crate) fn essential_branches(&mut self) -> Vec<H0Branch> {
        let mut branches = Vec::new();
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 {
                branches.push(self.branch[node]);
            }
        }
        branches
    }
}

fn build_global_interface_branches(
    summaries: &[SlabH0MergeTreeSummary],
    offsets: &[u32],
) -> Vec<H0Branch> {
    let total = *offsets.last().unwrap_or(&0) as usize;
    let mut branches = vec![H0Branch { id: 0, birth: 0 }; total];

    for summary in summaries {
        for face in [&summary.z_min_face, &summary.z_max_face] {
            for ((&local_node, &birth), &id) in face
                .node_ids
                .iter()
                .zip(face.values.iter())
                .zip(face.branch_ids.iter())
            {
                let global = global_node_id(offsets, summary.slab_id, local_node) as usize;
                branches[global] = H0Branch { id, birth };
            }
        }
    }
    branches
}

fn generate_cross_slab_events(
    summaries: &[SlabH0MergeTreeSummary],
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
                            value: left_value.max(right_face.values[right_idx]),
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
                                    value: left_value.max(right_face.values[right_idx]),
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
    summaries: &[SlabH0MergeTreeSummary],
    connectivity: Connectivity,
) -> Result<MergeTree> {
    reduce_merge_tree_impl(summaries, connectivity, false, None)
}

pub(crate) fn reduce_merge_tree_with_deferred_parent_repair_prebucketed(
    summary: &SlabH0MergeTreeSummary,
    compact_local_by_value: Vec<Vec<CompactH0BranchMerge>>,
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
    summaries: &[SlabH0MergeTreeSummary],
    connectivity: Connectivity,
    repair_deferred_parents: bool,
    prebucketed_local: Option<Vec<Vec<CompactH0BranchMerge>>>,
) -> Result<MergeTree> {
    let offsets = build_interface_offsets(summaries);
    let branches = build_global_interface_branches(summaries, &offsets);
    let mut global_uf = GlobalH0MergeTreeUnionFind::new(branches);

    let using_prebucketed_local = prebucketed_local.is_some();
    if let Some(local_by_value) = prebucketed_local.as_ref()
        && local_by_value.len() != NUM_U16_VALUES
    {
        anyhow::bail!(
            "H0 compact prebucketed local event table has {} buckets; expected {}",
            local_by_value.len(),
            NUM_U16_VALUES
        );
    }
    let mut local_by_value: Option<Vec<Vec<H0BranchMerge>>> =
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
                "H0 compact prebucketed reduction requires summary.local_merge_events to be empty"
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
        // Hierarchical packed fast path.  Keep deferred-parent state and
        // same-threshold plateau contraction in open-addressed integer tables
        // instead of the HashSet+HashMap pair used by the reference reducer.
        let mut resolver = if repair_deferred_parents {
            let mut resolver = PackedDeferredIdResolver::default();
            for event in compact.iter().flatten() {
                resolver.watch(event.parent_id);
            }
            Some(resolver)
        } else {
            None
        };
        let mut recorder = H0PackedTreeRecorder::default();
        let mut attach_merges: Vec<H0BranchMerge> = Vec::new();
        let mut duplicate_targets: HashSet<(u64, u64)> = HashSet::new();
        let mut matched_duplicates: HashSet<(u64, u64)> = HashSet::new();
        let mut resolved_local_parent_ids: Vec<u64> = Vec::new();
        let mut observed_global_redirects: Vec<(u64, u64)> = Vec::new();

        for value in 0..NUM_U16_VALUES {
            let value_u16 = value as u16;
            attach_merges.clear();
            duplicate_targets.clear();
            matched_duplicates.clear();
            resolved_local_parent_ids.clear();
            observed_global_redirects.clear();

            attach_merges.reserve(attach_by_value[value].len());
            for event in &attach_by_value[value] {
                if let Some(merge) = global_uf.attach(event.node, event.branch, value_u16) {
                    attach_merges.push(merge);
                }
            }

            if repair_deferred_parents {
                duplicate_targets.extend(
                    attach_merges
                        .iter()
                        .map(|merge| (merge.child.id, merge.parent.id)),
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

                recorder.record_compact_merge_parent_id(value_u16, event, repaired_parent_id)?;
                resolved_local_parent_ids.push(repaired_parent_id);
            }

            for &merge in &attach_merges {
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && matched_duplicates.contains(&(merge.child.id, merge.parent.id));
                if !duplicate_hierarchical_transition {
                    recorder.record_merge(merge)?;
                    observed_global_redirects.push((merge.child.id, merge.parent.id));
                }
            }
            for event in &interface_by_value[value] {
                if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                    recorder.record_merge(merge)?;
                    observed_global_redirects.push((merge.child.id, merge.parent.id));
                }
            }
            for event in &cross_by_value[value] {
                if let Some(merge) = global_uf.union(event.a, event.b, value_u16) {
                    recorder.record_merge(merge)?;
                    observed_global_redirects.push((merge.child.id, merge.parent.id));
                }
            }

            recorder.finish_threshold()?;
            if let Some(resolver) = resolver.as_mut() {
                // Preserve threshold-delayed observation order exactly:
                // local events first, then attach/interface/cross events.
                for (&event, &repaired_parent_id) in
                    compact[value].iter().zip(resolved_local_parent_ids.iter())
                {
                    resolver.observe(event.child_id, repaired_parent_id);
                }
                for &(child_id, parent_id) in &observed_global_redirects {
                    resolver.observe(child_id, parent_id);
                }
            }
        }

        for branch in global_uf.essential_branches() {
            recorder.add_essential(branch)?;
        }
        return recorder.into_tree();
    }

    // General reducer used by the flat/in-memory path. Keep this path
    // unchanged as the independent reference implementation.
    let local = local_by_value
        .as_ref()
        .expect("non-prebucketed H0 reducer must own local event buckets");
    let mut resolver = if repair_deferred_parents {
        Some(H0DeferredParentResolver::from_local_events(local))
    } else {
        None
    };
    let mut recorder = H0TreeRecorder::default();
    for value in 0..NUM_U16_VALUES {
        let value_u16 = value as u16;
        let mut observed_merges = Vec::new();
        let mut recorded_local_merges = HashSet::new();

        {
            let resolver_ref = resolver.as_ref();
            for &event in &local[value] {
                let repaired = if let Some(resolver) = resolver_ref {
                    H0BranchMerge {
                        parent: resolver.resolve(event.parent),
                        ..event
                    }
                } else {
                    event
                };
                recorder.record_merge(repaired)?;
                recorded_local_merges.insert((repaired.child.id, repaired.parent.id));
                observed_merges.push(repaired);
            }
        }

        for event in &attach_by_value[value] {
            if let Some(merge) = global_uf.attach(event.node, event.branch, value_u16) {
                let duplicate_hierarchical_transition = repair_deferred_parents
                    && recorded_local_merges.contains(&(merge.child.id, merge.parent.id));
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

    for branch in global_uf.essential_branches() {
        recorder.add_essential(branch)?;
    }

    recorder.into_tree()
}

pub fn compute_h0_merge_tree_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<MergeTree> {
    let start = Instant::now();
    println!("Computing in-memory slabwise H0 merge tree...");

    let summaries = process_all_slabs(volume, slab_depth, connectivity)?;
    println!("Processed {} H0 merge-tree slab summaries", summaries.len());
    let tree = reduce_merge_tree(&summaries, connectivity)?;

    println!(
        "In-memory H0 merge-tree computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(tree)
}

#[cfg(test)]
mod deferred_parent_tests {
    use super::*;

    #[test]
    fn global_h0_same_branch_replay_does_not_emit_a_second_death() {
        let branch = H0Branch { id: 10, birth: 3 };
        let mut uf = GlobalH0MergeTreeUnionFind::new(vec![branch, branch]);

        assert!(uf.attach(0, branch, 5).is_none());
        assert!(uf.union(0, 1, 5).is_none());
        assert_eq!(uf.essential_branches(), vec![branch]);
    }

    #[test]
    fn deferred_h0_parent_follows_an_earlier_global_death() {
        let child = H0Branch { id: 10, birth: 3 };
        let provisional_parent = H0Branch { id: 20, birth: 1 };
        let survivor = H0Branch { id: 30, birth: 0 };
        let local = [vec![H0BranchMerge {
            value: 9,
            child,
            parent: provisional_parent,
        }]];
        let mut resolver = H0DeferredParentResolver::from_local_events(&local);

        resolver.observe(H0BranchMerge {
            value: 5,
            child: provisional_parent,
            parent: survivor,
        });

        assert_eq!(resolver.resolve(provisional_parent), survivor);
    }

    #[test]
    fn deferred_h0_parent_compact_bucket_matches_full_bucket() {
        let provisional_parent = H0Branch { id: 20, birth: 1 };
        let survivor = H0Branch { id: 30, birth: 0 };
        let compact = [vec![CompactH0BranchMerge::from_merge(H0BranchMerge {
            value: 9,
            child: H0Branch { id: 10, birth: 3 },
            parent: provisional_parent,
        })]];
        let mut resolver = H0DeferredParentResolver::from_compact_local_events(&compact);

        resolver.observe(H0BranchMerge {
            value: 5,
            child: provisional_parent,
            parent: survivor,
        });

        assert_eq!(resolver.resolve(provisional_parent), survivor);
    }

    #[test]
    fn deferred_h0_parent_follows_multiple_earlier_deaths() {
        let child = H0Branch { id: 10, birth: 4 };
        let first_parent = H0Branch { id: 20, birth: 2 };
        let second_parent = H0Branch { id: 30, birth: 1 };
        let survivor = H0Branch { id: 40, birth: 0 };
        let local = [vec![H0BranchMerge {
            value: 12,
            child,
            parent: first_parent,
        }]];
        let mut resolver = H0DeferredParentResolver::from_local_events(&local);

        resolver.observe(H0BranchMerge {
            value: 6,
            child: first_parent,
            parent: second_parent,
        });
        resolver.observe(H0BranchMerge {
            value: 8,
            child: second_parent,
            parent: survivor,
        });

        assert_eq!(resolver.resolve(first_parent), survivor);
    }
}

#[cfg(test)]
mod hierarchical_leaf_attach_pruning_tests {
    use super::*;

    fn two_node_uf(values: &[u16]) -> LocalUnionFind<'_> {
        let mut uf = LocalUnionFind {
            uf_state: LocalUnionFindState::new(
                2,
                ActiveStateStrategy::ParentSentinel,
                UnionFindLayoutStrategy::Packed,
            ),
            elder_local: vec![0, 1],
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
    fn hierarchical_h0_finalizes_nonpromoting_internal_branch_early() {
        let boundary = H0Branch { id: 1, birth: 1 };
        let internal = H0Branch { id: 2, birth: 3 };
        let values = [boundary.birth, internal.birth];
        let mut uf = two_node_uf(&values);
        let mut stats = H0LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, false, &mut stats);
        let action = action.unwrap();
        match action {
            LocalAction::LocalMerge(merge) => {
                assert_eq!(merge.value, 5);
                assert_eq!(merge.child_local, 1);
                assert_eq!(merge.child_birth, internal.birth);
                assert_eq!(merge.parent_local, 0);
            }
            other => panic!("expected early local finalization, got {other:?}"),
        }
        assert_eq!(stats.one_boundary_internal, 1);
        assert_eq!(stats.finalized_early, 1);
        assert_eq!(stats.propagated_attach, 0);
    }

    #[test]
    fn hierarchical_h0_keeps_promoting_internal_branch_as_attach() {
        let boundary = H0Branch { id: 1, birth: 3 };
        let internal = H0Branch { id: 2, birth: 1 };
        let values = [boundary.birth, internal.birth];
        let mut uf = two_node_uf(&values);
        let mut stats = H0LeafAttachStats::default();

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
    fn h0_plateau_native_elides_zero_local_merge_before_history() {
        let values = [5u16, 5u16];
        let mut uf = two_node_uf(&values);
        uf.interface_rep[0] = NO_INTERFACE_REP;
        let mut stats = H0LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, true, &mut stats);
        assert!(matches!(action, Some(LocalAction::None)));
        assert_eq!(stats.plateau_zero_local_merges_elided, 1);
    }

    #[test]
    fn h0_plateau_native_keeps_positive_local_merge() {
        let values = [1u16, 3u16];
        let mut uf = two_node_uf(&values);
        uf.interface_rep[0] = NO_INTERFACE_REP;
        let mut stats = H0LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, true, true, &mut stats);
        assert!(matches!(action, Some(LocalAction::LocalMerge(_))));
        assert_eq!(stats.plateau_zero_local_merges_elided, 0);
    }

    #[test]
    fn h0_reference_leaf_policy_preserves_original_attach() {
        let boundary = H0Branch { id: 1, birth: 1 };
        let internal = H0Branch { id: 2, birth: 3 };
        let values = [boundary.birth, internal.birth];
        let mut uf = two_node_uf(&values);
        let mut stats = H0LeafAttachStats::default();

        let (_, action) = uf.union_from_current_root(0, 1, 5, false, false, &mut stats);
        assert!(matches!(action, Some(LocalAction::Attach(_))));
        assert_eq!(stats.finalized_early, 0);
        assert_eq!(stats.propagated_attach, 1);
    }

    #[test]
    fn h0_inline_leaf_history_contracts_diagonal_parent() {
        let mut history = LocalMergeHistory::new(true, true, 64);
        history.push(LocalH0Merge {
            value: 20,
            child_local: 10,
            child_birth: 18,
            parent_local: 50,
        });
        history.push(LocalH0Merge {
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
    fn h0_inline_leaf_history_keeps_equal_threshold_positive_parent() {
        let mut history = LocalMergeHistory::new(true, true, 64);
        history.push(LocalH0Merge {
            value: 15,
            child_local: 10,
            child_birth: 14,
            parent_local: 50,
        });
        history.push(LocalH0Merge {
            value: 15,
            child_local: 50,
            child_birth: 10,
            parent_local: 60,
        });
        assert_eq!(history.repaired_parent_refs, 0);
        assert_eq!(history.events[0].parent_local, 50);
        assert_eq!(history.retained_events(), 2);
    }

    #[test]
    fn h0_reference_leaf_history_keeps_diagonal_event() {
        let mut history = LocalMergeHistory::new(false, true, 64);
        history.push(LocalH0Merge {
            value: 15,
            child_local: 50,
            child_birth: 15,
            parent_local: 60,
        });
        assert_eq!(history.input_events, 1);
        assert_eq!(history.contracted_zero_events, 0);
        assert_eq!(history.retained_events(), 1);
    }
}
