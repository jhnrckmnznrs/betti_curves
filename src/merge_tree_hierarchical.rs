use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::interface_sparsify::{InterfaceFiltration, sparsify_cross_interface};
use crate::io::TiffStackReader;
use crate::merge_tree_common::{
    CompactH0BranchMerge, CompactH2BranchMerge, H0Branch, H0BranchMerge, H2Branch, H2BranchMerge,
    MergeTree, canonicalize_h0_plateaus, canonicalize_h2_plateaus,
};
use crate::merge_tree_h0::{
    AttachEvent as H0AttachEvent, BoundaryFaceNodes as H0BoundaryFaceNodes, H0LeafAttachStats,
    InterfaceMergeEvent as H0InterfaceMergeEvent, SlabH0MergeTreeSummary,
    process_slab_h0_merge_tree_hierarchical,
    reduce_merge_tree_with_deferred_parent_repair_prebucketed as reduce_h0_merge_tree_prebucketed,
};
use crate::merge_tree_h2::{
    AttachEvent as H2AttachEvent, BoundaryFaceNodes as H2BoundaryFaceNodes, H2LeafAttachStats,
    InterfaceMergeEvent as H2InterfaceMergeEvent, SlabH2MergeTreeSummary,
    process_slab_h2_merge_tree_hierarchical,
    reduce_merge_tree_with_deferred_parent_repair_prebucketed as reduce_h2_merge_tree_prebucketed,
};
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
};

#[derive(Debug)]
struct HierarchicalSummary<S> {
    z0: usize,
    z1: usize,
    summary: S,
}

type H0LeafBatchItem = (
    usize,
    usize,
    usize,
    SlabH0MergeTreeSummary,
    H0LeafAttachStats,
    f64,
    f64,
);
type H0LeafBatchResult = Result<H0LeafBatchItem>;

type H2LeafBatchItem = (
    usize,
    usize,
    usize,
    SlabH2MergeTreeSummary,
    H2LeafAttachStats,
    f64,
    f64,
);
type H2LeafBatchResult = Result<H2LeafBatchItem>;

fn slab_ranges(depth: usize, slab_depth: usize) -> Vec<(usize, usize, usize)> {
    assert!(slab_depth > 0, "slab_depth must be positive");
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

const NUM_U16_VALUES: usize = 65_536;
const DEFAULT_HIER_LEAF_MEMORY_BUDGET_MB: usize = 512;
const ESTIMATED_HIER_LEAF_BYTES_PER_VOXEL: usize = 128;
const DEFAULT_MAX_HIER_LEAF_WORKERS: usize = 4;

fn materialized_fan_in_reference_enabled() -> bool {
    std::env::var("BETTI_HIER_MATERIALIZE_FANIN")
        .ok()
        .is_some_and(|raw| matches!(raw.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn early_finalize_leaf_attaches_enabled() -> bool {
    !std::env::var("BETTI_HIER_EARLY_FINALIZE_LEAF_ATTACHES")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

fn inline_leaf_history_contraction_enabled() -> bool {
    !std::env::var("BETTI_HIER_INLINE_LEAF_HISTORY")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

fn recursive_history_contraction_enabled() -> bool {
    !std::env::var("BETTI_HIER_RECURSIVE_HISTORY")
        .ok()
        .is_some_and(|raw| {
            matches!(
                raw.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
}

fn empty_h0_event_buckets() -> Vec<Vec<CompactH0BranchMerge>> {
    (0..NUM_U16_VALUES).map(|_| Vec::new()).collect()
}

fn empty_h2_event_buckets() -> Vec<Vec<CompactH2BranchMerge>> {
    (0..NUM_U16_VALUES).map(|_| Vec::new()).collect()
}

#[derive(Debug, Clone, Copy)]
struct HistoryHandle {
    value: u16,
    index: usize,
}

/// Online finalized-history contraction for hierarchical H0.
///
/// Only positive-persistence finalized branches are retained for root replay.
/// When branch `c` is finalized, every already-generated retained event that
/// still names `c` as parent is known to lie below the same hierarchical
/// summary.  References whose filtration time occurs after `c` dies are
/// redirected to the survivor immediately.  A zero-persistence branch is
/// never an exported tree node, so all references to it are redirected and
/// its own finalized event is discarded.
#[derive(Debug)]
struct H0FinalizedHistory {
    buckets: Vec<Vec<CompactH0BranchMerge>>,
    by_parent: HashMap<u64, Vec<HistoryHandle>>,
    input_events: usize,
    contracted_zero_events: usize,
    repaired_parent_refs: usize,
}

impl H0FinalizedHistory {
    fn new() -> Self {
        Self {
            buckets: empty_h0_event_buckets(),
            by_parent: HashMap::new(),
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
        }
    }

    fn push_merge(&mut self, merge: H0BranchMerge) {
        self.input_events += 1;
        let zero = merge.child.birth == merge.value;
        self.redirect_existing(
            merge.child.id,
            merge.parent.id,
            merge.parent.birth,
            merge.value,
            zero,
        );
        if zero {
            self.contracted_zero_events += 1;
            return;
        }

        let value = merge.value;
        let index = self.buckets[value as usize].len();
        self.buckets[value as usize].push(CompactH0BranchMerge::from_merge(merge));
        self.by_parent
            .entry(merge.parent.id)
            .or_default()
            .push(HistoryHandle { value, index });
    }

    fn extend(&mut self, events: impl IntoIterator<Item = H0BranchMerge>) {
        for event in events {
            self.push_merge(event);
        }
    }

    fn redirect_existing(
        &mut self,
        child_id: u64,
        parent_id: u64,
        parent_birth: u16,
        death: u16,
        zero: bool,
    ) {
        let Some(handles) = self.by_parent.remove(&child_id) else {
            return;
        };
        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for handle in handles {
            // H0 is a sublevel filtration. A positive branch remains the
            // correct parent through its own death threshold and redirects
            // only events occurring strictly later. A diagonal branch is
            // contracted on its birth/death plateau, so equality redirects.
            if zero || handle.value > death {
                let event = &mut self.buckets[handle.value as usize][handle.index];
                debug_assert_eq!(event.parent_id, child_id);
                event.parent_id = parent_id;
                event.parent_birth = parent_birth;
                self.repaired_parent_refs += 1;
                moved.push(handle);
            } else {
                keep.push(handle);
            }
        }
        if !keep.is_empty() {
            self.by_parent.insert(child_id, keep);
        }
        if !moved.is_empty() {
            self.by_parent.entry(parent_id).or_default().extend(moved);
        }
    }

    fn retained_events(&self) -> usize {
        self.input_events - self.contracted_zero_events
    }
}

/// H2 counterpart of `H0FinalizedHistory`. H2 runs in the opposite
/// filtration direction, so a positive branch redirects references at
/// strictly lower death values; diagonal branches redirect equality too.
#[derive(Debug)]
struct H2FinalizedHistory {
    buckets: Vec<Vec<CompactH2BranchMerge>>,
    by_parent: HashMap<u64, Vec<HistoryHandle>>,
    input_events: usize,
    contracted_zero_events: usize,
    repaired_parent_refs: usize,
}

impl H2FinalizedHistory {
    fn new() -> Self {
        Self {
            buckets: empty_h2_event_buckets(),
            by_parent: HashMap::new(),
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
        }
    }

    fn push_merge(&mut self, merge: H2BranchMerge) {
        self.input_events += 1;
        let zero = merge.child_birth == merge.value;
        let (parent_id, parent_birth) = match merge.parent {
            H2Branch::Outside => (crate::merge_tree_common::OUTSIDE_BRANCH_ID, 0),
            H2Branch::Finite { id, birth } => (id, birth),
        };
        self.redirect_existing(merge.child_id, parent_id, parent_birth, merge.value, zero);
        if zero {
            self.contracted_zero_events += 1;
            return;
        }

        let value = merge.value;
        let index = self.buckets[value as usize].len();
        self.buckets[value as usize].push(CompactH2BranchMerge::from_merge(merge));
        if parent_id != crate::merge_tree_common::OUTSIDE_BRANCH_ID {
            self.by_parent
                .entry(parent_id)
                .or_default()
                .push(HistoryHandle { value, index });
        }
    }

    fn extend(&mut self, events: impl IntoIterator<Item = H2BranchMerge>) {
        for event in events {
            self.push_merge(event);
        }
    }

    fn redirect_existing(
        &mut self,
        child_id: u64,
        parent_id: u64,
        parent_birth: u16,
        death: u16,
        zero: bool,
    ) {
        let Some(handles) = self.by_parent.remove(&child_id) else {
            return;
        };
        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for handle in handles {
            if zero || handle.value < death {
                let event = &mut self.buckets[handle.value as usize][handle.index];
                debug_assert_eq!(event.parent_id, child_id);
                event.parent_id = parent_id;
                event.parent_birth = parent_birth;
                self.repaired_parent_refs += 1;
                moved.push(handle);
            } else {
                keep.push(handle);
            }
        }
        if !keep.is_empty() {
            self.by_parent.insert(child_id, keep);
        }
        if parent_id != crate::merge_tree_common::OUTSIDE_BRANCH_ID && !moved.is_empty() {
            self.by_parent.entry(parent_id).or_default().extend(moved);
        }
    }

    fn retained_events(&self) -> usize {
        self.input_events - self.contracted_zero_events
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct RecursiveHistoryStats {
    nodes: u64,
    input_events: u64,
    retained_events: u64,
    contracted_zero_events: u64,
    repaired_parent_refs: u64,
}

/// Vec-backed H0 history contractor used inside one hierarchy node.
///
/// Child summaries already contain contracted positive history. The node
/// rebuilds the small parent index over those retained records, observes new
/// finalized deaths in event order, contracts diagonal deaths immediately,
/// and returns only the positive history that can still matter above this
/// node. Unlike `H0FinalizedHistory`, this structure allocates no 65,536-way
/// threshold bucket table, so it is cheap enough to instantiate per combine.
#[derive(Debug)]
struct H0RecursiveHistory {
    events: Vec<H0BranchMerge>,
    by_parent: HashMap<u64, Vec<usize>>,
    input_events: u64,
    contracted_zero_events: u64,
    repaired_parent_refs: u64,
}

impl H0RecursiveHistory {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            by_parent: HashMap::new(),
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
        }
    }

    fn push(&mut self, merge: H0BranchMerge) {
        self.input_events += 1;
        let zero = merge.child.birth == merge.value;
        let Some(indices) = self.by_parent.remove(&merge.child.id) else {
            if zero {
                self.contracted_zero_events += 1;
                return;
            }
            let index = self.events.len();
            self.events.push(merge);
            self.by_parent
                .entry(merge.parent.id)
                .or_default()
                .push(index);
            return;
        };

        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for index in indices {
            let event = &mut self.events[index];
            if zero || event.value > merge.value {
                debug_assert_eq!(event.parent.id, merge.child.id);
                event.parent = merge.parent;
                self.repaired_parent_refs += 1;
                moved.push(index);
            } else {
                keep.push(index);
            }
        }
        if !keep.is_empty() {
            self.by_parent.insert(merge.child.id, keep);
        }
        if !moved.is_empty() {
            self.by_parent
                .entry(merge.parent.id)
                .or_default()
                .extend(moved);
        }
        if zero {
            self.contracted_zero_events += 1;
            return;
        }
        let index = self.events.len();
        self.events.push(merge);
        self.by_parent
            .entry(merge.parent.id)
            .or_default()
            .push(index);
    }

    fn extend(&mut self, events: impl IntoIterator<Item = H0BranchMerge>) {
        for event in events {
            self.push(event);
        }
    }

    fn finish(self, stats: &mut RecursiveHistoryStats) -> Vec<H0BranchMerge> {
        stats.nodes += 1;
        stats.input_events += self.input_events;
        stats.retained_events += self.events.len() as u64;
        stats.contracted_zero_events += self.contracted_zero_events;
        stats.repaired_parent_refs += self.repaired_parent_refs;
        self.events
    }
}

#[derive(Debug)]
struct H2RecursiveHistory {
    events: Vec<H2BranchMerge>,
    by_parent: HashMap<u64, Vec<usize>>,
    input_events: u64,
    contracted_zero_events: u64,
    repaired_parent_refs: u64,
}

impl H2RecursiveHistory {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            by_parent: HashMap::new(),
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
        }
    }

    fn push(&mut self, merge: H2BranchMerge) {
        self.input_events += 1;
        let zero = merge.child_birth == merge.value;
        let (parent_id, parent_birth) = match merge.parent {
            H2Branch::Outside => (crate::merge_tree_common::OUTSIDE_BRANCH_ID, 0),
            H2Branch::Finite { id, birth } => (id, birth),
        };

        let handles = self.by_parent.remove(&merge.child_id).unwrap_or_default();
        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for index in handles {
            let event = &mut self.events[index];
            if zero || event.value < merge.value {
                let current_parent = match event.parent {
                    H2Branch::Outside => crate::merge_tree_common::OUTSIDE_BRANCH_ID,
                    H2Branch::Finite { id, .. } => id,
                };
                debug_assert_eq!(current_parent, merge.child_id);
                event.parent = if parent_id == crate::merge_tree_common::OUTSIDE_BRANCH_ID {
                    H2Branch::Outside
                } else {
                    H2Branch::Finite {
                        id: parent_id,
                        birth: parent_birth,
                    }
                };
                self.repaired_parent_refs += 1;
                moved.push(index);
            } else {
                keep.push(index);
            }
        }
        if !keep.is_empty() {
            self.by_parent.insert(merge.child_id, keep);
        }
        if parent_id != crate::merge_tree_common::OUTSIDE_BRANCH_ID && !moved.is_empty() {
            self.by_parent.entry(parent_id).or_default().extend(moved);
        }
        if zero {
            self.contracted_zero_events += 1;
            return;
        }
        let index = self.events.len();
        self.events.push(merge);
        if parent_id != crate::merge_tree_common::OUTSIDE_BRANCH_ID {
            self.by_parent.entry(parent_id).or_default().push(index);
        }
    }

    fn extend(&mut self, events: impl IntoIterator<Item = H2BranchMerge>) {
        for event in events {
            self.push(event);
        }
    }

    fn finish(self, stats: &mut RecursiveHistoryStats) -> Vec<H2BranchMerge> {
        stats.nodes += 1;
        stats.input_events += self.input_events;
        stats.retained_events += self.events.len() as u64;
        stats.contracted_zero_events += self.contracted_zero_events;
        stats.repaired_parent_refs += self.repaired_parent_refs;
        self.events
    }
}

fn hierarchical_leaf_workers(volume: &TiffStackReader, slab_depth: usize) -> usize {
    let rayon_threads = rayon::current_num_threads().max(1);
    if let Ok(raw) = std::env::var("BETTI_HIER_LEAF_WORKERS") {
        if let Ok(requested) = raw.parse::<usize>() {
            if requested > 0 {
                return requested.min(rayon_threads);
            }
        }
        eprintln!(
            "warning: ignoring invalid BETTI_HIER_LEAF_WORKERS={raw:?}; expected a positive integer"
        );
    }

    let budget_mb = std::env::var("BETTI_HIER_LEAF_BUDGET_MB")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_HIER_LEAF_MEMORY_BUDGET_MB);
    let budget_bytes = budget_mb.saturating_mul(1024 * 1024);
    let leaf_depth = slab_depth.min(volume.depth.max(1));
    let leaf_voxels = volume
        .width
        .saturating_mul(volume.height)
        .saturating_mul(leaf_depth);
    let estimated_worker_bytes = leaf_voxels
        .saturating_mul(ESTIMATED_HIER_LEAF_BYTES_PER_VOXEL)
        .max(1);
    let memory_limited = (budget_bytes / estimated_worker_bytes).max(1);

    rayon_threads
        .min(DEFAULT_MAX_HIER_LEAF_WORKERS)
        .min(memory_limited)
        .max(1)
}

#[derive(Debug, Clone, Copy)]
enum H0Action {
    None,
    /// A branch death is globally final because the dying component has no
    /// parent-boundary representative left.
    Final(H0BranchMerge),
    /// The elder branch visible at a parent-boundary representative changes.
    /// This transition must propagate, but its branch death is still
    /// provisional because higher fan-in levels may connect that boundary
    /// component at an earlier filtration value.
    Attach(H0AttachEvent),
    Interface(H0InterfaceMergeEvent),
}

#[derive(Debug)]
struct H0PairUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    branch: Vec<H0Branch>,
    terminal_rep: Vec<u32>,
}

impl H0PairUnionFind {
    fn new(branch: Vec<H0Branch>, terminal_rep: Vec<u32>) -> Self {
        assert_eq!(branch.len(), terminal_rep.len());
        assert!(branch.len() <= u32::MAX as usize);
        Self {
            parent: (0..branch.len() as u32).collect(),
            rank: vec![0; branch.len()],
            branch,
            terminal_rep,
        }
    }

    fn find(&mut self, mut node: u32) -> u32 {
        while self.parent[node as usize] != node {
            let parent = self.parent[node as usize];
            let grandparent = self.parent[parent as usize];
            self.parent[node as usize] = grandparent;
            node = parent;
        }
        node
    }

    fn link(&mut self, mut a: u32, mut b: u32) -> u32 {
        let rank_a = self.rank[a as usize];
        let rank_b = self.rank[b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        if rank_a == rank_b {
            self.rank[a as usize] = rank_a.saturating_add(1);
        }
        a
    }

    fn union(&mut self, a: u32, b: u32, value: u16) -> H0Action {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return H0Action::None;
        }

        let branch_a = self.branch[root_a as usize];
        let branch_b = self.branch[root_b as usize];
        let rep_a = self.terminal_rep[root_a as usize];
        let rep_b = self.terminal_rep[root_b as usize];
        let has_a = rep_a != NO_INTERFACE_REP;
        let has_b = rep_b != NO_INTERFACE_REP;
        let older = branch_a.older(branch_b);
        let younger = branch_a.younger(branch_b);

        let action = if branch_a == branch_b {
            if has_a && has_b {
                H0Action::Interface(H0InterfaceMergeEvent {
                    value,
                    a: rep_a,
                    b: rep_b,
                })
            } else {
                H0Action::None
            }
        } else {
            match (has_a, has_b) {
                (false, false) => H0Action::Final(H0BranchMerge {
                    value,
                    child: younger,
                    parent: older,
                }),
                (true, false) => {
                    if older == branch_a {
                        H0Action::Final(H0BranchMerge {
                            value,
                            child: branch_b,
                            parent: branch_a,
                        })
                    } else {
                        H0Action::Attach(H0AttachEvent {
                            value,
                            interface_node: rep_a,
                            branch: branch_b,
                        })
                    }
                }
                (false, true) => {
                    if older == branch_b {
                        H0Action::Final(H0BranchMerge {
                            value,
                            child: branch_a,
                            parent: branch_b,
                        })
                    } else {
                        H0Action::Attach(H0AttachEvent {
                            value,
                            interface_node: rep_b,
                            branch: branch_a,
                        })
                    }
                }
                (true, true) => H0Action::Interface(H0InterfaceMergeEvent {
                    value,
                    a: rep_a,
                    b: rep_b,
                }),
            }
        };

        let new_root = self.link(root_a, root_b);
        self.branch[new_root as usize] = older;
        self.terminal_rep[new_root as usize] = if has_a { rep_a } else { rep_b };
        action
    }

    fn attach(&mut self, node: u32, branch: H0Branch, value: u16) -> H0Action {
        let root = self.find(node);
        let root_branch = self.branch[root as usize];
        if root_branch == branch {
            return H0Action::None;
        }
        let older = root_branch.older(branch);
        let rep = self.terminal_rep[root as usize];
        self.branch[root as usize] = older;

        if rep == NO_INTERFACE_REP || older == root_branch {
            H0Action::Final(H0BranchMerge {
                value,
                child: if older == root_branch {
                    branch
                } else {
                    root_branch
                },
                parent: older,
            })
        } else {
            H0Action::Attach(H0AttachEvent {
                value,
                interface_node: rep,
                branch,
            })
        }
    }

    fn internal_roots(&mut self) -> usize {
        let mut count = 0;
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 && self.terminal_rep[node] == NO_INTERFACE_REP {
                count += 1;
            }
        }
        count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum H0EventKind {
    Attach { node: u32, branch: H0Branch },
    Interface { a: u32, b: u32 },
    Cross { a: u32, b: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct H0Event {
    value: u16,
    kind: H0EventKind,
}

struct H0OrderedEventStream<'a> {
    left_attach: &'a [H0AttachEvent],
    right_attach: &'a [H0AttachEvent],
    left_interface: &'a [H0InterfaceMergeEvent],
    right_interface: &'a [H0InterfaceMergeEvent],
    cross: &'a [H0InterfaceMergeEvent],
    cursor: [usize; 5],
    remaining: usize,
    left_offset: u32,
    right_offset: u32,
}

impl<'a> H0OrderedEventStream<'a> {
    fn new(
        left_attach: &'a [H0AttachEvent],
        right_attach: &'a [H0AttachEvent],
        left_interface: &'a [H0InterfaceMergeEvent],
        right_interface: &'a [H0InterfaceMergeEvent],
        cross: &'a [H0InterfaceMergeEvent],
        left_offset: u32,
        right_offset: u32,
    ) -> Self {
        debug_assert!(left_attach.windows(2).all(|w| w[0].value <= w[1].value));
        debug_assert!(right_attach.windows(2).all(|w| w[0].value <= w[1].value));
        debug_assert!(left_interface.windows(2).all(|w| w[0].value <= w[1].value));
        debug_assert!(right_interface.windows(2).all(|w| w[0].value <= w[1].value));
        debug_assert!(cross.windows(2).all(|w| w[0].value <= w[1].value));

        Self {
            left_attach,
            right_attach,
            left_interface,
            right_interface,
            cross,
            cursor: [0; 5],
            remaining: left_attach.len()
                + right_attach.len()
                + left_interface.len()
                + right_interface.len()
                + cross.len(),
            left_offset,
            right_offset,
        }
    }
}

impl Iterator for H0OrderedEventStream<'_> {
    type Item = H0Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let mut best: Option<(u16, u8, u8)> = None;
        let mut consider = |stream: u8, priority: u8, value: u16| {
            let key = (value, priority, stream);
            if best.is_none_or(|current| key < current) {
                best = Some(key);
            }
        };

        if self.cursor[0] < self.left_attach.len() {
            consider(0, 0, self.left_attach[self.cursor[0]].value);
        }
        if self.cursor[1] < self.right_attach.len() {
            consider(1, 0, self.right_attach[self.cursor[1]].value);
        }
        if self.cursor[2] < self.left_interface.len() {
            consider(2, 1, self.left_interface[self.cursor[2]].value);
        }
        if self.cursor[3] < self.right_interface.len() {
            consider(3, 1, self.right_interface[self.cursor[3]].value);
        }
        if self.cursor[4] < self.cross.len() {
            consider(4, 2, self.cross[self.cursor[4]].value);
        }

        let (_, _, stream) = best.expect("H0 streaming fan-in lost a non-empty source");
        let event = match stream {
            0 => {
                let source = self.left_attach[self.cursor[0]];
                self.cursor[0] += 1;
                H0Event {
                    value: source.value,
                    kind: H0EventKind::Attach {
                        node: self.left_offset + source.interface_node,
                        branch: source.branch,
                    },
                }
            }
            1 => {
                let source = self.right_attach[self.cursor[1]];
                self.cursor[1] += 1;
                H0Event {
                    value: source.value,
                    kind: H0EventKind::Attach {
                        node: self.right_offset + source.interface_node,
                        branch: source.branch,
                    },
                }
            }
            2 => {
                let source = self.left_interface[self.cursor[2]];
                self.cursor[2] += 1;
                H0Event {
                    value: source.value,
                    kind: H0EventKind::Interface {
                        a: self.left_offset + source.a,
                        b: self.left_offset + source.b,
                    },
                }
            }
            3 => {
                let source = self.right_interface[self.cursor[3]];
                self.cursor[3] += 1;
                H0Event {
                    value: source.value,
                    kind: H0EventKind::Interface {
                        a: self.right_offset + source.a,
                        b: self.right_offset + source.b,
                    },
                }
            }
            4 => {
                let source = self.cross[self.cursor[4]];
                self.cursor[4] += 1;
                H0Event {
                    value: source.value,
                    kind: H0EventKind::Cross {
                        a: source.a,
                        b: source.b,
                    },
                }
            }
            _ => unreachable!(),
        };
        self.remaining -= 1;
        Some(event)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for H0OrderedEventStream<'_> {}

fn ordered_h0_event_stream<'a>(
    left_attach: &'a [H0AttachEvent],
    right_attach: &'a [H0AttachEvent],
    left_interface: &'a [H0InterfaceMergeEvent],
    right_interface: &'a [H0InterfaceMergeEvent],
    cross: &'a [H0InterfaceMergeEvent],
    left_offset: u32,
    right_offset: u32,
) -> H0OrderedEventStream<'a> {
    H0OrderedEventStream::new(
        left_attach,
        right_attach,
        left_interface,
        right_interface,
        cross,
        left_offset,
        right_offset,
    )
}

fn merge_h0_event_streams(
    left_attach: &[H0AttachEvent],
    right_attach: &[H0AttachEvent],
    left_interface: &[H0InterfaceMergeEvent],
    right_interface: &[H0InterfaceMergeEvent],
    cross: &[H0InterfaceMergeEvent],
    left_offset: u32,
    right_offset: u32,
) -> Vec<H0Event> {
    ordered_h0_event_stream(
        left_attach,
        right_attach,
        left_interface,
        right_interface,
        cross,
        left_offset,
        right_offset,
    )
    .collect()
}

fn apply_h0_action(
    action: H0Action,
    history: &mut H0FinalizedHistory,
    recursive_history: &mut Option<H0RecursiveHistory>,
    attaches: &mut Vec<H0AttachEvent>,
    interfaces: &mut Vec<H0InterfaceMergeEvent>,
) {
    match action {
        H0Action::None => {}
        H0Action::Final(merge) => {
            if let Some(local) = recursive_history.as_mut() {
                local.push(merge);
            } else {
                history.push_merge(merge);
            }
        }
        H0Action::Attach(attach) => attaches.push(attach),
        H0Action::Interface(event) => interfaces.push(event),
    }
}

fn seed_h0_face(face: &H0BoundaryFaceNodes, offset: u32, branches: &mut [H0Branch]) {
    for ((&node, &birth), &id) in face
        .node_ids
        .iter()
        .zip(face.values.iter())
        .zip(face.branch_ids.iter())
    {
        branches[(offset + node) as usize] = H0Branch { id, birth };
    }
}

fn combine_h0(
    mut left: HierarchicalSummary<SlabH0MergeTreeSummary>,
    mut right: HierarchicalSummary<SlabH0MergeTreeSummary>,
    connectivity: Connectivity,
    history: &mut H0FinalizedHistory,
    recursive_contract: bool,
    recursive_stats: &mut RecursiveHistoryStats,
) -> Result<HierarchicalSummary<SlabH0MergeTreeSummary>> {
    let combine_start = Instant::now();
    let setup_start = Instant::now();
    let mut recursive_history = if recursive_contract {
        let mut local = H0RecursiveHistory::new();
        local.extend(std::mem::take(&mut left.summary.local_merge_events));
        local.extend(std::mem::take(&mut right.summary.local_merge_events));
        Some(local)
    } else {
        None
    };
    if left.z1 != right.z0 {
        bail!("hierarchical H0 branch-tree summaries are not adjacent");
    }
    let width = left.summary.z_min_face.width;
    let height = left.summary.z_min_face.height;
    if right.summary.z_min_face.width != width || right.summary.z_min_face.height != height {
        bail!("hierarchical H0 branch-tree face dimensions differ");
    }
    let face_size = width.checked_mul(height).expect("face size overflow");
    let left_count = left.summary.interface_node_count as usize;
    let right_count = right.summary.interface_node_count as usize;
    let total_nodes = left_count
        .checked_add(right_count)
        .ok_or_else(|| anyhow::anyhow!("hierarchical H0 pair node count overflow"))?;
    if total_nodes > u32::MAX as usize {
        bail!("hierarchical H0 branch-tree pair exceeds u32 node capacity");
    }

    let left_offset = 0u32;
    let right_offset = u32::try_from(left_count).expect("left H0 interface count exceeds u32");
    let seed = H0Branch { id: 0, birth: 0 };
    let mut branches = vec![seed; total_nodes];
    seed_h0_face(&left.summary.z_min_face, left_offset, &mut branches);
    seed_h0_face(&left.summary.z_max_face, left_offset, &mut branches);
    seed_h0_face(&right.summary.z_min_face, right_offset, &mut branches);
    seed_h0_face(&right.summary.z_max_face, right_offset, &mut branches);

    let parent_depth = right.z1 - left.z0;
    let mut terminals = vec![NO_INTERFACE_REP; total_nodes];
    for face_index in 0..face_size {
        let child = left.summary.z_min_face.node_ids[face_index];
        terminals[(left_offset + child) as usize] =
            face_node_id(0, parent_depth, face_index, face_size);
    }
    for face_index in 0..face_size {
        let child = right.summary.z_max_face.node_ids[face_index];
        terminals[(right_offset + child) as usize] =
            face_node_id(parent_depth - 1, parent_depth, face_index, face_size);
    }

    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let cross_start = Instant::now();
    let mut cross_events = Vec::new();
    let cross = sparsify_cross_interface(
        &left.summary.z_max_face.values,
        &right.summary.z_min_face.values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SublevelMax,
        |edge| {
            let li = edge.left_face_index as usize;
            let ri = edge.right_face_index as usize;
            cross_events.push(H0InterfaceMergeEvent {
                value: edge.value,
                a: left_offset + left.summary.z_max_face.node_ids[li],
                b: right_offset + right.summary.z_min_face.node_ids[ri],
            });
            Ok(())
        },
    )?;

    let cross_seconds = cross_start.elapsed().as_secs_f64();
    let input_attach_events = left.summary.attach_events.len() + right.summary.attach_events.len();
    let input_interface_events =
        left.summary.interface_merge_events.len() + right.summary.interface_merge_events.len();
    let streamed_events = input_attach_events + input_interface_events + cross_events.len();
    let materialize_reference = materialized_fan_in_reference_enabled();
    let sort_seconds = 0.0f64;
    let mut merge_seconds = 0.0f64;
    let mut materialized_events = 0usize;
    let materialized = if materialize_reference {
        let merge_start = Instant::now();
        let events = merge_h0_event_streams(
            &left.summary.attach_events,
            &right.summary.attach_events,
            &left.summary.interface_merge_events,
            &right.summary.interface_merge_events,
            &cross_events,
            left_offset,
            right_offset,
        );
        merge_seconds = merge_start.elapsed().as_secs_f64();
        materialized_events = events.len();
        Some(events)
    } else {
        None
    };

    let reduce_start = Instant::now();
    let mut uf = H0PairUnionFind::new(branches, terminals);
    let mut parent_attaches = Vec::new();
    let mut parent_interfaces = Vec::new();
    if let Some(events) = materialized {
        for event in events {
            let action = match event.kind {
                H0EventKind::Attach { node, branch } => uf.attach(node, branch, event.value),
                H0EventKind::Interface { a, b } | H0EventKind::Cross { a, b } => {
                    uf.union(a, b, event.value)
                }
            };
            apply_h0_action(
                action,
                history,
                &mut recursive_history,
                &mut parent_attaches,
                &mut parent_interfaces,
            );
        }
    } else {
        for event in ordered_h0_event_stream(
            &left.summary.attach_events,
            &right.summary.attach_events,
            &left.summary.interface_merge_events,
            &right.summary.interface_merge_events,
            &cross_events,
            left_offset,
            right_offset,
        ) {
            let action = match event.kind {
                H0EventKind::Attach { node, branch } => uf.attach(node, branch, event.value),
                H0EventKind::Interface { a, b } | H0EventKind::Cross { a, b } => {
                    uf.union(a, b, event.value)
                }
            };
            apply_h0_action(
                action,
                history,
                &mut recursive_history,
                &mut parent_attaches,
                &mut parent_interfaces,
            );
        }
    }

    let internal = uf.internal_roots();
    if internal != 0 {
        bail!(
            "hierarchical H0 branch-tree composition left {internal} components without an outer-face representative"
        );
    }

    let reduce_seconds = reduce_start.elapsed().as_secs_f64();
    let finish_start = Instant::now();

    let lower_values = left.summary.z_min_face.values;
    let lower_branch_ids = left.summary.z_min_face.branch_ids;
    let upper_values = right.summary.z_max_face.values;
    let upper_branch_ids = right.summary.z_max_face.branch_ids;
    let lower_node_ids = (0..face_size)
        .map(|i| face_node_id(0, parent_depth, i, face_size))
        .collect();
    let upper_node_ids = (0..face_size)
        .map(|i| face_node_id(parent_depth - 1, parent_depth, i, face_size))
        .collect();
    let interface_node_count = u32::try_from(slab_interface_node_count(face_size, parent_depth))
        .expect("hierarchical H0 branch-tree parent interface count exceeds u32");

    let finish_seconds = finish_start.elapsed().as_secs_f64();
    let total_seconds = combine_start.elapsed().as_secs_f64();
    let history_input_events = recursive_history.as_ref().map_or(0, |h| h.input_events);
    let history_retained_events = recursive_history
        .as_ref()
        .map_or(0, |h| h.events.len() as u64);
    let history_contracted_zero_events = recursive_history
        .as_ref()
        .map_or(0, |h| h.contracted_zero_events);

    println!(
        "PROFILE_BRANCH_H0_HIER_COMBINE z0={} z1={} pair_nodes={} parent_interface_nodes={} parent_attach={} parent_interface={} cross_retained={} cross_candidates={} input_attach={} input_interface={} fanin_mode={} streamed_events={} materialized_events={} history_input_events={} history_retained_events={} history_contracted_zero_events={} setup_seconds={:.6} cross_seconds={:.6} sort_seconds={:.6} merge_seconds={:.6} reduce_seconds={:.6} finish_seconds={:.6} total_seconds={:.6}",
        left.z0,
        right.z1,
        total_nodes,
        interface_node_count,
        parent_attaches.len(),
        parent_interfaces.len(),
        cross.retained_edges,
        cross.candidate_edges,
        input_attach_events,
        input_interface_events,
        if materialize_reference {
            "materialized"
        } else {
            "streaming"
        },
        streamed_events,
        materialized_events,
        history_input_events,
        history_retained_events,
        history_contracted_zero_events,
        setup_seconds,
        cross_seconds,
        sort_seconds,
        merge_seconds,
        reduce_seconds,
        finish_seconds,
        total_seconds,
    );

    Ok(HierarchicalSummary {
        z0: left.z0,
        z1: right.z1,
        summary: SlabH0MergeTreeSummary {
            slab_id: 0,
            local_merge_events: recursive_history
                .take()
                .map(|local| local.finish(recursive_stats))
                .unwrap_or_default(),
            attach_events: parent_attaches,
            interface_merge_events: parent_interfaces,
            interface_node_count,
            z_min_face: H0BoundaryFaceNodes {
                width,
                height,
                node_ids: lower_node_ids,
                values: lower_values,
                branch_ids: lower_branch_ids,
            },
            z_max_face: H0BoundaryFaceNodes {
                width,
                height,
                node_ids: upper_node_ids,
                values: upper_values,
                branch_ids: upper_branch_ids,
            },
        },
    })
}

#[derive(Debug, Clone, Copy)]
enum H2Action {
    None,
    /// A finite-branch death is globally final because the dying component has
    /// no parent-boundary representative left.
    Final(H2BranchMerge),
    /// The elder state visible at a parent-boundary representative changes.
    /// Propagate the state transition without finalizing the boundary-visible
    /// finite branch at this fan-in level.
    Attach(H2AttachEvent),
    Interface(H2InterfaceMergeEvent),
}

#[derive(Debug)]
struct H2PairUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    branch: Vec<H2Branch>,
    terminal_rep: Vec<u32>,
}

impl H2PairUnionFind {
    fn new(branch: Vec<H2Branch>, terminal_rep: Vec<u32>) -> Self {
        assert_eq!(branch.len(), terminal_rep.len());
        assert!(branch.len() <= u32::MAX as usize);
        Self {
            parent: (0..branch.len() as u32).collect(),
            rank: vec![0; branch.len()],
            branch,
            terminal_rep,
        }
    }

    fn find(&mut self, mut node: u32) -> u32 {
        while self.parent[node as usize] != node {
            let parent = self.parent[node as usize];
            let grandparent = self.parent[parent as usize];
            self.parent[node as usize] = grandparent;
            node = parent;
        }
        node
    }

    fn link(&mut self, mut a: u32, mut b: u32) -> u32 {
        let rank_a = self.rank[a as usize];
        let rank_b = self.rank[b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        if rank_a == rank_b {
            self.rank[a as usize] = rank_a.saturating_add(1);
        }
        a
    }

    fn merge_record(value: u16, a: H2Branch, b: H2Branch) -> Option<H2BranchMerge> {
        if a == b {
            return None;
        }
        let older = a.older(b);
        a.younger(b).map(|(child_id, child_birth)| H2BranchMerge {
            value,
            child_id,
            child_birth,
            parent: older,
        })
    }

    fn merge_event(value: u16, a: H2Branch, b: H2Branch) -> H2Action {
        Self::merge_record(value, a, b).map_or(H2Action::None, H2Action::Final)
    }

    fn union(&mut self, a: u32, b: u32, value: u16) -> H2Action {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return H2Action::None;
        }
        let branch_a = self.branch[root_a as usize];
        let branch_b = self.branch[root_b as usize];
        let rep_a = self.terminal_rep[root_a as usize];
        let rep_b = self.terminal_rep[root_b as usize];
        let has_a = rep_a != NO_INTERFACE_REP;
        let has_b = rep_b != NO_INTERFACE_REP;
        let older = branch_a.older(branch_b);

        let action = match (has_a, has_b) {
            (false, false) => Self::merge_event(value, branch_a, branch_b),
            (true, false) => {
                if older == branch_a {
                    Self::merge_event(value, branch_a, branch_b)
                } else {
                    H2Action::Attach(H2AttachEvent {
                        value,
                        interface_node: rep_a,
                        branch: branch_b,
                    })
                }
            }
            (false, true) => {
                if older == branch_b {
                    Self::merge_event(value, branch_a, branch_b)
                } else {
                    H2Action::Attach(H2AttachEvent {
                        value,
                        interface_node: rep_b,
                        branch: branch_a,
                    })
                }
            }
            (true, true) => H2Action::Interface(H2InterfaceMergeEvent {
                value,
                a: rep_a,
                b: rep_b,
            }),
        };

        let new_root = self.link(root_a, root_b);
        self.branch[new_root as usize] = older;
        self.terminal_rep[new_root as usize] = if has_a { rep_a } else { rep_b };
        action
    }

    fn attach(&mut self, node: u32, branch: H2Branch, value: u16) -> H2Action {
        let root = self.find(node);
        let root_branch = self.branch[root as usize];
        if root_branch == branch {
            return H2Action::None;
        }
        let older = root_branch.older(branch);
        let rep = self.terminal_rep[root as usize];
        self.branch[root as usize] = older;

        if rep == NO_INTERFACE_REP || older == root_branch {
            Self::merge_event(value, root_branch, branch)
        } else {
            H2Action::Attach(H2AttachEvent {
                value,
                interface_node: rep,
                branch,
            })
        }
    }

    fn internal_finite_roots(&mut self) -> usize {
        let mut count = 0;
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32
                && self.terminal_rep[node] == NO_INTERFACE_REP
                && !matches!(self.branch[node], H2Branch::Outside)
            {
                count += 1;
            }
        }
        count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum H2EventKind {
    Outside { node: u32 },
    Attach { node: u32, branch: H2Branch },
    Interface { a: u32, b: u32 },
    Cross { a: u32, b: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct H2Event {
    value: u16,
    kind: H2EventKind,
}

struct H2FilteredAttachCursor<'a> {
    events: &'a [H2AttachEvent],
    scan: usize,
    cached: Option<H2AttachEvent>,
    want_outside: bool,
}

impl<'a> H2FilteredAttachCursor<'a> {
    fn new(events: &'a [H2AttachEvent], want_outside: bool) -> Self {
        Self {
            events,
            scan: 0,
            cached: None,
            want_outside,
        }
    }

    fn peek(&mut self) -> Option<H2AttachEvent> {
        if self.cached.is_none() {
            while self.scan < self.events.len() {
                let event = self.events[self.scan];
                self.scan += 1;
                if (event.branch == H2Branch::Outside) == self.want_outside {
                    self.cached = Some(event);
                    break;
                }
            }
        }
        self.cached
    }

    fn pop(&mut self) -> H2AttachEvent {
        let event = self
            .peek()
            .expect("H2 filtered attach cursor popped an empty stream");
        self.cached = None;
        event
    }
}

struct H2OrderedEventStream<'a> {
    left_outside: H2FilteredAttachCursor<'a>,
    right_outside: H2FilteredAttachCursor<'a>,
    left_finite: H2FilteredAttachCursor<'a>,
    right_finite: H2FilteredAttachCursor<'a>,
    left_interface: &'a [H2InterfaceMergeEvent],
    right_interface: &'a [H2InterfaceMergeEvent],
    cross: &'a [H2InterfaceMergeEvent],
    interface_cursor: [usize; 3],
    remaining: usize,
    left_offset: u32,
    right_offset: u32,
}

impl<'a> H2OrderedEventStream<'a> {
    fn new(
        left_attach: &'a [H2AttachEvent],
        right_attach: &'a [H2AttachEvent],
        left_interface: &'a [H2InterfaceMergeEvent],
        right_interface: &'a [H2InterfaceMergeEvent],
        cross: &'a [H2InterfaceMergeEvent],
        left_offset: u32,
        right_offset: u32,
    ) -> Self {
        debug_assert!(left_attach.windows(2).all(|w| w[0].value >= w[1].value));
        debug_assert!(right_attach.windows(2).all(|w| w[0].value >= w[1].value));
        debug_assert!(left_interface.windows(2).all(|w| w[0].value >= w[1].value));
        debug_assert!(right_interface.windows(2).all(|w| w[0].value >= w[1].value));
        debug_assert!(cross.windows(2).all(|w| w[0].value >= w[1].value));

        Self {
            left_outside: H2FilteredAttachCursor::new(left_attach, true),
            right_outside: H2FilteredAttachCursor::new(right_attach, true),
            left_finite: H2FilteredAttachCursor::new(left_attach, false),
            right_finite: H2FilteredAttachCursor::new(right_attach, false),
            left_interface,
            right_interface,
            cross,
            interface_cursor: [0; 3],
            remaining: left_attach.len()
                + right_attach.len()
                + left_interface.len()
                + right_interface.len()
                + cross.len(),
            left_offset,
            right_offset,
        }
    }
}

impl Iterator for H2OrderedEventStream<'_> {
    type Item = H2Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let left_outside = self.left_outside.peek();
        let right_outside = self.right_outside.peek();
        let left_finite = self.left_finite.peek();
        let right_finite = self.right_finite.peek();

        let mut best: Option<(u16, u8, u8)> = None;
        let mut consider = |stream: u8, priority: u8, value: u16| {
            let candidate = (value, priority, stream);
            let replace = match best {
                None => true,
                Some(current) => {
                    candidate.0 > current.0
                        || (candidate.0 == current.0
                            && (candidate.1, candidate.2) < (current.1, current.2))
                }
            };
            if replace {
                best = Some(candidate);
            }
        };

        if let Some(event) = left_outside {
            consider(0, 0, event.value);
        }
        if let Some(event) = right_outside {
            consider(1, 0, event.value);
        }
        if let Some(event) = left_finite {
            consider(2, 1, event.value);
        }
        if let Some(event) = right_finite {
            consider(3, 1, event.value);
        }
        if self.interface_cursor[0] < self.left_interface.len() {
            consider(4, 2, self.left_interface[self.interface_cursor[0]].value);
        }
        if self.interface_cursor[1] < self.right_interface.len() {
            consider(5, 2, self.right_interface[self.interface_cursor[1]].value);
        }
        if self.interface_cursor[2] < self.cross.len() {
            consider(6, 3, self.cross[self.interface_cursor[2]].value);
        }

        let (_, _, stream) = best.expect("H2 streaming fan-in lost a non-empty source");
        let event = match stream {
            0 => {
                let source = self.left_outside.pop();
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Outside {
                        node: self.left_offset + source.interface_node,
                    },
                }
            }
            1 => {
                let source = self.right_outside.pop();
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Outside {
                        node: self.right_offset + source.interface_node,
                    },
                }
            }
            2 => {
                let source = self.left_finite.pop();
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Attach {
                        node: self.left_offset + source.interface_node,
                        branch: source.branch,
                    },
                }
            }
            3 => {
                let source = self.right_finite.pop();
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Attach {
                        node: self.right_offset + source.interface_node,
                        branch: source.branch,
                    },
                }
            }
            4 => {
                let source = self.left_interface[self.interface_cursor[0]];
                self.interface_cursor[0] += 1;
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Interface {
                        a: self.left_offset + source.a,
                        b: self.left_offset + source.b,
                    },
                }
            }
            5 => {
                let source = self.right_interface[self.interface_cursor[1]];
                self.interface_cursor[1] += 1;
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Interface {
                        a: self.right_offset + source.a,
                        b: self.right_offset + source.b,
                    },
                }
            }
            6 => {
                let source = self.cross[self.interface_cursor[2]];
                self.interface_cursor[2] += 1;
                H2Event {
                    value: source.value,
                    kind: H2EventKind::Cross {
                        a: source.a,
                        b: source.b,
                    },
                }
            }
            _ => unreachable!(),
        };
        self.remaining -= 1;
        Some(event)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for H2OrderedEventStream<'_> {}

fn ordered_h2_event_stream<'a>(
    left_attach: &'a [H2AttachEvent],
    right_attach: &'a [H2AttachEvent],
    left_interface: &'a [H2InterfaceMergeEvent],
    right_interface: &'a [H2InterfaceMergeEvent],
    cross: &'a [H2InterfaceMergeEvent],
    left_offset: u32,
    right_offset: u32,
) -> H2OrderedEventStream<'a> {
    H2OrderedEventStream::new(
        left_attach,
        right_attach,
        left_interface,
        right_interface,
        cross,
        left_offset,
        right_offset,
    )
}

fn merge_h2_event_streams(
    left_attach: &[H2AttachEvent],
    right_attach: &[H2AttachEvent],
    left_interface: &[H2InterfaceMergeEvent],
    right_interface: &[H2InterfaceMergeEvent],
    cross: &[H2InterfaceMergeEvent],
    left_offset: u32,
    right_offset: u32,
) -> Vec<H2Event> {
    ordered_h2_event_stream(
        left_attach,
        right_attach,
        left_interface,
        right_interface,
        cross,
        left_offset,
        right_offset,
    )
    .collect()
}

fn apply_h2_action(
    action: H2Action,
    history: &mut H2FinalizedHistory,
    recursive_history: &mut Option<H2RecursiveHistory>,
    attaches: &mut Vec<H2AttachEvent>,
    interfaces: &mut Vec<H2InterfaceMergeEvent>,
) {
    match action {
        H2Action::None => {}
        H2Action::Final(merge) => {
            if let Some(local) = recursive_history.as_mut() {
                local.push(merge);
            } else {
                history.push_merge(merge);
            }
        }
        H2Action::Attach(attach) => attaches.push(attach),
        H2Action::Interface(event) => interfaces.push(event),
    }
}

fn seed_h2_face(face: &H2BoundaryFaceNodes, offset: u32, branches: &mut [H2Branch]) {
    for ((&node, &birth), &id) in face
        .node_ids
        .iter()
        .zip(face.values.iter())
        .zip(face.branch_ids.iter())
    {
        branches[(offset + node) as usize] = H2Branch::Finite { id, birth };
    }
}

fn combine_h2(
    mut left: HierarchicalSummary<SlabH2MergeTreeSummary>,
    mut right: HierarchicalSummary<SlabH2MergeTreeSummary>,
    connectivity: Connectivity,
    history: &mut H2FinalizedHistory,
    recursive_contract: bool,
    recursive_stats: &mut RecursiveHistoryStats,
) -> Result<HierarchicalSummary<SlabH2MergeTreeSummary>> {
    let combine_start = Instant::now();
    let setup_start = Instant::now();
    let mut recursive_history = if recursive_contract {
        let mut local = H2RecursiveHistory::new();
        local.extend(std::mem::take(&mut left.summary.local_merge_events));
        local.extend(std::mem::take(&mut right.summary.local_merge_events));
        Some(local)
    } else {
        None
    };
    if left.z1 != right.z0 {
        bail!("hierarchical H2 branch-tree summaries are not adjacent");
    }
    let width = left.summary.z_min_face.width;
    let height = left.summary.z_min_face.height;
    if right.summary.z_min_face.width != width || right.summary.z_min_face.height != height {
        bail!("hierarchical H2 branch-tree face dimensions differ");
    }
    let face_size = width.checked_mul(height).expect("face size overflow");
    let left_count = left.summary.interface_node_count as usize;
    let right_count = right.summary.interface_node_count as usize;
    let total_nodes = left_count
        .checked_add(right_count)
        .ok_or_else(|| anyhow::anyhow!("hierarchical H2 pair node count overflow"))?;
    if total_nodes > u32::MAX as usize {
        bail!("hierarchical H2 branch-tree pair exceeds u32 node capacity");
    }

    let left_offset = 0u32;
    let right_offset = u32::try_from(left_count).expect("left H2 interface count exceeds u32");
    let seed = H2Branch::Finite { id: 0, birth: 0 };
    let mut branches = vec![seed; total_nodes];
    seed_h2_face(&left.summary.z_min_face, left_offset, &mut branches);
    seed_h2_face(&left.summary.z_max_face, left_offset, &mut branches);
    seed_h2_face(&right.summary.z_min_face, right_offset, &mut branches);
    seed_h2_face(&right.summary.z_max_face, right_offset, &mut branches);

    let parent_depth = right.z1 - left.z0;
    let mut terminals = vec![NO_INTERFACE_REP; total_nodes];
    for face_index in 0..face_size {
        let child = left.summary.z_min_face.node_ids[face_index];
        terminals[(left_offset + child) as usize] =
            face_node_id(0, parent_depth, face_index, face_size);
    }
    for face_index in 0..face_size {
        let child = right.summary.z_max_face.node_ids[face_index];
        terminals[(right_offset + child) as usize] =
            face_node_id(parent_depth - 1, parent_depth, face_index, face_size);
    }

    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let cross_start = Instant::now();
    let mut cross_events = Vec::new();
    let cross = sparsify_cross_interface(
        &left.summary.z_max_face.values,
        &right.summary.z_min_face.values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SuperlevelMin,
        |edge| {
            let li = edge.left_face_index as usize;
            let ri = edge.right_face_index as usize;
            cross_events.push(H2InterfaceMergeEvent {
                value: edge.value,
                a: left_offset + left.summary.z_max_face.node_ids[li],
                b: right_offset + right.summary.z_min_face.node_ids[ri],
            });
            Ok(())
        },
    )?;

    let cross_seconds = cross_start.elapsed().as_secs_f64();
    let input_attach_events = left.summary.attach_events.len() + right.summary.attach_events.len();
    let input_interface_events =
        left.summary.interface_merge_events.len() + right.summary.interface_merge_events.len();
    let streamed_events = input_attach_events + input_interface_events + cross_events.len();
    let materialize_reference = materialized_fan_in_reference_enabled();
    let sort_seconds = 0.0f64;
    let mut merge_seconds = 0.0f64;
    let mut materialized_events = 0usize;
    let materialized = if materialize_reference {
        let merge_start = Instant::now();
        let events = merge_h2_event_streams(
            &left.summary.attach_events,
            &right.summary.attach_events,
            &left.summary.interface_merge_events,
            &right.summary.interface_merge_events,
            &cross_events,
            left_offset,
            right_offset,
        );
        merge_seconds = merge_start.elapsed().as_secs_f64();
        materialized_events = events.len();
        Some(events)
    } else {
        None
    };

    let reduce_start = Instant::now();
    let mut uf = H2PairUnionFind::new(branches, terminals);
    let mut parent_attaches = Vec::new();
    let mut parent_interfaces = Vec::new();
    if let Some(events) = materialized {
        for event in events {
            let action = match event.kind {
                H2EventKind::Outside { node } => uf.attach(node, H2Branch::Outside, event.value),
                H2EventKind::Attach { node, branch } => uf.attach(node, branch, event.value),
                H2EventKind::Interface { a, b } | H2EventKind::Cross { a, b } => {
                    uf.union(a, b, event.value)
                }
            };
            apply_h2_action(
                action,
                history,
                &mut recursive_history,
                &mut parent_attaches,
                &mut parent_interfaces,
            );
        }
    } else {
        for event in ordered_h2_event_stream(
            &left.summary.attach_events,
            &right.summary.attach_events,
            &left.summary.interface_merge_events,
            &right.summary.interface_merge_events,
            &cross_events,
            left_offset,
            right_offset,
        ) {
            let action = match event.kind {
                H2EventKind::Outside { node } => uf.attach(node, H2Branch::Outside, event.value),
                H2EventKind::Attach { node, branch } => uf.attach(node, branch, event.value),
                H2EventKind::Interface { a, b } | H2EventKind::Cross { a, b } => {
                    uf.union(a, b, event.value)
                }
            };
            apply_h2_action(
                action,
                history,
                &mut recursive_history,
                &mut parent_attaches,
                &mut parent_interfaces,
            );
        }
    }

    let internal = uf.internal_finite_roots();
    if internal != 0 {
        bail!(
            "hierarchical H2 branch-tree composition left {internal} finite components without an outer-face representative"
        );
    }

    let reduce_seconds = reduce_start.elapsed().as_secs_f64();
    let finish_start = Instant::now();

    let lower_values = left.summary.z_min_face.values;
    let lower_branch_ids = left.summary.z_min_face.branch_ids;
    let upper_values = right.summary.z_max_face.values;
    let upper_branch_ids = right.summary.z_max_face.branch_ids;
    let lower_node_ids = (0..face_size)
        .map(|i| face_node_id(0, parent_depth, i, face_size))
        .collect();
    let upper_node_ids = (0..face_size)
        .map(|i| face_node_id(parent_depth - 1, parent_depth, i, face_size))
        .collect();
    let interface_node_count = u32::try_from(slab_interface_node_count(face_size, parent_depth))
        .expect("hierarchical H2 branch-tree parent interface count exceeds u32");

    let finish_seconds = finish_start.elapsed().as_secs_f64();
    let total_seconds = combine_start.elapsed().as_secs_f64();
    let history_input_events = recursive_history.as_ref().map_or(0, |h| h.input_events);
    let history_retained_events = recursive_history
        .as_ref()
        .map_or(0, |h| h.events.len() as u64);
    let history_contracted_zero_events = recursive_history
        .as_ref()
        .map_or(0, |h| h.contracted_zero_events);

    println!(
        "PROFILE_BRANCH_H2_HIER_COMBINE z0={} z1={} pair_nodes={} parent_interface_nodes={} parent_attach={} parent_interface={} cross_retained={} cross_candidates={} input_attach={} input_interface={} fanin_mode={} streamed_events={} materialized_events={} history_input_events={} history_retained_events={} history_contracted_zero_events={} setup_seconds={:.6} cross_seconds={:.6} sort_seconds={:.6} merge_seconds={:.6} reduce_seconds={:.6} finish_seconds={:.6} total_seconds={:.6}",
        left.z0,
        right.z1,
        total_nodes,
        interface_node_count,
        parent_attaches.len(),
        parent_interfaces.len(),
        cross.retained_edges,
        cross.candidate_edges,
        input_attach_events,
        input_interface_events,
        if materialize_reference {
            "materialized"
        } else {
            "streaming"
        },
        streamed_events,
        materialized_events,
        history_input_events,
        history_retained_events,
        history_contracted_zero_events,
        setup_seconds,
        cross_seconds,
        sort_seconds,
        merge_seconds,
        reduce_seconds,
        finish_seconds,
        total_seconds,
    );

    Ok(HierarchicalSummary {
        z0: left.z0,
        z1: right.z1,
        summary: SlabH2MergeTreeSummary {
            slab_id: 0,
            local_merge_events: recursive_history
                .take()
                .map(|local| local.finish(recursive_stats))
                .unwrap_or_default(),
            attach_events: parent_attaches,
            interface_merge_events: parent_interfaces,
            interface_node_count,
            z_min_face: H2BoundaryFaceNodes {
                width,
                height,
                node_ids: lower_node_ids,
                values: lower_values,
                branch_ids: lower_branch_ids,
            },
            z_max_face: H2BoundaryFaceNodes {
                width,
                height,
                node_ids: upper_node_ids,
                values: upper_values,
                branch_ids: upper_branch_ids,
            },
        },
    })
}

fn fan_in<S>(
    mut current: HierarchicalSummary<S>,
    slots: &mut Vec<Option<HierarchicalSummary<S>>>,
    mut combine: impl FnMut(
        HierarchicalSummary<S>,
        HierarchicalSummary<S>,
    ) -> Result<HierarchicalSummary<S>>,
    combine_count: &mut usize,
    max_pair_nodes: &mut usize,
    interface_count: impl Fn(&S) -> usize,
) -> Result<()> {
    let mut level = 0usize;
    loop {
        if level == slots.len() {
            slots.push(Some(current));
            return Ok(());
        }
        if let Some(left) = slots[level].take() {
            *max_pair_nodes = (*max_pair_nodes)
                .max(interface_count(&left.summary) + interface_count(&current.summary));
            current = combine(left, current)?;
            *combine_count += 1;
            level += 1;
        } else {
            slots[level] = Some(current);
            return Ok(());
        }
    }
}

pub fn compute_h0_merge_tree_hierarchical_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<MergeTree> {
    let start = Instant::now();
    let ranges = slab_ranges(volume.depth, slab_depth);
    let leaf_count = ranges.len();
    let leaf_workers = hierarchical_leaf_workers(volume, slab_depth);
    let mut slots: Vec<Option<HierarchicalSummary<SlabH0MergeTreeSummary>>> = Vec::new();
    let mut finalized_history = H0FinalizedHistory::new();
    let mut combines = 0usize;
    let mut max_pair_nodes = 0usize;
    let mut max_live_summaries = 0usize;
    let mut leaf_batch_seconds = 0.0f64;
    let mut leaf_read_thread_seconds = 0.0f64;
    let mut leaf_compute_thread_seconds = 0.0f64;
    let mut bucket_seconds = 0.0f64;
    let mut fan_in_seconds = 0.0f64;
    let mut root_combine_seconds = 0.0f64;
    let early_finalize_leaf_attaches = early_finalize_leaf_attaches_enabled();
    let inline_leaf_history = inline_leaf_history_contraction_enabled();
    let recursive_history = recursive_history_contraction_enabled();
    let mut recursive_history_stats = RecursiveHistoryStats::default();
    let mut leaf_attach_stats = H0LeafAttachStats::default();

    for batch in ranges.chunks(leaf_workers) {
        let leaf_batch_start = Instant::now();
        let parallel_results: Vec<H0LeafBatchResult> = batch
            .par_iter()
            .map(|&(slab_id, z0, z1)| {
                let read_start = Instant::now();
                let block = volume.read_z_slab(z0, z1)?;
                let read_seconds = read_start.elapsed().as_secs_f64();
                let compute_start = Instant::now();
                let (summary, attach_stats) = process_slab_h0_merge_tree_hierarchical(
                    slab_id,
                    &block,
                    connectivity,
                    early_finalize_leaf_attaches,
                    inline_leaf_history,
                );
                let compute_seconds = compute_start.elapsed().as_secs_f64();
                Ok((
                    slab_id,
                    z0,
                    z1,
                    summary,
                    attach_stats,
                    read_seconds,
                    compute_seconds,
                ))
            })
            .collect();
        leaf_batch_seconds += leaf_batch_start.elapsed().as_secs_f64();
        let mut leaves = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;
        leaves.sort_by_key(|entry| entry.0);

        for (_slab_id, z0, z1, mut summary, attach_stats, read_seconds, compute_seconds) in leaves {
            leaf_attach_stats.one_boundary_internal += attach_stats.one_boundary_internal;
            leaf_attach_stats.finalized_early += attach_stats.finalized_early;
            leaf_attach_stats.propagated_attach += attach_stats.propagated_attach;
            leaf_attach_stats.local_history_input_events += attach_stats.local_history_input_events;
            leaf_attach_stats.local_history_retained_events +=
                attach_stats.local_history_retained_events;
            leaf_attach_stats.local_history_contracted_zero_events +=
                attach_stats.local_history_contracted_zero_events;
            leaf_attach_stats.local_history_repaired_parent_refs +=
                attach_stats.local_history_repaired_parent_refs;
            leaf_read_thread_seconds += read_seconds;
            leaf_compute_thread_seconds += compute_seconds;
            if !recursive_history {
                let bucket_start = Instant::now();
                finalized_history.extend(std::mem::take(&mut summary.local_merge_events));
                bucket_seconds += bucket_start.elapsed().as_secs_f64();
            }
            summary.slab_id = 0;
            let fan_in_start = Instant::now();
            fan_in(
                HierarchicalSummary { z0, z1, summary },
                &mut slots,
                |left, right| {
                    combine_h0(
                        left,
                        right,
                        connectivity,
                        &mut finalized_history,
                        recursive_history,
                        &mut recursive_history_stats,
                    )
                },
                &mut combines,
                &mut max_pair_nodes,
                |summary| summary.interface_node_count as usize,
            )?;
            fan_in_seconds += fan_in_start.elapsed().as_secs_f64();
            max_live_summaries = max_live_summaries.max(slots.iter().flatten().count());
        }
    }

    let mut remaining: Vec<_> = slots.into_iter().flatten().collect();
    if remaining.is_empty() {
        bail!("hierarchical H0 branch tree received an empty volume");
    }
    remaining.sort_by_key(|summary| summary.z0);
    let mut root = remaining.remove(0);
    for right in remaining {
        max_pair_nodes = max_pair_nodes.max(
            root.summary.interface_node_count as usize
                + right.summary.interface_node_count as usize,
        );
        let root_combine_start = Instant::now();
        root = combine_h0(
            root,
            right,
            connectivity,
            &mut finalized_history,
            recursive_history,
            &mut recursive_history_stats,
        )?;
        root_combine_seconds += root_combine_start.elapsed().as_secs_f64();
        combines += 1;
    }

    if recursive_history {
        let bucket_start = Instant::now();
        finalized_history.extend(std::mem::take(&mut root.summary.local_merge_events));
        bucket_seconds += bucket_start.elapsed().as_secs_f64();
    } else {
        debug_assert!(root.summary.local_merge_events.is_empty());
    }
    let central_history_input_events = finalized_history.input_events;
    let finalized_input_events = central_history_input_events
        .saturating_add(leaf_attach_stats.local_history_contracted_zero_events as usize)
        .saturating_add(recursive_history_stats.contracted_zero_events as usize);
    let contracted_zero_events = finalized_history
        .contracted_zero_events
        .saturating_add(leaf_attach_stats.local_history_contracted_zero_events as usize)
        .saturating_add(recursive_history_stats.contracted_zero_events as usize);
    let repaired_history_parent_refs = finalized_history
        .repaired_parent_refs
        .saturating_add(leaf_attach_stats.local_history_repaired_parent_refs as usize)
        .saturating_add(recursive_history_stats.repaired_parent_refs as usize);
    let finalized_events = finalized_history.retained_events();
    debug_assert_eq!(
        finalized_events,
        finalized_input_events - contracted_zero_events
    );
    debug_assert_eq!(
        finalized_events,
        finalized_history
            .buckets
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    );
    let root_reduce_start = Instant::now();
    let raw_tree =
        reduce_h0_merge_tree_prebucketed(&root.summary, finalized_history.buckets, connectivity)?;
    let root_reduce_seconds = root_reduce_start.elapsed().as_secs_f64();
    let canonical_start = Instant::now();
    let tree = canonicalize_h0_plateaus(raw_tree)?;
    let canonical_seconds = canonical_start.elapsed().as_secs_f64();
    println!(
        "PROFILE_BRANCH_H0_HIERARCHY leaf_slabs={} leaf_workers={} combines={} max_live_summaries={} max_pair_nodes={} final_interface_nodes={} leaf_attach_policy={} leaf_history_policy={} recursive_history_policy={} leaf_one_boundary_internal={} leaf_finalized_attach_early={} leaf_propagated_attach={} leaf_local_history_input_events={} leaf_local_history_retained_events={} leaf_local_history_contracted_zero_events={} leaf_local_history_repaired_parent_refs={} recursive_history_nodes={} recursive_history_input_events={} recursive_history_retained_events={} recursive_history_contracted_zero_events={} recursive_history_repaired_parent_refs={} central_history_input_events={} finalized_input_events={} finalized_events={} contracted_zero_events={} repaired_history_parent_refs={} compact_event_bytes={} full_event_bytes={} finalized_storage_bytes={} leaf_batch_seconds={:.6} leaf_read_thread_seconds={:.6} leaf_compute_thread_seconds={:.6} bucket_seconds={:.6} fan_in_seconds={:.6} root_combine_seconds={:.6} root_reduce_seconds={:.6} canonical_seconds={:.6} canonical_parenting=plateau",
        leaf_count,
        leaf_workers,
        combines,
        max_live_summaries,
        max_pair_nodes,
        slab_interface_node_count(volume.width * volume.height, volume.depth),
        if early_finalize_leaf_attaches {
            "early-finalize"
        } else {
            "reference"
        },
        if inline_leaf_history {
            "inline-contract"
        } else {
            "reference"
        },
        if recursive_history {
            "recursive-contract"
        } else {
            "reference"
        },
        leaf_attach_stats.one_boundary_internal,
        leaf_attach_stats.finalized_early,
        leaf_attach_stats.propagated_attach,
        leaf_attach_stats.local_history_input_events,
        leaf_attach_stats.local_history_retained_events,
        leaf_attach_stats.local_history_contracted_zero_events,
        leaf_attach_stats.local_history_repaired_parent_refs,
        recursive_history_stats.nodes,
        recursive_history_stats.input_events,
        recursive_history_stats.retained_events,
        recursive_history_stats.contracted_zero_events,
        recursive_history_stats.repaired_parent_refs,
        central_history_input_events,
        finalized_input_events,
        finalized_events,
        contracted_zero_events,
        repaired_history_parent_refs,
        std::mem::size_of::<CompactH0BranchMerge>(),
        std::mem::size_of::<H0BranchMerge>(),
        finalized_events.saturating_mul(std::mem::size_of::<CompactH0BranchMerge>()),
        leaf_batch_seconds,
        leaf_read_thread_seconds,
        leaf_compute_thread_seconds,
        bucket_seconds,
        fan_in_seconds,
        root_combine_seconds,
        root_reduce_seconds,
        canonical_seconds,
    );
    println!(
        "Hierarchical H0 branch-tree computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(tree)
}

pub fn compute_h2_merge_tree_hierarchical_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<MergeTree> {
    let start = Instant::now();
    let ranges = slab_ranges(volume.depth, slab_depth);
    let leaf_count = ranges.len();
    let leaf_workers = hierarchical_leaf_workers(volume, slab_depth);
    let global_shape = volume.shape();
    let mut slots: Vec<Option<HierarchicalSummary<SlabH2MergeTreeSummary>>> = Vec::new();
    let mut finalized_history = H2FinalizedHistory::new();
    let mut combines = 0usize;
    let mut max_pair_nodes = 0usize;
    let mut max_live_summaries = 0usize;
    let mut leaf_batch_seconds = 0.0f64;
    let mut leaf_read_thread_seconds = 0.0f64;
    let mut leaf_compute_thread_seconds = 0.0f64;
    let mut bucket_seconds = 0.0f64;
    let mut fan_in_seconds = 0.0f64;
    let mut root_combine_seconds = 0.0f64;
    let early_finalize_leaf_attaches = early_finalize_leaf_attaches_enabled();
    let inline_leaf_history = inline_leaf_history_contraction_enabled();
    let recursive_history = recursive_history_contraction_enabled();
    let mut recursive_history_stats = RecursiveHistoryStats::default();
    let mut leaf_attach_stats = H2LeafAttachStats::default();

    for batch in ranges.chunks(leaf_workers) {
        let leaf_batch_start = Instant::now();
        let parallel_results: Vec<H2LeafBatchResult> = batch
            .par_iter()
            .map(|&(slab_id, z0, z1)| {
                let read_start = Instant::now();
                let block = volume.read_z_slab(z0, z1)?;
                let read_seconds = read_start.elapsed().as_secs_f64();
                let compute_start = Instant::now();
                let (summary, attach_stats) = process_slab_h2_merge_tree_hierarchical(
                    slab_id,
                    &block,
                    global_shape,
                    background_connectivity,
                    early_finalize_leaf_attaches,
                    inline_leaf_history,
                );
                let compute_seconds = compute_start.elapsed().as_secs_f64();
                Ok((
                    slab_id,
                    z0,
                    z1,
                    summary,
                    attach_stats,
                    read_seconds,
                    compute_seconds,
                ))
            })
            .collect();
        leaf_batch_seconds += leaf_batch_start.elapsed().as_secs_f64();
        let mut leaves = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;
        leaves.sort_by_key(|entry| entry.0);

        for (_slab_id, z0, z1, mut summary, attach_stats, read_seconds, compute_seconds) in leaves {
            leaf_attach_stats.one_boundary_internal += attach_stats.one_boundary_internal;
            leaf_attach_stats.finalized_early += attach_stats.finalized_early;
            leaf_attach_stats.propagated_attach += attach_stats.propagated_attach;
            leaf_attach_stats.local_history_input_events += attach_stats.local_history_input_events;
            leaf_attach_stats.local_history_retained_events +=
                attach_stats.local_history_retained_events;
            leaf_attach_stats.local_history_contracted_zero_events +=
                attach_stats.local_history_contracted_zero_events;
            leaf_attach_stats.local_history_repaired_parent_refs +=
                attach_stats.local_history_repaired_parent_refs;
            leaf_read_thread_seconds += read_seconds;
            leaf_compute_thread_seconds += compute_seconds;
            if !recursive_history {
                let bucket_start = Instant::now();
                finalized_history.extend(std::mem::take(&mut summary.local_merge_events));
                bucket_seconds += bucket_start.elapsed().as_secs_f64();
            }
            summary.slab_id = 0;
            let fan_in_start = Instant::now();
            fan_in(
                HierarchicalSummary { z0, z1, summary },
                &mut slots,
                |left, right| {
                    combine_h2(
                        left,
                        right,
                        background_connectivity,
                        &mut finalized_history,
                        recursive_history,
                        &mut recursive_history_stats,
                    )
                },
                &mut combines,
                &mut max_pair_nodes,
                |summary| summary.interface_node_count as usize,
            )?;
            fan_in_seconds += fan_in_start.elapsed().as_secs_f64();
            max_live_summaries = max_live_summaries.max(slots.iter().flatten().count());
        }
    }

    let mut remaining: Vec<_> = slots.into_iter().flatten().collect();
    if remaining.is_empty() {
        bail!("hierarchical H2 branch tree received an empty volume");
    }
    remaining.sort_by_key(|summary| summary.z0);
    let mut root = remaining.remove(0);
    for right in remaining {
        max_pair_nodes = max_pair_nodes.max(
            root.summary.interface_node_count as usize
                + right.summary.interface_node_count as usize,
        );
        let root_combine_start = Instant::now();
        root = combine_h2(
            root,
            right,
            background_connectivity,
            &mut finalized_history,
            recursive_history,
            &mut recursive_history_stats,
        )?;
        root_combine_seconds += root_combine_start.elapsed().as_secs_f64();
        combines += 1;
    }

    if recursive_history {
        let bucket_start = Instant::now();
        finalized_history.extend(std::mem::take(&mut root.summary.local_merge_events));
        bucket_seconds += bucket_start.elapsed().as_secs_f64();
    } else {
        debug_assert!(root.summary.local_merge_events.is_empty());
    }
    let central_history_input_events = finalized_history.input_events;
    let finalized_input_events = central_history_input_events
        .saturating_add(leaf_attach_stats.local_history_contracted_zero_events as usize)
        .saturating_add(recursive_history_stats.contracted_zero_events as usize);
    let contracted_zero_events = finalized_history
        .contracted_zero_events
        .saturating_add(leaf_attach_stats.local_history_contracted_zero_events as usize)
        .saturating_add(recursive_history_stats.contracted_zero_events as usize);
    let repaired_history_parent_refs = finalized_history
        .repaired_parent_refs
        .saturating_add(leaf_attach_stats.local_history_repaired_parent_refs as usize)
        .saturating_add(recursive_history_stats.repaired_parent_refs as usize);
    let finalized_events = finalized_history.retained_events();
    debug_assert_eq!(
        finalized_events,
        finalized_input_events - contracted_zero_events
    );
    debug_assert_eq!(
        finalized_events,
        finalized_history
            .buckets
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    );
    let root_reduce_start = Instant::now();
    let raw_tree = reduce_h2_merge_tree_prebucketed(
        &root.summary,
        finalized_history.buckets,
        background_connectivity,
    )?;
    let root_reduce_seconds = root_reduce_start.elapsed().as_secs_f64();
    let canonical_start = Instant::now();
    let tree = canonicalize_h2_plateaus(raw_tree)?;
    let canonical_seconds = canonical_start.elapsed().as_secs_f64();
    println!(
        "PROFILE_BRANCH_H2_HIERARCHY leaf_slabs={} leaf_workers={} combines={} max_live_summaries={} max_pair_nodes={} final_interface_nodes={} leaf_attach_policy={} leaf_history_policy={} recursive_history_policy={} leaf_one_boundary_internal={} leaf_finalized_attach_early={} leaf_propagated_attach={} leaf_local_history_input_events={} leaf_local_history_retained_events={} leaf_local_history_contracted_zero_events={} leaf_local_history_repaired_parent_refs={} recursive_history_nodes={} recursive_history_input_events={} recursive_history_retained_events={} recursive_history_contracted_zero_events={} recursive_history_repaired_parent_refs={} central_history_input_events={} finalized_input_events={} finalized_events={} contracted_zero_events={} repaired_history_parent_refs={} compact_event_bytes={} full_event_bytes={} finalized_storage_bytes={} leaf_batch_seconds={:.6} leaf_read_thread_seconds={:.6} leaf_compute_thread_seconds={:.6} bucket_seconds={:.6} fan_in_seconds={:.6} root_combine_seconds={:.6} root_reduce_seconds={:.6} canonical_seconds={:.6} canonical_parenting=plateau",
        leaf_count,
        leaf_workers,
        combines,
        max_live_summaries,
        max_pair_nodes,
        slab_interface_node_count(volume.width * volume.height, volume.depth),
        if early_finalize_leaf_attaches {
            "early-finalize"
        } else {
            "reference"
        },
        if inline_leaf_history {
            "inline-contract"
        } else {
            "reference"
        },
        if recursive_history {
            "recursive-contract"
        } else {
            "reference"
        },
        leaf_attach_stats.one_boundary_internal,
        leaf_attach_stats.finalized_early,
        leaf_attach_stats.propagated_attach,
        leaf_attach_stats.local_history_input_events,
        leaf_attach_stats.local_history_retained_events,
        leaf_attach_stats.local_history_contracted_zero_events,
        leaf_attach_stats.local_history_repaired_parent_refs,
        recursive_history_stats.nodes,
        recursive_history_stats.input_events,
        recursive_history_stats.retained_events,
        recursive_history_stats.contracted_zero_events,
        recursive_history_stats.repaired_parent_refs,
        central_history_input_events,
        finalized_input_events,
        finalized_events,
        contracted_zero_events,
        repaired_history_parent_refs,
        std::mem::size_of::<CompactH2BranchMerge>(),
        std::mem::size_of::<H2BranchMerge>(),
        finalized_events.saturating_mul(std::mem::size_of::<CompactH2BranchMerge>()),
        leaf_batch_seconds,
        leaf_read_thread_seconds,
        leaf_compute_thread_seconds,
        bucket_seconds,
        fan_in_seconds,
        root_combine_seconds,
        root_reduce_seconds,
        canonical_seconds,
    );
    println!(
        "Hierarchical H2 branch-tree computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(tree)
}

#[cfg(test)]
mod finalized_history_contraction_tests {
    use super::*;

    #[test]
    fn h0_positive_death_redirects_only_strictly_later_references() {
        let mut history = H0FinalizedHistory::new();
        let parent = H0Branch { id: 50, birth: 10 };
        history.push_merge(H0BranchMerge {
            value: 12,
            child: H0Branch { id: 11, birth: 11 },
            parent,
        });
        history.push_merge(H0BranchMerge {
            value: 20,
            child: H0Branch { id: 10, birth: 18 },
            parent,
        });
        history.push_merge(H0BranchMerge {
            value: 15,
            child: parent,
            parent: H0Branch { id: 60, birth: 5 },
        });

        assert_eq!(history.buckets[12][0].parent_id, 50);
        assert_eq!(history.buckets[20][0].parent_id, 60);
        assert_eq!(history.buckets[15][0].child_id, 50);
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(history.contracted_zero_events, 0);
    }

    #[test]
    fn h0_zero_death_contracts_later_references_and_is_not_stored() {
        let mut history = H0FinalizedHistory::new();
        let zero = H0Branch { id: 70, birth: 15 };
        history.push_merge(H0BranchMerge {
            value: 19,
            child: H0Branch { id: 13, birth: 18 },
            parent: zero,
        });
        history.push_merge(H0BranchMerge {
            value: 15,
            child: zero,
            parent: H0Branch { id: 80, birth: 7 },
        });

        assert_eq!(history.buckets[19][0].parent_id, 80);
        assert!(history.buckets[15].iter().all(|event| event.child_id != 70));
        assert_eq!(history.contracted_zero_events, 1);
        assert_eq!(history.retained_events(), 1);
    }

    #[test]
    fn h2_positive_death_redirects_only_strictly_lower_references() {
        let mut history = H2FinalizedHistory::new();
        let parent = H2Branch::Finite { id: 50, birth: 20 };
        history.push_merge(H2BranchMerge {
            value: 18,
            child_id: 11,
            child_birth: 19,
            parent,
        });
        history.push_merge(H2BranchMerge {
            value: 10,
            child_id: 10,
            child_birth: 17,
            parent,
        });
        history.push_merge(H2BranchMerge {
            value: 15,
            child_id: 50,
            child_birth: 20,
            parent: H2Branch::Finite { id: 60, birth: 30 },
        });

        assert_eq!(history.buckets[18][0].parent_id, 50);
        assert_eq!(history.buckets[10][0].parent_id, 60);
        assert_eq!(history.buckets[15][0].child_id, 50);
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(history.contracted_zero_events, 0);
    }

    #[test]
    fn h2_zero_death_contracts_lower_references_and_is_not_stored() {
        let mut history = H2FinalizedHistory::new();
        let zero = H2Branch::Finite { id: 70, birth: 15 };
        history.push_merge(H2BranchMerge {
            value: 9,
            child_id: 13,
            child_birth: 14,
            parent: zero,
        });
        history.push_merge(H2BranchMerge {
            value: 15,
            child_id: 70,
            child_birth: 15,
            parent: H2Branch::Finite { id: 80, birth: 30 },
        });

        assert_eq!(history.buckets[9][0].parent_id, 80);
        assert!(history.buckets[15].iter().all(|event| event.child_id != 70));
        assert_eq!(history.contracted_zero_events, 1);
        assert_eq!(history.retained_events(), 1);
    }
}

#[cfg(test)]
mod boundary_transition_action_tests {
    use super::*;

    #[test]
    fn h0_older_attached_branch_propagates_without_premature_death() {
        let boundary = H0Branch { id: 1, birth: 10 };
        let incoming = H0Branch { id: 2, birth: 5 };
        let mut uf = H0PairUnionFind::new(vec![boundary], vec![0]);

        match uf.attach(0, incoming, 12) {
            H0Action::Attach(attach) => {
                assert_eq!(attach.value, 12);
                assert_eq!(attach.interface_node, 0);
                assert_eq!(attach.branch, incoming);
            }
            other => panic!("expected H0 propagated Attach, got {other:?}"),
        }
    }

    #[test]
    fn h0_older_internal_union_propagates_without_premature_death() {
        let boundary = H0Branch { id: 1, birth: 10 };
        let internal = H0Branch { id: 2, birth: 5 };
        let mut uf = H0PairUnionFind::new(vec![boundary, internal], vec![0, NO_INTERFACE_REP]);

        match uf.union(0, 1, 12) {
            H0Action::Attach(attach) => {
                assert_eq!(attach.value, 12);
                assert_eq!(attach.interface_node, 0);
                assert_eq!(attach.branch, internal);
            }
            other => panic!("expected H0 propagated Attach, got {other:?}"),
        }
    }

    #[test]
    fn h0_propagated_elder_change_finalizes_once_when_terminal_is_lost() {
        let boundary = H0Branch { id: 1, birth: 10 };
        let incoming = H0Branch { id: 2, birth: 5 };

        for _ in 0..3 {
            let mut boundary_visible = H0PairUnionFind::new(vec![boundary], vec![0]);
            assert!(matches!(
                boundary_visible.attach(0, incoming, 12),
                H0Action::Attach(_)
            ));
        }

        let mut internal = H0PairUnionFind::new(vec![boundary], vec![NO_INTERFACE_REP]);
        match internal.attach(0, incoming, 12) {
            H0Action::Final(merge) => {
                assert_eq!(merge.value, 12);
                assert_eq!(merge.child, boundary);
                assert_eq!(merge.parent, incoming);
            }
            other => panic!("expected H0 Final after terminal loss, got {other:?}"),
        }
    }

    #[test]
    fn h2_older_attached_branch_propagates_without_premature_death() {
        let boundary = H2Branch::Finite { id: 1, birth: 5 };
        let incoming = H2Branch::Finite { id: 2, birth: 10 };
        let mut uf = H2PairUnionFind::new(vec![boundary], vec![0]);

        match uf.attach(0, incoming, 4) {
            H2Action::Attach(attach) => {
                assert_eq!(attach.value, 4);
                assert_eq!(attach.interface_node, 0);
                assert_eq!(attach.branch, incoming);
            }
            other => panic!("expected H2 propagated Attach, got {other:?}"),
        }
    }

    #[test]
    fn h2_propagated_elder_change_finalizes_once_when_terminal_is_lost() {
        let boundary = H2Branch::Finite { id: 1, birth: 5 };
        let incoming = H2Branch::Finite { id: 2, birth: 10 };

        for _ in 0..3 {
            let mut boundary_visible = H2PairUnionFind::new(vec![boundary], vec![0]);
            assert!(matches!(
                boundary_visible.attach(0, incoming, 4),
                H2Action::Attach(_)
            ));
        }

        let mut internal = H2PairUnionFind::new(vec![boundary], vec![NO_INTERFACE_REP]);
        match internal.attach(0, incoming, 4) {
            H2Action::Final(merge) => {
                assert_eq!(merge.value, 4);
                assert_eq!(merge.child_id, 1);
                assert_eq!(merge.parent, incoming);
            }
            other => panic!("expected H2 Final after terminal loss, got {other:?}"),
        }
    }

    #[test]
    fn h2_outside_attach_propagates_without_premature_death() {
        let boundary = H2Branch::Finite { id: 1, birth: 5 };
        let mut uf = H2PairUnionFind::new(vec![boundary], vec![0]);

        match uf.attach(0, H2Branch::Outside, 4) {
            H2Action::Attach(attach) => {
                assert_eq!(attach.value, 4);
                assert_eq!(attach.interface_node, 0);
                assert_eq!(attach.branch, H2Branch::Outside);
            }
            other => panic!("expected H2 propagated Attach, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod ordered_fan_in_tests {
    use super::*;
    use std::cmp::Ordering;

    fn h0_priority(event: H0Event) -> u8 {
        match event.kind {
            H0EventKind::Attach { .. } => 0,
            H0EventKind::Interface { .. } => 1,
            H0EventKind::Cross { .. } => 2,
        }
    }

    fn h2_priority(event: H2Event) -> u8 {
        match event.kind {
            H2EventKind::Outside { .. } => 0,
            H2EventKind::Attach { .. } => 1,
            H2EventKind::Interface { .. } => 2,
            H2EventKind::Cross { .. } => 3,
        }
    }

    fn reference_h0_order(
        left_attach: &[H0AttachEvent],
        right_attach: &[H0AttachEvent],
        left_interface: &[H0InterfaceMergeEvent],
        right_interface: &[H0InterfaceMergeEvent],
        cross: &[H0InterfaceMergeEvent],
        left_offset: u32,
        right_offset: u32,
    ) -> Vec<H0Event> {
        let mut events = Vec::new();
        for event in left_attach {
            events.push(H0Event {
                value: event.value,
                kind: H0EventKind::Attach {
                    node: left_offset + event.interface_node,
                    branch: event.branch,
                },
            });
        }
        for event in right_attach {
            events.push(H0Event {
                value: event.value,
                kind: H0EventKind::Attach {
                    node: right_offset + event.interface_node,
                    branch: event.branch,
                },
            });
        }
        for event in left_interface {
            events.push(H0Event {
                value: event.value,
                kind: H0EventKind::Interface {
                    a: left_offset + event.a,
                    b: left_offset + event.b,
                },
            });
        }
        for event in right_interface {
            events.push(H0Event {
                value: event.value,
                kind: H0EventKind::Interface {
                    a: right_offset + event.a,
                    b: right_offset + event.b,
                },
            });
        }
        for event in cross {
            events.push(H0Event {
                value: event.value,
                kind: H0EventKind::Cross {
                    a: event.a,
                    b: event.b,
                },
            });
        }
        events.sort_by(|a, b| {
            let value = a.value.cmp(&b.value);
            if value == Ordering::Equal {
                h0_priority(*a).cmp(&h0_priority(*b))
            } else {
                value
            }
        });
        events
    }

    fn reference_h2_order(
        left_attach: &[H2AttachEvent],
        right_attach: &[H2AttachEvent],
        left_interface: &[H2InterfaceMergeEvent],
        right_interface: &[H2InterfaceMergeEvent],
        cross: &[H2InterfaceMergeEvent],
        left_offset: u32,
        right_offset: u32,
    ) -> Vec<H2Event> {
        let mut events = Vec::new();
        for (offset, attach, interface) in [
            (left_offset, left_attach, left_interface),
            (right_offset, right_attach, right_interface),
        ] {
            for event in attach {
                let kind = if event.branch == H2Branch::Outside {
                    H2EventKind::Outside {
                        node: offset + event.interface_node,
                    }
                } else {
                    H2EventKind::Attach {
                        node: offset + event.interface_node,
                        branch: event.branch,
                    }
                };
                events.push(H2Event {
                    value: event.value,
                    kind,
                });
            }
            for event in interface {
                events.push(H2Event {
                    value: event.value,
                    kind: H2EventKind::Interface {
                        a: offset + event.a,
                        b: offset + event.b,
                    },
                });
            }
        }
        for event in cross {
            events.push(H2Event {
                value: event.value,
                kind: H2EventKind::Cross {
                    a: event.a,
                    b: event.b,
                },
            });
        }
        events.sort_by(|a, b| {
            let value = b.value.cmp(&a.value);
            if value == Ordering::Equal {
                h2_priority(*a).cmp(&h2_priority(*b))
            } else {
                value
            }
        });
        events
    }

    #[test]
    fn h0_linear_merge_matches_stable_sort_with_ties() {
        let b1 = H0Branch { id: 11, birth: 1 };
        let b2 = H0Branch { id: 12, birth: 2 };
        let left_attach = vec![
            H0AttachEvent {
                value: 1,
                interface_node: 0,
                branch: b1,
            },
            H0AttachEvent {
                value: 3,
                interface_node: 1,
                branch: b2,
            },
            H0AttachEvent {
                value: 3,
                interface_node: 2,
                branch: b1,
            },
        ];
        let right_attach = vec![
            H0AttachEvent {
                value: 2,
                interface_node: 0,
                branch: b2,
            },
            H0AttachEvent {
                value: 3,
                interface_node: 1,
                branch: b1,
            },
        ];
        let left_interface = vec![
            H0InterfaceMergeEvent {
                value: 1,
                a: 0,
                b: 1,
            },
            H0InterfaceMergeEvent {
                value: 3,
                a: 1,
                b: 2,
            },
        ];
        let right_interface = vec![
            H0InterfaceMergeEvent {
                value: 3,
                a: 0,
                b: 1,
            },
            H0InterfaceMergeEvent {
                value: 4,
                a: 1,
                b: 2,
            },
        ];
        let cross = vec![
            H0InterfaceMergeEvent {
                value: 1,
                a: 100,
                b: 200,
            },
            H0InterfaceMergeEvent {
                value: 3,
                a: 101,
                b: 201,
            },
            H0InterfaceMergeEvent {
                value: 4,
                a: 102,
                b: 202,
            },
        ];

        let expected = reference_h0_order(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
            10,
            20,
        );
        let actual = merge_h0_event_streams(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
            10,
            20,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn h2_linear_merge_matches_stable_sort_with_mixed_attach_priorities() {
        let f1 = H2Branch::Finite { id: 21, birth: 9 };
        let f2 = H2Branch::Finite { id: 22, birth: 8 };
        let left_attach = vec![
            H2AttachEvent {
                value: 9,
                interface_node: 0,
                branch: H2Branch::Outside,
            },
            H2AttachEvent {
                value: 9,
                interface_node: 1,
                branch: f1,
            },
            H2AttachEvent {
                value: 8,
                interface_node: 2,
                branch: f2,
            },
            H2AttachEvent {
                value: 8,
                interface_node: 3,
                branch: H2Branch::Outside,
            },
        ];
        let right_attach = vec![
            H2AttachEvent {
                value: 9,
                interface_node: 0,
                branch: f2,
            },
            H2AttachEvent {
                value: 9,
                interface_node: 1,
                branch: H2Branch::Outside,
            },
            H2AttachEvent {
                value: 7,
                interface_node: 2,
                branch: H2Branch::Outside,
            },
        ];
        let left_interface = vec![
            H2InterfaceMergeEvent {
                value: 9,
                a: 0,
                b: 1,
            },
            H2InterfaceMergeEvent {
                value: 8,
                a: 1,
                b: 2,
            },
        ];
        let right_interface = vec![
            H2InterfaceMergeEvent {
                value: 8,
                a: 0,
                b: 1,
            },
            H2InterfaceMergeEvent {
                value: 7,
                a: 1,
                b: 2,
            },
        ];
        let cross = vec![
            H2InterfaceMergeEvent {
                value: 9,
                a: 100,
                b: 200,
            },
            H2InterfaceMergeEvent {
                value: 8,
                a: 101,
                b: 201,
            },
            H2InterfaceMergeEvent {
                value: 7,
                a: 102,
                b: 202,
            },
        ];

        let expected = reference_h2_order(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
            10,
            20,
        );
        let actual = merge_h2_event_streams(
            &left_attach,
            &right_attach,
            &left_interface,
            &right_interface,
            &cross,
            10,
            20,
        );
        assert_eq!(actual, expected);
    }
}
