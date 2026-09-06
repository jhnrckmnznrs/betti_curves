use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::merge_tree_common::{
    CompactH2BranchMerge, H2Branch, H2BranchMerge, H2PackedTreeRecorder, H2TreeRecorder, MergeTree,
    OUTSIDE_BRANCH_ID, PackedDeferredIdResolver,
};
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};

pub(crate) const NUM_U16_VALUES: usize = 65_536;

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
enum LocalAction {
    None,
    LocalMerge(H2BranchMerge),
    Attach(AttachEvent),
    InterfaceMerge(InterfaceMergeEvent),
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct H2LeafAttachStats {
    pub(crate) one_boundary_internal: u64,
    pub(crate) finalized_early: u64,
    pub(crate) propagated_attach: u64,
    pub(crate) local_history_input_events: u64,
    pub(crate) local_history_retained_events: u64,
    pub(crate) local_history_contracted_zero_events: u64,
    pub(crate) local_history_repaired_parent_refs: u64,
}

/// Leaf-local counterpart of the hierarchical finalized-history contractor.
/// See the H0 implementation for the structural argument. H2 uses the
/// reversed filtration inequality and never watches the Outside root as a
/// finite provisional parent.
#[derive(Debug)]
struct LocalMergeHistory {
    events: Vec<H2BranchMerge>,
    by_parent: HashMap<u64, Vec<usize>>,
    inline_contract: bool,
    input_events: u64,
    contracted_zero_events: u64,
    repaired_parent_refs: u64,
}

impl LocalMergeHistory {
    fn new(inline_contract: bool) -> Self {
        Self {
            events: Vec::new(),
            by_parent: HashMap::new(),
            inline_contract,
            input_events: 0,
            contracted_zero_events: 0,
            repaired_parent_refs: 0,
        }
    }

    fn push(&mut self, merge: H2BranchMerge) {
        self.input_events += 1;
        if !self.inline_contract {
            self.events.push(merge);
            return;
        }

        let zero = merge.child_birth == merge.value;
        let (parent_id, parent_birth) = match merge.parent {
            H2Branch::Outside => (OUTSIDE_BRANCH_ID, 0),
            H2Branch::Finite { id, birth } => (id, birth),
        };
        self.redirect_existing(merge.child_id, parent_id, parent_birth, merge.value, zero);
        if zero {
            self.contracted_zero_events += 1;
            return;
        }

        let index = self.events.len();
        self.events.push(merge);
        if parent_id != OUTSIDE_BRANCH_ID {
            self.by_parent.entry(parent_id).or_default().push(index);
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
        let Some(indices) = self.by_parent.remove(&child_id) else {
            return;
        };
        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for index in indices {
            let event = &mut self.events[index];
            if zero || event.value < death {
                let current_parent_id = match event.parent {
                    H2Branch::Outside => OUTSIDE_BRANCH_ID,
                    H2Branch::Finite { id, .. } => id,
                };
                debug_assert_eq!(current_parent_id, child_id);
                event.parent = if parent_id == OUTSIDE_BRANCH_ID {
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
            self.by_parent.insert(child_id, keep);
        }
        if parent_id != OUTSIDE_BRANCH_ID && !moved.is_empty() {
            self.by_parent.entry(parent_id).or_default().extend(moved);
        }
    }

    fn retained_events(&self) -> u64 {
        self.events.len() as u64
    }

    fn into_events(self) -> Vec<H2BranchMerge> {
        self.events
    }
}

#[derive(Debug)]
struct LocalUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    branch: Vec<H2Branch>,
    interface_rep: Vec<u32>,
}

impl LocalUnionFind {
    fn new(block: &Block) -> Self {
        let voxel_count = block.voxel_count();
        assert!(
            voxel_count <= u32::MAX as usize,
            "LocalUnionFind uses u32 indices; slab has too many voxels"
        );

        let slice_size = block.shape[0] * block.shape[1];
        let global_base = (block.z0 as u64)
            .checked_mul(slice_size as u64)
            .expect("global voxel ID overflow");
        let branch = block
            .values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, birth)| H2Branch::Finite {
                id: global_base + idx as u64,
                birth,
            })
            .collect();

        Self {
            parent: (0..voxel_count as u32).collect(),
            rank: vec![0u8; voxel_count],
            branch,
            interface_rep: vec![NO_INTERFACE_REP; voxel_count],
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

    fn set_interface_rep(&mut self, x: u32, interface_node: u32) {
        let root = self.find(x);
        self.interface_rep[root as usize] = interface_node;
    }

    fn mark_outside(&mut self, x: u32) {
        let root = self.find(x);
        self.branch[root as usize] = H2Branch::Outside;
    }

    fn union(
        &mut self,
        a: u32,
        b: u32,
        value: u16,
        finalize_nonpromoting_attach: bool,
        attach_stats: &mut H2LeafAttachStats,
    ) -> Option<LocalAction> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return None;
        }

        let branch_a = self.branch[root_a as usize];
        let branch_b = self.branch[root_b as usize];
        let older = branch_a.older(branch_b);
        let rep_a = self.interface_rep[root_a as usize];
        let rep_b = self.interface_rep[root_b as usize];

        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => match branch_a.younger(branch_b) {
                Some((child_id, child_birth)) => LocalAction::LocalMerge(H2BranchMerge {
                    value,
                    child_id,
                    child_birth,
                    parent: older,
                }),
                None => LocalAction::None,
            },
            (true, false) => {
                attach_stats.one_boundary_internal += 1;
                if finalize_nonpromoting_attach && older == branch_a {
                    attach_stats.finalized_early += 1;
                    match branch_a.younger(branch_b) {
                        Some((child_id, child_birth)) => LocalAction::LocalMerge(H2BranchMerge {
                            value,
                            child_id,
                            child_birth,
                            parent: branch_a,
                        }),
                        None => LocalAction::None,
                    }
                } else {
                    attach_stats.propagated_attach += 1;
                    LocalAction::Attach(AttachEvent {
                        value,
                        interface_node: rep_a,
                        branch: branch_b,
                    })
                }
            }
            (false, true) => {
                attach_stats.one_boundary_internal += 1;
                if finalize_nonpromoting_attach && older == branch_b {
                    attach_stats.finalized_early += 1;
                    match branch_a.younger(branch_b) {
                        Some((child_id, child_birth)) => LocalAction::LocalMerge(H2BranchMerge {
                            value,
                            child_id,
                            child_birth,
                            parent: branch_b,
                        }),
                        None => LocalAction::None,
                    }
                } else {
                    attach_stats.propagated_attach += 1;
                    LocalAction::Attach(AttachEvent {
                        value,
                        interface_node: rep_b,
                        branch: branch_a,
                    })
                }
            }
            (true, true) => LocalAction::InterfaceMerge(InterfaceMergeEvent {
                value,
                a: rep_a,
                b: rep_b,
            }),
        };

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
        self.interface_rep[root_a as usize] = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };

        Some(action)
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

fn union_active_neighbor(
    uf: &mut LocalUnionFind,
    active: &[u8],
    current: u32,
    neighbor: usize,
    value: u16,
    buffers: &mut LocalBuffers<'_>,
) {
    if active[neighbor] == 0 {
        return;
    }
    if let Some(action) = uf.union(
        current,
        neighbor as u32,
        value,
        buffers.finalize_nonpromoting_attach,
        &mut *buffers.attach_stats,
    ) {
        handle_action(action, buffers);
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
    let voxel_count = block.voxel_count();
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalUnionFind::new(block);
    let mut active = vec![0u8; voxel_count];
    let buckets = build_voxel_buckets_u16(&block.values);
    let mut local_pruner = NeighborhoodComponentPruner::new(background_connectivity);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut local_merge_history = LocalMergeHistory::new(inline_contract_local_history);
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();
    let mut attach_stats = H2LeafAttachStats::default();

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
            active[idx] = 1;

            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
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

            let mut buffers = LocalBuffers {
                local_merge_history: &mut local_merge_history,
                attach_events: &mut attach_events,
                interface_merge_events: &mut interface_merge_events,
                finalize_nonpromoting_attach,
                attach_stats: &mut attach_stats,
            };

            local_pruner.for_each_representative_neighbor(
                &active,
                x,
                y,
                z,
                width,
                height,
                depth,
                |neighbor| {
                    union_active_neighbor(
                        &mut uf,
                        &active,
                        idx_u32,
                        neighbor,
                        value_u16,
                        &mut buffers,
                    );
                },
            );
        }
    }

    attach_stats.local_history_input_events = local_merge_history.input_events;
    attach_stats.local_history_retained_events = local_merge_history.retained_events();
    attach_stats.local_history_contracted_zero_events = local_merge_history.contracted_zero_events;
    attach_stats.local_history_repaired_parent_refs = local_merge_history.repaired_parent_refs;
    let local_merge_events = local_merge_history.into_events();

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

    fn two_node_uf(boundary: H2Branch, internal: H2Branch) -> LocalUnionFind {
        LocalUnionFind {
            parent: vec![0, 1],
            rank: vec![0, 0],
            branch: vec![boundary, internal],
            interface_rep: vec![0, NO_INTERFACE_REP],
        }
    }

    #[test]
    fn hierarchical_h2_finalizes_nonpromoting_internal_branch_early() {
        let boundary = H2Branch::Finite { id: 1, birth: 12 };
        let internal = H2Branch::Finite { id: 2, birth: 9 };
        let mut uf = two_node_uf(boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let action = uf.union(0, 1, 5, true, &mut stats).unwrap();
        match action {
            LocalAction::LocalMerge(merge) => {
                assert_eq!(merge.value, 5);
                assert_eq!(merge.child_id, 2);
                assert_eq!(merge.child_birth, 9);
                assert_eq!(merge.parent, boundary);
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
        let mut uf = two_node_uf(boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let action = uf.union(0, 1, 5, true, &mut stats).unwrap();
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
        let mut uf = two_node_uf(boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        let action = uf.union(0, 1, 5, true, &mut stats).unwrap();
        match action {
            LocalAction::LocalMerge(merge) => {
                assert_eq!(merge.child_id, 2);
                assert_eq!(merge.parent, H2Branch::Outside);
            }
            other => panic!("expected finite-to-outside finalization, got {other:?}"),
        }
    }

    #[test]
    fn h2_reference_leaf_policy_preserves_original_attach() {
        let boundary = H2Branch::Finite { id: 1, birth: 12 };
        let internal = H2Branch::Finite { id: 2, birth: 9 };
        let mut uf = two_node_uf(boundary, internal);
        let mut stats = H2LeafAttachStats::default();

        assert!(matches!(
            uf.union(0, 1, 5, false, &mut stats),
            Some(LocalAction::Attach(_))
        ));
        assert_eq!(stats.finalized_early, 0);
        assert_eq!(stats.propagated_attach, 1);
    }

    #[test]
    fn h2_inline_leaf_history_contracts_diagonal_parent() {
        let mut history = LocalMergeHistory::new(true);
        history.push(H2BranchMerge {
            value: 10,
            child_id: 10,
            child_birth: 12,
            parent: H2Branch::Finite { id: 50, birth: 15 },
        });
        history.push(H2BranchMerge {
            value: 15,
            child_id: 50,
            child_birth: 15,
            parent: H2Branch::Finite { id: 60, birth: 20 },
        });

        assert_eq!(history.input_events, 2);
        assert_eq!(history.contracted_zero_events, 1);
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(history.retained_events(), 1);
        assert_eq!(
            history.events[0].parent,
            H2Branch::Finite { id: 60, birth: 20 }
        );
    }

    #[test]
    fn h2_inline_leaf_history_keeps_equal_threshold_positive_parent() {
        let mut history = LocalMergeHistory::new(true);
        history.push(H2BranchMerge {
            value: 15,
            child_id: 10,
            child_birth: 16,
            parent: H2Branch::Finite { id: 50, birth: 20 },
        });
        history.push(H2BranchMerge {
            value: 15,
            child_id: 50,
            child_birth: 20,
            parent: H2Branch::Finite { id: 60, birth: 25 },
        });
        assert_eq!(history.repaired_parent_refs, 0);
        assert_eq!(
            history.events[0].parent,
            H2Branch::Finite { id: 50, birth: 20 }
        );
        assert_eq!(history.retained_events(), 2);
    }

    #[test]
    fn h2_inline_leaf_history_repairs_to_outside() {
        let mut history = LocalMergeHistory::new(true);
        history.push(H2BranchMerge {
            value: 10,
            child_id: 10,
            child_birth: 12,
            parent: H2Branch::Finite { id: 50, birth: 15 },
        });
        history.push(H2BranchMerge {
            value: 15,
            child_id: 50,
            child_birth: 15,
            parent: H2Branch::Outside,
        });
        assert_eq!(history.repaired_parent_refs, 1);
        assert_eq!(history.events[0].parent, H2Branch::Outside);
    }
}
