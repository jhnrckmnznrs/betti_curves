use anyhow::{Result, bail};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};

pub(crate) const NUM_U16_VALUES: usize = 65_536;

/// A foreground H2 persistence interval [birth, death).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H2Interval {
    pub birth: u16,
    pub death: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FinitePair {
    /// Foreground H2 birth threshold.
    pub(crate) birth: u16,
    /// Foreground H2 death threshold.
    pub(crate) death: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttachEvent {
    pub(crate) value: u16,
    pub(crate) interface_node: u32,
    pub(crate) branch_birth: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OutsideEvent {
    pub(crate) value: u16,
    pub(crate) interface_node: u32,
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
}

#[derive(Debug)]
pub(crate) struct SlabH2Summary {
    pub(crate) slab_id: usize,
    pub(crate) finalized_pairs: Vec<FinitePair>,
    pub(crate) attach_events: Vec<AttachEvent>,
    pub(crate) outside_events: Vec<OutsideEvent>,
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

/// Birth metadata in the original-image superlevel coordinates.
///
/// `Outside` is older than every finite component. This is equivalent to
/// assigning the outside component birth -infinity in the sublevel filtration
/// of the negated padded image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundBirth {
    Outside,
    Finite(u16),
}

fn older_background_birth(a: BackgroundBirth, b: BackgroundBirth) -> BackgroundBirth {
    match (a, b) {
        (BackgroundBirth::Outside, _) | (_, BackgroundBirth::Outside) => BackgroundBirth::Outside,
        (BackgroundBirth::Finite(a_value), BackgroundBirth::Finite(b_value)) => {
            BackgroundBirth::Finite(a_value.max(b_value))
        }
    }
}

/// Returns the finite component killed by a successful background merge.
///
/// In a descending superlevel sweep, larger finite birth values are older.
/// The corresponding foreground H2 interval is [merge_value, younger_birth).
fn pair_from_background_merge(
    a: BackgroundBirth,
    b: BackgroundBirth,
    merge_value: u16,
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
enum LocalPersistenceAction {
    None,
    FinalPair(FinitePair),
    Attach(AttachEvent),
    Outside(OutsideEvent),
    InterfaceMerge(InterfaceMergeEvent),
}

#[derive(Debug)]
struct LocalBackgroundPersistenceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<BackgroundBirth>,
    interface_rep: Vec<u32>,
}

impl LocalBackgroundPersistenceUnionFind {
    fn new(values: &[u16]) -> Self {
        assert!(
            values.len() <= u32::MAX as usize,
            "LocalBackgroundPersistenceUnionFind uses u32 indices; slab has too many voxels"
        );

        Self {
            parent: (0..values.len() as u32).collect(),
            rank: vec![0u8; values.len()],
            birth: values
                .iter()
                .copied()
                .map(BackgroundBirth::Finite)
                .collect(),
            interface_rep: vec![NO_INTERFACE_REP; values.len()],
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
        self.birth[root as usize] = BackgroundBirth::Outside;
    }

    fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        merge_value: u16,
    ) -> Option<LocalPersistenceAction> {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);

        if root_a == root_b {
            return None;
        }

        let birth_a = self.birth[root_a as usize];
        let birth_b = self.birth[root_b as usize];
        let rep_a = self.interface_rep[root_a as usize];
        let rep_b = self.interface_rep[root_b as usize];

        let action = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (false, false) => pair_from_background_merge(birth_a, birth_b, merge_value)
                .map(LocalPersistenceAction::FinalPair)
                .unwrap_or(LocalPersistenceAction::None),

            (true, false) => match birth_b {
                BackgroundBirth::Finite(branch_birth) => {
                    LocalPersistenceAction::Attach(AttachEvent {
                        value: merge_value,
                        interface_node: rep_a,
                        branch_birth,
                    })
                }
                BackgroundBirth::Outside => {
                    if birth_a == BackgroundBirth::Outside {
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
                BackgroundBirth::Finite(branch_birth) => {
                    LocalPersistenceAction::Attach(AttachEvent {
                        value: merge_value,
                        interface_node: rep_b,
                        branch_birth,
                    })
                }
                BackgroundBirth::Outside => {
                    if birth_b == BackgroundBirth::Outside {
                        LocalPersistenceAction::None
                    } else {
                        LocalPersistenceAction::Outside(OutsideEvent {
                            value: merge_value,
                            interface_node: rep_b,
                        })
                    }
                }
            },

            (true, true) => LocalPersistenceAction::InterfaceMerge(InterfaceMergeEvent {
                value: merge_value,
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

        self.birth[root_a as usize] = older_background_birth(birth_a, birth_b);
        self.interface_rep[root_a as usize] = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };

        Some(action)
    }
}

struct LocalPersistenceBuffers<'a> {
    finalized_pairs: &'a mut Vec<FinitePair>,
    attach_events: &'a mut Vec<AttachEvent>,
    outside_events: &'a mut Vec<OutsideEvent>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent>,
}

fn handle_local_action(action: LocalPersistenceAction, buffers: &mut LocalPersistenceBuffers<'_>) {
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

fn union_active_neighbor_h2(
    uf: &mut LocalBackgroundPersistenceUnionFind,
    active: &[u8],
    current: u32,
    neighbor: usize,
    value: u16,
    buffers: &mut LocalPersistenceBuffers<'_>,
) {
    if active[neighbor] == 0 {
        return;
    }

    if let Some(action) = uf.union_with_persistence(current, neighbor as u32, value) {
        handle_local_action(action, buffers);
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

pub(crate) fn process_slab_h2_persistence(
    slab_id: usize,
    block: &Block,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
) -> SlabH2Summary {
    let voxel_count = block.voxel_count();
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalBackgroundPersistenceUnionFind::new(&block.values);
    let mut active = vec![0u8; voxel_count];
    let buckets = build_voxel_buckets_u16(&block.values);
    let mut local_pruner = NeighborhoodComponentPruner::new(background_connectivity);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut finalized_pairs = Vec::new();
    let mut attach_events = Vec::new();
    let mut outside_events = Vec::new();
    let mut interface_merge_events = Vec::new();

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
                    outside_events.push(OutsideEvent {
                        value: value_u16,
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

            local_pruner.for_each_representative_neighbor(
                &active,
                x,
                y,
                z,
                width,
                height,
                depth,
                |neighbor| {
                    union_active_neighbor_h2(
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

    let z_min_face = extract_boundary_face_nodes(block, 0);
    let z_max_face = extract_boundary_face_nodes(block, depth - 1);

    SlabH2Summary {
        slab_id,
        finalized_pairs,
        attach_events,
        outside_events,
        interface_merge_events,
        interface_node_count,
        z_min_face,
        z_max_face,
    }
}

fn extract_boundary_face_nodes(block: &Block, z_local: usize) -> BoundaryFaceNodes {
    let width = block.shape[0];
    let height = block.shape[1];
    let slice_size = width * height;
    let mut node_ids = vec![0u32; slice_size];
    let mut values = vec![0u16; slice_size];

    for y in 0..height {
        for x in 0..width {
            let face_idx = y * width + x;
            let idx = z_local * slice_size + face_idx;
            node_ids[face_idx] = face_node_id(z_local, block.shape[2], face_idx, slice_size);
            values[face_idx] = block.values[idx];
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
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<SlabH2Summary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);
    let [global_width, global_height, global_depth] = volume.shape();

    println!(
        "Processing {} H2-persistence slab summaries in parallel...",
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
    branch_birth: u16,
}

#[derive(Debug, Clone, Copy)]
struct GlobalPairEvent {
    value: u16,
    a: u32,
    b: u32,
}

/// Global background union-find with an explicit outside component.
///
/// The outside component is older than every finite component, matching birth
/// -infinity in the sublevel filtration of the negated padded image.
#[derive(Debug)]
pub(crate) struct GlobalBackgroundPersistenceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<BackgroundBirth>,
    outside_node: u32,
}

impl GlobalBackgroundPersistenceUnionFind {
    pub(crate) fn new(interface_births: Vec<u16>) -> Self {
        assert!(
            interface_births.len() < u32::MAX as usize,
            "too many interface nodes for an additional outside node"
        );

        let outside_node = interface_births.len() as u32;
        let mut birth: Vec<BackgroundBirth> = interface_births
            .into_iter()
            .map(BackgroundBirth::Finite)
            .collect();
        birth.push(BackgroundBirth::Outside);

        Self {
            parent: (0..birth.len() as u32).collect(),
            rank: vec![0u8; birth.len()],
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
        merge_value: u16,
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
        branch_birth: u16,
        merge_value: u16,
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
        merge_value: u16,
    ) -> Option<FinitePair> {
        self.union_with_persistence(interface_node, self.outside_node, merge_value)
    }

    pub(crate) fn remaining_finite_root_births(&mut self) -> Vec<u16> {
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

fn build_global_interface_births(summaries: &[SlabH2Summary], offsets: &[u32]) -> Vec<u16> {
    let total = *offsets.last().unwrap_or(&0) as usize;
    let mut births = vec![0u16; total];

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

fn generate_cross_slab_events(
    summaries: &[SlabH2Summary],
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
                        let value = left_value.min(right_face.values[right_idx]);
                        events.push(GlobalPairEvent { value, a, b });
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
                                let value = left_value.min(right_face.values[right_idx]);
                                events.push(GlobalPairEvent { value, a, b });
                            }
                        }
                    }
                }
            }
        }
    }

    events
}

fn push_finite_interval(intervals: &mut Vec<H2Interval>, pair: FinitePair) {
    if pair.birth < pair.death {
        intervals.push(H2Interval {
            birth: pair.birth,
            death: pair.death,
        });
    }
}

fn reduce_h2_persistence(
    summaries: &[SlabH2Summary],
    connectivity: Connectivity,
) -> Result<Vec<H2Interval>> {
    let offsets = build_interface_offsets(summaries);
    let births = build_global_interface_births(summaries, &offsets);
    let mut global_uf = GlobalBackgroundPersistenceUnionFind::new(births);

    let mut attach_by_value: Vec<Vec<GlobalAttachEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut outside_by_value: Vec<Vec<u32>> = (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut interface_by_value: Vec<Vec<GlobalPairEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut cross_by_value: Vec<Vec<GlobalPairEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();

    let mut intervals = Vec::new();

    for summary in summaries {
        for &pair in &summary.finalized_pairs {
            push_finite_interval(&mut intervals, pair);
        }

        for event in &summary.attach_events {
            let node = global_node_id(&offsets, summary.slab_id, event.interface_node);
            attach_by_value[event.value as usize].push(GlobalAttachEvent {
                node,
                branch_birth: event.branch_birth,
            });
        }

        for event in &summary.outside_events {
            let node = global_node_id(&offsets, summary.slab_id, event.interface_node);
            outside_by_value[event.value as usize].push(node);
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

    for value in (0..NUM_U16_VALUES).rev() {
        let merge_value = value as u16;

        // Outside is the oldest component. Processing outside events first also
        // resolves same-value ties consistently with the padded construction.
        for &node in &outside_by_value[value] {
            if let Some(pair) = global_uf.connect_outside(node, merge_value) {
                push_finite_interval(&mut intervals, pair);
            }
        }

        for event in &attach_by_value[value] {
            if let Some(pair) = global_uf.attach_branch(event.node, event.branch_birth, merge_value)
            {
                push_finite_interval(&mut intervals, pair);
            }
        }

        for event in &interface_by_value[value] {
            if let Some(pair) = global_uf.union_with_persistence(event.a, event.b, merge_value) {
                push_finite_interval(&mut intervals, pair);
            }
        }

        for event in &cross_by_value[value] {
            if let Some(pair) = global_uf.union_with_persistence(event.a, event.b, merge_value) {
                push_finite_interval(&mut intervals, pair);
            }
        }
    }

    let remaining = global_uf.remaining_finite_root_births();
    if !remaining.is_empty() {
        bail!(
            "H2 persistence reduction ended with {} background components not connected to outside",
            remaining.len()
        );
    }

    Ok(intervals)
}

pub fn compute_h2_persistence_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<H2Interval>> {
    let start = Instant::now();
    println!("Computing in-memory slabwise H2 persistence...");

    let summaries = process_all_slabs_h2(volume, slab_depth, background_connectivity)?;
    println!("Processed {} H2 slab summaries", summaries.len());

    let intervals = reduce_h2_persistence(&summaries, background_connectivity)?;

    println!(
        "In-memory H2 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(intervals)
}

pub fn write_h2_persistence_csv(path: &Path, intervals: &[H2Interval]) -> Result<()> {
    let mut file = AtomicOutput::create(path)?;
    writeln!(file, "birth,death")?;

    for interval in intervals {
        writeln!(file, "{},{}", interval.birth, interval.death)?;
    }

    file.commit()
}

pub fn betti2_curve_from_h2_intervals(intervals: &[H2Interval]) -> Vec<(u16, i64)> {
    let mut delta = vec![0i64; NUM_U16_VALUES + 1];

    for interval in intervals {
        delta[interval.birth as usize] += 1;
        delta[interval.death as usize] -= 1;
    }

    let mut sparse = Vec::new();
    let mut beta2 = 0i64;
    let mut previous: Option<i64> = None;

    for (value, change) in delta.iter().copied().enumerate().take(NUM_U16_VALUES) {
        beta2 += change;

        if previous != Some(beta2) {
            sparse.push((value as u16, beta2));
            previous = Some(beta2);
        }
    }

    sparse
}
