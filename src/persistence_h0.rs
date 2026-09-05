use anyhow::Result;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H0Interval {
    pub birth: u16,
    pub death: Option<u16>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FinitePair {
    pub(crate) birth: u16,
    pub(crate) death: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttachEvent {
    pub(crate) value: u16,
    pub(crate) interface_node: u32,
    pub(crate) branch_birth: u16,
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
pub(crate) struct SlabH0Summary {
    pub(crate) slab_id: usize,
    pub(crate) finalized_pairs: Vec<FinitePair>,
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
enum LocalPersistenceAction {
    FinalPair(FinitePair),
    Attach(AttachEvent),
    InterfaceMerge(InterfaceMergeEvent),
}

#[derive(Debug)]
struct LocalPersistenceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<u16>,
    interface_rep: Vec<u32>,
}

impl LocalPersistenceUnionFind {
    fn new(values: &[u16]) -> Self {
        assert!(
            values.len() <= u32::MAX as usize,
            "LocalPersistenceUnionFind uses u32 indices; slab has too many voxels"
        );

        Self {
            parent: (0..values.len() as u32).collect(),
            rank: vec![0u8; values.len()],
            birth: values.to_vec(),
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

    fn union_with_persistence(
        &mut self,
        a: u32,
        b: u32,
        death: u16,
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

        let rank_a = self.rank[root_a as usize];
        let rank_b = self.rank[root_b as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.parent[root_b as usize] = root_a;

        if rank_a == rank_b {
            self.rank[root_a as usize] += 1;
        }

        self.birth[root_a as usize] = birth_a.min(birth_b);
        self.interface_rep[root_a as usize] = if rep_a != NO_INTERFACE_REP {
            rep_a
        } else {
            rep_b
        };

        Some(action)
    }
}

fn handle_local_action(
    action: LocalPersistenceAction,
    finalized_pairs: &mut Vec<FinitePair>,
    attach_events: &mut Vec<AttachEvent>,
    interface_merge_events: &mut Vec<InterfaceMergeEvent>,
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

struct LocalPersistenceBuffers<'a> {
    finalized_pairs: &'a mut Vec<FinitePair>,
    attach_events: &'a mut Vec<AttachEvent>,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent>,
}

fn union_active_neighbor_h0(
    uf: &mut LocalPersistenceUnionFind,
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
        handle_local_action(
            action,
            buffers.finalized_pairs,
            buffers.attach_events,
            buffers.interface_merge_events,
        );
    }
}

pub(crate) fn process_slab_h0_persistence(
    slab_id: usize,
    block: &Block,
    connectivity: Connectivity,
) -> SlabH0Summary {
    let voxel_count = block.voxel_count();
    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalPersistenceUnionFind::new(&block.values);
    let mut active = vec![0u8; voxel_count];
    let buckets = build_voxel_buckets_u16(&block.values);
    let mut local_pruner = NeighborhoodComponentPruner::new(connectivity);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut finalized_pairs = Vec::new();
    let mut attach_events = Vec::new();
    let mut interface_merge_events = Vec::new();

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
            active[idx] = 1;

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

            local_pruner.for_each_representative_neighbor(
                &active,
                x,
                y,
                z,
                width,
                height,
                depth,
                |neighbor| {
                    union_active_neighbor_h0(
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

    SlabH0Summary {
        slab_id,
        finalized_pairs,
        attach_events,
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

fn process_all_slabs_h0(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<SlabH0Summary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);

    println!(
        "Processing {} H0-persistence slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabH0Summary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_h0_persistence(slab_id, &block, connectivity))
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

#[derive(Debug)]
pub(crate) struct GlobalPersistenceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    birth: Vec<u16>,
}

impl GlobalPersistenceUnionFind {
    pub(crate) fn new(birth: Vec<u16>) -> Self {
        assert!(
            birth.len() <= u32::MAX as usize,
            "GlobalPersistenceUnionFind uses u32 indices; too many interface nodes"
        );

        Self {
            parent: (0..birth.len() as u32).collect(),
            rank: vec![0u8; birth.len()],
            birth,
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
        death: u16,
    ) -> Option<FinitePair> {
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

        Some(pair)
    }

    pub(crate) fn attach_branch(
        &mut self,
        interface_node: u32,
        branch_birth: u16,
        death: u16,
    ) -> FinitePair {
        let root = self.find(interface_node);
        let root_birth = self.birth[root as usize];
        self.birth[root as usize] = root_birth.min(branch_birth);

        FinitePair {
            birth: root_birth.max(branch_birth),
            death,
        }
    }

    pub(crate) fn essential_births(&mut self) -> Vec<u16> {
        let mut births = Vec::new();
        for node in 0..self.parent.len() {
            let node_u32 = node as u32;
            if self.find(node_u32) == node_u32 {
                births.push(self.birth[node]);
            }
        }
        births
    }
}

fn build_global_interface_births(summaries: &[SlabH0Summary], offsets: &[u32]) -> Vec<u16> {
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
    summaries: &[SlabH0Summary],
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
                        let value = left_value.max(right_face.values[right_idx]);
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
                                let value = left_value.max(right_face.values[right_idx]);
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

fn push_finite_interval(intervals: &mut Vec<H0Interval>, pair: FinitePair) {
    if pair.birth < pair.death {
        intervals.push(H0Interval {
            birth: pair.birth,
            death: Some(pair.death),
        });
    }
}

fn reduce_h0_persistence(
    summaries: &[SlabH0Summary],
    connectivity: Connectivity,
) -> Vec<H0Interval> {
    let offsets = build_interface_offsets(summaries);
    let births = build_global_interface_births(summaries, &offsets);
    let mut global_uf = GlobalPersistenceUnionFind::new(births);

    let mut attach_by_value: Vec<Vec<GlobalAttachEvent>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
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

    for value in 0..NUM_U16_VALUES {
        let death = value as u16;

        for event in &attach_by_value[value] {
            let pair = global_uf.attach_branch(event.node, event.branch_birth, death);
            push_finite_interval(&mut intervals, pair);
        }

        for event in &interface_by_value[value] {
            if let Some(pair) = global_uf.union_with_persistence(event.a, event.b, death) {
                push_finite_interval(&mut intervals, pair);
            }
        }

        for event in &cross_by_value[value] {
            if let Some(pair) = global_uf.union_with_persistence(event.a, event.b, death) {
                push_finite_interval(&mut intervals, pair);
            }
        }
    }

    for birth in global_uf.essential_births() {
        intervals.push(H0Interval { birth, death: None });
    }

    intervals
}

pub fn compute_h0_persistence_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<H0Interval>> {
    let start = Instant::now();
    println!("Computing in-memory slabwise H0 persistence...");

    let summaries = process_all_slabs_h0(volume, slab_depth, connectivity)?;
    println!("Processed {} H0 slab summaries", summaries.len());

    let intervals = reduce_h0_persistence(&summaries, connectivity);

    println!(
        "In-memory H0 persistence computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(intervals)
}

pub fn write_h0_persistence_csv(path: &Path, intervals: &[H0Interval]) -> Result<()> {
    let mut file = AtomicOutput::create(path)?;
    writeln!(file, "birth,death")?;

    for interval in intervals {
        match interval.death {
            Some(death) => writeln!(file, "{},{}", interval.birth, death)?,
            None => writeln!(file, "{},inf", interval.birth)?,
        }
    }

    file.commit()
}

pub fn betti0_curve_from_h0_intervals(intervals: &[H0Interval]) -> Vec<(u16, i64)> {
    let mut delta = vec![0i64; NUM_U16_VALUES + 1];

    for interval in intervals {
        delta[interval.birth as usize] += 1;
        if let Some(death) = interval.death {
            delta[death as usize] -= 1;
        }
    }

    let mut sparse = Vec::new();
    let mut beta0 = 0i64;
    let mut previous: Option<i64> = None;

    for (value, change) in delta.iter().copied().enumerate().take(NUM_U16_VALUES) {
        beta0 += change;

        if previous != Some(beta0) {
            sparse.push((value as u16, beta0));
            previous = Some(beta0);
        }
    }

    sparse
}
